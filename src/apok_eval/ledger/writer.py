"""Writing one sealed run record.

::

    run = open_run("runs/fold-a", run_id="fold-a", purpose="held-out fit, fold A", config=config,
                   expected_bytes=2 << 30, index="ledger/index.jsonl")
    run.add_input("umbilicus", "/data/umbilicus.json")
    run.add_chunk_manifest("volume", read_log_entries, source_uri="https://.../volume.zarr")
    run.add_model("winding-model", "/models/ckpt.pth", source="trained by run wm-11")
    run.set_seed("optimizer", 1, config_pointer="/optimizer_random_seed")
    ...                                              # the run itself
    run.add_output("metrics", "metrics.json")         # relative to the run directory
    run.set_figures("metrics", pointer="/summary")
    seal = run.seal()

:func:`open_run` checks before any compute is spent, read-only checks first:
arguments; the run directory (a sealed record there is never touched; a
non-empty directory needs ``resume=True``); free disk against the caller's
stated ``expected_bytes``; the code state (it must be recordable in
``apok-record-v1``; a dirty tree needs ``allow_dirty=True``; for a checkout
the source manifest must already equal what the commit - and, when dirty,
the diff - imply, or ``verify`` would fail later); the index (it must verify
and must not list the run_id). Only then does it write: it creates the run
directory and probes it with a real write, writes ``code.diff`` for a dirty
tree, and creates and probes the index directory.

Digests are computed by the ledger from the files (``add_input``,
``add_model``, ``add_output``) or bytes (``add_remote_input``); no method
takes a digest from the caller. Chunk-manifest entries are the exception by
nature: their digests are of objects the reader fetched, and the ledger
records them in a sidecar whose own digest it computes.

:meth:`RunWriter.seal` refuses a run with no outputs, re-captures the code
state and refuses if the commit or source manifest changed since
``open_run``, assembles the record, runs every ``verify`` check except the
git comparison on it (so a sealed record verifies: a file changed since it
was added refuses the seal), publishes ``record.json`` atomically without
ever replacing an existing one, and appends to the index.
"""

from __future__ import annotations

import datetime as dt
import os
import shutil
from pathlib import Path, PurePath
from typing import Any, Iterable

import apok_eval
from apok_eval import jsonstrict
from apok_eval.canonical import canonical_dumps, digest_json, sha256_bytes
from apok_eval.ledger import manifest
from apok_eval.ledger.environment import capture_environment
from apok_eval.ledger.gitcode import CodeState, capture_code, compare_manifests
from apok_eval.ledger.index import append_index, check_index_accepts
from apok_eval.ledger.record import (
    DIFF_FILE,
    MANIFEST_DIR,
    RECORD_FILE,
    SCHEMA,
    UTC_FORMAT,
    code_problems,
    compute_seal,
    resolve_in_run_dir,
    seed_config_problem,
    seed_value_problem,
)
from apok_eval.ledger.util import (
    TMP_PREFIX,
    NotRegularFile,
    TargetExists,
    digest_and_size,
    is_name,
    is_safe_relpath,
    is_text,
    is_utf8,
    probe_writable,
    read_json_at,
    write_atomic,
)
from apok_eval.ledger.verify import check_record, summarise
from apok_eval.merkle import SCHEME, manifest_root
from apok_eval.refusal import Refusal


def _free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def _nearest_existing(path: Path) -> Path:
    while not path.exists():
        path = path.parent
    return path


def _canonical_copy(value: Any) -> Any:
    """A deep copy through canonical JSON, or ``None`` if ``value`` is not canonical JSON (or nests too deeply)."""
    try:
        return jsonstrict.loads(canonical_dumps(value))
    except (TypeError, ValueError, RecursionError):
        return None


def _is_path(value: Any) -> bool:
    """A str or os.PathLike naming UTF-8 text without NUL (``os.open`` raises ValueError for the others)."""
    try:
        text = os.fspath(value)
    except TypeError:
        return False
    return isinstance(text, str) and "\x00" not in text and is_utf8(text)


def _regular_file_digest(path: Path, what: str) -> tuple[str, int]:
    try:
        return digest_and_size(path)
    except NotRegularFile as exc:
        raise Refusal("LEDGER_FILE_MISSING", f"{what}: no regular file at {path} ({exc})", what=what, path=str(path))
    except OSError as exc:
        raise Refusal("LEDGER_FILE_UNREADABLE", f"{what}: {path} cannot be read: {exc}", what=what, path=str(path),
                      errno=exc.errno)


def code_block(code: CodeState) -> dict:
    """The record's ``code`` member for a captured code state."""
    return {
        "package_version": code.package_version, "commit": code.commit, "dirty": code.dirty,
        "diff_digest": sha256_bytes(code.diff) if code.diff is not None else None,
        "diff_file": DIFF_FILE if code.diff is not None else None, "source_root": code.source_root,
        "source_manifest_root": code.source_manifest_root, "source_manifest": code.source_manifest,
    }


def open_run(run_dir: str | os.PathLike[str], *, run_id: str, purpose: str, config: dict, expected_bytes: int,
             index: str | os.PathLike[str] | None, resume: bool = False, allow_dirty: bool = False,
             source_dir: str | os.PathLike[str] | None = None) -> "RunWriter":
    """Open a run for writing (see the module docstring).

    ``index`` is the index file to append to at seal (``None``: no index).
    ``source_dir`` is the package source whose identity is recorded; by
    default the directory of the running ``apok_eval`` package.
    """
    if not is_name(run_id):
        raise Refusal("LEDGER_RUN_ID_INVALID", f"run_id {run_id!r} does not match [a-z0-9][a-z0-9._-]{{0,127}}",
                      run_id=run_id)
    for field, value in (("run_dir", run_dir), ("index", index), ("source_dir", source_dir)):
        if value is not None and not _is_path(value):
            raise Refusal("LEDGER_ARGUMENT_INVALID", f"{field} {value!r} is not a valid path (a str or path-like "
                          "naming valid UTF-8 text without NUL)", field=field)
    if not is_text(purpose):
        raise Refusal("LEDGER_ARGUMENT_INVALID", "purpose must be a non-empty string", field="purpose")
    config_copy = _canonical_copy(config)
    if not isinstance(config_copy, dict):
        raise Refusal("LEDGER_ARGUMENT_INVALID", "config must be an object of canonical JSON values (no tuples, "
                      "NaN, non-string keys, lone surrogates or nesting past 256 levels)", field="config")
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int) or expected_bytes < 0:
        raise Refusal("LEDGER_ARGUMENT_INVALID", f"expected_bytes must be a non-negative int, got {expected_bytes!r}",
                      field="expected_bytes")
    source = Path(source_dir) if source_dir is not None else Path(apok_eval.__file__).resolve().parent
    if not source.is_dir():
        raise Refusal("LEDGER_ARGUMENT_INVALID", f"source_dir {source} is not a directory", field="source_dir")

    run_path = Path(os.path.abspath(run_dir))
    if (run_path / RECORD_FILE).exists():
        raise Refusal("LEDGER_ALREADY_SEALED", f"{run_path / RECORD_FILE} exists; a sealed run is never reopened",
                      run_dir=str(run_path))
    if run_path.is_dir() and any(run_path.iterdir()) and not resume:
        raise Refusal("LEDGER_RUN_DIR_NOT_EMPTY", f"run directory {run_path} is not empty (pass resume=True to "
                      "continue a run there)", run_dir=str(run_path))
    existing = _nearest_existing(run_path)
    free = _free_bytes(existing)
    if free < expected_bytes:
        raise Refusal("LEDGER_DISK_INSUFFICIENT", f"{free} bytes free at {existing}, the run states it needs "
                      f"{expected_bytes}", path=str(existing), free=free, expected_bytes=expected_bytes)

    code = capture_code(source)
    problems = code_problems(code_block(code))
    if problems:
        raise Refusal("LEDGER_SOURCE_UNSUPPORTED", f"the code identity of {source} cannot be recorded in {SCHEMA}: "
                      f"{problems[0]}", source_dir=str(source), field=problems[0].split(":", 1)[0],
                      commit=code.commit, problems=problems)
    if code.dirty and not allow_dirty:
        raise Refusal("LEDGER_DIRTY_TREE", f"the git tree holding {source} has uncommitted changes "
                      "(pass allow_dirty=True to record them as a diff)", source_dir=str(source),
                      changed=list(code.status[:20]))
    differences = {} if code.expected_manifest is None else compare_manifests(code.source_manifest,
                                                                               code.expected_manifest)
    if differences:
        raise Refusal("LEDGER_CODE_MISMATCH", f"the files under {source} differ from what commit {code.commit} "
                      f"{'and the diff of the dirty tree imply' if code.dirty else 'implies'} (ignored files?)",
                      commit=code.commit, **differences)

    index_path = Path(os.path.abspath(index)) if index is not None else None
    if index_path is not None:
        check_index_accepts(index_path, run_id)

    target = run_path
    try:
        run_path.mkdir(parents=True, exist_ok=True)
        probe_writable(run_path)
        if code.diff is not None:
            write_atomic(run_path / DIFF_FILE, code.diff, overwrite=True)
        if index_path is not None:
            target = index_path.parent
            target.mkdir(parents=True, exist_ok=True)
            probe_writable(target)
    except OSError as exc:
        raise Refusal("LEDGER_UNWRITABLE", f"cannot write to {target}: {exc}", path=str(target))

    created = dt.datetime.now(dt.timezone.utc).strftime(UTC_FORMAT)
    return RunWriter(run_path, run_id, purpose, config_copy, created, source, code, index_path)


class RunWriter:
    """One open run; build the record with the ``add_*``/``set_*`` methods, then :meth:`seal`."""

    def __init__(self, run_dir: Path, run_id: str, purpose: str, config: dict, created_utc: str, source_dir: Path,
                 code: CodeState, index: Path | None) -> None:
        self.run_dir = run_dir
        self.record_path = run_dir / RECORD_FILE
        self.run_id = run_id
        self._purpose = purpose
        self._config = config
        self._created_utc = created_utc
        self._source_dir = source_dir
        self._code = code
        self._index = index
        self._sections: dict[str, dict[str, Any]] = {
            "inputs": {}, "models": {}, "outputs": {}, "chunk_manifests": {}, "seeds": {}}
        self._seed_pointers: dict[str, str] = {}
        self._figures: dict | None = None
        self._predictions: dict | None = None
        self._seal: str | None = None

    # --- guards shared by every method ---------------------------------------------------

    def _require_open(self) -> None:
        if self._seal is not None:
            raise Refusal("LEDGER_ALREADY_SEALED", f"run {self.run_id} was sealed ({self._seal}); it cannot change",
                          run_id=self.run_id, seal=self._seal)

    def _new_name(self, section: str, name: Any) -> str:
        if not is_name(name):
            raise Refusal("LEDGER_NAME_INVALID", f"{section} name {name!r} does not match "
                          "[a-z0-9][a-z0-9._-]{0,127}", section=section, name=name)
        if name in self._sections[section]:
            raise Refusal("LEDGER_NAME_DUPLICATE", f"{section} already has an entry named {name!r}",
                          section=section, name=name)
        return name

    def _local_file(self, path: str | os.PathLike[str], what: str) -> dict:
        full = os.path.abspath(path)
        if not is_utf8(full) or "\x00" in full:
            raise Refusal("LEDGER_ARGUMENT_INVALID", f"{what}: path {full!r} is not a valid path (not UTF-8, or "
                          "holding NUL)", field="path", what=what)
        digest, size = _regular_file_digest(Path(full), what)
        return {"path": full, "digest": digest, "size": size}

    # --- what the run consumed ------------------------------------------------------------

    def add_input(self, name: str, path: str | os.PathLike[str], *, uri: str | None = None) -> dict:
        """Record a local input file; its digest and size are computed here."""
        self._require_open()
        self._new_name("inputs", name)
        if uri is not None and not is_text(uri):
            raise Refusal("LEDGER_ARGUMENT_INVALID", f"input {name!r}: uri must be null or a non-empty string",
                          field="uri", name=name)
        entry = {"name": name, "uri": uri, **self._local_file(path, f"input {name!r}")}
        self._sections["inputs"][name] = entry
        return dict(entry)

    def add_remote_input(self, name: str, data: bytes, *, uri: str) -> dict:
        """Record an input read from ``uri`` and never stored locally, from the bytes as fetched.

        ``verify`` cannot re-check it (it never fetches); only the seal covers it.
        """
        self._require_open()
        self._new_name("inputs", name)
        if not isinstance(data, bytes):
            raise Refusal("LEDGER_ARGUMENT_INVALID", f"input {name!r}: data must be the bytes as fetched",
                          field="data", name=name)
        if not is_text(uri):
            raise Refusal("LEDGER_ARGUMENT_INVALID", f"remote input {name!r} needs a uri", field="uri", name=name)
        entry = {"name": name, "uri": uri, "path": None, "digest": sha256_bytes(data), "size": len(data)}
        self._sections["inputs"][name] = entry
        return dict(entry)

    def add_model(self, name: str, path: str | os.PathLike[str], *, source: str, uri: str | None = None) -> dict:
        """Record a model file and where it came from (``source``: free text, e.g. the run that trained it)."""
        self._require_open()
        self._new_name("models", name)
        if not is_text(source):
            raise Refusal("LEDGER_ARGUMENT_INVALID", f"model {name!r}: source must be a non-empty string",
                          field="source", name=name)
        if uri is not None and not is_text(uri):
            raise Refusal("LEDGER_ARGUMENT_INVALID", f"model {name!r}: uri must be null or a non-empty string",
                          field="uri", name=name)
        entry = {"name": name, "uri": uri, **self._local_file(path, f"model {name!r}"), "source": source}
        self._sections["models"][name] = entry
        return dict(entry)

    def add_chunk_manifest(self, name: str, entries: Iterable[dict], *, source_uri: str) -> dict:
        """Record the objects read from one store; writes the sidecar ``manifests/<name>.jsonl``."""
        self._require_open()
        self._new_name("chunk_manifests", name)
        if not is_text(source_uri):
            raise Refusal("LEDGER_ARGUMENT_INVALID", f"chunk manifest {name!r}: source_uri must be a non-empty "
                          "string", field="source_uri", name=name)
        try:
            ordered = manifest.sorted_entries(entries)
        except (TypeError, ValueError) as exc:
            raise Refusal("LEDGER_MANIFEST_INVALID", f"chunk manifest {name!r}: {exc}", manifest=name,
                          problem=str(exc))
        data = manifest.to_jsonl(ordered)
        rel = f"{MANIFEST_DIR}/{name}.jsonl"
        try:
            (self.run_dir / MANIFEST_DIR).mkdir(exist_ok=True)
            # A resumed run directory may hold a symlinked manifests/: never write through it.
            write_atomic(resolve_in_run_dir(self.run_dir, rel), data, overwrite=True)
        except OSError as exc:
            raise Refusal("LEDGER_WRITE_FAILED", f"chunk manifest {name!r}: cannot write {rel}: {exc}", path=rel)
        present, absent = manifest.counts(ordered)
        entry = {"name": name, "source_uri": source_uri, "scheme": SCHEME, "count_present": present,
                 "count_absent": absent, "root": manifest_root(ordered), "manifest_file": rel,
                 "manifest_file_digest": sha256_bytes(data)}
        self._sections["chunk_manifests"][name] = entry
        return dict(entry)

    def set_seed(self, name: str, value: int, *, config_pointer: str) -> None:
        """Record a seed the run used; the configuration must hold the same int at ``config_pointer``."""
        self._require_open()
        self._new_name("seeds", name)
        problem = seed_value_problem(value)
        if problem:
            raise Refusal("LEDGER_SEED_INVALID", f"seed {name!r}: {problem}", seed=name)
        problem = seed_config_problem(self._config, config_pointer, value)
        if problem:
            raise Refusal("LEDGER_SEED_CONFIG_MISMATCH", f"seed {name!r}: {problem}", seed=name,
                          pointer=config_pointer)
        self._sections["seeds"][name] = value
        self._seed_pointers[name] = config_pointer

    def set_predictions(self, registry_input: str, ids: Iterable[str]) -> None:
        """Record the pre-registered predictions in force: a recorded input (the registry) and prediction ids."""
        self._require_open()
        if self._predictions is not None:
            raise Refusal("LEDGER_PREDICTIONS_INVALID", "predictions are already set for this run",
                          field="predictions")
        registry = self._sections["inputs"].get(registry_input)
        if registry is None:
            raise Refusal("LEDGER_PREDICTIONS_INVALID", f"the predictions registry {registry_input!r} is not a "
                          "recorded input", field="registry_input", registry_input=registry_input)
        id_list = [] if isinstance(ids, (str, bytes)) else list(ids)
        if not id_list or not all(is_text(i) for i in id_list) or len(set(id_list)) != len(id_list):
            raise Refusal("LEDGER_PREDICTIONS_INVALID", "prediction ids must be a non-empty collection of distinct "
                          "non-empty strings", field="ids")
        self._predictions = {"registry_input": registry_input, "registry_digest": registry["digest"],
                             "ids": sorted(id_list)}

    # --- what the run produced -------------------------------------------------------------

    def _run_relative(self, path: str | os.PathLike[str]) -> str:
        candidate = Path(path)
        full = candidate if candidate.is_absolute() else self.run_dir / candidate
        rel = PurePath(os.path.relpath(os.path.abspath(full), self.run_dir)).as_posix()
        reserved = (rel in (RECORD_FILE, DIFF_FILE) or rel.split("/")[0] == MANIFEST_DIR
                    or rel.rsplit("/", 1)[-1].startswith(TMP_PREFIX))
        if not is_safe_relpath(rel) or reserved:
            raise Refusal("LEDGER_PATH_UNSAFE", f"output path {os.fspath(path)!r} is outside the run directory or "
                          "reserved for the ledger", path=os.fspath(path), run_dir=str(self.run_dir))
        return rel

    def add_output(self, name: str, path: str | os.PathLike[str]) -> dict:
        """Record an output file inside the run directory (``path`` relative to it, or absolute inside it)."""
        self._require_open()
        self._new_name("outputs", name)
        rel = self._run_relative(path)
        digest, size = _regular_file_digest(resolve_in_run_dir(self.run_dir, rel), f"output {name!r}")
        entry = {"name": name, "path": rel, "digest": digest, "size": size}
        self._sections["outputs"][name] = entry
        return dict(entry)

    def set_figures(self, output_name: str, *, pointer: str = "") -> dict:
        """Quote the figures at ``pointer`` in a recorded JSON output; they must be a non-empty object."""
        self._require_open()
        if self._figures is not None:
            raise Refusal("LEDGER_FIGURES_INVALID", "figures are already set for this run", source=output_name)
        output = self._sections["outputs"].get(output_name)
        if output is None:
            raise Refusal("LEDGER_FIGURES_INVALID", f"figures source {output_name!r} is not a recorded output; "
                          "every quoted figure must be backed by an output", source=output_name)
        try:
            values = read_json_at(resolve_in_run_dir(self.run_dir, output["path"]), pointer)
        except (OSError, ValueError, LookupError) as exc:
            raise Refusal("LEDGER_FIGURES_INVALID", f"output {output_name!r} does not yield figures at "
                          f"{pointer!r}: {exc}", source=output_name, pointer=pointer)
        if not isinstance(values, dict) or not values:
            raise Refusal("LEDGER_FIGURES_INVALID", f"the figures at {pointer!r} in {output_name!r} are not a "
                          "non-empty object", source=output_name, pointer=pointer)
        self._figures = {"source": output_name, "pointer": pointer, "values": values}
        return dict(self._figures)

    # --- sealing ---------------------------------------------------------------------------

    def _record(self) -> dict:
        def listed(section: str) -> list[dict]:
            return [self._sections[section][name] for name in sorted(self._sections[section])]

        return {
            "schema": SCHEMA, "run_id": self.run_id, "created_utc": self._created_utc, "purpose": self._purpose,
            "code": code_block(self._code), "config": self._config, "config_digest": digest_json(self._config),
            "seeds": dict(self._sections["seeds"]), "seed_pointers": dict(self._seed_pointers),
            "inputs": listed("inputs"), "chunk_manifests": listed("chunk_manifests"), "models": listed("models"),
            "environment": capture_environment(), "outputs": listed("outputs"), "figures": self._figures,
            "predictions": self._predictions,
        }

    def seal(self) -> str:
        """Seal the run: write ``record.json`` (never over an existing one), append to the index, return the seal."""
        self._require_open()
        if not self._sections["outputs"]:
            raise Refusal("LEDGER_NO_OUTPUTS", f"run {self.run_id} has no outputs; a run that produced nothing "
                          "recordable is not sealed", run_id=self.run_id)
        now = capture_code(self._source_dir)
        if (now.commit, now.source_manifest_root) != (self._code.commit, self._code.source_manifest_root):
            raise Refusal("LEDGER_CODE_CHANGED", f"the code under {self._source_dir} changed while run "
                          f"{self.run_id} was open", run_id=self.run_id,
                          opened={"commit": self._code.commit, "source_manifest_root": self._code.source_manifest_root},
                          now={"commit": now.commit, "source_manifest_root": now.source_manifest_root})
        record = self._record()
        record["seal"] = compute_seal(record)
        summarise({"record": str(self.record_path), "checks": check_record(record, self.run_dir, check_code=False)},
                  f"run {self.run_id} would not verify; not sealed")
        try:
            write_atomic(self.record_path, canonical_dumps(record) + b"\n", overwrite=False)
        except TargetExists:
            raise Refusal("LEDGER_ALREADY_SEALED", f"{self.record_path} appeared while the run was open; it is not "
                          "replaced", run_dir=str(self.run_dir))
        except OSError as exc:
            raise Refusal("LEDGER_WRITE_FAILED", f"cannot write {self.record_path}: {exc}",
                          path=str(self.record_path))
        self._seal = record["seal"]
        if self._index is not None:
            append_index(self._index, run_id=self.run_id, seal=self._seal, record_path=self.record_path)
        return self._seal
