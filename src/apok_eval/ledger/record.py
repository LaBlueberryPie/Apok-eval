"""The record schema ``apok-record-v1``, its seal, and the run-directory layout.

A record is one canonical-JSON object (``apok_eval.canonical``) stored as its
canonical bytes plus one ``\\n`` in ``<run_dir>/record.json``. Its members
(all required; ``docs/ledger.md`` gives the meaning of each):

``schema`` ``run_id`` ``created_utc`` ``purpose`` ``code`` ``config``
``config_digest`` ``seeds`` ``seed_pointers`` ``inputs`` ``chunk_manifests``
``models`` ``environment`` ``outputs`` ``figures`` ``predictions`` ``seal``

``seal`` is ``digest_json`` of the record without its ``seal`` member. Anyone
who edits a record can recompute it, so the seal makes a record
*self-consistent*, not tamper-proof: tamper evidence comes from comparing the
seal with a value anchored elsewhere (``verify(expect_seal=...)``).

Run-directory layout: ``record.json``; ``manifests/<name>.jsonl`` (chunk
manifest sidecars, :mod:`apok_eval.ledger.manifest`); ``code.diff`` (the
``git diff --binary HEAD`` bytes when the tree was dirty); the run's outputs
anywhere else. Paths a record gives relative to the run directory are safe
relative POSIX paths and must not pass through a symlink.

:func:`record_problems` is the structural half of verification: exact member
sets, types, formats, and the cross-references (figures name an output,
predictions name an input). Everything it accepts can be recomputed or
compared by :mod:`apok_eval.ledger.verify` without further type checks.
"""

from __future__ import annotations

import datetime as dt
import os
import re
from pathlib import Path
from typing import Any, Callable

from apok_eval import jsonstrict
from apok_eval.canonical import digest_json, is_digest
from apok_eval.ledger.util import (
    is_name,
    is_safe_relpath,
    is_size,
    is_text,
    is_utf8,
    read_regular,
    resolve_pointer,
)
from apok_eval.merkle import SCHEME
from apok_eval.refusal import Refusal

SCHEMA = "apok-record-v1"
RECORD_FILE = "record.json"
MANIFEST_DIR = "manifests"
DIFF_FILE = "code.diff"
UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

TOP_KEYS = frozenset({
    "schema", "run_id", "created_utc", "purpose", "code", "config", "config_digest", "seeds", "seed_pointers",
    "inputs", "chunk_manifests", "models", "environment", "outputs", "figures", "predictions", "seal",
})
CODE_KEYS = frozenset({"package_version", "commit", "dirty", "diff_digest", "diff_file", "source_root",
                       "source_manifest_root", "source_manifest"})
INPUT_KEYS = frozenset({"name", "uri", "path", "digest", "size"})
MODEL_KEYS = frozenset({"name", "uri", "path", "digest", "size", "source"})
OUTPUT_KEYS = frozenset({"name", "path", "digest", "size"})
CHUNK_KEYS = frozenset({"name", "source_uri", "scheme", "count_present", "count_absent", "root", "manifest_file",
                        "manifest_file_digest"})
ENVIRONMENT_KEYS = frozenset({"python", "implementation", "platform", "machine", "packages"})
FIGURES_KEYS = frozenset({"source", "pointer", "values"})
PREDICTIONS_KEYS = frozenset({"registry_input", "registry_digest", "ids"})
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_UTC_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")


def compute_seal(record: dict) -> str:
    """Digest of the canonical JSON of ``record`` without its ``seal`` member."""
    return digest_json({key: value for key, value in record.items() if key != "seal"})


def read_record(path: str | os.PathLike[str]) -> tuple[bytes, dict]:
    """The raw bytes and the parsed object of a record file (a regular file; see ``util.read_regular``)."""
    try:
        raw = read_regular(path)
        record = jsonstrict.loads(raw)
    except (OSError, ValueError) as exc:
        raise Refusal("LEDGER_RECORD_UNREADABLE", f"record {path} cannot be read as a strict JSON object: {exc}",
                      record=str(path), problem=jsonstrict.problem_of(exc)) from None
    if not isinstance(record, dict):
        problem = f"the top level is a {type(record).__name__}, not an object"
        raise Refusal("LEDGER_RECORD_UNREADABLE", f"record {path} cannot be read as a strict JSON object: {problem}",
                      record=str(path), problem=problem)
    return raw, record


def load_record(path: str | os.PathLike[str]) -> dict:
    return read_record(path)[1]


def resolve_in_run_dir(run_dir: str | os.PathLike[str], rel: str) -> Path:
    """The file a run-relative record path names.

    Refuses a path that is not a safe relative path or that passes through a
    symlink (the realpath of the result must be the result itself).
    """
    base = Path(os.path.realpath(run_dir))
    candidate = base.joinpath(*rel.split("/")) if is_safe_relpath(rel) else None
    if candidate is None or os.path.realpath(candidate) != str(candidate):
        raise Refusal("LEDGER_PATH_UNSAFE",
                      f"path {rel!r} leaves the run directory {run_dir} or passes through a symlink",
                      path=rel, run_dir=str(run_dir))
    return candidate


def seed_value_problem(value: Any) -> str | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return f"a seed must be an int, not {type(value).__name__} ({value!r})"
    return None


def seed_config_problem(config: Any, pointer: Any, value: int) -> str | None:
    """Why ``config`` does not hold ``value`` at ``pointer``, or ``None``."""
    try:
        found = resolve_pointer(config, pointer)
    except (LookupError, ValueError) as exc:
        return f"pointer {pointer!r} does not resolve in the configuration: {exc}"
    if seed_value_problem(found) or found != value:
        return f"the configuration holds {found!r} at {pointer!r}, the seed is {value!r}"
    return None


def is_utc_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not _UTC_RE.fullmatch(value):
        return False
    try:
        dt.datetime.strptime(value, UTC_FORMAT)
    except ValueError:
        return False
    return True


# --- structure -------------------------------------------------------------------------


class _Problems:
    def __init__(self) -> None:
        self.items: list[str] = []

    def need(self, ok: bool, where: str, what: str) -> bool:
        if not ok:
            self.items.append(f"{where}: {what}")
        return ok

    def exact_keys(self, value: Any, keys: frozenset[str], where: str) -> bool:
        if not self.need(isinstance(value, dict), where, "must be an object"):
            return False
        missing, extra = sorted(keys - set(value)), sorted(set(value) - keys)
        return self.need(not missing and not extra, where, f"missing members {missing}, unexpected members {extra}")


def _absolute(value: Any) -> bool:
    return isinstance(value, str) and os.path.isabs(value) and "\x00" not in value and is_utf8(value)


def _optional_text(value: Any) -> bool:
    return value is None or is_text(value)


def code_problems(code: Any) -> list[str]:
    """Structural violations of a ``code`` block alone (``open_run`` checks the captured one before compute)."""
    p = _Problems()
    _code(p, code)
    return p.items


def _code(p: _Problems, code: Any) -> None:
    if not p.exact_keys(code, CODE_KEYS, "code"):
        return
    p.need(is_text(code["package_version"]), "code.package_version", "must be a non-empty string")
    commit = code["commit"]
    p.need(commit is None or (isinstance(commit, str) and bool(_COMMIT_RE.fullmatch(commit))), "code.commit",
           "must be null or 40 lowercase hex digits")
    if commit is None:
        p.need(code["dirty"] is None, "code.dirty", "must be null when there is no commit")
        p.need(code["source_root"] is None, "code.source_root", "must be null when there is no commit")
    else:
        p.need(isinstance(code["dirty"], bool), "code.dirty", "must be a bool when there is a commit")
        p.need(code["source_root"] == "." or is_safe_relpath(code["source_root"]), "code.source_root",
               "must be '.' or a safe relative path when there is a commit")
    if code["dirty"] is True:
        p.need(is_digest(code["diff_digest"]), "code.diff_digest", "must be a sha256 digest when dirty")
        p.need(is_safe_relpath(code["diff_file"]), "code.diff_file", "must be a safe relative path when dirty")
    else:
        p.need(code["diff_digest"] is None, "code.diff_digest", "must be null unless dirty")
        p.need(code["diff_file"] is None, "code.diff_file", "must be null unless dirty")
    p.need(is_digest(code["source_manifest_root"]), "code.source_manifest_root", "must be a sha256 digest")
    manifest = code["source_manifest"]
    if not p.need(isinstance(manifest, list), "code.source_manifest", "must be a list"):
        return
    previous = None
    for i, entry in enumerate(manifest):
        where = f"code.source_manifest[{i}]"
        if not p.exact_keys(entry, frozenset({"path", "digest"}), where):
            continue
        p.need(is_safe_relpath(entry["path"]), f"{where}.path", "must be a safe relative path")
        p.need(is_digest(entry["digest"]), f"{where}.digest", "must be a sha256 digest")
        if isinstance(entry["path"], str):
            key = entry["path"].encode("utf-8", "surrogatepass")
            p.need(previous is None or key > previous, f"{where}.path", "paths must be strictly sorted")
            previous = key


def _seeds(p: _Problems, seeds: Any, pointers: Any) -> None:
    ok = p.need(isinstance(seeds, dict), "seeds", "must be an object")
    ok = p.need(isinstance(pointers, dict), "seed_pointers", "must be an object") and ok
    if not ok:
        return
    p.need(set(seeds) == set(pointers), "seed_pointers", "must have exactly the names of seeds")
    for name in sorted(seeds):
        p.need(is_name(name), f"seeds.{name}", "is not a valid name")
    for name, pointer in sorted(pointers.items()):
        p.need(isinstance(pointer, str), f"seed_pointers.{name}", "must be a string")


def _entries(p: _Problems, section: str, value: Any, keys: frozenset[str],
             check: Callable[[_Problems, str, dict], None]) -> dict[str, dict]:
    """Validate a list of named entries (unique names, sorted); return the well-formed ones by name."""
    if not p.need(isinstance(value, list), section, "must be a list"):
        return {}
    by_name: dict[str, dict] = {}
    previous = None
    for i, entry in enumerate(value):
        where = f"{section}[{i}]"
        if not p.exact_keys(entry, keys, where):
            continue
        name = entry["name"]
        if p.need(is_name(name), f"{where}.name", "is not a valid name"):
            p.need(previous is None or name > previous, f"{where}.name", "names must be unique and sorted")
            previous = name
            by_name[name] = entry
        check(p, where, entry)
    return by_name


def _file(p: _Problems, where: str, entry: dict) -> None:
    p.need(is_digest(entry["digest"]), f"{where}.digest", "must be a sha256 digest")
    p.need(is_size(entry["size"]), f"{where}.size", "must be a non-negative int")


def _input(p: _Problems, where: str, entry: dict) -> None:
    p.need(entry["path"] is None or _absolute(entry["path"]), f"{where}.path", "must be null or an absolute path")
    p.need(_optional_text(entry["uri"]), f"{where}.uri", "must be null or a non-empty string")
    p.need(entry["path"] is not None or entry["uri"] is not None, where, "needs a path, a uri or both")
    _file(p, where, entry)


def _model(p: _Problems, where: str, entry: dict) -> None:
    p.need(_absolute(entry["path"]), f"{where}.path", "must be an absolute path")
    p.need(_optional_text(entry["uri"]), f"{where}.uri", "must be null or a non-empty string")
    p.need(is_text(entry["source"]), f"{where}.source", "must be a non-empty string")
    _file(p, where, entry)


def _output(p: _Problems, where: str, entry: dict) -> None:
    p.need(is_safe_relpath(entry["path"]), f"{where}.path", "must be a safe relative path")
    _file(p, where, entry)


def _chunks(p: _Problems, where: str, entry: dict) -> None:
    p.need(is_text(entry["source_uri"]), f"{where}.source_uri", "must be a non-empty string")
    p.need(entry["scheme"] == SCHEME, f"{where}.scheme", f"must be {SCHEME!r}")
    p.need(is_size(entry["count_present"]), f"{where}.count_present", "must be a non-negative int")
    p.need(is_size(entry["count_absent"]), f"{where}.count_absent", "must be a non-negative int")
    p.need(is_digest(entry["root"]), f"{where}.root", "must be a sha256 digest")
    p.need(is_safe_relpath(entry["manifest_file"]), f"{where}.manifest_file", "must be a safe relative path")
    p.need(is_digest(entry["manifest_file_digest"]), f"{where}.manifest_file_digest", "must be a sha256 digest")


def _environment(p: _Problems, env: Any) -> None:
    if not p.exact_keys(env, ENVIRONMENT_KEYS, "environment"):
        return
    for key in ("python", "implementation", "platform", "machine"):
        p.need(isinstance(env[key], str), f"environment.{key}", "must be a string")
    packages = env["packages"]
    if p.need(isinstance(packages, dict), "environment.packages", "must be an object"):
        for name, version in sorted(packages.items()):
            p.need(version is None or isinstance(version, str), f"environment.packages.{name}",
                   "must be null or a string")


def _figures(p: _Problems, figures: Any, outputs: dict[str, dict]) -> None:
    if figures is None or not p.exact_keys(figures, FIGURES_KEYS, "figures"):
        return
    p.need(isinstance(figures["source"], str) and figures["source"] in outputs, "figures.source",
           "must name a recorded output")
    pointer = figures["pointer"]
    p.need(isinstance(pointer, str) and (pointer == "" or pointer.startswith("/")), "figures.pointer",
           "must be a JSON pointer ('' or starting with '/')")
    p.need(isinstance(figures["values"], dict) and bool(figures["values"]), "figures.values",
           "must be a non-empty object")


def _predictions(p: _Problems, predictions: Any, inputs: dict[str, dict]) -> None:
    if predictions is None or not p.exact_keys(predictions, PREDICTIONS_KEYS, "predictions"):
        return
    registry = predictions["registry_input"]
    p.need(isinstance(registry, str) and registry in inputs, "predictions.registry_input",
           "must name a recorded input")
    p.need(is_digest(predictions["registry_digest"]), "predictions.registry_digest", "must be a sha256 digest")
    ids = predictions["ids"]
    p.need(isinstance(ids, list) and bool(ids) and all(is_text(i) for i in ids)
           and all(a < b for a, b in zip(ids, ids[1:])), "predictions.ids",
           "must be a non-empty, strictly sorted list of non-empty strings")


def record_problems(record: dict) -> list[str]:
    """Every structural violation of ``apok-record-v1`` in ``record`` (empty when well-formed).

    Each problem starts with the path of the offending member. The ``schema``
    member's value is the caller's to check first (a different schema is
    refused as unsupported, not as malformed).
    """
    p = _Problems()
    if not p.exact_keys(record, TOP_KEYS, "$"):
        return p.items
    p.need(is_name(record["run_id"]), "run_id", "must match [a-z0-9][a-z0-9._-]{0,127}")
    p.need(is_utc_timestamp(record["created_utc"]), "created_utc", "must be a UTC time YYYY-MM-DDTHH:MM:SSZ")
    p.need(is_text(record["purpose"]), "purpose", "must be a non-empty string")
    _code(p, record["code"])
    p.need(isinstance(record["config"], dict), "config", "must be an object")
    p.need(is_digest(record["config_digest"]), "config_digest", "must be a sha256 digest")
    _seeds(p, record["seeds"], record["seed_pointers"])
    inputs = _entries(p, "inputs", record["inputs"], INPUT_KEYS, _input)
    _entries(p, "chunk_manifests", record["chunk_manifests"], CHUNK_KEYS, _chunks)
    _entries(p, "models", record["models"], MODEL_KEYS, _model)
    _environment(p, record["environment"])
    outputs = _entries(p, "outputs", record["outputs"], OUTPUT_KEYS, _output)
    p.need(bool(record["outputs"]), "outputs", "a sealed record has at least one output")
    _figures(p, record["figures"], outputs)
    _predictions(p, record["predictions"], inputs)
    p.need(is_digest(record["seal"]), "seal", "must be a sha256 digest")
    return p.items
