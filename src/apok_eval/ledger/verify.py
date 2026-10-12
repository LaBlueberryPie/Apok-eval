"""``verify``: recompute everything a sealed record states.

Each check is a function that returns on success, raises its own
:class:`~apok_eval.refusal.Refusal` on failure, or raises :class:`NotChecked` with
the reason it cannot be performed. :func:`run_verification` runs every check
and returns a report listing each as ``PASS``, ``FAIL`` (code, reason,
details) or ``NOT CHECKED`` (reason); :func:`verify` then raises one
``LEDGER_VERIFY_FAILED`` whose details list the failed checks.

Checks, in report order (names as they appear in the report):

``record``
    the file parses as a strict JSON object (``LEDGER_RECORD_UNREADABLE``);
    nothing else can run without it.
``canonical``
    the file's bytes are the canonical JSON of that object plus ``\\n``
    (``LEDGER_RECORD_NOT_CANONICAL``), so no byte of the file is free.
``seal``
    ``seal`` equals the digest of the record without ``seal``
    (``LEDGER_SEAL_MISMATCH``).
``expected_seal``
    with ``expect_seal``, the recomputed seal equals that anchored value
    (``LEDGER_SEAL_UNEXPECTED``); NOT CHECKED without one.
``schema``
    ``schema`` is ``apok-record-v1`` (``LEDGER_SCHEMA_UNSUPPORTED``) and the
    structure is valid (``LEDGER_RECORD_MALFORMED``). If this fails, the
    content checks below are reported as one NOT CHECKED entry ``content``.
``config_digest``
    ``LEDGER_CONFIG_DIGEST_MISMATCH``.
``seeds``
    every seed is an int, not a bool (``LEDGER_SEED_INVALID``), and the
    configuration holds the same int at the seed's pointer
    (``LEDGER_SEED_CONFIG_MISMATCH``).
``input:<name>``, ``model:<name>``, ``output:<name>``
    a regular file exists (``LEDGER_FILE_MISSING``; for inputs and models NOT
    CHECKED instead when ``allow_missing_inputs``) and can be read
    (``LEDGER_FILE_UNREADABLE``, never excused), its size and sha256 match
    (``LEDGER_FILE_SIZE_MISMATCH``, ``LEDGER_FILE_DIGEST_MISMATCH``). Outputs
    are resolved inside the run directory without symlinks
    (``LEDGER_PATH_UNSAFE``). A remote-only input (``path`` null) is NOT
    CHECKED: verify never fetches.
``chunk_manifest:<name>``
    the sidecar exists and has the recorded digest; its lines are canonical,
    valid, sorted and unique (``LEDGER_MANIFEST_INVALID``); the Merkle root of
    its entries is the recorded root (``LEDGER_MANIFEST_ROOT_MISMATCH``); the
    present/absent counts match (``LEDGER_MANIFEST_COUNT_MISMATCH``). This
    is the sidecar's own consistency: it says nothing about the chunks.
``chunk_data:<name>``
    when ``source_uri`` is a local ``file://`` URI (no host, or
    ``localhost``), every key of the sidecar is read again through
    :class:`apok_eval.io.store.LocalStore` at that root, and a present entry must
    still have its digest and size, an absent one still be absent
    (``LEDGER_CHUNK_MISMATCH``, naming the keys; a key the store refuses to
    read counts as differing). A local store that is gone is
    ``LEDGER_FILE_MISSING`` (NOT CHECKED with ``allow_missing_inputs``). Any
    other ``source_uri`` is NOT CHECKED: verify never fetches, so the chunk
    bytes of a remote store are covered by the sidecar's digests and the seal
    only.
``figures``
    the figures equal the value at their pointer in the source output's
    parsed content, compared as canonical JSON so ``1``, ``1.0`` and
    ``true`` differ (``LEDGER_FIGURES_INVALID`` when the file is not strict
    JSON or the pointer does not resolve, ``LEDGER_FIGURES_MISMATCH``). NOT
    CHECKED when the record quotes no figures.
``predictions``
    ``registry_digest`` equals the recorded digest of the registry input
    (``LEDGER_PREDICTIONS_MISMATCH``). NOT CHECKED when none are recorded.
``code.source_manifest_root``
    the Merkle root of the source manifest (``LEDGER_SOURCE_ROOT_MISMATCH``).
``code.diff``
    for a dirty tree, ``code.diff`` exists and has ``diff_digest``; NOT
    CHECKED for a clean tree or no checkout.
``code.commit``
    with ``check_code``, the source manifest equals the manifest the recorded
    commit implies, computed from git objects - for a dirty tree, with the
    recorded diff applied (``LEDGER_CODE_MISMATCH``, also when that diff does
    not apply). A commit missing from a repository given explicitly as
    ``repo`` fails (``LEDGER_COMMIT_MISSING``). NOT CHECKED (with the reason)
    when disabled, without a commit, when the recorded diff cannot be read or
    lacks its digest (the ``code.diff`` check fails then), without a
    repository, when the commit is not in the default repository, or when git
    fails.
``environment``
    always NOT CHECKED: it cannot be recomputed; only the seal covers it.
"""

from __future__ import annotations

import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

import apok_eval
from apok_eval.canonical import canonical_dumps, digest_json, sha256_bytes
from apok_eval.ledger import gitcode, manifest
from apok_eval.ledger.record import (
    SCHEMA,
    compute_seal,
    read_record,
    record_problems,
    resolve_in_run_dir,
    seed_config_problem,
    seed_value_problem,
)
from apok_eval.fileio import entry_kind
from apok_eval.ledger.util import NotRegularFile, digest_and_size, read_json_at, read_regular
from apok_eval.merkle import manifest_root
from apok_eval.refusal import NOT_CHECKED, Refusal

PASS, FAIL = "PASS", "FAIL"


class NotChecked(Exception):
    """A check that cannot be performed; the message says why."""


def _outcome(name: str, check: Callable[..., None], *args: Any) -> dict:
    try:
        check(*args)
    except NotChecked as skipped:
        return {"check": name, "status": NOT_CHECKED, "reason": str(skipped)}
    except Refusal as refusal:
        return {"check": name, "status": FAIL, "code": refusal.code, "reason": refusal.reason,
                "details": refusal.details}
    return {"check": name, "status": PASS}


# --- the checks ----------------------------------------------------------------------


def check_canonical(raw: bytes, record: dict) -> None:
    if raw != canonical_dumps(record) + b"\n":
        raise Refusal("LEDGER_RECORD_NOT_CANONICAL",
                      "the record file is not exactly its canonical JSON followed by one newline",
                      size=len(raw))


def check_seal(record: dict) -> None:
    recomputed = compute_seal(record)
    if record.get("seal") != recomputed:
        raise Refusal("LEDGER_SEAL_MISMATCH", "the recorded seal is not the digest of the record's content",
                      recorded=record.get("seal"), recomputed=recomputed)


def check_expected_seal(record: dict, expect_seal: str | None) -> None:
    if expect_seal is None:
        raise NotChecked("no anchored seal was given (expect_seal)")
    recomputed = compute_seal(record)
    if recomputed != expect_seal:
        raise Refusal("LEDGER_SEAL_UNEXPECTED", f"the record's content has seal {recomputed}, "
                      f"the anchored seal is {expect_seal}", expected=expect_seal, recomputed=recomputed)


def check_schema(record: dict) -> None:
    if record.get("schema") != SCHEMA:
        raise Refusal("LEDGER_SCHEMA_UNSUPPORTED", f"schema {record.get('schema')!r} is not {SCHEMA!r}",
                      schema=record.get("schema"))
    problems = record_problems(record)
    if problems:
        raise Refusal("LEDGER_RECORD_MALFORMED", f"{len(problems)} structural problem(s), first: {problems[0]}",
                      field=problems[0].split(":", 1)[0], problems=problems)


def check_config_digest(record: dict) -> None:
    recomputed = digest_json(record["config"])
    if recomputed != record["config_digest"]:
        raise Refusal("LEDGER_CONFIG_DIGEST_MISMATCH", "config_digest is not the digest of the configuration",
                      field="config_digest", recorded=record["config_digest"], recomputed=recomputed)


def check_seeds(record: dict) -> None:
    for name, value in sorted(record["seeds"].items()):
        problem = seed_value_problem(value)
        if problem:
            raise Refusal("LEDGER_SEED_INVALID", f"seed {name!r}: {problem}", seed=name)
        problem = seed_config_problem(record["config"], record["seed_pointers"][name], value)
        if problem:
            raise Refusal("LEDGER_SEED_CONFIG_MISMATCH", f"seed {name!r}: {problem}", seed=name,
                          pointer=record["seed_pointers"][name])


def check_file(path: Path, digest: str, size: int | None, *, what: str, allow_missing: bool = False) -> None:
    """``path`` is a regular file with ``digest`` (and ``size`` unless ``None``)."""
    try:
        actual_digest, actual_size = digest_and_size(path)
    except NotRegularFile as exc:
        if allow_missing:
            raise NotChecked(f"{what}: no regular file at {path} (allowed by allow_missing_inputs)") from exc
        raise Refusal("LEDGER_FILE_MISSING", f"{what}: no regular file at {path} ({exc})", what=what, path=str(path))
    except OSError as exc:
        raise Refusal("LEDGER_FILE_UNREADABLE", f"{what}: {path} cannot be read: {exc}", what=what, path=str(path),
                      errno=exc.errno)
    if size is not None and actual_size != size:
        raise Refusal("LEDGER_FILE_SIZE_MISMATCH", f"{what}: {path} has {actual_size} bytes, recorded {size}",
                      what=what, path=str(path), recorded=size, actual=actual_size)
    if actual_digest != digest:
        raise Refusal("LEDGER_FILE_DIGEST_MISMATCH", f"{what}: {path} has digest {actual_digest}, recorded {digest}",
                      what=what, path=str(path), recorded=digest, actual=actual_digest)


def check_input(entry: dict, allow_missing: bool) -> None:
    if entry["path"] is None:
        raise NotChecked(f"input {entry['name']!r} is remote-only ({entry['uri']}); verify never fetches")
    check_file(Path(entry["path"]), entry["digest"], entry["size"], what=f"input {entry['name']!r}",
               allow_missing=allow_missing)


def check_model(entry: dict, allow_missing: bool) -> None:
    check_file(Path(entry["path"]), entry["digest"], entry["size"], what=f"model {entry['name']!r}",
               allow_missing=allow_missing)


def check_output(run_dir: Path, entry: dict) -> None:
    check_file(resolve_in_run_dir(run_dir, entry["path"]), entry["digest"], entry["size"],
               what=f"output {entry['name']!r}")


def check_chunk_manifest(run_dir: Path, entry: dict) -> None:
    what = f"chunk manifest {entry['name']!r}"
    path = resolve_in_run_dir(run_dir, entry["manifest_file"])
    check_file(path, entry["manifest_file_digest"], None, what=what)
    try:
        entries = manifest.parse_jsonl(read_regular(path))
    except (OSError, ValueError) as exc:
        raise Refusal("LEDGER_MANIFEST_INVALID", f"{what}: sidecar {entry['manifest_file']}: {exc}",
                      manifest=entry["name"], problem=str(exc))
    root = manifest_root(entries)
    if root != entry["root"]:
        raise Refusal("LEDGER_MANIFEST_ROOT_MISMATCH", f"{what}: the sidecar's Merkle root is {root}, "
                      f"recorded {entry['root']}", manifest=entry["name"], recorded=entry["root"], recomputed=root)
    present, absent = manifest.counts(entries)
    if (present, absent) != (entry["count_present"], entry["count_absent"]):
        raise Refusal("LEDGER_MANIFEST_COUNT_MISMATCH",
                      f"{what}: the sidecar has {present} present and {absent} absent entries, recorded "
                      f"{entry['count_present']} and {entry['count_absent']}", manifest=entry["name"],
                      recorded=[entry["count_present"], entry["count_absent"]], recomputed=[present, absent])


def _local_store_root(uri: str) -> Path | None:
    """The directory a local ``file://`` URI names, or None for any other URI."""
    parts = urllib.parse.urlsplit(uri)
    if parts.scheme != "file" or parts.netloc not in ("", "localhost") or parts.query or parts.fragment:
        return None
    path = urllib.request.url2pathname(parts.path)
    return Path(path) if os.path.isabs(path) else None


def check_chunk_data(run_dir: Path, entry: dict, allow_missing: bool) -> None:
    """Re-read every chunk of a local store and compare it with the sidecar (see the module docstring)."""
    from apok_eval.io.store import LocalStore  # only here: the ledger itself reads no volume

    what = f"chunk manifest {entry['name']!r}"
    root = _local_store_root(entry["source_uri"])
    if root is None:
        raise NotChecked(f"{entry['source_uri']!r} is not a local file:// store; verify never fetches, so the chunk "
                         "bytes are covered by the sidecar's digests and the seal only")
    try:
        entries = manifest.parse_jsonl(read_regular(resolve_in_run_dir(run_dir, entry["manifest_file"])))
    except (OSError, ValueError, Refusal) as exc:
        raise NotChecked(f"the sidecar cannot be read ({exc}); see chunk_manifest:{entry['name']}") from None
    try:
        kind = entry_kind(root, follow_symlinks=True)
    except (OSError, ValueError) as exc:  # EACCES on a parent, ENAMETOOLONG: never taken as missing
        raise Refusal("LEDGER_FILE_UNREADABLE", f"{what}: the local store {root} cannot be looked at: {exc}",
                      what=what, path=str(root), errno=getattr(exc, "errno", None)) from None
    if kind != "directory":
        if allow_missing:
            raise NotChecked(f"the local store {root} is not there (allow_missing_inputs)")
        raise Refusal("LEDGER_FILE_MISSING", f"{what}: the local store {root} it was read from is not a directory",
                      what=what, path=str(root))
    store = LocalStore(root)
    differing = []
    for chunk in entries:
        try:
            data = store.get(chunk["key"])
        except Refusal as exc:  # the store refuses to read the key now (unreadable, escaping, not a file)
            differing.append({"key": chunk["key"], "found": f"{exc.code}: {exc.reason}"})
            continue
        found = None if data is None else {"digest": sha256_bytes(data), "size": len(data)}
        recorded = None if chunk.get("absent") else {"digest": chunk["digest"], "size": chunk["size"]}
        if found != recorded:
            differing.append({"key": chunk["key"], "recorded": recorded or "absent", "found": found or "absent"})
    if differing:
        raise Refusal("LEDGER_CHUNK_MISMATCH", f"{what}: {len(differing)} of the {len(entries)} chunks read again from "
                      f"{root} differ from the record, first {differing[0]['key']!r}", manifest=entry["name"],
                      store=str(root), count=len(differing), differing=differing[:20])


def _differing_members(recorded: dict, actual: Any) -> list[str]:
    if not isinstance(actual, dict):
        return sorted(recorded)
    names = set(recorded) | set(actual)
    return sorted(n for n in names if n not in recorded or n not in actual
                  or canonical_dumps(recorded[n]) != canonical_dumps(actual[n]))


def check_figures(run_dir: Path, record: dict) -> None:
    figures = record["figures"]
    if figures is None:
        raise NotChecked("the record quotes no figures")
    source = next(o for o in record["outputs"] if o["name"] == figures["source"])
    path = resolve_in_run_dir(run_dir, source["path"])
    try:
        actual = read_json_at(path, figures["pointer"])
    except (OSError, ValueError, LookupError) as exc:
        raise Refusal("LEDGER_FIGURES_INVALID", f"figures source output {figures['source']!r} ({source['path']}) "
                      f"does not yield a value at {figures['pointer']!r}: {exc}", source=figures["source"],
                      pointer=figures["pointer"])
    if canonical_dumps(actual) != canonical_dumps(figures["values"]):
        raise Refusal("LEDGER_FIGURES_MISMATCH", f"the recorded figures differ from {source['path']} at "
                      f"{figures['pointer']!r}", source=figures["source"],
                      differing=_differing_members(figures["values"], actual))


def check_predictions(record: dict) -> None:
    predictions = record["predictions"]
    if predictions is None:
        raise NotChecked("the record lists no pre-registered predictions")
    registry = next(i for i in record["inputs"] if i["name"] == predictions["registry_input"])
    if registry["digest"] != predictions["registry_digest"]:
        raise Refusal("LEDGER_PREDICTIONS_MISMATCH",
                      f"predictions registry digest {predictions['registry_digest']} is not the digest of input "
                      f"{registry['name']!r} ({registry['digest']})", registry_input=registry["name"],
                      recorded=predictions["registry_digest"], input_digest=registry["digest"])


def check_source_root(record: dict) -> None:
    code = record["code"]
    recomputed = gitcode.source_root_digest(code["source_manifest"])
    if recomputed != code["source_manifest_root"]:
        raise Refusal("LEDGER_SOURCE_ROOT_MISMATCH", "source_manifest_root is not the Merkle root of the source "
                      "manifest", field="code.source_manifest_root", recorded=code["source_manifest_root"],
                      recomputed=recomputed)


def check_code_diff(run_dir: Path, record: dict) -> None:
    code = record["code"]
    if code["dirty"] is not True:
        raise NotChecked("no diff is recorded (the tree was clean, or not a git checkout)")
    check_file(resolve_in_run_dir(run_dir, code["diff_file"]), code["diff_digest"], None, what="code diff")


def _default_repo() -> Path | None:
    return gitcode.toplevel(Path(apok_eval.__file__).resolve().parent)


def _recorded_diff(run_dir: Path, code: dict) -> bytes:
    """The bytes of ``code.diff`` if they have ``diff_digest``; NOT CHECKED otherwise (``code.diff`` fails then)."""
    try:
        data = read_regular(resolve_in_run_dir(run_dir, code["diff_file"]))
    except (Refusal, OSError) as exc:
        raise NotChecked(f"the recorded diff {code['diff_file']} cannot be read ({exc}); see the code.diff check") \
            from exc
    if sha256_bytes(data) != code["diff_digest"]:
        raise NotChecked(f"the recorded diff {code['diff_file']} does not have diff_digest; see the code.diff check")
    return data


def check_commit(run_dir: Path, record: dict, enabled: bool, repo: str | os.PathLike[str] | None) -> None:
    code = record["code"]
    if not enabled:
        raise NotChecked("the code check was disabled")
    if code["commit"] is None:
        raise NotChecked("the record names no commit: the package source was not tracked by a git checkout")
    diff = _recorded_diff(run_dir, code) if code["dirty"] else None
    try:
        where = repo if repo is not None else _default_repo()
        if where is None:
            raise NotChecked(f"no git repository is available to look up commit {code['commit']}")
        try:
            available = gitcode.commit_available(where, code["commit"])
        except gitcode.GitUnavailable:
            raise
        except gitcode.GitError as exc:
            if repo is not None:  # an explicit repository git cannot open is a mistake, not a reason to skip
                raise Refusal("LEDGER_REPO_INVALID", f"the repository {repo} given to verify cannot be opened by "
                              f"git: {exc}", repo=str(repo), commit=code["commit"], error=str(exc)) from None
            raise
        if not available:
            if repo is not None:
                raise Refusal("LEDGER_COMMIT_MISSING", f"commit {code['commit']} is not in the repository {repo} "
                              "given to verify", commit=code["commit"], repo=str(repo))
            raise NotChecked(f"commit {code['commit']} is not in the repository at {where} (the work tree of the "
                             "running apok_eval package); pass the run's repository as repo to check it")
        expected = (gitcode.commit_manifest(where, code["commit"], code["source_root"]) if diff is None
                    else gitcode.applied_manifest(where, code["commit"], code["source_root"], diff))
    except gitcode.PatchError as exc:
        raise Refusal("LEDGER_CODE_MISMATCH", f"the recorded diff does not apply to commit {code['commit']}: {exc}",
                      commit=code["commit"], diff_does_not_apply=str(exc))
    except gitcode.GitError as exc:  # git missing (GitUnavailable), or failing after the repository was opened
        raise NotChecked(f"git failed: {exc}") from exc
    differences = gitcode.compare_manifests(code["source_manifest"], expected)
    if differences:
        raise Refusal("LEDGER_CODE_MISMATCH", f"the source manifest differs from commit {code['commit']} under "
                      f"{code['source_root']!r}", commit=code["commit"], **differences)


def check_environment(record: dict) -> None:
    raise NotChecked("the environment cannot be recomputed; it is covered by the seal only")


# --- running them ------------------------------------------------------------------------


def check_record(record: dict, run_dir: str | os.PathLike[str], *, expect_seal: str | None = None,
                 check_code: bool = True, allow_missing_inputs: bool = False,
                 repo: str | os.PathLike[str] | None = None) -> list[dict]:
    """Outcomes of every check after ``record``/``canonical`` for an in-memory record."""
    run_dir = Path(run_dir)
    results = [_outcome("seal", check_seal, record),
               _outcome("expected_seal", check_expected_seal, record, expect_seal)]
    schema = _outcome("schema", check_schema, record)
    results.append(schema)
    if schema["status"] != PASS:
        results.append({"check": "content", "status": NOT_CHECKED,
                        "reason": "the record's structure is invalid (see schema); its content was not checked"})
        return results
    results.append(_outcome("config_digest", check_config_digest, record))
    results.append(_outcome("seeds", check_seeds, record))
    results += [_outcome(f"input:{e['name']}", check_input, e, allow_missing_inputs) for e in record["inputs"]]
    results += [_outcome(f"model:{e['name']}", check_model, e, allow_missing_inputs) for e in record["models"]]
    results += [_outcome(f"output:{e['name']}", check_output, run_dir, e) for e in record["outputs"]]
    for e in record["chunk_manifests"]:
        results.append(_outcome(f"chunk_manifest:{e['name']}", check_chunk_manifest, run_dir, e))
        results.append(_outcome(f"chunk_data:{e['name']}", check_chunk_data, run_dir, e, allow_missing_inputs))
    results.append(_outcome("figures", check_figures, run_dir, record))
    results.append(_outcome("predictions", check_predictions, record))
    results.append(_outcome("code.source_manifest_root", check_source_root, record))
    results.append(_outcome("code.diff", check_code_diff, run_dir, record))
    results.append(_outcome("code.commit", check_commit, run_dir, record, check_code, repo))
    results.append(_outcome("environment", check_environment, record))
    return results


def _report(record_path: Path, seal: Any, results: list[dict]) -> dict:
    counts = {status: sum(r["status"] == status for r in results) for status in (PASS, FAIL, NOT_CHECKED)}
    return {"record": str(record_path), "seal": seal, "ok": counts[FAIL] == 0, "counts": counts, "checks": results}


def run_verification(record_path: str | os.PathLike[str], *, expect_seal: str | None = None,
                     check_code: bool = True, allow_missing_inputs: bool = False,
                     repo: str | os.PathLike[str] | None = None) -> dict:
    """Run every check on the record file at ``record_path``; return the report without raising.

    ``repo`` is where ``code.commit`` is looked up; by default, the git work
    tree holding the running ``apok_eval`` package, if any.
    """
    path = Path(record_path)
    loaded: dict[str, Any] = {}

    def load() -> None:
        loaded["raw"], loaded["record"] = read_record(path)

    results = [_outcome("record", load)]
    if "record" not in loaded:
        return _report(path, None, results)
    record = loaded["record"]
    results.append(_outcome("canonical", check_canonical, loaded["raw"], record))
    results += check_record(record, path.parent, expect_seal=expect_seal, check_code=check_code,
                            allow_missing_inputs=allow_missing_inputs, repo=repo)
    return _report(path, record.get("seal"), results)


def summarise(report: dict, what: str) -> dict:
    """Return ``report`` if nothing failed, else raise ``LEDGER_VERIFY_FAILED`` listing the failures."""
    failed = [{key: r[key] for key in ("check", "code", "reason", "details")}
              for r in report["checks"] if r["status"] == FAIL]
    if failed:
        raise Refusal("LEDGER_VERIFY_FAILED",
                      f"{what}: {len(failed)} check(s) failed: "
                      + ", ".join(f"{f['check']} ({f['code']})" for f in failed),
                      record=report["record"], failed=failed,
                      not_checked=[{"check": r["check"], "reason": r["reason"]}
                                   for r in report["checks"] if r["status"] == NOT_CHECKED])
    return report


def verify(record_path: str | os.PathLike[str], *, expect_seal: str | None = None, check_code: bool = True,
           allow_missing_inputs: bool = False, repo: str | os.PathLike[str] | None = None) -> dict:
    """Verify a sealed record; return the report, or raise ``LEDGER_VERIFY_FAILED``."""
    report = run_verification(record_path, expect_seal=expect_seal, check_code=check_code,
                              allow_missing_inputs=allow_missing_inputs, repo=repo)
    return summarise(report, f"record {record_path}")
