"""``verify``: the ledger's defining property, literally.

Changing any recorded byte, digest or figure makes ``verify`` fail:

* every byte of ``record.json`` flipped in turn;
* every leaf of the record changed (seal left stale) -> ``LEDGER_SEAL_MISMATCH``;
* every leaf changed AND the record re-sealed -> either a named recomputation
  check fails with the expected code and names the cause in its details (and
  the seal check passes), or - for the leaves only the seal protects - plain
  verify passes and ``expect_seal`` with the original seal fails; for a run
  sealed from a clean tree and for one sealed from a dirty tree;
* one byte flipped in each input, model, output, the chunk-manifest sidecar
  and a figure value -> the check of that file fails with the right code;
* sidecar lines deleted, reordered, added -> fail.
"""

from __future__ import annotations

import copy
import importlib
import datetime as dt
import errno
import json
import os
from pathlib import Path

import pytest

from apok_eval.canonical import canonical_dumps, digest_json, is_digest, sha256_bytes
from apok_eval.cli import main
from apok_eval.ledger import verify
from apok_eval.ledger.gitcode import source_root_digest
from apok_eval.ledger.record import compute_seal
from apok_eval.ledger.verify import FAIL, NOT_CHECKED, PASS, run_verification
from apok_eval.merkle import manifest_root
from apok_eval.refusal import Refusal

from ledger_fixtures import (
    FIGURES,
    build_run,
    deadline,
    failed_codes,
    failures,
    git,
    make_dirty_source_repo,
    make_source_repo,
    status_of,
    write_record,
)

verify_module = importlib.import_module("apok_eval.ledger.verify")  # the package exports a function `verify`
SEAL_ONLY = "seal-only"
MISSING_COMMIT = "0123456789abcdef" * 2 + "01234567"  # well formed, in no repository


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    return make_source_repo(tmp_path_factory.mktemp("ledger-source"))


@pytest.fixture(scope="module")
def sealed(tmp_path_factory, source):
    """A sealed run that tests only read (tampered records are written beside it under other names)."""
    return build_run(tmp_path_factory.mktemp("ledger-run"), source)


@pytest.fixture
def fresh(tmp_path, source):
    """A sealed run of the test's own, for tests that change files."""
    return build_run(tmp_path, source)


@pytest.fixture(scope="module")
def dirty_source(tmp_path_factory):
    return make_dirty_source_repo(tmp_path_factory.mktemp("ledger-dirty-source"))


@pytest.fixture(scope="module")
def sealed_dirty(tmp_path_factory, dirty_source):
    """A run sealed from a dirty tree (allow_dirty), read-only like ``sealed``."""
    return build_run(tmp_path_factory.mktemp("ledger-dirty-run"), dirty_source, allow_dirty=True)


@pytest.fixture
def fresh_dirty(tmp_path):
    source = make_dirty_source_repo(tmp_path / "source")
    return build_run(tmp_path, source, allow_dirty=True)


def refusal_of(path, **kwargs) -> Refusal:
    with pytest.raises(Refusal) as info:
        verify(path, **kwargs)
    return info.value


# --- the untouched run verifies --------------------------------------------------------


def test_sealed_run_verifies_with_every_check_accounted_for(sealed, source):
    report = verify(sealed.record_path, repo=source.repo)
    assert report["ok"] and report["seal"] == sealed.seal
    status = {c["check"]: c["status"] for c in report["checks"]}
    assert status == {
        "record": PASS, "canonical": PASS, "seal": PASS, "expected_seal": NOT_CHECKED, "schema": PASS,
        "config_digest": PASS, "seeds": PASS, "input:labels": PASS, "input:predictions": PASS,
        "input:remote-meta": NOT_CHECKED, "input:volume": PASS, "model:winding": PASS, "output:mesh": PASS,
        "output:metrics": PASS, "chunk_manifest:volume": PASS, "chunk_data:volume": NOT_CHECKED, "figures": PASS,
        "predictions": PASS,
        "code.source_manifest_root": PASS, "code.diff": NOT_CHECKED, "code.commit": PASS,
        "environment": NOT_CHECKED,
    }
    assert "never fetches" in status_of(report, "input:remote-meta")["reason"]
    assert "never fetches" in status_of(report, "chunk_data:volume")["reason"]  # a remote store's chunks
    assert "covered by the seal only" in status_of(report, "environment")["reason"]
    anchored = verify(sealed.record_path, expect_seal=sealed.seal, repo=source.repo)
    assert anchored["counts"] == {PASS: 18, FAIL: 0, NOT_CHECKED: 4}


# --- every byte -----------------------------------------------------------------------------


def test_flipping_any_byte_of_the_record_fails_verify(sealed):
    raw = sealed.record_path.read_bytes()
    target = sealed.run_dir / "flipped.json"
    for offset in range(len(raw)):
        flipped = bytearray(raw)
        flipped[offset] ^= 0x01
        target.write_bytes(bytes(flipped))
        report = run_verification(target, check_code=False)
        assert not report["ok"], f"flipping byte {offset} ({raw[offset:offset + 1]!r}) went unnoticed"


# --- every leaf -------------------------------------------------------------------------------


def leaves(obj, path=()):
    if isinstance(obj, dict) and obj:
        for key in sorted(obj):
            yield from leaves(obj[key], path + (key,))
    elif isinstance(obj, list) and obj:
        for i, item in enumerate(obj):
            yield from leaves(item, path + (i,))
    else:
        yield path, obj


def pattern(path) -> str:
    parts: list[str] = []
    for part in path:
        if isinstance(part, int):
            parts[-1] += "[]"
        else:
            parts.append(part)
    return ".".join(parts)


def set_at(record: dict, path, value) -> dict:
    out = copy.deepcopy(record)
    target = out
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    return out


def tamper(pat: str, value, sealed_run):
    """A type-appropriate change that keeps the record well formed where possible."""
    if pat == "schema":
        return "apok-record-v2"
    if pat == "code.commit":
        return MISSING_COMMIT
    if pat == "created_utc":
        fmt = "%Y-%m-%dT%H:%M:%SZ"
        return (dt.datetime.strptime(value, fmt) + dt.timedelta(seconds=1)).strftime(fmt)
    if pat == "figures.source":
        return "mesh"
    if pat == "predictions.registry_input":
        return "labels"
    if pat == "inputs[].path" and value is None:
        return "/nonexistent/remote-meta"
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.5
    if value is None:
        return "tampered"
    if is_digest(value):
        return sha256_bytes(f"tampered {pat} {value}".encode())
    return value + "x"


SINGULAR = {"inputs": "input", "models": "model", "outputs": "output", "chunk_manifests": "chunk manifest"}


def _in_run_dir(run, rel: str) -> str:
    return str(Path(os.path.realpath(run.run_dir)).joinpath(*rel.split("/")))


def expected_after_reseal(path, record: dict, tampered: dict, run, repo):
    """``(check, code, details subset)`` a re-sealed tamper of ``path`` must fail with, or SEAL_ONLY.

    The details subset is the named cause: the field, file, entry or value the
    check must report (test the reason, not only the refusal).
    """
    pat = pattern(path)
    leaf = tampered
    for part in path:
        leaf = leaf[part]
    if path[0] in SINGULAR:
        entry, changed = record[path[0]][path[1]], tampered[path[0]][path[1]]
        check = f"{SINGULAR[path[0]].replace(' ', '_')}:{entry['name']}"
        what = f"{SINGULAR[path[0]]} {entry['name']!r}"
    code = tampered["code"]
    if pat == "schema":
        return "schema", "LEDGER_SCHEMA_UNSUPPORTED", {"schema": leaf}
    if pat in ("run_id", "created_utc", "purpose", "code.package_version", "predictions.ids[]") \
            or pat.startswith("environment."):
        return SEAL_ONLY
    if pat == "inputs[].name":
        return ("schema", "LEDGER_RECORD_MALFORMED", {"field": "predictions.registry_input"}) \
            if entry["name"] == "predictions" else SEAL_ONLY
    if pat == "outputs[].name":
        return ("schema", "LEDGER_RECORD_MALFORMED", {"field": "figures.source"}) \
            if entry["name"] == "metrics" else SEAL_ONLY
    if pat in ("inputs[].uri", "models[].name", "models[].uri", "models[].source", "chunk_manifests[].name",
               "chunk_manifests[].source_uri"):
        return SEAL_ONLY
    if pat == "code.commit":
        return "code.commit", "LEDGER_COMMIT_MISSING", {"commit": MISSING_COMMIT, "repo": str(repo)}
    if pat == "code.source_root":
        return "code.commit", "LEDGER_CODE_MISMATCH", {"commit": code["commit"]}
    if pat == "code.dirty" or (pat in ("code.diff_digest", "code.diff_file") and not record["code"]["dirty"]):
        return "schema", "LEDGER_RECORD_MALFORMED", {"field": "code.diff_digest" if pat != "code.diff_file"
                                                     else "code.diff_file"}
    if pat == "code.diff_digest":
        return "code.diff", "LEDGER_FILE_DIGEST_MISMATCH", {"what": "code diff", "recorded": leaf}
    if pat == "code.diff_file":
        return "code.diff", "LEDGER_FILE_MISSING", {"what": "code diff", "path": _in_run_dir(run, leaf)}
    if pat == "chunk_manifests[].scheme":
        return "schema", "LEDGER_RECORD_MALFORMED", {"field": f"chunk_manifests[{path[1]}].scheme"}
    if pat.startswith("code.source_manifest"):
        return "code.source_manifest_root", "LEDGER_SOURCE_ROOT_MISMATCH", {
            "field": "code.source_manifest_root", "recorded": code["source_manifest_root"],
            "recomputed": source_root_digest(code["source_manifest"])}
    if pat == "config_digest" or pat.startswith("config."):
        return "config_digest", "LEDGER_CONFIG_DIGEST_MISMATCH", {
            "field": "config_digest", "recorded": tampered["config_digest"],
            "recomputed": digest_json(tampered["config"])}
    if pat.startswith(("seeds.", "seed_pointers.")):
        name = path[1]
        return "seeds", "LEDGER_SEED_CONFIG_MISMATCH", {"seed": name, "pointer": tampered["seed_pointers"][name]}
    if pat in ("inputs[].digest", "inputs[].size") and entry["path"] is None:
        return SEAL_ONLY  # remote-only: verify never fetches, so only the seal covers its bytes
    if path[0] in SINGULAR:
        field = path[-1]
        if field in ("path", "manifest_file"):
            where = changed[field] if path[0] in ("inputs", "models") else _in_run_dir(run, changed[field])
            return check, "LEDGER_FILE_MISSING", {"what": what, "path": where}
        located = changed["path"] if path[0] in ("inputs", "models") else \
            _in_run_dir(run, changed["manifest_file"] if path[0] == "chunk_manifests" else changed["path"])
        if field in ("digest", "manifest_file_digest"):
            return check, "LEDGER_FILE_DIGEST_MISMATCH", {"what": what, "path": located, "recorded": leaf}
        if field == "size":
            return check, "LEDGER_FILE_SIZE_MISMATCH", {"what": what, "path": located, "recorded": leaf}
        if field == "root":
            return check, "LEDGER_MANIFEST_ROOT_MISMATCH", {"manifest": entry["name"], "recorded": leaf}
        if field in ("count_present", "count_absent"):
            return check, "LEDGER_MANIFEST_COUNT_MISMATCH", {
                "manifest": entry["name"], "recorded": [changed["count_present"], changed["count_absent"]]}
    if pat in ("figures.source", "figures.pointer"):
        figures = tampered["figures"]
        return "figures", "LEDGER_FIGURES_INVALID", {"source": figures["source"], "pointer": figures["pointer"]}
    if pat.startswith("figures.values."):
        return "figures", "LEDGER_FIGURES_MISMATCH", {"source": "metrics", "differing": [path[2]]}
    if pat in ("predictions.registry_input", "predictions.registry_digest"):
        predictions = tampered["predictions"]
        return "predictions", "LEDGER_PREDICTIONS_MISMATCH", {
            "registry_input": predictions["registry_input"], "recorded": predictions["registry_digest"]}
    raise AssertionError(f"leaf {pat} has no expectation; every leaf must be classified")


def subset(details: dict, wanted: dict) -> dict:
    return {key: details.get(key) for key in wanted}


@pytest.mark.parametrize("tree", ["clean", "dirty"])
def test_every_leaf_is_protected(request, tree):
    run = request.getfixturevalue("sealed" if tree == "clean" else "sealed_dirty")
    repo = request.getfixturevalue("source" if tree == "clean" else "dirty_source").repo
    record = run.record()
    assert record["code"]["dirty"] is (tree == "dirty")
    target = run.run_dir / "tampered.json"
    tally = {"leaves": 0, "recomputed": 0, "seal_only": 0}
    for path, value in leaves(record):
        tally["leaves"] += 1
        pat = pattern(path)
        changed = tamper(pat, value, run)
        assert changed != value or type(changed) is not type(value), pat
        tampered = set_at(record, path, changed)

        # 1. changed, seal left as it was: the seal no longer matches, and the check names both seals.
        write_record(target, tampered, reseal=False)
        found = failures(refusal_of(target, repo=repo))
        assert found["seal"]["code"] == "LEDGER_SEAL_MISMATCH", pat
        assert found["seal"]["details"] == {"recorded": tampered["seal"], "recomputed": compute_seal(tampered)}, pat

        if path == ("seal",):
            continue
        # 2. changed and re-sealed: a recomputation check names the change, or only an anchored seal can.
        resealed = write_record(target, tampered, reseal=True)
        expected = expected_after_reseal(path, record, tampered, run, repo)
        if expected == SEAL_ONLY:
            tally["seal_only"] += 1
            assert verify(target, repo=repo)["ok"], pat
            found = failures(refusal_of(target, repo=repo, expect_seal=run.seal))
            assert set(found) == {"expected_seal"} and found["expected_seal"]["code"] == "LEDGER_SEAL_UNEXPECTED", pat
            assert found["expected_seal"]["details"] == {"expected": run.seal, "recomputed": resealed["seal"]}, pat
        else:
            tally["recomputed"] += 1
            check, code, cause = expected
            found = failures(refusal_of(target, repo=repo))
            assert "seal" not in found, pat
            assert check in found and found[check]["code"] == code, (pat, expected, found)
            assert subset(found[check]["details"], cause) == cause, (pat, found[check]["details"])
    assert tally["leaves"] == tally["recomputed"] + tally["seal_only"] + 1
    assert tally["recomputed"] > 2 * tally["seal_only"]
    print(f"\nleaf tally ({tree} tree): {tally}")


def test_a_fully_consistent_rewrite_needs_the_anchored_seal(fresh):
    # Drop the absent chunk from the sidecar and recompute every number the record states about it.
    sidecar = fresh.files["sidecar"]
    kept = [line for line in sidecar.read_bytes().split(b"\n")[:-1] if b'"absent"' not in line or b"1.0.0" in line]
    sidecar.write_bytes(b"".join(line + b"\n" for line in kept))
    record = fresh.record()
    entries = [json.loads(line) for line in kept]
    record["chunk_manifests"][0].update(manifest_file_digest=sha256_bytes(sidecar.read_bytes()),
                                        root=manifest_root(entries), count_absent=1)
    write_record(fresh.record_path, record, reseal=True)
    assert verify(fresh.record_path, check_code=False)["ok"]
    codes = failed_codes(refusal_of(fresh.record_path, check_code=False, expect_seal=fresh.seal))
    assert codes == {("expected_seal", "LEDGER_SEAL_UNEXPECTED")}


# --- every file -----------------------------------------------------------------------------


@pytest.mark.parametrize("which, check", [
    ("volume", "input:volume"), ("labels", "input:labels"), ("predictions", "input:predictions"),
    ("model", "model:winding"), ("metrics", "output:metrics"), ("mesh", "output:mesh"),
    ("sidecar", "chunk_manifest:volume"),
])
def test_flipping_one_byte_of_a_recorded_file_fails_its_check(fresh, which, check):
    path = fresh.files[which]
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x20
    path.write_bytes(bytes(data))
    found = failures(refusal_of(fresh.record_path, check_code=False))
    assert found[check]["code"] == "LEDGER_FILE_DIGEST_MISMATCH"
    assert found[check]["details"]["path"] == str(path)


def test_changing_a_figure_in_the_output_fails_twice(fresh):
    path = fresh.files["metrics"]
    text = path.read_text()
    assert text.count("0.93") == 1
    path.write_text(text.replace("0.93", "0.83"))
    found = failures(refusal_of(fresh.record_path, check_code=False))
    assert found["output:metrics"]["code"] == "LEDGER_FILE_DIGEST_MISMATCH"
    assert found["figures"]["code"] == "LEDGER_FIGURES_MISMATCH"
    assert found["figures"]["details"]["differing"] == ["achieved_rate"]


def test_a_file_of_another_size_is_named_by_its_size(fresh):
    fresh.files["mesh"].write_bytes(fresh.files["mesh"].read_bytes() + b"!")
    found = failures(refusal_of(fresh.record_path, check_code=False))
    assert found["output:mesh"]["code"] == "LEDGER_FILE_SIZE_MISMATCH"
    assert found["output:mesh"]["details"]["actual"] == found["output:mesh"]["details"]["recorded"] + 1


def test_missing_files(fresh):
    for name in ("volume", "model", "mesh"):
        fresh.files[name].unlink()
    found = failures(refusal_of(fresh.record_path, check_code=False))
    assert {k: v["code"] for k, v in found.items()} == {
        "input:volume": "LEDGER_FILE_MISSING", "model:winding": "LEDGER_FILE_MISSING",
        "output:mesh": "LEDGER_FILE_MISSING"}
    assert found["input:volume"]["details"]["path"] == str(fresh.files["volume"])


def test_allow_missing_inputs_marks_inputs_and_models_not_checked_but_never_outputs(fresh):
    fresh.files["volume"].unlink()
    fresh.files["model"].unlink()
    report = verify(fresh.record_path, check_code=False, allow_missing_inputs=True)
    for check in ("input:volume", "model:winding"):
        assert status_of(report, check)["status"] == NOT_CHECKED
        assert "allow_missing_inputs" in status_of(report, check)["reason"]
    fresh.files["mesh"].unlink()
    found = failures(refusal_of(fresh.record_path, check_code=False, allow_missing_inputs=True))
    assert set(found) == {"output:mesh"} and found["output:mesh"]["code"] == "LEDGER_FILE_MISSING"


def test_allow_missing_inputs_still_checks_present_inputs(fresh):
    fresh.files["labels"].write_text("{}")
    found = failures(refusal_of(fresh.record_path, check_code=False, allow_missing_inputs=True))
    assert found["input:labels"]["code"] in ("LEDGER_FILE_SIZE_MISMATCH", "LEDGER_FILE_DIGEST_MISMATCH")


def test_an_output_reached_through_a_symlink_is_refused(fresh, tmp_path):
    mesh = fresh.files["mesh"]
    elsewhere = tmp_path / "elsewhere.bin"
    elsewhere.write_bytes(mesh.read_bytes())
    mesh.unlink()
    mesh.symlink_to(elsewhere)
    found = failures(refusal_of(fresh.record_path, check_code=False))
    assert found["output:mesh"]["code"] == "LEDGER_PATH_UNSAFE"
    assert found["output:mesh"]["details"]["path"] == "out/mesh.bin"


def test_figures_are_not_read_through_a_symlink(fresh, tmp_path):
    metrics = fresh.files["metrics"]
    elsewhere = tmp_path / "metrics-elsewhere.json"
    elsewhere.write_bytes(metrics.read_bytes())  # the same bytes: only the path is wrong
    metrics.unlink()
    metrics.symlink_to(elsewhere)
    found = failures(refusal_of(fresh.record_path, check_code=False))
    for check in ("output:metrics", "figures"):
        assert found[check]["code"] == "LEDGER_PATH_UNSAFE" and found[check]["details"]["path"] == "metrics.json"


def test_a_sidecar_reached_through_a_symlink_is_refused(fresh, tmp_path):
    sidecar = fresh.files["sidecar"]
    elsewhere = tmp_path / "volume-elsewhere.jsonl"
    elsewhere.write_bytes(sidecar.read_bytes())
    sidecar.unlink()
    sidecar.symlink_to(elsewhere)
    found = failures(refusal_of(fresh.record_path, check_code=False))
    assert found["chunk_manifest:volume"]["code"] == "LEDGER_PATH_UNSAFE"
    assert found["chunk_manifest:volume"]["details"]["path"] == "manifests/volume.jsonl"


def test_a_code_diff_reached_through_a_symlink_is_refused(fresh_dirty, tmp_path):
    diff = fresh_dirty.run_dir / "code.diff"
    elsewhere = tmp_path / "diff-elsewhere"
    elsewhere.write_bytes(diff.read_bytes())
    diff.unlink()
    diff.symlink_to(elsewhere)
    report = run_verification(fresh_dirty.record_path, repo=fresh_dirty.source.repo)
    check = status_of(report, "code.diff")
    assert check["code"] == "LEDGER_PATH_UNSAFE" and check["details"]["path"] == "code.diff"
    commit = status_of(report, "code.commit")
    assert commit["status"] == NOT_CHECKED and "see the code.diff check" in commit["reason"]


def test_an_output_directory_that_is_a_symlink_is_refused(fresh, tmp_path):
    out = fresh.run_dir / "out"
    moved = tmp_path / "moved-out"
    out.rename(moved)
    out.symlink_to(moved, target_is_directory=True)
    found = failures(refusal_of(fresh.record_path, check_code=False))
    assert found["output:mesh"]["code"] == "LEDGER_PATH_UNSAFE"


# --- the chunk-manifest sidecar ------------------------------------------------------------------


def _lines(fresh) -> list[bytes]:
    return fresh.files["sidecar"].read_bytes().split(b"\n")[:-1]


def _rewrite_sidecar(fresh, data: bytes, *, update_record: bool) -> None:
    fresh.files["sidecar"].write_bytes(data)
    if update_record:
        record = fresh.record()
        record["chunk_manifests"][0]["manifest_file_digest"] = sha256_bytes(data)
        write_record(fresh.record_path, record, reseal=True)


EXTRA_LINE = canonical_dumps({"key": "volume.zarr/0/0.0.2", "digest": sha256_bytes(b"x"), "size": 1})

SIDECAR_EDITS = {
    "delete a line": (lambda ls: ls[:2] + ls[3:], "LEDGER_MANIFEST_ROOT_MISMATCH"),
    "reorder two lines": (lambda ls: [ls[1], ls[0]] + ls[2:], "LEDGER_MANIFEST_INVALID"),
    "add a line": (lambda ls: ls[:5] + [EXTRA_LINE] + ls[5:], "LEDGER_MANIFEST_ROOT_MISMATCH"),
    "repeat a line": (lambda ls: ls[:3] + [ls[2]] + ls[3:], "LEDGER_MANIFEST_INVALID"),
    "non-canonical line": (lambda ls: [ls[0].replace(b",", b", ")] + ls[1:], "LEDGER_MANIFEST_INVALID"),
    "blank line": (lambda ls: ls[:1] + [b""] + ls[1:], "LEDGER_MANIFEST_INVALID"),
    "absent entry with a digest": (lambda ls: [line.replace(b'{"absent":true,', b'{"absent":true,"digest":"x",')
                                               for line in ls], "LEDGER_MANIFEST_INVALID"),
}


def test_the_sidecar_holds_present_and_absent_entries_in_merkle_order(sealed):
    lines = sealed.files["sidecar"].read_bytes().split(b"\n")[:-1]
    keys = [json.loads(line)["key"] for line in lines]
    assert keys == sorted(keys, key=lambda k: k.encode()) and len(keys) == 8
    assert sum(b'"absent":true' in line for line in lines) == 2


@pytest.mark.parametrize("edit", sorted(SIDECAR_EDITS))
def test_editing_sidecar_lines_fails(fresh, edit):
    transform, code_after_reseal = SIDECAR_EDITS[edit]
    edited = b"".join(line + b"\n" for line in transform(_lines(fresh)))
    _rewrite_sidecar(fresh, edited, update_record=False)
    found = failures(refusal_of(fresh.record_path, check_code=False))
    assert found["chunk_manifest:volume"]["code"] == "LEDGER_FILE_DIGEST_MISMATCH"
    # The adversary also updates the sidecar digest and re-seals: the content checks still catch it.
    _rewrite_sidecar(fresh, edited, update_record=True)
    found = failures(refusal_of(fresh.record_path, check_code=False))
    assert set(found) == {"chunk_manifest:volume"}
    assert found["chunk_manifest:volume"]["code"] == code_after_reseal
    assert found["chunk_manifest:volume"]["details"]["manifest"] == "volume"


def test_a_sidecar_line_is_named_when_invalid(fresh):
    lines = _lines(fresh)
    _rewrite_sidecar(fresh, b"".join(line + b"\n" for line in [lines[1], lines[0]] + lines[2:]),
                     update_record=True)
    found = failures(refusal_of(fresh.record_path, check_code=False))
    assert "line 2" in found["chunk_manifest:volume"]["reason"]


def test_a_torn_sidecar_fails(fresh):
    _rewrite_sidecar(fresh, fresh.files["sidecar"].read_bytes()[:-1], update_record=True)
    found = failures(refusal_of(fresh.record_path, check_code=False))
    assert found["chunk_manifest:volume"]["code"] == "LEDGER_MANIFEST_INVALID"
    assert "newline" in found["chunk_manifest:volume"]["reason"]


def test_counts_are_recomputed_even_when_the_root_is_updated(fresh):
    lines = _lines(fresh)
    kept = [line for line in lines if b"0.0.1" not in line]
    data = b"".join(line + b"\n" for line in kept)
    fresh.files["sidecar"].write_bytes(data)
    record = fresh.record()
    record["chunk_manifests"][0].update(manifest_file_digest=sha256_bytes(data),
                                        root=manifest_root([json.loads(line) for line in kept]))
    write_record(fresh.record_path, record, reseal=True)
    found = failures(refusal_of(fresh.record_path, check_code=False))
    assert found["chunk_manifest:volume"]["code"] == "LEDGER_MANIFEST_COUNT_MISMATCH"
    assert found["chunk_manifest:volume"]["details"]["recomputed"] == [6, 1]


def test_a_missing_sidecar_fails(fresh):
    fresh.files["sidecar"].unlink()
    found = failures(refusal_of(fresh.record_path, check_code=False))
    assert found["chunk_manifest:volume"]["code"] == "LEDGER_FILE_MISSING"


# --- reading the record itself ----------------------------------------------------------------


@pytest.mark.parametrize("content, problem", [
    (None, "NotRegularFile: "), (b"", "invalid JSON: Expecting value"), (b"not json", "invalid JSON: Expecting value"),
    (b"[1, 2]", "the top level is a list, not an object"), (b'{"a": 1, "a": 2}', "duplicate key 'a'"),
    (b'{"a": NaN}', "non-standard constant NaN"), (b'{"a": 1e400}', "number 1e400 overflows a double"),
    (b'{"a": "\xff"}', "not UTF-8"), (b'{"a": "\\ud800"}', "lone surrogate '\\ud800'"),
])
def test_an_unreadable_record_fails_the_record_check(tmp_path, content, problem):
    path = tmp_path / "record.json"
    if content is not None:
        path.write_bytes(content)
    refusal = refusal_of(path)
    assert failed_codes(refusal) == {("record", "LEDGER_RECORD_UNREADABLE")}
    details = failures(refusal)["record"]["details"]
    assert details["record"] == str(path) and details["problem"].startswith(problem)  # the parser's own words
    assert refusal.details["record"] == str(path)


def test_a_record_with_the_same_content_but_other_bytes_is_not_canonical(sealed):
    target = sealed.run_dir / "pretty.json"
    target.write_text(json.dumps(sealed.record(), indent=1, sort_keys=True) + "\n")
    report = run_verification(target, check_code=False)
    assert status_of(report, "seal")["status"] == PASS
    assert status_of(report, "canonical")["code"] == "LEDGER_RECORD_NOT_CANONICAL"
    target.write_bytes(sealed.record_path.read_bytes().rstrip(b"\n"))
    assert status_of(run_verification(target, check_code=False), "canonical")["code"] == \
        "LEDGER_RECORD_NOT_CANONICAL"


def test_expected_seal_compares_the_recomputed_seal(sealed):
    record = sealed.record()
    record["seal"] = sha256_bytes(b"a stale seal field")
    target = sealed.run_dir / "stale-seal.json"
    write_record(target, record, reseal=False)
    report = run_verification(target, check_code=False, expect_seal=sealed.seal)
    assert status_of(report, "seal")["code"] == "LEDGER_SEAL_MISMATCH"
    assert status_of(report, "expected_seal")["status"] == PASS
    wrong = run_verification(sealed.record_path, check_code=False, expect_seal=sha256_bytes(b"other"))
    assert status_of(wrong, "expected_seal")["code"] == "LEDGER_SEAL_UNEXPECTED"
    assert status_of(wrong, "expected_seal")["details"]["recomputed"] == sealed.seal


# --- structure ------------------------------------------------------------------------------------


def _mutate(record: dict, dotted: str, value):
    target = record
    parts = [int(p) if p.isdigit() else p for p in dotted.split(".")]
    for part in parts[:-1]:
        target = target[part]
    if value is DELETE:
        del target[parts[-1]]
    else:
        target[parts[-1]] = value


DELETE = object()
MALFORMED = [
    ("purpose", DELETE, "$"),
    ("extra", 1, "$"),
    ("run_id", "Upper", "run_id"),
    ("run_id", "abc\n", "run_id"),
    ("created_utc", "2025-06-01 12:00:00", "created_utc"),
    ("created_utc", "2025-13-06T12:00:00Z", "created_utc"),
    ("purpose", " ", "purpose"),
    ("code", [], "code"),
    ("code.package_version", "", "code.package_version"),
    ("code.commit", "ABCDEF" + "0" * 34, "code.commit"),
    ("code.dirty", None, "code.dirty"),
    ("code.source_root", "/abs", "code.source_root"),
    ("code.source_manifest", {}, "code.source_manifest"),
    ("code.source_manifest.0", {"path": "a"}, "code.source_manifest[0]"),
    ("code.source_manifest.0.path", "../up", "code.source_manifest[0].path"),
    ("code.source_manifest.0.path", "zzz.py", "code.source_manifest[1].path"),
    ("code.source_manifest.0.digest", "sha256:short", "code.source_manifest[0].digest"),
    ("code.source_manifest_root", "md5:x", "code.source_manifest_root"),
    ("config", [], "config"),
    ("config_digest", None, "config_digest"),
    ("seeds", [], "seeds"),
    ("seed_pointers", [], "seed_pointers"),
    ("seed_pointers.eval", DELETE, "seed_pointers"),
    ("seed_pointers.eval", 3, "seed_pointers.eval"),
    ("seeds", {"Bad Name": 7, "eval": 11}, "seeds.Bad Name"),
    ("inputs", {}, "inputs"),
    ("inputs.0.extra", 1, "inputs[0]"),
    ("inputs.0.name", "Labels", "inputs[0].name"),
    ("inputs.1.name", "aaa", "inputs[1].name"),
    ("inputs.3.name", "remote-meta", "inputs[3].name"),  # a duplicate, in order: names are unique too
    ("inputs.0.path", "relative/path", "inputs[0].path"),
    ("inputs.2.uri", None, "inputs[2]"),
    ("inputs.0.uri", "", "inputs[0].uri"),
    ("inputs.0.digest", "sha256:" + "A" * 64, "inputs[0].digest"),
    ("inputs.0.size", -1, "inputs[0].size"),
    ("inputs.0.size", True, "inputs[0].size"),
    ("models.0.path", None, "models[0].path"),
    ("models.0.source", "", "models[0].source"),
    ("models.0.uri", 5, "models[0].uri"),
    ("outputs.0.path", "/etc/passwd", "outputs[0].path"),
    ("outputs.0.path", "out/../x", "outputs[0].path"),
    ("outputs.0.digest", None, "outputs[0].digest"),
    ("outputs", [], "outputs"),
    ("chunk_manifests.0.source_uri", "", "chunk_manifests[0].source_uri"),
    ("chunk_manifests.0.count_present", 1.0, "chunk_manifests[0].count_present"),
    ("chunk_manifests.0.count_absent", -2, "chunk_manifests[0].count_absent"),
    ("chunk_manifests.0.root", "x", "chunk_manifests[0].root"),
    ("chunk_manifests.0.manifest_file", "manifests\\volume.jsonl", "chunk_manifests[0].manifest_file"),
    ("chunk_manifests.0.manifest_file_digest", 1, "chunk_manifests[0].manifest_file_digest"),
    ("environment.packages", [], "environment.packages"),
    ("environment.packages.numpy", 2, "environment.packages.numpy"),
    ("environment.python", 3.13, "environment.python"),
    ("environment.hostname", "vm", "environment"),
    ("figures.source", "nope", "figures.source"),
    ("figures.pointer", "summary", "figures.pointer"),
    ("figures.values", {}, "figures.values"),
    ("figures.values", [1], "figures.values"),
    ("figures", [], "figures"),
    ("predictions.registry_input", "nope", "predictions.registry_input"),
    ("predictions.registry_digest", "x", "predictions.registry_digest"),
    ("predictions.ids", [], "predictions.ids"),
    ("predictions.ids", ["p-002", "p-001"], "predictions.ids"),
    ("predictions.ids", ["p-001", "p-001"], "predictions.ids"),
    ("predictions.ids", ["p-001", ""], "predictions.ids"),
    ("predictions", {"ids": []}, "predictions"),
]


@pytest.mark.parametrize("dotted, value, where", MALFORMED, ids=[f"{d}={v!r}"[:60] for d, v, _ in MALFORMED])
def test_structural_problems_are_refused_and_named(sealed, dotted, value, where):
    record = sealed.record()
    _mutate(record, dotted, value)
    target = sealed.run_dir / "malformed.json"
    write_record(target, record, reseal=True)
    report = run_verification(target, check_code=False)
    schema = status_of(report, "schema")
    assert schema["code"] == "LEDGER_RECORD_MALFORMED", schema
    assert any(p.split(":", 1)[0] == where for p in schema["details"]["problems"]), schema["details"]["problems"]
    assert status_of(report, "content")["status"] == NOT_CHECKED


def test_a_seal_that_is_not_a_digest_is_malformed(sealed):
    record = sealed.record()
    record["seal"] = "sealed by hand"
    target = sealed.run_dir / "bad-seal.json"
    write_record(target, record, reseal=False)
    report = run_verification(target, check_code=False)
    assert status_of(report, "seal")["code"] == "LEDGER_SEAL_MISMATCH"
    assert status_of(report, "schema")["details"]["problems"] == ["seal: must be a sha256 digest"]


def test_dirty_record_structure_needs_diff_fields(sealed):
    record = sealed.record()
    record["code"]["dirty"] = True
    target = sealed.run_dir / "dirty-no-diff.json"
    write_record(target, record, reseal=True)
    problems = status_of(run_verification(target, check_code=False), "schema")["details"]["problems"]
    assert {p.split(":")[0] for p in problems} == {"code.diff_digest", "code.diff_file"}


def test_no_commit_requires_null_dirty_and_source_root(sealed):
    record = sealed.record()
    record["code"]["commit"] = None
    target = sealed.run_dir / "no-commit.json"
    write_record(target, record, reseal=True)
    problems = status_of(run_verification(target, check_code=False), "schema")["details"]["problems"]
    assert {p.split(":")[0] for p in problems} == {"code.dirty", "code.source_root"}


def test_an_unsupported_schema_is_named(sealed):
    record = sealed.record()
    record["schema"] = "apok-record-v0"
    target = sealed.run_dir / "v0.json"
    write_record(target, record, reseal=True)
    check = status_of(run_verification(target, check_code=False), "schema")
    assert check["code"] == "LEDGER_SCHEMA_UNSUPPORTED" and check["details"]["schema"] == "apok-record-v0"


# --- seeds, figures, predictions, code --------------------------------------------------------------


@pytest.mark.parametrize("seed, config_value, code", [
    (True, 7, "LEDGER_SEED_INVALID"),
    (7.0, 7, "LEDGER_SEED_INVALID"),
    ("7", 7, "LEDGER_SEED_INVALID"),
    (1, True, "LEDGER_SEED_CONFIG_MISMATCH"),
    (7, 7.0, "LEDGER_SEED_CONFIG_MISMATCH"),
    (7, 8, "LEDGER_SEED_CONFIG_MISMATCH"),
])
def test_seed_checks(sealed, seed, config_value, code):
    record = sealed.record()
    record["seeds"]["optimizer"] = seed
    record["config"]["fit"]["optimizer_random_seed"] = config_value
    target = sealed.run_dir / "seeds.json"
    write_record(target, record, reseal=True)
    check = status_of(run_verification(target, check_code=False), "seeds")
    assert check["code"] == code and check["details"]["seed"] == "optimizer"


def test_a_seed_pointer_that_does_not_resolve_is_named(sealed):
    record = sealed.record()
    record["seed_pointers"]["eval"] = "/eval/missing"
    target = sealed.run_dir / "seed-pointer.json"
    write_record(target, record, reseal=True)
    check = status_of(run_verification(target, check_code=False), "seeds")
    assert check["code"] == "LEDGER_SEED_CONFIG_MISMATCH"
    assert check["details"] == {"seed": "eval", "pointer": "/eval/missing"}


@pytest.mark.parametrize("member, value", [("n", 120.0), ("certified", 1), ("alpha", 0.1000001)])
def test_figures_compare_as_canonical_json(sealed, member, value):
    assert FIGURES[member] == value or member == "alpha"  # Python equality would let 120 == 120.0, True == 1
    record = sealed.record()
    record["figures"]["values"][member] = value
    target = sealed.run_dir / "figures.json"
    write_record(target, record, reseal=True)
    check = status_of(run_verification(target, check_code=False), "figures")
    assert check["code"] == "LEDGER_FIGURES_MISMATCH" and check["details"]["differing"] == [member]


def test_figures_missing_or_extra_members_are_named(sealed):
    record = sealed.record()
    del record["figures"]["values"]["bound"]
    record["figures"]["values"]["extra"] = 1
    target = sealed.run_dir / "figures-members.json"
    write_record(target, record, reseal=True)
    check = status_of(run_verification(target, check_code=False), "figures")
    assert check["details"]["differing"] == ["bound", "extra"]


def test_figures_from_a_file_that_is_not_strict_json(fresh):
    fresh.files["metrics"].write_text('{"summary": {"n": NaN}}')
    found = failures(refusal_of(fresh.record_path, check_code=False))
    assert found["figures"]["code"] == "LEDGER_FIGURES_INVALID"
    assert found["figures"]["details"] == {"source": "metrics", "pointer": "/summary"}


def test_figures_whose_output_is_missing(fresh):
    fresh.files["metrics"].unlink()
    found = failures(refusal_of(fresh.record_path, check_code=False))
    assert found["figures"]["code"] == "LEDGER_FIGURES_INVALID"


def test_no_figures_and_no_predictions_are_reported_not_checked(sealed):
    record = sealed.record()
    record["figures"] = None
    record["predictions"] = None
    target = sealed.run_dir / "bare.json"
    write_record(target, record, reseal=True)
    report = verify(target, check_code=False)
    assert status_of(report, "figures") == {"check": "figures", "status": NOT_CHECKED,
                                            "reason": "the record quotes no figures"}
    assert status_of(report, "predictions")["status"] == NOT_CHECKED


def test_code_check_names_the_differing_file(sealed, source):
    record = sealed.record()
    record["code"]["commit"] = source.commits[0]
    target = sealed.run_dir / "old-commit.json"
    write_record(target, record, reseal=True)
    check = status_of(run_verification(target, repo=source.repo), "code.commit")
    assert check["code"] == "LEDGER_CODE_MISMATCH"
    assert check["details"]["content_differs"] == ["core.py"] and check["details"]["commit"] == source.commits[0]


def test_code_check_with_extra_and_missing_files(sealed, source):
    record = sealed.record()
    manifest = record["code"]["source_manifest"]
    manifest.append({"path": "zz_extra.py", "digest": sha256_bytes(b"")})
    del manifest[0]
    target = sealed.run_dir / "extra-file.json"
    write_record(target, record, reseal=True)
    check = status_of(run_verification(target, repo=source.repo), "code.commit")
    assert check["details"]["missing_from_commit"] == ["zz_extra.py"]
    assert check["details"]["missing_from_record"] == ["__init__.py"]


def test_code_check_not_checked_reasons(sealed, source, tmp_path, monkeypatch):
    def reason(**kwargs) -> str:
        check = status_of(run_verification(sealed.record_path, **kwargs), "code.commit")
        assert check["status"] == NOT_CHECKED, check
        return check["reason"]

    assert "disabled" in reason(check_code=False)
    other = tmp_path / "other"
    other.mkdir()
    git(other, "init", "-q")
    monkeypatch.setattr(verify_module, "_default_repo", lambda: other)
    assert f"is not in the repository at {other}" in reason()
    monkeypatch.setattr(verify_module, "_default_repo", lambda: None)
    assert "no git repository is available" in reason()


def test_a_commit_missing_from_an_explicitly_given_repo_fails(sealed, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    git(other, "init", "-q")
    check = status_of(run_verification(sealed.record_path, repo=other), "code.commit")
    assert check["code"] == "LEDGER_COMMIT_MISSING"
    assert check["details"] == {"commit": sealed.record()["code"]["commit"], "repo": str(other)}


@pytest.mark.parametrize("make", ["missing", "file", "not-a-repository"])
def test_an_explicit_repo_git_cannot_open_fails_rather_than_skipping(sealed, tmp_path, make):
    repo = {"missing": tmp_path / "nonexistent", "file": sealed.record_path,
            "not-a-repository": tmp_path / "plain"}[make]
    if make == "not-a-repository":
        repo.mkdir()
    report = run_verification(sealed.record_path, repo=repo)
    check = status_of(report, "code.commit")
    assert not report["ok"] and check["code"] == "LEDGER_REPO_INVALID"
    assert check["details"]["repo"] == str(repo) and check["details"]["commit"] == sealed.record()["code"]["commit"]


def test_a_resealed_commit_is_seal_only_without_a_repo_that_has_it(sealed, source, tmp_path, monkeypatch):
    record = sealed.record()
    record["code"]["commit"] = MISSING_COMMIT
    target = sealed.run_dir / "missing-commit.json"
    write_record(target, record, reseal=True)
    monkeypatch.setattr(verify_module, "_default_repo", lambda: source.repo)  # the default holds the history
    report = verify(target)
    assert status_of(report, "code.commit")["status"] == NOT_CHECKED
    found = failures(refusal_of(target, expect_seal=sealed.seal))
    assert set(found) == {"expected_seal"}
    assert failed_codes(refusal_of(target, repo=source.repo)) == {("code.commit", "LEDGER_COMMIT_MISSING")}


def test_code_check_without_git(sealed, source, monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    check = status_of(run_verification(sealed.record_path, repo=source.repo), "code.commit")
    assert check["status"] == NOT_CHECKED and "git could not be run" in check["reason"]


# --- the CLI ---------------------------------------------------------------------------------------


def test_cli_verify_passes_and_prints_every_check(sealed, source, capsys):
    assert main(["ledger", "verify", str(sealed.record_path), "--repo", str(source.repo)]) == 0
    out = capsys.readouterr().out
    assert "PASS         code.commit" in out and "NOT CHECKED  environment" in out
    assert f"seal {sealed.seal}" in out


def test_cli_verify_json_and_flags(fresh, capsys):
    fresh.files["volume"].unlink()
    assert main(["ledger", "verify", str(fresh.record_path), "--json", "--no-code-check",
                 "--allow-missing-inputs", "--expect-seal", fresh.seal]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] and status_of(report, "input:volume")["status"] == NOT_CHECKED
    assert status_of(report, "expected_seal")["status"] == PASS
    assert status_of(report, "code.commit")["reason"] == "the code check was disabled"


def test_cli_verify_refuses_with_exit_3(fresh, capsys):
    fresh.files["mesh"].write_bytes(b"other")
    assert main(["ledger", "verify", str(fresh.record_path), "--no-code-check", "--json"]) == 3
    refusal = json.loads(capsys.readouterr().out)
    assert refusal["code"] == "LEDGER_VERIFY_FAILED"
    assert [f["check"] for f in refusal["details"]["failed"]] == ["output:mesh"]
    assert main(["ledger", "verify", str(fresh.record_path), "--no-code-check"]) == 3
    captured = capsys.readouterr()
    assert json.loads(captured.out)["code"] == "LEDGER_VERIFY_FAILED"  # stdout: the refusal document only
    assert "FAIL         output:mesh  LEDGER_FILE_SIZE_MISMATCH" in captured.err


def test_cli_verify_with_a_wrong_anchored_seal(sealed, capsys):
    assert main(["ledger", "verify", str(sealed.record_path), "--no-code-check", "--expect-seal",
                 sha256_bytes(b"anchor")]) == 3
    assert "LEDGER_SEAL_UNEXPECTED" in capsys.readouterr().out


def test_cli_show(sealed, capsys):
    assert main(["ledger", "show", str(sealed.record_path)]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["seal"] == sealed.seal
    assert "does not verify" in captured.err
    assert main(["ledger", "show", str(sealed.run_dir / "absent.json")]) == 3
    assert "LEDGER_RECORD_UNREADABLE" in capsys.readouterr().out


def test_report_paths_are_strings(sealed):
    report = run_verification(sealed.record_path, check_code=False)
    json.dumps(report)
    assert os.fspath(report["record"]) == str(Path(sealed.record_path))


# --- a run sealed from a dirty tree: the commit plus the recorded diff ---------------------------------


def test_a_dirty_run_verifies_against_its_commit_and_diff(sealed_dirty, dirty_source):
    report = verify(sealed_dirty.record_path, repo=dirty_source.repo, expect_seal=sealed_dirty.seal)
    assert {c["check"]: c["status"] for c in report["checks"] if c["check"].startswith("code.")} == {
        "code.source_manifest_root": PASS, "code.diff": PASS, "code.commit": PASS}
    code = sealed_dirty.record()["code"]
    paths = {e["path"] for e in code["source_manifest"]}
    assert {"sub/untracked.py", "sub/empty.py"} <= paths  # untracked files of the package are covered too
    diff = (sealed_dirty.run_dir / "code.diff").read_bytes()
    assert b"+ANSWER = 44" in diff and b"+synthetic, edited outside" in diff and b"b/src/apok/sub/empty.py" in diff


@pytest.mark.parametrize("tree", ["clean", "dirty"])
def test_a_consistent_source_manifest_edit_fails_the_commit_check(request, tree):
    run = request.getfixturevalue("sealed" if tree == "clean" else "sealed_dirty")
    repo = request.getfixturevalue("source" if tree == "clean" else "dirty_source").repo
    record = run.record()
    manifest = record["code"]["source_manifest"]
    for entry in manifest:
        if entry["path"] == "core.py":
            entry["digest"] = sha256_bytes(b"ANSWER = 666\n")
    record["code"]["source_manifest_root"] = source_root_digest(manifest)  # consistent: the root check passes
    target = run.run_dir / "edited-manifest.json"
    write_record(target, record, reseal=True)
    found = failures(refusal_of(target, repo=repo))
    assert set(found) == {"code.commit"} and found["code.commit"]["code"] == "LEDGER_CODE_MISMATCH"
    assert found["code.commit"]["details"]["content_differs"] == ["core.py"]


def _reseal_with_diff(run, diff: bytes) -> None:
    (run.run_dir / "code.diff").write_bytes(diff)
    record = run.record()
    record["code"]["diff_digest"] = sha256_bytes(diff)
    write_record(run.record_path, record, reseal=True)


def test_an_edited_diff_with_its_digest_updated_fails_the_commit_check(fresh_dirty):
    diff = (fresh_dirty.run_dir / "code.diff").read_bytes()
    _reseal_with_diff(fresh_dirty, diff.replace(b"+NEW = 1", b"+NEW = 2"))
    found = failures(refusal_of(fresh_dirty.record_path, repo=fresh_dirty.source.repo))
    assert set(found) == {"code.commit"}
    assert found["code.commit"]["details"]["content_differs"] == ["sub/untracked.py"]


def test_a_diff_that_does_not_apply_to_the_commit_fails(fresh_dirty):
    diff = (fresh_dirty.run_dir / "code.diff").read_bytes()
    _reseal_with_diff(fresh_dirty, diff.replace(b"-ANSWER = 42", b"-ANSWER = 40"))
    found = failures(refusal_of(fresh_dirty.record_path, repo=fresh_dirty.source.repo))
    assert found["code.commit"]["code"] == "LEDGER_CODE_MISMATCH"
    assert "src/apok/core.py" in found["code.commit"]["details"]["diff_does_not_apply"]


def test_a_diff_that_lost_its_digest_leaves_the_commit_unchecked(fresh_dirty):
    diff_path = fresh_dirty.run_dir / "code.diff"
    diff_path.write_bytes(diff_path.read_bytes() + b"\n")
    report = run_verification(fresh_dirty.record_path, repo=fresh_dirty.source.repo)
    assert status_of(report, "code.diff")["code"] == "LEDGER_FILE_DIGEST_MISMATCH"
    commit = status_of(report, "code.commit")
    assert commit["status"] == NOT_CHECKED and "does not have diff_digest" in commit["reason"]
    diff_path.unlink()
    report = run_verification(fresh_dirty.record_path, repo=fresh_dirty.source.repo)
    assert status_of(report, "code.diff")["code"] == "LEDGER_FILE_MISSING"
    assert "cannot be read" in status_of(report, "code.commit")["reason"]


# --- files that cannot be read, or are not files -----------------------------------------------------------


def test_an_input_path_that_cannot_be_opened_fails_and_is_named(sealed):
    record = sealed.record()
    long_path = "/" + "a" * 300  # a valid absolute path that no file system can open
    record["inputs"][0]["path"] = long_path
    target = sealed.run_dir / "long-path.json"
    write_record(target, record, reseal=True)
    for allow in (False, True):  # unreadable is not missing: allow_missing_inputs does not excuse it
        found = failures(refusal_of(target, check_code=False, allow_missing_inputs=allow))
        check = found[f"input:{record['inputs'][0]['name']}"]
        assert check["code"] == "LEDGER_FILE_UNREADABLE"
        assert check["details"] == {"what": "input 'labels'", "path": long_path, "errno": errno.ENAMETOOLONG}


def test_a_read_error_is_a_failed_check_not_a_crash(fresh, monkeypatch):
    def denied(path):
        raise PermissionError(errno.EACCES, "Permission denied", str(path))

    monkeypatch.setattr(verify_module, "digest_and_size", denied)
    report = run_verification(fresh.record_path, check_code=False)
    check = status_of(report, "model:winding")
    assert check["code"] == "LEDGER_FILE_UNREADABLE"
    assert check["details"] == {"what": "model 'winding'", "path": str(fresh.files["model"]), "errno": errno.EACCES}


def test_a_figures_output_replaced_by_a_fifo_fails_without_blocking(fresh):
    metrics = fresh.files["metrics"]
    metrics.unlink()
    os.mkfifo(metrics)
    with deadline():
        found = failures(refusal_of(fresh.record_path, check_code=False))
    assert found["output:metrics"]["code"] == "LEDGER_FILE_MISSING"
    assert found["figures"]["code"] == "LEDGER_FIGURES_INVALID"
    assert "not a regular file" in found["figures"]["reason"]


def test_a_sidecar_replaced_by_a_fifo_fails_without_blocking(fresh, monkeypatch):
    sidecar = fresh.files["sidecar"]
    sidecar.unlink()
    os.mkfifo(sidecar)
    with deadline():
        found = failures(refusal_of(fresh.record_path, check_code=False))
    assert found["chunk_manifest:volume"]["code"] == "LEDGER_FILE_MISSING"


def test_a_sidecar_that_turns_unreadable_after_its_digest_was_checked(fresh, monkeypatch):
    def gone(path):
        raise OSError(errno.EIO, "Input/output error", str(path))

    monkeypatch.setattr(verify_module, "read_regular", gone)
    found = failures(refusal_of(fresh.record_path, check_code=False))
    assert found["chunk_manifest:volume"]["code"] == "LEDGER_MANIFEST_INVALID"
    assert found["chunk_manifest:volume"]["details"]["manifest"] == "volume"


def test_a_record_that_is_a_fifo_or_a_device_is_unreadable(tmp_path):
    fifo = tmp_path / "record.json"
    os.mkfifo(fifo)
    for path in (fifo, Path("/dev/zero")):
        with deadline():
            refusal = refusal_of(path)
        assert failed_codes(refusal) == {("record", "LEDGER_RECORD_UNREADABLE")}
        assert "not a regular file" in failures(refusal)["record"]["reason"]


@pytest.mark.parametrize("depth", [257, 2000, 100_000])
def test_a_record_nested_too_deeply_is_unreadable_not_a_crash(tmp_path, depth, capsys):
    path = tmp_path / "deep.json"
    path.write_bytes(b'{"a":' + b"[" * depth + b"]" * depth + b"}")
    refusal = refusal_of(path)
    assert failed_codes(refusal) == {("record", "LEDGER_RECORD_UNREADABLE")}
    assert "nested" in failures(refusal)["record"]["reason"]
    assert main(["ledger", "verify", str(path), "--json"]) == 3
    assert json.loads(capsys.readouterr().out)["code"] == "LEDGER_VERIFY_FAILED"


def test_figures_that_cannot_be_serialised_canonically_fail(fresh):
    fresh.files["metrics"].write_bytes(b'{"summary": {"label": "\\ud800"}}')  # a lone surrogate
    found = failures(refusal_of(fresh.record_path, check_code=False))
    assert found["figures"]["code"] == "LEDGER_FIGURES_INVALID"
    assert found["figures"]["details"] == {"source": "metrics", "pointer": "/summary"}


# --- the chunks of a local store are read again ------------------------------------------------------


@pytest.fixture
def local_volume_run(tmp_path, source):
    """A sealed run whose chunk manifest is the read log of a local OME-Zarr store, recorded by file:// URI."""
    from io_zarr_fixtures import write_small

    from apok_eval.io.frame import VolumeFrame
    from apok_eval.io.omezarr import open_volume
    from apok_eval.io.store import LocalStore
    from apok_eval.ledger import open_run

    store_root = tmp_path / "vol.zarr"
    write_small(store_root)
    volume = open_volume(LocalStore(store_root), VolumeFrame("20241024131838", (9, 7, 5), 7.91, "test"))
    volume.read((0, 0, 0), (9, 7, 5))
    volume.read((0, 0, 0), (9, 7, 5), 1)
    (store_root / "0" / "0.0.0").unlink()  # one chunk absent from the start, as the read log records it
    volume = open_volume(LocalStore(store_root), VolumeFrame("20241024131838", (9, 7, 5), 7.91, "test"))
    volume.read((0, 0, 0), (9, 7, 5))
    (tmp_path / "data.bin").write_bytes(b"x")
    run = open_run(tmp_path / "run", run_id="local", purpose="a local volume read", config={}, expected_bytes=0,
                   index=None, source_dir=source.source_dir)
    run.add_input("data", tmp_path / "data.bin")
    run.add_chunk_manifest("volume", volume.log.entries(), source_uri=store_root.as_uri())
    (run.run_dir / "out.json").write_text("{}")
    run.add_output("out", "out.json")
    run.seal()
    return run.record_path, store_root


def test_the_chunks_of_a_local_store_are_read_again_and_pass_unchanged(local_volume_run, source):
    record, _ = local_volume_run
    assert status_of(verify(record, repo=source.repo), "chunk_data:volume")["status"] == PASS


@pytest.mark.parametrize("change, key, recorded, found", [
    ("flip", "0/0.1.0", "digest", "digest"),      # a chunk the run read, one byte changed
    ("delete", "0/0.1.0", "digest", "absent"),    # a chunk the run read, gone
    ("create", "0/0.0.0", "absent", "digest"),    # a chunk the run found absent, now present
])
def test_a_changed_chunk_of_a_local_store_fails_verify_naming_it(local_volume_run, source, change, key, recorded,
                                                                   found):
    record, store_root = local_volume_run
    path = store_root.joinpath(*key.split("/"))
    if change == "flip":
        data = bytearray(path.read_bytes())
        data[0] ^= 0xFF
        path.write_bytes(bytes(data))
    elif change == "delete":
        path.unlink()
    else:
        path.write_bytes(bytes(64))
    report = run_verification(record, repo=source.repo)
    check = status_of(report, "chunk_data:volume")
    assert not report["ok"] and check["code"] == "LEDGER_CHUNK_MISMATCH"
    assert check["details"]["count"] == 1 and check["details"]["store"] == str(store_root)
    (entry,) = check["details"]["differing"]
    assert entry["key"] == key
    assert (entry["recorded"] == "absent") == (recorded == "absent") and (entry["found"] == "absent") == (found == "absent")
    assert status_of(report, "chunk_manifest:volume")["status"] == PASS  # the sidecar itself is unchanged


def test_a_local_store_that_is_gone_is_missing_unless_allowed(local_volume_run, source):
    import shutil

    record, store_root = local_volume_run
    shutil.rmtree(store_root)
    check = status_of(run_verification(record, repo=source.repo), "chunk_data:volume")
    assert check["code"] == "LEDGER_FILE_MISSING" and check["details"]["path"] == str(store_root)
    allowed = status_of(run_verification(record, repo=source.repo, allow_missing_inputs=True), "chunk_data:volume")
    assert allowed["status"] == NOT_CHECKED


def test_a_local_store_that_cannot_be_looked_at_is_unreadable(local_volume_run, source, monkeypatch):
    import errno

    record, store_root = local_volume_run

    def blocked(path, **kwargs):
        raise PermissionError(errno.EACCES, "Permission denied", str(path))

    monkeypatch.setattr(verify_module, "entry_kind", blocked)
    check = status_of(run_verification(record, repo=source.repo), "chunk_data:volume")
    assert check["code"] == "LEDGER_FILE_UNREADABLE" and check["details"]["errno"] == errno.EACCES


def test_a_chunk_the_store_now_refuses_to_read_differs(local_volume_run, source):
    record, store_root = local_volume_run
    chunk = store_root / "0" / "0.1.0"
    chunk.unlink()
    chunk.mkdir()  # a directory where the chunk was: the store refuses it
    check = status_of(run_verification(record, repo=source.repo), "chunk_data:volume")
    assert check["code"] == "LEDGER_CHUNK_MISMATCH"
    assert check["details"]["differing"][0]["found"].startswith("IO_STORE_NOT_A_FILE")


@pytest.mark.parametrize("uri, local", [
    ("file:///data/vol.zarr", "/data/vol.zarr"), ("file://localhost/data/vol%20a.zarr", "/data/vol a.zarr"),
    ("file://host/data/vol.zarr", None), ("https://example.org/vol.zarr", None), ("file:vol.zarr", None),
    ("file:///data/vol.zarr?x=1", None), ("s3://bucket/vol.zarr", None),
])
def test_only_a_local_file_uri_names_a_store_to_read_again(uri, local):
    from apok_eval.ledger.verify import _local_store_root

    assert _local_store_root(uri) == (None if local is None else Path(local))
