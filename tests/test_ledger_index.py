"""The append-only index: every field of every line tampered, lines deleted,
swapped and forged, the anchor, and the record cross-check."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from apok_eval.canonical import canonical_dumps, sha256_bytes
from apok_eval.cli import main
from apok_eval.ledger import append_index, verify_index
from apok_eval.refusal import Refusal

from ledger_fixtures import build_run, deadline, make_source_repo, write_record

RUNS = ("run-a", "run-b", "run-c")
FIELDS = ("seq", "run_id", "seal", "record_path", "prev_entry_digest")


@pytest.fixture(scope="module")
def ledger(tmp_path_factory):
    """Three sealed runs in one index; tests work on a copy (see ``index``)."""
    base = tmp_path_factory.mktemp("index")
    source = make_source_repo(base)
    index = base / "ledger" / "index.jsonl"
    runs = {run_id: build_run(base, source, run_id=run_id, index=index) for run_id in RUNS}
    return base, index, runs


@pytest.fixture
def index(ledger, tmp_path):
    """A copy of the index beside the original, so record paths still resolve."""
    _, original, _ = ledger
    copy = original.parent / f"copy-{tmp_path.name}.jsonl"
    shutil.copyfile(original, copy)
    return copy


def lines(path) -> list[bytes]:
    return path.read_bytes().split(b"\n")[:-1]


def write_lines(path, items: list[bytes]) -> None:
    path.write_bytes(b"".join(item + b"\n" for item in items))


def entry(line: bytes) -> dict:
    return json.loads(line)


def refused(code: str, path, **kwargs) -> Refusal:
    with pytest.raises(Refusal) as info:
        verify_index(path, **kwargs)
    assert info.value.code == code, (info.value.code, info.value.reason)
    return info.value


def rechain(items: list[dict]) -> list[bytes]:
    """Lines for ``items`` with seq and prev_entry_digest recomputed: a consistent rewrite."""
    out: list[bytes] = []
    for number, item in enumerate(items, start=1):
        item = dict(item, seq=number, prev_entry_digest=sha256_bytes(out[-1]) if out else None)
        out.append(canonical_dumps(item))
    return out


# --- the untouched index ---------------------------------------------------------------------


def test_the_index_verifies(ledger):
    _, index, runs = ledger
    summary = verify_index(index)
    assert summary["entries"] == 3 and summary["records_checked"] == 3
    assert summary["head"] == sha256_bytes(lines(index)[-1])
    assert [entry(line)["run_id"] for line in lines(index)] == list(RUNS)
    assert [entry(line)["seal"] for line in lines(index)] == [runs[r].seal for r in RUNS]
    assert entry(lines(index)[1])["prev_entry_digest"] == sha256_bytes(lines(index)[0])
    for line in lines(index):
        assert verify_index(index, anchor=sha256_bytes(line))["anchor"] == sha256_bytes(line)


def test_an_empty_index_verifies(tmp_path):
    (tmp_path / "index.jsonl").write_bytes(b"")
    assert verify_index(tmp_path / "index.jsonl") == {"index": str(tmp_path / "index.jsonl"), "entries": 0,
                                                       "head": None, "anchor": None, "records_checked": 0}


# --- every field of every line ---------------------------------------------------------------


def tampered_value(field: str, value):
    if field == "seq":
        return value + 1
    if field in ("seal", "prev_entry_digest"):
        return sha256_bytes(f"forged {value}".encode())
    return value + "x"


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("number", [1, 2, 3])
def test_tampering_any_field_of_any_line_fails(index, number, field):
    items = lines(index)
    changed = entry(items[number - 1])
    changed[field] = tampered_value(field, changed[field])
    items[number - 1] = canonical_dumps(changed)
    write_lines(index, items)
    if field == "seq":
        refusal = refused("LEDGER_INDEX_SEQUENCE_BROKEN", index)
        assert refusal.details["line"] == number
    elif field == "prev_entry_digest":
        refusal = refused("LEDGER_INDEX_CHAIN_BROKEN", index)
        assert refusal.details["line"] == number
    elif number < 3:
        refusal = refused("LEDGER_INDEX_CHAIN_BROKEN", index)  # the next line no longer links
        assert refusal.details["line"] == number + 1
    else:
        refusal = refused("LEDGER_INDEX_RECORD_MISMATCH", index)  # nothing links the last line: its record does
        assert refusal.details["line"] == 3
        assert field.split("_")[0] in refusal.details["problem"] or "cannot be read" in refusal.details["problem"]


@pytest.mark.parametrize("field", ["run_id", "seal", "record_path"])
def test_a_consistently_rewritten_chain_is_caught_by_records_and_anchor(index, field):
    original = lines(index)
    items = [entry(line) for line in original]
    items[0][field] = tampered_value(field, items[0][field])
    write_lines(index, rechain(items))
    assert verify_index(index, check_records=False)["entries"] == 3
    refusal = refused("LEDGER_INDEX_RECORD_MISMATCH", index)
    assert refusal.details["line"] == 1 and refusal.details["run_id"] == items[0]["run_id"]
    refusal = refused("LEDGER_INDEX_ANCHOR_MISSING", index, check_records=False,
                      anchor=sha256_bytes(original[1]))
    assert refusal.details["anchor"] == sha256_bytes(original[1])


# --- deleting, swapping, forging lines --------------------------------------------------------------


@pytest.mark.parametrize("number", [1, 2])
def test_deleting_a_line_breaks_the_sequence(index, number):
    items = lines(index)
    del items[number - 1]
    write_lines(index, items)
    refusal = refused("LEDGER_INDEX_SEQUENCE_BROKEN", index)
    assert refusal.details["line"] == number and refusal.details["seq"] == number + 1


def test_deleting_the_last_line_is_caught_only_by_an_anchor(index):
    items = lines(index)
    head = sha256_bytes(items[-1])
    write_lines(index, items[:-1])
    assert verify_index(index)["entries"] == 2
    refusal = refused("LEDGER_INDEX_ANCHOR_MISSING", index, anchor=head)
    assert refusal.details["anchor"] == head


@pytest.mark.parametrize("pair", [(0, 1), (1, 2), (0, 2)])
def test_swapping_two_lines_fails(index, pair):
    items = lines(index)
    i, j = pair
    items[i], items[j] = items[j], items[i]
    write_lines(index, items)
    refusal = refused("LEDGER_INDEX_SEQUENCE_BROKEN", index)
    assert refusal.details["line"] == i + 1


def test_a_forged_appended_line_is_caught_by_the_record_check(index):
    items = lines(index)
    forged = {"seq": 4, "run_id": "run-d", "seal": sha256_bytes(b"forged"), "record_path": "../runs/run-d/record.json",
              "prev_entry_digest": sha256_bytes(items[-1])}
    write_lines(index, items + [canonical_dumps(forged)])
    assert verify_index(index, check_records=False)["entries"] == 4
    refusal = refused("LEDGER_INDEX_RECORD_MISMATCH", index)
    assert refusal.details == {"line": 4, "run_id": "run-d", "record_path": "../runs/run-d/record.json",
                               "problem": refusal.details["problem"]}
    assert "cannot be read" in refusal.details["problem"]


def test_a_forged_line_pointing_at_a_real_record_is_caught(index):
    items = lines(index)
    real = entry(items[0])
    forged = {"seq": 4, "run_id": "run-d", "seal": real["seal"], "record_path": real["record_path"],
              "prev_entry_digest": sha256_bytes(items[-1])}
    write_lines(index, items + [canonical_dumps(forged)])
    refusal = refused("LEDGER_INDEX_RECORD_MISMATCH", index)
    assert "run_id is 'run-a', the index says 'run-d'" in refusal.details["problem"]


def test_a_repeated_run_id_is_refused(index):
    items = [entry(line) for line in lines(index)]
    items.append(dict(items[0]))
    write_lines(index, rechain(items))
    refusal = refused("LEDGER_INDEX_DUPLICATE_RUN", index, check_records=False)
    assert refusal.details["line"] == 4 and refusal.details["run_id"] == "run-a"


def test_an_edited_record_no_longer_matches_its_index_line(ledger, index):
    _, _, runs = ledger
    run = runs["run-b"]
    original = run.record_path.read_bytes()
    try:
        record = run.record()
        record["purpose"] = "rewritten after the fact"
        write_record(run.record_path, record, reseal=True)
        refusal = refused("LEDGER_INDEX_RECORD_MISMATCH", index)
        assert refusal.details["line"] == 2 and "seal is" in refusal.details["problem"]
        record["seal"] = run.seal  # the old seal kept on new content: the recomputed seal differs
        write_record(run.record_path, record, reseal=False)
        refusal = refused("LEDGER_INDEX_RECORD_MISMATCH", index)
        assert "recomputed seal" in refusal.details["problem"]
    finally:
        run.record_path.write_bytes(original)


# --- malformed lines and files ------------------------------------------------------------------


@pytest.mark.parametrize("edit, fragment", [
    (lambda e: e.update(extra=1), "members must be exactly"),
    (lambda e: e.pop("seal"), "members must be exactly"),
    (lambda e: e.update(seq=True), "seq must be a positive int"),
    (lambda e: e.update(seq=0), "seq must be a positive int"),
    (lambda e: e.update(run_id="Run A"), "run_id"),
    (lambda e: e.update(seal="sha256:1234"), "seal must be"),
    (lambda e: e.update(record_path="/abs/record.json"), "record_path"),
    (lambda e: e.update(record_path="a//b"), "record_path"),
    (lambda e: e.update(record_path="./b"), "record_path"),
    (lambda e: e.update(record_path="../ledger/../runs/run-b/record.json"), "record_path"),  # another spelling
    (lambda e: e.update(record_path="../runs/run-b/.."), "record_path"),
    (lambda e: e.update(record_path=".."), "record_path"),
    (lambda e: e.update(record_path="..\\runs\\run-b\\record.json"), "record_path"),
    (lambda e: e.update(record_path="../runs/run-b/record.json\x00"), "record_path"),
    (lambda e: e.update(prev_entry_digest="none"), "prev_entry_digest"),
])
def test_malformed_entries_are_named(index, edit, fragment):
    items = lines(index)
    changed = entry(items[1])
    edit(changed)
    items[1] = canonical_dumps(changed)
    write_lines(index, items)
    refusal = refused("LEDGER_INDEX_MALFORMED", index)
    assert refusal.details["line"] == 2 and any(fragment in p for p in refusal.details["problems"])


@pytest.mark.parametrize("raw, problem", [
    (lambda items: items[:1] + [items[1].replace(b",", b", ", 1)] + items[2:], "the line is not canonical JSON"),
    (lambda items: items[:1] + [b""] + items[1:],
     "the line is not strict JSON: invalid JSON: Expecting value at line 1 column 1"),
    (lambda items: items[:1] + [b"not json"] + items[1:],
     "the line is not strict JSON: invalid JSON: Expecting value at line 1 column 1"),
    (lambda items: items[:1] + [b"[1]"] + items[1:], "the line is not a JSON object"),
    (lambda items: items[:1] + [b'{"seq": 2, "seq": 2}'] + items[1:],
     "the line is not strict JSON: duplicate key 'seq'"),
    (lambda items: items[:1] + [b'{"seq": NaN}'] + items[1:],
     "the line is not strict JSON: non-standard constant NaN (NaN, Infinity and -Infinity are not JSON)"),
    # Canonical, but past the depth every ledger reader accepts.
    (lambda items: items[:1] + [b"[" * 257 + b"]" * 257] + items[1:],
     "the line is not strict JSON: arrays and objects nested deeper than 256 levels"),
    # Python's json raises RecursionError here, which is not a ValueError.
    (lambda items: items[:1] + [b"[" * 100_000 + b"]" * 100_000] + items[1:],
     "the line is not strict JSON: arrays and objects nested deeper than the parser can recurse (the limit is 256 "
     "levels)"),
], ids=["spaced", "blank", "not-json", "array", "duplicate-key", "nan", "deeper-than-256", "deeper-than-the-parser"])
def test_lines_that_are_not_canonical_objects_are_refused(index, raw, problem):
    write_lines(index, raw(lines(index)))
    refusal = refused("LEDGER_INDEX_MALFORMED", index)
    assert refusal.details["line"] == 2 and refusal.details["problems"] == [problem]
    assert refusal.reason.endswith(f"line 2: {problem}")


def test_a_torn_last_line_is_refused(index):
    index.write_bytes(index.read_bytes()[:-5])
    refusal = refused("LEDGER_INDEX_MALFORMED", index)
    assert refusal.details["line"] == 3 and "newline" in refusal.reason


def test_a_missing_index_is_unreadable(tmp_path):
    refusal = refused("LEDGER_INDEX_UNREADABLE", tmp_path / "nowhere.jsonl")
    assert refusal.details["index"] == str(tmp_path / "nowhere.jsonl")


# --- appending ---------------------------------------------------------------------------------------


def test_append_refuses_a_run_id_already_indexed(index):
    before = index.read_bytes()
    with pytest.raises(Refusal) as info:
        append_index(index, run_id="run-b", seal=sha256_bytes(b"x"), record_path=index.parent / "r.json")
    assert info.value.code == "LEDGER_INDEX_DUPLICATE_RUN" and info.value.details["run_id"] == "run-b"
    assert index.read_bytes() == before


def test_append_refuses_a_malformed_entry(index):
    before = index.read_bytes()
    with pytest.raises(Refusal) as info:
        append_index(index, run_id="run-d", seal="not a digest", record_path=index.parent / "r.json")
    assert info.value.code == "LEDGER_INDEX_MALFORMED" and "seal" in info.value.details["problems"][0]
    assert index.read_bytes() == before


def test_append_refuses_to_extend_a_broken_chain(index):
    items = lines(index)
    write_lines(index, [items[1], items[0], items[2]])
    with pytest.raises(Refusal) as info:
        append_index(index, run_id="run-d", seal=sha256_bytes(b"x"), record_path=index.parent / "r.json")
    assert info.value.code == "LEDGER_INDEX_SEQUENCE_BROKEN"


def test_append_into_a_missing_directory_fails(tmp_path):
    with pytest.raises(Refusal) as info:
        append_index(tmp_path / "missing" / "index.jsonl", run_id="r", seal=sha256_bytes(b"x"),
                     record_path=tmp_path / "r.json")
    assert info.value.code == "LEDGER_WRITE_FAILED"
    assert info.value.details["path"] == str(tmp_path / "missing" / "index.jsonl")


def test_append_returns_the_entry_digest(tmp_path):
    index = tmp_path / "index.jsonl"
    first = append_index(index, run_id="r1", seal=sha256_bytes(b"1"), record_path=tmp_path / "runs" / "r1.json")
    second = append_index(index, run_id="r2", seal=sha256_bytes(b"2"), record_path=tmp_path / "r2.json")
    assert first["record_path"] == "runs/r1.json" and first["prev_entry_digest"] is None
    assert second["prev_entry_digest"] == first["entry_digest"] == sha256_bytes(lines(index)[0])
    assert verify_index(index, check_records=False)["head"] == second["entry_digest"]


# --- the CLI ------------------------------------------------------------------------------------------


def test_cli_verify_index(ledger, index, capsys):
    assert main(["ledger", "verify-index", str(index)]) == 0
    assert json.loads(capsys.readouterr().out)["entries"] == 3
    head = sha256_bytes(lines(index)[-1])
    write_lines(index, lines(index)[:-1])
    assert main(["ledger", "verify-index", str(index), "--no-record-check", "--anchor", head]) == 3
    assert json.loads(capsys.readouterr().out)["code"] == "LEDGER_INDEX_ANCHOR_MISSING"


# --- record paths: one spelling, regular files only, symlinked directories ------------------------------------


def test_the_last_line_pointing_at_a_copy_of_its_record_needs_the_anchor(ledger, index):
    base, _, runs = ledger
    items = lines(index)
    head = sha256_bytes(items[-1])
    copy_dir = base / f"copy-of-run-c-{index.stem}"
    copy_dir.mkdir()
    shutil.copyfile(runs["run-c"].record_path, copy_dir / "record.json")
    last = entry(items[-1])
    last["record_path"] = f"../{copy_dir.name}/record.json"
    write_lines(index, items[:-1] + [canonical_dumps(last)])
    # The same record bytes elsewhere: run_id and seal still agree, so only the anchored head notices.
    assert verify_index(index)["entries"] == 3
    refusal = refused("LEDGER_INDEX_ANCHOR_MISSING", index, anchor=head)
    assert refusal.details["anchor"] == head


def _single_line_index(directory, record_path: str):
    line = {"seq": 1, "run_id": "evil", "seal": sha256_bytes(b"x"), "record_path": record_path,
            "prev_entry_digest": None}
    index = directory / "index.jsonl"
    index.write_bytes(canonical_dumps(line) + b"\n")
    return index


def test_a_record_path_to_a_device_or_a_fifo_is_refused_without_reading_it(tmp_path):
    ups = "/".join([".."] * (len(Path(os.path.realpath(tmp_path)).parts) + 2))
    os.mkfifo(tmp_path / "pipe.json")
    for record_path in ("pipe.json", f"{ups}/dev/zero"):  # the FIFO first: a reader that ignored types would hang
        index = _single_line_index(tmp_path, record_path)
        with deadline():
            refusal = refused("LEDGER_INDEX_RECORD_MISMATCH", index)
        assert refusal.details["record_path"] == record_path and "not a regular file" in refusal.details["problem"]


@pytest.mark.parametrize("kind", ["fifo", "device"])
def test_an_index_that_is_not_a_regular_file(tmp_path, kind):
    index = tmp_path / "index.jsonl"
    if kind == "fifo":
        os.mkfifo(index)
    else:
        index.symlink_to("/dev/null")  # appends would vanish without a trace
    with deadline():
        refusal = refused("LEDGER_INDEX_UNREADABLE", index)
        with pytest.raises(Refusal) as info:
            append_index(index, run_id="r", seal=sha256_bytes(b"x"), record_path=tmp_path / "r.json")
    assert refusal.details["index"] == str(index) and "not a regular file" in refusal.reason
    assert info.value.code == "LEDGER_WRITE_FAILED" and info.value.details["path"] == str(index)
    if kind == "device":
        assert "not a regular file" in info.value.reason


def test_a_record_path_that_is_not_utf8_is_never_appended(tmp_path):
    index = tmp_path / "index.jsonl"
    with pytest.raises(Refusal) as info:
        append_index(index, run_id="r", seal=sha256_bytes(b"x"),
                     record_path=os.fsdecode(os.fsencode(tmp_path) + b"/bad\xffname.json"))
    assert info.value.code == "LEDGER_INDEX_MALFORMED" and "record_path" in info.value.details["problems"][0]
    assert index.read_bytes() == b""


def test_an_index_in_a_symlinked_directory_verifies_after_sealing(tmp_path):
    source = make_source_repo(tmp_path / "source")
    real_ledger = tmp_path / "elsewhere" / "ledger"
    real_ledger.mkdir(parents=True)
    (tmp_path / "ledger").symlink_to(real_ledger, target_is_directory=True)
    index = tmp_path / "ledger" / "index.jsonl"
    run = build_run(tmp_path, source, index=index)
    (line,) = lines(index)
    assert entry(line)["record_path"] == "../../runs/fold-a/record.json"  # from the resolved index directory
    summary = verify_index(index)
    assert summary["entries"] == 1 and summary["records_checked"] == 1
    assert entry(line)["seal"] == run.seal


@pytest.mark.parametrize("name", ["index\x00.jsonl", "index\ud800.jsonl"])
def test_an_index_path_that_is_not_a_path_is_unreadable_not_a_crash(tmp_path, name):
    exc = refused("LEDGER_INDEX_UNREADABLE", str(tmp_path / name))
    assert exc.details == {"index": str(tmp_path / name)}
