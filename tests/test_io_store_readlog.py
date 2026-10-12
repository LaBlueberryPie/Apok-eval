"""The read log: an apok_eval.merkle chunk manifest of every object fetched, refusing objects that change."""

from __future__ import annotations

import hashlib

import pytest

from apok_eval.canonical import sha256_bytes
from apok_eval.io.readlog import LoggedStore, ReadLog
from apok_eval.io.store import LocalStore
from apok_eval.merkle import SCHEME, manifest_root, merkle_root
from apok_eval.refusal import Refusal


def test_present_and_absent_entries_follow_the_manifest_convention():
    log = ReadLog()
    log.record("0/1.2.3", b"compressed bytes")
    log.record("0/9.9.9", None)
    assert log.entries() == [
        {"key": "0/1.2.3", "digest": sha256_bytes(b"compressed bytes"), "size": 16},
        {"key": "0/9.9.9", "absent": True},
    ]


def test_entries_are_sorted_by_utf8_key_bytes_whatever_the_read_order():
    log = ReadLog()
    for key in ("b", "a/z", "A", "a.b", "a"):
        log.record(key, key.encode())
    assert [e["key"] for e in log.entries()] == ["A", "a", "a.b", "a/z", "b"]


def test_root_is_the_manifest_root_of_the_entries_and_matches_a_hand_computed_tree():
    log = ReadLog()
    log.record(".zgroup", b'{"zarr_format":2}')
    log.record("0/0.0.0", None)
    assert log.root() == manifest_root(log.entries())
    empty = "sha256:" + hashlib.sha256(b"").hexdigest()
    assert log.root() == merkle_root([(".zgroup", sha256_bytes(b'{"zarr_format":2}')), ("0/0.0.0#absent", empty)])
    assert ReadLog().root() == manifest_root([])
    assert log.as_dict() == {"scheme": SCHEME, "root": log.root(), "entries": log.entries()}
    assert len(log) == 2


def test_the_root_changes_when_any_object_changes_or_goes_missing():
    def root(objects):
        log = ReadLog()
        for key, data in objects.items():
            log.record(key, data)
        return log.root()

    base = {".zattrs": b"{}", "0/0.0.0": b"\x01"}
    assert root(base) != root({**base, "0/0.0.0": b"\x02"})
    assert root(base) != root({**base, "0/0.0.0": None})
    assert root(base) != root({".zattrs": b"{}", "0/0.0.1": b"\x01"})


def test_rereading_identical_bytes_is_accepted():
    log = ReadLog()
    log.record("k", b"same")
    log.record("k", bytearray(b"same"))
    assert len(log) == 1


@pytest.mark.parametrize("first,second", [(b"one", b"two"), (b"one", None), (None, b"now present"),
                                          (b"one", b"one\x00")])
def test_an_object_that_changes_between_reads_is_refused_and_the_first_record_kept(first, second):
    log = ReadLog()
    log.record("0/1.1.1", first)
    before = log.entries()
    with pytest.raises(Refusal) as exc:
        log.record("0/1.1.1", second)
    assert exc.value.code == "IO_OBJECT_CHANGED"
    assert exc.value.details["key"] == "0/1.1.1"
    assert exc.value.details["before"] == before[0]
    assert exc.value.details["now"]["key"] == "0/1.1.1"
    assert log.entries() == before


def test_record_refuses_keys_a_store_could_not_have_served():
    with pytest.raises(Refusal) as exc:
        ReadLog().record("a#absent", b"x")
    assert exc.value.code == "IO_STORE_KEY_INVALID" and exc.value.details["key"] == "a#absent"


def test_a_logged_store_records_every_get_with_the_raw_stored_bytes(tmp_path):
    raw = b"\x78\x9c raw, not decoded"
    (tmp_path / "k").write_bytes(raw)
    log = ReadLog()
    store = LoggedStore(LocalStore(tmp_path), log)
    assert store.get("k") == raw
    assert store.get("missing") is None
    assert log.entries() == [{"key": "k", "digest": sha256_bytes(raw), "size": 19},
                             {"key": "missing", "absent": True}]
    assert store.describe() == {"kind": "local", "root": str(tmp_path.resolve())}
    (tmp_path / "missing").write_bytes(b"appeared")
    with pytest.raises(Refusal) as exc:
        store.get("missing")
    assert exc.value.code == "IO_OBJECT_CHANGED" and exc.value.details["key"] == "missing"
