"""Known-answer and property tests for the shared core: refusals, canonical
JSON, digests and the Merkle scheme."""

from __future__ import annotations

import hashlib
import socket
import struct

import pytest

from apok_eval.canonical import canonical_dumps, digest_json, digest_raw, is_digest, sha256_bytes, sha256_file
from apok_eval.merkle import merkle_root
from apok_eval.refusal import REGISTRY, Refusal, register_codes


def _d(data: bytes) -> str:
    return sha256_bytes(data)


# --- refusal -----------------------------------------------------------------


def test_refusal_requires_registered_code():
    with pytest.raises(ValueError, match="unregistered"):
        Refusal("NOT_A_REAL_CODE_XYZ", "anything")


def test_refusal_requires_reason():
    register_codes("tests", {"TEST_FOUNDATION_CODE": "used by the foundation tests"})
    with pytest.raises(ValueError, match="without a reason"):
        Refusal("TEST_FOUNDATION_CODE", "  ")
    r = Refusal("TEST_FOUNDATION_CODE", "because", path="x")
    assert (r.code, r.reason, r.details) == ("TEST_FOUNDATION_CODE", "because", {"path": "x"})
    assert r.as_dict()["refused"] is True


@pytest.mark.parametrize("bad", ["lower_case", "NOUNDERSCORE", "_LEADING", "TRAILING_", "DOUBLE__UNDERSCORE"])
def test_register_rejects_malformed_codes(bad):
    with pytest.raises(ValueError):
        register_codes("tests", {bad: "x"})


def test_register_rejects_conflicting_redefinition():
    register_codes("tests", {"TEST_STABLE_CODE": "first meaning"})
    register_codes("tests", {"TEST_STABLE_CODE": "first meaning"})  # idempotent
    with pytest.raises(ValueError, match="already registered"):
        register_codes("tests", {"TEST_STABLE_CODE": "second meaning"})
    with pytest.raises(ValueError, match="already registered"):
        register_codes("other-owner", {"TEST_STABLE_CODE": "first meaning"})
    assert "TEST_STABLE_CODE" in REGISTRY


# --- canonical JSON and digests ---------------------------------------------------


def test_canonical_known_answer():
    assert canonical_dumps({"b": 1, "a": [True, None, 1.5, "é"]}) == '{"a":[true,null,1.5,"é"],"b":1}'.encode()


@pytest.mark.parametrize(
    "bad",
    [float("nan"), float("inf"), {"a": float("-inf")}, (1, 2), {1: "x"}, {"a": {2}}, b"bytes", object()],
)
def test_canonical_rejects_ambiguous_values(bad):
    with pytest.raises((TypeError, ValueError)):
        canonical_dumps(bad)


def test_canonical_rejects_numpy_scalars():
    np = pytest.importorskip("numpy")
    with pytest.raises(TypeError):
        canonical_dumps({"x": np.float32(1.0)})


def test_digest_format_and_file_digest(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc")
    assert sha256_bytes(b"abc") == "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert sha256_file(p) == sha256_bytes(b"abc")
    assert is_digest(sha256_bytes(b""))
    assert not is_digest("sha256:XYZ") and not is_digest("md5:" + "0" * 32)
    assert digest_json({"a": 1}) == sha256_bytes(b'{"a":1}')
    with pytest.raises(ValueError):
        digest_raw("sha256:short")


# --- Merkle scheme (spelled out independently from apok_eval.merkle) -----------------


def _leaf(key: str, digest: str) -> bytes:
    k = key.encode()
    return hashlib.sha256(b"\x00" + struct.pack(">I", len(k)) + k + bytes.fromhex(digest[7:])).digest()


def _node(a: bytes, b: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + a + b).digest()


def test_merkle_known_answers():
    a, b, c = _d(b"a"), _d(b"b"), _d(b"c")
    assert merkle_root([]) == "sha256:" + hashlib.sha256(b"\x02").hexdigest()
    assert merkle_root([("k", a)]) == "sha256:" + _leaf("k", a).hex()
    two = _node(_leaf("0.0.0", a), _leaf("0.0.1", b))
    assert merkle_root([("0.0.1", b), ("0.0.0", a)]) == "sha256:" + two.hex()
    three = _node(two, _leaf("0.0.2", c))  # odd node promoted, then paired
    assert merkle_root([("0.0.2", c), ("0.0.0", a), ("0.0.1", b)]) == "sha256:" + three.hex()


def test_merkle_binds_keys_digests_and_membership():
    a, b, c = _d(b"a"), _d(b"b"), _d(b"c")
    base = merkle_root([("x", a), ("y", b), ("z", c)])
    assert merkle_root([("z", c), ("x", a), ("y", b)]) == base  # order-free
    assert merkle_root([("x", b), ("y", a), ("z", c)]) != base  # digest moved to another key
    assert merkle_root([("x", a), ("y", b), ("z", a)]) != base  # one digest changed
    assert merkle_root([("x", a), ("y", b)]) != base  # leaf removed
    assert merkle_root([("x", a), ("y", b), ("z", c), ("zz", c)]) != base  # leaf added
    assert merkle_root([("x", a), ("y", b), ("z", c), ("z2", c)]) != merkle_root([("x", a), ("y", b), ("z", c)])


def test_merkle_rejects_duplicates_and_bad_input():
    a = _d(b"a")
    with pytest.raises(ValueError, match="duplicate"):
        merkle_root([("k", a), ("k", a)])
    with pytest.raises(ValueError):
        merkle_root([("", a)])
    with pytest.raises(ValueError):
        merkle_root([("k", "sha256:nothex")])


# --- the network guard itself must work -----------------------------------------


def test_no_environment_proxy_is_visible_to_tests():
    import urllib.request

    assert urllib.request.getproxies() == {}


def test_network_guard_blocks_non_loopback():
    with pytest.raises(OSError, match="non-loopback|may not"):
        socket.getaddrinfo("vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com", 443)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(OSError, match="may not connect"):
            s.connect(("8.8.8.8", 53))
    finally:
        s.close()


# --- chunk manifests ------------------------------------------------------------


def test_manifest_root_distinguishes_absent_from_present():
    from apok_eval.merkle import manifest_root

    a = _d(b"chunk")
    present = [{"key": "0/0.0.0", "digest": a, "size": 5}]
    absent = [{"key": "0/0.0.0", "absent": True}]
    empty_present = [{"key": "0/0.0.0", "digest": _d(b""), "size": 0}]
    roots = {manifest_root(present), manifest_root(absent), manifest_root(empty_present), manifest_root([])}
    assert len(roots) == 4
    assert manifest_root(absent) == merkle_root([("0/0.0.0#absent", _d(b""))])


@pytest.mark.parametrize(
    "bad",
    [
        [{"key": "a#b", "digest": "sha256:" + "0" * 64, "size": 1}],
        [{"key": "k", "absent": True, "digest": "sha256:" + "0" * 64}],
        [{"key": "k", "digest": "sha256:" + "0" * 64}],
        [{"key": "k", "digest": "sha256:" + "0" * 64, "size": 1, "extra": 1}],
        [{"key": "k", "digest": "md5:00", "size": 1}],
        [{"key": "k", "digest": "sha256:" + "0" * 64, "size": True}],
        [{"key": "k", "digest": "sha256:" + "0" * 64, "size": -1}],
        [{"key": "k", "digest": "sha256:" + "0" * 64, "size": 1.0}],
        [{"key": "k", "absent": True, "note": "extra"}],
        [{"key": "k", "absent": False}],
        [{"key": "k", "absent": 1}],
    ],
)
def test_manifest_rejects_malformed_entries(bad):
    from apok_eval.merkle import manifest_root

    with pytest.raises(ValueError):
        manifest_root(bad)


def test_manifest_refuses_a_chunk_recorded_both_present_and_absent():
    from apok_eval.merkle import manifest_root

    both = [{"key": "k", "absent": True}, {"key": "k", "digest": "sha256:" + "0" * 64, "size": 1}]
    for entries in (both, both[::-1]):
        with pytest.raises(ValueError, match="chunk 'k' appears more than once"):
            manifest_root(entries)


# --- small helpers exist once ---------------------------------------------------------


def test_shared_helpers_are_one_definition_each():
    import apok_eval.canonical as canonical
    import apok_eval.io.cli as io_cli
    import apok_eval.io.frame as frame
    import apok_eval.io.omezarr as omezarr
    import apok_eval.io.pointcollection as pointcollection
    import apok_eval.io.umbilicus as umbilicus
    import apok_eval.ledger.util as ledger_util
    import importlib
    import apok_eval.licence.registry as licence_registry
    import apok_eval.refusal as refusal

    ledger_verify = importlib.import_module("apok_eval.ledger.verify")  # apok_eval.ledger.verify is also a function
    assert omezarr.finite_number is pointcollection.finite_number is umbilicus.finite_number is frame.finite_number
    assert ledger_util.is_text is licence_registry.is_text is canonical.is_text
    assert ledger_util.is_utf8 is canonical.is_utf8
    assert io_cli.NOT_CHECKED is ledger_verify.NOT_CHECKED is refusal.NOT_CHECKED == "NOT CHECKED"
    for module, name in ((omezarr, "_number"), (pointcollection, "_finite"), (umbilicus, "_finite_number"),
                         (licence_registry, "_text")):
        assert not hasattr(module, name), f"{module.__name__}.{name} duplicates a shared helper"


@pytest.mark.parametrize("value, expected", [
    (1, 1.0), (-2.5, -2.5), (0, 0.0), (2 ** 1023, float(2 ** 1023)), (True, None), (False, None),
    (2 ** 1024, None), (float("nan"), None), (float("inf"), None), ("1", None), (None, None), ([1], None),
])
def test_finite_number(value, expected):
    from apok_eval.io.frame import finite_number

    assert finite_number(value) == expected and (expected is None or type(finite_number(value)) is float)


@pytest.mark.parametrize("value, ok", [
    ("x", True), (" x ", True), ("", False), ("  \t", False), ("a\ud800", False), (3, False), (None, False),
])
def test_is_text(value, ok):
    from apok_eval.canonical import is_text

    assert is_text(value) is ok
