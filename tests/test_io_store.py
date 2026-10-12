"""Stores: safe keys, the local root contract, the canonical S3 mapping and anonymous cleared HTTP reads.

Remote reads go only to a loopback server; the s3:// tests check the URL mapping
and the clearance questions and never issue a request to a real host. Every
store that is built gets a REAL clearance, issued by apok_eval.licence.check_run
against a registry written to tmp_path (io_zarr_fixtures.clearance_for); a
stand-in is refused, whether it merely has a covers() method or passes
isinstance(x, LicenceClearance) (a spec'd mock, a __class__ claim, a subclass).
"""

from __future__ import annotations

import contextlib
import os
import socket
from unittest import mock

import numpy as np
import pytest
from io_zarr_fixtures import DuckClearance, LoopbackServer, clearance_for

from apok_eval.clearance import Clearance
from apok_eval.io.store import (S3_HTTPS_RULE, USER_AGENT, HttpStore, LocalStore, Store, _vetted_connection,
                           check_key, s3_to_https)
from apok_eval.licence import LicenceClearance
from apok_eval.licence.uri import canonical_uri
from apok_eval.refusal import Refusal


def refused(code: str, fn, *args, **kwargs) -> Refusal:
    with pytest.raises(Refusal) as exc:
        fn(*args, **kwargs)
    assert exc.value.code == code, exc.value.as_dict()
    return exc.value


def outcome(fn, *args) -> Refusal | None:
    """The Refusal ``fn(*args)`` raises, or None when it returns (so a caller can first check what it sent)."""
    try:
        fn(*args)
    except Refusal as exc:
        return exc
    return None


# --- keys -------------------------------------------------------------------------------------


@pytest.mark.parametrize("key", [".zgroup", "0/.zarray", "0/1.2.3", "0/1/2/3", "meta.json", "a_b-C.9"])
def test_safe_keys_pass_unchanged(key):
    assert check_key(key) == key


@pytest.mark.parametrize("key", ["", "/a", "a/", "a//b", ".", "..", "a/../b", "./a", "a b", "a#b", "a\\b", "ä",
                                 "a?b", "%2e%2e", "a:b", 5, None, b"a"])
def test_unsafe_keys_are_refused_naming_the_key(key):
    assert refused("IO_STORE_KEY_INVALID", check_key, key).details["key"] == key


# --- local store ------------------------------------------------------------------------------


def test_local_store_reads_files_and_reports_absence(tmp_path):
    (tmp_path / "0").mkdir()
    (tmp_path / "0" / "1.2.3").write_bytes(b"\x00chunk")
    store = LocalStore(tmp_path)
    assert isinstance(store, Store)
    assert store.get("0/1.2.3") == b"\x00chunk"
    assert store.get("0/9.9.9") is None
    assert store.get("missing/dir/key") is None
    assert store.get("0/1.2.3/below-a-file") is None
    assert store.describe() == {"kind": "local", "root": str(tmp_path.resolve())}


@pytest.mark.parametrize("make_root", [lambda p: p / "missing", lambda p: p / "file", lambda p: None,
                                       lambda p: ""])
def test_local_store_root_must_be_an_existing_directory(tmp_path, make_root):
    (tmp_path / "file").write_text("x")
    root = make_root(tmp_path)
    assert refused("IO_STORE_ROOT_INVALID", LocalStore, root).details["root"] == str(root)


def test_an_empty_root_is_refused_not_read_as_the_working_directory(tmp_path, monkeypatch):
    (tmp_path / "k").write_bytes(b"in the working directory")
    monkeypatch.chdir(tmp_path)
    assert LocalStore(".").get("k") == b"in the working directory"  # an explicit "." is the caller's choice
    assert refused("IO_STORE_ROOT_INVALID", LocalStore, "").details["root"] == ""


def test_local_store_refuses_unsafe_keys_before_touching_the_filesystem(tmp_path):
    (tmp_path / "secret").write_text("x")
    store = LocalStore(tmp_path / ".." / tmp_path.name)
    assert refused("IO_STORE_KEY_INVALID", store.get, "../secret").details["key"] == "../secret"


def test_symlinks_inside_the_root_are_followed(tmp_path):
    root = tmp_path / "root"
    (root / "real").mkdir(parents=True)
    (root / "real" / "c").write_bytes(b"inside")
    os.symlink(root / "real", root / "alias")
    os.symlink("real/c", root / "link-to-file")
    store = LocalStore(root)
    assert store.get("alias/c") == b"inside"
    assert store.get("link-to-file") == b"inside"
    assert store.get("alias/missing/deeper") is None  # below a working link: absent
    assert store.get("link-to-file/below") is None  # below a link to a file, as below a file: absent


def test_a_symlinked_root_is_resolved_once(tmp_path):
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "k").write_bytes(b"v")
    os.symlink(tmp_path / "real", tmp_path / "via-link")
    store = LocalStore(tmp_path / "via-link")
    assert store.root == (tmp_path / "real").resolve()
    assert store.get("k") == b"v"


@pytest.mark.parametrize("key,link,target,resolved", [
    ("leak", "leak", "outside/secret", "outside/secret"),          # a file outside
    ("out/secret", "out", "outside", "outside/secret"),            # a directory outside, existing file
    ("out/nothing-here", "out", "outside", "outside/nothing-here"),  # outside and absent: not reported absent
    ("dangling-out", "dangling-out", "outside/none", "outside/none"),
])
def test_symlinks_leaving_the_root_are_refused_naming_key_and_target(tmp_path, key, link, target, resolved):
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "secret").write_text("x")
    os.symlink(tmp_path / target, root / link)
    exc = refused("IO_STORE_ESCAPE", LocalStore(root).get, key)
    assert exc.details["key"] == key
    assert exc.details["resolved"] == str(tmp_path.resolve() / resolved)


@pytest.mark.parametrize("make", ["directory", "dangling", "loop"])
def test_non_regular_entries_are_refused_not_reported_absent(tmp_path, make):
    if make == "directory":
        (tmp_path / "k").mkdir()
    elif make == "dangling":
        os.symlink(tmp_path / "missing-target", tmp_path / "k")
    else:
        os.symlink(tmp_path / "loop-b", tmp_path / "k")
        os.symlink(tmp_path / "k", tmp_path / "loop-b")
    exc = refused("IO_STORE_NOT_A_FILE", LocalStore(tmp_path).get, "k")
    assert exc.details == {"key": "k", "resolved": exc.details["resolved"]}  # the key itself, not a component


@pytest.mark.parametrize("make", ["dangling", "loop"])
@pytest.mark.parametrize("key,component", [("lv/1/0/0", "lv/1"), ("lv/1/0", "lv/1"), ("lv/1/x/y/z", "lv/1")])
def test_a_key_below_a_dangling_or_looping_link_is_refused_naming_the_link(tmp_path, make, key, component):
    (tmp_path / "lv" / "0" / "0").mkdir(parents=True)
    (tmp_path / "lv" / "0" / "0" / "0").write_bytes(b"chunk")
    if make == "dangling":  # e.g. a chunk-row directory moved away: its chunks must not read as absent
        os.symlink(tmp_path / "lv" / "moved-away", tmp_path / "lv" / "1")
    else:
        os.symlink(tmp_path / "lv" / "2", tmp_path / "lv" / "1")
        os.symlink(tmp_path / "lv" / "1", tmp_path / "lv" / "2")
    store = LocalStore(tmp_path)
    exc = refused("IO_STORE_NOT_A_FILE", store.get, key)
    assert exc.details["key"] == key and exc.details["component"] == component
    assert store.get("lv/0/0/0") == b"chunk" and store.get("lv/0/0/1") is None and store.get("lv/3/0/0") is None


# --- unreadable is never absent ------------------------------------------------------------------


def test_a_key_that_cannot_be_looked_up_is_refused_not_read_as_absent(tmp_path):
    import errno

    (tmp_path / "0").mkdir()
    store = LocalStore(tmp_path)
    key = "0/" + "x" * 300  # longer than NAME_MAX: lstat fails with ENAMETOOLONG, which says nothing of absence
    exc = refused("IO_STORE_UNREADABLE", store.get, key)
    assert exc.details["key"] == key and exc.details["errno"] == errno.ENAMETOOLONG
    assert exc.details["path"].endswith("x" * 300)


def _denied_below(monkeypatch, blocked):
    """os.lstat and os.stat fail with EACCES for paths below ``blocked``, as for an unsearchable directory."""
    import errno

    for name in ("lstat", "stat"):
        real = getattr(os, name)

        def fake(path, *args, _real=real, **kwargs):
            if str(path).startswith(str(blocked) + os.sep):
                raise PermissionError(errno.EACCES, "Permission denied", str(path))
            return _real(path, *args, **kwargs)

        monkeypatch.setattr(os, name, fake)


def test_a_chunk_below_an_unsearchable_directory_is_refused_not_read_as_fill(tmp_path, monkeypatch):
    import errno

    from io_zarr_fixtures import small_levels, write_small

    from apok_eval.io.frame import VolumeFrame
    from apok_eval.io.omezarr import open_volume

    write_small(tmp_path)  # level 1 is nested ('/'): 1/0 holds the chunks of z-row 0
    volume = open_volume(LocalStore(tmp_path), VolumeFrame("20241024131838", (9, 7, 5), 7.91, "test"))
    _denied_below(monkeypatch, tmp_path / "1" / "0")
    exc = refused("IO_STORE_UNREADABLE", volume.read, (0, 0, 0), (2, 2, 2), 1)
    assert exc.details == {"key": "1/0/0/0", "path": str(tmp_path / "1" / "0" / "0" / "0"), "errno": errno.EACCES}
    assert np.array_equal(volume.read((0, 0, 0), (2, 2, 2), 0)[0], small_levels()[0][:2, :2, :2])  # level 0 reads


def test_an_unsearchable_level_is_refused_not_reported_as_a_missing_zarray(tmp_path, monkeypatch):
    import errno

    from io_zarr_fixtures import write_small

    from apok_eval.io.frame import VolumeFrame
    from apok_eval.io.omezarr import open_volume

    write_small(tmp_path)
    _denied_below(monkeypatch, tmp_path / "2")
    exc = refused("IO_STORE_UNREADABLE", open_volume, LocalStore(tmp_path),
                  VolumeFrame("20241024131838", (9, 7, 5), 7.91, "test"))
    assert exc.details["key"] == "2/.zarray" and exc.details["errno"] == errno.EACCES


def test_a_file_that_cannot_be_read_is_refused_with_its_errno(tmp_path, monkeypatch):
    import errno

    from apok_eval.io import store as store_module

    (tmp_path / "k").write_bytes(b"x")

    def denied(path):
        raise PermissionError(errno.EACCES, "Permission denied", str(path))

    monkeypatch.setattr(store_module, "read_regular", denied)
    exc = refused("IO_STORE_UNREADABLE", LocalStore(tmp_path).get, "k")
    assert exc.details == {"key": "k", "path": str(tmp_path.resolve() / "k"), "errno": errno.EACCES}


@pytest.mark.skipif(not os.path.exists("/proc/self/mem"), reason="needs Linux /proc")
def test_a_regular_file_whose_read_fails_is_refused_not_absent():
    import errno

    exc = refused("IO_STORE_UNREADABLE", LocalStore("/proc/self").get, "mem")  # reading offset 0 gives EIO
    assert exc.details["key"] == "mem" and exc.details["errno"] == errno.EIO


@pytest.mark.parametrize("root", ["nul\x00root", "surrogate\udcff"])
def test_a_root_that_is_not_a_path_is_refused(root):
    assert refused("IO_STORE_ROOT_INVALID", LocalStore, root).details["root"] == root


# --- the S3 mapping ---------------------------------------------------------------------------


@pytest.mark.parametrize("uri,url", [
    ("s3://vesuvius-challenge-open-data/PHerc0172/volumes/20241024131838-7.910um-53keV-masked.zarr",
     "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0172/volumes/"
     "20241024131838-7.910um-53keV-masked.zarr"),
    ("s3://apok-test-bucket/a/b.zarr/", "https://apok-test-bucket.s3.us-east-1.amazonaws.com/a/b.zarr"),
    ("s3://apok-test-bucket", "https://apok-test-bucket.s3.us-east-1.amazonaws.com"),
    ("s3://abc/", "https://abc.s3.us-east-1.amazonaws.com"),
])
def test_s3_uris_map_to_one_virtual_hosted_us_east_1_form(uri, url):
    assert s3_to_https(uri) == url
    assert S3_HTTPS_RULE == "s3://<bucket>/<key> -> https://<bucket>.s3.us-east-1.amazonaws.com/<key>"


@pytest.mark.parametrize("uri,cause", [
    ("s3://Upper-Case/x", ("bucket", "Upper-Case")), ("s3://my.bucket/x", ("bucket", "my.bucket")),
    ("s3://ab/x", ("bucket", "ab")), ("s3://-lead/x", ("bucket", "-lead")), ("s3://a_b_c/x", ("bucket", "a_b_c")),
    ("s3://user@bucket/x", ("bucket", "user@bucket")), ("s3://bucket:443/x", ("bucket", "bucket:443")),
    ("s3://bucket/a//b", ("path", "/a//b")), ("s3://bucket/a/../b", ("path", "/a/../b")),
    ("s3://bucket/%41", ("path", "/%41")),
    ("s3://bucket/x?versionId=1", ("uri", "s3://bucket/x?versionId=1")),
    ("s3://bucket/x#frag", ("uri", "s3://bucket/x#frag")), ("https://bucket/x", ("uri", "https://bucket/x")),
    (7, ("uri", 7)),
])
def test_malformed_s3_uris_are_refused_naming_the_part(uri, cause):
    exc = refused("IO_READ_URI_INVALID", s3_to_https, uri)
    assert exc.details[cause[0]] == cause[1]
    assert exc.details["uri"] == uri


# --- the licence gate's answers ------------------------------------------------------------------

REAL = object()  # GateSpy.answer when the gate's own covers() answers


class GateSpy:
    """Every question put to any LicenceClearance during a test, answered by the gate's own code unless
    ``answer`` is set (to simulate a gate whose answer for some URI is not the bool True)."""

    def __init__(self) -> None:
        self.asked: list[str] = []
        self.answer = REAL


@pytest.fixture
def gate(monkeypatch):
    spy = GateSpy()
    real_covers = LicenceClearance.covers

    def covers(self, uri):
        spy.asked.append(uri)
        return real_covers(self, uri) if spy.answer is REAL else spy.answer

    monkeypatch.setattr(LicenceClearance, "covers", covers)
    return spy


@pytest.mark.parametrize("uri,normalised", [
    ("S3://apok-test-bucket/vol.zarr", "s3://apok-test-bucket/vol.zarr"),  # urlsplit lowercases the scheme
    ("HTTPS://example.org/vol.zarr", "https://example.org/vol.zarr"),
    (" s3://apok-test-bucket/vol.zarr", "s3://apok-test-bucket/vol.zarr"),  # leading whitespace stripped
    ("\x01https://example.org/vol.zarr", "https://example.org/vol.zarr"),  # leading control character stripped
    ("https://exa\nmple.org/vol.zarr", "https://example.org/vol.zarr"),  # tab, CR, LF removed anywhere
    ("s3://apok-test\t-bucket/vol.zarr", "s3://apok-test-bucket/vol.zarr"),
    ("https://example.org/vol.zarr?", "https://example.org/vol.zarr"),  # an empty query dropped
    ("https://example.org/vol.zarr#", "https://example.org/vol.zarr"),  # an empty fragment dropped
    ("https://exämple.org/vol.zarr", "https://exämple.org/vol.zarr"),  # unchanged, but not ASCII (IDNA)
    ("https://example.org/a b", "https://example.org/a b"),  # unchanged, but a space
    ("http://[::1/vol.zarr", None),  # urlsplit itself refuses an unclosed IPv6 bracket
])
def test_base_uris_not_in_normal_form_are_refused_before_the_clearance_is_asked(tmp_path, gate, uri, normalised):
    clearance = clearance_for(tmp_path, "s3://apok-test-bucket/vol.zarr", "https://example.org/vol.zarr")
    exc = refused("IO_READ_URI_INVALID", HttpStore, uri, clearance)
    assert exc.details == {"uri": uri, "normalised": normalised}
    assert gate.asked == []
    if uri.lower().lstrip().startswith("s3"):
        assert refused("IO_READ_URI_INVALID", s3_to_https, uri).details["normalised"] == normalised


# --- HttpStore construction: only a gate-issued clearance ---------------------------------------


class SubclassedClearance(LicenceClearance):
    """NOT gate-issued: a subclass that skips the gate's token check and answers True."""

    __slots__ = ()

    def __init__(self) -> None:
        pass

    def covers(self, uri: str) -> bool:
        return True


class ClaimsToBeAClearance:
    """NOT gate-issued: names LicenceClearance as its ``__class__`` (which isinstance believes) and answers True."""

    @property
    def __class__(self):
        return LicenceClearance

    def covers(self, uri: str) -> bool:
        return True


def spec_mock(make=lambda: mock.Mock(spec=LicenceClearance)):
    """NOT gate-issued: the usual accidental stand-in in a test, answering True."""
    stand_in = make()
    stand_in.covers.return_value = True
    return stand_in


def build_and_read(uri: str, clearance) -> bytes | None:
    return HttpStore(uri, clearance).get(".zgroup")


@pytest.mark.parametrize("make,received,passes_isinstance", [
    (DuckClearance, "io_zarr_fixtures.DuckClearance", False),  # covers() -> True: the runtime protocol accepts it
    (spec_mock, "unittest.mock.Mock", True),
    (lambda: spec_mock(lambda: mock.create_autospec(LicenceClearance, instance=True)),
     "unittest.mock.NonCallableMagicMock", True),
    (ClaimsToBeAClearance, "test_io_store.ClaimsToBeAClearance", True),
    (SubclassedClearance, "test_io_store.SubclassedClearance", True),
    (lambda: None, "builtins.NoneType", False),
    (object, "builtins.object", False),
    (lambda: "s3://apok-test-bucket/vol.zarr", "builtins.str", False),
])
def test_anything_but_a_gate_issued_clearance_is_refused_naming_its_type(tmp_path, make, received,
                                                                         passes_isinstance):
    stand_in = make()
    (tmp_path / "vol.zarr").mkdir()
    (tmp_path / "vol.zarr" / ".zgroup").write_bytes(b'{"zarr_format": 2}')
    with LoopbackServer(tmp_path) as server:
        base = server.url + "/vol.zarr"
        exc = outcome(build_and_read, base, stand_in)
        assert server.requests == []  # checked first: a store built on a stand-in answering True reads .zgroup
    assert exc is not None and exc.code == "IO_READ_NOT_CLEARED", exc
    assert exc.details == {"uri": base, "clearance_type": received}  # refused building the store (no key)
    assert received in exc.reason
    if isinstance(stand_in, DuckClearance):
        assert isinstance(stand_in, Clearance)  # what the old structural check accepted
        assert stand_in.asked == []  # its covers() is never even called
    assert isinstance(stand_in, LicenceClearance) is passes_isinstance  # what isinstance against the class says
    if isinstance(stand_in, mock.NonCallableMock):
        stand_in.covers.assert_not_called()


def test_a_real_clearance_that_does_not_cover_the_base_refuses_before_any_request(tmp_path):
    for name in ("vol.zarr", "vol.zarr2"):
        (tmp_path / name).mkdir()
        (tmp_path / name / ".zgroup").write_bytes(b'{"zarr_format": 2}')
    with LoopbackServer(tmp_path) as server:
        base = server.url + "/vol.zarr"
        for clearance, uri in ((clearance_for(tmp_path, server.url + "/other.zarr"), base),  # another asset
                               (clearance_for(tmp_path, base), server.url),  # a parent of the cleared URI
                               (clearance_for(tmp_path, base), server.url + "/vol.zarr2")):  # not whole segments
            assert isinstance(clearance, LicenceClearance)
            exc = outcome(build_and_read, uri, clearance)
            assert server.requests == []  # checked first: a store built on this clearance would have read
            assert exc is not None and exc.code == "IO_READ_NOT_CLEARED", exc
            assert exc.details == {"uri": uri, "clearance_type": "apok_eval.licence.gate.LicenceClearance"}  # at the base


@pytest.mark.parametrize("answer", [False, "yes", 1, None])
def test_a_gate_answer_other_than_the_bool_true_does_not_clear_the_base(tmp_path, gate, answer):
    base = "http://127.0.0.1:9/vol.zarr"
    clearance = clearance_for(tmp_path, base)
    gate.answer = answer
    exc = refused("IO_READ_NOT_CLEARED", HttpStore, base, clearance)
    assert exc.details["uri"] == base and gate.asked == [base]


def test_the_clearance_is_asked_about_the_base_and_every_object_in_the_callers_form(tmp_path, gate):
    store = HttpStore("s3://apok-test-bucket/vol.zarr/", clearance_for(tmp_path, "s3://apok-test-bucket/vol.zarr"))
    assert gate.asked == ["s3://apok-test-bucket/vol.zarr/"]
    assert store.object_uri("0/.zarray") == "s3://apok-test-bucket/vol.zarr/0/.zarray"
    assert store.describe() == {
        "kind": "http", "base_uri": "s3://apok-test-bucket/vol.zarr/",
        "base_url": "https://apok-test-bucket.s3.us-east-1.amazonaws.com/vol.zarr",
        "s3_mapping": S3_HTTPS_RULE, "retries": 0, "retry_events": []}
    gate.answer = False  # the gate says no to the object, so nothing is requested from the bucket
    exc = refused("IO_READ_NOT_CLEARED", store.get, "0/.zarray")
    assert exc.details == {"uri": "s3://apok-test-bucket/vol.zarr/0/.zarray", "key": "0/.zarray"}
    assert gate.asked[1:] == ["s3://apok-test-bucket/vol.zarr/0/.zarray"]  # the caller's form, not the https URL
    gate.answer = REAL
    https = "https://example.org/vol.zarr"
    assert HttpStore(https, clearance_for(tmp_path, https)).describe()["s3_mapping"] is None


@pytest.mark.parametrize("uri,detail", [
    ("ftp://127.0.0.1/x", ("scheme", "ftp")), ("file:///etc", ("scheme", "file")), ("x", ("scheme", "")),
    (None, ("scheme", None)),
    ("http://example.org/vol.zarr", ("host", "example.org")), ("http://169.254.169.254/latest", ("host", "169.254.169.254")),
    ("http://localhost/vol.zarr", ("host", "localhost")),  # a name, not a loopback literal: the gate never clears it
    ("https://user:pw@example.org/x", ("uri", "https://user:pw@example.org/x")),
    ("https://example.org/x?q=1", ("uri", "https://example.org/x?q=1")),
    ("https://example.org/x#f", ("uri", "https://example.org/x#f")),
    ("https:///x", ("uri", "https:///x")), ("https://example.org:99999/x", ("uri", "https://example.org:99999/x")),
    ("https://example.org/a//b", ("path", "/a//b")), ("http://127.0.0.1/a/../b", ("path", "/a/../b")),
])
def test_base_uris_outside_the_allowed_forms_are_refused(tmp_path, gate, uri, detail):
    clearance = clearance_for(tmp_path, "https://example.org/x", "http://127.0.0.1/a")
    exc = refused("IO_READ_URI_INVALID", HttpStore, uri, clearance)
    assert exc.details[detail[0]] == detail[1]
    assert gate.asked == []  # the gate is not consulted about URIs apok-eval would never read


def test_loopback_ip_literal_http_forms_are_accepted(tmp_path):
    uris = ("http://127.0.0.1:8/x", "http://[::1]:8/x", "http://127.9.9.9/x")
    clearance = clearance_for(tmp_path, *uris, "https://example.org/a.zarr")
    for uri in uris:
        assert HttpStore(uri, clearance).base_url == uri
    assert HttpStore("https://example.org/a.zarr/", clearance).base_url == "https://example.org/a.zarr"


@pytest.mark.parametrize("option,value", [
    ("retries", -1), ("retries", 6), ("retries", True), ("retries", 1.0),
    ("retry_wait_s", -0.1), ("retry_wait_s", float("nan")), ("retry_wait_s", "1"), ("retry_wait_s", 1e300),
    ("retry_wait_s", 3600.5), ("timeout_s", 0), ("timeout_s", float("inf")), ("timeout_s", None),
    ("timeout_s", 1e300), ("timeout_s", 3601),
])
def test_out_of_range_options_are_refused_naming_the_option(tmp_path, option, value):
    clearance = clearance_for(tmp_path, "http://127.0.0.1/x")
    exc = refused("IO_READ_OPTION_INVALID", HttpStore, "http://127.0.0.1/x", clearance, **{option: value})
    assert exc.details["field"] == option


# --- HttpStore reads ----------------------------------------------------------------------------


@pytest.fixture
def server(tmp_path):
    (tmp_path / "vol.zarr" / "0").mkdir(parents=True)
    (tmp_path / "vol.zarr" / ".zgroup").write_bytes(b'{"zarr_format": 2}')
    (tmp_path / "vol.zarr" / "0" / "0.0.0").write_bytes(bytes(range(256)))
    with LoopbackServer(tmp_path) as srv:
        srv.clearance = clearance_for(tmp_path, srv.url + "/vol.zarr")
        yield srv


def store_for(server, **options) -> HttpStore:
    return HttpStore(server.url + "/vol.zarr", server.clearance, retry_wait_s=0, **options)


def test_200_returns_the_bytes_as_served_and_404_is_absent(server, gate):
    store = store_for(server)
    assert store.get(".zgroup") == b'{"zarr_format": 2}'
    assert store.get("0/0.0.0") == bytes(range(256))
    assert store.get("0/9.9.9") is None
    assert server.paths() == ["/vol.zarr/.zgroup", "/vol.zarr/0/0.0.0", "/vol.zarr/0/9.9.9"]
    assert gate.asked == [server.url + p for p in ["/vol.zarr", *server.paths()]]  # the base, then each object


def test_requests_are_anonymous_plain_gets(server):
    store_for(server).get(".zgroup")
    headers = {k.lower(): v for k, v in server.requests[0]["headers"].items()}
    assert headers["user-agent"] == USER_AGENT
    assert headers["accept-encoding"] == "identity"  # added by http.client: the stored bytes, not a re-encoding
    assert not {"authorization", "cookie", "x-amz-security-token", "x-amz-date"} & set(headers)


def test_environment_proxies_are_not_used(server, monkeypatch):
    with socket.socket() as dead:  # a bound, never-accepting port: any proxied request would fail
        dead.bind(("127.0.0.1", 0))
        proxy = f"http://127.0.0.1:{dead.getsockname()[1]}"
        for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
            monkeypatch.setenv(name, proxy)
        for name in ("no_proxy", "NO_PROXY"):
            monkeypatch.delenv(name, raising=False)
        assert store_for(server, timeout_s=5).get(".zgroup") == b'{"zarr_format": 2}'


@pytest.mark.parametrize("answer", [False, "yes", 1, None])
def test_an_object_the_gate_does_not_answer_true_for_is_never_requested(server, gate, answer):
    # Covering the base does not imply covering an object under it where the gate remaps a path prefix (next
    # test): a clearance of https://data.aws.ash2txt.org does not cover its samples/ objects. A real clearance for
    # that host needs a person's own acceptance of the Vesuvius Challenge terms, so the gate's answer is patched here.
    store = store_for(server)
    gate.answer = answer
    exc = refused("IO_READ_NOT_CLEARED", store.get, ".zgroup")
    assert exc.details == {"uri": server.url + "/vol.zarr/.zgroup", "key": ".zgroup"}
    assert server.requests == []


def test_coverage_is_not_monotone_where_the_gate_remaps_a_path_so_each_object_is_asked():
    # The premise of the per-object question (docs/io-zarr.md, section 1): the gate reads the gateway path
    # https://data.aws.ash2txt.org/samples/<k> as s3://vesuvius-challenge/<k>, which is not under the host root.
    base = canonical_uri("https://data.aws.ash2txt.org")
    remapped = canonical_uri("https://data.aws.ash2txt.org/samples/PHerc0332/x.zarr/.zgroup")
    assert str(remapped) == "s3://vesuvius-challenge/PHerc0332/x.zarr/.zgroup" and not remapped.is_under(base)
    assert canonical_uri("https://data.aws.ash2txt.org/other/x.zarr/.zgroup").is_under(base)


def test_unsafe_keys_are_refused_before_any_request(server):
    refused("IO_STORE_KEY_INVALID", store_for(server).get, "../secret")
    assert server.requests == []


def test_401_is_refused_as_auth_required_and_never_retried(server):
    server.route("/vol.zarr/k", (401, b"Unauthorized", {"WWW-Authenticate": "Basic"}))
    exc = refused("IO_READ_AUTH_REQUIRED", store_for(server, retries=2).get, "k")
    assert exc.details["status"] == 401 and len(server.requests) == 1


def test_listings_are_read_with_a_trailing_slash_and_cleared_first(server, gate):
    server.route("/vol.zarr/0/", (200, b"<html>index</html>"))
    server.route("/vol.zarr/", (200, b"<html>root</html>"))
    store = store_for(server)
    assert store.get_listing("0") == b"<html>index</html>" and store.get_listing() == b"<html>root</html>"
    assert gate.asked[-2:] == [server.url + "/vol.zarr/0/", server.url + "/vol.zarr/"]
    refused("IO_STORE_KEY_INVALID", store.get_listing, "../x")
    gate.answer = False
    exc = refused("IO_READ_NOT_CLEARED", store.get_listing, "0")
    assert exc.details == {"uri": server.url + "/vol.zarr/0/", "key": "0"}


def test_a_base_that_names_a_file_is_read_whole(server, gate):
    store = HttpStore(server.url + "/vol.zarr/.zgroup", server.clearance, retry_wait_s=0)
    assert store.get_base() == b'{"zarr_format": 2}'
    gate.answer = False
    assert refused("IO_READ_NOT_CLEARED", store.get_base).details["key"] == ""


def test_403_is_refused_never_absent(server):
    server.route("/vol.zarr/0/1.0.0", (403, b"<Error>AccessDenied</Error>"))
    exc = refused("IO_READ_FORBIDDEN", store_for(server).get, "0/1.0.0")
    assert exc.details["key"] == "0/1.0.0" and exc.details["status"] == 403


@pytest.mark.parametrize("status", [302, 301, 204, 206, 418])
def test_other_statuses_are_refused_with_the_status_and_never_retried(server, status):
    server.route("/vol.zarr/k", (status, b"", {"Location": "/vol.zarr/.zgroup"}))
    exc = refused("IO_READ_HTTP_STATUS", store_for(server, retries=2).get, "k")
    assert exc.details["status"] == status and exc.details["key"] == "k" and exc.details["attempts"] == 1
    assert server.paths() == ["/vol.zarr/k"]  # neither retried nor, for a redirect, followed


def test_5xx_without_retries_is_refused_after_one_attempt(server):
    server.route("/vol.zarr/k", (500, b""), (200, b"would have worked"))
    exc = refused("IO_READ_HTTP_STATUS", store_for(server).get, "k")
    assert exc.details["status"] == 500 and exc.details["attempts"] == 1
    assert server.paths() == ["/vol.zarr/k"]


def test_5xx_is_retried_within_the_bound_and_every_retry_recorded(server):
    server.route("/vol.zarr/k", (503, b""), (500, b""), (200, b"late"))
    store = store_for(server, retries=2)
    assert store.get("k") == b"late"
    assert store.retry_events == [{"key": "k", "status": 503, "retry": 1}, {"key": "k", "status": 500, "retry": 2}]
    assert store.describe()["retry_events"] == store.retry_events
    assert server.paths() == ["/vol.zarr/k"] * 3


def test_5xx_beyond_the_bound_is_refused_after_exactly_retries_plus_one_attempts(server):
    server.route("/vol.zarr/k", (503, b""))
    store = store_for(server, retries=2)
    exc = refused("IO_READ_HTTP_STATUS", store.get, "k")
    assert exc.details["status"] == 503 and exc.details["attempts"] == 3
    assert server.paths() == ["/vol.zarr/k"] * 3
    assert store.retry_events == [{"key": "k", "status": 503, "retry": 1}, {"key": "k", "status": 503, "retry": 2}]


def test_a_truncated_body_is_a_network_refusal(server):
    server.route("/vol.zarr/k", (200, b"x" * 10, {"Content-Length": "100"}))
    exc = refused("IO_READ_NETWORK", store_for(server).get, "k")
    assert exc.details["key"] == "k" and exc.details["error"] == "IncompleteRead"


def test_the_largest_waits_and_timeouts_are_accepted(tmp_path):
    clearance = clearance_for(tmp_path, "http://127.0.0.1/x")
    store = HttpStore("http://127.0.0.1/x", clearance, retry_wait_s=3600, timeout_s=3600)
    assert (store.retry_wait_s, store.timeout_s) == (3600.0, 3600.0)


def test_a_malformed_chunked_response_is_a_network_refusal(tmp_path):
    import threading

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        def answer():
            connection, _ = listener.accept()
            with connection:
                connection.recv(65536)
                connection.sendall(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n-5\r\nxxxxx\r\n0\r\n\r\n")

        worker = threading.Thread(target=answer, daemon=True)
        worker.start()
        base = f"http://127.0.0.1:{port}/vol.zarr"
        exc = refused("IO_READ_NETWORK", HttpStore(base, clearance_for(tmp_path, base), timeout_s=5).get, "k")
        worker.join(5)
    assert exc.details["url"] == base + "/k" and exc.details["error"] == "ValueError"


def test_a_connection_failure_is_a_network_refusal(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}/vol.zarr"
    exc = refused("IO_READ_NETWORK", HttpStore(base, clearance_for(tmp_path, base), timeout_s=5).get, ".zgroup")
    assert exc.details["url"] == base + "/.zgroup" and exc.details["error"] == "ConnectionRefusedError"
    assert exc.details["attempts"] == 1


# --- transport retries: a failed transfer is retried like a 5xx ---------------------------------------------

OK_HELLO = b"HTTP/1.0 200 OK\r\nContent-Length: 5\r\n\r\nhello"


@contextlib.contextmanager
def scripted(*actions):
    """A loopback server answering its n-th connection with ``actions[n]`` (the last repeating) after reading the
    request: ``"reset"`` closes with a TCP reset, ``"stall"`` answers nothing until the client gives up, bytes are
    sent as the whole answer. Yields its ``http://127.0.0.1:<port>`` and the list of actions taken so far."""
    import struct
    import threading

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    listener.settimeout(0.05)
    taken: list = []
    stop = threading.Event()

    def serve():
        while not stop.is_set():
            try:
                connection, _ = listener.accept()
            except (TimeoutError, OSError):
                continue
            action = actions[min(len(taken), len(actions) - 1)]
            taken.append(action)
            with connection, contextlib.suppress(OSError):  # a client that goes away early ends only its connection
                connection.settimeout(10)
                connection.recv(65536)
                if action == "reset":
                    connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                elif action == "stall":
                    connection.recv(65536)  # returns when the client closes, having timed out
                else:
                    connection.sendall(action)

    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{listener.getsockname()[1]}", taken
    finally:
        stop.set()
        worker.join(5)
        listener.close()


def scripted_store(tmp_path, url, **options) -> HttpStore:
    base = url + "/vol.zarr"
    return HttpStore(base, clearance_for(tmp_path, base), retry_wait_s=0, **options)


def test_a_reset_connection_is_retried_and_the_retry_names_the_error(tmp_path):
    with scripted("reset", OK_HELLO) as (url, taken):
        store = scripted_store(tmp_path, url, retries=2, timeout_s=5)
        assert store.get("k") == b"hello"
    assert taken == ["reset", OK_HELLO]
    assert store.retry_events == [{"key": "k", "error": "ConnectionResetError", "retry": 1}]
    assert store.describe()["retry_events"] == store.retry_events


def test_a_connection_reset_every_time_is_refused_after_retries_plus_one_attempts(tmp_path):
    with scripted("reset") as (url, taken):
        store = scripted_store(tmp_path, url, retries=2, timeout_s=5)
        exc = refused("IO_READ_NETWORK", store.get, "k")
    assert exc.details == {"key": "k", "url": url + "/vol.zarr/k", "error": "ConnectionResetError", "attempts": 3}
    assert "ConnectionResetError" in exc.reason and "3 attempt(s)" in exc.reason
    assert taken == ["reset"] * 3
    assert store.retry_events == [{"key": "k", "error": "ConnectionResetError", "retry": r} for r in (1, 2)]


def test_a_request_that_stalls_past_the_timeout_is_retried(tmp_path):
    with scripted("stall", OK_HELLO) as (url, taken):
        store = scripted_store(tmp_path, url, retries=1, timeout_s=1.0)  # the answered retry has 1 s to arrive
        assert store.get("k") == b"hello"
    assert taken == ["stall", OK_HELLO]
    assert store.retry_events == [{"key": "k", "error": "TimeoutError", "retry": 1}]


def test_without_retries_a_transport_failure_is_refused_at_once(tmp_path):
    with scripted("reset", OK_HELLO) as (url, taken):
        store = scripted_store(tmp_path, url, timeout_s=5)
        exc = refused("IO_READ_NETWORK", store.get, "k")
    assert exc.details["error"] == "ConnectionResetError" and exc.details["attempts"] == 1
    assert taken == ["reset"] and store.retry_events == []


def test_a_tls_handshake_that_times_out_is_retried_then_refused(tmp_path, monkeypatch):
    # A single TLS handshake timeout must not end a long read. The name resolves, by a stand-in resolver, to a
    # loopback server that reads the ClientHello and never answers it.
    base = "https://mirror.example/vol.zarr"
    store = HttpStore(base, clearance_for(tmp_path, base), retries=1, retry_wait_s=0, timeout_s=0.3)
    with scripted("stall") as (url, taken):
        port = int(url.rsplit(":", 1)[1])
        connect = socket.create_connection
        monkeypatch.setattr(socket, "getaddrinfo", lambda host, p, *a, **k: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", p))])
        monkeypatch.setattr(socket, "create_connection", lambda address, *a, **k: connect(("127.0.0.1", port), *a, **k))
        exc = refused("IO_READ_NETWORK", store.get, ".zgroup")
    assert exc.details["error"] == "TimeoutError" and exc.details["attempts"] == 2
    assert "handshake" in exc.reason
    assert taken == ["stall", "stall"]
    assert store.retry_events == [{"key": ".zgroup", "error": "TimeoutError", "retry": 1}]


def test_retries_wait_retry_wait_s_times_the_attempt_whatever_failed(tmp_path, monkeypatch):
    import types

    import apok_eval.io.store as store_module

    waits: list = []
    monkeypatch.setattr(store_module, "time", types.SimpleNamespace(sleep=waits.append))
    with LoopbackServer(tmp_path) as server:
        server.route("/vol.zarr/k", (503, b""), (200, b"ok"))
        base = server.url + "/vol.zarr"
        store = HttpStore(base, clearance_for(tmp_path, base), retries=3, retry_wait_s=10.0)
        assert store.get("k") == b"ok"
    with scripted("reset", "reset", OK_HELLO) as (url, taken):
        store = HttpStore(url + "/vol.zarr", clearance_for(tmp_path, url + "/vol.zarr"), retries=3,
                          retry_wait_s=10.0, timeout_s=5)
        assert store.get("k") == b"hello"
    assert waits == [10.0, 10.0, 20.0]  # the 5xx's one retry; then two transport retries, 10 s x attempt


def test_a_refusal_on_the_way_to_a_request_is_never_retried(tmp_path, resolver):
    base = "https://lan.example/vol.zarr"
    store = HttpStore(base, clearance_for(tmp_path, base), retries=5, retry_wait_s=0)
    fake = resolver({"lan.example": [["10.0.0.7"]]})
    exc = refused("IO_READ_ADDRESS_FORBIDDEN", store.get, "k")
    assert exc.details["address"] == "10.0.0.7"
    assert fake.lookups == ["lan.example"] and fake.connects == [] and store.retry_events == []


def test_every_read_retries_a_transport_failure(tmp_path):
    with scripted("reset", b"HTTP/1.0 200 OK\r\nContent-Length: 4\r\n\r\nbase", "reset",
                  b"HTTP/1.0 200 OK\r\nContent-Length: 7\r\n\r\nlisting", "reset",
                  b"HTTP/1.0 206 Partial Content\r\nContent-Length: 2\r\n\r\nel") as (url, taken):
        base = url + "/vol.zarr/f.json"
        store = HttpStore(base, clearance_for(tmp_path, base), retries=1, retry_wait_s=0, timeout_s=5)
        assert store.get_base() == b"base"
        listing = HttpStore(url + "/vol.zarr", clearance_for(tmp_path, url + "/vol.zarr"), retries=1,
                            retry_wait_s=0, timeout_s=5)
        assert listing.get_listing() == b"listing"
        assert listing.get_range("k", 1, 2) == b"el"
    assert [e["error"] for e in store.retry_events + listing.retry_events] == ["ConnectionResetError"] * 3


def test_download_through_a_reset_writes_the_whole_file_and_a_final_failure_leaves_nothing(tmp_path):
    import hashlib

    out = tmp_path / "out"
    out.mkdir()
    with scripted("reset", OK_HELLO, "reset") as (url, taken):
        store = scripted_store(tmp_path, url, retries=1, timeout_s=5)
        assert store.download("k", out / "k") == (5, "sha256:" + hashlib.sha256(b"hello").hexdigest())
        assert (out / "k").read_bytes() == b"hello" and not (out / "k.partial").exists()
        assert store.retry_events == [{"key": "k", "error": "ConnectionResetError", "retry": 1}]
        exc = refused("IO_READ_NETWORK", store.download, "j", out / "j")
    assert exc.details == {"key": "j", "url": url + "/vol.zarr/j", "error": "ConnectionResetError", "attempts": 2}
    assert taken == ["reset", OK_HELLO, "reset", "reset"]
    assert sorted(p.name for p in out.iterdir()) == ["k"]


def test_a_partial_download_is_removed_when_the_retry_fails_otherwise(tmp_path):
    # Attempt 1 writes part of the body and breaks off; attempt 2 is answered 503: the part must not survive.
    with LoopbackServer(tmp_path) as server:
        server.route("/pool/x", (200, b"abc", {"Content-Length": "10"}), (503, b""))
        base = server.url + "/pool"
        store = HttpStore(base, clearance_for(tmp_path, base), retries=1, retry_wait_s=0)
        exc = refused("IO_READ_HTTP_STATUS", store.download, "x", tmp_path / "x")
    assert exc.details == {"key": "x", "url": base + "/x", "status": 503, "attempts": 2}
    assert store.retry_events == [{"key": "x", "error": "IncompleteRead", "retry": 1}]
    assert not (tmp_path / "x").exists() and not (tmp_path / "x.partial").exists()


@pytest.mark.parametrize("where,size", [("missing-directory", 64 << 10), ("full-disk", 64 << 10), ("full-disk", 3)],
                         ids=["missing-directory", "full-disk-on-write", "full-disk-on-close"])
def test_a_download_whose_local_file_cannot_be_written_is_refused_at_once_never_retried(tmp_path, where, size):
    # A local file error is not a failed transfer: a retry would request the object again for nothing, and record a
    # transport retry that never happened.
    if where == "full-disk" and not os.path.exists("/dev/full"):
        pytest.skip("no /dev/full here")
    target = tmp_path / "no-such-dir" / "x" if where == "missing-directory" else tmp_path / "x"
    partial = target.with_name("x.partial")
    if where == "full-disk":
        partial.symlink_to("/dev/full")  # every write there fails with ENOSPC
    with LoopbackServer(tmp_path) as server:
        server.route("/pool/x", (200, os.urandom(size)))  # 64 KiB passes the 8 KiB file buffer; 3 bytes wait for close
        base = server.url + "/pool"
        store = HttpStore(base, clearance_for(tmp_path, base), retries=3, retry_wait_s=0)
        exc = refused("IO_READ_SINK_UNWRITABLE", store.download, "x", target)
    error = "FileNotFoundError" if where == "missing-directory" else "OSError"
    assert exc.details == {"key": "x", "path": str(partial), "error": error} and error in exc.reason
    assert server.paths() == ["/pool/x"] and store.retry_events == []
    assert not partial.is_symlink() and not partial.exists() and not target.exists()


# --- addresses: the instance-metadata service is never contacted ---------------------------------


class Resolver:
    """Stands in for DNS: ``answers`` maps a host to the addresses it resolves to (a list per lookup, the
    last one repeating); a numeric host resolves to itself. Every lookup and connect attempt is recorded,
    and no connection is made."""

    def __init__(self, answers: dict[str, list]) -> None:
        self.answers = {host: list(a) for host, a in answers.items()}
        self.lookups: list[str] = []
        self.connects: list[tuple] = []

    def getaddrinfo(self, host, port, *args, **kwargs):
        self.lookups.append(host)
        script = self.answers.get(host)
        addresses = (script.pop(0) if len(script) > 1 else script[0]) if script else [host]
        return [(socket.AF_INET6 if ":" in a else socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, port))
                for a in addresses]

    def connect(self, sock, address):
        self.connects.append(address[:2])
        raise ConnectionRefusedError("recorded, not connected")


@pytest.fixture
def resolver(monkeypatch):
    def install(answers):
        fake = Resolver(answers)
        monkeypatch.setattr(socket, "getaddrinfo", fake.getaddrinfo)
        monkeypatch.setattr(socket.socket, "connect", lambda sock, address: fake.connect(sock, address))
        return fake
    return install


@pytest.mark.parametrize("base,answers,host,address", [
    ("https://instance-data.example/latest", {"instance-data.example": [["169.254.169.254"]]},
     "instance-data.example", "169.254.169.254"),
    ("https://v6-metadata.example/latest", {"v6-metadata.example": [["fd00:ec2::254"]]},
     "v6-metadata.example", "fd00:ec2::254"),
    ("https://mirror.example/vol.zarr", {"mirror.example": [["52.216.1.1", "169.254.169.254"]]},
     "mirror.example", "169.254.169.254"),  # one forbidden answer among global ones is enough
    ("https://lan.example/vol.zarr", {"lan.example": [["10.0.0.7"]]}, "lan.example", "10.0.0.7"),
])
def test_no_connection_is_made_to_an_address_that_is_neither_loopback_nor_global(tmp_path, resolver, base, answers,
                                                                                host, address):
    store = HttpStore(base, clearance_for(tmp_path, base))
    fake = resolver(answers)
    exc = refused("IO_READ_ADDRESS_FORBIDDEN", store.get, "iam")
    assert exc.details == {"host": host, "port": 443, "address": address}
    assert fake.connects == []


def test_plain_http_connections_are_address_checked_too(tmp_path, resolver):
    # A loopback literal resolves to itself in practice (localhost, a name, is refused as a URI); a resolver that
    # answers otherwise is still checked, so the http handler connects through the same vetting as https.
    base = "http://127.0.0.1:9/vol.zarr"
    store = HttpStore(base, clearance_for(tmp_path, base))
    fake = resolver({"127.0.0.1": [["169.254.169.254"]]})
    exc = refused("IO_READ_ADDRESS_FORBIDDEN", store.get, ".zgroup")
    assert exc.details == {"host": "127.0.0.1", "port": 9, "address": "169.254.169.254"}
    assert fake.connects == []


@pytest.mark.parametrize("uri,host,answers,address", [
    ("https://169.254.169.254/latest", "169.254.169.254", {}, "169.254.169.254"),
    ("https://[fd00:ec2::254]/latest", "fd00:ec2::254", {}, "fd00:ec2::254"),
    ("https://2852039166/latest", "2852039166", {"2852039166": [["169.254.169.254"]]}, "169.254.169.254"),
])
def test_ip_literal_hosts_cannot_be_cleared_and_the_address_check_refuses_them_too(tmp_path, resolver, uri, host,
                                                                                  answers, address):
    with pytest.raises(Refusal) as exc:  # no registry may declare an IP-literal host, so no clearance covers one
        clearance_for(tmp_path, uri)
    assert exc.value.code == "LICENCE_URI_AMBIGUOUS" and exc.value.details["what"].startswith("registry hosts[")
    fake = resolver(answers)  # and the connection layer refuses them on its own (defence in depth)
    exc = refused("IO_READ_ADDRESS_FORBIDDEN", _vetted_connection, (host, 443))
    assert exc.details == {"host": host, "port": 443, "address": address}
    assert fake.connects == []


def test_a_global_address_is_connected_to_as_resolved_once_and_each_is_tried(tmp_path, resolver):
    base = "https://s3.example/vol.zarr"
    store = HttpStore(base, clearance_for(tmp_path, base))
    # The second lookup would rebind the name to the metadata service; it is never made.
    fake = resolver({"s3.example": [["52.216.1.1", "52.216.1.2"], ["169.254.169.254"]]})
    exc = refused("IO_READ_NETWORK", store.get, ".zgroup")
    assert exc.details["error"] == "ConnectionRefusedError"
    assert fake.connects == [("52.216.1.1", 443), ("52.216.1.2", 443)]
    assert fake.lookups.count("s3.example") == 1


def test_download_streams_to_disk_hashing_as_it_goes_and_leaves_nothing_on_failure(tmp_path):
    import hashlib

    data = os.urandom(5 * 1024 + 7)
    (tmp_path / "pool").mkdir()
    (tmp_path / "pool" / "channel_0.u8").write_bytes(data)
    out = tmp_path / "out"
    out.mkdir()
    with LoopbackServer(tmp_path) as server:
        base = server.url + "/pool"
        store = HttpStore(base, clearance_for(tmp_path, base), retries=1, retry_wait_s=0)
        got = store.download("channel_0.u8", out / "c0", chunk=1000)
        assert got == (len(data), "sha256:" + hashlib.sha256(data).hexdigest())
        assert (out / "c0").read_bytes() == data and not (out / "c0.partial").exists()
        assert store.download("absent", out / "a") is None and not (out / "a").exists()
        server.route("/pool/flaky", (503, b""), (200, b"ok"))
        assert store.download("flaky", out / "f") == (2, "sha256:" + hashlib.sha256(b"ok").hexdigest())
        assert store.retry_events == [{"key": "flaky", "status": 503, "retry": 1}]
        for path, answers, code in (("/pool/auth", [(401, b"")], "IO_READ_AUTH_REQUIRED"),
                                    ("/pool/forbidden", [(403, b"")], "IO_READ_FORBIDDEN"),
                                    ("/pool/down", [(500, b""), (502, b"")], "IO_READ_HTTP_STATUS"),
                                    ("/pool/short", [(200, b"abc", {"Content-Length": "10"})], "IO_READ_NETWORK")):
            server.route(path, *answers)
            key = path.rsplit("/", 1)[1]
            exc = refused(code, store.download, key, out / key)
            assert not (out / key).exists() and not (out / f"{key}.partial").exists(), key
            assert exc.details["key"] == key
        assert refused("IO_READ_HTTP_STATUS", store.download, "down", out / "d").details["attempts"] == 2
    other = server.url + "/other"
    with pytest.raises(Refusal) as r:
        store.download("../other/x", out / "x")
    assert r.value.code.startswith("IO_") and not (out / "x").exists() and other


def test_an_s3_folder_is_listed_by_list_objects_v2_after_clearance(tmp_path):
    from apok_eval.io.store import s3_list_url

    assert s3_list_url("s3://apok-test-bucket/a/b-1/") == (
        "https://apok-test-bucket.s3.us-east-1.amazonaws.com/?list-type=2&delimiter=%2F&max-keys=1000&prefix=a/b-1/")
    assert s3_list_url("s3://apok-test-bucket/") == (
        "https://apok-test-bucket.s3.us-east-1.amazonaws.com/?list-type=2&delimiter=%2F&max-keys=1000&prefix=")
    for bad in ("s3://apok-test-bucket/a", "https://apok-test-bucket/a/", "s3://apok-test-bucket/a/../b/"):
        refused("IO_READ_URI_INVALID", s3_list_url, bad)
    clearance = clearance_for(tmp_path, "s3://apok-test-bucket/seg/mesh")
    store = HttpStore("s3://apok-test-bucket/seg/mesh", clearance)
    asked = []
    with mock.patch.object(HttpStore, "_get", lambda self, url, key: asked.append((url, key)) or b"<xml/>"):
        assert store.get_listing("intermediate") == b"<xml/>" and store.get_listing() == b"<xml/>"
    assert asked == [
        ("https://apok-test-bucket.s3.us-east-1.amazonaws.com/?list-type=2&delimiter=%2F&max-keys=1000&prefix="
         "seg/mesh/intermediate/", "intermediate/"),
        ("https://apok-test-bucket.s3.us-east-1.amazonaws.com/?list-type=2&delimiter=%2F&max-keys=1000&prefix="
         "seg/mesh/", "/")]
    refused("IO_STORE_KEY_INVALID", store.get_listing, "../y")
    with mock.patch.object(HttpStore, "_get", lambda self, url, key: asked.append(url) or b""), \
            mock.patch.object(LicenceClearance, "covers", lambda self, uri: asked.append(uri) or False):
        exc = refused("IO_READ_NOT_CLEARED", store.get_listing, "intermediate")
    assert exc.details["uri"] == "s3://apok-test-bucket/seg/mesh/intermediate/"
    assert asked[-1] == "s3://apok-test-bucket/seg/mesh/intermediate/"  # the gate was asked; nothing was requested


def test_byte_ranges_are_cleared_and_sliced(server, gate):
    store = store_for(server)
    server.route("/vol.zarr/f", (200, b"0123456789"))
    assert store.get_range("f", 2, 3) == b"234"
    assert server.requests[-1]["headers"].get("Range") == "bytes=2-4"
    server.route("/vol.zarr/g", (206, b"abc"))
    assert store.get_range("g", 100, 3) == b"abc"
    server.route("/vol.zarr/h", (416, b""))
    assert store.get_range("h", 10 ** 6, 3) == b""
    assert store.get_range("absent", 0, 3) is None
    for start, length in ((-1, 3), (0, 0), (0, 1 << 27), (0.5, 3)):
        refused("IO_READ_OPTION_INVALID", store.get_range, "f", start, length)
    gate.answer = False
    refused("IO_READ_NOT_CLEARED", store.get_range, "f", 0, 3)
