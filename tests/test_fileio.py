"""apok_eval.fileio: regular-file reads that never block, and the one atomic writer."""

from __future__ import annotations

import os

import pytest

from apok_eval import fileio
from apok_eval.canonical import sha256_bytes
from apok_eval.fileio import NotRegularFile, TargetExists, digest_and_size, read_regular, write_atomic


def test_a_regular_file_is_read_and_digested_from_its_bytes(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc" * 1000)
    assert read_regular(path) == b"abc" * 1000
    assert digest_and_size(path) == (sha256_bytes(b"abc" * 1000), 3000)


@pytest.mark.parametrize("make, kind", [
    (lambda p: p.mkdir(), "a directory"),
    (lambda p: os.mkfifo(p), "a FIFO"),
    (lambda p: os.symlink("/dev/null", p), "a character device"),
])
def test_anything_but_a_regular_file_is_refused_without_blocking_and_named(tmp_path, make, kind):
    path = tmp_path / "x"
    make(path)
    for read in (read_regular, digest_and_size):
        error = _raised_within(10, read, path)
        if error is None and kind == "a FIFO":  # the reader is blocked on the FIFO: release it, then fail
            os.close(os.open(path, os.O_WRONLY | os.O_NONBLOCK))
        assert isinstance(error, NotRegularFile), f"{read.__name__} did not refuse {kind}: {error!r}"
        assert not error.missing and error.kind == kind
        assert f"not a regular file ({kind}, mode " in str(error) and error.filename == str(path)


def _raised_within(seconds: float, fn, *args):
    """What ``fn(*args)`` raised, run in a thread; None if it is still running after ``seconds`` (blocked)."""
    import threading

    outcome: list = []

    def run():
        try:
            fn(*args)
            outcome.append(AssertionError("returned without raising"))
        except BaseException as exc:  # noqa: BLE001 - the test inspects it
            outcome.append(exc)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(seconds)
    return outcome[0] if outcome else None


@pytest.mark.parametrize("rel", ["missing", "missing/below", "file/below"])
def test_nothing_there_is_missing(tmp_path, rel):
    (tmp_path / "file").write_bytes(b"x")
    with pytest.raises(NotRegularFile) as info:
        read_regular(tmp_path / rel)
    assert info.value.missing and info.value.kind is None and "no such file" in str(info.value)


def test_atomic_write_never_clobbers_without_overwrite(tmp_path):
    target = tmp_path / "f.json"
    write_atomic(str(target), b"one", overwrite=False)
    with pytest.raises(TargetExists):
        write_atomic(str(target), b"two", overwrite=False)
    assert issubclass(TargetExists, FileExistsError)
    assert target.read_bytes() == b"one"
    write_atomic(target, b"three", overwrite=True)
    assert target.read_bytes() == b"three"
    assert sorted(os.listdir(tmp_path)) == ["f.json"]  # no temporary file left behind


def test_atomic_overwrite_replaces_a_symlink_instead_of_following_it(tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"keep")
    link = tmp_path / "d" / "f.json"
    link.parent.mkdir()
    os.symlink(outside, link)
    write_atomic(str(link), b"new", overwrite=True)
    assert not os.path.islink(link) and link.read_bytes() == b"new"
    assert outside.read_bytes() == b"keep"


def test_atomic_write_fsyncs_the_file_and_then_its_directory(tmp_path, monkeypatch):
    synced: list[str] = []
    real_fsync = os.fsync

    def fsync(fd):
        synced.append("dir" if os.path.isdir(f"/proc/self/fd/{fd}") else "file")
        return real_fsync(fd)

    monkeypatch.setattr(fileio.os, "fsync", fsync)
    write_atomic(tmp_path / "a.bin", b"x", overwrite=False)
    assert synced == ["file", "dir"]


def test_the_tifxyz_writer_publishes_meta_json_last_through_the_shared_writer(tmp_path, monkeypatch):
    import numpy as np

    from apok_eval.io import tifxyz

    written: list[str] = []
    real = fileio.write_atomic

    def spy(path, data, *, overwrite):
        written.append(os.path.basename(path))
        return real(path, data, overwrite=overwrite)

    monkeypatch.setattr(tifxyz, "write_atomic", spy)
    plane = np.ones((2, 2), "<f4")
    tifxyz.write_tifxyz(tmp_path / "s", x=plane, y=plane, z=plane, meta={"format": "tifxyz", "type": "seg",
                                                                          "uuid": "s", "scale": [1.0, 1.0]})
    assert written == ["x.tif", "y.tif", "z.tif", "meta.json"]  # meta.json last


def test_entry_kind_names_what_is_there_and_never_guesses_absent(tmp_path):
    from apok_eval.fileio import entry_kind

    (tmp_path / "f").write_bytes(b"x")
    (tmp_path / "d").mkdir()
    os.symlink(tmp_path / "f", tmp_path / "link")
    os.symlink(tmp_path / "gone", tmp_path / "dangling")
    os.mkfifo(tmp_path / "fifo")
    assert [entry_kind(tmp_path / n) for n in ("f", "d", "link", "dangling", "fifo", "none", "f/below")] == \
        ["file", "directory", "symlink", "symlink", "other", None, None]
    assert entry_kind(tmp_path / "link", follow_symlinks=True) == "file"
    assert entry_kind(tmp_path / "dangling", follow_symlinks=True) is None
    with pytest.raises(OSError) as info:
        entry_kind(tmp_path / ("x" * 300))  # ENAMETOOLONG is an error, not "nothing there"
    assert info.value.errno == 36
    with pytest.raises(ValueError):
        entry_kind(tmp_path / "nul\x00")
