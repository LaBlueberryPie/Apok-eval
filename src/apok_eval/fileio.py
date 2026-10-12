"""File reading and writing shared by every module.

* **Regular files only.** :func:`read_regular`, :func:`digest_and_size` and
  :func:`open_regular` open a path without blocking (``O_NONBLOCK``) and
  refuse, with :class:`NotRegularFile`, anything that is not a regular file
  once open: a FIFO would block a reader forever and a device such as
  ``/dev/zero`` never ends. The type is checked on the open descriptor, so
  the file read is the file checked. :class:`NotRegularFile` says which
  condition was found (``missing``: nothing there; otherwise ``kind``: a
  directory, FIFO, character or block device, or socket), so a caller's
  refusal can name it. Any other failure (``EACCES``, ``EIO``,
  ``ENAMETOOLONG``) is the ``OSError`` the system raised; a path holding NUL
  is the ``ValueError`` Python raises.
* **Looking without guessing.** :func:`entry_kind` says what is at a path
  (``"file"``, ``"directory"``, ``"symlink"``, ``"other"``) or ``None`` when
  nothing is (``ENOENT``/``ENOTDIR``). Unlike ``os.path.isfile`` and its
  kin, which answer ``False`` for any error, it lets every other error
  propagate: an unsearchable directory or a name too long is not "absent".
* **Read once.** :func:`read_regular` returns the bytes; a caller that both
  digests and parses a file digests those bytes (:func:`apok_eval.canonical.sha256_bytes`)
  and parses the same bytes, so the digest names the bytes judged.
* **Names for people.** :func:`portable_name` writes a file name holding bytes
  that are not UTF-8 with those bytes as ``\\xNN``, so it can go into JSON.
* **Atomic publication.** :func:`write_atomic` writes a temporary file
  (``.apok-tmp-<hex>``, created exclusively with mode 0o666 so the umask
  applies) in the target directory and fsyncs it, then publishes it with
  ``os.replace`` (overwrite allowed; a symbolic link at the target is
  replaced, never followed) or ``os.link`` (no-clobber: ``os.rename`` would
  silently replace a file that appeared meanwhile; :class:`TargetExists`),
  and fsyncs the directory, so files published in order reach the disk in
  order. Only the target and its temporary file are touched.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import stat
import uuid
from pathlib import Path
from typing import BinaryIO

from apok_eval.canonical import DIGEST_PREFIX

TMP_PREFIX = ".apok-tmp-"
_CHUNK = 1 << 20
_KINDS = ((stat.S_ISDIR, "a directory"), (stat.S_ISFIFO, "a FIFO"), (stat.S_ISCHR, "a character device"),
          (stat.S_ISBLK, "a block device"), (stat.S_ISSOCK, "a socket"))


class TargetExists(FileExistsError):
    """A no-clobber publication found the target already present."""


class NotRegularFile(OSError):
    """No regular file at a path: nothing there (``missing``), or a directory, FIFO, device or socket (``kind``)."""

    def __init__(self, path: str | os.PathLike[str], problem: str, *, kind: str | None = None) -> None:
        super().__init__(f"{os.fspath(path)}: {problem}")
        self.filename = os.fspath(path)
        self.problem = problem
        self.kind = kind
        self.missing = kind is None

    def __str__(self) -> str:  # OSError would print "[Errno None] None: <filename>"
        return str(self.args[0])


def portable_name(name: str) -> str:
    """``name`` unchanged when it is valid Unicode; else its undecodable bytes written as ``\\xNN``.

    ``os.scandir``, ``os.readlink`` and ``sys.argv`` decode bytes that are not UTF-8 to lone surrogates
    (``surrogateescape``), which no UTF-8 (and so no canonical or strict JSON) text can hold. The written
    form is for reading by people: ``\\xff`` is also what a name holding those four characters looks like.
    """
    try:
        name.encode("utf-8")
    except UnicodeEncodeError:
        return os.fsencode(name).decode("utf-8", "backslashreplace")
    return name


def file_kind(mode: int) -> str:
    """A reader's name for the file type in ``mode`` ("a regular file", "a FIFO", ...)."""
    if stat.S_ISREG(mode):
        return "a regular file"
    return next((name for test, name in _KINDS if test(mode)), f"an unknown file type ({stat.filemode(mode)})")


def entry_kind(path: str | os.PathLike[str], *, follow_symlinks: bool = False) -> str | None:
    """What is at ``path`` (see the module docstring); with ``follow_symlinks`` a dangling link is ``None`` and a
    link is reported as what it leads to. ``OSError`` other than ``ENOENT``/``ENOTDIR`` (``EACCES``,
    ``ENAMETOOLONG``, ``ELOOP``) and the ``ValueError`` of a path holding NUL propagate."""
    try:
        mode = (os.stat(path) if follow_symlinks else os.lstat(path)).st_mode
    except (FileNotFoundError, NotADirectoryError):
        return None
    if stat.S_ISREG(mode):
        return "file"
    if stat.S_ISDIR(mode):
        return "directory"
    return "symlink" if stat.S_ISLNK(mode) else "other"


def open_regular(path: str | os.PathLike[str]) -> BinaryIO:
    """``path`` opened for reading if it is a regular file; :class:`NotRegularFile` otherwise."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise NotRegularFile(path, "no such file") from exc
    try:
        mode = os.fstat(fd).st_mode
        if not stat.S_ISREG(mode):
            kind = file_kind(mode)
            raise NotRegularFile(path, f"not a regular file ({kind}, mode {stat.filemode(mode)})", kind=kind)
        return os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise


def read_regular(path: str | os.PathLike[str]) -> bytes:
    """All bytes of the regular file ``path`` (see :func:`open_regular`)."""
    with open_regular(path) as fh:
        return fh.read()


def digest_and_size(path: str | os.PathLike[str]) -> tuple[str, int]:
    """``sha256:`` digest and byte count of the regular file ``path``, from one pass over its bytes."""
    h = hashlib.sha256()
    size = 0
    with open_regular(path) as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            h.update(block)
            size += len(block)
    return DIGEST_PREFIX + h.hexdigest(), size


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_temp(directory: Path, data: bytes) -> Path:
    tmp = directory / f"{TMP_PREFIX}{uuid.uuid4().hex}"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    return tmp


def write_atomic(path: str | os.PathLike[str], data: bytes, *, overwrite: bool) -> None:
    """Publish ``data`` at ``path`` atomically (see the module docstring).

    Raises :class:`TargetExists` when ``overwrite`` is false and ``path``
    exists, ``OSError`` for any other failure.
    """
    path = Path(os.path.abspath(path))
    tmp = _write_temp(path.parent, data)
    try:
        if overwrite:
            os.replace(tmp, path)
        else:
            try:
                os.link(tmp, path)
            except FileExistsError as exc:
                raise TargetExists(str(path)) from exc
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
    _fsync_dir(path.parent)


def probe_writable(directory: str | os.PathLike[str]) -> None:
    """Write, fsync and remove a small file in ``directory``; raises ``OSError`` if that fails."""
    os.unlink(_write_temp(Path(directory), b"apok write probe\n"))
