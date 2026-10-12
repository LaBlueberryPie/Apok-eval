"""Code identity of a run: commit, working-tree state, diff and source manifest.

* **Source manifest**: one ``{"path", "digest"}`` entry for every file under
  the package source directory (``src/apok_eval`` for a checkout), ``path``
  relative to that directory in POSIX form, ``digest`` the sha256 of the
  file's bytes; entries sorted by the UTF-8 bytes of ``path``. Directories
  named ``__pycache__`` are skipped (interpreter caches, never committed);
  every other file counts, including untracked and ignored ones. A symlink, a
  non-regular file, a file name that is not a safe UTF-8 relative path (a
  backslash, an undecodable byte) or a file that cannot be read anywhere in
  the tree is refused (``LEDGER_SOURCE_UNSUPPORTED``): its digest could not
  be compared with git, or its name could not be recorded.
  ``source_manifest_root`` is ``apok_eval.merkle.merkle_root`` over
  ``(path, digest)``.
* **Checkout or not.** The source directory is a *checkout of the package*
  when it lies inside a git work tree whose ``HEAD`` or index holds at least
  one file under it. Otherwise ``commit``, ``dirty``, ``diff`` and
  ``source_root`` are all ``None``, on positive evidence only: git finds no
  repository *and* no ``.git`` entry exists in the source directory or any
  parent; or the enclosing work tree tracks nothing under the source
  directory (a package installed into an ignored virtualenv inside some
  unrelated project: that project's ``HEAD`` says nothing about this code).
  A ``.git`` entry git cannot open (a moved or pruned worktree) is a
  :class:`GitError`, never "no checkout"; so is a tracked package whose
  branch has no commit yet.
* **Git state** of a checkout: ``commit`` is ``HEAD``; ``dirty`` is true
  when ``git status --porcelain --untracked-files=all`` lists anything for
  the whole work tree (ignored files do not count); ``source_root`` is the
  source directory relative to the top of the work tree.
* **Diff** of a dirty checkout: ``git diff HEAD`` of the whole work tree with
  :data:`DIFF_OPTIONS` (tracked files), followed by one new-file patch
  (``git diff --no-index`` against ``/dev/null``) for every untracked file
  under the source directory outside ``__pycache__``. Untracked files
  elsewhere make the tree dirty but are not in the diff.
* **Expected manifest**: the source manifest the git state implies, computed
  from git objects alone. For a clean checkout: the commit's files under
  ``source_root`` (``git ls-tree -r``, the sha256 of each blob's bytes). For
  a dirty one: those files written to a temporary directory and patched with
  the diff (``git apply`` outside any repository, limited to
  ``source_root``), then digested; :class:`PatchError` if the diff does not
  apply. A submodule appears with the digest ``"commit:<object id>"`` and a
  symlink with ``"symlink:<sha256 of its target>"``; neither equals a file
  digest.

Git runs with ``LC_ALL=C`` (its messages are matched), literal pathspecs and
``GIT_NO_REPLACE_OBJECTS`` (a commit's own objects, not ``git replace``
substitutes), and without the variables that point it at another
repository, index or object store or stop its discovery
(:data:`_DROPPED_ENV`). The source repository is an input and is
never written to: ``status`` runs with ``--no-optional-locks``; ``git diff``
refreshes stale stat data in the index it reads, so it reads a temporary
copy of the index; patches are applied in a temporary directory, without
the user's or the system's git configuration.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from apok_eval import __version__
from apok_eval.canonical import sha256_bytes
from apok_eval.ledger.util import digest_and_size, is_safe_relpath
from apok_eval.merkle import merkle_root
from apok_eval.refusal import Refusal

EXCLUDED_DIR = "__pycache__"
DIFF_OPTIONS = ("--no-ext-diff", "--no-textconv", "--no-color", "--binary", "--no-renames", "--src-prefix=a/",
                "--dst-prefix=b/")
_NOT_A_REPO = "not a git repository"
_DROPPED_ENV = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR", "GIT_OBJECT_DIRECTORY",
                "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_NAMESPACE", "GIT_CEILING_DIRECTORIES")
_WILDMATCH_SPECIAL = re.compile(r"([\\*?\[])")


class GitError(Exception):
    """git could not be run, or a git command failed."""


class GitUnavailable(GitError):
    """git itself could not be run (not installed, not executable): nothing is known about any repository."""


class PatchError(GitError):
    """A diff does not apply to the files of the commit it was recorded against."""


def _git_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in _DROPPED_ENV}
    env.update(LC_ALL="C", GIT_TERMINAL_PROMPT="0", GIT_LITERAL_PATHSPECS="1", GIT_NO_REPLACE_OBJECTS="1")
    env.update(extra or {})
    return env


def git_run(repo: str | os.PathLike[str], *args: str, stdin: bytes | None = None,
            env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["git", "--no-optional-locks", "-C", os.fspath(repo), *args], input=stdin,
                              capture_output=True, env=_git_env(env), check=False)
    except OSError as exc:
        raise GitUnavailable(f"git could not be run: {exc}") from exc


def _failed(args: tuple[str, ...], proc: subprocess.CompletedProcess) -> GitError:
    message = proc.stderr.decode("utf-8", "replace").strip()
    return GitError(f"`git {' '.join(args)}` exited {proc.returncode}: {message}")


def git(repo: str | os.PathLike[str], *args: str, stdin: bytes | None = None,
        env: dict[str, str] | None = None) -> bytes:
    proc = git_run(repo, *args, stdin=stdin, env=env)
    if proc.returncode != 0:
        raise _failed(args, proc)
    return proc.stdout


def _git_entry_at_or_above(path: str | os.PathLike[str]) -> Path | None:
    start = Path(os.path.realpath(path))
    for directory in (start, *start.parents):
        if os.path.lexists(directory / ".git"):
            return directory / ".git"
    return None


def toplevel(path: str | os.PathLike[str]) -> Path | None:
    """Top of the git work tree containing ``path``; ``None`` only when there is no repository at all."""
    try:
        out = git(path, "rev-parse", "--show-toplevel")
    except GitError as exc:
        if _NOT_A_REPO not in str(exc):
            raise
        entry = _git_entry_at_or_above(path)
        if entry is not None:
            raise GitError(f"{exc}; yet {entry} exists: a checkout git cannot open (moved or pruned?)") from exc
        return None
    return Path(out.decode("utf-8").rstrip("\n"))


def _walk(root: Path) -> Iterator[tuple[str, str, int]]:
    """``(relative POSIX path, full path, lstat mode)`` of every non-directory under ``root``, caches skipped."""
    for dirpath, dirnames, filenames in os.walk(root, onerror=_reraise):
        dirnames[:] = sorted(d for d in dirnames if d != EXCLUDED_DIR)
        for name in dirnames + sorted(filenames):
            full = os.path.join(dirpath, name)
            mode = os.lstat(full).st_mode
            if not stat.S_ISDIR(mode):  # os.walk lists a symlink to a directory among the directories
                yield Path(full).relative_to(root).as_posix(), full, mode


def _reraise(error: OSError) -> None:
    raise error


def _sorted_manifest(entries: list[dict]) -> list[dict]:
    return sorted(entries, key=lambda entry: entry["path"].encode("utf-8", "surrogateescape"))


def source_manifest(source_dir: Path) -> list[dict]:
    """The source manifest of ``source_dir`` (see the module docstring)."""
    entries = []
    try:
        for rel, full, mode in _walk(source_dir):
            if not stat.S_ISREG(mode) or not is_safe_relpath(rel):
                raise Refusal("LEDGER_SOURCE_UNSUPPORTED",
                              f"source tree {source_dir} contains {rel!r}: a symlink, a non-regular file or a "
                              "name that is not a safe UTF-8 relative path", source_dir=str(source_dir), path=rel)
            entries.append({"path": rel, "digest": digest_and_size(full)[0]})
    except OSError as exc:
        raise Refusal("LEDGER_SOURCE_UNSUPPORTED", f"source tree {source_dir} cannot be read: {exc}",
                      source_dir=str(source_dir), path=str(exc.filename))
    return _sorted_manifest(entries)


def source_root_digest(manifest: list[dict]) -> str:
    return merkle_root((entry["path"], entry["digest"]) for entry in manifest)


@dataclass(frozen=True)
class CodeState:
    """What :func:`capture_code` found; ``expected_manifest`` is set for a checkout."""

    package_version: str
    source_manifest: list[dict]
    commit: str | None = None
    dirty: bool | None = None
    diff: bytes | None = None
    source_root: str | None = None
    status: tuple[str, ...] = ()
    expected_manifest: list[dict] | None = None

    @property
    def source_manifest_root(self) -> str:
        return source_root_digest(self.source_manifest)


def _resolve_commit(repo: str | os.PathLike[str], rev: str) -> str | None:
    """The commit ``rev`` names in ``repo``, ``None`` if there is none; ``GitError`` when git itself fails.

    ``rev-parse --verify --quiet`` exits 1 for a missing object or an unborn
    branch and 128 for a fatal error (no repository, ...).
    """
    args = ("rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}")
    proc = git_run(repo, *args)
    if proc.returncode not in (0, 1):
        raise _failed(args, proc)
    return proc.stdout.decode("ascii").strip() if proc.returncode == 0 else None


def commit_available(repo: str | os.PathLike[str], commit: str) -> bool:
    """Whether ``repo`` has the commit object; ``GitError`` when git itself fails."""
    return _resolve_commit(repo, commit) is not None


def _tracks(top: Path, head: str | None, source_root: str) -> bool:
    """Whether the index or ``HEAD`` holds any file under ``source_root``."""
    if git(top, "ls-files", "-z", "--", source_root):
        return True
    return head is not None and bool(git(top, "ls-tree", "-r", "-z", "--name-only", head, "--", source_root))


def _status(top: Path) -> list[tuple[str, str]]:
    """``(XY, path)`` of each porcelain-v1 status entry (renames off: a rename is a deletion and an addition)."""
    raw = git(top, "status", "--porcelain=v1", "-z", "--no-renames", "--untracked-files=all")
    return [(item[:2].decode("ascii"), item[3:].decode("utf-8", "surrogateescape"))
            for item in raw.split(b"\0") if item]


def _under(path: str, source_root: str) -> bool:
    return source_root == "." or path.startswith(source_root + "/")


def working_diff(top: Path, source_root: str, status: list[tuple[str, str]]) -> bytes:
    """The diff of a dirty checkout (see the module docstring)."""
    try:
        with tempfile.TemporaryDirectory(prefix="apok-ledger-index-") as tmp:
            index = Path(tmp) / "index"
            real = top / git(top, "rev-parse", "--git-path", "index").decode("utf-8", "surrogateescape").rstrip("\n")
            if real.exists():
                # copy2 keeps the mtime: git re-reads entries not older than its index file ("racily
                # clean"), and a copy stamped now would hide a same-size edit made within that second.
                shutil.copy2(real, index)
            # core.splitIndex=false: a split index would write a new shared index into the repository.
            parts = [git(top, "-c", "core.splitIndex=false", "diff", *DIFF_OPTIONS, "HEAD",
                         env={"GIT_INDEX_FILE": str(index)})]
    except OSError as exc:
        raise GitError(f"cannot copy the index of {top}: {exc}") from exc
    for xy, path in status:
        if xy == "??" and _under(path, source_root) and EXCLUDED_DIR not in path.split("/"):
            args = ("diff", "--no-index", *DIFF_OPTIONS, "--", os.devnull, path)
            proc = git_run(top, *args)
            if proc.returncode != 1:  # 1: the file differs from /dev/null, as every file does
                raise _failed(args, proc)
            parts.append(proc.stdout)
    return b"".join(parts)


def _tree_items(repo: str | os.PathLike[str], commit: str, source_root: str) -> list[tuple[str, str, str, str]]:
    """``(path relative to source_root, mode, type, object id)`` of each entry of ``commit`` under it."""
    listing = git(repo, "ls-tree", "-r", "-z", "--full-tree", commit, "--", source_root)
    prefix = "" if source_root == "." else source_root + "/"
    items = []
    for record in listing.split(b"\0"):
        if not record:
            continue
        meta, _, raw_path = record.partition(b"\t")
        mode, kind, oid = meta.decode("ascii").split(" ")
        path = raw_path.decode("utf-8", "surrogateescape")
        if not path.startswith(prefix):
            continue
        rel = path[len(prefix):]
        if EXCLUDED_DIR not in rel.split("/"):
            items.append((rel, mode, kind, oid))
    return items


def _blob_bytes(repo: str | os.PathLike[str], oids: list[str]) -> dict[str, bytes]:
    if not oids:
        return {}
    out = git(repo, "cat-file", "--batch", stdin="".join(f"{oid}\n" for oid in oids).encode("ascii"))
    blobs: dict[str, bytes] = {}
    pos = 0
    for _ in oids:
        end = out.index(b"\n", pos)
        header = out[pos:end].decode("ascii").split(" ")
        if len(header) != 3 or header[1] != "blob":
            raise GitError(f"git cat-file --batch answered {' '.join(header)!r}")
        size = int(header[2])
        blobs[header[0]] = out[end + 1:end + 1 + size]
        pos = end + 1 + size + 1
    return blobs


def _symlink_marker(target: bytes) -> str:
    return "symlink:" + sha256_bytes(target)


def _item_digest(mode: str, kind: str, oid: str, blobs: dict[str, bytes]) -> str:
    if kind != "blob":
        return f"{kind}:{oid}"
    return _symlink_marker(blobs[oid]) if mode == "120000" else sha256_bytes(blobs[oid])


def commit_manifest(repo: str | os.PathLike[str], commit: str, source_root: str) -> list[dict]:
    """The expected manifest of a clean checkout of ``commit`` (see the module docstring)."""
    items = _tree_items(repo, commit, source_root)
    blobs = _blob_bytes(repo, sorted({oid for _, _, kind, oid in items if kind == "blob"}))
    return _sorted_manifest([{"path": rel, "digest": _item_digest(mode, kind, oid, blobs)}
                             for rel, mode, kind, oid in items])


def _materialize(repo: str | os.PathLike[str], items: list[tuple[str, str, str, str]], base: Path) -> dict[str, str]:
    """Write the blobs of ``items`` under ``base``; return the digests of the entries that are not blobs.

    Symlinks are created after every file, so no file is written through one.
    """
    blobs = _blob_bytes(repo, sorted({oid for _, _, kind, oid in items if kind == "blob"}))
    others: dict[str, str] = {}
    links: list[tuple[Path, bytes]] = []
    for rel, mode, kind, oid in items:
        if kind != "blob":
            others[rel] = f"{kind}:{oid}"
            continue
        if not is_safe_relpath(rel):
            raise GitError(f"the commit holds a path that cannot be written safely: {rel!r}")
        dest = base.joinpath(*rel.split("/"))
        dest.parent.mkdir(parents=True, exist_ok=True)
        if mode == "120000":
            links.append((dest, blobs[oid]))
        else:
            dest.write_bytes(blobs[oid])
    for dest, target in links:
        os.symlink(os.fsdecode(target), dest)
    return others


def _file_digest(full: str, mode: int) -> str:
    """Digest of a file ``git apply`` left behind: a regular file or a symlink, the only kinds it makes."""
    return digest_and_size(full)[0] if stat.S_ISREG(mode) else _symlink_marker(os.fsencode(os.readlink(full)))


def _wildmatch_literal(text: str) -> str:
    """``text`` as a git wildmatch pattern that matches only itself."""
    return _WILDMATCH_SPECIAL.sub(r"\\\1", text)


def applied_manifest(repo: str | os.PathLike[str], commit: str, source_root: str, diff: bytes) -> list[dict]:
    """The expected manifest of a dirty checkout: ``commit`` patched with ``diff`` (see the module docstring)."""
    items = _tree_items(repo, commit, source_root)
    try:
        with tempfile.TemporaryDirectory(prefix="apok-ledger-apply-") as tmp:
            work = Path(os.path.realpath(tmp)) / "tree"
            base = work.joinpath(*source_root.split("/"))
            base.mkdir(parents=True)
            others = _materialize(repo, items, base)
            if diff:
                limit = () if source_root == "." else (f"--include={_wildmatch_literal(source_root)}/*",)
                proc = git_run(work, "apply", *limit, "-", stdin=diff, env={
                    "GIT_CEILING_DIRECTORIES": str(work.parent), "GIT_CONFIG_NOSYSTEM": "1",
                    "GIT_CONFIG_GLOBAL": os.devnull})
                if proc.returncode != 0:
                    raise PatchError(f"git apply exited {proc.returncode}: "
                                     f"{proc.stderr.decode('utf-8', 'replace').strip()}")
            entries = [{"path": rel, "digest": _file_digest(full, mode)}
                       for rel, full, mode in (_walk(base) if base.is_dir() else ())]
    except OSError as exc:
        raise GitError(f"cannot rebuild the files of commit {commit} in a temporary directory: {exc}") from exc
    return _sorted_manifest(entries + [{"path": rel, "digest": digest} for rel, digest in others.items()])


def capture_code(source_dir: Path) -> CodeState:
    """Code identity of the package at ``source_dir`` (see the module docstring)."""
    manifest = source_manifest(source_dir)
    try:
        top = toplevel(source_dir)
        if top is None:
            return CodeState(__version__, manifest)
        source_root = Path(os.path.relpath(os.path.realpath(source_dir), os.path.realpath(top))).as_posix()
        head = _resolve_commit(top, "HEAD")
        if not _tracks(top, head, source_root):
            return CodeState(__version__, manifest)
        if head is None:
            raise GitError(f"the index of {top} tracks files under {source_root!r}, but HEAD has no commit yet")
        status = _status(top)
        diff = working_diff(top, source_root, status) if status else None
        expected = (commit_manifest(top, head, source_root) if diff is None
                    else applied_manifest(top, head, source_root, diff))
    except GitError as exc:
        raise Refusal("LEDGER_GIT_FAILED", f"cannot establish the git state of {source_dir}: {exc}",
                      source_dir=str(source_dir))
    return CodeState(__version__, manifest, head, bool(status), diff, source_root,
                     tuple(f"{xy} {path}" for xy, path in status), expected)


def compare_manifests(recorded: list[dict], expected: list[dict]) -> dict[str, list[str]]:
    """Paths that differ between two source manifests; empty when they agree."""
    have = {entry["path"]: entry["digest"] for entry in recorded}
    want = {entry["path"]: entry["digest"] for entry in expected}
    differences = {
        "missing_from_commit": sorted(set(have) - set(want)),
        "missing_from_record": sorted(set(want) - set(have)),
        "content_differs": sorted(p for p in set(have) & set(want) if have[p] != want[p]),
    }
    return {kind: paths for kind, paths in differences.items() if paths}
