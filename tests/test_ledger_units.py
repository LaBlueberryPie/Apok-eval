"""Unit tests for the ledger's building blocks: JSON pointers, names and
paths, atomic writes, the chunk-manifest sidecar format, the git
helpers and the environment block."""

from __future__ import annotations

import json
import os
import subprocess
import time

import pytest

from apok_eval.canonical import canonical_dumps, sha256_bytes
from apok_eval.ledger import gitcode, manifest
from apok_eval.ledger.environment import capture_environment
from apok_eval.ledger import open_run
from apok_eval.ledger.util import (
    NotRegularFile,
    TargetExists,
    digest_and_size,
    is_name,
    is_safe_relpath,
    is_text,
    read_regular,
    resolve_pointer,
    write_atomic,
)
from apok_eval.merkle import manifest_root

from ledger_fixtures import chunk_entries, deadline, git, make_source_repo, no_git_above


# --- regular files ------------------------------------------------------------------------------------


def test_reading_refuses_what_is_not_a_regular_file(tmp_path):
    os.mkfifo(tmp_path / "fifo")
    (tmp_path / "dir").mkdir()
    (tmp_path / "file").write_bytes(b"bytes")
    for path in (tmp_path / "fifo", tmp_path / "dir", tmp_path / "missing", tmp_path / "file" / "deeper",
                 "/dev/zero"):
        for read in (read_regular, digest_and_size):
            with deadline(), pytest.raises(NotRegularFile) as info:
                read(path)
            assert info.value.filename == str(path) and str(path) in str(info.value)
    assert read_regular(tmp_path / "file") == b"bytes"
    (tmp_path / "link").symlink_to(tmp_path / "file")
    assert read_regular(tmp_path / "link") == b"bytes"  # callers decide about symlinks (resolve_in_run_dir)


# --- JSON pointers ---------------------------------------------------------------------------------

DOC = {"a": [10, {"b": 2}], "x/y": {"~k": 3}, "": 4}


@pytest.mark.parametrize("pointer, value", [("", DOC), ("/a/0", 10), ("/a/1/b", 2), ("/x~1y/~0k", 3), ("/", 4)])
def test_pointer_resolution(pointer, value):
    assert resolve_pointer(DOC, pointer) == value


@pytest.mark.parametrize("pointer", ["/missing", "/a/2", "/a/01", "/a/-", "/a/0/deeper", "/a/b"])
def test_pointers_that_do_not_resolve(pointer):
    with pytest.raises(LookupError):
        resolve_pointer(DOC, pointer)


@pytest.mark.parametrize("pointer", ["a", "/x~2", "/~", 5, None])
def test_malformed_pointers(pointer):
    with pytest.raises(ValueError):
        resolve_pointer(DOC, pointer)


# --- names, text and paths ------------------------------------------------------------------------


@pytest.mark.parametrize("value, ok", [("a", True), ("fold-a.v2_1", True), ("0", True), ("a" * 128, True),
                                       ("a" * 129, False), ("", False), ("A", False), (".a", False),
                                       ("a\n", False), ("a b", False), ("a/b", False), (1, False)])
def test_names(value, ok):
    assert is_name(value) is ok


@pytest.mark.parametrize("value, ok", [("x", True), (" x ", True), ("", False), ("  ", False), ("\ud800", False),
                                       (None, False)])
def test_text(value, ok):
    assert is_text(value) is ok


@pytest.mark.parametrize("value, ok", [("a", True), ("a/b.c", True), ("", False), ("/a", False), ("a//b", False),
                                       ("a/./b", False), ("a/../b", False), ("..", False), ("a\\b", False),
                                       ("a\x00", False), ("a/\udcff", False), ("a/", False), (3, False)])
def test_safe_relative_paths(value, ok):
    assert is_safe_relpath(value) is ok


def test_digest_and_size(tmp_path):
    path = tmp_path / "f"
    path.write_bytes(b"abc" * 1000)
    assert digest_and_size(path) == (sha256_bytes(b"abc" * 1000), 3000)


def test_write_atomic(tmp_path):
    target = tmp_path / "t.json"
    write_atomic(target, b"one", overwrite=False)
    with pytest.raises(TargetExists):
        write_atomic(target, b"two", overwrite=False)
    assert target.read_bytes() == b"one"
    write_atomic(target, b"three", overwrite=True)
    assert target.read_bytes() == b"three"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["t.json"]
    assert target.stat().st_mode & 0o777 == 0o666 & ~_umask()


def _umask() -> int:
    import os

    mask = os.umask(0)
    os.umask(mask)
    return mask


# --- the chunk-manifest sidecar ---------------------------------------------------------------------


def test_sidecar_round_trip_and_root():
    entries = manifest.sorted_entries(chunk_entries())
    data = manifest.to_jsonl(entries)
    assert manifest.parse_jsonl(data) == entries
    assert manifest_root(entries) == manifest_root(chunk_entries())
    assert manifest.counts(entries) == (6, 2)
    assert data.count(b"\n") == 8 and data.endswith(b"\n")
    assert [e["key"] for e in entries] == sorted((e["key"] for e in entries), key=str.encode)


def test_counts_count_every_absent_entry():
    entries = [{"key": "a", "absent": True}, {"key": "b", "digest": sha256_bytes(b""), "size": 0},
               {"key": "c", "absent": True}]
    assert manifest.counts(entries) == (1, 2)


def test_an_empty_sidecar_is_an_empty_manifest():
    assert manifest.parse_jsonl(b"") == [] and manifest.to_jsonl([]) == b""


@pytest.mark.parametrize("data, fragment", [
    (b'{"absent":true,"key":"a"}', "newline"),
    (b'{"absent":true,"key":"b"}\n{"absent":true,"key":"a"}\n', "line 2"),
    (b'{"absent":true,"key":"a"}\n{"absent":true,"key":"a"}\n', "line 2"),
    (b'{"key":"a","absent":true}\n', "canonical"),
    (b'{"absent":true,"key":"a"}\n\n', "line 2"),
    (b'{"absent":1,"key":"a"}\n', "absent chunk"),
    (b'[1]\n', "not an object"),
    (b'{"absent":true,"key":"a","key":"b"}\n', "duplicate"),
])
def test_bad_sidecars_name_the_line(data, fragment):
    with pytest.raises(ValueError, match=fragment):
        manifest.parse_jsonl(data)


def test_entry_problem_accepts_only_the_two_shapes():
    assert manifest.entry_problem({"key": "a", "absent": True}) is None
    assert manifest.entry_problem({"key": "a", "digest": sha256_bytes(b""), "size": 0}) is None
    assert manifest.entry_problem({"key": "a", "digest": sha256_bytes(b""), "size": 0, "absent": True})
    assert manifest.entry_problem({"key": 3, "absent": True})


_D = sha256_bytes(b"")


@pytest.mark.parametrize("entry", [
    {"key": "a", "absent": True}, {"key": "a", "digest": _D, "size": 0},
    {"key": "a", "absent": True, "x": 1}, {"key": "a", "absent": 1}, {"key": "a", "absent": False},
    {"key": "a", "digest": _D, "size": True}, {"key": "a", "digest": _D, "size": -1},
    {"key": "a", "digest": _D, "size": 1.0}, {"key": "a", "digest": _D}, {"key": "a", "digest": _D, "size": 0, "x": 1},
    {"key": "a", "digest": "sha256:" + "0" * 63, "size": 0}, {"key": "a", "digest": "md5:00", "size": 0},
    {"key": "", "absent": True}, {"key": "a#b", "absent": True}, {"key": 3, "absent": True},
])
def test_entry_problem_states_the_rules_merkle_enforces(entry):
    """docs/ledger.md says the sidecar's entry rules are apok_eval.merkle's; the two must not drift."""
    try:
        manifest_root([entry])
        merkle_accepts = True
    except ValueError:
        merkle_accepts = False
    assert (manifest.entry_problem(entry) is None) == merkle_accepts


def test_duplicate_keys_are_refused_by_both():
    entries = [{"key": "a", "absent": True}, {"key": "a", "digest": _D, "size": 0}]
    with pytest.raises(ValueError, match="chunk 'a' appears more than once"):
        manifest_root(entries)
    with pytest.raises(ValueError, match="more than once"):
        manifest.sorted_entries(entries)


# --- git helpers ------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    return make_source_repo(tmp_path_factory.mktemp("units-source"))


def test_commit_manifest_matches_the_working_tree_at_head(source):
    expected = gitcode.commit_manifest(source.repo, source.commits[1], "src/apok")
    assert expected == gitcode.source_manifest(source.source_dir)
    old = {e["path"]: e["digest"] for e in gitcode.commit_manifest(source.repo, source.commits[0], "src/apok")}
    assert old["core.py"] == sha256_bytes(b"ANSWER = 41\n")


def test_commit_manifest_of_the_whole_tree_and_of_a_prefix_that_is_not_a_directory(source):
    whole = {e["path"] for e in gitcode.commit_manifest(source.repo, source.commits[1], ".")}
    assert {"README", ".gitignore", "src/apok/core.py"} <= whole
    assert gitcode.commit_manifest(source.repo, source.commits[1], "src/apok/core.py") == []
    assert gitcode.commit_manifest(source.repo, source.commits[1], "src/ap") == []


def test_commit_manifest_marks_submodules(tmp_path):
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "a.py").write_text("a")
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    oid = "1" * 40
    git(repo, "update-index", "--add", "--cacheinfo", f"160000,{oid},pkg/vendored")
    git(repo, "commit", "-q", "-m", "with a gitlink")
    manifest_ = gitcode.commit_manifest(repo, git(repo, "rev-parse", "HEAD"), "pkg")
    assert {"path": "vendored", "digest": f"commit:{oid}"} in manifest_


def test_commit_manifest_skips_committed_caches(tmp_path):
    repo = tmp_path / "repo"
    (repo / "pkg" / "__pycache__").mkdir(parents=True)
    (repo / "pkg" / "a.py").write_text("a")
    (repo / "pkg" / "__pycache__" / "a.pyc").write_bytes(b"c")
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "caches committed by mistake")
    assert [e["path"] for e in gitcode.commit_manifest(repo, git(repo, "rev-parse", "HEAD"), "pkg")] == ["a.py"]


def test_commit_available(source, tmp_path):
    assert gitcode.commit_available(source.repo, source.commits[0])
    assert not gitcode.commit_available(source.repo, "2" * 40)
    with pytest.raises(gitcode.GitError):
        gitcode.commit_available(tmp_path / "missing", source.commits[0])


def test_toplevel(source, tmp_path, monkeypatch):
    assert gitcode.toplevel(source.source_dir / "sub").resolve() == source.repo.resolve()
    assert no_git_above(tmp_path), "this test needs a temporary directory outside every git checkout"
    assert gitcode.toplevel(tmp_path) is None  # no repository, and no .git entry anywhere above
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(source.source_dir))  # ignored: never hides a checkout
    assert gitcode.toplevel(source.source_dir / "sub").resolve() == source.repo.resolve()
    with pytest.raises(gitcode.GitError):
        gitcode.toplevel(tmp_path / "missing")
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / ".git").write_text("gitdir: /nonexistent/worktree\n")
    with pytest.raises(gitcode.GitError, match="exists"):
        gitcode.toplevel(broken)


def test_git_ignores_a_git_dir_from_the_environment(source, tmp_path, monkeypatch):
    other = tmp_path / "other"
    other.mkdir()
    git(other, "init", "-q")
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    state = gitcode.capture_code(source.source_dir)
    assert state.commit == source.commits[1]


def test_status_does_not_touch_the_index_file(source):
    index_file = source.repo / ".git" / "index"
    before = (index_file.stat().st_mtime_ns, index_file.read_bytes())
    (source.source_dir / "core.py").touch()  # stale stat data that `git status` would normally refresh
    gitcode.capture_code(source.source_dir)
    assert (index_file.stat().st_mtime_ns, index_file.read_bytes()) == before


def test_a_dirty_run_never_touches_the_index_file_and_misses_no_racy_edit(tmp_path):
    """``git diff`` rewrites the index it reads to refresh stale stat data; the ledger diffs a copy.

    The copy must keep the index's mtime: git re-reads an entry whose mtime is not older than the index
    file ("racily clean"). Here the edit keeps size and mtime, so only that rule reveals it.
    """
    source = make_source_repo(tmp_path / "own")
    repo, core, mod = source.repo, source.source_dir / "core.py", source.source_dir / "sub" / "mod.py"
    git(repo, "config", "core.checkStat", "minimal")  # compare mtime seconds and size only: deterministic
    past = time.time_ns() - 200 * 10**9
    os.utime(core, ns=(past, past))
    os.utime(mod, ns=(past - 50 * 10**9, past - 50 * 10**9))
    git(repo, "update-index", "--refresh")  # the index entries now hold these past mtimes
    index_file = repo / ".git" / "index"
    os.utime(index_file, ns=(past - 10**9, past - 10**9))
    # core.py's entry is not older than the index file: racily clean, so git compares its content.
    core.write_text("ANSWER = 43\n")  # same size as "ANSWER = 42\n", same mtime: only that rule sees it
    os.utime(core, ns=(past, past))
    # mod.py's entry is older than the index file: its new mtime is stale stat data that `git diff` refreshes
    # by rewriting the index it reads.
    mod.touch()
    before = (index_file.stat().st_mtime_ns, index_file.read_bytes())
    run = open_run(tmp_path / "run", run_id="r", purpose="p", config={}, expected_bytes=0, index=None,
                   source_dir=source.source_dir, allow_dirty=True)
    (run.run_dir / "out.txt").write_text("o")
    run.add_output("out", "out.txt")
    run.seal()
    assert (index_file.stat().st_mtime_ns, index_file.read_bytes()) == before
    assert b"+ANSWER = 43" in (run.run_dir / "code.diff").read_bytes()


def test_applied_manifest_of_the_whole_tree_and_with_a_symlink(tmp_path):
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "a.py").write_text("A = 1\n")
    (repo / "top.txt").write_text("t\n")
    (repo / "pkg" / "link").symlink_to("a.py")
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "base")
    head = git(repo, "rev-parse", "HEAD")
    (repo / "pkg" / "a.py").write_text("A = 2\n")
    (repo / "new.txt").write_text("n\n")
    status = gitcode._status(repo)
    whole = gitcode.working_diff(repo, ".", status)
    assert b"b/new.txt" in whole and b"+A = 2" in whole  # source_root "." takes every untracked file
    assert b"b/new.txt" not in gitcode.working_diff(repo, "pkg", status)
    applied = {e["path"]: e["digest"] for e in gitcode.applied_manifest(repo, head, ".", whole)}
    assert applied["pkg/a.py"] == sha256_bytes(b"A = 2\n") and applied["new.txt"] == sha256_bytes(b"n\n")
    marker = "symlink:" + sha256_bytes(b"a.py")
    assert applied["pkg/link"] == marker
    assert {e["path"]: e["digest"] for e in gitcode.commit_manifest(repo, head, "pkg")}["link"] == marker
    assert gitcode.applied_manifest(repo, head, "pkg", b"") == gitcode.commit_manifest(repo, head, "pkg")


def test_a_commit_holding_an_unsafe_path_is_never_written(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    blob = subprocess.run(["git", "-C", str(repo), "hash-object", "-w", "--stdin"], input=b"x", capture_output=True,
                          check=True).stdout.decode().strip()
    inner = subprocess.run(["git", "-C", str(repo), "mktree"], input=f"100644 blob {blob}\t..\n".encode(),
                           capture_output=True, check=True).stdout.decode().strip()
    tree = subprocess.run(["git", "-C", str(repo), "mktree"], input=f"040000 tree {inner}\tpkg\n".encode(),
                          capture_output=True, check=True).stdout.decode().strip()
    commit = git(repo, "commit-tree", tree, "-m", "a crafted tree with a '..' entry")
    with pytest.raises(gitcode.GitError, match="cannot be written safely"):
        gitcode.applied_manifest(repo, commit, "pkg", b"")
    assert not (tmp_path / "x").exists()


def test_compare_manifests():
    a = [{"path": "x", "digest": "d1"}, {"path": "y", "digest": "d2"}]
    b = [{"path": "y", "digest": "d3"}, {"path": "z", "digest": "d4"}]
    assert gitcode.compare_manifests(a, a) == {}
    assert gitcode.compare_manifests(a, b) == {"missing_from_commit": ["x"], "missing_from_record": ["z"],
                                               "content_differs": ["y"]}


def test_blob_reader_handles_binary_and_newlines(tmp_path):
    repo = tmp_path / "repo"
    (repo / "p").mkdir(parents=True)
    payloads = {"a.bin": b"\n\n\x00binary\n", "b.txt": b"", "c.txt": b"tail without newline"}
    for name, data in payloads.items():
        (repo / "p" / name).write_bytes(data)
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "blobs")
    found = gitcode.commit_manifest(repo, git(repo, "rev-parse", "HEAD"), "p")
    assert found == [{"path": k, "digest": sha256_bytes(v)} for k, v in sorted(payloads.items())]


# --- environment -----------------------------------------------------------------------------------


def test_environment_is_canonical_and_explicit_about_missing_packages(monkeypatch):
    import apok_eval.ledger.environment as environment

    monkeypatch.setattr(environment, "_version", lambda name: None if name in ("tifffile", "imagecodecs") else "1.0")
    env = capture_environment()
    canonical_dumps(env)
    assert env["packages"]["tifffile"] is None and "imagecodecs" not in env["packages"]
    assert env["packages"]["zarr"] == "1.0"


def test_environment_records_every_declared_dependency_and_the_lzw_codec(monkeypatch):
    """pyproject's dependencies are recorded even when absent (null); imagecodecs, which decides whether an
    LZW tifxyz decodes or is refused, is recorded when installed."""
    import re
    import tomllib
    from pathlib import Path

    import apok_eval.ledger.environment as environment

    pyproject = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    declared = {re.match(r"[A-Za-z0-9_.-]+", spec).group(0).lower() for spec in pyproject["project"]["dependencies"]}
    assert declared == set(environment.REQUIRED_PACKAGES)
    assert "imagecodecs" in pyproject["project"]["optional-dependencies"]["lzw"]
    monkeypatch.setattr(environment, "_version", lambda name: None)
    assert capture_environment()["packages"] == {"apok-eval": environment.__version__, "numpy": None, "tifffile": None,
                                                 "numcodecs": None, "scipy": None}
    monkeypatch.setattr(environment, "_version", lambda name: "2.0" if name == "imagecodecs" else None)
    assert capture_environment()["packages"]["imagecodecs"] == "2.0"


def test_environment_versions_come_from_distribution_metadata():
    import importlib.metadata

    env = capture_environment()
    assert env["packages"]["numpy"] == importlib.metadata.version("numpy")
    assert json.loads(canonical_dumps(env)) == env


def test_git_is_installed_for_these_tests():
    assert subprocess.run(["git", "--version"], capture_output=True).returncode == 0
