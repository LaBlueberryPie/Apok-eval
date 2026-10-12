"""The writer: what a sealed record contains, and every refusal on the way to it."""

from __future__ import annotations

import errno
import getpass
import json
import os
import platform
import shutil
import socket
import subprocess
import tempfile

import pytest

import apok_eval.ledger.writer as writer_module
from apok_eval.canonical import canonical_dumps, digest_json, sha256_bytes, sha256_file
from apok_eval.ledger import open_run, verify
from apok_eval.ledger.record import compute_seal
from apok_eval.ledger.verify import NOT_CHECKED
from apok_eval.merkle import manifest_root
from apok_eval.refusal import Refusal

from ledger_fixtures import (
    CONFIG,
    FIGURES,
    build_run,
    chunk_entries,
    deadline,
    failures,
    git,
    make_source_repo,
    no_git_above,
    status_of,
)


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    return make_source_repo(tmp_path_factory.mktemp("writer-source"))


@pytest.fixture
def own_source(tmp_path):
    """A source repository the test may make dirty."""
    return make_source_repo(tmp_path / "own")


def opened(tmp_path, source, **overrides):
    run_dir = overrides.pop("run_dir", tmp_path / "run")
    kwargs = dict(run_id="r1", purpose="a test run", config=CONFIG, expected_bytes=0, index=None,
                  source_dir=source.source_dir)
    kwargs.update(overrides)
    return open_run(run_dir, **kwargs)


def refused(code: str, fn, *args, **kwargs) -> Refusal:
    with pytest.raises(Refusal) as info:
        fn(*args, **kwargs)
    assert info.value.code == code, (info.value.code, info.value.reason)
    return info.value


def minimal_seal(run) -> str:
    (run.run_dir / "out.txt").write_text("done\n")
    run.add_output("out", "out.txt")
    return run.seal()


# --- what a sealed record holds --------------------------------------------------------------


def test_sealed_record_content(tmp_path, source):
    run = build_run(tmp_path, source)
    raw = run.record_path.read_bytes()
    record = json.loads(raw)
    assert raw == canonical_dumps(record) + b"\n"
    assert record["seal"] == compute_seal(record) == run.seal
    assert record["config"] == CONFIG and record["config_digest"] == digest_json(CONFIG)
    assert record["seeds"] == {"eval": 11, "optimizer": 7}
    assert record["seed_pointers"] == {"eval": "/eval/seed", "optimizer": "/fit/optimizer_random_seed"}
    inputs = {e["name"]: e for e in record["inputs"]}
    assert [e["name"] for e in record["inputs"]] == sorted(inputs)
    for name in ("volume", "labels", "predictions"):
        path = run.files[name]
        assert inputs[name]["path"] == str(path)
        assert inputs[name]["digest"] == sha256_file(path) and inputs[name]["size"] == path.stat().st_size
    assert inputs["volume"]["uri"] == "file:///synthetic/volume.bin" and inputs["labels"]["uri"] is None
    assert inputs["remote-meta"]["path"] is None
    assert inputs["remote-meta"]["digest"] == sha256_bytes(b'{"multiscales":[]}')
    (model,) = record["models"]
    assert model["digest"] == sha256_file(run.files["model"]) and model["source"].startswith("synthetic")
    outputs = {e["name"]: e for e in record["outputs"]}
    assert outputs["mesh"]["path"] == "out/mesh.bin" and outputs["metrics"]["path"] == "metrics.json"
    assert record["figures"] == {"source": "metrics", "pointer": "/summary", "values": FIGURES}
    assert record["predictions"] == {"registry_input": "predictions",
                                     "registry_digest": inputs["predictions"]["digest"], "ids": ["p-001", "p-002"]}
    (chunks,) = record["chunk_manifests"]
    assert chunks["root"] == manifest_root(chunk_entries())
    assert (chunks["count_present"], chunks["count_absent"]) == (6, 2)
    assert chunks["manifest_file"] == "manifests/volume.jsonl" and chunks["scheme"] == "apok-merkle-v1"
    assert chunks["manifest_file_digest"] == sha256_file(run.files["sidecar"])


def test_code_block_of_a_clean_checkout(tmp_path, source):
    code = build_run(tmp_path, source).record()["code"]
    assert code["commit"] == source.commits[1] and code["dirty"] is False
    assert code["diff_digest"] is None and code["diff_file"] is None
    assert code["source_root"] == "src/apok" and code["package_version"]
    paths = [e["path"] for e in code["source_manifest"]]
    assert paths == ["__init__.py", "core.py", "data/table.json", "sub/__init__.py", "sub/mod.py"]
    assert dict((e["path"], e["digest"]) for e in code["source_manifest"])["core.py"] == \
        sha256_bytes(b"ANSWER = 42\n")


def test_environment_records_versions_and_nothing_private(tmp_path, source, monkeypatch):
    real = {platform.node(), socket.gethostname(), getpass.getuser()}
    # Distinctive stand-ins, so that a host or user name embedded anywhere in a value is found as a substring.
    host, user = "apok-private-host-7f3c", "apok-private-user-9d2e"
    monkeypatch.setattr(platform, "node", lambda: host)
    monkeypatch.setattr(socket, "gethostname", lambda: host)
    for variable in ("LOGNAME", "USER", "LNAME", "USERNAME"):
        monkeypatch.setenv(variable, user)
    assert getpass.getuser() == user
    env = build_run(tmp_path, source).record()["environment"]
    assert set(env) == {"python", "implementation", "platform", "machine", "packages"}
    assert env["python"] == platform.python_version() and env["machine"] == platform.machine()
    assert {"apok-eval", "numpy", "tifffile", "numcodecs", "scipy"} <= set(env["packages"])
    assert set(env["packages"]) <= {"apok-eval", "numpy", "tifffile", "numcodecs", "scipy", "imagecodecs", "zarr"}
    assert env["packages"]["numpy"]
    text = canonical_dumps(env).decode()
    assert host not in text and user not in text
    # The real names too, where they are long enough not to occur by chance (a host called "vm" would).
    assert not [name for name in real if len(name) >= 6 and name in text]


def test_created_utc_is_a_utc_timestamp(tmp_path, source):
    created = build_run(tmp_path, source).record()["created_utc"]
    assert len(created) == 20 and created.endswith("Z") and created[10] == "T"


def test_the_writer_never_accepts_a_digest(tmp_path, source):
    run = opened(tmp_path, source)
    data = tmp_path / "in.bin"
    data.write_bytes(b"real bytes")
    claimed = sha256_bytes(b"what the caller claims")
    with pytest.raises(TypeError):
        run.add_input("a", data, digest=claimed)
    with pytest.raises(TypeError):
        run.add_model("m", data, source="s", digest=claimed)
    with pytest.raises(TypeError):
        run.add_remote_input("r", b"x", uri="https://example.invalid/x", digest=claimed)
    (run.run_dir / "o.txt").write_text("o")
    with pytest.raises(TypeError):
        run.add_output("o", "o.txt", digest=claimed)
    assert run.add_input("a", data)["digest"] == sha256_bytes(b"real bytes")


def test_a_run_outside_any_checkout_records_no_commit(tmp_path):
    assert no_git_above(tmp_path), "this test needs a temporary directory outside every git checkout"
    src = tmp_path / "plain" / "apok"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("")
    run = build_run(tmp_path, src)
    code = run.record()["code"]
    assert (code["commit"], code["dirty"], code["source_root"], code["diff_digest"]) == (None, None, None, None)
    report = verify(run.record_path)
    assert "names no commit" in status_of(report, "code.commit")["reason"]
    assert status_of(report, "code.diff")["status"] == NOT_CHECKED


def _ignored_package(tmp_path):
    """A package installed into an ignored virtualenv inside an unrelated project checkout."""
    project = tmp_path / "project"
    package = project / ".venv" / "lib" / "site-packages" / "apok"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('__version__ = "0.1.0"\n')
    (package / "core.py").write_text("X = 1\n")
    (project / ".gitignore").write_text(".venv/\n")
    (project / "notes.txt").write_text("an unrelated project\n")
    git(project, "init", "-q")
    git(project, "add", "-A")
    git(project, "commit", "-q", "-m", "unrelated")
    return project, package


@pytest.mark.parametrize("project_tree", ["clean", "dirty"])
def test_a_package_the_enclosing_checkout_does_not_track_records_no_commit(tmp_path, project_tree):
    project, package = _ignored_package(tmp_path)
    if project_tree == "dirty":
        (project / "notes.txt").write_text("edited\n")
    run = build_run(tmp_path, package)  # neither LEDGER_CODE_MISMATCH (clean) nor the project's HEAD (dirty)
    code = run.record()["code"]
    assert (code["commit"], code["dirty"], code["source_root"], code["diff_digest"]) == (None, None, None, None)
    assert [e["path"] for e in code["source_manifest"]] == ["__init__.py", "core.py"]
    assert status_of(verify(run.record_path, repo=project), "code.commit")["status"] == NOT_CHECKED


def test_an_untracked_package_in_a_checkout_records_no_commit(tmp_path):
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "a.py").write_text("A = 1\n")
    (repo / "README").write_text("r\n")
    git(repo, "init", "-q")
    git(repo, "add", "README")
    git(repo, "commit", "-q", "-m", "without the package")
    run = build_run(tmp_path, repo / "pkg")
    assert run.record()["code"]["commit"] is None


# --- opening a run -------------------------------------------------------------------------------


@pytest.mark.parametrize("run_id", ["", "Upper", "-lead", "a" * 129, "abc\n", "a b", None, 5, "a/b"])
def test_run_id_is_validated(tmp_path, source, run_id):
    refusal = refused("LEDGER_RUN_ID_INVALID", opened, tmp_path, source, run_id=run_id)
    assert refusal.details["run_id"] == run_id
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("field, value", [
    ("purpose", ""), ("purpose", "   "), ("purpose", None), ("purpose", "\ud800"),
    ("config", [1]), ("config", {"a": (1, 2)}), ("config", {"a": float("nan")}), ("config", {1: "x"}),
    ("config", "{}"),
    ("expected_bytes", -1), ("expected_bytes", True), ("expected_bytes", 1.5), ("expected_bytes", None),
])
def test_open_arguments_are_validated(tmp_path, source, field, value):
    refusal = refused("LEDGER_ARGUMENT_INVALID", opened, tmp_path, source, **{field: value})
    assert refusal.details["field"] == field


@pytest.mark.parametrize("depth", [300, 5000])
def test_a_config_nested_too_deeply_is_refused(tmp_path, source, depth):
    config: dict = {}
    for _ in range(depth):
        config = {"a": config}
    refusal = refused("LEDGER_ARGUMENT_INVALID", opened, tmp_path, source, config=config)
    assert refusal.details["field"] == "config" and "nesting past 256 levels" in refusal.reason


def test_source_dir_must_exist(tmp_path, source):
    refusal = refused("LEDGER_ARGUMENT_INVALID", opened, tmp_path, source, source_dir=tmp_path / "missing")
    assert refusal.details["field"] == "source_dir"


def test_the_config_is_copied_at_open(tmp_path, source):
    config = {"a": {"b": 1}}
    run = opened(tmp_path, source, config=config)
    config["a"]["b"] = 2
    minimal_seal(run)
    assert json.loads(run.record_path.read_bytes())["config"] == {"a": {"b": 1}}


def test_a_non_empty_run_dir_needs_resume(tmp_path, source):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "partial.bin").write_bytes(b"from an earlier attempt")
    refusal = refused("LEDGER_RUN_DIR_NOT_EMPTY", opened, tmp_path, source)
    assert refusal.details["run_dir"] == str(run_dir)
    run = opened(tmp_path, source, resume=True)
    run.add_output("partial", "partial.bin")
    run.seal()
    assert verify(run.record_path, check_code=False)["ok"]


def test_an_empty_existing_run_dir_is_fine(tmp_path, source):
    (tmp_path / "run").mkdir()
    minimal_seal(opened(tmp_path, source))


def test_a_sealed_run_dir_is_never_reopened(tmp_path, source):
    minimal_seal(opened(tmp_path, source))
    before = (tmp_path / "run" / "record.json").read_bytes()
    for resume in (False, True):
        refusal = refused("LEDGER_ALREADY_SEALED", opened, tmp_path, source, run_id="r2", resume=resume)
        assert refusal.details["run_dir"] == str(tmp_path / "run")
    assert (tmp_path / "run" / "record.json").read_bytes() == before


def test_insufficient_disk_is_refused_before_anything_is_written(tmp_path, source, monkeypatch):
    monkeypatch.setattr(writer_module, "_free_bytes", lambda path: 1000)
    refusal = refused("LEDGER_DISK_INSUFFICIENT", opened, tmp_path, source, expected_bytes=1001,
                      run_dir=tmp_path / "a" / "b")
    assert refusal.details == {"path": str(tmp_path), "free": 1000, "expected_bytes": 1001}
    assert not (tmp_path / "a").exists()
    opened(tmp_path, source, expected_bytes=1000)  # exactly enough is enough


def test_real_free_space_is_measured(tmp_path, source):
    refusal = refused("LEDGER_DISK_INSUFFICIENT", opened, tmp_path, source, expected_bytes=1 << 62)
    assert refusal.details["free"] < 1 << 62


def test_an_unwritable_run_dir_is_refused(tmp_path, source):
    blocker = tmp_path / "file"
    blocker.write_text("a file where a directory should be")
    refusal = refused("LEDGER_UNWRITABLE", opened, tmp_path, source, run_dir=blocker / "run")
    assert refusal.details["path"] == str(blocker / "run")


def test_an_unwritable_index_dir_is_refused(tmp_path, source):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    refusal = refused("LEDGER_UNWRITABLE", opened, tmp_path, source, index=blocker / "ledger" / "index.jsonl")
    assert refusal.details["path"] == str(blocker / "ledger")


def test_a_dirty_tree_is_refused(tmp_path, own_source):
    (own_source.source_dir / "core.py").write_text("ANSWER = 43\n")
    refusal = refused("LEDGER_DIRTY_TREE", opened, tmp_path, own_source)
    assert refusal.details["source_dir"] == str(own_source.source_dir)
    assert any("src/apok/core.py" in line for line in refusal.details["changed"])
    assert not (tmp_path / "run").exists()


def test_an_untracked_file_makes_the_tree_dirty(tmp_path, own_source):
    (own_source.repo / "notes.txt").write_text("untracked")
    refusal = refused("LEDGER_DIRTY_TREE", opened, tmp_path, own_source)
    assert any("notes.txt" in line for line in refusal.details["changed"])


def test_an_ignored_file_does_not_make_the_tree_dirty(tmp_path, own_source):
    (own_source.source_dir / "__pycache__" / "mod.cpython-313.pyc").write_bytes(b"cache")
    minimal_seal(opened(tmp_path, own_source))


def test_allow_dirty_records_the_diff(tmp_path, own_source):
    (own_source.source_dir / "core.py").write_text("ANSWER = 43\n")
    run = build_run(tmp_path, own_source, allow_dirty=True)
    code = run.record()["code"]
    diff = (run.run_dir / "code.diff").read_bytes()
    assert code["dirty"] is True and code["diff_file"] == "code.diff" and code["diff_digest"] == sha256_bytes(diff)
    assert b"-ANSWER = 42" in diff and b"+ANSWER = 43" in diff
    assert dict((e["path"], e["digest"]) for e in code["source_manifest"])["core.py"] == \
        sha256_bytes(b"ANSWER = 43\n")
    report = verify(run.record_path, repo=own_source.repo)
    assert status_of(report, "code.diff")["status"] == "PASS"
    assert status_of(report, "code.commit")["status"] == "PASS"  # the commit with the diff applied
    (run.run_dir / "code.diff").write_bytes(diff + b"\n")
    found = failures(pytest.raises(Refusal, verify, run.record_path, repo=own_source.repo).value)
    assert found["code.diff"]["code"] == "LEDGER_FILE_DIGEST_MISMATCH"


def _seal_dirty_and_verify(tmp_path, repo, package):
    run = open_run(tmp_path / "run", run_id="r", purpose="p", config={}, expected_bytes=0, index=None,
                   source_dir=package, allow_dirty=True)
    minimal_seal(run)
    report = verify(run.record_path, repo=repo)
    assert status_of(report, "code.commit")["status"] == "PASS"
    return json.loads(run.record_path.read_bytes())["code"]


def _repo_with(tmp_path, files: dict[str, str], links: dict[str, str] = {}):
    repo = tmp_path / "repo"
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    for rel, target in links.items():
        (repo / rel).symlink_to(target)
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "base")
    return repo


def test_a_dirty_package_whose_directory_name_holds_glob_characters(tmp_path):
    repo = _repo_with(tmp_path, {"lib/pk[1]*?/mod.py": "A = 1\n", "lib/pk1/mod.py": "B = 1\n"})
    (repo / "lib" / "pk[1]*?" / "mod.py").write_text("A = 2\n")
    (repo / "lib" / "pk1" / "mod.py").write_text("B = 2\n")  # what an unescaped pattern would select instead
    code = _seal_dirty_and_verify(tmp_path, repo, repo / "lib" / "pk[1]*?")
    assert code["source_root"] == "lib/pk[1]*?"


def test_a_committed_symlink_replaced_by_a_file_in_a_dirty_tree(tmp_path):
    repo = _repo_with(tmp_path, {"pkg/a.py": "A = 1\n"}, links={"pkg/link.py": "a.py"})
    (repo / "pkg" / "link.py").unlink()
    (repo / "pkg" / "link.py").write_text("now a file\n")
    code = _seal_dirty_and_verify(tmp_path, repo, repo / "pkg")
    assert [e["path"] for e in code["source_manifest"]] == ["a.py", "link.py"]


def test_a_staged_rename_and_a_deletion_in_a_dirty_tree(tmp_path):
    repo = _repo_with(tmp_path, {"pkg/a.py": "A = 1\n", "pkg/b.py": "B = 1\n", "pkg/c.py": "C = 1\n"})
    git(repo, "mv", "pkg/a.py", "pkg/renamed.py")
    (repo / "pkg" / "c.py").unlink()
    code = _seal_dirty_and_verify(tmp_path, repo, repo / "pkg")
    assert [e["path"] for e in code["source_manifest"]] == ["b.py", "renamed.py"]


def test_an_ignored_file_in_a_dirty_package_is_refused_at_open(tmp_path, own_source):
    (own_source.repo / ".gitignore").write_text("__pycache__/\n*.local\n")  # uncommitted: the tree is dirty
    (own_source.source_dir / "settings.local").write_text("ignored, yet part of the package directory")
    refusal = refused("LEDGER_CODE_MISMATCH", opened, tmp_path, own_source, allow_dirty=True)
    assert refusal.details["missing_from_commit"] == ["settings.local"]
    assert "diff of the dirty tree" in refusal.reason
    assert not (tmp_path / "run").exists()


def test_a_clean_tree_whose_files_differ_from_the_commit_is_refused(tmp_path, own_source):
    (own_source.repo / ".gitignore").write_text("__pycache__/\n*.local\n")
    git(own_source.repo, "commit", "-qam", "ignore local files")
    (own_source.source_dir / "settings.local").write_text("ignored, yet part of the package directory")
    refusal = refused("LEDGER_CODE_MISMATCH", opened, tmp_path, own_source)
    assert refusal.details["missing_from_commit"] == ["settings.local"]


def test_a_tracked_package_without_a_commit_is_a_git_failure(tmp_path):
    repo = tmp_path / "unborn"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "a.py").write_text("A = 1\n")
    git(repo, "init", "-q")
    git(repo, "add", "pkg")
    refusal = refused("LEDGER_GIT_FAILED", open_run, tmp_path / "run", run_id="r", purpose="p", config={},
                      expected_bytes=0, index=None, source_dir=repo / "pkg")
    assert refusal.details["source_dir"] == str(repo / "pkg") and "no commit yet" in refusal.reason


def test_an_untracked_package_in_a_repository_without_commits_records_no_commit(tmp_path):
    repo = tmp_path / "unborn"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "a.py").write_text("A = 1\n")
    git(repo, "init", "-q")
    run = open_run(tmp_path / "run", run_id="r", purpose="p", config={}, expected_bytes=0, index=None,
                   source_dir=repo / "pkg")
    minimal_seal(run)
    assert json.loads(run.record_path.read_bytes())["code"]["commit"] is None


def test_a_checkout_git_cannot_open_is_a_git_failure_not_no_checkout(tmp_path, own_source):
    worktree = tmp_path / "wt"
    git(own_source.repo, "worktree", "add", "-q", str(worktree))
    shutil.rmtree(own_source.repo / ".git" / "worktrees")  # the worktree's metadata pruned: git cannot open it
    package = worktree / "src" / "apok"
    (package / "core.py").write_text("ANSWER = 99  # uncommitted\n")
    refusal = refused("LEDGER_GIT_FAILED", open_run, tmp_path / "run", run_id="r", purpose="p", config={},
                      expected_bytes=0, index=None, source_dir=package, allow_dirty=True)
    assert refusal.details["source_dir"] == str(package)
    assert f"{worktree / '.git'} exists" in refusal.reason
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("variable", ["GIT_CEILING_DIRECTORIES", "GIT_DIR", "GIT_WORK_TREE"])
def test_the_environment_cannot_hide_the_checkout(tmp_path, own_source, monkeypatch, variable):
    (own_source.source_dir / "core.py").write_text("ANSWER = 43\n")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    git(elsewhere, "init", "-q")
    value = {"GIT_CEILING_DIRECTORIES": str(own_source.repo), "GIT_DIR": str(elsewhere / ".git"),
             "GIT_WORK_TREE": str(elsewhere)}[variable]
    monkeypatch.setenv(variable, value)
    refusal = refused("LEDGER_DIRTY_TREE", opened, tmp_path, own_source)
    assert refusal.details["changed"] == [" M src/apok/core.py"]  # this checkout, seen through its own index


def test_replacement_objects_are_not_the_commit(tmp_path, own_source):
    git(own_source.repo, "replace", own_source.commits[1], own_source.commits[0])  # `git replace` HEAD by the first
    run = opened(tmp_path, own_source)  # the real HEAD: clean, and its files are the package's
    minimal_seal(run)
    assert status_of(verify(run.record_path, repo=own_source.repo), "code.commit")["status"] == "PASS"


def test_a_package_removed_from_the_index_is_still_a_checkout_of_head(tmp_path, own_source):
    git(own_source.repo, "rm", "-r", "-q", "--cached", "src/apok")  # staged deletion; the files stay, untracked
    code = _seal_dirty_and_verify(tmp_path, own_source.repo, own_source.source_dir)
    assert code["commit"] == own_source.commits[1] and code["dirty"] is True


def test_untracked_caches_in_the_package_stay_out_of_the_diff(tmp_path, own_source):
    (own_source.repo / ".gitignore").unlink()  # caches are no longer ignored: untracked, and the tree is dirty
    code = _seal_dirty_and_verify(tmp_path, own_source.repo, own_source.source_dir)
    assert code["dirty"] is True
    diff = (tmp_path / "run" / "code.diff").read_bytes()
    assert b"b/.gitignore" in diff and b"src/apok/__pycache__/" not in diff


def test_a_failing_new_file_diff_is_a_git_failure(tmp_path, own_source, monkeypatch):
    import apok_eval.ledger.gitcode as gitcode

    (own_source.source_dir / "new.py").write_text("N = 1\n")
    real = gitcode.git_run

    def failing(repo, *args, **kwargs):
        if "--no-index" in args:
            return subprocess.CompletedProcess(args, 2, b"", b"simulated failure")
        return real(repo, *args, **kwargs)

    monkeypatch.setattr(gitcode, "git_run", failing)
    refusal = refused("LEDGER_GIT_FAILED", opened, tmp_path, own_source, allow_dirty=True)
    assert "simulated failure" in refusal.reason and refusal.details["source_dir"] == str(own_source.source_dir)


def test_patches_apply_without_the_user_configuration(tmp_path, own_source, monkeypatch):
    config = tmp_path / "gitconfig"
    config.write_text("[apply]\n\twhitespace = error\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    (own_source.source_dir / "core.py").write_text("ANSWER = 44   \n")  # trailing whitespace: an apply error there
    _seal_dirty_and_verify(tmp_path, own_source.repo, own_source.source_dir)


def test_patches_apply_with_the_temporary_directory_inside_a_repository(tmp_path, own_source, monkeypatch):
    host = tmp_path / "host"
    (host / "tmp").mkdir(parents=True)
    git(host, "init", "-q")
    monkeypatch.setattr(tempfile, "tempdir", str(host / "tmp"))
    (own_source.source_dir / "core.py").write_text("ANSWER = 44\n")
    _seal_dirty_and_verify(tmp_path, own_source.repo, own_source.source_dir)


def test_a_sha256_repository_is_refused_before_anything_is_written(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src" / "apok").mkdir(parents=True)
    (repo / "src" / "apok" / "__init__.py").write_text("")
    if subprocess.run(["git", "init", "-q", "--object-format=sha256", str(repo)], capture_output=True).returncode:
        pytest.skip("this git cannot create a SHA-256 repository")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "sha256")
    commit = git(repo, "rev-parse", "HEAD")
    assert len(commit) == 64
    refusal = refused("LEDGER_SOURCE_UNSUPPORTED", open_run, tmp_path / "run", run_id="r", purpose="p", config={},
                      expected_bytes=0, index=None, source_dir=repo / "src" / "apok")
    assert refusal.details["field"] == "code.commit" and refusal.details["commit"] == commit
    assert not (tmp_path / "run").exists()


def test_a_missing_git_is_a_git_failure(tmp_path, source, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    refusal = refused("LEDGER_GIT_FAILED", opened, tmp_path, source)
    assert "git could not be run" in refusal.reason


@pytest.mark.parametrize("kind", ["file symlink", "dir symlink", "fifo", "non-utf8 name", "backslash name"])
def test_unsupported_entries_in_the_source_tree(tmp_path, kind):
    src = tmp_path / "apok"
    src.mkdir()
    (src / "real.py").write_text("")
    name = {"file symlink": "link.py", "dir symlink": "linkdir", "fifo": "pipe", "non-utf8 name": None,
            "backslash name": "odd\\name.py"}[kind]
    if kind == "file symlink":
        (src / name).symlink_to(src / "real.py")
    elif kind == "dir symlink":
        (src / name).symlink_to(tmp_path, target_is_directory=True)
    elif kind == "fifo":
        os.mkfifo(src / name)
    elif kind == "backslash name":
        (src / name).write_text("")  # recordable by the file system, not as a safe relative path
    else:
        with open(os.path.join(os.fsencode(src), b"bad\xffname.py"), "wb"):
            pass
    refusal = refused("LEDGER_SOURCE_UNSUPPORTED", open_run, tmp_path / "run", run_id="r", purpose="p",
                      config={}, expected_bytes=0, index=None, source_dir=src)
    assert refusal.details["path"] == (name or "bad\udcffname.py")
    assert not (tmp_path / "run").exists()


def test_an_unreadable_source_file_is_refused(tmp_path, monkeypatch):
    import apok_eval.ledger.gitcode as gitcode

    src = tmp_path / "apok"
    src.mkdir()
    (src / "secret.py").write_text("")

    def denied(path):
        raise PermissionError(errno.EACCES, "Permission denied", str(path))

    monkeypatch.setattr(gitcode, "digest_and_size", denied)
    refusal = refused("LEDGER_SOURCE_UNSUPPORTED", open_run, tmp_path / "run", run_id="r", purpose="p",
                      config={}, expected_bytes=0, index=None, source_dir=src)
    assert refusal.details["path"] == str(src / "secret.py") and "Permission denied" in refusal.reason


def test_index_preflight_at_open(tmp_path, source):
    index = tmp_path / "ledger" / "index.jsonl"
    build_run(tmp_path, source, run_id="r1", index=index)
    refusal = refused("LEDGER_INDEX_DUPLICATE_RUN", opened, tmp_path, source, run_dir=tmp_path / "again",
                      run_id="r1", index=index)
    assert refusal.details["run_id"] == "r1"
    index.write_bytes(index.read_bytes()[:-1])  # a torn append
    refused("LEDGER_INDEX_MALFORMED", opened, tmp_path, source, run_dir=tmp_path / "other", run_id="r2",
            index=index)


# --- adding things ---------------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["input", "model", "output", "chunks", "seed", "remote"])
def test_names_are_validated_and_unique_per_section(tmp_path, source, method):
    run = opened(tmp_path, source)
    data = tmp_path / "f.bin"
    data.write_bytes(b"f")
    (run.run_dir / "o.bin").write_bytes(b"o")
    add = {
        "input": lambda name: run.add_input(name, data),
        "model": lambda name: run.add_model(name, data, source="s"),
        "output": lambda name: run.add_output(name, "o.bin"),
        "chunks": lambda name: run.add_chunk_manifest(name, [], source_uri="https://example.invalid/s"),
        "seed": lambda name: run.set_seed(name, 7, config_pointer="/fit/optimizer_random_seed"),
        "remote": lambda name: run.add_remote_input(name, b"r", uri="https://example.invalid/r"),
    }[method]
    refusal = refused("LEDGER_NAME_INVALID", add, "Bad Name")
    assert refusal.details["name"] == "Bad Name"
    add("good")
    refusal = refused("LEDGER_NAME_DUPLICATE", add, "good")
    assert refusal.details["name"] == "good"


def test_the_same_name_may_appear_in_different_sections(tmp_path, source):
    run = opened(tmp_path, source)
    data = tmp_path / "f.bin"
    data.write_bytes(b"f")
    run.add_input("x", data)
    run.add_model("x", data, source="s")


@pytest.mark.parametrize("call, field", [
    (lambda run, f: run.add_input("a", f, uri=""), "uri"),
    (lambda run, f: run.add_remote_input("a", "text, not bytes", uri="https://example.invalid/a"), "data"),
    (lambda run, f: run.add_remote_input("a", b"x", uri=" "), "uri"),
    (lambda run, f: run.add_model("a", f, source=""), "source"),
    (lambda run, f: run.add_model("a", f, source="s", uri=""), "uri"),
    (lambda run, f: run.add_chunk_manifest("a", [], source_uri=""), "source_uri"),
    (lambda run, f: run.add_input("a", os.fsdecode(os.fsencode(f) + b"\xff")), "path"),
])
def test_add_arguments_are_validated(tmp_path, source, call, field):
    run = opened(tmp_path, source)
    data = tmp_path / "f.bin"
    data.write_bytes(b"f")
    refusal = refused("LEDGER_ARGUMENT_INVALID", call, run, data)
    assert refusal.details["field"] == field


def test_inputs_and_models_must_be_regular_files(tmp_path, source):
    run = opened(tmp_path, source)
    refusal = refused("LEDGER_FILE_MISSING", run.add_input, "a", tmp_path / "missing.bin")
    assert refusal.details["path"] == str(tmp_path / "missing.bin")
    refusal = refused("LEDGER_FILE_MISSING", run.add_model, "m", tmp_path, source="s")
    assert refusal.details["what"] == "model 'm'"


def test_an_input_that_cannot_be_read_is_refused_and_named(tmp_path, source, monkeypatch):
    run = opened(tmp_path, source)
    long_path = "/" + "b" * 300
    refusal = refused("LEDGER_FILE_UNREADABLE", run.add_input, "a", long_path)
    assert refusal.details == {"what": "input 'a'", "path": long_path, "errno": errno.ENAMETOOLONG}
    data = tmp_path / "in.bin"
    data.write_bytes(b"x")

    def denied(path):
        raise PermissionError(errno.EACCES, "Permission denied", str(path))

    monkeypatch.setattr(writer_module, "digest_and_size", denied)
    refusal = refused("LEDGER_FILE_UNREADABLE", run.add_model, "m", data, source="s")
    assert refusal.details == {"what": "model 'm'", "path": str(data), "errno": errno.EACCES}


def test_a_fifo_input_is_refused_without_blocking(tmp_path, source):
    run = opened(tmp_path, source)
    os.mkfifo(tmp_path / "pipe")
    with deadline():
        refusal = refused("LEDGER_FILE_MISSING", run.add_input, "a", tmp_path / "pipe")
    assert refusal.details["path"] == str(tmp_path / "pipe") and "not a regular file" in refusal.reason


def test_an_input_symlink_is_followed_and_its_target_digested(tmp_path, source):
    run = opened(tmp_path, source)
    target = tmp_path / "target.bin"
    target.write_bytes(b"target bytes")
    (tmp_path / "link.bin").symlink_to(target)
    entry = run.add_input("a", tmp_path / "link.bin")
    assert entry["digest"] == sha256_bytes(b"target bytes") and entry["path"] == str(tmp_path / "link.bin")


@pytest.mark.parametrize("path", ["../outside.bin", "/etc/hostname", "record.json", "code.diff",
                                  "manifests/x.jsonl", ".apok-tmp-123", "sub/.apok-tmp-1", "."])
def test_unsafe_or_reserved_output_paths(tmp_path, source, path):
    run = opened(tmp_path, source)
    refusal = refused("LEDGER_PATH_UNSAFE", run.add_output, "o", path)
    assert refusal.details["path"] == path


def test_an_output_through_a_symlink_is_refused(tmp_path, source):
    run = opened(tmp_path, source)
    (tmp_path / "real.bin").write_bytes(b"r")
    (run.run_dir / "link.bin").symlink_to(tmp_path / "real.bin")
    refusal = refused("LEDGER_PATH_UNSAFE", run.add_output, "o", "link.bin")
    assert refusal.details["path"] == "link.bin"


def test_an_output_must_exist(tmp_path, source):
    run = opened(tmp_path, source)
    refusal = refused("LEDGER_FILE_MISSING", run.add_output, "o", "nothing.bin")
    assert refusal.details["path"].endswith("nothing.bin")


def test_an_absolute_output_path_inside_the_run_dir_is_recorded_relative(tmp_path, source):
    run = opened(tmp_path, source)
    (run.run_dir / "deep").mkdir()
    (run.run_dir / "deep" / "o.bin").write_bytes(b"o")
    assert run.add_output("o", run.run_dir / "deep" / "o.bin")["path"] == "deep/o.bin"


@pytest.mark.parametrize("value", [True, False, 7.0, "7", None])
def test_seeds_must_be_ints(tmp_path, source, value):
    run = opened(tmp_path, source)
    refusal = refused("LEDGER_SEED_INVALID", run.set_seed, "s", value, config_pointer="/fit/optimizer_random_seed")
    assert refusal.details["seed"] == "s"


@pytest.mark.parametrize("config, pointer, value", [
    ({"seed": 7}, "/missing", 7),
    ({"seed": 7}, "seed", 7),
    ({"seed": 7}, "/seed", 8),
    ({"seed": True}, "/seed", 1),
    ({"seed": 7.0}, "/seed", 7),
    ({"seed": [7]}, "/seed/1", 7),
    ({"a/b": {"~": 7}}, "/a/b/~", 7),
])
def test_seeds_must_be_where_the_config_says(tmp_path, source, config, pointer, value):
    run = opened(tmp_path, source, config=config)
    refusal = refused("LEDGER_SEED_CONFIG_MISMATCH", run.set_seed, "s", value, config_pointer=pointer)
    assert refusal.details == {"seed": "s", "pointer": pointer}


def test_seed_pointers_follow_rfc_6901(tmp_path, source):
    run = opened(tmp_path, source, config={"a/b": {"~": [3, 7]}})
    run.set_seed("s", 7, config_pointer="/a~1b/~0/1")


@pytest.mark.parametrize("entries, fragment", [
    ([{"key": "a", "digest": sha256_bytes(b""), "size": 0}] * 2, "more than once"),
    ([{"key": "a#b", "absent": True}], "'#'"),
    ([{"key": "", "absent": True}], "non-empty"),
    ([{"key": "a", "absent": True, "digest": sha256_bytes(b"")}], "absent chunk"),
    ([{"key": "a", "absent": True, "note": 1}], "absent chunk"),
    ([{"key": "a", "absent": False}], "absent chunk"),
    ([{"key": "a", "digest": sha256_bytes(b"")}], "exactly key, digest and size"),
    ([{"key": "a", "digest": sha256_bytes(b""), "size": 1, "x": 0}], "exactly key, digest and size"),
    ([{"key": "a", "digest": "md5:00", "size": 1}], "not a sha256 digest"),
    ([{"key": "a", "digest": sha256_bytes(b""), "size": True}], "not a non-negative int"),
    ([{"key": "a", "digest": sha256_bytes(b""), "size": -1}], "not a non-negative int"),
    (["not an object"], "not an object"),
    ([{"key": "\ud800", "absent": True}], "utf-8"),
])
def test_chunk_manifest_entries_are_validated(tmp_path, source, entries, fragment):
    run = opened(tmp_path, source)
    refusal = refused("LEDGER_MANIFEST_INVALID", run.add_chunk_manifest, "store", entries,
                      source_uri="https://example.invalid/s")
    assert refusal.details["manifest"] == "store" and fragment in refusal.reason
    assert not (run.run_dir / "manifests" / "store.jsonl").exists()


@pytest.mark.parametrize("planted", ["manifests dir", "sidecar file"])
def test_a_sidecar_is_never_written_through_a_symlink(tmp_path, source, planted):
    inputs = tmp_path / "input-data"
    inputs.mkdir()
    (inputs / "store.jsonl").write_text("precious input bytes\n")
    run_dir = tmp_path / "run"
    (run_dir / "manifests").mkdir(parents=True) if planted == "sidecar file" else run_dir.mkdir()
    if planted == "manifests dir":  # left in a resumed run directory, pointing at an input directory
        (run_dir / "manifests").symlink_to(inputs, target_is_directory=True)
    else:
        (run_dir / "manifests" / "store.jsonl").symlink_to(inputs / "store.jsonl")
    run = opened(tmp_path, source, run_dir=run_dir, resume=True)
    refusal = refused("LEDGER_PATH_UNSAFE", run.add_chunk_manifest, "store", chunk_entries(),
                      source_uri="https://example.invalid/s")
    assert refusal.details["path"] == "manifests/store.jsonl"
    assert (inputs / "store.jsonl").read_text() == "precious input bytes\n"


def test_an_empty_chunk_manifest_is_recorded(tmp_path, source):
    run = opened(tmp_path, source)
    entry = run.add_chunk_manifest("none", [], source_uri="https://example.invalid/s")
    assert (entry["count_present"], entry["count_absent"]) == (0, 0)
    assert (run.run_dir / "manifests" / "none.jsonl").read_bytes() == b""
    minimal_seal(run)
    assert verify(run.record_path, check_code=False)["ok"]


def test_figures_must_be_backed_by_a_recorded_output(tmp_path, source):
    run = opened(tmp_path, source)
    (run.run_dir / "m.json").write_text(json.dumps({"s": {"n": 1}, "list": [1], "empty": {}}))
    refusal = refused("LEDGER_FIGURES_INVALID", run.set_figures, "m", pointer="/s")
    assert refusal.details["source"] == "m" and "not a recorded output" in refusal.reason
    run.add_output("m", "m.json")
    for pointer in ("/missing", "s"):
        refusal = refused("LEDGER_FIGURES_INVALID", run.set_figures, "m", pointer=pointer)
        assert refusal.details == {"source": "m", "pointer": pointer}
    for pointer in ("/list", "/empty", "/s/n"):
        refusal = refused("LEDGER_FIGURES_INVALID", run.set_figures, "m", pointer=pointer)
        assert "non-empty object" in refusal.reason
    assert run.set_figures("m", pointer="/s")["values"] == {"n": 1}
    refusal = refused("LEDGER_FIGURES_INVALID", run.set_figures, "m", pointer="/s")
    assert "already set" in refusal.reason


@pytest.mark.parametrize("content", [b"not json", b'{"n": NaN}', b'{"n": 1, "n": 2}', b"\xff",
                                     b'{"s": "\\ud800"}', b'{"n": ' + b"[" * 300 + b"]" * 300 + b"}"])
def test_figures_need_a_strict_json_output(tmp_path, source, content):
    run = opened(tmp_path, source)
    (run.run_dir / "m.json").write_bytes(content)
    run.add_output("m", "m.json")
    refusal = refused("LEDGER_FIGURES_INVALID", run.set_figures, "m")
    assert refusal.details == {"source": "m", "pointer": ""}


def test_figures_from_an_output_replaced_by_a_fifo_or_a_symlink(tmp_path, source):
    run = opened(tmp_path, source)
    (run.run_dir / "m.json").write_text('{"n": 1}')
    run.add_output("m", "m.json")
    (run.run_dir / "m.json").unlink()
    os.mkfifo(run.run_dir / "m.json")
    with deadline():
        refusal = refused("LEDGER_FIGURES_INVALID", run.set_figures, "m")
    assert refusal.details == {"source": "m", "pointer": ""} and "not a regular file" in refusal.reason
    (run.run_dir / "m.json").unlink()
    (tmp_path / "elsewhere.json").write_text('{"n": 1}')
    (run.run_dir / "m.json").symlink_to(tmp_path / "elsewhere.json")
    refusal = refused("LEDGER_PATH_UNSAFE", run.set_figures, "m")
    assert refusal.details["path"] == "m.json"


def test_predictions_are_validated(tmp_path, source):
    run = opened(tmp_path, source)
    registry = tmp_path / "registry.json"
    registry.write_text("{}")
    refusal = refused("LEDGER_PREDICTIONS_INVALID", run.set_predictions, "registry", ["p1"])
    assert refusal.details["field"] == "registry_input"
    run.add_input("registry", registry)
    for ids in ([], "p1", ["p1", "p1"], ["p1", ""], [3], (i for i in ())):
        refusal = refused("LEDGER_PREDICTIONS_INVALID", run.set_predictions, "registry", ids)
        assert refusal.details["field"] == "ids"
    run.set_predictions("registry", (i for i in ["p2", "p1"]))
    refusal = refused("LEDGER_PREDICTIONS_INVALID", run.set_predictions, "registry", ["p3"])
    assert refusal.details["field"] == "predictions"
    minimal_seal(run)
    assert json.loads(run.record_path.read_bytes())["predictions"]["ids"] == ["p1", "p2"]


# --- sealing ---------------------------------------------------------------------------------------


def test_a_run_without_outputs_is_not_sealed(tmp_path, source):
    run = opened(tmp_path, source)
    refusal = refused("LEDGER_NO_OUTPUTS", run.seal)
    assert refusal.details["run_id"] == "r1"
    assert not run.record_path.exists()


def test_a_sealed_writer_refuses_every_change(tmp_path, source):
    run = opened(tmp_path, source)
    seal = minimal_seal(run)
    data = tmp_path / "f.bin"
    data.write_bytes(b"f")
    for call in (lambda: run.add_input("late", data), lambda: run.add_output("late", "out.txt"),
                 lambda: run.add_model("late", data, source="s"), run.seal,
                 lambda: run.set_figures("out"), lambda: run.set_seed("late", 7, config_pointer="/x"),
                 lambda: run.add_chunk_manifest("late", [], source_uri="u"), lambda: run.set_predictions("x", ["p"]),
                 lambda: run.add_remote_input("late", b"x", uri="u")):
        refusal = refused("LEDGER_ALREADY_SEALED", call)
        assert refusal.details == {"run_id": "r1", "seal": seal}


def test_a_record_that_appears_while_open_is_not_replaced(tmp_path, source):
    run = opened(tmp_path, source)
    (run.run_dir / "out.txt").write_text("done\n")
    run.add_output("out", "out.txt")
    run.record_path.write_bytes(b"someone else's record")
    refusal = refused("LEDGER_ALREADY_SEALED", run.seal)
    assert refusal.details["run_dir"] == str(run.run_dir)
    assert run.record_path.read_bytes() == b"someone else's record"
    assert not [p for p in run.run_dir.iterdir() if p.name.startswith(".apok-tmp-")]


def test_a_failed_record_write_is_refused(tmp_path, source, monkeypatch):
    run = opened(tmp_path, source)
    (run.run_dir / "out.txt").write_text("done\n")
    run.add_output("out", "out.txt")

    def failing_link(src, dst):
        raise OSError(errno.EIO, "simulated I/O error")

    monkeypatch.setattr(os, "link", failing_link)
    refusal = refused("LEDGER_WRITE_FAILED", run.seal)
    assert refusal.details["path"] == str(run.record_path)
    assert not run.record_path.exists()
    monkeypatch.undo()
    assert run.seal()  # nothing was sealed, so sealing can be retried


def test_a_failed_sidecar_write_is_refused(tmp_path, source, monkeypatch):
    run = opened(tmp_path, source)

    def failing_replace(src, dst):
        raise OSError(errno.ENOSPC, "simulated full disk")

    monkeypatch.setattr(os, "replace", failing_replace)
    refusal = refused("LEDGER_WRITE_FAILED", run.add_chunk_manifest, "store", chunk_entries(),
                      source_uri="https://example.invalid/s")
    assert refusal.details["path"] == "manifests/store.jsonl"


def test_a_file_changed_after_it_was_added_refuses_the_seal(tmp_path, source):
    run = opened(tmp_path, source)
    data = tmp_path / "in.bin"
    data.write_bytes(b"before")
    run.add_input("in", data)
    data.write_bytes(b"after!")
    (run.run_dir / "out.txt").write_text("done\n")
    run.add_output("out", "out.txt")
    refusal = refused("LEDGER_VERIFY_FAILED", run.seal)
    assert [(f["check"], f["code"]) for f in refusal.details["failed"]] == \
        [("input:in", "LEDGER_FILE_DIGEST_MISMATCH")]
    assert not run.record_path.exists()


def test_figures_changed_after_they_were_quoted_refuse_the_seal(tmp_path, source):
    run = opened(tmp_path, source)
    (run.run_dir / "m.json").write_text('{"n": 1}')
    run.add_output("m", "m.json")
    run.set_figures("m")
    (run.run_dir / "m.json").write_text('{"n": 2}')
    refusal = refused("LEDGER_VERIFY_FAILED", run.seal)
    assert {f["check"] for f in refusal.details["failed"]} == {"output:m", "figures"}


def test_code_edited_while_the_run_is_open_refuses_the_seal(tmp_path, own_source):
    run = opened(tmp_path, own_source)
    (run.run_dir / "out.txt").write_text("done\n")
    run.add_output("out", "out.txt")
    (own_source.source_dir / "sub" / "mod.py").write_text("def f():\n    return 'changed'\n")
    refusal = refused("LEDGER_CODE_CHANGED", run.seal)
    assert refusal.details["opened"]["commit"] == refusal.details["now"]["commit"] == own_source.commits[1]
    assert refusal.details["opened"]["source_manifest_root"] != refusal.details["now"]["source_manifest_root"]


def test_a_commit_made_while_the_run_is_open_refuses_the_seal(tmp_path, own_source):
    run = opened(tmp_path, own_source)
    (run.run_dir / "out.txt").write_text("done\n")
    run.add_output("out", "out.txt")
    git(own_source.repo, "commit", "-q", "--allow-empty", "-m", "empty")
    refusal = refused("LEDGER_CODE_CHANGED", run.seal)
    assert refusal.details["now"]["commit"] != own_source.commits[1]


def test_an_index_broken_while_the_run_is_open_refuses_the_append_not_the_record(tmp_path, source):
    index = tmp_path / "ledger" / "index.jsonl"
    run = opened(tmp_path, source, index=index)
    index.write_bytes(b'{"torn":')
    (run.run_dir / "out.txt").write_text("done\n")
    run.add_output("out", "out.txt")
    refusal = refused("LEDGER_INDEX_MALFORMED", run.seal)
    assert refusal.details["index"] == str(index)
    assert verify(run.record_path, check_code=False)["ok"]
    refused("LEDGER_ALREADY_SEALED", run.seal)


def test_seal_appends_to_the_index(tmp_path, source):
    index = tmp_path / "ledger" / "index.jsonl"
    run = build_run(tmp_path, source, index=index)
    (line,) = index.read_bytes().splitlines()
    entry = json.loads(line)
    assert entry == {"seq": 1, "run_id": "fold-a", "seal": run.seal, "record_path": "../runs/fold-a/record.json",
                     "prev_entry_digest": None}


@pytest.mark.parametrize("field", ["run_dir", "index", "source_dir"])
@pytest.mark.parametrize("bad", ["a\x00b", "a\ud800b", 5])
def test_a_path_argument_that_is_not_a_path_is_refused_naming_it(tmp_path, source, field, bad):
    path = bad if not isinstance(bad, str) else str(tmp_path / bad)
    exc = refused("LEDGER_ARGUMENT_INVALID", opened, tmp_path, source, **{field: path})
    assert exc.details == {"field": field}
    assert not (tmp_path / "run").exists()  # refused before anything is created


@pytest.mark.parametrize("bad", ["in\x00put.bin", "in\ud800put.bin"])
def test_an_input_or_model_path_that_is_not_a_path_is_refused(tmp_path, source, bad):
    run = opened(tmp_path, source)
    exc = refused("LEDGER_ARGUMENT_INVALID", run.add_input, "x", str(tmp_path / bad))
    assert exc.details == {"field": "path", "what": "input 'x'"}
    exc = refused("LEDGER_ARGUMENT_INVALID", run.add_model, "m", str(tmp_path / bad), source="here")
    assert exc.details == {"field": "path", "what": "model 'm'"}
