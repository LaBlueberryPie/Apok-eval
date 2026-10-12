"""Residuals from a final review's confirmation pass, each pinned by a test that fails without its fix."""

from __future__ import annotations

import errno
import json

import pytest

from apok_eval import cli
from apok_eval.refusal import Refusal, register_codes


def test_a_refusal_with_unprintable_details_still_exits_3_with_a_marked_document(monkeypatch, capsys):
    register_codes("tests", {"TEST_UNPRINTABLE_DETAILS": "a refusal whose details hold a set"})

    def refuse(_args):
        raise Refusal("TEST_UNPRINTABLE_DETAILS", "details hold a set", values={1, 2})

    parser = cli.build_parser()
    real_parse = parser.parse_args
    monkeypatch.setattr(cli, "build_parser", lambda: parser)
    monkeypatch.setattr(parser, "parse_args", lambda argv=None: type("A", (), {"func": staticmethod(refuse)})())
    assert cli.main([]) == cli.EXIT_REFUSED
    doc = json.loads(capsys.readouterr().out)
    assert doc["code"] == "TEST_UNPRINTABLE_DETAILS" and doc["refused"] is True
    assert "{1, 2}" in doc["details"]["unprintable_details"] and doc["details"]["printer_error"]
    assert real_parse  # the real parser was built; only its parse was replaced


def test_a_damaged_sidecar_of_a_local_store_run_leaves_chunk_data_not_checked(tmp_path):
    """verify.check_chunk_data: an unreadable sidecar is reported once, as the manifest's FAIL; the chunk
    re-read is NOT CHECKED with that reason (and verify returns a report rather than crashing)."""
    from io_zarr_fixtures import write_small
    from ledger_fixtures import make_source_repo

    from apok_eval.io.frame import VolumeFrame
    from apok_eval.io.omezarr import open_volume
    from apok_eval.io.store import LocalStore
    from apok_eval.ledger import open_run
    from apok_eval.ledger.verify import NOT_CHECKED, run_verification

    source = make_source_repo(tmp_path / "own")
    store_root = tmp_path / "vol.zarr"
    write_small(store_root)
    volume = open_volume(LocalStore(store_root), VolumeFrame("20241024131838", (9, 7, 5), 7.91, "test"))
    volume.read((0, 0, 0), (9, 7, 5))
    run = open_run(tmp_path / "run", run_id="local", purpose="a local volume read", config={}, expected_bytes=0,
                   index=None, source_dir=source.source_dir)
    run.add_chunk_manifest("volume", volume.log.entries(), source_uri=store_root.as_uri())
    (run.run_dir / "out.json").write_text("{}")
    run.add_output("out", "out.json")
    run.seal()
    sidecar = next((run.run_dir / "manifests").glob("*.jsonl"))
    sidecar.write_bytes(b"not json\n")
    report = run_verification(run.record_path, repo=source.repo)
    checks = {c["check"]: c for c in report["checks"]}
    assert checks["chunk_manifest:volume"]["status"] == "FAIL"
    assert checks["chunk_data:volume"]["status"] == NOT_CHECKED
    assert "the sidecar cannot be read" in checks["chunk_data:volume"]["reason"]


def test_an_index_that_cannot_be_copied_is_a_git_failure_naming_it(tmp_path, monkeypatch):
    """gitcode.working_diff copies git's index before diffing a dirty tree; an OSError there is a GitError, which
    capture_code refuses as LEDGER_GIT_FAILED naming the copy."""
    from ledger_fixtures import make_source_repo

    from apok_eval.ledger import gitcode

    source = make_source_repo(tmp_path / "own")
    (source.source_dir / "core.py").write_text("ANSWER = 43\n")  # dirty: the diff path runs

    def no_copy(*args, **kwargs):
        raise PermissionError(errno.EACCES, "Permission denied", str(args[0]))

    monkeypatch.setattr(gitcode.shutil, "copy2", no_copy)
    with pytest.raises(Refusal) as exc:
        gitcode.capture_code(source.source_dir)
    assert exc.value.code == "LEDGER_GIT_FAILED" and "cannot copy the index" in exc.value.reason
