"""What every ``apok-eval`` subcommand prints: one JSON printer, strict JSON out, the same conventions everywhere.

* Every JSON document a subcommand prints goes through :func:`apok_eval.cli.print_json`, so apok-eval's own output is
  accepted by its own strict parser (:func:`apok_eval.jsonstrict.loads`): a path holding bytes that are not
  UTF-8 is written with those bytes as ``\\xNN`` (:func:`apok_eval.fileio.portable_name`), a non-finite number in
  a refusal's details as its ``repr`` (``"nan"``, ``"inf"``), bytes as text with ``\\xNN`` escapes; any other
  value that is not JSON is a ``TypeError``, never coerced.
* Every command group needs a subcommand (``apok-eval ledger``, ``apok-eval io``).
* With ``--json``, a check that fails prints its per-check report as JSON on stderr; stdout holds only the
  refusal document.
"""

from __future__ import annotations

import ast
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest

from apok_eval import jsonstrict
from apok_eval.cli import json_safe, main, print_json
from apok_eval.io.tifxyz import write_tifxyz

SRC = Path(__file__).resolve().parents[1] / "src" / "apok_eval"
META = {"format": "tifxyz", "type": "seg", "uuid": "inspect-surface", "scale": [0.05, 0.05]}


def undecodable(parent: Path, name: bytes) -> str:
    """``parent/name`` as Python spells a file name holding bytes that are not UTF-8 (lone surrogates)."""
    path = os.fsdecode(os.fsencode(parent) + b"/" + name)
    assert any("\udc80" <= ch <= "\udcff" for ch in path)
    return path


# --- strict JSON out ------------------------------------------------------------------------------


def test_inspect_json_of_a_surface_under_a_non_utf8_name_is_strict_json(tmp_path, capsys):
    path = undecodable(tmp_path, b"surf\xff")
    rows, cols = np.mgrid[0:2, 0:3]
    write_tifxyz(path, x=cols.astype(np.float32), y=rows.astype(np.float32),
                 z=(1 + rows * 3 + cols).astype(np.float32), meta=META)
    assert main(["io", "inspect", path, "--json"]) == 0
    out = capsys.readouterr().out
    record = jsonstrict.loads(out.encode("utf-8"))
    portable = os.fsencode(path).decode("utf-8", "backslashreplace")
    assert record["path"] == portable and record["path"].endswith("surf\\xff")
    assert record["summary"]["surface"]["path"] == portable


def test_the_inspect_record_itself_writes_paths_as_its_summary_does(tmp_path):
    """The record ``inspect_path`` returns (not only its printed form) is canonical JSON, path included."""
    from apok_eval.canonical import canonical_dumps
    from apok_eval.io.cli import inspect_path

    mesh = undecodable(tmp_path, b"mesh\xfe.obj")
    Path(mesh).write_bytes(b"v 1 2 3\nv 2 3 4\nv 3 4 4\nf 1 2 3\n")
    pcl = undecodable(tmp_path, b"points\xfe.json")
    Path(pcl).write_bytes(b'{"vc_pointcollections_json_version": "1", "collections": {}}')
    umbilicus = undecodable(tmp_path, b"umbilicus\xfe.json")
    Path(umbilicus).write_bytes(b'{"control_points": [{"z": 0, "y": 3, "x": 2}, {"z": 8, "y": 3, "x": 2}]}')
    for path, where in ((mesh, ("source", "path")), (pcl, ("path",)), (pcl, ("basename",)), (umbilicus, ("path",))):
        record = inspect_path(path)
        assert record["path"] == os.fsencode(path).decode("utf-8", "backslashreplace")
        summary = record["summary"]
        for key in where:
            summary = summary[key]
        assert "\\xfe" in summary, (path, where)
        canonical_dumps(record)


def test_a_refusal_naming_a_non_utf8_path_is_strict_json(tmp_path, capsys):
    path = undecodable(tmp_path, b"nope\xff")
    assert main(["io", "inspect", path]) == 3
    refusal = jsonstrict.loads(capsys.readouterr().out.encode("utf-8"))
    assert refusal["code"] == "IO_INSPECT_PATH_INVALID"
    assert refusal["details"]["path"].endswith("nope\\xff")


def test_json_safe_writes_what_strict_json_cannot_hold_and_refuses_the_rest():
    value = {"s": "a\udcffb", "nan": float("nan"), "inf": float("-inf"), "b": b"k\xff", "t": (1, 2.5),
             "p": Path("x") / "y", "i": np.int64(7), "f": np.float32(0.5), "ok": [True, None, "é"],
             "k\udcfe": 1}
    assert json_safe(value) == {"s": "a\\xffb", "nan": "nan", "inf": "-inf", "b": "k\\xff", "t": [1, 2.5],
                                "p": "x/y", "i": 7, "f": 0.5, "ok": [True, None, "é"], "k\\xfe": 1}
    for bad in (object(), {1: "non-string key"}, {"set": {1}}):
        with pytest.raises(TypeError):
            json_safe(bad)
    buffer = io.StringIO()
    print_json(value, file=buffer)
    assert jsonstrict.loads(buffer.getvalue().encode("utf-8")) == json_safe(value)
    assert buffer.getvalue() == json.dumps(json_safe(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def test_every_command_prints_json_only_through_print_json():
    """No module of the command line calls json.dumps itself, except the one serialiser behind print_json and the
    refusal printer (cli.py ``_dumps``) and the text renderer's one-value helper (io/cli.py ``_brief``, a value
    inside a line of text, not a document)."""
    allowed = {("cli.py", "_dumps"), ("io/cli.py", "_brief")}
    calls = set()
    for path in [SRC / "cli.py", *SRC.glob("*/cli.py")]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        owner = {}
        for fn in ast.walk(tree):
            if isinstance(fn, ast.FunctionDef):
                for node in ast.walk(fn):
                    owner[id(node)] = fn.name  # the innermost function wins: ast.walk visits outer ones first
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "dumps" \
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "json":
                calls.add((str(path.relative_to(SRC)), owner.get(id(node), "<module>")))
    assert calls - allowed == set()
    assert ("cli.py", "_dumps") in calls


# --- command groups need a subcommand ----------------------------------------------------------------


@pytest.mark.parametrize("group", ["ledger", "io"])
def test_a_command_group_without_a_subcommand_is_a_usage_error_naming_the_group(group, capsys):
    with pytest.raises(SystemExit) as exit_:
        main([group])
    assert exit_.value.code == 2
    err = capsys.readouterr().err
    assert f"usage: apok-eval {group}" in err and "the following arguments are required" in err


# --- --json reports a failing check on stderr ----------------------------------------------------------


def test_ledger_verify_json_prints_the_failing_report_on_stderr(tmp_path, capsys):
    record = tmp_path / "run" / "record.json"
    record.parent.mkdir()
    record.write_bytes(b"{}")
    assert main(["ledger", "verify", "--json", str(record)]) == 3
    out, err = capsys.readouterr()
    refusal = jsonstrict.loads(out.encode("utf-8"))
    assert refusal["code"] == "LEDGER_VERIFY_FAILED"
    report = jsonstrict.loads(err.encode("utf-8"))
    assert report["ok"] is False and report["record"] == str(record)
    assert any(check["status"] == "FAIL" for check in report["checks"])
