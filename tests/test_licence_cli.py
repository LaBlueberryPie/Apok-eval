"""``apok-eval licence-check``: exit 0 with a per-input report when cleared, exit 3
with the refusal as JSON otherwise."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from licence_fixtures import (
    CC_VOLUME_ID,
    CC_VOLUME_URI,
    SPIRAL_URI,
    SYNTHETIC_URI,
    acceptance,
    registry_doc,
    run,
    write_json,
)

import apok_eval
from apok_eval.cli import EXIT_OK, EXIT_REFUSED, main


@pytest.fixture
def registry(tmp_path):
    return write_json(tmp_path / "registry.json", registry_doc())


def _cli(capsys, *argv):
    code = main(["licence-check", *map(str, argv)])
    return code, capsys.readouterr().out


def _cli_err(capsys, *argv):
    code = main(["licence-check", *map(str, argv)])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_cleared_run_prints_each_input_and_exits_0(tmp_path, registry, capsys):
    config = write_json(tmp_path / "run.json", run((CC_VOLUME_ID, CC_VOLUME_URI), ("spiral-input", SPIRAL_URI),
                                                   accepted=acceptance("BUCKET-TERMS", "SERVER-TERMS", "RESERVATION")))
    code, out = _cli(capsys, config, "--registry", registry)
    assert code == EXIT_OK
    assert "CLEARED 2 input(s) for a run" in out
    assert f"input 0: asset {CC_VOLUME_ID} (volume)" in out
    assert "licence:          EDUCELAB-SCROLLS - EduceLab-Scrolls data-server licence" in out
    assert "    - no redistribution without written approval" in out
    assert "additional terms: RESERVATION" in out and "additional terms: none" in out
    assert "required terms:   SERVER-TERMS, RESERVATION" in out and "required terms:   BUCKET-TERMS" in out
    assert out.count("decision:         cleared") == 2
    assert "acceptance: Jane Example on 2025-06-01" in out
    assert "RESERVATION: publication reservation (text not pinned by the registry; binds this person only)" in out


def test_json_output_is_the_clearance_record(tmp_path, registry, capsys):
    config = write_json(tmp_path / "run.json", run(("synthetic", SYNTHETIC_URI + "/x")))
    code, out = _cli(capsys, config, "--registry", registry, "--json")
    assert code == EXIT_OK
    record = json.loads(out)
    assert record["schema"] == "apok-licence-clearance-v1" and record["acceptance"] is None
    assert record["inputs"][0]["uri"] == SYNTHETIC_URI + "/x" and record["registry"]["source"] == str(registry)


def test_refusal_exits_3_with_code_and_cause(tmp_path, registry, capsys):
    config = write_json(tmp_path / "run.json", run(("spiral-input", SPIRAL_URI)))
    code, out = _cli(capsys, config, "--registry", registry)
    assert code == EXIT_REFUSED
    refusal = json.loads(out)
    assert refusal["code"] == "LICENCE_TERMS_ABSENT"
    assert refusal["details"]["required"] == ["SERVER-TERMS", "RESERVATION"]


def test_a_refusal_also_reports_every_input_on_its_own(tmp_path, registry, capsys):
    config = write_json(tmp_path / "run.json", run(("synthetic", SYNTHETIC_URI + "/a"), ("spiral-input", SPIRAL_URI),
                                                   ("nowhere", SYNTHETIC_URI)))
    code, out, err = _cli_err(capsys, config, "--registry", registry)
    assert code == EXIT_REFUSED and json.loads(out)["code"] == "LICENCE_ASSET_UNKNOWN"  # stdout: the refusal only
    assert "licence-check: REFUSED; judged on its own, 2 of 3 input(s) would be refused" in err
    blocks = err.split("\ninput ")
    assert blocks[1].startswith("0: asset synthetic") and "decision:         cleared" in blocks[1]
    assert "licence:          MIT" in blocks[1] and "    - keep the notice" in blocks[1]
    assert blocks[2].startswith("1: asset spiral-input") and "licence:          EDUCELAB-SCROLLS" in blocks[2]
    assert "additional terms: RESERVATION" in blocks[2]
    assert "decision:         refused LICENCE_TERMS_ABSENT: the inputs require terms" in blocks[2]
    assert blocks[3].startswith("2: asset nowhere") and "decision:         refused LICENCE_ASSET_UNKNOWN" in blocks[3]
    assert "licence:          none (unknown asset or licence null)" in blocks[3]


def test_a_json_refusal_reports_every_input_as_json_on_stderr(tmp_path, registry, capsys):
    config = write_json(tmp_path / "run.json", run(("synthetic", SYNTHETIC_URI), (CC_VOLUME_ID, CC_VOLUME_URI),
                                                   accepted=acceptance("BUCKET-TERMS")))
    code, out, err = _cli_err(capsys, config, "--registry", registry, "--for-release", "--json")
    assert code == EXIT_REFUSED and json.loads(out)["code"] == "LICENCE_RELEASE_PUBLICATION"
    decisions = json.loads(err)
    assert [d["decision"] for d in decisions] == ["cleared", "refused"]
    assert decisions[1]["refusal"]["code"] == "LICENCE_RELEASE_PUBLICATION"
    assert decisions[1]["licence"] == "CC-BY-NC-4.0"


def test_a_refusal_of_the_whole_run_has_no_per_input_report(tmp_path, registry, capsys):
    config = write_json(tmp_path / "run.json", run(("synthetic", SYNTHETIC_URI),
                                                   accepted=acceptance("BUCKET-TERMS", person="Jane and John")))
    code, out, err = _cli_err(capsys, config, "--registry", registry)
    assert code == EXIT_REFUSED and json.loads(out)["code"] == "LICENCE_ACCEPTANCE_MALFORMED" and err == ""


def test_for_release_refuses_what_a_run_clears(tmp_path, registry, capsys):
    config = write_json(tmp_path / "run.json", run((CC_VOLUME_ID, CC_VOLUME_URI), accepted=acceptance("BUCKET-TERMS")))
    assert _cli(capsys, config, "--registry", registry)[0] == EXIT_OK
    code, out = _cli(capsys, config, "--registry", registry, "--for-release")
    assert code == EXIT_REFUSED and json.loads(out)["code"] == "LICENCE_RELEASE_PUBLICATION"


def test_default_registry_is_the_shipped_one(tmp_path, capsys):
    config = write_json(tmp_path / "run.json", run(("apok-synthetic-fixtures", "synthetic://apok-fixtures/a")))
    code, out = _cli(capsys, config)
    assert code == EXIT_OK and "registry: apok_eval.licence/registry.json (sha256:" in out
    config = write_json(tmp_path / "run.json", run(("synthetic", SYNTHETIC_URI)))
    code, out = _cli(capsys, config)
    assert code == EXIT_REFUSED and json.loads(out)["code"] == "LICENCE_ASSET_UNKNOWN"


def test_unreadable_inputs_are_refusals_not_crashes(tmp_path, registry, capsys):
    code, out = _cli(capsys, tmp_path / "missing.json", "--registry", registry)
    assert code == EXIT_REFUSED and json.loads(out)["code"] == "LICENCE_RUN_UNREADABLE"
    config = write_json(tmp_path / "run.json", run(("synthetic", SYNTHETIC_URI)))
    code, out = _cli(capsys, config, "--registry", tmp_path / "missing-registry.json")
    assert code == EXIT_REFUSED and json.loads(out)["code"] == "LICENCE_REGISTRY_UNREADABLE"


def test_the_installed_command_line_exits_3_on_refusal(tmp_path, registry):
    config = write_json(tmp_path / "run.json", run(("synthetic", "https://dl.ash2txt.org/x")))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(Path(apok_eval.__file__).resolve().parent.parent),
                                                       os.environ["PYTHONPATH"]]))  # keeps the net guard
    proc = subprocess.run([sys.executable, "-m", "apok_eval.cli", "licence-check", str(config), "--registry", str(registry)],
                          capture_output=True, text=True, env=env, timeout=60)
    assert proc.returncode == 3, proc.stderr
    assert json.loads(proc.stdout)["code"] == "LICENCE_URI_OUTSIDE_ASSET"
