"""``apok-eval e7v3``, ``demo``, ``verify`` and ``inspect`` on the demo's synthetic scroll (apok_eval.eval.cli,
apok_eval.eval.local, apok_eval.eval.demo). Nothing here is scroll data."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from apok_eval import jsonstrict
from apok_eval.cli import main
from apok_eval.eval import demo, local
from apok_eval.eval import synthetic as syn
from apok_eval.ledger.verify import run_verification
from apok_eval.licence.registry import SHIPPED_REGISTRY, read_registry
from apok_eval.refusal import Refusal

FAST = {"registration": {"null_draws": 20, "band_draws": 20, "rotation_draws": 20}}
SLAB = ("--z0", "96", "--z1", "800")
FRAME = ("--frame-shape", *(str(n) for n in demo.FRAME))


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("world")
    paths = demo.write_world(root)
    paths["fast"] = root / "fast.json"
    paths["fast"].write_text(json.dumps(FAST))
    return paths


def args(world, *extra, fibres="fibres", fit="fit"):
    # --allow-dirty: a record of a run from an edited checkout keeps the edits as a diff (the tests run there too)
    return ["e7v3", "--segments", str(world[fit]), "--fibres", str(world[fibres]), "--umbilicus",
            str(world["umbilicus"]), *SLAB, "--settings", str(world["fast"]), "--quiet", "--allow-dirty", *extra]


def refusal(capsys, *argv) -> dict:
    assert main(list(argv)) == 3
    doc = json.loads(capsys.readouterr().out)
    assert doc["refused"] is True
    return doc


# --- e7v3 on local files ------------------------------------------------------------------------------------


def test_e7v3_without_a_volume_scores_with_the_control_not_run_and_seals_a_record(world, tmp_path, capsys):
    out = tmp_path / "run"
    assert main(args(world, *FRAME, "--out", str(out), "--run-id", "local-test")) == 0
    text = capsys.readouterr().out
    assert "CT positive control: NOT RUN" in text and "licence: NOT CHECKED" in text
    result = jsonstrict.loads((out / "e7v3.json").read_bytes())
    fig = result["figures"]
    assert fig["control"] == "NOT RUN" and fig["ct"] is None
    assert {k: fig["e7v3"][k] for k in ("all", "H", "V")} == {"all": "PASS", "H": "PASS", "V": "PASS"}
    assert fig["offsets_voxels"]["V"] == pytest.approx(demo.V_BEHIND, abs=1.0)
    assert result["licence"] == {"checked": False, "reason": "no registry given: the inputs were not licence-checked"}
    record = json.loads((out / "record.json").read_text())
    assert record["run_id"] == "local-test" and record["figures"]["values"] == fig
    assert record["config"]["settings"]["registration"]["null_draws"] == 20  # every value in force is kept
    assert record["config"]["settings"]["registration"]["band"] == 0.125
    assert set(record["seeds"]) == {"nulls", "split", "ceiling", "ct_sampling"}
    names = {m["name"]: m for m in record["chunk_manifests"]}
    assert set(names) == {"segments", "fibres"} and names["fibres"]["count_present"] == 2 * demo.N_FIBRES
    assert names["segments"]["source_uri"] == Path(world["fit"]).resolve().as_uri()
    report = run_verification(out / "record.json")
    assert report["ok"], report
    checks = {c["check"]: c["status"] for c in report["checks"]}
    assert checks["chunk_data:fibres"] == checks["chunk_data:segments"] == checks["figures"] == "PASS"
    assert main(["verify", str(out / "record.json")]) == 0


def test_a_changed_input_fails_verify(world, tmp_path, capsys):
    fibres = tmp_path / "fibres"
    shutil.copytree(world["fibres"], fibres)
    out = tmp_path / "run"
    assert main(args({**world, "fibres": fibres}, *FRAME, "--out", str(out))) == 0
    capsys.readouterr()
    first = sorted(fibres.glob("*.json"))[0]
    first.write_bytes(first.read_bytes() + b" ")  # one byte more: still a fibre, no longer the one read
    report = run_verification(out / "record.json")
    assert not report["ok"]
    failed = {c["check"]: c for c in report["checks"] if c["status"] == "FAIL"}
    assert set(failed) == {"chunk_data:fibres"} and failed["chunk_data:fibres"]["code"] == "LEDGER_CHUNK_MISMATCH"
    doc = refusal(capsys, "verify", str(out / "record.json"))
    assert doc["code"] == "LEDGER_VERIFY_FAILED"


def test_the_record_binds_the_umbilicus_bytes_the_run_parsed(world, tmp_path):
    umbilicus = tmp_path / "umbilicus.json"
    shutil.copyfile(world["umbilicus"], umbilicus)
    inputs = local.Inputs(segments=str(world["fit"]), fibres=str(world["fibres"]), umbilicus=str(umbilicus),
                          frame_shape=demo.FRAME)
    result, listing = local.evaluate(inputs, local.Settings(z_range=(96.0, 800.0), **FAST))
    read = umbilicus.read_bytes()
    parsed = result["umbilicus"]
    assert (parsed["digest"], parsed["size"]) == ("sha256:" + hashlib.sha256(read).hexdigest(), len(read))
    # the axis moves 40 voxels between the run's read and its record: the record would bind bytes the run never used
    (tmp_path / "moved").mkdir()
    moved = syn.write_umbilicus_file(tmp_path / "moved", (demo.CENTRE[0], demo.CENTRE[1] + 40.0),
                                     z_top=float(demo.FRAME[0]))
    umbilicus.write_bytes(moved.read_bytes())
    assert umbilicus.read_bytes() != read
    out = tmp_path / "run-moved"
    with pytest.raises(Refusal) as refused:
        local.seal_run(result, listing, out, run_id="moved", purpose="a test", allow_dirty=True)
    assert refused.value.code == "EVAL_RUN_INPUT_CHANGED"
    assert refused.value.details["read_digest"] == parsed["digest"] != refused.value.details["now_digest"]
    assert not (out / "record.json").exists() and not (out / "e7v3.json").exists()
    umbilicus.write_bytes(read)  # the bytes the run read: it seals, and the record lists exactly those
    out = tmp_path / "same"
    local.seal_run(result, listing, out, run_id="same", purpose="a test", allow_dirty=True)
    record = json.loads((out / "record.json").read_text())
    assert [(i["name"], i["digest"], i["size"]) for i in record["inputs"]] == \
        [("umbilicus", parsed["digest"], parsed["size"])]
    assert run_verification(out / "record.json")["ok"]


def test_e7v3_with_the_volume_runs_the_control_first(world, tmp_path, capsys):
    assert main(args(world, "--volume", str(world["ct"]), "--json", "--no-record")) == 0
    doc = json.loads(capsys.readouterr().out)
    fig = doc["figures"]
    assert doc["seal"] is None and doc["record"] is None
    assert fig["control"] == "CONFIRMED" and fig["ct"]["verdicts"]["fibres"] == "LOCKED"
    assert fig["ct"]["period"] in (15, 16, 17) and fig["e7v3"]["all"] == "PASS"
    assert main(args(world, "--volume", str(world["ct"]), "--json", "--no-record", fibres="fibres_off")) == 0
    off = json.loads(capsys.readouterr().out)["figures"]
    assert off["control"] == "REFUTED" and off["e7v3"] == {"all": "NOT MEASURED"}


def test_the_volume_level_is_the_frame_unless_a_declared_shape_disagrees(world, capsys):
    doc = refusal(capsys, *args(world, "--volume", str(world["ct"]), "--frame-shape", "896", "448", "447",
                                "--no-record"))
    assert doc["code"] == "EVAL_RUN_FRAME" and doc["details"]["level_shape"] == list(demo.FRAME)
    doc = refusal(capsys, *args(world, "--no-record"))
    assert doc["code"] == "EVAL_RUN_FRAME"
    doc = refusal(capsys, *args(world, "--volume", str(world["ct"]), "--level", "1", "--no-record"))
    assert doc["code"] == "EVAL_RUN_VOLUME"


def test_a_level_that_is_not_an_unsigned_c_array_is_refused(world, tmp_path, capsys):
    ct = tmp_path / "ct.zarr"
    shutil.copytree(world["ct"], ct)
    meta = json.loads((ct / "0" / ".zarray").read_text())
    (ct / "0" / ".zarray").write_text(json.dumps({**meta, "dtype": "<f4"}))
    doc = refusal(capsys, *args(world, "--volume", str(ct), "--no-record"))
    assert doc["code"] == "EVAL_RUN_VOLUME" and doc["details"]["dtype"] == "<f4"
    (ct / "0" / ".zarray").write_text(json.dumps({**meta, "compressor": {"id": "bz2", "level": 1}}))
    assert refusal(capsys, *args(world, "--volume", str(ct), "--no-record"))["code"] == "IO_CODEC_UNSUPPORTED"


def test_segments_need_winding_ranges_or_a_windings_map(world, tmp_path, capsys):
    segs = tmp_path / "segs"
    segs.mkdir()
    for d, new in (("demo-w004-006", "inner"), ("demo-w007-010", "demo-w007-010")):
        shutil.copytree(world["fit"] / d, segs / new)
    doc = refusal(capsys, *args({**world, "fit": segs}, *FRAME, "--no-record"))
    assert doc["code"] == "EVAL_SEGMENT_NAME" and doc["details"]["segment"] == "inner"
    good = tmp_path / "windings.json"
    good.write_text(json.dumps({"inner": [4, 6]}))
    assert main(args({**world, "fit": segs}, *FRAME, "--windings", str(good), "--no-record", "--json")) == 0
    assert json.loads(capsys.readouterr().out)["figures"]["e7v3"]["all"] == "PASS"
    for bad, code in (({"elsewhere": [1, 2]}, "EVAL_RUN_WINDINGS"), ({"inner": [6, 4]}, "EVAL_RUN_WINDINGS"),
                      ({"inner": [4, True]}, "EVAL_RUN_WINDINGS"), ([4, 6], "EVAL_RUN_WINDINGS"),
                      ({"inner": [9, 10]}, "EVAL_SEGMENT_RANGES")):
        good.write_text(json.dumps(bad))
        assert refusal(capsys, *args({**world, "fit": segs}, *FRAME, "--windings", str(good),
                                     "--no-record"))["code"] == code
    empty = tmp_path / "empty"
    empty.mkdir()
    assert refusal(capsys, *args({**world, "fit": empty}, *FRAME, "--no-record"))["code"] == "EVAL_RUN_INPUT"


def test_a_segment_outside_the_frame_is_refused(world, capsys):
    doc = refusal(capsys, *args(world, "--frame-shape", "896", "300", "448", "--no-record"))
    assert doc["code"] == "EVAL_RUN_SEGMENT_OUTSIDE_FRAME" and doc["details"]["frame_shape_zyx"] == [896, 300, 448]


def test_fibre_files_that_refuse_are_counted_and_none_at_all_is_refused(world, tmp_path, capsys):
    fibres = tmp_path / "fibres"
    shutil.copytree(world["fibres"], fibres)
    (fibres / "broken.json").write_text('{"type": "vc3d_fiber", "type": 1}')
    (fibres / "bad name.json").write_text("{}")
    (fibres / "notes.txt").write_text("not a fibre")
    assert main(args({**world, "fibres": fibres}, *FRAME, "--no-record", "--json")) == 0
    fig = json.loads(capsys.readouterr().out)["figures"]
    assert fig["fibres"] == {"files": 2 * demo.N_FIBRES + 2, "read": 2 * demo.N_FIBRES,
                             "refused": {"EVAL_FIBRE_JSON_INVALID": 1, "IO_STORE_KEY_INVALID": 1}}
    junk = tmp_path / "junk"
    junk.mkdir()
    (junk / "a.json").write_text("[]")
    doc = refusal(capsys, *args({**world, "fibres": junk}, *FRAME, "--no-record"))
    assert doc["code"] == "EVAL_RUN_FIBRES" and doc["details"]["refused"] == {"EVAL_FIBRE_STRUCTURE": 1}


def test_settings_files_are_strict_and_name_only_the_rules_settings(world, tmp_path, capsys):
    s = tmp_path / "s.json"
    for doc, code in (({"registration": {"bandd": 0.1}}, "EVAL_RUN_SETTINGS"),
                      ({"registration": {"z_range": [0, 1]}}, "EVAL_RUN_SETTINGS"),
                      ({"seed": 3}, "EVAL_RUN_SETTINGS"),
                      ({"face": {"offset_step": 0.03}}, "EVAL_FACE_PARAMETER"),
                      ({"registration": {"null_draws": 5}}, "EVAL_REG_PARAMETER"),
                      ({"ct": {"window": 99}}, "EVAL_CT_PARAMETER")):
        s.write_text(json.dumps(doc))
        argv = [a if a != str(world["fast"]) else str(s) for a in args(world, *FRAME, "--no-record")]
        assert refusal(capsys, *argv)["code"] == code, doc
    s.write_text('{"margin": 1, "margin": 2}')
    argv = [a if a != str(world["fast"]) else str(s) for a in args(world, *FRAME, "--no-record")]
    assert "duplicate key" in refusal(capsys, *argv)["details"]["problem"]


def test_settings_record_every_value_in_force():
    settings = local.load_settings(None, (0.0, 100.0))
    d = settings.as_dict()
    assert d["registration"]["z_range"] == [0.0, 100.0] and d["ct"]["z_range"] == [0.0, 100.0]
    assert (d["registration"]["seed"], d["face"]["split_seed"], d["face"]["ceiling_seed"], d["ct"]["seed"]) == \
        (1, 2, 3, 1)
    jsonstrict.dumps(d)


def test_the_control_rule():
    rule = local.control_verdict
    assert rule({"null": "NOT LOCKED", "fibres": "LOCKED"}) == "CONFIRMED"
    assert rule({"null": "NOT LOCKED", "fibres": "NOT LOCKED"}) == "REFUTED"
    assert rule({"null": "NOT LOCKED", "fibres": "UNCLEAR"}) == "INCONCLUSIVE"
    assert rule({"null": "LOCKED", "fibres": "LOCKED"}) == "INCONCLUSIVE"
    assert rule({"null": "INCONCLUSIVE", "fibres": "INCONCLUSIVE"}) == "INCONCLUSIVE"
    assert rule({}) == "INCONCLUSIVE"


def test_an_existing_run_directory_is_never_reused(world, tmp_path, capsys):
    out = tmp_path / "run"
    out.mkdir()
    (out / "x").write_text("x")
    assert refusal(capsys, *args(world, *FRAME, "--out", str(out)))["code"] == "LEDGER_RUN_DIR_NOT_EMPTY"


def test_the_leakage_audit_scores_the_far_points_again(world, capsys):
    assert main(args(world, *FRAME, "--patches", str(world["fit"]), "--no-record", "--json")) == 0
    fig = json.loads(capsys.readouterr().out)["figures"]
    # The H fibres lie on the patches' sheets (near); the V fibres 4 voxels behind them (far, beyond 2 voxels).
    assert 0.3 < fig["leakage"]["share_near"] < 0.7
    assert fig["leakage"]["e7v3_far"]["V"] == "PASS" and fig["leakage"]["e7v3_far"]["H"] == "NOT MEASURED"


# --- the licence gate on local inputs -------------------------------------------------------------------------


def _registry(tmp_path, root: Path, licence: str) -> Path:
    doc = read_registry(SHIPPED_REGISTRY)
    doc["hosts"][root.resolve().as_uri()] = {"allowed_licences": [licence], "requires_terms": [], "note": "test root"}
    doc["assets"]["test-data"] = {"kind": "synthetic" if licence == "MIT" else "dataset",
                                  "locations": [root.resolve().as_uri()], "licence": licence,
                                  "additional_terms": [], "aliases": [], "evidence": "test fixture"}
    path = tmp_path / f"registry-{licence}.json"
    path.write_text(json.dumps(doc))
    return path


def test_the_gate_refuses_undeclared_inputs_before_reading_anything(world, tmp_path, capsys):
    reg = tmp_path / "shipped.json"
    reg.write_bytes(SHIPPED_REGISTRY.read_bytes())
    doc = refusal(capsys, *args(world, *FRAME, "--registry", str(reg), "--no-record"))
    assert doc["code"] == "EVAL_RUN_UNDECLARED" and doc["details"]["input"] == "segments"


def test_the_gate_clears_declared_inputs_and_the_record_keeps_the_clearance(world, tmp_path, capsys):
    root = Path(world["umbilicus"]).parent
    reg = _registry(tmp_path, root, "MIT")
    out = tmp_path / "run"
    assert main(args(world, *FRAME, "--registry", str(reg), "--out", str(out))) == 0
    capsys.readouterr()
    record = json.loads((out / "record.json").read_text())
    clearance = record["config"]["licence"]["clearance"]
    assert record["config"]["licence"]["checked"] is True and clearance["acceptance"] is None
    assert [i["asset"] for i in clearance["inputs"]] == ["test-data"] * 3 and clearance["required_terms"] == []
    assert run_verification(out / "record.json")["ok"]


def test_data_under_terms_needs_your_own_terms_file(world, tmp_path, capsys):
    root = Path(world["umbilicus"]).parent
    reg = _registry(tmp_path, root, "CC-BY-NC-4.0")
    doc = refusal(capsys, *args(world, *FRAME, "--registry", str(reg), "--no-record"))
    assert doc["code"] == "LICENCE_TERMS_ABSENT" and doc["details"]["required"] == ["VESUVIUS-DATA-TERMS"]
    pinned = read_registry(SHIPPED_REGISTRY)["terms"]["VESUVIUS-DATA-TERMS"]["text_sha256"]
    terms = tmp_path / "terms.json"
    acceptance = {"person": "Jane Example", "date": "2025-06-01",
                  "accepted": [{"terms": "VESUVIUS-DATA-TERMS", "text_sha256": pinned}]}
    terms.write_text(json.dumps({"terms_accepted_by": acceptance}))
    assert main(args(world, *FRAME, "--registry", str(reg), "--terms", str(terms), "--no-record", "--json")) == 0
    capsys.readouterr()
    terms.write_text(json.dumps({"terms_accepted_by": acceptance, "inputs": []}))
    doc = refusal(capsys, *args(world, *FRAME, "--registry", str(reg), "--terms", str(terms), "--no-record"))
    assert doc["code"] == "EVAL_CLI_FILE_INVALID"
    terms.write_text(json.dumps({"terms_accepted_by": acceptance}))
    doc = refusal(capsys, *args(world, *FRAME, "--terms", str(terms), "--no-record"))
    assert doc["code"] == "EVAL_CLI_FILE_INVALID" and doc["details"]["problem"] == "no --registry"


# --- the synthetic world, verify and inspect ------------------------------------------------------------------


def test_the_synthetic_ct_is_bright_on_the_sheets_and_absent_outside_its_band(world):
    level = Path(world["ct"]) / "0"
    stored = sorted(p.name for p in level.iterdir() if not p.name.startswith("."))
    zs = {int(name.split(".")[0]) for name in stored}
    assert zs == set(range(demo.CT_Z[0] // demo.CHUNKS[0], demo.CT_Z[1] // demo.CHUNKS[0]))
    # one point on the sheet of winding 6, one half a gap outside it on the same ray
    pts = syn.spiral_point(np.array([6.0, 6.0]), 480.0, phase=demo.PHASE, centre=demo.CENTRE,
                           radius=lambda w: demo.sheet_radius(w, 480.0) + np.array([0.0, demo.SPACING / 2]))
    block = syn.layered_block((464, 0, 0), (32, 448, 448), sheet_radius=demo.sheet_radius, lo=4, hi=10,
                              centre=demo.CENTRE, phase=demo.PHASE, spacing=demo.SPACING, noise=0.0)
    on = block[480 - 464, int(round(pts[0, 1])), int(round(pts[0, 2]))]
    between = block[480 - 464, int(round(pts[1, 1])), int(round(pts[1, 2]))]
    assert on > 150 and between < 60
    assert block[16, 224, 224] == 0  # air inside the innermost sheet


def test_inspect_and_verify_are_the_io_and_ledger_commands(world, tmp_path, capsys):
    assert main(["inspect", str(world["ct"]), "--no-frame", "--json"]) == 0
    a = json.loads(capsys.readouterr().out)
    assert main(["io", "inspect", str(world["ct"]), "--no-frame", "--json"]) == 0
    assert a == json.loads(capsys.readouterr().out) and a["kind"] == "ome-zarr-v2"
    assert main(["inspect", str(world["fit"] / "demo-w004-006"), "--policy", "spiral", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["kind"] == "tifxyz"


def test_the_demo_gives_every_expected_verdict_and_every_record_verifies(tmp_path, capsys):
    """The shipped demo, with the default settings, end to end (README, "Quickstart")."""
    keep = tmp_path / "demo"
    assert main(["demo", "--keep", str(keep), "--quiet"]) == 0
    text = capsys.readouterr().out
    assert text.rstrip().endswith("demo passed: every verdict as expected and every record verifies")
    for sc in demo.SCENARIOS:
        record = keep / "runs" / sc.name / "record.json"
        report = run_verification(record)
        assert report["ok"] and report["counts"]["FAIL"] == 0
        fig = json.loads((record.parent / "e7v3.json").read_text())["figures"]
        got = {"control": fig["control"], **fig["e7v3"]}
        assert {k: got[k] for k in sc.expected} == sc.expected
        clearance = json.loads(record.read_text())["config"]["licence"]["clearance"]
        assert {i["asset"] for i in clearance["inputs"]} == {"apok-eval-demo"}
    assert refusal(capsys, "demo", "--keep", str(keep))["code"] == "EVAL_RUN_INPUT"  # never over an old one
