"""Umbilicus: declared variants, sorted control points, the two out-of-range policies, metadata against a
declared frame, a round trip, and every refusal."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from apok_eval.canonical import canonical_dumps
from apok_eval.io import umbilicus as U
from apok_eval.io.frame import VolumeFrame
from apok_eval.refusal import Refusal

FITTER = {"control_points": [{"z": 10, "y": 20.0, "x": 40.0, "score": 0.9},
                             {"z": 0, "y": 10.0, "x": 20.0, "estimates": [1, 2], "inside": True},
                             {"z": 30, "y": 20.0, "x": 10.0}]}
FRAME = VolumeFrame(volume_id="20990101000004", shape_zyx=(64, 48, 50), voxel_size_um=2.4, source="test")


def put(tmp_path, content, name="umbilicus.json") -> str:
    path = tmp_path / name
    if isinstance(content, bytes):
        path.write_bytes(content)
    elif isinstance(content, str):
        path.write_text(content)
    else:
        path.write_text(json.dumps(content))
    return str(path)


def refusal(fn, *args, **kwargs) -> Refusal:
    with pytest.raises(Refusal) as exc:
        fn(*args, **kwargs)
    return exc.value


def with_meta(**meta) -> dict:
    return {**FITTER, "metadata": meta}


# --------------------------------------------------------------------------------------------- reading


def test_fitter_form_is_sorted_by_z_and_keeps_its_digest(tmp_path):
    path = put(tmp_path, FITTER)
    u = U.read_umbilicus(path)
    assert u.control_zyx.tolist() == [[0, 10, 20], [10, 20, 40], [30, 20, 10]]
    assert u.z_span == (0.0, 30.0) and u.variant == "fitter" and u.coordinate_scale is None
    assert u.digest == "sha256:" + hashlib.sha256((tmp_path / "umbilicus.json").read_bytes()).hexdigest()
    assert u.metadata is None and u.findings == ()


def test_interpolation_and_both_out_of_range_policies(tmp_path):
    u = U.read_umbilicus(put(tmp_path, FITTER))
    z = [-10, 0, 5, 10, 20, 30, 40]
    ext = u.yx_at(z, out_of_range="extrapolate")
    assert ext.yx.tolist() == [[0, 0], [10, 20], [15, 30], [20, 40], [20, 25], [20, 10], [20, -5]]
    assert ext.outside.tolist() == [True, False, False, False, False, False, True] and ext.outside_count == 2
    clamp = u.yx_at(z, out_of_range="clamp")
    assert clamp.yx.tolist() == [[10, 20], [10, 20], [15, 30], [20, 40], [20, 25], [20, 10], [20, 10]]
    assert clamp.outside.tolist() == ext.outside.tolist() and clamp.out_of_range == "clamp"
    single = u.yx_at(np.float32(2.5), out_of_range="extrapolate")
    assert single.z.tolist() == [2.5] and single.yx.tolist() == [[12.5, 25.0]]


def test_clamp_returns_end_points_exactly_at_the_ends(tmp_path):
    points = {"control_points": [{"z": 0.1, "y": 0.7, "x": 0.3}, {"z": 0.3, "y": 0.1, "x": 0.9}]}
    u = U.read_umbilicus(put(tmp_path, points))
    assert u.yx_at([0.3], out_of_range="clamp").yx.tolist() == [[0.1, 0.9]]
    assert not u.yx_at([0.3], out_of_range="clamp").outside[0]


def test_coordinate_scale_applies_to_all_axes_only_when_given(tmp_path):
    u = U.read_umbilicus(put(tmp_path, FITTER), coordinate_scale=2)
    assert u.control_zyx.tolist() == [[0, 20, 40], [20, 40, 80], [60, 40, 20]]
    assert u.coordinate_scale == 2.0 and u.as_dict()["coordinate_scale"] == 2.0
    assert U.read_umbilicus(put(tmp_path, FITTER)).control_zyx[1].tolist() == [10, 20, 40]


@pytest.mark.parametrize("document,key", [
    ([[10, 20, 40], [0, 10, 20], [30, 20, 10]], "root array"),
    ({"points": [[10, 20, 40], {"z": 0, "y": 10, "x": 20}, [30, 20, 10]]}, "points"),
    ({"control_points": [[10, 20, 40], [0, 10, 20], {"x": 10, "y": 20, "z": 30}]}, "control_points"),
])
def test_vc3d_json_forms_only_when_declared(tmp_path, document, key):
    path = put(tmp_path, document)
    u = U.read_umbilicus(path, variant="vc3d-json")
    assert u.control_zyx.tolist() == [[0, 10, 20], [10, 20, 40], [30, 20, 10]], key
    exc = refusal(U.read_umbilicus, path)  # the fitter reads none of these
    assert exc.code == "IO_UMBILICUS_STRUCTURE"


def test_vc3d_text_and_thaumato_text_read_the_same_syntax_differently(tmp_path):
    text = "# z, y, x\n\n10, 20, 40\n 0,10,20 \n30, 20, 10\n"
    u = U.read_umbilicus(put(tmp_path, text, "umbilicus.txt"), variant="vc3d-text")
    assert u.control_zyx.tolist() == [[0, 10, 20], [10, 20, 40], [30, 20, 10]]
    thaumato = U.read_umbilicus(put(tmp_path, "510, 520, 540\n500,600,700\n", "u.txt"), variant="thaumato-text")
    assert thaumato.control_zyx.tolist() == [[20, 10, 40], [100, 0, 200]]  # y=a-500, z=b-500, x=c-500


def test_comma_text_without_a_declared_variant_is_refused(tmp_path):
    for name in ("umbilicus.txt", "umbilicus.json"):
        path = put(tmp_path, "10, 20, 40\n0, 10, 20\n", name)
        for variant in ("fitter", "vc3d-json"):
            exc = refusal(U.read_umbilicus, path, variant=variant)
            assert exc.code == "IO_UMBILICUS_TEXT_UNDECLARED" and exc.details == {"path": path}
            assert "vc3d-text" in exc.reason and "thaumato-text" in exc.reason


def test_metadata_block_is_parsed_and_checked_against_the_frame(tmp_path):
    meta = {"voxelsize_um": 2.4, "volume": "20990101000004", "volume_width": 50, "volume_height": 48,
            "volume_slices": 64, "tool": "vc_gen_umbilicus"}
    u = U.read_umbilicus(put(tmp_path, with_meta(**meta)), frame=FRAME)
    assert u.metadata == meta and u.findings == ()


def test_downsampled_stamp_matches_the_frame_through_coordinate_scale(tmp_path):
    meta = {"voxelsize_um": 4.8, "volume_width": 25, "volume_height": 24, "volume_slices": 32}
    small = {"control_points": [{"z": 0, "y": 5, "x": 10}, {"z": 31, "y": 6, "x": 12}], "metadata": meta}
    u = U.read_umbilicus(put(tmp_path, small), coordinate_scale=2, frame=FRAME)
    assert u.control_zyx.tolist() == [[0, 10, 20], [62, 12, 24]]
    odd = VolumeFrame(volume_id="20990101000004", shape_zyx=(63, 47, 49), voxel_size_um=2.4, source="test")
    U.read_umbilicus(put(tmp_path, small), coordinate_scale=2, frame=odd)  # ceil(63/2) = 32 still describes it
    exc = refusal(U.read_umbilicus, put(tmp_path, small), coordinate_scale=2,
                  frame=VolumeFrame(volume_id="20990101000004", shape_zyx=(66, 48, 50), voxel_size_um=2.4,
                                    source="test"))
    assert exc.code == "IO_UMBILICUS_FRAME_MISMATCH" and exc.details["field"] == "volume_slices"


def test_a_finer_stamp_matches_the_frame_as_vc3d_tests_a_rescale(tmp_path):
    # VC3D's rescale test (ScrollUmbilicus.cpp:262-315) at factor 1/2: the frame's count is the floor or ceiling
    # of half the stamped one. 127 -> 63.5 -> 64 and 97 -> 48.5 -> 48 describe FRAME (64, 48, 50).
    meta = {"voxelsize_um": 1.2, "volume_width": 100, "volume_height": 97, "volume_slices": 127}
    fine = {"control_points": [{"z": 0, "y": 10, "x": 20}, {"z": 126, "y": 12, "x": 24}], "metadata": meta}
    u = U.read_umbilicus(put(tmp_path, fine), coordinate_scale=0.5, frame=FRAME)
    assert u.control_zyx.tolist() == [[0, 5, 10], [63, 6, 12]]
    for slices in (125, 130):  # 62.5 and 65 do not round to 64
        path = put(tmp_path, {**fine, "metadata": {**meta, "volume_slices": slices}})
        exc = refusal(U.read_umbilicus, path, coordinate_scale=0.5, frame=FRAME)
        assert exc.code == "IO_UMBILICUS_FRAME_MISMATCH"
        assert exc.details == {"path": path, "field": "volume_slices", "stamped": slices, "declared": 64}


def test_an_inexact_reciprocal_scale_is_tested_through_its_step(tmp_path):
    # 1/3 is not a double; VC3D divides by the integer step, and 1 / float(1/3) is exactly 3.0.
    meta = {"voxelsize_um": 0.8, "volume_width": 150, "volume_height": 144, "volume_slices": 191}
    fine = {"control_points": [{"z": 0, "y": 30, "x": 60}, {"z": 150, "y": 36, "x": 72}], "metadata": meta}
    U.read_umbilicus(put(tmp_path, fine), coordinate_scale=1 / 3, frame=FRAME)  # 191 / 3 = 63.67 -> 64
    path = put(tmp_path, {**fine, "metadata": {**meta, "volume_slices": 195}})  # 65
    exc = refusal(U.read_umbilicus, path, coordinate_scale=1 / 3, frame=FRAME)
    assert exc.code == "IO_UMBILICUS_FRAME_MISMATCH" and exc.details["stamped"] == 195


@pytest.mark.parametrize("meta,field", [
    ({"volume_width": 50, "volume_height": 48, "volume_slices": 63}, "volume_slices"),
    ({"volume_width": 50, "volume_height": 49, "volume_slices": 64}, "volume_height"),
    ({"volume_width": 51, "volume_height": 48, "volume_slices": 64}, "volume_width"),
    ({"voxelsize_um": 2.41}, "voxelsize_um"),
    ({"voxelsize_um": 4.8}, "voxelsize_um"),
])
def test_metadata_that_contradicts_the_frame_is_refused(tmp_path, meta, field):
    exc = refusal(U.read_umbilicus, put(tmp_path, with_meta(**meta)), frame=FRAME)
    assert exc.code == "IO_UMBILICUS_FRAME_MISMATCH" and exc.details["field"] == field


def test_voxel_size_tolerance_is_relative(tmp_path):
    U.read_umbilicus(put(tmp_path, with_meta(voxelsize_um=2.4 * (1 + 5e-7))), frame=FRAME)


def test_control_points_must_lie_inside_the_frame(tmp_path):
    exc = refusal(U.read_umbilicus, put(tmp_path, FITTER), coordinate_scale=3, frame=FRAME)
    assert exc.code == "IO_FRAME_POINTS_OUTSIDE" and exc.details["example_zyx"] == [0.0, 30.0, 60.0]
    assert exc.details["count"] == 3 and "umbilicus.json" in exc.details["what"]


def test_findings(tmp_path):
    u = U.read_umbilicus(put(tmp_path, "0,1,2\n3,4,5\n", "umbilicus.json"), variant="vc3d-text")
    assert [f["finding"] for f in u.findings] == ["extension-disagrees-with-variant"]
    u = U.read_umbilicus(put(tmp_path, FITTER, "umbilicus.txt"))
    assert [f["finding"] for f in u.findings] == ["extension-disagrees-with-variant"]
    u = U.read_umbilicus(put(tmp_path, with_meta(coordinate_scale=0.5, volume="s1_2um.zarr")), frame=FRAME)
    assert [f["finding"] for f in u.findings] == ["metadata-coordinate-scale-not-applied", "metadata-volume-differs"]
    assert u.findings[1]["volume"] == "s1_2um.zarr"
    assert U.read_umbilicus(put(tmp_path, with_meta(volume="s1_2um.zarr"))).findings == ()


def test_null_metadata_is_no_metadata(tmp_path):
    assert U.read_umbilicus(put(tmp_path, {**FITTER, "metadata": None}), frame=FRAME).metadata is None


def test_round_trip(tmp_path):
    meta = {"voxelsize_um": 2.4, "volume_width": 50, "volume_height": 48, "volume_slices": 64}
    first = U.read_umbilicus(put(tmp_path, with_meta(**meta)))
    digest = U.write_umbilicus(tmp_path / "out.json", first.control_zyx, first.metadata)
    U.write_umbilicus(tmp_path / "out2.json", first.control_zyx[::-1].copy(), first.metadata)
    assert (tmp_path / "out.json").read_bytes() == (tmp_path / "out2.json").read_bytes()
    assert digest == "sha256:" + hashlib.sha256((tmp_path / "out.json").read_bytes()).hexdigest()
    second = U.read_umbilicus(tmp_path / "out.json", frame=FRAME)
    assert np.array_equal(first.control_zyx, second.control_zyx) and second.metadata == meta
    text = (tmp_path / "out.json").read_text()
    assert text == json.dumps(json.loads(text), indent=4, sort_keys=True) + "\n"
    assert U.read_umbilicus(tmp_path / "out.json", variant="vc3d-json").control_zyx.tolist() == \
        first.control_zyx.tolist()


def test_inspect(tmp_path):
    summary = U.inspect_umbilicus(put(tmp_path, FITTER), coordinate_scale=0.5)
    assert summary["control_points"] == 3 and summary["z_span"] == [0.0, 15.0] and summary["axes"] == ["z", "y", "x"]
    canonical_dumps(summary)


# -------------------------------------------------------------------------------------------- refusals


def test_unknown_variant_and_missing_file(tmp_path):
    exc = refusal(U.read_umbilicus, put(tmp_path, FITTER), variant="vc3d")
    assert exc.code == "IO_UMBILICUS_VARIANT_UNKNOWN" and exc.details == {"variant": "vc3d"}
    for target, problem in ((tmp_path / "absent.json", "missing"), (tmp_path, "not a regular file (a directory)")):
        exc = refusal(U.read_umbilicus, target)
        assert exc.code == "IO_UMBILICUS_FILE_MISSING" and exc.details == {"path": str(target), "problem": problem}


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf"), True, "2", 10 ** 400])
def test_coordinate_scale_must_be_finite_positive(tmp_path, scale):
    exc = refusal(U.read_umbilicus, put(tmp_path, FITTER), coordinate_scale=scale)
    assert exc.code == "IO_UMBILICUS_SCALE_INVALID" and exc.details == {"value": repr(scale)}


@pytest.mark.parametrize("data,why", [
    (b"{", "invalid JSON"),
    (b'{"control_points": [], "control_points": []}', "duplicate key"),
    (b'{"control_points": [{"z": NaN, "y": 0, "x": 0}]}', "non-standard constant NaN"),
    (b'{"control_points": [{"z": 1e400, "y": 0, "x": 0}]}', "overflows"),
    (b'{"control_points": [{"z": ' + b"1" * 5000 + b', "y": 0, "x": 0}]}', "more than int() converts"),
    (b"[" * 100000 + b"]" * 100000, "nested deeper"),
    (b"\xff\xfe", "not UTF-8"),
    (b"1, 2\n", "invalid JSON"),  # two columns: not comma text either
    # These two read before apok_eval.jsonstrict: the surrogate as a str that cannot be encoded, the nesting as data.
    (json.dumps({**FITTER, "metadata": {"volume": "v\udc00"}}).encode(),
     "lone surrogate '\\udc00' in the string at $.metadata.volume"),
    (b'{"control_points": [{"z": 0, "y": 1, "x": 2, "estimates": ' + b"[" * 300 + b"]" * 300 + b"}, "
     b'{"z": 5, "y": 1, "x": 2}]}', "arrays and objects nested deeper than 256 levels"),
], ids=["syntax", "duplicate", "nan", "overflow", "huge-int", "deep-nesting", "utf8", "two-columns", "lone-surrogate",
        "deeper-than-256"])
def test_strict_json(tmp_path, data, why):
    path = put(tmp_path, data)
    exc = refusal(U.read_umbilicus, path, variant="vc3d-json")
    assert exc.code == "IO_UMBILICUS_JSON_INVALID" and set(exc.details) == {"path", "problem"}
    assert exc.details["path"] == path and why in exc.details["problem"]
    assert exc.reason == f"{path}: {exc.details['problem']}"


@pytest.mark.parametrize("variant", ["fitter", "vc3d-json"])
def test_points_and_control_points_together_are_ambiguous(tmp_path, variant):
    path = put(tmp_path, {**FITTER, "points": [[0, 1, 2], [3, 4, 5]]})
    exc = refusal(U.read_umbilicus, path, variant=variant)
    assert exc.code == "IO_UMBILICUS_AMBIGUOUS" and exc.details == {"path": path}


@pytest.mark.parametrize("variant,document,index", [
    ("fitter", [], None),
    ("fitter", {"points": [[0, 1, 2]]}, None),
    ("fitter", {"control_points": {"z": 1}}, None),
    ("vc3d-json", {"points": {"z": 1}}, None),
    ("vc3d-json", {"control_points": "x"}, None),
    ("vc3d-json", {"other": []}, None),
    ("vc3d-json", 7, None),
    ("fitter", {"control_points": [{"z": 0, "y": 1, "x": 2}, [1, 2, 3]]}, 1),
    ("fitter", {"control_points": [{"z": 0, "y": 1}]}, 0),
    ("vc3d-json", [[0, 1, 2], [1, 2, 3, 4]], 1),
    ("vc3d-json", [[0, 1, 2], [1, 2]], 1),
    ("vc3d-json", [[0, 1, 2], "1,2,3"], 1),
])
def test_structure(tmp_path, variant, document, index):
    exc = refusal(U.read_umbilicus, put(tmp_path, document), variant=variant)
    assert exc.code == "IO_UMBILICUS_STRUCTURE" and exc.details["index"] == index


@pytest.mark.parametrize("variant,text,line", [
    ("vc3d-text", "0, 1, 2\n1, 2\n", 2),
    ("vc3d-text", "0, 1, 2\n1, 2, 3, 4\n", 2),
    ("vc3d-text", "0, 1, 2\n1,, 2, 3\n", 2),
    ("vc3d-text", "0, 1, 2\n1, 2, x\n", 2),
    ("vc3d-text", "0, 1, 2 # trailing comment\n", 1),
    ("vc3d-text", "0, 1, 2\n1, 2, ٣\n", 2),
    ("thaumato-text", "500, 500, ٥\n", 1),
    ("thaumato-text", "500, 500, 500\n\n501, 501, 501\n", 2),
    ("thaumato-text", "# header\n500, 500, 500\n", 1),
    ("thaumato-text", "500, 500, 500.5\n", 1),
    ("thaumato-text", "500, 500\n", 1),
])
def test_text_syntax(tmp_path, variant, text, line):
    exc = refusal(U.read_umbilicus, put(tmp_path, text, "u.txt"), variant=variant)
    assert exc.code == "IO_UMBILICUS_TEXT_SYNTAX" and exc.details["line"] == line


def test_text_must_be_utf8(tmp_path):
    exc = refusal(U.read_umbilicus, put(tmp_path, b"0,1,2\n\xff\n", "u.txt"), variant="vc3d-text")
    assert exc.code == "IO_UMBILICUS_TEXT_SYNTAX" and exc.details["line"] is None


@pytest.mark.parametrize("variant,document,where,axis", [
    ("fitter", {"control_points": [{"z": 0, "y": 1, "x": 2}, {"z": "1", "y": 1, "x": 2}]}, {"index": 1}, "z"),
    ("fitter", {"control_points": [{"z": 0, "y": True, "x": 2}, {"z": 1, "y": 1, "x": 2}]}, {"index": 0}, "y"),
    ("fitter", {"control_points": [{"z": 0, "y": 1, "x": None}, {"z": 1, "y": 1, "x": 2}]}, {"index": 0}, "x"),
    ("vc3d-json", [[0, 1, 2], [1, 10 ** 400, 2]], {"index": 1}, "y"),
])
def test_values_must_be_finite_numbers(tmp_path, variant, document, where, axis):
    exc = refusal(U.read_umbilicus, put(tmp_path, document), variant=variant)
    assert exc.code == "IO_UMBILICUS_VALUE_INVALID" and exc.details == {"path": str(tmp_path / "umbilicus.json"),
                                                                        "axis": axis, **where}


def test_text_value_overflow_names_the_line(tmp_path):
    exc = refusal(U.read_umbilicus, put(tmp_path, "0, 1, 2\n1e400, 1, 2\n", "u.txt"), variant="vc3d-text")
    assert exc.code == "IO_UMBILICUS_VALUE_INVALID" and exc.details["line"] == 2 and exc.details["axis"] == "z"


def test_thaumato_integer_past_int_digit_limit_is_an_invalid_value(tmp_path):
    path = put(tmp_path, "1, " + "2" * 5000 + ", 3\n4, 5, 6\n", "u.txt")  # b is z (Thaumato is y, z, x)
    exc = refusal(U.read_umbilicus, path, variant="thaumato-text")
    assert exc.code == "IO_UMBILICUS_VALUE_INVALID" and exc.details == {"path": path, "axis": "z", "line": 1}
    padded = put(tmp_path, "0" * 4999 + "1, 0, 0\n4, 5, 6\n", "p.txt")  # a = 1 in 5000 characters
    assert U.read_umbilicus(padded, variant="thaumato-text").control_zyx.tolist() == [[-500, -499, -500],
                                                                                      [-495, -496, -494]]


@pytest.mark.parametrize("variant,content,name,where,axis", [
    ("fitter", {"control_points": [{"z": 1e300, "y": 1e300, "x": 1}, {"z": 2e300, "y": 1, "x": 1},
                                   {"z": 0, "y": 0, "x": 0}]}, "u.json", {"index": 0}, "z"),
    ("vc3d-json", [[0, 1e300, 0], [1, 0, 0]], "u.json", {"index": 0}, "y"),
    ("vc3d-text", "0, 0, 0\n1, 2, -1e300\n", "u.txt", {"line": 2}, "x"),
])
def test_a_scale_that_overflows_a_coordinate_is_refused(tmp_path, variant, content, name, where, axis):
    # Unchecked, two z values became +inf, slipped past the duplicate test (inf - inf is NaN) and gave NaN centres.
    path = put(tmp_path, content, name)
    exc = refusal(U.read_umbilicus, path, variant=variant, coordinate_scale=1e10)
    assert exc.code == "IO_UMBILICUS_VALUE_INVALID"
    assert exc.details == {"path": path, "axis": axis, "scale": 1e10, **where}
    assert "overflows a double" in exc.reason
    edge = U.read_umbilicus(path, variant=variant, coordinate_scale=1e7)  # 2e300 * 1e7 = 2e307 still fits
    assert np.isfinite(edge.control_zyx).all()


@pytest.mark.parametrize("points", [[], [{"z": 1, "y": 2, "x": 3}]])
def test_too_few_control_points(tmp_path, points):
    exc = refusal(U.read_umbilicus, put(tmp_path, {"control_points": points}))
    assert exc.code == "IO_UMBILICUS_TOO_FEW" and exc.details["count"] == len(points)


def test_duplicate_z_names_both_sources(tmp_path):
    document = {"control_points": [{"z": 5, "y": 0, "x": 0}, {"z": 1, "y": 0, "x": 0}, {"z": 5.0, "y": 1, "x": 1}]}
    exc = refusal(U.read_umbilicus, put(tmp_path, document))
    assert exc.code == "IO_UMBILICUS_DUPLICATE_Z"
    assert exc.details["z"] == 5.0 and exc.details["sources"] == [0, 2] and exc.details["source_kind"] == "index"
    exc = refusal(U.read_umbilicus, put(tmp_path, "0,1,2\n-0.0,1,2\n", "u.txt"), variant="vc3d-text")
    assert exc.code == "IO_UMBILICUS_DUPLICATE_Z" and exc.details["sources"] == [1, 2]
    assert exc.details["source_kind"] == "line"


@pytest.mark.parametrize("meta,field", [
    ([1], "metadata"),
    ("v", "metadata"),
    ({"voxelsize_um": 0}, "voxelsize_um"),
    ({"voxelsize_um": -2.4}, "voxelsize_um"),
    ({"voxelsize_um": "2.4"}, "voxelsize_um"),
    ({"voxelsize_um": True}, "voxelsize_um"),
    ({"voxelsize_um": None}, "voxelsize_um"),
    ({"voxelsize_um": 10 ** 400}, "voxelsize_um"),
    ({"volume": ""}, "volume"),
    ({"volume": 7}, "volume"),
    ({"volume_width": 50, "volume_height": 48, "volume_slices": 64.0}, "volume_slices"),
    ({"volume_width": 0, "volume_height": 48, "volume_slices": 64}, "volume_width"),
    ({"volume_width": 50, "volume_height": True, "volume_slices": 64}, "volume_height"),
    ({"volume_width": 50, "volume_height": 48}, "volume_dimensions"),
    ({"volume_slices": 64}, "volume_dimensions"),
])
def test_metadata_block_is_validated(tmp_path, meta, field):
    exc = refusal(U.read_umbilicus, put(tmp_path, {**FITTER, "metadata": meta}))
    assert exc.code == "IO_UMBILICUS_METADATA_INVALID" and exc.details["field"] == field


def test_partial_triplet_names_what_is_missing(tmp_path):
    exc = refusal(U.read_umbilicus, put(tmp_path, with_meta(volume_width=50)))
    assert exc.details["missing"] == ["volume_slices", "volume_height"]


@pytest.mark.parametrize("policy", ["Clamp", "nearest", None])
def test_out_of_range_policy_is_required_and_checked(tmp_path, policy):
    u = U.read_umbilicus(put(tmp_path, FITTER))
    exc = refusal(u.yx_at, [1.0], out_of_range=policy)
    assert exc.code == "IO_UMBILICUS_POLICY_UNKNOWN" and exc.details == {"policy": policy}
    with pytest.raises(TypeError):
        u.yx_at([1.0])  # no default policy


@pytest.mark.parametrize("z,problem,dtype,shape", [
    (np.nan, "not finite", "float64", [1]),
    ([0.0, np.inf], "not finite", "float64", [2]),
    ("1", "not numeric", "<U1", [1]),
    ([[1.0, 2.0]], "not a scalar or 1-D", "float64", [1, 2]),
    (True, "not numeric", "bool", [1]),
    (None, "not numeric", "object", [1]),
])
def test_queries_must_be_finite_numbers(tmp_path, z, problem, dtype, shape):
    u = U.read_umbilicus(put(tmp_path, FITTER))
    exc = refusal(u.yx_at, z, out_of_range="clamp")
    assert exc.code == "IO_UMBILICUS_QUERY_INVALID"
    assert exc.details == {"problem": problem, "dtype": dtype, "shape": shape} and problem in exc.reason


@pytest.mark.parametrize("control,metadata,code,field", [
    (np.array([[0.0, 1.0, 2.0], [1.0, np.nan, 2.0]]), None, "IO_UMBILICUS_WRITE_INVALID", "control_zyx"),
    (np.array([[0.0, 1.0], [1.0, 2.0]]), None, "IO_UMBILICUS_WRITE_INVALID", "control_zyx"),
    ([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]], None, "IO_UMBILICUS_WRITE_INVALID", "control_zyx"),
    (np.array([[True, False, True], [False, True, False]]), None, "IO_UMBILICUS_WRITE_INVALID", "control_zyx"),
    (np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]), {"voxelsize_um": float("nan")}, "IO_UMBILICUS_WRITE_INVALID",
     "metadata"),
    (np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]), {"volume_width": 3}, "IO_UMBILICUS_METADATA_INVALID",
     "volume_dimensions"),
])
def test_writer_refuses_what_it_could_not_read_back(tmp_path, control, metadata, code, field):
    exc = refusal(U.write_umbilicus, tmp_path / "o.json", control, metadata)
    assert exc.code == code and exc.details["field"] == field
    assert not (tmp_path / "o.json").exists()


def deep(depth: int) -> list:
    value: list = []
    for _ in range(depth):
        value = [value]
    return value


@pytest.mark.parametrize("metadata,problem", [({"note": 10 ** 5000}, "ValueError"),
                                              ({"note": deep(100000)}, "RecursionError"),
                                              ({"note": deep(300)}, "would not read back as strict JSON: arrays "
                                                                    "and objects nested deeper than 256 levels")],
                         ids=["huge-int", "deep-nesting", "deeper-than-the-reader-reads"])
def test_writer_refuses_metadata_the_reader_would_refuse(tmp_path, metadata, problem):
    exc = refusal(U.write_umbilicus, tmp_path / "o.json", np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]), metadata)
    assert exc.code == "IO_UMBILICUS_WRITE_INVALID" and exc.details["field"] == "metadata"
    assert problem in exc.details["problem"]
    assert not (tmp_path / "o.json").exists()


def test_the_depth_limit_applies_to_the_document_that_nests_the_metadata(tmp_path):
    # deep(n) is n + 1 levels and the document nests the metadata one level deeper still. The writer once checked
    # the metadata alone: metadata of exactly 256 levels passed, and the document then escaped as a StrictJsonError.
    control = np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]])
    U.write_umbilicus(tmp_path / "ok.json", control, {"note": deep(253)})  # 255 levels, 256 in the document
    assert U.read_umbilicus(tmp_path / "ok.json").metadata == {"note": deep(253)}
    exc = refusal(U.write_umbilicus, tmp_path / "o.json", control, {"note": deep(254)})  # 256 levels, 257 there
    assert exc.code == "IO_UMBILICUS_WRITE_INVALID" and exc.details["field"] == "metadata"
    assert exc.details["problem"] == "would not read back as strict JSON: arrays and objects nested deeper than 256 " \
                                     "levels"
    assert not (tmp_path / "o.json").exists()


def test_writer_refuses_too_few_and_duplicate_points(tmp_path):
    exc = refusal(U.write_umbilicus, tmp_path / "o.json", np.array([[0.0, 1.0, 2.0]]))
    assert exc.code == "IO_UMBILICUS_TOO_FEW"
    exc = refusal(U.write_umbilicus, tmp_path / "o.json", np.array([[0.0, 1.0, 2.0], [0.0, 2.0, 3.0]]))
    assert exc.code == "IO_UMBILICUS_DUPLICATE_Z"


def test_writer_never_clobbers(tmp_path):
    target = tmp_path / "o.json"
    target.write_text("precious")
    control = np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]])
    exc = refusal(U.write_umbilicus, target, control)
    assert exc.code == "IO_UMBILICUS_WRITE_EXISTS" and exc.details == {"path": str(target)}
    assert target.read_text() == "precious"
    U.write_umbilicus(target, control, overwrite=True)
    assert U.read_umbilicus(target).control_zyx.tolist() == control.tolist()


@pytest.mark.parametrize("slices", [10 ** 400, 2 ** 53 + 1])
def test_a_volume_dimension_beyond_2_to_the_53_is_invalid_metadata_not_an_overflow(tmp_path, slices):
    meta = {"volume_width": 20, "volume_height": 20, "volume_slices": slices}
    path = put(tmp_path, with_meta(**meta))
    exc = refusal(U.read_umbilicus, path, coordinate_scale=0.5, frame=FRAME)
    assert exc.code == "IO_UMBILICUS_METADATA_INVALID" and exc.details == {"path": path, "field": "volume_slices"}
    U.read_umbilicus(put(tmp_path, with_meta(volume_width=2 ** 53, volume_height=20, volume_slices=20)))


def test_a_metadata_volume_naming_another_volume_id_is_refused(tmp_path):
    path = put(tmp_path, with_meta(volume="20230205180739"))
    exc = refusal(U.read_umbilicus, path, frame=FRAME)
    assert exc.code == "IO_FRAME_VOLUME_MISMATCH" and exc.details["field"] == "volume"
    assert exc.details["stated"] == "20230205180739" and exc.details["declared"] == FRAME.volume_id
    assert U.read_umbilicus(put(tmp_path, with_meta(volume=FRAME.volume_id)), frame=FRAME).findings == ()
    assert U.read_umbilicus(put(tmp_path, with_meta(volume="20230205180739"))).findings == ()  # no frame declared
