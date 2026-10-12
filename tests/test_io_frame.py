"""The shared coordinate contract: frame declarations and the in-volume check."""

from __future__ import annotations

import numpy as np
import pytest

from apok_eval.io.frame import VolumeFrame, check_points_within
from apok_eval.refusal import Refusal

GOOD = dict(volume_id="20990101000004", shape_zyx=(100, 50, 40), voxel_size_um=2.4, source="test")


def test_valid_frame_round_trips_to_dict():
    f = VolumeFrame(**GOOD)
    assert f.as_dict() == {"volume_id": "20990101000004", "shape_zyx": [100, 50, 40], "voxel_size_um": 2.4,
                           "axes": ["z", "y", "x"], "source": "test"}


@pytest.mark.parametrize(
    "field,value",
    [
        ("volume_id", "2099010100000"), ("volume_id", "PHercParis4"), ("volume_id", 20990101000004),
        ("shape_zyx", (100, 50)), ("shape_zyx", [100, 50, 40]), ("shape_zyx", (100, 0, 40)),
        ("shape_zyx", (100, True, 40)), ("shape_zyx", (100.0, 50, 40)),
        ("voxel_size_um", 0), ("voxel_size_um", -1.0), ("voxel_size_um", float("nan")),
        ("voxel_size_um", True), ("voxel_size_um", "2.4"),
        ("source", ""), ("source", "  "),
    ],
)
def test_malformed_frames_are_refused_naming_the_field(field, value):
    with pytest.raises(Refusal) as exc:
        VolumeFrame(**{**GOOD, field: value})
    assert exc.value.code == "IO_FRAME_INVALID"
    assert exc.value.details["field"] == field


def test_points_inside_pass_and_boundaries_are_half_open():
    f = VolumeFrame(**GOOD)
    check_points_within(f, np.array([[0, 0, 0], [99.999, 49.5, 39.0]]), "fixture")
    check_points_within(f, np.zeros((0, 3)), "empty")
    for bad in ([100, 0, 0], [0, 50, 0], [0, 0, 40], [-0.001, 0, 0]):
        with pytest.raises(Refusal) as exc:
            check_points_within(f, np.array([[1, 1, 1], bad], dtype=float), "fixture")
        assert exc.value.code == "IO_FRAME_POINTS_OUTSIDE"
        assert exc.value.details["count"] == 1 and exc.value.details["example_zyx"] == [float(c) for c in bad]


@pytest.mark.parametrize("pts", [np.zeros((3, 2)), np.zeros(3), np.array([[1, np.nan, 1]]), np.array([["a", "b", "c"]])])
def test_malformed_point_arrays_are_refused(pts):
    with pytest.raises(Refusal) as exc:
        check_points_within(VolumeFrame(**GOOD), pts, "fixture")
    assert exc.value.code == "IO_FRAME_POINTS_MALFORMED"


@pytest.mark.parametrize("field,value", [
    ("voxel_size_um", 10 ** 400), ("voxel_size_um", -(10 ** 400)), ("voxel_size_um", 2 ** 53 + 1),
    ("shape_zyx", (10 ** 400, 1, 1)), ("shape_zyx", (2 ** 60, 1, 1)), ("shape_zyx", (1, 1, 2 ** 53 + 1)),
])
def test_integers_a_float64_cannot_hold_exactly_are_refused_not_an_overflow(field, value):
    with pytest.raises(Refusal) as exc:
        VolumeFrame(**{**GOOD, field: value})
    assert exc.value.code == "IO_FRAME_INVALID" and exc.value.details["field"] == field
    assert "2**53" in exc.value.reason


def test_extents_up_to_2_to_the_53_are_frames_and_compare_exactly():
    frame = VolumeFrame(**{**GOOD, "shape_zyx": (2 ** 53, 1, 1), "voxel_size_um": 2 ** 53})
    check_points_within(frame, np.array([[2.0 ** 53 - 1, 0, 0]]), "edge")


def test_from_dict_reads_the_frame_file_schema():
    doc = {**{k: v for k, v in GOOD.items() if k != "shape_zyx"}, "shape_zyx": [100, 50, 40], "axes": ["z", "y", "x"]}
    assert VolumeFrame.from_dict(doc, where="t") == VolumeFrame(**GOOD)
    without_source = {k: v for k, v in doc.items() if k != "source"}
    assert VolumeFrame.from_dict(without_source, where="t", source="run:/frame").source == "run:/frame"
    for document, details in (
        (doc, {"field": "keys", "missing": [], "unknown": ["source"]}),  # a source given twice
        (without_source, {"field": "keys", "missing": ["source"], "unknown": []}),
    ):
        with pytest.raises(Refusal) as exc:
            VolumeFrame.from_dict(document, where="t", **({"source": "run:/frame"} if document is doc else {}))
        assert exc.value.code == "IO_FRAME_INVALID" and exc.value.details == {"declaration": "t", **details}


def test_the_identity_an_artefact_states_is_compared_with_the_frame():
    from apok_eval.io.frame import stated_identity

    frame = VolumeFrame(**GOOD)
    assert stated_identity(frame, {}, "t") == {"volume": "not stated by the artefact",
                                               "target_volume": "not stated by the artefact",
                                               "base_shape_zyx": "not stated by the artefact"}
    agrees = {"volume": "20990101000004", "target_volume": "20990101000004-2.4um", "base_shape_zyx": [100, 50, 40]}
    assert set(stated_identity(frame, agrees, "t").values()) == {"agrees"}
    for key, value in (("volume", "20990101000005"), ("target_volume", "20990101000005.zarr"),
                       ("base_shape_zyx", [100, 50, 41])):
        with pytest.raises(Refusal) as exc:
            stated_identity(frame, {**agrees, key: value}, "t")
        assert exc.value.code == "IO_FRAME_VOLUME_MISMATCH" and exc.value.details["field"] == key
    loose = stated_identity(frame, {"volume": "s1.zarr", "target_volume": "209901010000041",
                                    "base_shape_zyx": [100, 50, True]}, "t")
    assert all(v.startswith("not comparable") for v in loose.values())  # a 15-digit name is no 14-digit id
