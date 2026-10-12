"""VC3D point collections: lossless round trip, canonical ids, the fitter's role rule, and every refusal."""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

from apok_eval.canonical import canonical_dumps
from apok_eval.io import pointcollection as P
from apok_eval.io.frame import VolumeFrame
from apok_eval.refusal import Refusal


def doc(**collections) -> dict:
    return {"vc_pointcollections_json_version": "1", "collections": collections}


def collection(points: dict, name="c", **extra) -> dict:
    return {"name": name, "points": points, **extra}


def point(x, y, z, wind=None, **extra) -> dict:
    out = {"p": [x, y, z], **extra}
    if wind is not None:
        out["wind_a"] = wind
    return out


RICH = {
    "vc_pointcollections_json_version": "1",
    "vc_open_data_coordinate_space": "scroll",
    "vc_open_data_source_coordinate_level": 1,
    "collections": {
        "1": {
            "name": "strip",
            "color": [0.2, 0.8, 1.0],
            "metadata": {"winding_is_absolute": False},
            "tags": {"k": "v"},
            "windings_linked": [2],
            "anchor2d": [3.5, 4.5],
            "autoFillMode": 3,
            "autoFillConstant": 2.0,
            "points": {
                "2": {"p": [1.0, 2.0, 3.0], "creation_time": 1700000000000, "wind_a": None, "links": [10]},
                "10": {"p": [4, 5, 6], "creation_time": 1700000000001, "wind_a": 1, "fiber_dir": "h"},
                "1": {"p": [7.5, 8.25, 9.125], "wind_a": 0.1},
            },
        },
        "2": {"name": "other", "points": {"2": {"p": [0.0, 0.0, 1e-300], "wind_a": 2.5, "creation_time": -5}}},
    },
}


def write(tmp_path, document, name="relative_windings.json") -> str:
    path = tmp_path / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return str(path)


def refusal(fn, *args, **kwargs) -> Refusal:
    with pytest.raises(Refusal) as exc:
        fn(*args, **kwargs)
    return exc.value


# --------------------------------------------------------------------------------------------- reading


def test_reader_exposes_zyx_chain_order_and_defaults(tmp_path):
    pcl = P.read_point_collections(write(tmp_path, RICH), role="relative")
    first, second = pcl.collections
    assert (first.collection_id, first.name, first.point_ids) == (1, "strip", (1, 2, 10))
    assert first.zyx.tolist() == [[9.125, 8.25, 7.5], [3.0, 2.0, 1.0], [6.0, 5.0, 4.0]]
    assert np.isnan(first.wind_a[1]) and first.wind_a[0] == 0.1 and first.wind_a[2] == 1.0
    assert first.creation_time == (0, 1700000000000, 1700000000001)
    assert first.metadata == {"winding_is_absolute": False}
    assert second.creation_time == (-5,) and second.metadata is None
    digest = "sha256:" + hashlib.sha256((tmp_path / "relative_windings.json").read_bytes()).hexdigest()
    assert pcl.digest == digest
    assert pcl.identities() == [(digest, 1, 1), (digest, 1, 2), (digest, 1, 10), (digest, 2, 2)]
    assert pcl.points_zyx().shape == (4, 3)
    assert pcl.document == RICH


def test_round_trip_is_lossless_and_bytes_are_deterministic(tmp_path):
    first = P.read_point_collections(write(tmp_path, RICH), role="relative")
    d1 = P.write_point_collections(tmp_path / "a.json", first.document, role="relative")
    P.write_point_collections(tmp_path / "b.json", copy.deepcopy(first.document), role="relative")
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
    assert d1 == "sha256:" + hashlib.sha256((tmp_path / "a.json").read_bytes()).hexdigest()
    second = P.read_point_collections(tmp_path / "a.json", role="relative")
    assert second.document == RICH
    assert isinstance(second.document["collections"]["1"]["points"]["10"]["p"][0], int)
    assert second.document["collections"]["2"]["points"]["2"]["p"][2] == 1e-300
    for a, b in zip(first.collections, second.collections, strict=True):
        assert a.point_ids == b.point_ids and np.array_equal(a.zyx, b.zyx)
        assert np.array_equal(a.wind_a, b.wind_a, equal_nan=True)


def test_writer_sorts_keys_as_strings_with_indent_four():
    text = P.encode_point_collections(RICH).decode()
    assert text.endswith("}\n") and '\n    "collections": {' in text
    assert text.index('"10":') < text.index('"2":')  # "10" sorts before "2", as nlohmann's std::map does
    assert json.loads(text) == RICH
    assert text == json.dumps(RICH, indent=4, sort_keys=True, ensure_ascii=False) + "\n"


def test_unicode_names_survive(tmp_path):
    document = doc(**{"0": collection({"0": point(1, 2, 3)}, name="fibre é")})
    P.write_point_collections(tmp_path / "u.json", document, role=None)
    assert "é" in (tmp_path / "u.json").read_text(encoding="utf-8")
    assert P.read_point_collections(tmp_path / "u.json", role=None).collections[0].name == "fibre é"


@pytest.mark.parametrize("key", ["0", "7", "18446744073709551615"])
def test_canonical_ids_are_accepted(tmp_path, key):
    pcl = P.read_point_collections(write(tmp_path, doc(**{key: collection({key: point(1, 2, 3)})})), role=None)
    assert pcl.collections[0].collection_id == int(key) and pcl.collections[0].point_ids == (int(key),)


@pytest.mark.parametrize("key", ["-1", " 7", "7 ", "1_0", "07", "+1", "1.0", "", "18446744073709551616", "١",
                                 "1" + "0" * 20, pytest.param("1" * 5000, id="5000-digits")])
@pytest.mark.parametrize("kind", ["collection", "point"])
def test_non_canonical_ids_are_refused(tmp_path, key, kind):
    if kind == "collection":
        document, collection_id = doc(**{key: collection({"0": point(1, 2, 3)})}), None
    else:
        document, collection_id = doc(**{"3": collection({key: point(1, 2, 3)})}), 3
    exc = refusal(P.read_point_collections, write(tmp_path, document), role=None)
    assert exc.code == "IO_PCL_ID_NONCANONICAL"
    assert (exc.details["kind"], exc.details["key"], exc.details["collection_id"]) == (kind, key, collection_id)


@pytest.mark.parametrize("value", [1, "2", "1 ", None, 1.0, True])
def test_version_must_be_the_string_1(tmp_path, value):
    document = doc()
    if value is None:
        del document["vc_pointcollections_json_version"]
    else:
        document["vc_pointcollections_json_version"] = value
    exc = refusal(P.read_point_collections, write(tmp_path, document), role=None)
    assert exc.code == "IO_PCL_VERSION" and exc.details["value"] == value


@pytest.mark.parametrize("document,field,cid", [
    ([], "document", None),
    ({"vc_pointcollections_json_version": "1"}, "collections", None),
    ({"vc_pointcollections_json_version": "1", "collections": []}, "collections", None),
    (doc(**{"4": []}), "collection", 4),
    (doc(**{"4": {"points": {}}}), "name", 4),
    (doc(**{"4": {"name": 7, "points": {}}}), "name", 4),
    (doc(**{"4": {"name": "n"}}), "points", 4),
    (doc(**{"4": {"name": "n", "points": []}}), "points", 4),
    (doc(**{"4": {"name": "n", "points": {}, "metadata": []}}), "metadata", 4),
])
def test_structure(tmp_path, document, field, cid):
    exc = refusal(P.read_point_collections, write(tmp_path, document), role=None)
    assert exc.code == "IO_PCL_STRUCTURE" and exc.details["field"] == field
    assert exc.details.get("collection_id") == cid


@pytest.mark.parametrize("pt,field", [
    ([1, 2, 3], "point"),
    ({"wind_a": 1}, "p"),
    ({"p": [1, 2]}, "p"),
    ({"p": [1, 2, 3, 4]}, "p"),
    ({"p": [1, "2", 3]}, "p"),
    ({"p": [1, True, 3]}, "p"),
    ({"p": [1, None, 3]}, "p"),
    ({"p": [1, 10 ** 400, 3]}, "p"),
    ({"p": {"x": 1}}, "p"),
    ({"p": [1, 2, 3], "wind_a": "1"}, "wind_a"),
    ({"p": [1, 2, 3], "wind_a": False}, "wind_a"),
    ({"p": [1, 2, 3], "wind_a": [1]}, "wind_a"),
    ({"p": [1, 2, 3], "creation_time": 1.5}, "creation_time"),
    ({"p": [1, 2, 3], "creation_time": True}, "creation_time"),
    ({"p": [1, 2, 3], "creation_time": 2 ** 63}, "creation_time"),
    ({"p": [1, 2, 3], "creation_time": -(2 ** 63) - 1}, "creation_time"),
    ({"p": [1, 2, 3], "creation_time": None}, "creation_time"),
])
def test_points(tmp_path, pt, field):
    exc = refusal(P.read_point_collections, write(tmp_path, doc(**{"5": collection({"9": pt})})), role=None)
    assert exc.code == "IO_PCL_POINT_INVALID"
    assert (exc.details["collection_id"], exc.details["point_id"], exc.details["field"]) == (5, 9, field)


def test_int64_creation_time_bounds_are_accepted(tmp_path):
    pts = {"0": {"p": [1, 2, 3], "creation_time": 2 ** 63 - 1}, "1": {"p": [1, 2, 3], "creation_time": -(2 ** 63)}}
    pcl = P.read_point_collections(write(tmp_path, doc(**{"0": collection(pts)})), role=None)
    assert pcl.collections[0].creation_time == (2 ** 63 - 1, -(2 ** 63))


@pytest.mark.parametrize("text,why", [
    (b"{", "invalid JSON"),
    (b'{"vc_pointcollections_json_version": "1", "collections": {}, "collections": {}}', "duplicate key"),
    (b'{"vc_pointcollections_json_version": "1", "collections": {"0": {"name": "n", "points": '
     b'{"0": {"p": [1, 2, 3], "wind_a": NaN}}}}}', "non-standard constant NaN"),
    (b'{"vc_pointcollections_json_version": "1", "x": 1e999, "collections": {}}', "overflows"),
    (b'{"vc_pointcollections_json_version": "1", "x": "\xfe", "collections": {}}', "not UTF-8"),
    (b'{"vc_pointcollections_json_version": "1", "collections": {"0": {"name": "n", "points": '
     b'{"0": {"p": [1, 2, 3], "creation_time": ' + b"1" * 5000 + b'}}}}}', "more than int() converts"),
    (b'{"vc_pointcollections_json_version": "1", "x": ' + b"[" * 100000 + b"]" * 100000 + b', "collections": {}}',
     "nested deeper"),
    # These two read before apok_eval.jsonstrict: the surrogate as a str that cannot be encoded, the nesting as data.
    (b'{"vc_pointcollections_json_version": "1", "collections": {"0": {"name": "\\udc00", "points": {}}}}',
     "lone surrogate '\\udc00' in the string at $.collections[\"0\"].name"),
    (b'{"vc_pointcollections_json_version": "1", "x": ' + b"[" * 300 + b"]" * 300 + b', "collections": {}}',
     "arrays and objects nested deeper than 256 levels"),
], ids=["syntax", "duplicate", "nan", "overflow", "utf8", "huge-int", "deep-nesting", "lone-surrogate",
        "deeper-than-256"])
def test_strict_json(tmp_path, text, why):
    path = tmp_path / "f.json"
    path.write_bytes(text)
    exc = refusal(P.read_point_collections, path, role=None)
    assert exc.code == "IO_PCL_JSON_INVALID" and set(exc.details) == {"path", "problem"}
    assert exc.details["path"] == str(path) and why in exc.details["problem"]
    assert exc.reason == f"{path}: {exc.details['problem']}"


def test_missing_file_is_refused_not_skipped(tmp_path):
    for target, problem in ((tmp_path / "absent.json", "missing"), (tmp_path, "not a regular file (a directory)")):
        exc = refusal(P.read_point_collections, target, role=None)
        assert exc.code == "IO_PCL_FILE_MISSING" and exc.details == {"path": str(target), "problem": problem}
        assert ("does not exist" in exc.reason) == (problem == "missing")


# ---------------------------------------------------------------------------------------------- roles


@pytest.mark.parametrize("basename,declared,role,absolute", [
    ("abs_winding.json", None, "absolute", True),
    ("abs_windings.json", None, "legacy", False),
    ("relative_windings.json", None, "legacy", False),
    ("x.json", "absolute", "absolute", True),
    ("abs_winding.json", "relative", "relative", False),
    ("abs_winding.json", "same_winding", "same_winding", False),
    ("y.json", "drawn_control_points", "drawn_control_points", False),
])
def test_role_derivation_matches_the_fitter(basename, declared, role, absolute):
    assert P.derive_role(basename, declared) == (role, absolute)


@pytest.mark.parametrize("declared", ["abs", "", "Absolute", "legacy"])
def test_unknown_declared_role(tmp_path, declared):
    exc = refusal(P.read_point_collections, write(tmp_path, RICH), role=declared)
    assert exc.code == "IO_PCL_ROLE_UNKNOWN" and exc.details["role"] == declared


def test_metadata_flag_is_not_authoritative(tmp_path):
    flagged_true = doc(**{"1": collection({"0": point(1, 2, 3, 0)}, metadata={"winding_is_absolute": True})})
    pcl = P.read_point_collections(write(tmp_path, flagged_true, "relative_windings.json"), role=None)
    assert not pcl.absolute and pcl.role == "legacy"
    assert pcl.findings == ({"finding": "winding-flag-disagrees", "collection_id": 1, "flag": True,
                             "vc3d_reads_absolute": True, "role": "legacy"},)
    flagged_false = doc(**{"1": collection({"0": point(1, 2, 3, 3)}, metadata={"winding_is_absolute": False})})
    pcl = P.read_point_collections(write(tmp_path, flagged_false, "abs_winding.json"), role=None)
    assert pcl.absolute and pcl.annotations == ()
    assert pcl.findings[0]["finding"] == "winding-flag-disagrees" and pcl.findings[0]["flag"] is False


@pytest.mark.parametrize("metadata,flag", [(None, "absent"), ({}, "absent"), ({"winding_is_absolute": "yes"}, "yes"),
                                           ({"winding_is_absolute": 0}, 0)])
def test_absent_or_non_boolean_flag_reads_as_true_in_vc3d(tmp_path, metadata, flag):
    body = collection({"0": point(1, 2, 3)})
    if metadata is not None:
        body["metadata"] = metadata
    relative = P.read_point_collections(write(tmp_path, doc(**{"1": body}), "points.json"), role="relative")
    assert relative.findings == ({"finding": "winding-flag-disagrees", "collection_id": 1, "flag": flag,
                                  "vc3d_reads_absolute": True, "role": "relative"},)
    body["points"]["0"]["wind_a"] = 1
    absolute = P.read_point_collections(write(tmp_path, doc(**{"1": body}), "points.json"), role="absolute")
    assert absolute.findings == ()


@pytest.mark.parametrize("basename", ["abs_windings.json", "Abs_Winding.json", "abs-winding.json",
                                      "absolute_winding.json"])
def test_near_miss_basenames_are_flagged(tmp_path, basename):
    document = doc(**{"1": collection({"0": point(1, 2, 3, 0)}, metadata={"winding_is_absolute": False})})
    pcl = P.read_point_collections(write(tmp_path, document, basename), role=None)
    assert not pcl.absolute
    assert [f["finding"] for f in pcl.findings] == ["basename-near-miss"]
    assert pcl.findings[0]["basename"] == basename


def test_unrelated_basenames_are_not_near_misses(tmp_path):
    document = doc(**{"1": collection({"0": point(1, 2, 3, 0)}, metadata={"winding_is_absolute": False})})
    for basename in ("same_windings.json", "absorb.json", "winding_abs.json"):
        assert P.read_point_collections(write(tmp_path, document, basename), role=None).findings == ()


def test_declared_role_conflicting_with_a_conventional_name_is_flagged(tmp_path):
    document = doc(**{"1": collection({"0": point(1, 2, 3, 0)}, metadata={"winding_is_absolute": False})})
    pcl = P.read_point_collections(write(tmp_path, document, "abs_winding.json"), role="relative")
    assert pcl.findings == ({"finding": "role-conflicts-with-basename", "basename": "abs_winding.json",
                             "declared_role": "relative", "conventional_role": "absolute"},)
    assert P.read_point_collections(write(tmp_path, document, "relative_windings.json"), role="relative").findings == ()


@pytest.mark.parametrize("wind,expected", [(0, 0.0), (-1.5, -1.5), (None, None), (0.0, 0.0)])
def test_absolute_points_need_positive_finite_wind_a(tmp_path, wind, expected):
    pts = {"0": point(1, 2, 3, 2), "4": point(1, 2, 3, wind)}
    exc = refusal(P.read_point_collections, write(tmp_path, doc(**{"8": collection(pts)}), "abs_winding.json"),
                  role=None)
    assert exc.code == "IO_PCL_ABSOLUTE_WINDING"
    assert (exc.details["collection_id"], exc.details["point_id"], exc.details["wind_a"]) == (8, 4, expected)


def test_absolute_rule_applies_by_declared_role_and_not_to_relative_files(tmp_path):
    pts = {"0": point(1, 2, 3, 0)}
    path = write(tmp_path, doc(**{"8": collection(pts)}), "anything.json")
    assert refusal(P.read_point_collections, path, role="absolute").code == "IO_PCL_ABSOLUTE_WINDING"
    assert not P.read_point_collections(path, role="same_winding").absolute
    ok = write(tmp_path, doc(**{"8": collection({"0": point(1, 2, 3, 1e-9), "1": point(1, 2, 3, 7)})}),
               "abs_winding.json")
    assert P.read_point_collections(ok, role=None).absolute


def test_annotation_report_matches_the_fitter_normalisation(tmp_path):
    document = doc(**{
        "1": collection({"0": point(0, 0, 0, 0), "1": point(0, 0, 1, 1), "2": point(0, 0, 2, 2)}),
        "2": collection({"0": point(0, 0, 0), "1": point(0, 0, 1)}),
        "3": collection({"0": point(0, 0, 0, 3), "5": point(0, 0, 1), "9": point(0, 0, 2, 4), "12": point(0, 0, 3)}),
        "4": collection({}),
    })
    pcl = P.read_point_collections(write(tmp_path, document, "same_windings.json"), role="same_winding")
    assert list(pcl.annotations) == [
        {"collection_id": 1, "state": "all", "has_winding_annotations": True, "unannotated_point_ids": []},
        {"collection_id": 2, "state": "none", "has_winding_annotations": False, "unannotated_point_ids": [0, 1]},
        {"collection_id": 3, "state": "mixed", "has_winding_annotations": True, "unannotated_point_ids": [5, 12]},
        {"collection_id": 4, "state": "empty", "has_winding_annotations": False, "unannotated_point_ids": []},
    ]
    # Reported, not applied: the unannotated points are still there, still NaN.
    assert pcl.collections[2].point_ids == (0, 5, 9, 12) and np.isnan(pcl.collections[2].wind_a[1])


def test_point_findings(tmp_path):
    document = doc(**{
        "1": collection({"0": point(0, 0, 0, fiber_dir="h"), "1": point(0, 0, 1, fiber_dir="x")}),
        "2": collection({"0": point(0, 0, 0), "1": point(0, 0, 1), "7": point(0, 0, 1)}),
    })
    pcl = P.read_point_collections(write(tmp_path, document), role="relative")
    kinds = [f for f in pcl.findings if f["finding"] != "winding-flag-disagrees"]
    assert kinds == [
        {"finding": "fiber-dir-dropped-by-vc3d", "collection_id": 1, "point_id": 1, "value": "x"},
        {"finding": "point-ids-not-file-global", "count": 2, "example_point_id": 0,
         "detail": "identity is (file, collection id, point id)"},
    ]


def test_frame_check_uses_zyx(tmp_path):
    document = doc(**{"1": collection({"0": point(30, 2, 1), "1": point(1, 2, 3)})})
    pcl = P.read_point_collections(write(tmp_path, document), role="relative")
    frame = VolumeFrame(volume_id="20990101000004", shape_zyx=(4, 4, 31), voxel_size_um=7.91, source="test")
    assert P.check_point_collections_within(pcl, frame) == 2
    narrow = VolumeFrame(volume_id="20990101000004", shape_zyx=(4, 4, 30), voxel_size_um=7.91, source="test")
    exc = refusal(P.check_point_collections_within, pcl, narrow)
    assert exc.code == "IO_FRAME_POINTS_OUTSIDE" and exc.details["example_zyx"] == [1.0, 2.0, 30.0]
    empty = P.read_point_collections(write(tmp_path, doc()), role="relative")
    assert P.check_point_collections_within(empty, frame) == 0


def test_inspect(tmp_path):
    summary = P.inspect_point_collections(write(tmp_path, RICH), role=None)
    assert summary["role"] == "legacy" and summary["axes"] == ["z", "y", "x"]
    assert summary["collections"] == [{"id": 1, "name": "strip", "points": 3, "annotated": 2},
                                      {"id": 2, "name": "other", "points": 1, "annotated": 1}]
    assert summary["top_level_keys"] == sorted(RICH)
    canonical_dumps(summary)


# --------------------------------------------------------------------------------------------- writer


def deep(depth: int) -> list:
    value: list = []
    for _ in range(depth):
        value = [value]
    return value


@pytest.mark.parametrize("bad,problem", [
    ({**RICH, "extra": (1, 2)}, "tuple is not canonical JSON (at .extra)"),
    ({**RICH, "extra": float("nan")}, "non-finite float at .extra"),
    ({**RICH, 3: "non-string key"}, "non-string key 3 at $"),
    ({**RICH, "extra": 10 ** 5000}, "Exceeds the limit"),
    ({**RICH, "extra": deep(100000)}, "RecursionError"),
    ({**RICH, "extra": deep(300)}, "would not read back as strict JSON: arrays and objects nested deeper than 256"),
], ids=["tuple", "nan", "int-key", "huge-int", "deep-nesting", "deeper-than-the-reader-reads"])
def test_writer_refuses_non_json_values_naming_where(tmp_path, bad, problem):
    exc = refusal(P.write_point_collections, tmp_path / "o.json", bad, role=None)
    assert exc.code == "IO_PCL_WRITE_INVALID" and problem in exc.details["problem"]
    assert not (tmp_path / "o.json").exists()


def test_writer_refuses_what_the_reader_would_refuse(tmp_path):
    exc = refusal(P.write_point_collections, tmp_path / "o.json", {**RICH, "vc_pointcollections_json_version": 1},
                  role=None)
    assert exc.code == "IO_PCL_VERSION"
    exc = refusal(P.write_point_collections, tmp_path / "o.json", doc(**{"01": collection({})}), role=None)
    assert exc.code == "IO_PCL_ID_NONCANONICAL"
    assert not (tmp_path / "o.json").exists()


def test_writer_applies_the_absolute_rule_of_its_role(tmp_path):
    unannotated = doc(**{"0": collection({"0": point(1, 2, 3), "4": point(4, 5, 6, wind=2.0)})})
    target = tmp_path / "abs_winding.json"  # role-less and named so: absolute, as the reader derives it
    exc = refusal(P.write_point_collections, target, unannotated, role=None)
    assert exc.code == "IO_PCL_ABSOLUTE_WINDING"
    assert exc.details == {"path": str(target), "collection_id": 0, "point_id": 0, "wind_a": None}
    exc = refusal(P.write_point_collections, tmp_path / "w.json", unannotated, role="absolute")
    assert exc.code == "IO_PCL_ABSOLUTE_WINDING" and exc.details["point_id"] == 0
    exc = refusal(P.write_point_collections, tmp_path / "w.json", unannotated, role="Absolute")
    assert exc.code == "IO_PCL_ROLE_UNKNOWN" and exc.details == {"role": "Absolute"}
    assert sorted(p.name for p in tmp_path.iterdir()) == []
    P.write_point_collections(tmp_path / "w.json", unannotated, role="relative")  # relative files may be unannotated
    annotated = doc(**{"0": collection({"0": point(1, 2, 3, wind=1.0)})})
    P.write_point_collections(target, annotated, role=None)
    assert P.read_point_collections(target, role=None).absolute
    with pytest.raises(TypeError):
        P.write_point_collections(tmp_path / "x.json", annotated)  # no default role


def test_collections_are_ordered_by_numeric_id_whatever_the_file_order(tmp_path):
    document = doc(**{"10": collection({"3": point(1, 2, 3)}, name="ten"),
                      "2": collection({"1": point(4, 5, 6)}, name="two")})
    pcl = P.read_point_collections(write(tmp_path, document), role=None)
    assert [c.collection_id for c in pcl.collections] == [2, 10]
    assert [ident[1:] for ident in pcl.identities()] == [(2, 1), (10, 3)]
    assert pcl.points_zyx().tolist() == [[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]


def test_writer_never_clobbers(tmp_path):
    target = tmp_path / "o.json"
    target.write_text("precious")
    exc = refusal(P.write_point_collections, target, RICH, role=None)
    assert exc.code == "IO_PCL_WRITE_EXISTS" and exc.details["path"] == str(target)
    assert target.read_text() == "precious"
    P.write_point_collections(target, RICH, role=None, overwrite=True)
    assert json.loads(target.read_text()) == RICH
