"""``apok-eval io inspect``: explicit detection, the coordinate contract, and the wiring of the apok_eval.io package.

Every input is generated in tmp_path; nothing is fetched. Each summary is compared with what the module's own
public function returns for the same file, so the command adds detection and the frame, nothing else.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile
from io_zarr_fixtures import edit_json, write_small

import apok_eval
from apok_eval.cli import main
from apok_eval.io import cli as inspect_cli
from apok_eval.io.frame import VolumeFrame
from apok_eval.io.mesh import TriangleMesh, inspect_mesh, write_ply
from apok_eval.io.omezarr import open_volume
from apok_eval.io.pointcollection import inspect_point_collections
from apok_eval.io.store import LocalStore
from apok_eval.io.tifxyz import POLICIES, inspect_tifxyz, write_tifxyz
from apok_eval.io.umbilicus import inspect_umbilicus
from apok_eval.refusal import Refusal

SMALL = VolumeFrame("20241024131838", (9, 7, 5), 7.91, "test:small")  # write_small's level-0 shape
META = {"format": "tifxyz", "type": "seg", "uuid": "inspect-surface", "scale": [0.05, 0.05]}


# --- running the command -----------------------------------------------------------------------


def run(capsys, *argv) -> tuple[int, str]:
    rc = main(["io", "inspect", *map(str, argv)])
    return rc, capsys.readouterr().out


def record(capsys, *argv) -> dict:
    rc, out = run(capsys, *argv, "--json")
    assert rc == 0, out
    return json.loads(out)


def refused(capsys, code: str, *argv) -> dict:
    rc, out = run(capsys, *argv)
    assert rc == 3, out
    refusal = json.loads(out)
    assert refusal["code"] == code, refusal
    return refusal


def as_json(value):
    return json.loads(json.dumps(value))


# --- inputs ---------------------------------------------------------------------------------------


def tifxyz_dir(tmp_path, *, outside=False, mask=None, meta=None) -> Path:
    rows, cols = np.mgrid[0:2, 0:3]
    z = (1 + rows * 3 + cols).astype(np.float32)  # every vertex valid under every policy
    if outside:
        z[1, 2] = 20.0  # beyond the frame's z extent 9
    path = tmp_path / "surface"
    write_tifxyz(path, x=cols.astype(np.float32), y=rows.astype(np.float32), z=z, meta=meta or META)
    if mask is not None:
        tifffile.imwrite(path / "mask.tif", mask)
    return path


def zarr_dir(tmp_path) -> Path:
    return write_small(tmp_path / "vol.zarr")


def obj_file(tmp_path, *, outside=False) -> Path:
    path = tmp_path / "mesh.obj"
    path.write_text(f"v 1 2 3\nv 2 3 4\nv {7 if outside else 3} 4 4\nf 1 2 3\n")  # x y z; x 7 is beyond 5
    return path


def ply_file(tmp_path) -> Path:
    mesh = TriangleMesh(vertices_xyz=np.array([[1, 2, 3], [2, 3, 4], [3, 4, 4]], dtype=np.float32),
                        faces=np.array([[0, 1, 2]], dtype=np.int64))
    path = tmp_path / "mesh.PLY"
    write_ply(path, mesh)
    return path


def put_json(tmp_path, document, name: str) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def pcl_file(tmp_path, *, outside=False) -> Path:
    points = {"1": {"p": [1, 2, 3], "wind_a": 1.5}, "2": {"p": [9 if outside else 2, 3, 4]}}
    return put_json(tmp_path, {"vc_pointcollections_json_version": "1",
                               "collections": {"1": {"name": "a", "points": points}}}, "points.json")


def umbilicus_file(tmp_path, *, outside=False) -> Path:
    return put_json(tmp_path, {"control_points": [{"z": 0, "y": 3, "x": 2},
                                                  {"z": 9 if outside else 8, "y": 3, "x": 2}]}, "umbilicus.json")


def frame_file(tmp_path, document=None, name="frame.json") -> Path:
    return put_json(tmp_path, SMALL.as_dict() if document is None else document, name)


# --- detection: each kind, summarised by its own module ---------------------------------------


def test_a_tifxyz_directory_is_summarised_as_inspect_tifxyz_does(tmp_path, capsys):
    path = tifxyz_dir(tmp_path)
    out = record(capsys, path)
    assert out["schema"] == "apok-io-inspect-v1" and out["path"] == str(path) and out["kind"] == "tifxyz"
    assert out["detected_by"] == ["x.tif", "y.tif", "z.tif"]
    assert out["read_as"] == {"policies": list(POLICIES)}
    assert out["summary"] == as_json(inspect_tifxyz(path))
    assert out["frame"] == {"checked": False, "status": "NOT CHECKED",
                            "why": "no --frame given: the coordinates were compared with no declared volume"}


def test_policy_restricts_a_tifxyz_summary_to_that_policy(tmp_path, capsys):
    path = tifxyz_dir(tmp_path)
    out = record(capsys, path, "--policy", "lasagna")
    assert out["read_as"] == {"policies": ["lasagna"]}
    assert out["summary"] == as_json(inspect_tifxyz(path, ("lasagna",)))
    assert list(out["summary"]["validity"]["policies"]) == ["lasagna"]


def test_a_refusing_policy_without_a_frame_is_reported_as_inspect_tifxyz_reports_it(tmp_path, capsys):
    path = tifxyz_dir(tmp_path, mask=np.ones((2, 3), dtype=np.uint8))  # 0/1 mask: apok-strict refuses
    out = record(capsys, path, "--policy", "apok-strict")
    assert out["summary"]["refused_policies"]["apok-strict"]["code"] == "IO_TIFXYZ_MASK_VALUES"


@pytest.mark.parametrize("make,markers", [(obj_file, ["extension .obj"]), (ply_file, ["extension .ply"])])
def test_obj_and_ply_files_are_meshes_by_extension_in_any_case(tmp_path, capsys, make, markers):
    path = make(tmp_path)
    out = record(capsys, path)
    assert (out["kind"], out["detected_by"], out["read_as"]) == ("mesh", markers, {})
    assert out["summary"] == as_json(inspect_mesh(path))
    assert out["frame"]["status"] == "NOT CHECKED"


def test_a_point_collections_file_is_read_role_less_and_says_so(tmp_path, capsys):
    path = pcl_file(tmp_path)
    out = record(capsys, path)
    assert out["kind"] == "point-collections"
    assert out["detected_by"] == ["key vc_pointcollections_json_version", "key collections"]
    assert out["read_as"] == {"role": None}
    assert out["summary"] == as_json(inspect_point_collections(path, role=None))
    assert out["summary"]["declared_role"] is None and out["summary"]["role"] == "legacy"


@pytest.mark.parametrize("document,markers,variant", [
    ({"control_points": [{"z": 0, "y": 3, "x": 2}, {"z": 8, "y": 3, "x": 2}]}, ["key control_points"], "fitter"),
    ({"points": [[0, 3, 2], [8, 3, 2]]}, ["key points"], "vc3d-json"),
    ([[0, 3, 2], [8, 3, 2]], ["top-level array"], "vc3d-json"),
])
def test_an_umbilicus_is_read_under_the_variant_its_structure_names(tmp_path, capsys, document, markers, variant):
    path = put_json(tmp_path, document, "umbilicus.json")
    out = record(capsys, path)
    assert (out["kind"], out["detected_by"]) == ("umbilicus", markers)
    assert out["read_as"] == {"variant": variant, "coordinate_scale": None}
    assert out["summary"] == as_json(inspect_umbilicus(path, variant=variant))
    assert out["summary"]["variant"] == variant


def test_control_points_given_as_arrays_are_read_as_the_fitter_reads_them_and_refused(tmp_path, capsys):
    path = put_json(tmp_path, {"control_points": [[0, 3, 2], [8, 3, 2]]}, "umbilicus.json")  # VC3D would read it
    assert refused(capsys, "IO_UMBILICUS_STRUCTURE", path)["details"]["index"] == 0


def test_both_umbilicus_keys_reach_the_reader_which_refuses_them_as_ambiguous(tmp_path, capsys):
    path = put_json(tmp_path, {"points": [[0, 3, 2], [8, 3, 2]], "control_points": []}, "umbilicus.json")
    refused(capsys, "IO_UMBILICUS_AMBIGUOUS", path)


def test_a_json_extension_in_any_case_is_read(tmp_path, capsys):
    path = put_json(tmp_path, [[0, 3, 2], [8, 3, 2]], "UMBILICUS.JSON")
    assert record(capsys, path)["kind"] == "umbilicus"


def test_a_marker_only_picks_the_reader_which_names_what_is_wrong(tmp_path, capsys):
    path = tifxyz_dir(tmp_path)
    (path / "y.tif").unlink()
    assert refused(capsys, "IO_TIFXYZ_FILE_MISSING", path)["details"]["file"] == "y.tif"


def test_a_dangling_link_counts_as_a_marker(tmp_path, capsys):
    (tmp_path / "s").mkdir()
    os.symlink(tmp_path / "missing.tif", tmp_path / "s" / "x.tif")  # detected as tifxyz, then the reader judges it
    assert refused(capsys, "IO_TIFXYZ_FILE_MISSING", tmp_path / "s")["details"]["file"] == "meta.json"
    (tmp_path / "v").mkdir()
    os.symlink(tmp_path / "v" / "gone", tmp_path / "v" / ".zgroup")
    assert refused(capsys, "IO_STORE_NOT_A_FILE", tmp_path / "v", "--no-frame")["details"]["key"] == ".zgroup"


# --- detection refuses rather than guesses ------------------------------------------------------


@pytest.mark.parametrize("uri", ["s3://apok-test-bucket/vol.zarr", "https://example.org/mesh.obj",
                                 "file:///tmp/vol.zarr", "HTTP://127.0.0.1:8/x"])
def test_a_remote_inspect_is_refused_before_anything_is_read(capsys, uri):
    refusal = refused(capsys, "IO_INSPECT_PATH_INVALID", uri)
    assert refusal["details"] == {"path": uri, "problem": "remote"}


def test_a_path_naming_nothing_is_refused(tmp_path, capsys):
    path = str(tmp_path / "absent.obj")
    assert refused(capsys, "IO_INSPECT_PATH_INVALID", path)["details"] == {"path": path, "problem": "missing"}


@pytest.mark.parametrize("entries", [[], ["zarr.json"], ["meta.json", "notes.txt"], ["x.tiff", "zgroup"]])
def test_a_directory_without_markers_is_unknown(tmp_path, capsys, entries):
    for name in entries:
        (tmp_path / name).write_text("{}")
    details = refused(capsys, "IO_INSPECT_UNKNOWN", tmp_path)["details"]
    assert details["path"] == str(tmp_path) and details["problem"] == "no marker"
    assert "x.tif" in details["looked_for"] and ".zgroup" in details["looked_for"]


def test_a_directory_with_markers_of_two_kinds_is_ambiguous(tmp_path, capsys):
    path = zarr_dir(tmp_path)
    (path / "z.tif").write_bytes(b"")
    details = refused(capsys, "IO_INSPECT_AMBIGUOUS", path)["details"]
    assert details["kinds"] == ["tifxyz", "ome-zarr-v2"]
    assert details["markers"] == {"tifxyz": ["z.tif"], "ome-zarr-v2": [".zgroup", ".zattrs"]}


@pytest.mark.parametrize("content,error", [
    (b"not json", "invalid JSON: Expecting value at line 1 column 1"),
    (b'{"control_points": [}', "invalid JSON: Expecting value at line 1 column 21"),
    (b"\xff\xfe{}", "not UTF-8 (invalid start byte at byte 0)"),
    (b"\xef\xbb\xbf[]", "starts with a byte-order mark (U+FEFF), which JSON text must not carry"),
    (b"", "invalid JSON: Expecting value at line 1 column 1"),
    (b"[" * 100_000, "arrays and objects nested deeper than the parser can recurse (the limit is 256 levels)"),
    # Python's json read these five, and the sniff handed each to a reader by its keys; they are not strict JSON,
    # so which reader they are for is not guessed.
    (b'{"collections": {}, "collections": {}}', "duplicate key 'collections'"),
    (b'{"control_points": [{"z": NaN, "y": 3, "x": 2}]}',
     "non-standard constant NaN (NaN, Infinity and -Infinity are not JSON)"),
    (b'{"control_points": [], "metadata": {"voxelsize_um": 1e400}}', "number 1e400 overflows a double"),
    (b'{"points": [], "note": "\\ud800"}',
     "lone surrogate '\\ud800' in the string at $.note: not Unicode text (it cannot be encoded as UTF-8)"),
    (b'{"collections": {}, "note": ' + b"[" * 300 + b"]" * 300 + b"}",
     "arrays and objects nested deeper than 256 levels"),
], ids=["text", "syntax", "utf-16", "bom", "empty", "deeper-than-the-parser", "duplicate-key", "nan", "overflow",
        "lone-surrogate", "deeper-than-256"])
def test_a_json_file_that_is_not_strict_json_is_unknown_naming_the_problem(tmp_path, capsys, content, error):
    path = tmp_path / "u.json"
    path.write_bytes(content)
    details = refused(capsys, "IO_INSPECT_UNKNOWN", path)["details"]
    assert details == {"path": str(path), "problem": "not strict JSON", "error": error}


@pytest.mark.parametrize("make", ["io-error", "permission"])
def test_a_json_file_that_cannot_be_read_is_unknown_naming_the_error(tmp_path, capsys, monkeypatch, make):
    path = tmp_path / "u.json"
    if make == "io-error":  # a regular file whose read fails: offset 0 of /proc/self/mem is unmapped (EIO)
        if not os.path.isfile("/proc/self/mem"):
            pytest.skip("needs Linux /proc/self/mem")
        os.symlink("/proc/self/mem", path)
        error = "OSError"
    else:  # a mode-000 file refuses a non-root reader; root ignores modes, so the read is made to refuse instead
        path.write_text("[]")

        def denied(file, *args, **kwargs):
            raise PermissionError(13, "Permission denied", str(file))

        monkeypatch.setattr(inspect_cli, "read_regular", denied)
        error = "PermissionError"
    details = refused(capsys, "IO_INSPECT_UNKNOWN", path)["details"]
    assert details == {"path": str(path), "problem": "unreadable", "error": error}


@pytest.mark.parametrize("document", [{}, {"format": "tifxyz", "scale": [1, 1]}, 5, "points", None,
                                      {"Points": [], "COLLECTIONS": {}}])
def test_a_json_file_without_markers_is_unknown(tmp_path, capsys, document):
    path = put_json(tmp_path, document, "thing.json")
    details = refused(capsys, "IO_INSPECT_UNKNOWN", path)["details"]
    assert details["problem"] == "no marker" and "control_points" in details["looked_for"]


@pytest.mark.parametrize("document,markers", [
    ({"collections": {}, "control_points": []}, {"point-collections": ["key collections"],
                                                 "umbilicus": ["key control_points"]}),
    ({"vc_pointcollections_json_version": "1", "points": []},
     {"point-collections": ["key vc_pointcollections_json_version"], "umbilicus": ["key points"]}),
])
def test_a_json_file_with_markers_of_two_kinds_is_ambiguous(tmp_path, capsys, document, markers):
    details = refused(capsys, "IO_INSPECT_AMBIGUOUS", put_json(tmp_path, document, "x.json"))["details"]
    assert details["kinds"] == ["point-collections", "umbilicus"] and details["markers"] == markers


@pytest.mark.parametrize("name", ["umbilicus.txt", "surface.tif", "mesh.stl", "noextension"])
def test_any_other_file_is_unknown_naming_its_extension(tmp_path, capsys, name):
    path = tmp_path / name
    path.write_text("0, 1, 2\n5, 1, 2\n")
    details = refused(capsys, "IO_INSPECT_UNKNOWN", path)["details"]
    assert details == {"path": str(path), "problem": f"extension {os.path.splitext(name)[1]!r}"}


@pytest.mark.parametrize("make", ["dangling", "fifo"])
def test_something_neither_a_file_nor_a_directory_is_unknown(tmp_path, capsys, make):
    path = tmp_path / "thing.obj"
    if make == "dangling":
        os.symlink(tmp_path / "missing.obj", path)
    else:
        os.mkfifo(path)
    details = refused(capsys, "IO_INSPECT_UNKNOWN", path)["details"]
    assert details == {"path": str(path), "problem": "not a file or directory"}


# --- OME-Zarr: a frame, or an explicit --no-frame ---------------------------------------------


def test_an_ome_zarr_volume_needs_a_frame_or_an_explicit_no_frame(tmp_path, capsys):
    path = zarr_dir(tmp_path)
    refusal = refused(capsys, "IO_INSPECT_FRAME_REQUIRED", path)
    assert refusal["details"] == {"path": str(path), "kind": "ome-zarr-v2"}
    assert "--frame" in refusal["reason"] and "--no-frame" in refusal["reason"]


def test_a_framed_ome_zarr_inspect_is_open_volume_describe_and_records_the_check(tmp_path, capsys):
    path = zarr_dir(tmp_path)
    frame = frame_file(tmp_path)
    out = record(capsys, path, "--frame", frame)
    assert (out["kind"], out["detected_by"]) == ("ome-zarr-v2", [".zgroup", ".zattrs"])
    assert out["read_as"] == {"pyramid_tolerance": "exact", "axes_declared": None}
    assert out["summary"] == as_json(open_volume(LocalStore(path), SMALL).describe())
    assert out["frame"] == {"checked": True, "file": str(frame), "declared": SMALL.as_dict(),
                            "rule": "level 0's stored shape equals the declared shape_zyx", "count": None}


def test_a_frame_that_disagrees_with_level_0_is_refused(tmp_path, capsys):
    other = VolumeFrame("20241024131838", (9, 7, 6), 7.91, "test:wrong-shape").as_dict()
    refusal = refused(capsys, "IO_ZARR_SHAPE_MISMATCH", zarr_dir(tmp_path), "--frame", frame_file(tmp_path, other))
    assert refusal["details"]["stored"] == [9, 7, 5] and refusal["details"]["declared"] == [9, 7, 6]


def test_no_frame_runs_every_other_check_and_says_the_frame_was_not_checked(tmp_path, capsys):
    path = zarr_dir(tmp_path)
    rc, out = run(capsys, path, "--no-frame", "--json")
    assert rc == 0 and inspect_cli._STAND_IN_ID not in out and "stand-in" not in out  # the stand-in never leaves
    out = json.loads(out)
    assert out["frame"] == {"checked": False, "status": "NOT CHECKED",
                            "why": "--no-frame given: level 0's shape was compared with no declared frame, and no "
                                   "volume id or voxel size is known"}
    framed = as_json(open_volume(LocalStore(path), SMALL).describe())
    assert out["summary"] == {**framed, "frame": "NOT CHECKED"}


def test_no_frame_still_refuses_a_damaged_pyramid(tmp_path, capsys):
    path = zarr_dir(tmp_path)
    edit_json(path / "1" / ".zarray", lambda meta: meta.update(shape=[5, 4, 2]))
    refusal = refused(capsys, "IO_ZARR_PYRAMID_SHAPE", path, "--no-frame")
    assert refusal["details"]["level"] == 1


def test_no_frame_still_needs_axes_from_the_store(tmp_path, capsys):
    path = write_small(tmp_path / "vol.zarr", axes=None)
    assert refused(capsys, "IO_ZARR_AXES_UNDECLARED", path, "--no-frame")["details"]["key"] == ".zattrs"


@pytest.mark.parametrize("shape", [[9, 7], [9, 7, 0], [True, 7, 5], [9.0, 7, 5], "9,7,5", None])
def test_no_frame_with_a_malformed_level_0_shape_is_refused_by_the_reader_not_by_a_stand_in(tmp_path, capsys, shape):
    path = zarr_dir(tmp_path)
    edit_json(path / "0" / ".zarray", lambda meta: meta.update(shape=shape))
    refusal = refused(capsys, "IO_ZARR_ARRAY_META", path, "--no-frame")
    assert refusal["details"]["key"] == "0/.zarray" and refusal["details"]["field"] == "shape"


def test_no_frame_with_level_0_metadata_absent_or_not_json_is_refused_by_the_reader(tmp_path, capsys):
    path = zarr_dir(tmp_path)
    (path / "0" / ".zarray").unlink()
    assert refused(capsys, "IO_ZARR_ARRAY_META", path, "--no-frame")["details"]["field"] == ".zarray"
    # Undecodable; nested past the parser's recursion limit; a duplicated key (the stand-in, reading with the
    # reader's parser, fails too and leaves the refusal to the reader).
    for content in (b"\xff not json", b"[" * 100_000, b'{"shape": [9, 7, 5], "shape": [9, 7, 5]}'):
        (path / "0" / ".zarray").write_bytes(content)
        assert refused(capsys, "IO_OBJECT_NOT_JSON", path, "--no-frame")["details"]["key"] == "0/.zarray"


def test_no_frame_keeps_the_readers_order_of_refusals(tmp_path, capsys):
    # The stand-in reads 0/.zarray first; a store refusal there must not overtake the group check.
    path = zarr_dir(tmp_path)
    edit_json(path / ".zgroup", lambda meta: meta.update(zarr_format=3))
    os.rename(path / "0", tmp_path / "level0-moved")
    os.symlink(path / "level0-gone", path / "0")  # dangling, inside the store root
    framed = refused(capsys, "IO_ZARR_GROUP", path, "--frame", frame_file(tmp_path))
    assert refused(capsys, "IO_ZARR_GROUP", path, "--no-frame")["details"] == framed["details"]
    (path / ".zgroup").write_text('{"zarr_format": 2}')
    assert refused(capsys, "IO_STORE_NOT_A_FILE", path, "--no-frame")["details"]["key"] == "0/.zarray"


# --- frames for the other kinds -----------------------------------------------------------------


def test_a_tifxyz_frame_check_needs_a_policy(tmp_path, capsys):
    path = tifxyz_dir(tmp_path)
    refusal = refused(capsys, "IO_INSPECT_OPTION_INVALID", path, "--frame", frame_file(tmp_path))
    assert refusal["details"] == {"path": str(path), "option": "--policy", "kind": "tifxyz"}


def test_a_tifxyz_frame_check_counts_the_vertices_the_policy_keeps(tmp_path, capsys):
    path = tifxyz_dir(tmp_path)
    frame = frame_file(tmp_path)
    out = record(capsys, path, "--frame", frame, "--policy", "apok-strict")
    assert out["frame"] == {"checked": True, "file": str(frame), "declared": SMALL.as_dict(),
                            "rule": "every vertex valid under the policy lies inside the declared level-0 shape",
                            "count": 6}
    assert out["read_as"] == {"policies": ["apok-strict"]}
    identity = out["summary"].pop("stated_identity")
    assert identity == {"volume": "not stated by the artefact", "target_volume": "not stated by the artefact",
                        "base_shape_zyx": "not stated by the artefact"}
    assert out["summary"] == as_json(inspect_tifxyz(path, ("apok-strict",)))


@pytest.mark.parametrize("meta, field", [
    ({"volume": "20230205180739"}, "volume"),
    ({"target_volume": "20230205180739-7.91um-54keV"}, "target_volume"),
    ({"base_shape_zyx": [9, 7, 6]}, "base_shape_zyx"),
])
def test_a_tifxyz_whose_meta_names_another_volume_is_refused(tmp_path, capsys, meta, field):
    path = tifxyz_dir(tmp_path, meta={**META, **meta})
    refusal = refused(capsys, "IO_FRAME_VOLUME_MISMATCH", path, "--frame", frame_file(tmp_path), "--policy", "vc3d")
    assert refusal["details"]["field"] == field and refusal["details"]["stated"] == meta[field]


def test_a_tifxyz_whose_meta_names_the_declared_volume_agrees(tmp_path, capsys):
    meta = {**META, "volume": SMALL.volume_id, "target_volume": f"{SMALL.volume_id}.zarr",
            "base_shape_zyx": list(SMALL.shape_zyx)}
    out = record(capsys, tifxyz_dir(tmp_path, meta=meta), "--frame", frame_file(tmp_path), "--policy", "vc3d")
    assert out["summary"]["stated_identity"] == {"volume": "agrees", "target_volume": "agrees",
                                                 "base_shape_zyx": "agrees"}
    meta = {**META, "volume": "s1_2um.zarr", "base_shape_zyx": "9x7x5"}
    out = record(capsys, tifxyz_dir(tmp_path / "other", meta=meta), "--frame", frame_file(tmp_path), "--policy", "vc3d")
    assert out["summary"]["stated_identity"]["volume"].startswith("not comparable")
    assert out["summary"]["stated_identity"]["base_shape_zyx"].startswith("not comparable")


def test_a_tifxyz_vertex_outside_the_frame_is_refused(tmp_path, capsys):
    path = tifxyz_dir(tmp_path, outside=True)
    refusal = refused(capsys, "IO_FRAME_POINTS_OUTSIDE", path, "--frame", frame_file(tmp_path), "--policy", "vc3d")
    assert refusal["details"]["count"] == 1 and refusal["details"]["example_zyx"] == [20.0, 1.0, 2.0]


def test_a_tifxyz_frame_check_under_a_refusing_policy_is_refused(tmp_path, capsys):
    path = tifxyz_dir(tmp_path, mask=np.ones((2, 3), dtype=np.uint8))
    refused(capsys, "IO_TIFXYZ_MASK_VALUES", path, "--frame", frame_file(tmp_path), "--policy", "apok-strict")


@pytest.mark.parametrize("make,count,what", [
    (obj_file, 3, "mesh mesh.obj"), (pcl_file, 2, "point collections points.json"),
    (umbilicus_file, 2, None),
])
def test_points_inside_the_frame_are_counted_and_outside_it_refused(tmp_path, capsys, make, count, what):
    frame = frame_file(tmp_path)
    inside = make(tmp_path)
    out = record(capsys, inside, "--frame", frame)
    assert out["frame"]["checked"] is True and out["frame"]["count"] == count
    assert out["frame"]["rule"] == inspect_cli.FRAME_RULES[out["kind"]]
    refusal = refused(capsys, "IO_FRAME_POINTS_OUTSIDE", make(tmp_path, outside=True), "--frame", frame)
    assert refusal["details"]["count"] == 1 and refusal["details"]["volume_id"] == SMALL.volume_id
    if what is not None:
        assert refusal["details"]["what"] == what


def test_an_umbilicus_frame_check_includes_its_metadata(tmp_path, capsys):
    document = {"control_points": [{"z": 0, "y": 3, "x": 2}, {"z": 8, "y": 3, "x": 2}],
                "metadata": {"voxelsize_um": 3.0}}
    refusal = refused(capsys, "IO_UMBILICUS_FRAME_MISMATCH", put_json(tmp_path, document, "u.json"),
                      "--frame", frame_file(tmp_path))
    assert refusal["details"]["field"] == "voxelsize_um"


def test_no_frame_on_another_kind_says_so(tmp_path, capsys):
    out = record(capsys, obj_file(tmp_path), "--no-frame")
    assert out["frame"]["why"].startswith("--no-frame given")


@pytest.mark.parametrize("make", [zarr_dir, obj_file, pcl_file, umbilicus_file])
def test_policy_applies_to_tifxyz_only(tmp_path, capsys, make):
    path = make(tmp_path)
    refusal = refused(capsys, "IO_INSPECT_OPTION_INVALID", path, "--policy", "vc3d", "--no-frame")
    assert refusal["details"]["option"] == "--policy" and refusal["details"]["path"] == str(path)
    assert refusal["details"]["kind"] != "tifxyz"


@pytest.mark.parametrize("argv", [["--frame", "f.json", "--no-frame"], ["--policy", "strictest"]])
def test_usage_errors_exit_2(tmp_path, capsys, argv):
    with pytest.raises(SystemExit) as exc:
        main(["io", "inspect", str(obj_file(tmp_path)), *argv])
    assert exc.value.code == 2


def test_a_frame_and_no_frame_together_are_refused_by_the_function_too(tmp_path):
    path = str(obj_file(tmp_path))
    with pytest.raises(Refusal) as exc:
        inspect_cli.inspect_path(path, frame_path=str(frame_file(tmp_path)), no_frame=True)
    assert exc.value.code == "IO_INSPECT_OPTION_INVALID"
    assert exc.value.details == {"path": path, "option": "--no-frame", "kind": None}


@pytest.mark.parametrize("policy", ["no-such-policy", "VC3D", ""])
def test_an_unknown_policy_is_refused_by_the_function_too(tmp_path, policy):
    path = str(tifxyz_dir(tmp_path))
    with pytest.raises(Refusal) as exc:
        inspect_cli.inspect_path(path, policy=policy)
    assert exc.value.code == "IO_INSPECT_OPTION_INVALID"
    assert exc.value.details == {"path": path, "option": "--policy", "value": policy}


def test_apok_io_without_a_subcommand_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as exit_:
        main(["io"])
    assert exit_.value.code == 2 and "usage: apok-eval io" in capsys.readouterr().err


# --- the frame file -----------------------------------------------------------------------------


def frame_doc(**changes) -> dict:
    doc = {k: v for k, v in SMALL.as_dict().items() if k != "axes"}
    doc.update(changes)
    return {k: v for k, v in doc.items() if v is not ...}


def test_a_frame_file_without_axes_is_accepted(tmp_path, capsys):
    out = record(capsys, obj_file(tmp_path), "--frame", frame_file(tmp_path, frame_doc()))
    assert out["frame"]["declared"] == SMALL.as_dict()


FRAME_HEAD = b'{"volume_id": "20241024131838", "shape_zyx": [9, 7, 5], '


@pytest.mark.parametrize("content,problem", [
    (None, "NotRegularFile: "),
    (b'{"volume_id": "20241024131838",',
     "invalid JSON: Expecting property name enclosed in double quotes at line 1 column 32"),
    (b'{"volume_id": "20241024131838", "volume_id": "20990101000001", "shape_zyx": [9, 7, 5], '
     b'"voxel_size_um": 7.91, "source": "s"}', "duplicate key 'volume_id'"),
    (FRAME_HEAD + b'"voxel_size_um": NaN, "source": "s"}',
     "non-standard constant NaN (NaN, Infinity and -Infinity are not JSON)"),
    (FRAME_HEAD + b'"voxel_size_um": 7.91, "source": "\\ud800"}',
     "lone surrogate '\\ud800' in the string at $.source: not Unicode text"),
    (b'{"volume_id": "20241024131838", "source": "\xff"}',
     "not UTF-8 (invalid start byte at byte 43)"),
    (b"\xef\xbb\xbf" + FRAME_HEAD + b'"voxel_size_um": 7.91, "source": "s"}',
     "starts with a byte-order mark (U+FEFF)"),
    (b"[" * 100_000, "arrays and objects nested deeper than the parser can recurse"),
    # Python's json read these two: 1e400 as infinity, for VolumeFrame to judge, and the nesting as a source.
    (FRAME_HEAD + b'"voxel_size_um": 1e400, "source": "s"}', "number 1e400 overflows a double"),
    (FRAME_HEAD + b'"voxel_size_um": 7.91, "source": ' + b"[" * 300 + b"]" * 300 + b"}",
     "arrays and objects nested deeper than 256 levels"),
], ids=["missing", "syntax", "duplicate-key", "nan", "lone-surrogate", "not-utf8", "bom", "deeper-than-the-parser",
        "overflow", "deeper-than-256"])
def test_a_frame_file_that_is_not_strict_json_is_refused_naming_the_problem(tmp_path, capsys, content, problem):
    path = tmp_path / "frame.json"
    if content is not None:
        path.write_bytes(content)
    refusal = refused(capsys, "IO_INSPECT_FRAME_INVALID", obj_file(tmp_path), "--frame", path)
    assert refusal["details"]["path"] == str(path) and refusal["details"]["problem"].startswith(problem)


@pytest.mark.parametrize("name,problem", [
    ("frame\x00.json", "ValueError: "),  # the message itself depends on the Python version
    ("frame\ud800.json", "UnicodeEncodeError: 'utf-8' codec can't encode character '\\ud800'"),
], ids=["nul", "lone-surrogate"])
def test_a_frame_path_open_refuses_is_refused_naming_the_problem(tmp_path, name, problem):
    # open() raises ValueError, not OSError, for these paths (argv cannot carry them; inspect_path can).
    path = str(tmp_path / name)
    with pytest.raises(Refusal) as info:
        inspect_cli.inspect_path(str(obj_file(tmp_path)), frame_path=path)
    assert info.value.code == "IO_INSPECT_FRAME_INVALID"
    assert info.value.details["path"] == path and info.value.details["problem"].startswith(problem)


@pytest.mark.parametrize("document,details", [
    ([SMALL.as_dict()], {"field": "frame", "problem": "not an object"}),
    (frame_doc(source=...), {"field": "keys", "missing": ["source"], "unknown": []}),
    (frame_doc(units="um"), {"field": "keys", "missing": [], "unknown": ["units"]}),
    (frame_doc(axes=["x", "y", "z"]), {"field": "axes", "value": ["x", "y", "z"]}),
    (frame_doc(axes="zyx"), {"field": "axes", "value": "zyx"}),
])
def test_a_frame_file_of_the_wrong_shape_is_refused_naming_the_fields(tmp_path, capsys, document, details):
    # The schema is VolumeFrame.from_dict's: the same declaration is refused the same way through the API.
    path = frame_file(tmp_path, document)
    refusal = refused(capsys, "IO_FRAME_INVALID", obj_file(tmp_path), "--frame", path)
    assert refusal["details"] == {"declaration": f"--frame {str(path)!r}", **details}


@pytest.mark.parametrize("change,field", [
    ({"volume_id": "2024"}, "volume_id"), ({"shape_zyx": [9, 7]}, "shape_zyx"), ({"shape_zyx": "975"}, "shape_zyx"),
    ({"shape_zyx": [9, 7, True]}, "shape_zyx"), ({"voxel_size_um": 0}, "voxel_size_um"), ({"source": " "}, "source"),
])
def test_frame_values_are_judged_by_volume_frame(tmp_path, capsys, change, field):
    path = frame_file(tmp_path, frame_doc(**change))
    assert refused(capsys, "IO_FRAME_INVALID", obj_file(tmp_path), "--frame", path)["details"]["field"] == field


@pytest.mark.parametrize("change,field", [
    ({"voxel_size_um": 10 ** 400}, "voxel_size_um"),  # too large for a float at all
    ({"voxel_size_um": -(10 ** 400)}, "voxel_size_um"),  # the magnitude counts, whatever the sign
    ({"voxel_size_um": 2 ** 53 + 1}, "voxel_size_um"),  # a float, but not exactly this one
    ({"shape_zyx": [10 ** 400, 7, 5]}, "shape_zyx"),
    ({"shape_zyx": [9, 7, 2 ** 53 + 1]}, "shape_zyx"),
])
def test_an_integer_float64_cannot_hold_exactly_is_refused_naming_the_field(tmp_path, capsys, change, field):
    path = frame_file(tmp_path, frame_doc(**change))
    refusal = refused(capsys, "IO_FRAME_INVALID", obj_file(tmp_path), "--frame", path)
    assert refusal["details"] == {"field": field, "value": change[field]}
    with pytest.raises(Refusal) as info:  # the same refusal through the Python API, never an OverflowError
        VolumeFrame.from_dict(frame_doc(**change), where="api")
    assert info.value.code == "IO_FRAME_INVALID" and info.value.details["field"] == field


def test_integers_up_to_2_to_the_53_are_frame_values(tmp_path, capsys):
    path = frame_file(tmp_path, frame_doc(shape_zyx=[2 ** 53, 7, 5], voxel_size_um=-(2 ** 53)))
    assert refused(capsys, "IO_FRAME_INVALID", obj_file(tmp_path), "--frame", path)["details"]["field"] == \
        "voxel_size_um"  # past the bound check: VolumeFrame judges the (negative) voxel size
    path = frame_file(tmp_path, frame_doc(shape_zyx=[2 ** 53, 7, 5], voxel_size_um=2 ** 53))
    out = record(capsys, obj_file(tmp_path), "--frame", path)
    assert out["frame"]["declared"]["shape_zyx"] == [2 ** 53, 7, 5] and out["frame"]["count"] == 3


def test_the_frame_file_is_read_only_after_detection_and_options_pass(tmp_path, capsys):
    absent = tmp_path / "no-frame-here.json"
    refused(capsys, "IO_INSPECT_UNKNOWN", tmp_path, "--frame", absent)
    refused(capsys, "IO_INSPECT_OPTION_INVALID", tifxyz_dir(tmp_path), "--frame", absent)


# --- text rendering -------------------------------------------------------------------------------


def test_the_text_rendering_names_the_kind_the_markers_and_the_frame_status(tmp_path, capsys):
    path = obj_file(tmp_path)
    rc, text = run(capsys, path)
    assert rc == 0
    lines = text.splitlines()
    assert lines[0] == f"apok-eval io inspect: mesh at {path}"
    assert lines[1] == "  detected by: extension .obj"
    assert "  frame: NOT CHECKED (no --frame given: the coordinates were compared with no declared volume)" in lines
    assert "    vertices: 3" in lines and "    faces: 1" in lines and '    coordinate_type: "double"' in lines
    assert "    source: <" in text  # nested values are counted, not printed; --json has them


def test_the_text_rendering_of_a_checked_frame_and_reading_choices(tmp_path, capsys):
    rc, text = run(capsys, pcl_file(tmp_path), "--frame", frame_file(tmp_path))
    assert rc == 0
    assert "  read as: role=null" in text
    assert (f"  frame: CHECKED against volume {SMALL.volume_id} (shape_zyx [9, 7, 5], 7.91 um, from "
            f"{tmp_path / 'frame.json'}): every point lies inside the declared level-0 shape; 2 checked") in text


def test_brief_values_print_scalars_and_short_flat_lists_and_count_the_rest():
    assert inspect_cli._brief(None) == "null" and inspect_cli._brief("a") == '"a"'
    assert inspect_cli._brief([1, 2, 3]) == "[1, 2, 3]" and inspect_cli._brief({"a": 1}) == '{"a": 1}'
    assert inspect_cli._brief(list(range(7))) == "<7 items>"
    assert inspect_cli._brief([[1], [2]]) == "<2 items>" and inspect_cli._brief({"a": {}}) == "<1 entries>"


# --- the package: every code registered, imports one way ------------------------------------------


def _python(code: str) -> str:
    src = str(Path(apok_eval.__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([src, os.environ["PYTHONPATH"]])}  # keeps the net guard
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                          check=True).stdout


def test_importing_apok_io_registers_every_io_refusal_code():
    owners = json.loads(_python(
        "import json, apok_eval.io\n"
        "from apok_eval.refusal import REGISTRY\n"
        "print(json.dumps(sorted({v.split(':')[0] for k, v in REGISTRY.items() if k.startswith('IO_')})))"))
    assert owners == ["apok_eval.io.cli", "apok_eval.io.frame", "apok_eval.io.mesh", "apok_eval.io.omezarr", "apok_eval.io.pointcollection",
                      "apok_eval.io.readlog", "apok_eval.io.store", "apok_eval.io.tifxyz", "apok_eval.io.umbilicus"]


@pytest.mark.parametrize("first", ["apok_eval.io.store", "apok_eval.io.frame", "apok_eval.io.cli", "apok_eval.io.omezarr"])
def test_any_io_module_imports_first_without_a_cycle(first):
    assert _python(f"import {first}, apok_eval.refusal; print('IO_INSPECT_UNKNOWN' in apok_eval.refusal.REGISTRY)") == "True\n"


def test_the_licence_gate_does_not_import_apok_io():
    loaded = _python("import sys, apok_eval.licence; print(sorted(m for m in sys.modules if m.startswith('apok_eval.io')))")
    assert loaded == "[]\n"
