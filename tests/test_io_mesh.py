"""Triangle meshes: OBJ and PLY round trips, the explicit xyz -> zyx exposure, and every refusal."""

from __future__ import annotations

import hashlib
import struct

import numpy as np
import pytest

from apok_eval.canonical import canonical_dumps
from apok_eval.io import mesh as M
from apok_eval.io.frame import VolumeFrame
from apok_eval.refusal import Refusal

OBJ_FULL = b"""# written by hand
mtllib none.mtl
o patch
v 1.5 2.0 3.25
v -4 5e2 .5
v 7.000000 8.000000 9.000000
v 0 0 1
vt 0.1 0.2
vt 0.3 0.4
vt 0.5 0.6
vn 0 0 1
vn 0 1 0
g surface
s off
usemtl stone
f 1/1/1 2/2/1 3/3/2
f 4/3/2 -3/-2/-1 -4/-3/-2
"""


def refusal(fn, *args, **kwargs) -> Refusal:
    with pytest.raises(Refusal) as exc:
        fn(*args, **kwargs)
    return exc.value


def obj(text: str) -> bytes:
    return text.encode()


def ply_binary(vertices, faces, *, vertex_props=("float x", "float y", "float z"), vfmt="<fff",
               face_prop="list uchar int vertex_indices", ffmt="<B3i", counts=None, extra=b"", header_extra=""):
    head = ["ply", "format binary_little_endian 1.0", f"element vertex {len(vertices)}"]
    head += [f"property {p}" for p in vertex_props]
    head += [f"element face {len(faces)}", f"property {face_prop}"]
    if header_extra:
        head.insert(2, header_extra)
    body = b"".join(struct.pack(vfmt, *v) for v in vertices)
    for i, f in enumerate(faces):
        n = counts[i] if counts else len(f)
        body += struct.pack(ffmt[:2] + f"{len(f)}" + ffmt[3:], n, *f)
    return ("\n".join(head + ["end_header"]) + "\n").encode() + body + extra


TETRA = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 3.0)]
TETRA_FACES = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]

PLY_ASCII = b"""ply
format ascii 1.0
comment made by hand
obj_info synthetic
element vertex 4
property float x
property float y
property float z
property uchar red
property double quality
element face 2
property list uchar int vertex_indices
end_header
0 0 0 255 0.5
1 0 0 0 -1e-3
0 2.5 0 7 2
0 0 3 9 1e10
3 0 1 2
3 1 2 3
"""


# --------------------------------------------------------------------------------------------- OBJ


def test_obj_reads_all_statements_and_resolves_negative_indices():
    mesh = M.parse_obj(OBJ_FULL, "full.obj")
    assert mesh.vertices_xyz.tolist() == [[1.5, 2.0, 3.25], [-4.0, 500.0, 0.5], [7.0, 8.0, 9.0], [0.0, 0.0, 1.0]]
    assert mesh.faces.tolist() == [[0, 1, 2], [3, 1, 0]]
    assert mesh.face_texcoords.tolist() == [[0, 1, 2], [2, 1, 0]]
    assert mesh.face_normals.tolist() == [[0, 0, 1], [1, 1, 0]]
    assert mesh.texcoords.shape == (3, 2) and mesh.normals.shape == (2, 3)
    assert mesh.source == {"format": "obj", "comments": 1,
                           "ignored": {"mtllib": 1, "o": 1, "g": 1, "s": 1, "usemtl": 1}}


def test_vertices_zyx_reverses_the_stored_xyz():
    mesh = M.parse_obj(OBJ_FULL)
    assert mesh.vertices_zyx.tolist()[0] == [3.25, 2.0, 1.5]
    assert mesh.vertices_zyx.flags.c_contiguous


@pytest.mark.parametrize("face,vt,vn", [("1 2 3", False, False), ("1/1 2/2 3/3", True, False),
                                        ("1//1 2//1 3//1", False, True), ("1/1/1 2/2/1 3/3/1", True, True)])
def test_obj_face_forms(face, vt, vn):
    mesh = M.parse_obj(obj(f"v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nvn 0 0 1\nf {face}\n"))
    assert mesh.faces.tolist() == [[0, 1, 2]]
    assert (mesh.face_texcoords is not None) == vt and (mesh.face_normals is not None) == vn


def test_obj_negative_and_positive_indices_give_the_same_mesh():
    positive = M.parse_obj(obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\nv 0 0 1\nf 2 3 4\n"))
    negative = M.parse_obj(obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\nv 0 0 1\nf -3 -2 -1\n"))
    assert positive.faces.tolist() == negative.faces.tolist() == [[0, 1, 2], [1, 2, 3]]


def test_obj_forward_positive_reference_is_checked_against_the_total():
    mesh = M.parse_obj(obj("f 1 2 3\nv 0 0 0\nv 1 0 0\nv 0 1 0\n"))
    assert mesh.faces.tolist() == [[0, 1, 2]]


def test_vc_tifxyz2obj_style_output_reads():
    text = "v 1.000000 2.000000 3.000000\nvt 0.000000 0.000000\nvn 0.000000 0.000000 1.000000\n" * 3
    mesh = M.parse_obj(obj(text + "f 3/3/3 1/1/1 2/2/2\n"))
    assert mesh.faces.tolist() == [[2, 0, 1]] and mesh.face_normals.tolist() == [[2, 0, 1]]


def test_obj_round_trip_and_deterministic_bytes(tmp_path):
    src = tmp_path / "in.obj"
    src.write_bytes(OBJ_FULL)
    first = M.read_mesh(src)
    digest = M.write_obj(tmp_path / "a.obj", first)
    M.write_obj(tmp_path / "b.obj", first)
    assert (tmp_path / "a.obj").read_bytes() == (tmp_path / "b.obj").read_bytes()
    assert digest == "sha256:" + hashlib.sha256((tmp_path / "a.obj").read_bytes()).hexdigest()
    second = M.read_obj(tmp_path / "a.obj")
    for name in ("vertices_xyz", "faces", "texcoords", "normals", "face_texcoords", "face_normals"):
        assert np.array_equal(getattr(first, name), getattr(second, name)), name
    assert first.source["digest"] == "sha256:" + hashlib.sha256(OBJ_FULL).hexdigest()


def test_obj_writer_text_is_pinned():
    mesh = M.TriangleMesh(vertices_xyz=np.array([[0.1, -0.0, 1e-5], [1, 2, 3], [4, 5, 6]], dtype=np.float64),
                          faces=np.array([[0, 1, 2]]), normals=np.array([[0.0, 0.0, 1.0]]),
                          face_normals=np.array([[0, 0, 0]]))
    assert M.encode_obj(mesh) == (b"v 0.1 -0.0 1e-05\nv 1.0 2.0 3.0\nv 4.0 5.0 6.0\nvn 0.0 0.0 1.0\n"
                                  b"f 1//1 2//1 3//1\n")
    plain = M.TriangleMesh(vertices_xyz=np.zeros((3, 3)), faces=np.array([[2, 1, 0]]),
                           texcoords=np.array([[0.5, 0.25, 1.0]]), face_texcoords=np.array([[0, 0, 0]]))
    assert M.encode_obj(plain).endswith(b"vt 0.5 0.25 1.0\nf 3/1 2/1 1/1\n")


def test_float32_vertices_survive_an_obj_round_trip_exactly():
    v = np.array([[0.1, 0.2, 0.3], [1e-7, 3.4e38, -2.5]], dtype=np.float32)
    mesh = M.TriangleMesh(vertices_xyz=np.vstack([v, v[:1]]), faces=np.array([[0, 1, 2]]))
    back = M.parse_obj(M.encode_obj(mesh))
    assert np.array_equal(back.vertices_xyz.astype(np.float32), mesh.vertices_xyz)


# --------------------------------------------------------------------------------------------- PLY


def test_ply_ascii_reads_typed_columns():
    mesh = M.parse_ply(PLY_ASCII, "a.ply")
    assert mesh.vertices_xyz.dtype == np.float32
    assert mesh.vertices_xyz.tolist() == [[0, 0, 0], [1, 0, 0], [0, 2.5, 0], [0, 0, 3]]
    assert mesh.faces.tolist() == [[0, 1, 2], [1, 2, 3]] and mesh.faces.dtype == np.int64
    names = [(n, t) for n, t, _ in mesh.vertex_properties]
    assert names == [("red", "uchar"), ("quality", "double")]
    assert mesh.vertex_properties[0][2].dtype == np.uint8 and mesh.vertex_properties[0][2].tolist() == [255, 0, 7, 9]
    assert mesh.vertex_properties[1][2].tolist() == [0.5, -1e-3, 2.0, 1e10]
    assert mesh.source == {"format": "ply", "encoding": "ascii", "comments": ["made by hand"],
                           "obj_info": ["synthetic"]}


def test_ply_binary_reads_and_round_trips(tmp_path):
    src = tmp_path / "t.PLY"
    src.write_bytes(ply_binary(TETRA, TETRA_FACES, vertex_props=("double x", "double y", "double z"),
                               vfmt="<ddd"))
    first = M.read_mesh(src)
    assert first.vertices_xyz.dtype == np.float64 and first.faces.tolist() == [list(f) for f in TETRA_FACES]
    M.write_ply(tmp_path / "a.ply", first)
    M.write_ply(tmp_path / "b.ply", first)
    assert (tmp_path / "a.ply").read_bytes() == (tmp_path / "b.ply").read_bytes()
    second = M.read_ply(tmp_path / "a.ply")
    assert np.array_equal(first.vertices_xyz, second.vertices_xyz) and second.vertices_xyz.dtype == np.float64
    assert np.array_equal(first.faces, second.faces)


def test_ply_ascii_to_binary_round_trip_keeps_types_and_properties(tmp_path):
    first = M.parse_ply(PLY_ASCII)
    data = M.encode_ply(first)
    assert data.startswith(b"ply\nformat binary_little_endian 1.0\nelement vertex 4\nproperty float x\n"
                           b"property float y\nproperty float z\nproperty uchar red\nproperty double quality\n"
                           b"element face 2\nproperty list uchar int vertex_indices\nend_header\n")
    second = M.parse_ply(data)
    assert np.array_equal(first.vertices_xyz, second.vertices_xyz) and second.vertices_xyz.dtype == np.float32
    assert np.array_equal(first.faces, second.faces)
    for (n1, t1, v1), (n2, t2, v2) in zip(first.vertex_properties, second.vertex_properties, strict=True):
        assert (n1, t1) == (n2, t2) and v1.dtype == v2.dtype and np.array_equal(v1, v2)
    assert len(data) == len(data.split(b"end_header\n")[0]) + 11 + 4 * (12 + 1 + 8) + 2 * 13


def test_ply_type_aliases_and_uint_indices():
    data = ply_binary(TETRA, TETRA_FACES, vertex_props=("float32 x", "float32 y", "float32 z"),
                      face_prop="list uint8 uint32 vertex_indices", ffmt="<B3I")
    assert M.parse_ply(data).faces.tolist() == [list(f) for f in TETRA_FACES]


def test_empty_ply_and_obj():
    mesh = M.parse_ply(ply_binary([], []))
    assert mesh.vertices_xyz.shape == (0, 3) and mesh.faces.shape == (0, 3)
    assert M.parse_ply(M.encode_ply(mesh)).faces.shape == (0, 3)
    assert M.parse_obj(b"# nothing\n").vertices_xyz.shape == (0, 3)
    assert M.encode_obj(M.parse_obj(b"")) == b""


def test_cross_format_geometry_agrees():
    from_obj = M.parse_obj(OBJ_FULL)
    with pytest.raises(Refusal):
        M.encode_ply(from_obj)
    via_ply = M.parse_ply(M.encode_ply(from_obj.without_attributes()))
    assert np.array_equal(via_ply.vertices_xyz, from_obj.vertices_xyz)
    assert np.array_equal(via_ply.faces, from_obj.faces)
    from_ply = M.parse_ply(PLY_ASCII)
    via_obj = M.parse_obj(M.encode_obj(from_ply.without_attributes()))
    assert np.array_equal(via_obj.vertices_xyz, from_ply.vertices_xyz.astype(np.float64))


def test_mesh_frame_check_compares_in_zyx():
    mesh = M.parse_ply(ply_binary(TETRA, TETRA_FACES))
    frame = VolumeFrame(volume_id="20990101000004", shape_zyx=(4, 3, 2), voxel_size_um=2.4, source="test")
    M.check_mesh_within(mesh, frame, "tetra")
    small = VolumeFrame(volume_id="20990101000004", shape_zyx=(3, 3, 2), voxel_size_um=2.4, source="test")
    exc = refusal(M.check_mesh_within, mesh, small, "tetra")
    assert exc.code == "IO_FRAME_POINTS_OUTSIDE" and exc.details["example_zyx"] == [3.0, 0.0, 0.0]


def test_inspect_mesh(tmp_path):
    (tmp_path / "m.obj").write_bytes(OBJ_FULL)
    summary = M.inspect_mesh(tmp_path / "m.obj")
    assert summary["vertices"] == 4 and summary["faces"] == 2 and summary["coordinate_type"] == "double"
    assert summary["bounds_zyx"] == [[0.5, 0.0, -4.0], [9.0, 500.0, 7.0]]
    assert summary["texcoords"] == [3, 2] and summary["source"]["path"] == str(tmp_path / "m.obj")
    canonical_dumps(summary)
    (tmp_path / "m.ply").write_bytes(PLY_ASCII)
    canonical_dumps(M.inspect_mesh(tmp_path / "m.ply"))


# ------------------------------------------------------------------------------------------ refusals


def test_unknown_extension(tmp_path):
    exc = refusal(M.read_mesh, tmp_path / "m.stl")
    assert exc.code == "IO_MESH_FORMAT_UNKNOWN" and exc.details["path"] == str(tmp_path / "m.stl")


@pytest.mark.parametrize("text,line", [
    (b"v 1 2\n", 1), (b"v 0 0 0\nv 1 2 x\n", 2), (b"vn 1 2\n", 1), (b"vt 1\n", 1), (b"vt 1 2 3 4\n", 1),
    (b"vt 1 2\nvt 1 2 3\n", 2), (b"v 1 2 3 0.5 0.5 0.5\n", 1), (b"v nan 0 0\n", 1), (b"v 1_0 0 0\n", 1),
    (b"v 0 0 0\nf 1/2/3/4 1 1\n", 2), (b"v 0 0 0\nf 1/ 1/ 1/\n", 2), (b"v 0 0 0\nf a b c\n", 2),
    (b"v 0 0 0\nf 1/x 1/x 1/x\n", 2), (b"v 0 0 0\nf 1/1/x 1/1/x 1/1/x\n", 2), (b"v 0 0 0\nf +1 1 1\n", None),
    ("v ١ 0 0\n".encode(), 1), ("v 0 0 0\nf ١ 1 1\n".encode(), 2),  # non-ASCII digits
])
def test_obj_syntax(text, line):
    if line is None:  # a leading '+' is a valid integer: this one parses
        assert M.parse_obj(text).faces.tolist() == [[0, 0, 0]]
        return
    exc = refusal(M.parse_obj, text, "s.obj")
    assert exc.code == "IO_MESH_OBJ_SYNTAX" and exc.details == {"path": "s.obj", "line": line}


def test_obj_not_utf8():
    exc = refusal(M.parse_obj, b"v 0 0 0\n\xff\n", "u.obj")
    assert exc.code == "IO_MESH_OBJ_SYNTAX" and exc.details == {"path": "u.obj", "line": None}


@pytest.mark.parametrize("keyword", ["l", "p", "curv", "vp"])
def test_obj_geometry_apok_does_not_read_is_refused(keyword):
    exc = refusal(M.parse_obj, obj(f"v 0 0 0\nv 1 0 0\n{keyword} 1 2\n"), "u.obj")
    assert exc.code == "IO_MESH_OBJ_UNSUPPORTED" and exc.details == {"path": "u.obj", "line": 3, "keyword": keyword}


@pytest.mark.parametrize("face,count", [("1 2 3 4", 4), ("1 2", 2), ("1 2 3 4 5", 5)])
def test_obj_non_triangles_are_refused_not_triangulated(face, count):
    exc = refusal(M.parse_obj, obj(f"v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\nv 2 2 2\nf {face}\n"), "q.obj")
    assert exc.code == "IO_MESH_FACE_NOT_TRIANGLE" and exc.details == {"path": "q.obj", "line": 6, "count": count}


@pytest.mark.parametrize("faces,line", [("f 1 2 3\nf 1/1 2/1 3/1\n", 7), ("f 1 2/1 3\n", 6),
                                        ("f 1//1 2//1 3//1\nf 1/1/1 2/1/1 3/1/1\n", 7)])
def test_obj_mixed_face_forms(faces, line):
    exc = refusal(M.parse_obj, obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvn 0 0 1\n" + faces), "m.obj")
    assert exc.code == "IO_MESH_FACE_FORM_MIXED" and exc.details == {"path": "m.obj", "line": line}


@pytest.mark.parametrize("text,kind,index,line", [
    ("v 0 0 0\nf 0 1 1\n", "v", 0, 2),
    ("v 0 0 0\nf -2 1 1\n", "v", -2, 2),
    ("v 0 0 0\nv 0 0 0\nv 0 0 0\nf 1 2 4\n", "v", 4, 4),
    ("v 0 0 0\nvt 0 0\nf 1/1 1/2 1/1\n", "vt", 2, 3),
    ("v 0 0 0\nvn 0 0 1\nf 1//1 1//1 1//-2\n", "vn", -2, 3),
    ("v 0 0 0\nvn 0 0 1\nf 1//1 1//1 1//3\n", "vn", 3, 3),
])
def test_obj_index_out_of_range(text, kind, index, line):
    exc = refusal(M.parse_obj, obj(text), "i.obj")
    assert exc.code == "IO_MESH_INDEX_OUT_OF_RANGE"
    assert exc.details == {"path": "i.obj", "line": line, "kind": kind, "index": index}


def test_obj_overflowing_number():
    exc = refusal(M.parse_obj, b"v 1e400 0 0\n", "o.obj")
    assert exc.code == "IO_MESH_VALUE_INVALID" and exc.details == {"path": "o.obj", "line": 1}


@pytest.mark.parametrize("statement", ["f 1 2 3 # tri", "f 1 2 3#tri", "v 0 0 1 # apex", "vt 0 0 # uv",
                                       "vn 0 0 1 #n"])
def test_obj_trailing_inline_comment_is_a_syntax_error_named_as_such(statement):
    # Without the check "f 1 2 3 # tri" was refused as a five-vertex face: the right refusal, the wrong cause.
    text = obj(f"v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvn 0 0 1\n{statement}\n")
    exc = refusal(M.parse_obj, text, "c.obj")
    assert exc.code == "IO_MESH_OBJ_SYNTAX" and exc.details == {"path": "c.obj", "line": 6}
    assert "inline comment" in exc.reason and "#" in exc.reason


def test_a_hash_in_a_statement_without_geometry_is_not_a_comment():
    mesh = M.parse_obj(obj("o part#1\nusemtl stone#2\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"))
    assert mesh.source["ignored"] == {"o": 1, "usemtl": 1} and mesh.faces.tolist() == [[0, 1, 2]]


@pytest.mark.parametrize("corner,kind,digits", [
    ("1" * 5000, "v", 5000),  # past int()'s 4300-digit limit: a refusal, not a ValueError
    ("-" + "9" * 20, "v", 20),
    ("1/" + "2" * 5000, "vt", 5000),
    ("1//" + "3" * 4400, "vn", 4400),
], ids=["v-5000", "v-20-negative", "vt-5000", "vn-4400"])
def test_obj_index_with_more_digits_than_any_count_is_out_of_range(corner, kind, digits):
    text = obj(f"v 0 0 0\nvt 0 0\nvn 0 0 1\nf {corner} {corner} {corner}\n")
    exc = refusal(M.parse_obj, text, "d.obj")
    assert exc.code == "IO_MESH_INDEX_OUT_OF_RANGE"
    assert exc.details == {"path": "d.obj", "line": 4, "kind": kind, "index": None, "digits": digits}


def test_obj_index_digits_are_counted_without_leading_zeros():
    padded = "0" * 4999 + "1"  # index 1, written with 5000 characters
    assert M.parse_obj(obj(f"v 0 0 0\nf {padded} 1 1\n")).faces.tolist() == [[0, 0, 0]]
    nineteen = "1" + "0" * 18  # 19 significant digits pass the digit bound and fail the total check
    exc = refusal(M.parse_obj, obj(f"v 0 0 0\nf 1 1 {nineteen}\n"), "d.obj")
    assert exc.code == "IO_MESH_INDEX_OUT_OF_RANGE"
    assert exc.details == {"path": "d.obj", "line": 2, "kind": "v", "index": 10 ** 18}


@pytest.mark.parametrize("data", [
    b"PLY\nformat ascii 1.0\nend_header\n",
    b"ply\nformat ascii 1.0\nelement vertex 0\n",
    b"",
])
def test_ply_header_frame(data):
    exc = refusal(M.parse_ply, data, "h.ply")
    assert exc.code == "IO_MESH_PLY_HEADER" and exc.details == {"path": "h.ply"}


@pytest.mark.parametrize("lines,line", [
    (["format ascii 1.0", "format ascii 1.0"], 3),
    (["format ascii 1.0", "property float x"], 3),
    (["format ascii 1.0", "element vertex -1"], 3),
    (["format ascii 1.0", "element vertex 1 2"], 3),
    (["format ascii 1.0", "element vertex 0", "property list uchar x"], 4),
    (["format ascii 1.0", "element vertex 0", "property float"], 4),
    (["format ascii 1.0", "element vertex 0", "property float x y"], 4),
    (["format ascii 1.0", "element vertex 0", "flag on"], 4),
    (["format", "element vertex 0"], 2),
])
def test_ply_header_lines(lines, line):
    data = ("\n".join(["ply", *lines, "end_header"]) + "\n").encode()
    exc = refusal(M.parse_ply, data, "h.ply")
    assert exc.code == "IO_MESH_PLY_HEADER" and exc.details == {"path": "h.ply", "line": line}


def test_ply_header_without_format_or_with_non_ascii():
    exc = refusal(M.parse_ply, b"ply\ncomment none\nend_header\n", "h.ply")
    assert exc.code == "IO_MESH_PLY_HEADER" and exc.details == {"path": "h.ply", "line": None}
    exc = refusal(M.parse_ply, "ply\ncomment café\nformat ascii 1.0\nend_header\n".encode(), "h.ply")
    assert exc.code == "IO_MESH_PLY_HEADER" and exc.details == {"path": "h.ply", "line": 2}


@pytest.mark.parametrize("fmt", ["binary_big_endian 1.0", "ascii 2.0", "binary 1.0"])
def test_ply_unsupported_formats(fmt):
    exc = refusal(M.parse_ply, f"ply\nformat {fmt}\nend_header\n".encode(), "b.ply")
    assert exc.code == "IO_MESH_PLY_UNSUPPORTED" and exc.details == {"path": "b.ply", "format": fmt}


@pytest.mark.parametrize("vertex_props,face_prop,prop", [
    (("float16 x", "float y", "float z"), "list uchar int vertex_indices", None),
    (("float x", "float y", "float z"), "list uchar int64 vertex_indices", None),
    (("int x", "int y", "int z"), "list uchar int vertex_indices", "x/y/z"),
    (("float x", "double y", "float z"), "list uchar int vertex_indices", "x/y/z"),
    (("float x", "float y", "float z"), "list int int vertex_indices", "vertex_indices"),
    (("float x", "float y", "float z"), "list uchar short vertex_indices", "vertex_indices"),
    (("float x", "float y", "float z"), "list uchar float vertex_indices", "vertex_indices"),
])
def test_ply_property_types(vertex_props, face_prop, prop):
    data = ply_binary([], [], vertex_props=vertex_props, face_prop=face_prop)
    exc = refusal(M.parse_ply, data, "t.ply")
    assert exc.code == "IO_MESH_PLY_PROPERTY_TYPE"
    if prop is None:
        assert exc.details["type"] in ("float16", "int64")
    else:
        assert exc.details["property"] == prop


@pytest.mark.parametrize("header,detail", [
    ("element vertex 0\nproperty float x\nproperty float y\nproperty float z\nelement face 0\n"
     "property list uchar int vertex_indices\nelement edge 0\nproperty int a",
     ("elements", ["vertex", "face", "edge"])),
    ("element face 0\nproperty list uchar int vertex_indices\nelement vertex 0\nproperty float x",
     ("elements", ["face", "vertex"])),
    ("element vertex 0\nproperty float x\nproperty float y\nelement face 0\nproperty list uchar int vertex_indices",
     ("element", "vertex")),
    ("element vertex 0\nproperty float x\nproperty float x\nproperty float y\nproperty float z\nelement face 0\n"
     "property list uchar int vertex_indices", ("element", "vertex")),
    ("element vertex 0\nproperty float x\nproperty float y\nproperty float z\nproperty list uchar int n\n"
     "element face 0\nproperty list uchar int vertex_indices", ("element", "vertex")),
    ("element vertex 0\nproperty float x\nproperty float y\nproperty float z\nelement face 0\n"
     "property list uchar int vertex_index", ("element", "face")),
    ("element vertex 0\nproperty float x\nproperty float y\nproperty float z\nelement face 0\n"
     "property list uchar int vertex_indices\nproperty uchar red", ("element", "face")),
    ("element vertex 0\nproperty float x\nproperty float y\nproperty float z\nelement face 0\n"
     "property int vertex_indices", ("element", "face")),
])
def test_ply_layout(header, detail):
    data = f"ply\nformat binary_little_endian 1.0\n{header}\nend_header\n".encode()
    exc = refusal(M.parse_ply, data, "l.ply")
    assert exc.code == "IO_MESH_PLY_LAYOUT" and exc.details[detail[0]] == detail[1]


def test_ply_non_triangles_are_refused_binary_and_ascii():
    data = ply_binary(TETRA, [(0, 1, 2), (0, 1, 2, 3), (1, 2, 3)])
    exc = refusal(M.parse_ply, data, "q.ply")
    assert exc.code == "IO_MESH_FACE_NOT_TRIANGLE" and exc.details == {"path": "q.ply", "face": 1, "count": 4}
    text = PLY_ASCII.replace(b"3 1 2 3\n", b"4 1 2 3 0\n")
    exc = refusal(M.parse_ply, text, "q.ply")
    assert exc.code == "IO_MESH_FACE_NOT_TRIANGLE" and exc.details == {"path": "q.ply", "face": 1, "count": 4}


@pytest.mark.parametrize("faces,face", [([(0, 1, 4)], 0), ([(0, 1, 2), (0, -1, 2)], 1)])
def test_ply_index_out_of_range(faces, face):
    exc = refusal(M.parse_ply, ply_binary(TETRA, faces), "i.ply")
    assert exc.code == "IO_MESH_INDEX_OUT_OF_RANGE" and exc.details == {"path": "i.ply", "face": face, "kind": "v"}


def test_ply_uint_index_beyond_int32_is_out_of_range_not_wrapped():
    data = ply_binary(TETRA, [(0, 1, 2 ** 32 - 1)], face_prop="list uchar uint vertex_indices", ffmt="<B3I")
    exc = refusal(M.parse_ply, data, "i.ply")
    assert exc.code == "IO_MESH_INDEX_OUT_OF_RANGE" and exc.details["face"] == 0


def test_ply_non_finite_vertex():
    data = ply_binary([(0, 0, 0), (1, float("nan"), 0), (0, 1, 0)], [(0, 1, 2)])
    exc = refusal(M.parse_ply, data, "n.ply")
    assert exc.code == "IO_MESH_VALUE_INVALID" and exc.details == {"path": "n.ply", "vertex": 1}


@pytest.mark.parametrize("old,new,where,value", [
    (b"0 2.5 0 7 2", b"0 1e39 0 7 2", "vertex 2", "1e39"),  # overflows the declared float
    (b"0 2.5 0 7 2", b"0 2.5 0 7 1e400", "vertex 2", "1e400"),  # overflows the declared double
    (b"0 2.5 0 7 2", b"0 2.5 0 7 two", "vertex 2", "two"),
    (b"0 2.5 0 7 2", b"0 2.5 0 300 2", "vertex 2", "300"),
    (b"0 2.5 0 7 2", b"0 2.5 0 7.5 2", "vertex 2", "7.5"),
    (b"0 2.5 0 7 2", b"0 nan 0 7 2", "vertex 2", "nan"),
    (b"3 1 2 3\n", b"3 1 2 3.0\n", "face 1", "3.0"),
    (b"3 1 2 3\n", b"-3 1 2 3\n", "face 1", "-3"),
])
def test_ply_ascii_values_must_parse_as_their_type(old, new, where, value):
    exc = refusal(M.parse_ply, PLY_ASCII.replace(old, new), "v.ply")
    assert exc.code == "IO_MESH_VALUE_INVALID"
    assert exc.details == {"path": "v.ply", "where": where, "value": value}


@pytest.mark.parametrize("token", ["2" * 5000, "1" + "0" * 19], ids=["5000-digits", "20-digits"])
def test_ply_ascii_integer_with_too_many_digits_is_invalid_not_a_crash(token):
    exc = refusal(M.parse_ply, PLY_ASCII.replace(b"3 1 2 3\n", f"3 1 2 {token}\n".encode()), "v.ply")
    assert exc.code == "IO_MESH_VALUE_INVALID"
    assert exc.details == {"path": "v.ply", "where": "face 1", "value": token}


def test_ply_ascii_zero_padded_integers_read_their_value():
    mesh = M.parse_ply(PLY_ASCII.replace(b"3 1 2 3\n", ("3 1 2 " + "0" * 4999 + "3\n").encode()))
    assert mesh.faces.tolist() == [[0, 1, 2], [1, 2, 3]]


def test_ply_truncated():
    full = ply_binary(TETRA, TETRA_FACES)
    header_len = len(full) - 4 * 12 - 4 * 13
    exc = refusal(M.parse_ply, full[: header_len + 4 * 12 - 1], "t.ply")
    assert exc.code == "IO_MESH_TRUNCATED" and exc.details == {"path": "t.ply", "element": "vertex"}
    exc = refusal(M.parse_ply, full[:-1], "t.ply")
    assert exc.code == "IO_MESH_TRUNCATED" and exc.details == {"path": "t.ply", "element": "face"}
    exc = refusal(M.parse_ply, PLY_ASCII.replace(b"3 1 2 3\n", b"3 1 2\n"), "t.ply")
    assert exc.code == "IO_MESH_TRUNCATED" and exc.details == {"path": "t.ply", "element": "face"}
    exc = refusal(M.parse_ply, PLY_ASCII.split(b"0 0 3 9 1e10")[0], "t.ply")
    assert exc.code == "IO_MESH_TRUNCATED" and exc.details == {"path": "t.ply", "element": "vertex"}


@pytest.mark.parametrize("vertices,faces,element", [
    (10 ** 13, 0, "vertex"),  # an array this size would need 36 TiB
    (10 ** 20, 0, "vertex"),  # beyond any numpy dimension
    (4, 10 ** 15, "face"),
])
def test_ply_ascii_counts_beyond_the_data_are_truncation_not_an_allocation(vertices, faces, element):
    data = PLY_ASCII.replace(b"element vertex 4", f"element vertex {vertices}".encode())
    data = data.replace(b"element face 2", f"element face {faces}".encode())
    if not faces:
        data = data.split(b"3 0 1 2")[0]  # only vertex values follow the header
    exc = refusal(M.parse_ply, data, "big.ply")
    assert exc.code == "IO_MESH_TRUNCATED" and exc.details == {"path": "big.ply", "element": element}


def test_ply_trailing_data():
    exc = refusal(M.parse_ply, ply_binary(TETRA, TETRA_FACES, extra=b"\x00"), "x.ply")
    assert exc.code == "IO_MESH_TRAILING_DATA" and exc.details == {"path": "x.ply", "extra_bytes": 1}
    exc = refusal(M.parse_ply, PLY_ASCII + b"7 8\n", "x.ply")
    assert exc.code == "IO_MESH_TRAILING_DATA" and exc.details == {"path": "x.ply", "extra_values": 2}


# ------------------------------------------------------------------------------------- writer refusals


def good_mesh(**changes) -> M.TriangleMesh:
    base = dict(vertices_xyz=np.array(TETRA, dtype=np.float64), faces=np.array(TETRA_FACES, dtype=np.int64))
    return M.TriangleMesh(**{**base, **changes})


@pytest.mark.parametrize("changes,field", [
    ({"vertices_xyz": np.array(TETRA, dtype=np.float16)}, "vertices_xyz"),
    ({"vertices_xyz": np.array(TETRA, dtype=np.float64)[:, :2]}, "vertices_xyz"),
    ({"vertices_xyz": [list(v) for v in TETRA]}, "vertices_xyz"),
    ({"vertices_xyz": np.array([[0, 0, np.inf]] + TETRA[1:], dtype=np.float64)}, "vertices_xyz"),
    ({"faces": np.array([[0, 1, 4]])}, "faces"),
    ({"faces": np.array([[0, 1, -1]])}, "faces"),
    ({"faces": np.array([[0.0, 1.0, 2.0]])}, "faces"),
    ({"faces": np.array([[0, 1, 2, 3]])}, "faces"),
    ({"faces": None}, "faces"),
    ({"texcoords": np.zeros((2, 4))}, "texcoords"),
    ({"texcoords": np.full((2, 2), np.nan)}, "texcoords"),
    ({"texcoords": np.zeros((2, 2), dtype=np.int64)}, "texcoords"),
    ({"normals": np.zeros((2, 2))}, "normals"),
    ({"face_texcoords": np.zeros((4, 3), dtype=np.int64)}, "face_texcoords"),
    ({"texcoords": np.zeros((1, 2)), "face_texcoords": np.ones((4, 3), dtype=np.int64)}, "face_texcoords"),
    ({"texcoords": np.zeros((1, 2)), "face_texcoords": np.zeros((3, 3), dtype=np.int64)}, "face_texcoords"),
    ({"normals": np.zeros((1, 3)), "face_normals": np.zeros((4, 2), dtype=np.int64)}, "face_normals"),
])
def test_writers_refuse_malformed_meshes(changes, field):
    for encode in (M.encode_obj, M.encode_ply):
        exc = refusal(encode, good_mesh(**changes))
        assert exc.code == "IO_MESH_WRITE_INVALID" and exc.details["field"] == field


def test_well_formed_attributes_are_written_to_obj():
    mesh = good_mesh(texcoords=np.zeros((1, 2)), face_texcoords=np.zeros((4, 3), dtype=np.int64),
                     normals=np.zeros((2, 3)), face_normals=np.ones((4, 3), dtype=np.uint8))
    assert M.parse_obj(M.encode_obj(mesh)).face_normals.tolist() == [[1, 1, 1]] * 4


def test_each_writer_refuses_what_its_format_cannot_carry():
    exc = refusal(M.encode_obj, good_mesh(vertex_properties=(("red", "uchar", np.zeros(4, np.uint8)),)))
    assert exc.code == "IO_MESH_WRITE_INVALID" and exc.details["field"] == "vertex_properties"
    for field, value in (("texcoords", np.zeros((1, 2))), ("normals", np.zeros((1, 3)))):
        exc = refusal(M.encode_ply, good_mesh(**{field: value}))
        assert exc.code == "IO_MESH_WRITE_INVALID" and exc.details["field"] == field
    exc = refusal(M.encode_ply, good_mesh(texcoords=np.zeros((1, 2)),
                                          face_texcoords=np.zeros((4, 3), dtype=np.int64)))
    assert exc.details["field"] == "texcoords"


@pytest.mark.parametrize("prop", [
    ("red", "uchar", np.zeros(4, np.uint16)),
    ("red", "uchar", np.zeros(3, np.uint8)),
    ("red", "int64", np.zeros(4, np.int64)),
    ("x", "float", np.zeros(4, np.float32)),
    ("two words", "uchar", np.zeros(4, np.uint8)),
    ("red", "uchar", [0, 0, 0, 0]),
])
def test_ply_writer_refuses_bad_vertex_properties(prop):
    exc = refusal(M.encode_ply, good_mesh(vertex_properties=(prop,)))
    assert exc.code == "IO_MESH_WRITE_INVALID" and exc.details["property"] == prop[0]


def test_ply_writer_refuses_duplicate_properties_and_accepts_aliases():
    props = (("q", "float32", np.zeros(4, np.float32)), ("q", "float", np.zeros(4, np.float32)))
    exc = refusal(M.encode_ply, good_mesh(vertex_properties=props))
    assert exc.details["property"] == "q"
    data = M.encode_ply(good_mesh(vertex_properties=props[:1]))
    assert b"property float q\n" in data


def test_ply_writer_refuses_more_vertices_than_int32_indices():
    huge = np.broadcast_to(np.float64(0), (2 ** 31 + 1, 3))  # no allocation; refused before any scan
    exc = refusal(M.encode_ply, good_mesh(vertices_xyz=huge, faces=np.zeros((0, 3), dtype=np.int64)))
    assert exc.code == "IO_MESH_WRITE_INVALID" and exc.details["field"] == "faces"


def test_writers_never_clobber(tmp_path):
    for write, name in ((M.write_obj, "m.obj"), (M.write_ply, "m.ply")):
        target = tmp_path / name
        target.write_bytes(b"precious")
        exc = refusal(write, target, good_mesh())
        assert exc.code == "IO_MESH_WRITE_EXISTS" and exc.details["path"] == str(target)
        assert target.read_bytes() == b"precious"
        write(target, good_mesh(), overwrite=True)
        assert M.read_mesh(target).faces.tolist() == [list(f) for f in TETRA_FACES]
