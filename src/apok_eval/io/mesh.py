"""Triangular meshes: strict OBJ and PLY readers, deterministic OBJ and binary-PLY writers.

Both formats store vertex coordinates as **(x, y, z)**; :class:`TriangleMesh`
keeps them in that order (``vertices_xyz``) and exposes apok-eval's (z, y, x) order
explicitly as ``vertices_zyx``. Faces are triangles of 0-based vertex indices;
a face with any other vertex count is refused, never triangulated.

OBJ (Wavefront text): ``v x y z``, ``vt u v [w]``, ``vn x y z`` and ``f`` with
three corners of one form for the whole file (``v``, ``v/vt``, ``v//vn`` or
``v/vt/vn``); 1-based positive indices (checked against the file's totals)
and negative indices (relative to the elements defined so far). ``o``, ``g``,
``s``, ``usemtl`` and ``mtllib`` carry no geometry: they are counted in
``source["ignored"]``. Any other statement (``l``, ``p``, curves, vertex
colours as extra ``v`` numbers, ...) is refused rather than dropped, and so is
a ``#`` comment trailing a ``v``/``vt``/``vn``/``f`` statement.

PLY: ``ascii 1.0`` and ``binary_little_endian 1.0``; exactly the elements
``vertex`` (scalar properties, ``x``/``y``/``z`` all ``float`` or all
``double``; others kept in ``vertex_properties``) then ``face`` (one property,
``list uchar int|uint vertex_indices``). Short data and data after the last
element are refused. ASCII data is accumulated as it is read, so a header
count larger than the data is refused as truncated without allocating it.

Writers are deterministic and add nothing (no comments, no timestamps). Each
refuses data its format cannot carry instead of dropping it: OBJ has no
generic vertex properties, and PLY (as written here) no OBJ-style per-corner
texture/normal indices; :meth:`TriangleMesh.without_attributes` makes the
choice explicit.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from collections.abc import Iterator
from typing import Any

import numpy as np

from apok_eval.canonical import sha256_bytes
from apok_eval.io.frame import VolumeFrame, check_points_within
from apok_eval.fileio import NotRegularFile, portable_name, read_regular, write_atomic
from apok_eval.refusal import Refusal, register_codes

register_codes(
    "apok_eval.io.mesh",
    {
        "IO_MESH_FORMAT_UNKNOWN": "a mesh path's extension is neither .obj nor .ply",
        "IO_MESH_FILE_MISSING": "a mesh path is not a valid path, names nothing, or names no regular file",
        "IO_MESH_OBJ_SYNTAX": "an OBJ line is malformed",
        "IO_MESH_OBJ_UNSUPPORTED": "an OBJ statement carries geometry apok-eval does not read",
        "IO_MESH_FACE_NOT_TRIANGLE": "a face has other than three vertices",
        "IO_MESH_FACE_FORM_MIXED": "OBJ face corners mix the v, v/vt, v//vn and v/vt/vn forms",
        "IO_MESH_INDEX_OUT_OF_RANGE": "a face refers to a vertex, texture coordinate or normal that does not exist",
        "IO_MESH_VALUE_INVALID": "a coordinate is not finite, or a value does not parse as its declared type",
        "IO_MESH_PLY_HEADER": "a PLY header is malformed",
        "IO_MESH_PLY_UNSUPPORTED": "a PLY format other than ascii 1.0 or binary_little_endian 1.0",
        "IO_MESH_PLY_PROPERTY_TYPE": "a PLY property type is unknown or not allowed for that property",
        "IO_MESH_PLY_LAYOUT": "PLY elements are not vertex (x, y, z, ...) then face (vertex_indices)",
        "IO_MESH_TRUNCATED": "mesh data ends before the declared element counts",
        "IO_MESH_TRAILING_DATA": "mesh data continues after the declared elements",
        "IO_MESH_WRITE_INVALID": "a mesh given to a writer cannot be written in that format",
        "IO_MESH_WRITE_EXISTS": "the output file exists and overwrite was not requested",
    },
)

#: PLY type name -> numpy type code (little-endian where it matters). Aliases map to the same code.
PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}
#: numpy type code -> the PLY 1.0 name the writer emits.
PLY_NAMES = {"i1": "char", "u1": "uchar", "i2": "short", "u2": "ushort", "i4": "int", "u4": "uint",
             "f4": "float", "f8": "double"}
_OBJ_IGNORED = ("o", "g", "s", "usemtl", "mtllib")
# ASCII digits only: Python's float() and int() would also accept other scripts' digits.
_NUMBER = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
_INTEGER = re.compile(r"[+-]?[0-9]+\Z")
#: Significant digits beyond any index or PLY integer (int64 has 19); int() fails past 4300 digits.
_MAX_INTEGER_DIGITS = 19
_GEOMETRY = ("v", "vt", "vn", "f")


@dataclass(frozen=True, eq=False)
class TriangleMesh:
    """A triangle mesh; coordinates (x, y, z) as stored in OBJ/PLY, indices 0-based."""

    vertices_xyz: np.ndarray  # (N, 3) float32 or float64
    faces: np.ndarray  # (F, 3) int64
    texcoords: np.ndarray | None = None  # OBJ vt, (T, 2) or (T, 3) float64
    normals: np.ndarray | None = None  # OBJ vn, (M, 3) float64
    face_texcoords: np.ndarray | None = None  # (F, 3) int64 into texcoords
    face_normals: np.ndarray | None = None  # (F, 3) int64 into normals
    vertex_properties: tuple[tuple[str, str, np.ndarray], ...] = ()  # PLY (name, type, (N,) values)
    source: dict[str, Any] = field(default_factory=dict)

    @property
    def vertices_zyx(self) -> np.ndarray:
        """(N, 3) vertices in apok-eval's (z, y, x) order: column 0 is the file's z."""
        return np.ascontiguousarray(self.vertices_xyz[:, ::-1])

    def without_attributes(self) -> TriangleMesh:
        """The same vertices and faces with every texture, normal and extra property dropped."""
        return TriangleMesh(vertices_xyz=self.vertices_xyz, faces=self.faces, source=dict(self.source))

    def as_dict(self) -> dict[str, Any]:
        zyx = self.vertices_zyx.astype(np.float64)
        return {
            "vertices": int(len(self.vertices_xyz)),
            "faces": int(len(self.faces)),
            "coordinate_type": PLY_NAMES[self.vertices_xyz.dtype.str[1:]],
            "axes": ["z", "y", "x"],
            "bounds_zyx": None if not len(zyx) else [[float(v) for v in zyx.min(axis=0)],
                                                      [float(v) for v in zyx.max(axis=0)]],
            "texcoords": None if self.texcoords is None else list(self.texcoords.shape),
            "normals": None if self.normals is None else list(self.normals.shape),
            "vertex_properties": [[name, kind] for name, kind, _ in self.vertex_properties],
            # A path is written as the tifxyz summary writes it: bytes that are not UTF-8 as \xNN.
            "source": {k: portable_name(v) if k == "path" else v for k, v in self.source.items()},
        }


def read_mesh(path: str | os.PathLike[str]) -> TriangleMesh:
    """Read ``path`` as OBJ or PLY, chosen by its extension (``.obj`` / ``.ply``, any case)."""
    name = os.fspath(path)
    if "\x00" in name:
        raise Refusal("IO_MESH_FILE_MISSING", f"{name!r} is not a valid path (it holds a NUL byte)", path=name,
                      problem="not a valid path")
    ext = os.path.splitext(name)[1].lower()
    if ext not in (".obj", ".ply"):
        raise Refusal("IO_MESH_FORMAT_UNKNOWN", f"{name}: extension {ext!r} is neither .obj nor .ply", path=name)
    return read_obj(name) if ext == ".obj" else read_ply(name)


def _read_bytes(path: str) -> bytes:
    """The bytes of the regular file ``path``; refused when nothing is there or it is no regular file (a
    directory, FIFO or device is never opened for reading). Other I/O errors propagate as ``OSError``."""
    try:
        return read_regular(path)
    except NotRegularFile as exc:
        problem = "missing" if exc.missing else f"not a regular file ({exc.kind})"
        raise Refusal("IO_MESH_FILE_MISSING", f"{path}: {exc.problem}", path=path, problem=problem) from None
    except ValueError:  # a NUL byte: os.open refuses the path itself
        raise Refusal("IO_MESH_FILE_MISSING", f"{path!r} is not a valid path", path=path,
                      problem="not a valid path") from None


# ---------------------------------------------------------------------------------------------- OBJ


def read_obj(path: str | os.PathLike[str]) -> TriangleMesh:
    name = os.fspath(path)
    data = _read_bytes(name)
    mesh = parse_obj(data, name)
    mesh.source.update(path=name, digest=sha256_bytes(data))
    return mesh


class _ObjFaces:
    """Accumulates OBJ faces, resolving indices and holding the file's one corner form."""

    def __init__(self) -> None:
        self.v: list[list[int]] = []
        self.vt: list[list[int]] = []
        self.vn: list[list[int]] = []
        self.lines: list[int] = []
        self.form: tuple[bool, bool] | None = None


def parse_obj(data: bytes, name: str = "<obj>") -> TriangleMesh:
    """Parse OBJ bytes (see the module docstring for the accepted subset)."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Refusal("IO_MESH_OBJ_SYNTAX", f"{name}: not UTF-8 text (byte {exc.start})", path=name,
                      line=None) from None
    vertices: list[list[float]] = []
    texcoords: list[list[float]] = []
    normals: list[list[float]] = []
    faces = _ObjFaces()
    ignored: dict[str, int] = {}
    comments = 0
    for line_no, raw in enumerate(text.split("\n"), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            comments += line.startswith("#")
            continue
        keyword, *args = line.split()
        if keyword in _GEOMETRY and "#" in line:
            raise Refusal("IO_MESH_OBJ_SYNTAX", f"{name}:{line_no}: {keyword} statement carries an inline comment "
                          f"({line[line.index('#'):]!r}); comments must be lines of their own", path=name,
                          line=line_no)
        if keyword == "v":
            vertices.append(_obj_numbers(name, line_no, keyword, args, (3,)))
        elif keyword == "vt":
            values = _obj_numbers(name, line_no, keyword, args, (2, 3))
            if texcoords and len(values) != len(texcoords[0]):
                raise Refusal("IO_MESH_OBJ_SYNTAX", f"{name}:{line_no}: vt has {len(values)} numbers, earlier "
                              f"vt lines {len(texcoords[0])}", path=name, line=line_no)
            texcoords.append(values)
        elif keyword == "vn":
            normals.append(_obj_numbers(name, line_no, keyword, args, (3,)))
        elif keyword == "f":
            _obj_face(name, line_no, args, faces, (len(vertices), len(texcoords), len(normals)))
        elif keyword in _OBJ_IGNORED:
            ignored[keyword] = ignored.get(keyword, 0) + 1
        else:
            raise Refusal("IO_MESH_OBJ_UNSUPPORTED", f"{name}:{line_no}: statement {keyword!r} is not read by "
                          "apok-eval (only v, vt, vn, f; o/g/s/usemtl/mtllib are ignored)", path=name, line=line_no,
                          keyword=keyword)
    has_vt, has_vn = faces.form or (False, False)
    for kind, indices, total, used in (("v", faces.v, len(vertices), True), ("vt", faces.vt, len(texcoords), has_vt),
                                       ("vn", faces.vn, len(normals), has_vn)):
        if used:
            _obj_check_totals(name, kind, indices, faces.lines, total)
    return TriangleMesh(
        vertices_xyz=np.array(vertices, dtype=np.float64).reshape(-1, 3),
        faces=np.array(faces.v, dtype=np.int64).reshape(-1, 3),
        texcoords=np.array(texcoords, dtype=np.float64) if texcoords else None,
        normals=np.array(normals, dtype=np.float64).reshape(-1, 3) if normals else None,
        face_texcoords=np.array(faces.vt, dtype=np.int64).reshape(-1, 3) if has_vt else None,
        face_normals=np.array(faces.vn, dtype=np.int64).reshape(-1, 3) if has_vn else None,
        source={"format": "obj", "comments": comments, "ignored": ignored},
    )


def _obj_numbers(name: str, line_no: int, keyword: str, args: list[str], counts: tuple[int, ...]) -> list[float]:
    if len(args) not in counts:
        raise Refusal("IO_MESH_OBJ_SYNTAX", f"{name}:{line_no}: {keyword} takes {' or '.join(map(str, counts))} "
                      f"numbers, got {len(args)}", path=name, line=line_no)
    values = []
    for token in args:
        if not _NUMBER.match(token):
            raise Refusal("IO_MESH_OBJ_SYNTAX", f"{name}:{line_no}: {token!r} is not a decimal number",
                          path=name, line=line_no)
        value = float(token)
        if not math.isfinite(value):
            raise Refusal("IO_MESH_VALUE_INVALID", f"{name}:{line_no}: {token} overflows a double", path=name,
                          line=line_no)
        values.append(value)
    return values


def _obj_face(name: str, line_no: int, args: list[str], faces: _ObjFaces, defined: tuple[int, int, int]) -> None:
    if len(args) != 3:
        raise Refusal("IO_MESH_FACE_NOT_TRIANGLE", f"{name}:{line_no}: face has {len(args)} vertices; only "
                      "triangles are read (nothing is triangulated)", path=name, line=line_no, count=len(args))
    corners = [_obj_corner(name, line_no, token) for token in args]
    for corner in corners:
        form = (corner[1] is not None, corner[2] is not None)
        if faces.form is None:
            faces.form = form
        if form != faces.form:
            raise Refusal("IO_MESH_FACE_FORM_MIXED", f"{name}:{line_no}: face corner form {_form_text(form)} "
                          f"differs from the file's {_form_text(faces.form)}", path=name, line=line_no)
    resolved = [[_obj_index(name, line_no, kind, c[slot], defined[slot]) for c in corners]
                for slot, kind in enumerate(("v", "vt", "vn")) if corners[0][slot] is not None]
    faces.v.append(resolved[0])
    if faces.form[0]:
        faces.vt.append(resolved[1])
    if faces.form[1]:
        faces.vn.append(resolved[-1])
    faces.lines.append(line_no)


def _form_text(form: tuple[bool, bool]) -> str:
    return {(False, False): "v", (True, False): "v/vt", (False, True): "v//vn", (True, True): "v/vt/vn"}[form]


def _obj_corner(name: str, line_no: int, token: str) -> tuple[int, int | None, int | None]:
    parts = token.split("/")
    ok = len(parts) <= 3 and _INTEGER.match(parts[0]) is not None
    if len(parts) == 2:
        ok = ok and _INTEGER.match(parts[1]) is not None
    if len(parts) == 3:
        ok = ok and (parts[1] == "" or _INTEGER.match(parts[1]) is not None) and _INTEGER.match(parts[2]) is not None
    if not ok:
        raise Refusal("IO_MESH_OBJ_SYNTAX", f"{name}:{line_no}: face corner {token!r} is not v, v/vt, v//vn or "
                      "v/vt/vn", path=name, line=line_no)
    values = [_bounded_int(part) if part else None for part in parts]
    for kind, part, value in zip(("v", "vt", "vn"), parts, values):
        if part and value is None:
            digits = len(part.lstrip("+-").lstrip("0"))
            raise Refusal("IO_MESH_INDEX_OUT_OF_RANGE", f"{name}:{line_no}: {kind} index of {digits} significant "
                          "digits exceeds any element count", path=name, line=line_no, kind=kind, index=None,
                          digits=digits)
    vt = values[1] if len(parts) > 1 else None
    vn = values[2] if len(parts) == 3 else None
    return values[0], vt, vn  # type: ignore[return-value]


def _bounded_int(token: str) -> int | None:
    """The value of an ASCII integer token, or None past 19 significant digits (leading zeros are not counted)."""
    digits = token.lstrip("+-").lstrip("0")
    if len(digits) > _MAX_INTEGER_DIGITS:
        return None
    return -int(digits or "0") if token.startswith("-") else int(digits or "0")


def _obj_index(name: str, line_no: int, kind: str, index: int, defined: int) -> int:
    # Positive indices are 1-based and checked against the file's totals once it is read;
    # negative ones count back from the elements defined so far.
    resolved = index - 1 if index > 0 else defined + index
    if index == 0 or resolved < 0:
        raise Refusal("IO_MESH_INDEX_OUT_OF_RANGE", f"{name}:{line_no}: {kind} index {index} does not exist "
                      f"({defined} defined so far)", path=name, line=line_no, kind=kind, index=index)
    return resolved


def _obj_check_totals(name: str, kind: str, indices: list[list[int]], lines: list[int], total: int) -> None:
    for face, line_no in zip(indices, lines, strict=True):
        if max(face) >= total:
            raise Refusal("IO_MESH_INDEX_OUT_OF_RANGE", f"{name}:{line_no}: {kind} index {max(face) + 1} exceeds "
                          f"the {total} defined", path=name, line=line_no, kind=kind, index=max(face) + 1)


# ---------------------------------------------------------------------------------------------- PLY


@dataclass(frozen=True)
class _Property:
    name: str
    type: str  # numpy code for a scalar, or the item code of a list
    count_type: str | None = None  # numpy code of a list's length prefix


@dataclass
class _Element:
    name: str
    count: int
    properties: list[_Property] = field(default_factory=list)


def read_ply(path: str | os.PathLike[str]) -> TriangleMesh:
    name = os.fspath(path)
    data = _read_bytes(name)
    mesh = parse_ply(data, name)
    mesh.source.update(path=name, digest=sha256_bytes(data))
    return mesh


def parse_ply(data: bytes, name: str = "<ply>") -> TriangleMesh:
    """Parse PLY bytes (see the module docstring for the accepted subset)."""
    encoding, elements, notes, body = _ply_header(data, name)
    vertex, face = _ply_layout(name, elements)
    if encoding == "ascii":
        columns, corners = _ply_ascii_body(name, body, vertex, face)
    else:
        columns, corners = _ply_binary_body(name, body, vertex, face)
    xyz = np.stack([columns["x"], columns["y"], columns["z"]], axis=1)
    bad = ~np.isfinite(xyz).all(axis=1)
    if bad.any():
        index = int(np.argmax(bad))
        raise Refusal("IO_MESH_VALUE_INVALID", f"{name}: vertex {index} has a non-finite coordinate",
                      path=name, vertex=index)
    faces = corners.astype(np.int64)
    out_of_range = ((faces < 0) | (faces >= vertex.count)).any(axis=1)
    if out_of_range.any():
        index = int(np.argmax(out_of_range))
        raise Refusal("IO_MESH_INDEX_OUT_OF_RANGE", f"{name}: face {index} refers to vertex "
                      f"{faces[index].tolist()} of {vertex.count}", path=name, face=index, kind="v")
    extras = tuple((p.name, PLY_NAMES[p.type], columns[p.name]) for p in vertex.properties
                   if p.name not in ("x", "y", "z"))
    return TriangleMesh(vertices_xyz=xyz, faces=faces, vertex_properties=extras,
                        source={"format": "ply", "encoding": encoding, **notes})


def _ply_header(data: bytes, name: str) -> tuple[str, list[_Element], dict[str, list[str]], bytes]:
    lines, body_at = [], None
    pos = 0
    while pos < len(data):
        end = data.find(b"\n", pos)
        end = len(data) if end < 0 else end
        line = data[pos:end].rstrip(b"\r")
        pos = end + 1
        lines.append(line)
        if line == b"end_header":
            body_at = pos
            break
    if not lines or lines[0] != b"ply" or body_at is None:
        raise Refusal("IO_MESH_PLY_HEADER", f"{name}: a PLY file starts with 'ply' and its header ends with "
                      "'end_header'", path=name)
    encoding: str | None = None
    elements: list[_Element] = []
    notes: dict[str, list[str]] = {"comments": [], "obj_info": []}
    for line_no, raw in enumerate(lines[1:-1], start=2):
        try:
            text = raw.decode("ascii")
        except UnicodeDecodeError:
            raise Refusal("IO_MESH_PLY_HEADER", f"{name}: header line {line_no} is not ASCII", path=name,
                          line=line_no) from None
        keyword, _, rest = text.partition(" ")
        words = rest.split()
        if keyword in ("comment", "obj_info"):
            notes["comments" if keyword == "comment" else "obj_info"].append(rest)
        elif keyword == "format" and encoding is None and len(words) == 2:
            if (words[0], words[1]) not in (("ascii", "1.0"), ("binary_little_endian", "1.0")):
                raise Refusal("IO_MESH_PLY_UNSUPPORTED", f"{name}: format {rest!r} is not ascii 1.0 or "
                              "binary_little_endian 1.0", path=name, format=rest)
            encoding = words[0]
        elif keyword == "element" and len(words) == 2 and words[1].isdigit():
            elements.append(_Element(words[0], int(words[1])))
        elif keyword == "property" and elements and len(words) in (2, 4) and (len(words) == 2) != (words[0] == "list"):
            elements[-1].properties.append(_ply_property(name, line_no, words))
        else:
            raise Refusal("IO_MESH_PLY_HEADER", f"{name}: header line {line_no} {text!r} is not understood",
                          path=name, line=line_no)
    if encoding is None:
        raise Refusal("IO_MESH_PLY_HEADER", f"{name}: the header declares no format", path=name, line=None)
    return encoding, elements, notes, data[body_at:]


def _ply_property(name: str, line_no: int, words: list[str]) -> _Property:
    types = words[1:3] if words[0] == "list" else words[:1]
    for type_name in types:
        if type_name not in PLY_TYPES:
            raise Refusal("IO_MESH_PLY_PROPERTY_TYPE", f"{name}: header line {line_no}: unknown property type "
                          f"{type_name!r}", path=name, line=line_no, type=type_name)
    if words[0] == "list":
        return _Property(words[3], PLY_TYPES[words[2]], PLY_TYPES[words[1]])
    return _Property(words[1], PLY_TYPES[words[0]])


def _ply_layout(name: str, elements: list[_Element]) -> tuple[_Element, _Element]:
    if [e.name for e in elements] != ["vertex", "face"]:
        raise Refusal("IO_MESH_PLY_LAYOUT", f"{name}: elements are {[e.name for e in elements]}, not "
                      "['vertex', 'face']", path=name, elements=[e.name for e in elements])
    vertex, face = elements
    names = [p.name for p in vertex.properties]
    if any(p.count_type is not None for p in vertex.properties) or len(set(names)) != len(names) \
            or not {"x", "y", "z"} <= set(names):
        raise Refusal("IO_MESH_PLY_LAYOUT", f"{name}: vertex properties {names} must be distinct scalars "
                      "including x, y and z", path=name, element="vertex", properties=names)
    coordinate_types = {p.type for p in vertex.properties if p.name in ("x", "y", "z")}
    if coordinate_types not in ({"f4"}, {"f8"}):
        raise Refusal("IO_MESH_PLY_PROPERTY_TYPE", f"{name}: x, y and z must all be float or all double, not "
                      f"{sorted(PLY_NAMES[t] for t in coordinate_types)}", path=name, property="x/y/z")
    if len(face.properties) != 1 or face.properties[0].name != "vertex_indices" \
            or face.properties[0].count_type is None:
        raise Refusal("IO_MESH_PLY_LAYOUT", f"{name}: the face element must hold exactly one list property "
                      "named vertex_indices", path=name, element="face",
                      properties=[p.name for p in face.properties])
    indices = face.properties[0]
    if indices.count_type != "u1" or indices.type not in ("i4", "u4"):
        raise Refusal("IO_MESH_PLY_PROPERTY_TYPE", f"{name}: vertex_indices must be list uchar int|uint, not list "
                      f"{PLY_NAMES[indices.count_type]} {PLY_NAMES[indices.type]}", path=name,
                      property="vertex_indices")
    return vertex, face


def _ply_binary_body(name: str, body: bytes, vertex: _Element, face: _Element
                     ) -> tuple[dict[str, np.ndarray], np.ndarray]:
    vertex_dtype = np.dtype([(p.name, "<" + p.type) for p in vertex.properties])
    need = vertex.count * vertex_dtype.itemsize
    if len(body) < need:
        raise Refusal("IO_MESH_TRUNCATED", f"{name}: {len(body)} data bytes, {need} needed for "
                      f"{vertex.count} vertices", path=name, element="vertex")
    records = np.frombuffer(body, vertex_dtype, count=vertex.count)
    columns = {p.name: records[p.name].astype(p.type) for p in vertex.properties}
    face_dtype = np.dtype([("n", "u1"), ("i", "<" + face.properties[0].type, (3,))])
    available = min(face.count, (len(body) - need) // face_dtype.itemsize)
    faces = np.frombuffer(body, face_dtype, count=available, offset=need)
    _ply_require_triangles(name, faces["n"])
    if available < face.count:
        raise Refusal("IO_MESH_TRUNCATED", f"{name}: data ends after {available} of {face.count} faces",
                      path=name, element="face")
    end = need + face.count * face_dtype.itemsize
    if end != len(body):
        raise Refusal("IO_MESH_TRAILING_DATA", f"{name}: {len(body) - end} bytes follow the last face",
                      path=name, extra_bytes=len(body) - end)
    return columns, faces["i"]


def _ply_require_triangles(name: str, counts: np.ndarray, first: int = 0) -> None:
    """Refuse at the first entry of ``counts`` (face ``first`` onwards) that is not 3."""
    bad = counts != 3
    if bad.any():
        index = first + int(np.argmax(bad))
        count = int(counts[index - first])
        raise Refusal("IO_MESH_FACE_NOT_TRIANGLE", f"{name}: face {index} has {count} vertices; "
                      "only triangles are read (nothing is triangulated)", path=name, face=index, count=count)


def _ply_ascii_body(name: str, body: bytes, vertex: _Element, face: _Element
                    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
    # Values are gathered in lists, never in arrays sized by the header: a count of 10**15 in a tiny
    # file must end as IO_MESH_TRUNCATED, not as a petabyte allocation.
    tokens = iter(body.split())
    values: dict[str, list[int | float]] = {p.name: [] for p in vertex.properties}
    for i in range(vertex.count):
        for p in vertex.properties:
            values[p.name].append(_ascii_value(name, _next_token(name, tokens, "vertex"), p.type, f"vertex {i}"))
    corners: list[list[int | float]] = []
    for j in range(face.count):
        count = _ascii_value(name, _next_token(name, tokens, "face"), "u1", f"face {j}")
        _ply_require_triangles(name, np.array([count]), j)
        corners.append([_ascii_value(name, _next_token(name, tokens, "face"), face.properties[0].type, f"face {j}")
                        for _ in range(3)])
    leftover = sum(1 for _ in tokens)
    if leftover:
        raise Refusal("IO_MESH_TRAILING_DATA", f"{name}: {leftover} values follow the last face", path=name,
                      extra_values=leftover)
    columns = {p.name: np.array(values[p.name], dtype=p.type) for p in vertex.properties}
    return columns, np.array(corners, dtype=face.properties[0].type).reshape(-1, 3)


def _next_token(name: str, tokens: Iterator[bytes], element: str) -> bytes:
    token = next(tokens, None)
    if token is None:
        raise Refusal("IO_MESH_TRUNCATED", f"{name}: data ends inside the {element} element", path=name,
                      element=element)
    return token


def _ascii_value(name: str, token: bytes, code: str, where: str) -> int | float:
    text = token.decode("ascii", "replace")
    value: int | float | None
    if code[0] == "f":
        # A decimal that overflows its declared precision would silently become infinity.
        with np.errstate(over="ignore"):
            value = float(text) if _NUMBER.match(text) else None
            ok = value is not None and bool(np.isfinite(np.dtype(code).type(value)))
    else:
        info = np.iinfo(code)
        value = _bounded_int(text) if _INTEGER.match(text) else None
        ok = value is not None and info.min <= value <= info.max
    if not ok:
        raise Refusal("IO_MESH_VALUE_INVALID", f"{name}: {where}: {text!r} is not a finite {PLY_NAMES[code]}",
                      path=name, where=where, value=text)
    return value  # type: ignore[return-value]


# ------------------------------------------------------------------------------------------- writers


def _check_mesh(mesh: TriangleMesh) -> None:
    """Refuse a mesh whose arrays are not the shapes, types and ranges the writers assume."""
    v = mesh.vertices_xyz
    if not isinstance(v, np.ndarray) or v.ndim != 2 or v.shape[1] != 3 or v.dtype.str[1:] not in ("f4", "f8") \
            or not np.isfinite(v).all():
        raise Refusal("IO_MESH_WRITE_INVALID", "vertices_xyz must be a finite (N, 3) float32/float64 array",
                      field="vertices_xyz")
    _check_attribute("texcoords", mesh.texcoords, (2, 3))
    _check_attribute("normals", mesh.normals, (3,))
    for field_name, indices, target in (("faces", mesh.faces, v), ("face_texcoords", mesh.face_texcoords,
                                                                      mesh.texcoords),
                                        ("face_normals", mesh.face_normals, mesh.normals)):
        if indices is None and field_name != "faces":
            continue
        size = -1 if target is None else len(target)
        if not isinstance(indices, np.ndarray) or indices.ndim != 2 or indices.shape[1] != 3 \
                or indices.dtype.kind not in "iu" or (len(indices) and (indices.min() < 0 or indices.max() >= size)) \
                or len(indices) != len(mesh.faces):
            raise Refusal("IO_MESH_WRITE_INVALID", f"{field_name} must be an (F, 3) integer array indexing an "
                          "existing array, one row per face", field=field_name)


def _check_attribute(field_name: str, values: np.ndarray | None, widths: tuple[int, ...]) -> None:
    if values is None:
        return
    if not isinstance(values, np.ndarray) or values.ndim != 2 or values.shape[1] not in widths \
            or values.dtype.kind != "f" or not np.isfinite(values).all():
        raise Refusal("IO_MESH_WRITE_INVALID", f"{field_name} must be a finite float array with "
                      f"{' or '.join(map(str, widths))} columns", field=field_name)


def _obj_number(value: float) -> str:
    return repr(float(value))


def encode_obj(mesh: TriangleMesh) -> bytes:
    """Deterministic OBJ text: v, vt, vn, then f (1-based, one form), numbers as shortest round-trip reprs."""
    _check_mesh(mesh)
    if mesh.vertex_properties:
        raise Refusal("IO_MESH_WRITE_INVALID", "OBJ cannot carry vertex properties "
                      f"{[p[0] for p in mesh.vertex_properties]}; drop them explicitly (without_attributes)",
                      field="vertex_properties")
    lines = ["v " + " ".join(_obj_number(c) for c in row) for row in mesh.vertices_xyz]
    lines += ["vt " + " ".join(_obj_number(c) for c in row) for row in (mesh.texcoords if mesh.texcoords is not None
                                                                        else ())]
    lines += ["vn " + " ".join(_obj_number(c) for c in row) for row in (mesh.normals if mesh.normals is not None
                                                                        else ())]
    for i, face in enumerate(mesh.faces):
        corners = []
        for k in range(3):
            corner = str(int(face[k]) + 1)
            if mesh.face_texcoords is not None:
                corner += f"/{int(mesh.face_texcoords[i, k]) + 1}"
            if mesh.face_normals is not None:
                corner += ("/" if mesh.face_texcoords is not None else "//") + str(int(mesh.face_normals[i, k]) + 1)
            corners.append(corner)
        lines.append("f " + " ".join(corners))
    return "".join(line + "\n" for line in lines).encode("ascii")


def encode_ply(mesh: TriangleMesh) -> bytes:
    """Deterministic binary_little_endian PLY: x, y, z (float or double), vertex properties, then faces."""
    vertices = mesh.vertices_xyz
    if isinstance(vertices, np.ndarray) and vertices.ndim >= 1 and vertices.shape[0] > 2 ** 31:
        # Checked before anything scans the vertices: indices are declared as int32.
        raise Refusal("IO_MESH_WRITE_INVALID", f"{vertices.shape[0]} vertices cannot be indexed by the int32 "
                      "vertex_indices the PLY writer declares", field="faces")
    _check_mesh(mesh)
    for field_name in ("texcoords", "normals", "face_texcoords", "face_normals"):
        if getattr(mesh, field_name) is not None:
            raise Refusal("IO_MESH_WRITE_INVALID", f"PLY as written here has no {field_name}; drop them "
                          "explicitly (without_attributes)", field=field_name)
    n = len(mesh.vertices_xyz)
    coordinate = mesh.vertices_xyz.dtype.str[1:]
    fields = [("x", "<" + coordinate), ("y", "<" + coordinate), ("z", "<" + coordinate)]
    for prop_name, prop_type, values in mesh.vertex_properties:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", prop_name) or prop_name in [f[0] for f in fields] \
                or PLY_TYPES.get(prop_type) is None or not isinstance(values, np.ndarray) \
                or values.dtype.str[1:] != PLY_TYPES[prop_type] or values.shape != (n,):
            raise Refusal("IO_MESH_WRITE_INVALID", f"vertex property {prop_name!r} ({prop_type}) must be a unique "
                          "identifier with an (N,) array of exactly its declared type", field="vertex_properties",
                          property=prop_name)
        fields.append((prop_name, "<" + PLY_TYPES[prop_type]))
    records = np.empty(n, dtype=np.dtype(fields))
    for axis, column in enumerate("xyz"):
        records[column] = mesh.vertices_xyz[:, axis]
    for prop_name, _, values in mesh.vertex_properties:
        records[prop_name] = values
    faces = np.empty(len(mesh.faces), dtype=np.dtype([("n", "u1"), ("i", "<i4", (3,))]))
    faces["n"] = 3
    faces["i"] = mesh.faces
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    header += [f"property {PLY_NAMES[code[1:]]} {field_name}" for field_name, code in fields]
    header += [f"element face {len(mesh.faces)}", "property list uchar int vertex_indices", "end_header"]
    return "".join(line + "\n" for line in header).encode("ascii") + records.tobytes() + faces.tobytes()


def _write(path: str | os.PathLike[str], data: bytes, overwrite: bool) -> str:
    target = os.fspath(path)
    if os.path.lexists(target) and not overwrite:
        raise Refusal("IO_MESH_WRITE_EXISTS", f"{target} exists; pass overwrite=True to replace it", path=target)
    write_atomic(target, data, overwrite=overwrite)
    return sha256_bytes(data)


def write_obj(path: str | os.PathLike[str], mesh: TriangleMesh, *, overwrite: bool = False) -> str:
    """Write :func:`encode_obj` bytes to ``path``; return their digest."""
    return _write(path, encode_obj(mesh), overwrite)


def write_ply(path: str | os.PathLike[str], mesh: TriangleMesh, *, overwrite: bool = False) -> str:
    """Write :func:`encode_ply` bytes to ``path``; return their digest."""
    return _write(path, encode_ply(mesh), overwrite)


def check_mesh_within(mesh: TriangleMesh, frame: VolumeFrame, what: str = "mesh") -> None:
    """Refuse unless every vertex lies inside the declared ``frame`` (compared in z, y, x)."""
    check_points_within(frame, mesh.vertices_zyx.astype(np.float64), what)


def inspect_mesh(path: str | os.PathLike[str]) -> dict[str, Any]:
    """The summary ``apok-eval io inspect`` prints (it reads once and summarises the object it checked)."""
    return read_mesh(path).as_dict()
