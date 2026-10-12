"""VC3D point collections ("PointCollections v1"): a strict reader, a deterministic writer, the fitter's role rule.

Document (villa volume-cartographer/core/src/PointCollections.cpp and
spiral-fitting/point_collection.py; ``docs/io-surfaces.md`` cites lines): one
JSON object with ``vc_pointcollections_json_version`` equal to the **string**
"1" and ``collections``, an object keyed by canonical decimal uint64 ids
("0" or no leading zero, no sign, no spaces, no underscores, < 2**64). Each
collection has a string ``name`` and ``points``, an object keyed by canonical
uint64 point ids; ``metadata``, when present, is an object. Each point has
``p`` = [x, y, z] (three finite numbers), ``wind_a`` (a finite number, or
null/absent meaning unannotated, held as NaN) and ``creation_time`` (an int64;
absent means 0). Every other key at every level (links, fiber_dir, tags,
windings_linked, anchor2d, autoFillMode, autoFillConstant, color,
vc_open_data_* stamps, ...) is kept verbatim in ``document`` and not
interpreted. NaN/Infinity tokens, duplicate keys and numbers that overflow a
double are refused: VC3D's nlohmann::json would not read such a file.

Points are exposed in (z, y, x) order. A point's identity is (file digest,
collection id, point id): ids are not file-global in practice. Within a
collection the fitter's chain order is ascending point id.

Role (as spiral-fitting/fit_spiral.py:759-769 derives it): absolute iff the
declared role is "absolute", or, for a role-less file, iff its basename is
exactly ``abs_winding.json``; a role-less file of any other name is "legacy"
(relative). ``metadata.winding_is_absolute`` is not authoritative; where VC3D
would read it differently (missing or non-boolean reads as true) a finding is
recorded. Absolute files must give every point a finite ``wind_a`` > 0.
Non-absolute files get an annotation report (what the fitter would keep,
zero-fill or strip); nothing is applied. The writer takes the same role
argument and refuses an absolute file the reader would refuse.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from apok_eval import jsonstrict
from apok_eval.canonical import sha256_bytes
from apok_eval.io.frame import VolumeFrame, check_points_within, finite_number
from apok_eval.fileio import NotRegularFile, portable_name, read_regular, write_atomic
from apok_eval.refusal import Refusal, register_codes

register_codes(
    "apok_eval.io.pointcollection",
    {
        "IO_PCL_FILE_MISSING": "a point-collection file does not exist or is not a regular file",
        "IO_PCL_JSON_INVALID": "the file is not strict JSON (UTF-8, syntax, duplicate keys, overflow, NaN tokens)",
        "IO_PCL_VERSION": "vc_pointcollections_json_version is not the string \"1\"",
        "IO_PCL_STRUCTURE": "the document, a collection, or a required key has the wrong JSON type",
        "IO_PCL_ID_NONCANONICAL": "a collection or point key is not a canonical decimal uint64",
        "IO_PCL_POINT_INVALID": "a point's p, wind_a or creation_time is malformed",
        "IO_PCL_ROLE_UNKNOWN": "a declared role is not one of the fitter's PCL roles",
        "IO_PCL_ABSOLUTE_WINDING": "a point of an absolute-role file lacks a finite wind_a > 0",
        "IO_PCL_WRITE_INVALID": "a document given to the writer is not JSON the strict reader accepts",
        "IO_PCL_WRITE_EXISTS": "the output file exists and overwrite was not requested",
    },
)

VERSION_KEY = "vc_pointcollections_json_version"
#: The fitter's PCL roles (spiral-fitting/fit_session.py:163-167).
ROLES = ("absolute", "relative", "same_winding", "drawn_control_points")
ABSOLUTE_BASENAME = "abs_winding.json"
#: Conventional file name per role (spiral-fitting/fit_session.py:418-423).
CONVENTIONAL_BASENAMES = {"abs_winding.json": "absolute", "relative_windings.json": "relative",
                          "same_windings.json": "same_winding", "drawn_control_points.json": "drawn_control_points"}
# At most 20 digits (2**64 - 1 has 20), which also keeps int() clear of its 4300-digit limit.
_UINT64 = re.compile(r"(?:0|[1-9][0-9]{0,19})\Z")
_UINT64_MAX = 2 ** 64 - 1
_INT64 = (-(2 ** 63), 2 ** 63 - 1)


@dataclass(frozen=True, eq=False)
class PointCollection:
    """One collection; arrays are parallel to ``point_ids`` (ascending, the fitter's chain order)."""

    collection_id: int
    name: str
    point_ids: tuple[int, ...]
    zyx: np.ndarray  # (n, 3) float64 from p = [x, y, z]
    wind_a: np.ndarray  # (n,) float64, NaN where null or absent
    creation_time: tuple[int, ...]
    metadata: dict[str, Any] | None


@dataclass(frozen=True, eq=False)
class PointCollectionFile:
    path: str
    basename: str
    digest: str
    document: dict[str, Any]  # as parsed: every key and value kept
    declared_role: str | None
    role: str  # "absolute", the declared role, or "legacy"
    absolute: bool
    collections: tuple[PointCollection, ...]  # ascending collection id
    annotations: tuple[dict[str, Any], ...]  # non-absolute files only
    findings: tuple[dict[str, Any], ...]

    def identities(self) -> list[tuple[str, int, int]]:
        """(file digest, collection id, point id) of every point, in :meth:`points_zyx` order."""
        return [(self.digest, c.collection_id, pid) for c in self.collections for pid in c.point_ids]

    def points_zyx(self) -> np.ndarray:
        return np.concatenate([c.zyx for c in self.collections]) if self.collections else np.zeros((0, 3))

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": portable_name(self.path),  # bytes that are not UTF-8 as \xNN, as the tifxyz summary
            "basename": portable_name(self.basename),
            "digest": self.digest,
            "declared_role": self.declared_role,
            "role": self.role,
            "absolute": self.absolute,
            "axes": ["z", "y", "x"],
            "collections": [{"id": c.collection_id, "name": c.name, "points": len(c.point_ids),
                             "annotated": int(np.isfinite(c.wind_a).sum())} for c in self.collections],
            "top_level_keys": sorted(self.document),
            "annotations": list(self.annotations),
            "findings": list(self.findings),
        }


def derive_role(basename: str, declared_role: str | None) -> tuple[str, bool]:
    """(role, absolute) exactly as the fitter stamps them; refuses an unknown declared role."""
    if declared_role is not None:
        if declared_role not in ROLES:
            raise Refusal("IO_PCL_ROLE_UNKNOWN", f"role {declared_role!r} is not one of {list(ROLES)} "
                          "(None means role-less)", role=declared_role)
        return declared_role, declared_role == "absolute"
    absolute = basename == ABSOLUTE_BASENAME
    return ("absolute" if absolute else "legacy"), absolute


def read_point_collections(path: str | os.PathLike[str], *, role: str | None) -> PointCollectionFile:
    """Read one PointCollections file. ``role`` is required: a fitter role, or None for a role-less file."""
    name = os.fspath(path)
    basename = os.path.basename(name)
    derived, absolute = derive_role(basename, role)
    try:
        data = read_regular(name)
    except NotRegularFile as exc:
        problem = "missing" if exc.missing else f"not a regular file ({exc.kind})"
        why = "does not exist (the fitter would skip it with a warning)" if exc.missing else f"is {exc.kind}, " \
            "not a regular file"
        raise Refusal("IO_PCL_FILE_MISSING", f"{name} {why}", path=name, problem=problem) from None
    except ValueError:  # a NUL byte: os.open refuses the path itself
        raise Refusal("IO_PCL_FILE_MISSING", f"{name!r} is not a valid path", path=name,
                      problem="not a valid path") from None
    try:
        document = jsonstrict.loads(data)
    except jsonstrict.StrictJsonError as exc:
        raise Refusal("IO_PCL_JSON_INVALID", f"{name}: {exc}", path=name, problem=str(exc)) from None
    collections = parse_point_collections(document, name)
    if absolute:
        _require_absolute_windings(name, collections)
    findings = _role_findings(basename, role, derived, absolute, collections) + _point_findings(document,
                                                                                               collections)
    return PointCollectionFile(
        path=name, basename=basename, digest=sha256_bytes(data), document=document, declared_role=role,
        role=derived, absolute=absolute, collections=collections,
        annotations=() if absolute else tuple(_annotation(c) for c in collections), findings=tuple(findings),
    )


def parse_point_collections(document: Any, where: str) -> tuple[PointCollection, ...]:
    """Check the document's structure and return its collections (ascending id)."""
    if not isinstance(document, dict):
        raise Refusal("IO_PCL_STRUCTURE", f"{where}: the document must be a JSON object", path=where,
                      field="document")
    if document.get(VERSION_KEY) != "1":  # only the string compares equal: a number 1 is refused
        raise Refusal("IO_PCL_VERSION", f"{where}: {VERSION_KEY} must be the string \"1\"", path=where,
                      value=document.get(VERSION_KEY))
    collections = document.get("collections")
    if not isinstance(collections, dict):
        raise Refusal("IO_PCL_STRUCTURE", f"{where}: collections must be an object keyed by id", path=where,
                      field="collections")
    parsed = [_collection(where, _canonical_id(where, "collection", key, None), body)
              for key, body in collections.items()]
    return tuple(sorted(parsed, key=lambda c: c.collection_id))


def _canonical_id(where: str, kind: str, key: str, collection_id: int | None) -> int:
    if not _UINT64.match(key) or int(key) > _UINT64_MAX:
        raise Refusal("IO_PCL_ID_NONCANONICAL", f"{where}: {kind} key {key!r} is not a canonical decimal uint64",
                      path=where, kind=kind, key=key, collection_id=collection_id)
    return int(key)


def _collection(where: str, cid: int, body: Any) -> PointCollection:
    if not isinstance(body, dict):
        raise Refusal("IO_PCL_STRUCTURE", f"{where}: collection {cid} must be an object", path=where,
                      collection_id=cid, field="collection")
    for key, kind in (("name", str), ("points", dict)):
        if not isinstance(body.get(key), kind):
            raise Refusal("IO_PCL_STRUCTURE", f"{where}: collection {cid} {key} must be a JSON "
                          f"{'string' if kind is str else 'object'}", path=where, collection_id=cid, field=key)
    if "metadata" in body and not isinstance(body["metadata"], dict):
        raise Refusal("IO_PCL_STRUCTURE", f"{where}: collection {cid} metadata must be an object", path=where,
                      collection_id=cid, field="metadata")
    points = sorted((_canonical_id(where, "point", key, cid), value) for key, value in body["points"].items())
    zyx, wind, created = [], [], []
    for pid, point in points:
        z, w, t = _point(where, cid, pid, point)
        zyx.append(z)
        wind.append(w)
        created.append(t)
    return PointCollection(collection_id=cid, name=body["name"], point_ids=tuple(pid for pid, _ in points),
                           zyx=np.array(zyx, dtype=np.float64).reshape(-1, 3), wind_a=np.array(wind, dtype=np.float64),
                           creation_time=tuple(created), metadata=body.get("metadata"))


def _point(where: str, cid: int, pid: int, point: Any) -> tuple[list[float], float, int]:
    at = f"{where}: collection {cid} point {pid}"
    if not isinstance(point, dict):
        raise Refusal("IO_PCL_POINT_INVALID", f"{at}: a point must be an object", path=where, collection_id=cid,
                      point_id=pid, field="point")
    p = point.get("p")
    coords = [finite_number(c) for c in p] if isinstance(p, list) and len(p) == 3 else None
    if coords is None or None in coords:
        raise Refusal("IO_PCL_POINT_INVALID", f"{at}: p must be three finite numbers [x, y, z], not {p!r}",
                      path=where, collection_id=cid, point_id=pid, field="p")
    wind = point.get("wind_a")
    wind_a = math.nan if wind is None else finite_number(wind)
    if wind_a is None:
        raise Refusal("IO_PCL_POINT_INVALID", f"{at}: wind_a must be a finite number or null, not {wind!r}",
                      path=where, collection_id=cid, point_id=pid, field="wind_a")
    created = point.get("creation_time", 0)
    if isinstance(created, bool) or not isinstance(created, int) or not _INT64[0] <= created <= _INT64[1]:
        raise Refusal("IO_PCL_POINT_INVALID", f"{at}: creation_time must be an int64, not {created!r}",
                      path=where, collection_id=cid, point_id=pid, field="creation_time")
    return [coords[2], coords[1], coords[0]], wind_a, created


def _require_absolute_windings(where: str, collections: tuple[PointCollection, ...]) -> None:
    for c in collections:
        for pid, wind in zip(c.point_ids, c.wind_a, strict=True):
            if not (math.isfinite(wind) and wind > 0):
                raise Refusal("IO_PCL_ABSOLUTE_WINDING", f"{where}: absolute-role collection {c.collection_id} "
                              f"point {pid} has wind_a {None if math.isnan(wind) else float(wind)}; the fitter "
                              "needs a finite wind_a > 0", path=where, collection_id=c.collection_id,
                              point_id=pid, wind_a=None if math.isnan(wind) else float(wind))


def _annotation(c: PointCollection) -> dict[str, Any]:
    """What the fitter's normalisation would do to one non-absolute collection (point_collection.py:878-900)."""
    annotated = np.isfinite(c.wind_a)
    if not len(annotated):
        state = "empty"
    elif annotated.all():
        state = "all"  # kept as is
    elif not annotated.any():
        state = "none"  # every wind_a set to 0.0: a same-winding collection
    else:
        state = "mixed"  # the unannotated points are deleted
    return {"collection_id": c.collection_id, "state": state, "has_winding_annotations": bool(annotated.any()),
            "unannotated_point_ids": [pid for pid, ok in zip(c.point_ids, annotated, strict=True) if not ok]}


def _looks_absolute(basename: str) -> bool:
    folded = re.sub(r"[^a-z0-9]", "", basename.lower())
    return folded.startswith("abs") and "winding" in folded


def _role_findings(basename: str, declared: str | None, derived: str, absolute: bool,
                   collections: tuple[PointCollection, ...]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if declared is None and not absolute and _looks_absolute(basename):
        findings.append({"finding": "basename-near-miss", "basename": basename,
                         "detail": f"only a role-less file named exactly {ABSOLUTE_BASENAME} is absolute; the "
                                   "fitter treats this one as relative"})
    conventional = CONVENTIONAL_BASENAMES.get(basename)
    if declared is not None and conventional not in (None, declared):
        findings.append({"finding": "role-conflicts-with-basename", "basename": basename, "declared_role": declared,
                         "conventional_role": conventional})
    for c in collections:
        flag = (c.metadata or {}).get("winding_is_absolute", "absent")
        vc3d_reads = flag if isinstance(flag, bool) else True  # VC3D's default (PointCollections.cpp:94-99)
        if vc3d_reads != absolute:
            findings.append({"finding": "winding-flag-disagrees", "collection_id": c.collection_id, "flag": flag,
                             "vc3d_reads_absolute": vc3d_reads, "role": derived})
    return findings


def _point_findings(document: dict[str, Any], collections: tuple[PointCollection, ...]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for c in collections:
        points = document["collections"][str(c.collection_id)]["points"]
        for pid in c.point_ids:
            fiber_dir = points[str(pid)].get("fiber_dir")
            if fiber_dir is not None and fiber_dir not in ("h", "v"):
                findings.append({"finding": "fiber-dir-dropped-by-vc3d", "collection_id": c.collection_id,
                                 "point_id": pid, "value": fiber_dir})
    seen: dict[int, int] = {}
    shared = set()
    for c in collections:
        for pid in c.point_ids:
            if pid in seen:
                shared.add(pid)
            seen[pid] = c.collection_id
    if shared:
        findings.append({"finding": "point-ids-not-file-global", "count": len(shared),
                         "example_point_id": min(shared),
                         "detail": "identity is (file, collection id, point id)"})
    return findings


def encode_point_collections(document: dict[str, Any]) -> bytes:
    """Deterministic bytes: sorted keys (ids as strings, so "10" before "2"), indent 4, trailing newline.

    Checks everything the reader checks except the role rule, which depends on
    the file name: :func:`write_point_collections` applies it.
    """
    return _encode(document)[0]


def _encode(document: dict[str, Any]) -> tuple[bytes, tuple[PointCollection, ...]]:
    try:
        data = jsonstrict.dumps(document)
    except jsonstrict.StrictJsonError as exc:
        raise Refusal("IO_PCL_WRITE_INVALID", f"the document is not JSON the strict reader accepts: {exc}",
                      problem=str(exc)) from None
    return data, parse_point_collections(document, "document given to the writer")


def write_point_collections(path: str | os.PathLike[str], document: dict[str, Any], *, role: str | None,
                            overwrite: bool = False) -> str:
    """Write :func:`encode_point_collections` bytes to ``path``; return their digest.

    ``role`` is required, as for :func:`read_point_collections`: when it (or,
    role-less, the basename ``abs_winding.json``) makes the file absolute,
    every point needs a finite ``wind_a`` > 0, as the reader demands.
    """
    target = os.fspath(path)
    _, absolute = derive_role(os.path.basename(target), role)
    data, collections = _encode(document)
    if absolute:
        _require_absolute_windings(target, collections)
    if os.path.lexists(target) and not overwrite:
        raise Refusal("IO_PCL_WRITE_EXISTS", f"{target} exists; pass overwrite=True to replace it", path=target)
    write_atomic(target, data, overwrite=overwrite)
    return sha256_bytes(data)


def check_point_collections_within(pcl: PointCollectionFile, frame: VolumeFrame) -> int:
    """Refuse unless every point lies inside the declared ``frame``; return the number of points."""
    points = pcl.points_zyx()
    check_points_within(frame, points, f"point collections {pcl.basename}")
    return int(len(points))


def inspect_point_collections(path: str | os.PathLike[str], *, role: str | None) -> dict[str, Any]:
    """The summary ``apok-eval io inspect`` prints (it reads once and summarises the object it checked)."""
    return read_point_collections(path, role=role).as_dict()
