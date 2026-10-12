"""``apok-eval io inspect PATH [--frame FRAME.json | --no-frame] [--policy NAME] [--json]``.

Says what a local input is, reads it with its module's public functions and
prints a summary: a short text rendering, or with ``--json`` the whole record
(``apok-io-inspect-v1``). A refusal propagates to :func:`apok_eval.cli.main`, which
prints it as JSON and exits 3. Local paths only: a URI (``s3://``,
``https://``, ...) is refused, because a remote inspect is out of scope.

**Detection** is explicit, by markers, and refuses rather than guesses:

* ``tifxyz``: a directory with an entry ``x.tif``, ``y.tif`` or ``z.tif``;
* ``ome-zarr-v2``: a directory with an entry ``.zgroup`` or ``.zattrs``;
* ``mesh``: a file whose extension is ``.obj`` or ``.ply`` (any case);
* ``point-collections``: a ``.json`` file (any case) whose top level is an
  object with the key ``vc_pointcollections_json_version`` or ``collections``;
* ``umbilicus``: a ``.json`` file whose top level is an array, or an object
  with the key ``control_points`` or ``points``.

Markers of two kinds are ``IO_INSPECT_AMBIGUOUS``; none, a ``.json`` file that
cannot be read or is not strict JSON, any other extension, or something that is
neither a regular file nor a directory is ``IO_INSPECT_UNKNOWN``; a URI or a
path naming nothing is ``IO_INSPECT_PATH_INVALID``. A marker only picks the
reader: a partial tifxyz directory, say, is refused by the tifxyz reader naming
the missing file. Every JSON document inspect reads (the ``.json`` sniff, the
``--no-frame`` stand-in's ``0/.zarray``, the frame file) is parsed by
:func:`apok_eval.jsonstrict.loads`, the parser every JSON reader in ``apok_eval.io`` uses:
a ``.json`` file that parser refuses (a duplicate key, NaN, nesting past 256
levels, ...) cannot be told point collections or an umbilicus, so the sniff
refuses it naming the parser's problem rather than guess from a lenient parse.

**Reading** (``read_as`` in the record states every choice made):

* tifxyz: :func:`apok_eval.io.tifxyz.inspect_tifxyz` under every validity policy,
  or only ``--policy``; with ``--frame``, ``--policy`` is required
  (``IO_INSPECT_OPTION_INVALID``: the policy decides which vertices are points,
  never assumed) and the surface is read once, its validity report made under
  that policy (which refuses if the policy does) and
  :func:`~apok_eval.io.tifxyz.check_within_frame` applied.
* OME-Zarr v2: :func:`apok_eval.io.omezarr.open_volume` on a
  :class:`~apok_eval.io.store.LocalStore` (metadata only, no chunk is read), default
  pyramid tolerance ``exact``, axes as the store declares them (a store without
  axes is refused, ``IO_ZARR_AXES_UNDECLARED``: inspect declares none).
  ``--frame`` is required (the coordinate contract: a frame is declared, never
  assumed) unless ``--no-frame`` is given (``IO_INSPECT_FRAME_REQUIRED``). With
  ``--no-frame`` every check but the frame's runs: the volume is opened
  against a stand-in frame whose shape is level 0's own, so the shape
  comparison is vacuous, and the stand-in never leaves this module - the
  summary's ``frame`` is ``"NOT CHECKED"``.
* mesh: :func:`apok_eval.io.mesh.read_mesh`; the summary is
  :func:`~apok_eval.io.mesh.inspect_mesh`'s; ``--frame`` applies
  :func:`~apok_eval.io.mesh.check_mesh_within`.
* point collections: read role-less (``role=None``: the role is derived from
  the file name exactly as the fitter derives it for an undeclared role, and
  recorded as ``declared_role: null``); the summary is
  ``inspect_point_collections(path, role=None)``'s; ``--frame`` applies
  :func:`~apok_eval.io.pointcollection.check_point_collections_within`.
* umbilicus: variant ``fitter`` when the object's only marker is
  ``control_points``, otherwise ``vc3d-json`` (a top-level array or ``points``
  is read only by VC3D); no coordinate scale; ``--frame`` is passed to
  :func:`apok_eval.io.umbilicus.read_umbilicus`, which checks the control points and
  the metadata against it.

``--policy`` must name one of :data:`apok_eval.io.tifxyz.POLICIES` and applies to
tifxyz only (``IO_INSPECT_OPTION_INVALID`` otherwise: the command line's
``choices`` stop an unknown name, and :func:`inspect_path` refuses it itself).
Without ``--frame`` the record's ``frame`` says ``NOT CHECKED`` and why.

**The frame file** is strict JSON (:func:`apok_eval.jsonstrict.loads`: UTF-8 without
a byte-order mark, no duplicate key, no NaN/Infinity, no number that overflows a
double, no lone-surrogate escape, no nesting past 256 levels); a file that is
not, or cannot be read (a path ``open()`` refuses, a FIFO or device included),
is ``IO_INSPECT_FRAME_INVALID`` naming the problem. Its document is read by
:meth:`apok_eval.io.frame.VolumeFrame.from_dict`, the one frame-file schema (exactly
``volume_id``, ``shape_zyx``, ``voxel_size_um`` and ``source``, optionally
``axes`` = ``["z", "y", "x"]``; an integer beyond 2**53 refused), so a
declaration is refused the same way (``IO_FRAME_INVALID``, naming the
``field``) here and through the Python API.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any

from apok_eval import jsonstrict
from apok_eval.cli import print_json
from apok_eval.fileio import portable_name, read_regular
from apok_eval.io.frame import VolumeFrame, stated_identity
from apok_eval.io.mesh import check_mesh_within, read_mesh
from apok_eval.io.omezarr import open_volume
from apok_eval.io.pointcollection import check_point_collections_within, read_point_collections
from apok_eval.io.store import LocalStore
from apok_eval.io.tifxyz import POLICIES, check_within_frame, inspect_tifxyz, read_tifxyz, validity_report
from apok_eval.io.umbilicus import read_umbilicus
from apok_eval.refusal import NOT_CHECKED, Refusal, register_codes

register_codes(
    "apok_eval.io.cli",
    {
        "IO_INSPECT_PATH_INVALID": "an inspect PATH is a remote URI or names nothing on the local filesystem",
        "IO_INSPECT_UNKNOWN": "an inspect PATH is none of the kinds apok-eval io inspect detects",
        "IO_INSPECT_AMBIGUOUS": "an inspect PATH carries the markers of more than one kind",
        "IO_INSPECT_FRAME_REQUIRED": "an OME-Zarr inspect has neither --frame nor an explicit --no-frame",
        "IO_INSPECT_FRAME_INVALID": "a --frame file is not a strict JSON frame declaration",
        "IO_INSPECT_OPTION_INVALID": "an inspect option does not apply to the detected kind, or one it needs is "
                                     "missing",
    },
)

SCHEMA = "apok-io-inspect-v1"
TIFXYZ_MARKERS = ("x.tif", "y.tif", "z.tif")
ZARR_MARKERS = (".zgroup", ".zattrs")
MESH_EXTENSIONS = (".obj", ".ply")
PCL_KEYS = ("vc_pointcollections_json_version", "collections")
UMBILICUS_KEYS = ("control_points", "points")
#: What each kind's frame check compares (the record's ``frame.rule``).
FRAME_RULES = {
    "tifxyz": "every vertex valid under the policy lies inside the declared level-0 shape",
    "ome-zarr-v2": "level 0's stored shape equals the declared shape_zyx",
    "mesh": "every vertex lies inside the declared level-0 shape",
    "point-collections": "every point lies inside the declared level-0 shape",
    "umbilicus": "every control point lies inside the declared level-0 shape, and any metadata's stamped "
                 "dimensions and voxel size agree with the frame",
}
_URI = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://")
_STAND_IN_ID = "00000000000000"  # the --no-frame stand-in's volume id; never printed


@dataclass(frozen=True)
class Detected:
    kind: str
    markers: tuple[str, ...]


# --- detection -------------------------------------------------------------------------------


def detect(path: str) -> Detected:
    """The kind of the local input at ``path``, by the markers in the module docstring; refuses otherwise."""
    if _URI.match(path):
        raise Refusal("IO_INSPECT_PATH_INVALID", f"{path!r} is a URI; apok-eval io inspect reads local paths only (a "
                      "remote inspect is out of scope)", path=path, problem="remote")
    if not os.path.lexists(path):
        raise Refusal("IO_INSPECT_PATH_INVALID", f"nothing exists at {path!r}", path=path, problem="missing")
    if os.path.isdir(path):
        found = {"tifxyz": [m for m in TIFXYZ_MARKERS if os.path.lexists(os.path.join(path, m))],
                 "ome-zarr-v2": [m for m in ZARR_MARKERS if os.path.lexists(os.path.join(path, m))]}
        return _one_kind(path, found, "a directory with x.tif, y.tif or z.tif (tifxyz) or with .zgroup or .zattrs "
                         "(OME-Zarr v2; a Zarr v3 zarr.json is not read)")
    if not os.path.isfile(path):
        raise Refusal("IO_INSPECT_UNKNOWN", f"{path!r} is neither a regular file nor a directory (a dangling link, "
                      "a FIFO, a device?)", path=path, problem="not a file or directory")
    extension = os.path.splitext(path)[1].lower()
    if extension in MESH_EXTENSIONS:
        return Detected("mesh", (f"extension {extension}",))
    if extension != ".json":
        raise Refusal("IO_INSPECT_UNKNOWN", f"{path!r}: extension {extension!r} is none of .obj, .ply (mesh) or "
                      ".json (point collections, umbilicus)", path=path, problem=f"extension {extension!r}")
    return _one_kind(path, _json_markers(path), "a JSON object with vc_pointcollections_json_version or "
                     "collections (point collections), or a JSON array or an object with control_points or "
                     "points (umbilicus)")


def _json_markers(path: str) -> dict[str, list[str]]:
    try:
        data = read_regular(path)
    except OSError as exc:  # a regular file may still refuse reading: no permission, an I/O error
        raise Refusal("IO_INSPECT_UNKNOWN", f"{path!r} cannot be read ({type(exc).__name__}: {exc}), so whether "
                      "it holds point collections or an umbilicus cannot be told", path=path, problem="unreadable",
                      error=type(exc).__name__) from None
    try:
        top = jsonstrict.loads(data)
    except jsonstrict.StrictJsonError as exc:  # the sniff: what the readers would refuse is not guessed at
        raise Refusal("IO_INSPECT_UNKNOWN", f"{path!r} is not strict JSON ({exc}), so whether it holds point "
                      "collections or an umbilicus cannot be told", path=path, problem="not strict JSON",
                      error=str(exc)) from None
    if isinstance(top, list):
        return {"point-collections": [], "umbilicus": ["top-level array"]}
    keys = top if isinstance(top, dict) else {}
    return {"point-collections": [f"key {k}" for k in PCL_KEYS if k in keys],
            "umbilicus": [f"key {k}" for k in UMBILICUS_KEYS if k in keys]}


def _one_kind(path: str, found: dict[str, list[str]], looked_for: str) -> Detected:
    kinds = [kind for kind, markers in found.items() if markers]
    if len(kinds) > 1:
        raise Refusal("IO_INSPECT_AMBIGUOUS", f"{path!r} carries the markers of {' and '.join(kinds)} "
                      f"({found}); inspect will not pick one", path=path, kinds=kinds,
                      markers={kind: found[kind] for kind in kinds})
    if not kinds:
        raise Refusal("IO_INSPECT_UNKNOWN", f"{path!r} has no marker of a kind inspect reads; looked for "
                      f"{looked_for}", path=path, problem="no marker", looked_for=looked_for)
    return Detected(kinds[0], tuple(found[kinds[0]]))


# --- the frame file --------------------------------------------------------------------------


def load_frame(path: str) -> VolumeFrame:
    """The :class:`VolumeFrame` declared in the file at ``path`` (format in the module docstring)."""
    try:
        document = jsonstrict.loads(read_regular(path))
    except (OSError, ValueError) as exc:  # a StrictJsonError; a FIFO or device (NotRegularFile); a NUL (ValueError)
        problem = jsonstrict.problem_of(exc)
        raise Refusal("IO_INSPECT_FRAME_INVALID", f"--frame {path!r} is not readable strict JSON: {problem}",
                      path=path, problem=problem) from None
    return VolumeFrame.from_dict(document, where=f"--frame {path!r}")


# --- reading ---------------------------------------------------------------------------------


def inspect_path(path: str, *, frame_path: str | None = None, no_frame: bool = False,
                 policy: str | None = None) -> dict[str, Any]:
    """The ``apok-io-inspect-v1`` record for the local input at ``path``; refuses as the module docstring says."""
    if frame_path is not None and no_frame:
        raise Refusal("IO_INSPECT_OPTION_INVALID", "--frame and --no-frame contradict each other", path=path,
                      option="--no-frame", kind=None)
    if policy is not None and policy not in POLICIES:
        raise Refusal("IO_INSPECT_OPTION_INVALID", f"--policy {policy!r} is none of {list(POLICIES)}", path=path,
                      option="--policy", value=policy)
    detected = detect(path)
    kind = detected.kind
    if policy is not None and kind != "tifxyz":
        raise Refusal("IO_INSPECT_OPTION_INVALID", f"--policy chooses a tifxyz validity policy; {path!r} is "
                      f"{kind}", path=path, option="--policy", kind=kind)
    if kind == "tifxyz" and frame_path is not None and policy is None:
        raise Refusal("IO_INSPECT_OPTION_INVALID", "a frame check on a tifxyz surface needs --policy: the policy "
                      "decides which vertices are points, and inspect does not assume one", path=path,
                      option="--policy", kind=kind)
    if kind == "ome-zarr-v2" and frame_path is None and not no_frame:
        raise Refusal("IO_INSPECT_FRAME_REQUIRED", f"{path!r} is an OME-Zarr volume: give --frame FRAME.json (its "
                      "level-0 shape is checked against the declared frame, never assumed) or say --no-frame",
                      path=path, kind=kind)
    frame = None if frame_path is None else load_frame(frame_path)
    try:
        read_as, summary, count = _READERS[kind](path, detected, frame, policy)
    except OSError as exc:  # the readers refuse malformed content; an I/O failure is theirs to raise, not to guess
        raise Refusal("IO_INSPECT_UNKNOWN", f"{path!r} cannot be read as {kind} ({type(exc).__name__}: {exc})",
                      path=path, problem="unreadable", error=type(exc).__name__, errno=exc.errno) from None
    if frame is None:
        why = ("--no-frame given" if no_frame else "no --frame given") + (
            ": level 0's shape was compared with no declared frame, and no volume id or voxel size is known"
            if kind == "ome-zarr-v2" else ": the coordinates were compared with no declared volume")
        frame_record = {"checked": False, "status": NOT_CHECKED, "why": why}
    else:
        frame_record = {"checked": True, "file": portable_name(frame_path), "declared": frame.as_dict(),
                        "rule": FRAME_RULES[kind], "count": count}
    # Paths in the record are written as the tifxyz summary writes them (bytes that are not UTF-8 as \xNN).
    return {"schema": SCHEMA, "path": portable_name(path), "kind": kind, "detected_by": list(detected.markers),
            "read_as": read_as, "frame": frame_record, "summary": summary}


def _read_tifxyz(path: str, detected: Detected, frame: VolumeFrame | None, policy: str | None):
    if frame is None:
        policies = POLICIES if policy is None else (policy,)
        return {"policies": list(policies)}, inspect_tifxyz(path, policies), None
    surface = read_tifxyz(path)
    report = validity_report(surface, (policy,))
    count = check_within_frame(surface, frame, policy)
    # What meta.json itself states about the volume, as compared (check_within_frame refused a disagreement).
    identity = stated_identity(frame, surface.meta, f"tifxyz {surface.dir_name}")
    return ({"policies": [policy]}, {"surface": surface.as_dict(), "validity": report.as_dict(), "refused_policies": {},
                                     "stated_identity": identity}, count)


def _read_zarr(path: str, detected: Detected, frame: VolumeFrame | None, policy: str | None):
    store = LocalStore(path)
    volume = open_volume(store, _stand_in_frame(store) if frame is None else frame)
    summary = volume.describe()
    if frame is None:
        summary["frame"] = NOT_CHECKED  # the stand-in is not a declaration
    return {"pyramid_tolerance": volume.pyramid_tolerance, "axes_declared": None}, summary, None


def _stand_in_frame(store: LocalStore) -> VolumeFrame:
    """For ``--no-frame`` only: a frame whose shape is level 0's own, so open_volume's frame check is vacuous.

    The shape is read from ``0/.zarray`` with the parser open_volume uses (:func:`apok_eval.jsonstrict.loads`).
    Whenever open_volume accepts that object it parses the same shape, so the comparison passes; whenever
    this reading fails, open_volume refuses level 0 itself before it compares, so (1, 1, 1) is never compared.
    """
    try:
        meta = jsonstrict.loads(store.get("0/.zarray") or b"null")
    except (Refusal, jsonstrict.StrictJsonError):  # open_volume meets the same object and refuses it in order
        meta = None
    shape = meta.get("shape") if isinstance(meta, dict) else None
    own = (isinstance(shape, list) and len(shape) == 3
           and all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in shape))
    return VolumeFrame(_STAND_IN_ID, tuple(shape) if own else (1, 1, 1), 1.0,
                       "apok-eval io inspect --no-frame stand-in: NOT a declaration")


def _read_mesh(path: str, detected: Detected, frame: VolumeFrame | None, policy: str | None):
    mesh = read_mesh(path)
    if frame is not None:
        check_mesh_within(mesh, frame, f"mesh {os.path.basename(path)}")
    return {}, mesh.as_dict(), None if frame is None else int(len(mesh.vertices_xyz))


def _read_point_collections(path: str, detected: Detected, frame: VolumeFrame | None, policy: str | None):
    pcl = read_point_collections(path, role=None)
    count = None if frame is None else check_point_collections_within(pcl, frame)
    return {"role": None}, pcl.as_dict(), count


def _read_umbilicus(path: str, detected: Detected, frame: VolumeFrame | None, policy: str | None):
    variant = "fitter" if detected.markers == ("key control_points",) else "vc3d-json"
    umbilicus = read_umbilicus(path, variant=variant, frame=frame)
    count = None if frame is None else int(len(umbilicus.control_zyx))
    return {"variant": variant, "coordinate_scale": None}, umbilicus.as_dict(), count


_READERS = {"tifxyz": _read_tifxyz, "ome-zarr-v2": _read_zarr, "mesh": _read_mesh,
            "point-collections": _read_point_collections, "umbilicus": _read_umbilicus}


# --- command line ----------------------------------------------------------------------------


def render(record: dict[str, Any]) -> str:
    """A short text rendering of an inspect record; ``--json`` prints the record itself."""
    lines = [f"apok-eval io inspect: {record['kind']} at {record['path']}",
             f"  detected by: {', '.join(record['detected_by'])}"]
    if record["read_as"]:
        lines.append("  read as: " + ", ".join(f"{k}={_brief(v)}" for k, v in sorted(record["read_as"].items())))
    frame = record["frame"]
    if frame["checked"]:
        declared = frame["declared"]
        count = "" if frame["count"] is None else f"; {frame['count']} checked"
        lines.append(f"  frame: CHECKED against volume {declared['volume_id']} (shape_zyx {declared['shape_zyx']}, "
                     f"{declared['voxel_size_um']} um, from {frame['file']}): {frame['rule']}{count}")
    else:
        lines.append(f"  frame: {NOT_CHECKED} ({frame['why']})")
    lines.append("  summary (--json prints all of it):")
    lines += [f"    {key}: {_brief(value)}" for key, value in sorted(record["summary"].items())]
    return "\n".join(lines)


def _brief(value: Any) -> str:
    items = list(value.values()) if isinstance(value, dict) else value if isinstance(value, list) else None
    if items is not None and (len(items) > 6 or any(isinstance(v, (dict, list)) for v in items)):
        return f"<{len(items)} {'entries' if isinstance(value, dict) else 'items'}>"
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _inspect(args: argparse.Namespace) -> int:
    record = inspect_path(args.path, frame_path=args.frame, no_frame=args.no_frame, policy=args.policy)
    if args.json:
        print_json(record)
    else:
        print(render(record))
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    io = subparsers.add_parser("io", help="input formats: inspect", description="apok-eval io: inspect a local input")
    sub = io.add_subparsers(dest="io_command", metavar="SUBCOMMAND", required=True)
    inspect = sub.add_parser("inspect", help="say what a local input is and summarise it",
                             description="Detect what a local input is (tifxyz, OME-Zarr v2, mesh, point "
                             "collections, umbilicus), read it with its module, check it against a declared frame "
                             "when one is given, and print a summary.")
    inspect.add_argument("path", metavar="PATH", help="a tifxyz or OME-Zarr v2 directory, a point-collections or "
                         "umbilicus .json, or an .obj/.ply mesh")
    frame = inspect.add_mutually_exclusive_group()
    frame.add_argument("--frame", metavar="FRAME.json", default=None,
                       help="the declared volume frame (volume_id, shape_zyx, voxel_size_um, source) to check against")
    frame.add_argument("--no-frame", action="store_true",
                       help="inspect an OME-Zarr volume without a frame; the summary says NOT CHECKED")
    inspect.add_argument("--policy", choices=POLICIES, default=None,
                         help="tifxyz only: the one validity policy to apply (required with --frame)")
    inspect.add_argument("--json", action="store_true", help="print the whole inspect record as JSON")
    inspect.set_defaults(func=_inspect)
