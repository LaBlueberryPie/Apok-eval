"""Umbilicus files: declared-variant readers, linear interpolation with a declared out-of-range policy.

The umbilicus maps slice z to the scroll centre (y, x). Readers disagree
(``docs/io-surfaces.md`` cites villa for each), so the caller declares which
reading applies (:data:`VARIANTS`):

- ``fitter`` (default; spiral-fitting/umbilicus.py:25-33): a JSON object whose
  ``control_points`` is a list of objects with numeric ``z``, ``y``, ``x``;
  other keys of a point (score, estimates, inside) are kept, not read.
- ``vc3d-json`` (volume-cartographer/core/src/Umbilicus.cpp:276-324): a root
  array, or an object with a ``points`` or ``control_points`` array, of
  ``[z, y, x]`` arrays or ``{z, y, x}`` objects.
- ``vc3d-text``: one ``z, y, x`` per line; blank lines and ``#`` comments allowed.
- ``thaumato-text`` (spiral-fitting/umbilicus.py:6-13): integer lines ``a, b, c``
  meaning y = a - 500, z = b - 500, x = c - 500.

Comma text read as JSON is refused as undeclared: the two text readings use the
same syntax with different axes. An object holding both ``points`` and
``control_points`` is refused under every variant (VC3D reads one, the fitter
the other). At least two control points, all finite, with distinct z, are
required, also after ``coordinate_scale``, which multiplies all three
coordinates only when the caller passes it, and is recorded (a product that
overflows a double is refused). Control points are sorted by z (stable).

A VC3D ``metadata`` block (voxelsize_um > 0, a non-empty ``volume`` string,
and either all or none of volume_width/height/slices as positive ints) is
checked; with a declared :class:`~apok_eval.io.frame.VolumeFrame` the stamped
dimensions and voxel size must describe that frame after ``coordinate_scale``
(each stamped dimension the floor or ceiling of the frame's divided by the
scale, or, for a scale below 1, the frame's the floor or ceiling of the stamped
one times the scale, as VC3D tests a rescale) and the control points must lie
inside it.

Interpolation is linear between neighbouring control points. Outside the
control-point z span the caller must choose ``"extrapolate"`` (the fitter's
interp1d, continuing the end segments) or ``"clamp"`` (VC3D, holding the end
values; VC3D's consumers further round z to a whole slice, which is not
modelled); every query reports whether it fell outside the span.
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
from apok_eval.io.frame import MAX_EXACT_INT, VolumeFrame, check_points_within, finite_number, stated_identity
from apok_eval.fileio import NotRegularFile, portable_name, read_regular, write_atomic
from apok_eval.refusal import Refusal, register_codes

register_codes(
    "apok_eval.io.umbilicus",
    {
        "IO_UMBILICUS_FILE_MISSING": "the umbilicus file does not exist or is not a regular file",
        "IO_UMBILICUS_VARIANT_UNKNOWN": "the declared umbilicus variant is not one apok-eval reads",
        "IO_UMBILICUS_JSON_INVALID": "a JSON umbilicus is not strict JSON",
        "IO_UMBILICUS_TEXT_UNDECLARED": "comma text was read without declaring VC3D z,y,x or Thaumato (y,z,x)+500",
        "IO_UMBILICUS_TEXT_SYNTAX": "a line of a text umbilicus is malformed",
        "IO_UMBILICUS_STRUCTURE": "the JSON root or a control point does not have the declared variant's shape",
        "IO_UMBILICUS_AMBIGUOUS": "both 'points' and 'control_points' are present; VC3D and the fitter disagree",
        "IO_UMBILICUS_VALUE_INVALID": "a control-point coordinate is not a finite number",
        "IO_UMBILICUS_TOO_FEW": "fewer than two control points",
        "IO_UMBILICUS_DUPLICATE_Z": "two control points share a z value",
        "IO_UMBILICUS_SCALE_INVALID": "coordinate_scale is not a finite positive number",
        "IO_UMBILICUS_METADATA_INVALID": "the VC3D metadata block is malformed, including a partial volume triplet",
        "IO_UMBILICUS_FRAME_MISMATCH": "the metadata block contradicts the declared frame",
        "IO_UMBILICUS_POLICY_UNKNOWN": "the out-of-range policy is not 'extrapolate' or 'clamp'",
        "IO_UMBILICUS_QUERY_INVALID": "a query z is not a finite number",
        "IO_UMBILICUS_WRITE_INVALID": "control points given to the writer are not a finite (N, 3) array",
        "IO_UMBILICUS_WRITE_EXISTS": "the output file exists and overwrite was not requested",
    },
)

VARIANTS = ("fitter", "vc3d-json", "vc3d-text", "thaumato-text")
OUT_OF_RANGE = ("extrapolate", "clamp")
#: VC3D metadata dimension keys in (z, y, x) order (volume-cartographer/.../Umbilicus.hpp:18-33).
DIMENSION_KEYS = ("volume_slices", "volume_height", "volume_width")
#: Relative tolerance when a stamped voxel size is compared with the declared frame's.
VOXEL_SIZE_RTOL = 1e-6
# ASCII digits only: Python's float() and int() would also accept other scripts' digits.
_DECIMAL = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
_INTEGER = re.compile(r"[+-]?[0-9]+\Z")


@dataclass(frozen=True, eq=False)
class UmbilicusQuery:
    z: np.ndarray  # (M,) float64
    yx: np.ndarray  # (M, 2) float64
    outside: np.ndarray  # (M,) bool: z outside the control-point span
    out_of_range: str

    @property
    def outside_count(self) -> int:
        return int(self.outside.sum())


@dataclass(frozen=True, eq=False)
class Umbilicus:
    path: str
    digest: str  # of the bytes parsed
    size: int  # their length
    variant: str
    control_zyx: np.ndarray  # (N, 3) float64, ascending z, coordinate_scale applied
    coordinate_scale: float | None
    metadata: dict[str, Any] | None  # the VC3D block verbatim; None when absent or null
    findings: tuple[dict[str, Any], ...]

    @property
    def z_span(self) -> tuple[float, float]:
        return float(self.control_zyx[0, 0]), float(self.control_zyx[-1, 0])

    def yx_at(self, z: Any, *, out_of_range: str) -> UmbilicusQuery:
        """Centre (y, x) at each ``z``; ``out_of_range`` is "extrapolate" or "clamp" (no default)."""
        if out_of_range not in OUT_OF_RANGE:
            raise Refusal("IO_UMBILICUS_POLICY_UNKNOWN", f"out_of_range must be one of {list(OUT_OF_RANGE)}, not "
                          f"{out_of_range!r}", policy=out_of_range)
        query = np.atleast_1d(np.asarray(z))
        problem = ("not numeric" if query.dtype.kind not in "iuf" else "not a scalar or 1-D" if query.ndim != 1
                   else "not finite" if not np.isfinite(query).all() else None)
        if problem is not None:
            raise Refusal("IO_UMBILICUS_QUERY_INVALID", f"z is {problem}; it must be a finite number or a 1-D array "
                          "of them", problem=problem, dtype=str(query.dtype), shape=list(query.shape))
        query = query.astype(np.float64)
        zc, yxc = self.control_zyx[:, 0], self.control_zyx[:, 1:]
        # Linear interpolation in the form scipy's interp1d uses, end segments continued outside the span.
        hi = np.clip(np.searchsorted(zc, query, side="left"), 1, len(zc) - 1)
        lo = hi - 1
        slope = (yxc[hi] - yxc[lo]) / (zc[hi] - zc[lo])[:, None]
        yx = slope * (query - zc[lo])[:, None] + yxc[lo]
        if out_of_range == "clamp":  # VC3D returns the end control points exactly at and beyond the ends
            yx = np.where((query <= zc[0])[:, None], yxc[0], yx)
            yx = np.where((query >= zc[-1])[:, None], yxc[-1], yx)
        outside = (query < zc[0]) | (query > zc[-1])
        return UmbilicusQuery(z=query, yx=yx, outside=outside, out_of_range=out_of_range)

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": portable_name(self.path),  # bytes that are not UTF-8 as \xNN, as the tifxyz summary
            "digest": self.digest,
            "variant": self.variant,
            "control_points": int(len(self.control_zyx)),
            "z_span": list(self.z_span),
            "axes": ["z", "y", "x"],
            "coordinate_scale": self.coordinate_scale,
            "metadata": self.metadata,
            "findings": list(self.findings),
        }


def read_umbilicus(path: str | os.PathLike[str], *, variant: str = "fitter", coordinate_scale: float | None = None,
                   frame: VolumeFrame | None = None) -> Umbilicus:
    """Read one umbilicus file under the declared ``variant`` (see the module docstring)."""
    name = os.fspath(path)
    if variant not in VARIANTS:
        raise Refusal("IO_UMBILICUS_VARIANT_UNKNOWN", f"variant {variant!r} is not one of {list(VARIANTS)}",
                      variant=variant)
    scale = _scale(coordinate_scale)
    try:
        data = read_regular(name)
    except NotRegularFile as exc:
        problem = "missing" if exc.missing else f"not a regular file ({exc.kind})"
        why = "does not exist" if exc.missing else f"is {exc.kind}, not a regular file"
        raise Refusal("IO_UMBILICUS_FILE_MISSING", f"{name} {why}", path=name, problem=problem) from None
    except ValueError:  # a NUL byte: os.open refuses the path itself
        raise Refusal("IO_UMBILICUS_FILE_MISSING", f"{name!r} is not a valid path", path=name,
                      problem="not a valid path") from None
    metadata = None
    if variant in ("fitter", "vc3d-json"):
        document = _load_json(name, data)
        points = _json_points(name, document, variant)
        metadata = _metadata(name, document)
    else:
        points = _text_points(name, data, variant)
    control = _control_points(name, points, scale)
    findings = _findings(name, variant, metadata, frame)
    if frame is not None:
        _check_frame(name, metadata, frame, scale, control)
    return Umbilicus(path=name, digest=sha256_bytes(data), size=len(data), variant=variant, control_zyx=control,
                     coordinate_scale=scale, metadata=metadata, findings=tuple(findings))


def _scale(coordinate_scale: Any) -> float | None:
    if coordinate_scale is None:
        return None
    scale = finite_number(coordinate_scale)
    if scale is None or scale <= 0:
        raise Refusal("IO_UMBILICUS_SCALE_INVALID", f"coordinate_scale {coordinate_scale!r} is not a finite "
                      "positive number", value=repr(coordinate_scale))
    return scale


def _load_json(name: str, data: bytes) -> Any:
    try:
        return jsonstrict.loads(data)
    except jsonstrict.StrictJsonError as exc:
        if _looks_like_comma_text(data):
            raise Refusal("IO_UMBILICUS_TEXT_UNDECLARED", f"{name} is comma text, which reads as VC3D z,y,x or "
                          "as Thaumato (y,z,x)+500; declare variant='vc3d-text' or 'thaumato-text'",
                          path=name) from None
        raise Refusal("IO_UMBILICUS_JSON_INVALID", f"{name}: {exc}", path=name, problem=str(exc)) from None


def _looks_like_comma_text(data: bytes) -> bool:
    try:
        lines = [line.strip() for line in data.decode("utf-8").splitlines()]
    except UnicodeDecodeError:
        return False
    rows = [line for line in lines if line and not line.startswith("#")]
    return bool(rows) and all(len(parts := [t.strip() for t in row.split(",")]) == 3
                              and all(_DECIMAL.match(t) for t in parts) for row in rows)


#: A control point before checking: (("index", list position) or ("line", line number), (z, y, x) as read).
_RawPoint = tuple[tuple[str, int], Any]


def _json_points(name: str, document: Any, variant: str) -> list[_RawPoint]:
    if isinstance(document, dict) and "points" in document and "control_points" in document:
        raise Refusal("IO_UMBILICUS_AMBIGUOUS", f"{name} holds both 'points' (read by VC3D) and 'control_points' "
                      "(read by the fitter)", path=name)
    if variant == "vc3d-json" and isinstance(document, list):
        entries: Any = document
    elif variant == "vc3d-json" and isinstance(document, dict) and "points" in document:
        entries = document["points"]
    else:
        entries = document.get("control_points") if isinstance(document, dict) else None
    if not isinstance(entries, list):
        expected = ("an object with a control_points array" if variant == "fitter"
                    else "a root array or an object with a points or control_points array")
        raise Refusal("IO_UMBILICUS_STRUCTURE", f"{name}: the {variant} variant reads {expected}", path=name,
                      index=None)
    points = []
    for index, entry in enumerate(entries):
        if isinstance(entry, dict) and all(axis in entry for axis in "zyx"):
            values = (entry["z"], entry["y"], entry["x"])
        elif variant == "vc3d-json" and isinstance(entry, list) and len(entry) == 3:
            values = tuple(entry)
        else:
            form = "{z, y, x} object" if variant == "fitter" else "[z, y, x] array or {z, y, x} object"
            raise Refusal("IO_UMBILICUS_STRUCTURE", f"{name}: control point {index} is not a {form}", path=name,
                          index=index)
        points.append((("index", index), values))
    return points


def _text_points(name: str, data: bytes, variant: str) -> list[_RawPoint]:
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise Refusal("IO_UMBILICUS_TEXT_SYNTAX", f"{name}: not UTF-8 text (byte {exc.start})", path=name,
                      line=None) from None
    points = []
    for line_no, raw in enumerate(lines, start=1):
        line = raw.strip()
        if variant == "vc3d-text" and (not line or line.startswith("#")):
            continue
        tokens = [t.strip() for t in line.split(",")]
        pattern = _DECIMAL if variant == "vc3d-text" else _INTEGER
        if len(tokens) != 3 or not all(pattern.match(t) for t in tokens):
            kind = "z, y, x decimals" if variant == "vc3d-text" else "three integers a, b, c"
            raise Refusal("IO_UMBILICUS_TEXT_SYNTAX", f"{name}:{line_no}: {raw!r} is not {kind}", path=name,
                          line=line_no)
        if variant == "vc3d-text":
            values = tuple(float(t) for t in tokens)
        else:
            a, b, c = (_whole(t) for t in tokens)
            values = (b - 500, a - 500, c - 500)  # (z, y, x) from Thaumato's (y, z, x) + 500
        points.append((("line", line_no), values))
    return points


def _whole(token: str) -> int | float:
    """An integer token exactly, or, past int()'s digit limit, as a double (infinity unless zero-padded)."""
    try:
        return int(token)
    except ValueError:  # too many digits for int(): read as a double, which the finiteness check refuses
        return float(token)


def _control_points(name: str, points: list[_RawPoint], scale: float | None) -> np.ndarray:
    rows = []
    for (kind, where), values in points:
        row = [finite_number(v) for v in values]
        for axis, value, number in zip("zyx", values, row, strict=True):
            if number is None:
                raise Refusal("IO_UMBILICUS_VALUE_INVALID", f"{name}: control point at {kind} {where}: {axis} = "
                              f"{value!r} is not a finite number", path=name, axis=axis, **{kind: where})
        if scale is not None:
            for axis, number in zip("zyx", row, strict=True):
                if not math.isfinite(number * scale):
                    raise Refusal("IO_UMBILICUS_VALUE_INVALID", f"{name}: control point at {kind} {where}: {axis} = "
                                  f"{number!r} times coordinate_scale {scale!r} overflows a double", path=name,
                                  axis=axis, scale=scale, **{kind: where})
            row = [number * scale for number in row]
        rows.append(row)
    if len(rows) < 2:
        raise Refusal("IO_UMBILICUS_TOO_FEW", f"{name}: {len(rows)} control point(s); interpolation needs two",
                      path=name, count=len(rows))
    control = np.array(rows, dtype=np.float64)
    order = np.argsort(control[:, 0], kind="stable")
    control = control[order]
    repeated = np.flatnonzero(np.diff(control[:, 0]) == 0)
    if len(repeated):
        i = int(repeated[0])
        (kind, first), (_, second) = points[int(order[i])][0], points[int(order[i + 1])][0]
        raise Refusal("IO_UMBILICUS_DUPLICATE_Z", f"{name}: control points at {kind} {first} and {second} share "
                      f"z = {control[i, 0]!r}", path=name, z=float(control[i, 0]), source_kind=kind,
                      sources=[first, second])
    return control


def _metadata(name: str, document: Any) -> dict[str, Any] | None:
    """The VC3D metadata block, checked as VC3D's reader and resolver check it."""
    if not isinstance(document, dict) or document.get("metadata") is None:
        return None
    meta = document["metadata"]
    if not isinstance(meta, dict):
        raise Refusal("IO_UMBILICUS_METADATA_INVALID", f"{name}: metadata must be an object or null", path=name,
                      field="metadata")
    voxel = finite_number(meta.get("voxelsize_um"))
    if "voxelsize_um" in meta and (voxel is None or voxel <= 0):
        raise Refusal("IO_UMBILICUS_METADATA_INVALID", f"{name}: metadata.voxelsize_um must be a positive "
                      f"number, not {meta['voxelsize_um']!r}", path=name, field="voxelsize_um")
    if "volume" in meta and (not isinstance(meta["volume"], str) or not meta["volume"]):
        raise Refusal("IO_UMBILICUS_METADATA_INVALID", f"{name}: metadata.volume must be a non-empty string",
                      path=name, field="volume")
    present = [key for key in DIMENSION_KEYS if key in meta]
    for key in present:
        value = meta[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise Refusal("IO_UMBILICUS_METADATA_INVALID", f"{name}: metadata.{key} must be a positive integer, "
                          f"not {value!r}", path=name, field=key)
        if value > MAX_EXACT_INT:  # compared with the frame in float arithmetic, which overflows at 2**1024
            raise Refusal("IO_UMBILICUS_METADATA_INVALID", f"{name}: metadata.{key} is beyond 2**53, more voxels "
                          "than any volume and more than a float holds exactly", path=name, field=key)
    if 0 < len(present) < 3:
        missing = [key for key in DIMENSION_KEYS if key not in meta]
        raise Refusal("IO_UMBILICUS_METADATA_INVALID", f"{name}: an incomplete volume triplet declares nothing; "
                      f"missing {missing}", path=name, field="volume_dimensions", missing=missing)
    return meta


def _check_frame(name: str, metadata: dict[str, Any] | None, frame: VolumeFrame, scale: float | None,
                 control: np.ndarray) -> None:
    if metadata is not None:  # a volume id the metadata names must be the frame's (the shape is checked below)
        stated_identity(frame, {"volume": metadata.get("volume")}, f"umbilicus {os.path.basename(name)} metadata")
    factor = 1.0 if scale is None else scale
    if metadata is not None and all(key in metadata for key in DIMENSION_KEYS):
        for key, stamped, declared in zip(DIMENSION_KEYS, (metadata[k] for k in DIMENSION_KEYS), frame.shape_zyx,
                                                    strict=True):
            # VC3D's test of a rescale factor (ScrollUmbilicus.cpp:262-315): downsampling a count c by an
            # integer step n gives floor(c/n) or ceil(c/n). factor >= 1: the stamped grid is the frame
            # downsampled by factor; factor < 1: the frame is the stamped grid downsampled by 1/factor.
            coarse, fine, step = (declared, stamped, factor) if factor >= 1 else (stamped, declared, 1 / factor)
            if fine not in (math.floor(coarse / step), math.ceil(coarse / step)):
                raise Refusal("IO_UMBILICUS_FRAME_MISMATCH", f"{name}: metadata.{key} = {stamped} does not "
                              f"describe the declared frame's {declared} at coordinate_scale {factor}",
                              path=name, field=key, stamped=stamped, declared=declared)
    if metadata is not None and "voxelsize_um" in metadata:
        expected = frame.voxel_size_um * factor
        if not math.isclose(metadata["voxelsize_um"], expected, rel_tol=VOXEL_SIZE_RTOL):
            raise Refusal("IO_UMBILICUS_FRAME_MISMATCH", f"{name}: metadata.voxelsize_um = "
                          f"{metadata['voxelsize_um']} but the declared frame at coordinate_scale {factor} "
                          f"implies {expected}", path=name, field="voxelsize_um", stamped=metadata["voxelsize_um"],
                          declared=expected)
    check_points_within(frame, control, f"umbilicus {os.path.basename(name)}")


def _findings(name: str, variant: str, metadata: dict[str, Any] | None,
              frame: VolumeFrame | None) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    is_json_name = os.path.splitext(name)[1].lower() == ".json"
    if is_json_name != (variant in ("fitter", "vc3d-json")):
        findings.append({"finding": "extension-disagrees-with-variant", "variant": variant,
                         "detail": "VC3D picks its reader by extension: .json is JSON, anything else comma text"})
    if metadata is not None and "coordinate_scale" in metadata:
        findings.append({"finding": "metadata-coordinate-scale-not-applied", "value": metadata["coordinate_scale"],
                         "detail": "vc_gen_umbilicus's grid-units-per-voxel; apok-eval applies only a declared scale"})
    if metadata is not None and frame is not None and "volume" in metadata and metadata["volume"] != frame.volume_id:
        # A volume id that differs was refused by stated_identity; what is left names no volume id.
        findings.append({"finding": "metadata-volume-differs", "volume": metadata["volume"],
                         "frame_volume_id": frame.volume_id,
                         "detail": "metadata.volume names no 14-digit volume id, so it is not compared as an identity"})
    return findings


def encode_umbilicus(control_zyx: np.ndarray, metadata: dict[str, Any] | None = None) -> bytes:
    """Fitter-form JSON: {"control_points": [{"x", "y", "z"}, ...] ascending z, "metadata"?}, sorted, indent 4."""
    if not isinstance(control_zyx, np.ndarray) or control_zyx.ndim != 2 or control_zyx.shape[1] != 3 \
            or control_zyx.dtype.kind not in "iuf" or not np.isfinite(control_zyx).all():
        raise Refusal("IO_UMBILICUS_WRITE_INVALID", "control_zyx must be a finite (N, 3) numeric array",
                      field="control_zyx")
    where = "umbilicus given to the writer"
    document: dict[str, Any] = {"control_points": [{"z": z, "y": y, "x": x}
                                                   for z, y, x in control_zyx.astype(np.float64).tolist()]}
    control = _control_points(where, _json_points(where, document, "fitter"), None)
    document["control_points"] = [{"z": z, "y": y, "x": x} for z, y, x in control.tolist()]
    if metadata is not None:
        document["metadata"] = metadata
    # The whole document is checked, not the metadata alone: it nests the metadata one level deeper, so
    # metadata at exactly the reader's depth limit is a document past it. The control points are finite
    # floats by now, so only the metadata can make the document unwritable.
    try:
        data = jsonstrict.dumps(document)
    except jsonstrict.StrictJsonError as exc:
        raise Refusal("IO_UMBILICUS_WRITE_INVALID", f"metadata must hold only JSON the reader accepts: {exc}",
                      field="metadata", problem=str(exc)) from None
    _metadata(where, document)
    return data


def write_umbilicus(path: str | os.PathLike[str], control_zyx: np.ndarray, metadata: dict[str, Any] | None = None,
                    *, overwrite: bool = False) -> str:
    """Write :func:`encode_umbilicus` bytes to ``path``; return their digest."""
    target = os.fspath(path)
    data = encode_umbilicus(control_zyx, metadata)
    if os.path.lexists(target) and not overwrite:
        raise Refusal("IO_UMBILICUS_WRITE_EXISTS", f"{target} exists; pass overwrite=True to replace it",
                      path=target)
    write_atomic(target, data, overwrite=overwrite)
    return sha256_bytes(data)


def inspect_umbilicus(path: str | os.PathLike[str], *, variant: str = "fitter",
                      coordinate_scale: float | None = None) -> dict[str, Any]:
    """The summary ``apok-eval io inspect`` prints (it reads once and summarises the object it checked)."""
    return read_umbilicus(path, variant=variant, coordinate_scale=coordinate_scale).as_dict()
