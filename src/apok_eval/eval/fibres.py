"""VC3D fibre traces (``vc3d_fiber`` JSON) in the spiral frame, by villa's convention.

Ported, with credit, from villa ``e0bbb8b`` (MIT; its notice is in ``THIRD_PARTY_NOTICES.md``): ``vesuvius/src/vc3d_fiber_format/__init__.py``
(``parse_vc3d_fiber_format``) and ``spiral-fitting/spiral_helpers.py`` (``load_fiber_point_collection``).

- **Points.** The dense ``line_points`` trimmed to the span from the first to the last control point, each
  control point taken to its nearest line point (squared distance in float32, first index on a tie). When
  ``line_points`` has fewer entries than there are control points, the control points themselves.
- **Scale.** ``coordinate_base_shape_zyx`` absent: 0.25 (fibres are stored at 4x the point-collection
  scale). Present: the unique dyadic 2^k, k in [-5, 5], under which the fibre's domain and the dataset's
  base shape agree (same level: within 1 per axis; otherwise the smaller shape is the larger divided by
  2^|k|, rounded up or down). None or several refuse (``EVAL_FIBRE_DOMAIN``).
- **Numbers.** As villa: coordinates are rounded to float32 before scaling; the scale is a power of two,
  so scaling is exact.
- **Axes.** Files store (x, y, z); apok-eval returns (z, y, x).

apok-eval parses with ``apok_eval.jsonstrict`` (villa's ``json.load`` accepts NaN, duplicate keys and other
inputs apok-eval refuses), and it refuses where villa warns and skips: a version-1 fibre without control points and a
control-point array of the wrong shape (``EVAL_FIBRE_NO_CONTROLS``). For versions 3 and 4 apok-eval checks the
control-point structure villa checks (keys, ``position``, ``tags``, ``segment_to_next`` on every control
point but the last) and requires ``segment_to_next`` to be an object; it does not validate the segment
metadata villa parses inside it, because the evaluator reads only coordinates. Every point must lie in the declared
frame to within one voxel per axis (``EVAL_FIBRE_OUTSIDE_FRAME``): a fibre in the wrong units is refused,
not measured.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from apok_eval import jsonstrict
from apok_eval.canonical import sha256_bytes
from apok_eval.refusal import Refusal, register_codes

register_codes(
    "apok_eval.eval.fibres",
    {
        "EVAL_FIBRE_JSON_INVALID": "a fibre file is not strict JSON",
        "EVAL_FIBRE_STRUCTURE": "a fibre document does not have the vc3d_fiber structure villa accepts",
        "EVAL_FIBRE_DOMAIN": "coordinate_base_shape_zyx is malformed, or no unique dyadic scale maps it to the "
                             "declared base shape",
        "EVAL_FIBRE_NO_CONTROLS": "a fibre has no usable control points (villa skips it; apok-eval refuses it)",
        "EVAL_FIBRE_OUTSIDE_FRAME": "a scaled fibre point lies outside the declared frame by more than one voxel",
    },
)

FIBRE_VERSIONS = (1, 3, 4)
OPTIMIZATION_MODES = ("lasagna", "native_fiber_trace3d")
#: Scale for a fibre that does not declare its domain (spiral_helpers.py: "4x the scale" of the PCL JSONs).
DEFAULT_SCALE = 0.25
_CONTROL_KEYS = frozenset({"position", "segment_to_next", "tags"})


@dataclass(frozen=True, eq=False)
class Fibre:
    """One fibre trace in the spiral frame. ``points_zyx`` is float64 (N, 3) and read-only."""

    name: str
    digest: str
    version: int
    optimization_mode: str
    points_zyx: np.ndarray
    control_count: int
    line_count: int
    trimmed: bool  # False when the control points stood in for a short line
    coordinate_scale: float
    scale_source: str  # "coordinate_base_shape_zyx" or "default-0.25"
    coordinate_base_shape_zyx: list[int] | None
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "digest": self.digest, "version": self.version,
            "optimization_mode": self.optimization_mode, "points": int(len(self.points_zyx)),
            "control_count": self.control_count, "line_count": self.line_count, "trimmed": self.trimmed,
            "coordinate_scale": self.coordinate_scale, "scale_source": self.scale_source,
            "coordinate_base_shape_zyx": self.coordinate_base_shape_zyx,
        }


def read_fibre(data: bytes, *, name: str, base_shape_zyx: tuple[int, int, int] | list[int]) -> Fibre:
    """Parse one fibre file's bytes; ``base_shape_zyx`` is the declared frame's shape (z, y, x)."""
    base = _shape(base_shape_zyx, "base_shape_zyx", name)
    try:
        doc = jsonstrict.loads(data)
    except jsonstrict.StrictJsonError as exc:
        raise Refusal("EVAL_FIBRE_JSON_INVALID", f"{name}: {exc}", fibre=name, problem=str(exc)) from None
    if not isinstance(doc, dict):
        raise Refusal("EVAL_FIBRE_STRUCTURE", f"{name}: the document must be an object", fibre=name,
                      field="(root)")
    scale, source, domain = DEFAULT_SCALE, "default-0.25", None
    if "coordinate_base_shape_zyx" in doc and doc["coordinate_base_shape_zyx"] is not None:
        domain = _shape(doc["coordinate_base_shape_zyx"], "coordinate_base_shape_zyx", name)
        scale, source = dyadic_scale(domain, base, name), "coordinate_base_shape_zyx"
    version = doc.get("version", 1)
    if version == 1 and not isinstance(version, bool) and not doc.get("control_points"):
        raise Refusal("EVAL_FIBRE_NO_CONTROLS", f"{name}: a version-1 fibre without control points", fibre=name,
                      version=1)
    version, mode, line, controls = _parse(doc, name)
    if controls.ndim != 2 or len(controls) == 0:
        raise Refusal("EVAL_FIBRE_NO_CONTROLS", f"{name}: no control points", fibre=name, version=version)

    if len(line) >= len(controls):
        d2 = ((line[None, :, :] - controls[:, None, :]) ** 2).sum(axis=-1)  # float32, as villa
        nearest = d2.argmin(axis=1)
        points, trimmed = line[int(nearest.min()):int(nearest.max()) + 1], True
    else:
        points, trimmed = controls, False
    points = (points * np.float32(scale)).astype(np.float64)[:, ::-1].copy()
    _check_frame(points, base, name)
    points.flags.writeable = False
    meta = {k: doc[k] for k in ("name", "generation", "sequence", "started_at", "hv_classification") if k in doc}
    return Fibre(name=name, digest=sha256_bytes(data), version=version, optimization_mode=mode, points_zyx=points,
                 control_count=len(controls), line_count=len(line), trimmed=trimmed, coordinate_scale=float(scale),
                 scale_source=source, coordinate_base_shape_zyx=domain, meta=meta)


#: villa ``config.py:659-660`` and ``:622`` at ``e0bbb8b``: the H/V rule's thresholds and the fibre decimation spacing.
HV_MIN_Z_FRACTION = 0.8
HV_MIN_AUTO_CERTAINTY = 0.5
HV_POINT_SPACING = 40.0


def decimate(points: np.ndarray, spacing: float) -> np.ndarray:
    """villa's ``_decimate_ordered_points_min_spacing``: keep the first point, then each point at least
    ``spacing`` from the last one kept (no forced points: apok-eval does not read fibre links)."""
    if spacing <= 0 or len(points) <= 1:
        return points
    rows = np.asarray(points, dtype=np.float64).tolist()
    keep, (a, b, c) = [0], rows[0]
    for i in range(1, len(rows)):
        x, y, z = rows[i]
        if math.sqrt((x - a) ** 2 + (y - b) ** 2 + (z - c) ** 2) >= spacing:
            keep.append(i)
            a, b, c = x, y, z
    return points[keep]


def hv_family(fibre: Fibre, *, min_z_fraction: float = HV_MIN_Z_FRACTION,
              min_auto_certainty: float = HV_MIN_AUTO_CERTAINTY, spacing: float = HV_POINT_SPACING) -> tuple[str, str]:
    """``("H" | "V" | "U", source)``: villa's ``classify_fiber_hv`` (``spiral_helpers.py:286-320``) on the fibre's
    ``hv_classification`` and its points decimated as villa's loader does (``fiber_collection_hv_tag``).

    Precedence: VC3D's manual tag; its automatic tag when ``automatic_certainty`` ≥ ``min_auto_certainty``; then
    V when the z extent is at least ``min_z_fraction`` of the path length, H when at most 1 − ``min_z_fraction``,
    else untagged ("U"). ``source`` is "manual", "automatic", "geometry" or "none". Malformed metadata falls through
    as in villa: a non-object is ignored, an unreadable certainty counts as 0.
    """
    hv = fibre.meta.get("hv_classification")
    hv = hv if isinstance(hv, dict) else {}
    manual = str(hv.get("manual_tag") or "").strip().upper()
    if manual in ("H", "V"):
        return manual, "manual"
    automatic = str(hv.get("automatic_tag") or "").strip().upper()
    if automatic in ("H", "V"):
        try:
            certainty = float(hv.get("automatic_certainty", 0.0))
        except (TypeError, ValueError):
            certainty = 0.0
        if certainty >= min_auto_certainty:
            return automatic, "automatic"
    zyx = decimate(np.asarray(fibre.points_zyx, dtype=np.float64), spacing)
    if len(zyx) < 2:
        return "U", "none"
    length = float(np.linalg.norm(np.diff(zyx, axis=0), axis=-1).sum())
    if length <= 0.0:
        return "U", "none"
    z_fraction = float(zyx[:, 0].max() - zyx[:, 0].min()) / length
    if z_fraction >= min_z_fraction:
        return "V", "geometry"
    if z_fraction <= 1.0 - min_z_fraction:
        return "H", "geometry"
    return "U", "geometry"


def dyadic_scale(source: list[int], base: list[int], name: str = "fibre") -> float:
    """villa's domain match: the unique 2^k, k in [-5, 5], mapping ``source`` onto ``base``."""
    scales = []
    for exponent in range(-5, 6):
        factor = 2 ** abs(exponent)
        large, small = (source, base) if exponent <= 0 else (base, source)
        if all((abs(a - b) <= 1) if exponent == 0 else (b in ((a + factor - 1) // factor, max(1, a // factor)))
               for a, b in zip(large, small)):
            scales.append(2.0 ** exponent)
    if len(scales) != 1:
        why = "no dyadic scale" if not scales else f"several dyadic scales {scales}"
        raise Refusal("EVAL_FIBRE_DOMAIN", f"{name}: {why} maps coordinate_base_shape_zyx {source} onto the "
                      f"declared base shape {base}", fibre=name, source_shape=source, base_shape=base,
                      candidates=scales)
    return scales[0]


def _shape(value: Any, label: str, name: str) -> list[int]:
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v <= 0 for v in value)):
        raise Refusal("EVAL_FIBRE_DOMAIN", f"{name}: {label} must be three positive integers, not {value!r}",
                      fibre=name, field=label, value=repr(value))
    return [int(v) for v in value]


def _parse(doc: dict, name: str) -> tuple[int, str, np.ndarray, np.ndarray]:
    if doc.get("type", "vc3d_fiber") != "vc3d_fiber":
        raise Refusal("EVAL_FIBRE_STRUCTURE", f"{name}: type must be 'vc3d_fiber'", fibre=name, field="type")
    version = doc.get("version", 1)
    if not isinstance(version, int) or isinstance(version, bool) or version not in FIBRE_VERSIONS:
        raise Refusal("EVAL_FIBRE_STRUCTURE", f"{name}: version must be one of {list(FIBRE_VERSIONS)}, not {version!r}",
                      fibre=name, field="version")
    if version >= 3 and "type" not in doc:
        raise Refusal("EVAL_FIBRE_STRUCTURE", f"{name}: a version-3/4 fibre must state its type",
                      fibre=name, field="type")
    if version >= 3 and "optimization_mode" not in doc:
        raise Refusal("EVAL_FIBRE_STRUCTURE", f"{name}: a version-3/4 fibre must state optimization_mode",
                      fibre=name, field="optimization_mode")
    mode = doc.get("optimization_mode", "lasagna")
    if not isinstance(mode, str) or mode not in OPTIMIZATION_MODES:
        raise Refusal("EVAL_FIBRE_STRUCTURE", f"{name}: optimization_mode must be one of {list(OPTIMIZATION_MODES)}",
                      fibre=name, field="optimization_mode")
    line = _points(doc.get("line_points"), "line_points", name)
    raw = doc.get("control_points")
    if not isinstance(raw, list):
        raise Refusal("EVAL_FIBRE_STRUCTURE", f"{name}: control_points must be a list",
                      fibre=name, field="control_points")
    if version == 1:
        controls = _points(raw, "control_points", name)
    else:
        positions = []
        for i, control in enumerate(raw):
            where = f"control_points[{i}]"
            if not isinstance(control, dict) or not set(control) <= _CONTROL_KEYS:
                raise Refusal("EVAL_FIBRE_STRUCTURE", f"{name}: a version-3/4 control point may hold only position, "
                              f"segment_to_next and tags",
                              fibre=name, field=where)
            if "position" not in control:
                raise Refusal("EVAL_FIBRE_STRUCTURE", f"{name}: a version-3/4 control point needs position",
                              fibre=name, field=where)
            tags = control.get("tags", [])
            if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
                raise Refusal("EVAL_FIBRE_STRUCTURE", f"{name}: control point tags must be a list of strings",
                              fibre=name, field=where)
            last = i + 1 == len(raw)
            if last and "segment_to_next" in control:
                raise Refusal("EVAL_FIBRE_STRUCTURE", f"{name}: the final control point cannot carry segment_to_next",
                              fibre=name, field=where)
            if not last and not isinstance(control.get("segment_to_next"), dict):
                raise Refusal("EVAL_FIBRE_STRUCTURE", f"{name}: every control point but the last needs a "
                              f"segment_to_next object",
                              fibre=name, field=where)
            positions.append(control["position"])
        controls = _points(positions, "control_points", name)
    generation = doc.get("generation", 1)
    if version >= 3:
        if not isinstance(generation, int) or isinstance(generation, bool) or generation < 0:
            raise Refusal("EVAL_FIBRE_STRUCTURE", f"{name}: generation must be a non-negative integer",
                          fibre=name, field="generation")
    else:
        try:
            int(generation)
        except (TypeError, ValueError, OverflowError):
            raise Refusal("EVAL_FIBRE_STRUCTURE", f"{name}: generation must be an integer",
                          fibre=name, field="generation") from None
    return version, mode, line, controls


def _points(raw: Any, label: str, name: str) -> np.ndarray:
    if not isinstance(raw, list):
        raise Refusal("EVAL_FIBRE_STRUCTURE", f"{name}: {label} must be a list", fibre=name, field=label)
    for i, point in enumerate(raw):
        if (not isinstance(point, list) or len(point) != 3
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in point)):
            raise Refusal("EVAL_FIBRE_STRUCTURE", f"{name}: {label}[{i}] must be [x, y, z] numbers",
                          fibre=name, field=f"{label}[{i}]")
    if not raw:
        return np.zeros((0, 3), dtype=np.float32)
    with np.errstate(over="ignore"):
        points = np.asarray(raw, dtype=np.float32).reshape(-1, 3)
    if not np.isfinite(points).all():
        raise Refusal("EVAL_FIBRE_STRUCTURE", f"{name}: {label} holds a number that is not finite in float32",
                      fibre=name, field=label)
    return points


def _check_frame(points_zyx: np.ndarray, base: list[int], name: str) -> None:
    if not len(points_zyx):
        return
    low = points_zyx.min(axis=0)
    high = points_zyx.max(axis=0)
    shape = np.asarray(base, dtype=np.float64)
    if (low < -1).any() or (high > shape).any():
        raise Refusal("EVAL_FIBRE_OUTSIDE_FRAME", f"{name}: scaled points span z,y,x {low.tolist()} to "
                      f"{high.tolist()}, outside the frame [0, {base}) by more than one voxel", fibre=name,
                      low_zyx=low.tolist(), high_zyx=high.tolist(), frame_shape_zyx=base)
