"""The coordinate contract (the rule: "Coordinates (z, y, x) in full-resolution
voxels of the volume the dataset declares — read off the artefact and checked
before any read, never assumed").

A :class:`VolumeFrame` is a *declaration*: which volume coordinates refer to,
its level-0 shape, and its voxel size. It comes from the asset registry or the
run configuration, never from an artefact's own unit labels (OME-Zarr writers
label relative factors as micrometres; tifxyz meshes record no volume at all).
Readers compare what an artefact says against the declared frame and refuse
on any disagreement.

The frame's numbers are compared as float64 (:func:`check_points_within`), so
every integer in a declaration must be one float64 holds exactly: a shape
extent or an integral voxel size beyond 2**53 is ``IO_FRAME_INVALID`` (a
voxel size of 10**400 cannot even be converted). :meth:`VolumeFrame.from_dict`
reads the frame-file schema (``volume_id``, ``shape_zyx``, ``voxel_size_um``,
``source``, optionally ``axes`` = ``["z", "y", "x"]``), which ``apok-eval io inspect
--frame`` and run configurations use.

:func:`stated_identity` reads what an artefact itself says about its volume
(the rule: "read off the artefact and checked before any read, never
assumed") and compares it with the declared frame: a ``volume`` or
``target_volume`` that names a 14-digit volume id (VC3D's
``Segmentation::ensureScrollSource`` and ``vc_grow_seg_from_seed`` write
them; a directory name counts by its leading id) must be the frame's, and a
``base_shape_zyx`` of three positive ints (spiral combined previews, lasagna
exports) must be its shape, else ``IO_FRAME_VOLUME_MISMATCH``. A fact the
artefact does not state is reported ``not stated by the artefact``; a value
that names no volume id is reported as not comparable, never guessed.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from apok_eval.refusal import Refusal, register_codes

register_codes(
    "apok_eval.io.frame",
    {
        "IO_FRAME_INVALID": "a VolumeFrame declaration is malformed",
        "IO_FRAME_POINTS_OUTSIDE": "coordinates fall outside the declared volume's level-0 shape",
        "IO_FRAME_POINTS_MALFORMED": "a point array is not an (N, 3) finite ZYX array",
        "IO_FRAME_VOLUME_MISMATCH": "an artefact states a volume id or level-0 shape other than the declared frame's",
    },
)

AXES = ("z", "y", "x")
IDENTITY_KEYS = ("volume", "target_volume")
_STATED_VOLUME_ID = re.compile(r"^([0-9]{14})(?![0-9])")
FRAME_KEYS = ("volume_id", "shape_zyx", "voxel_size_um", "source")
MAX_EXACT_INT = 2 ** 53  # float64 holds every integer up to this magnitude exactly
_VOLUME_ID = re.compile(r"^[0-9]{14}$")


def finite_number(value: Any) -> float | None:
    """``value`` as a double if it is a finite non-bool JSON number (huge ints included), else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        as_double = float(value)
    except OverflowError:  # an int of 2**1024 or more: no float holds it
        return None
    return as_double if math.isfinite(as_double) else None


@dataclass(frozen=True)
class VolumeFrame:
    """Level-0 voxel grid of one declared volume. Axes are always (z, y, x)."""

    volume_id: str
    shape_zyx: tuple[int, int, int]
    voxel_size_um: float
    source: str  # where the declaration came from, e.g. "registry:<asset id>"

    def __post_init__(self) -> None:
        if not isinstance(self.volume_id, str) or not _VOLUME_ID.match(self.volume_id):
            raise Refusal("IO_FRAME_INVALID", "volume_id must be a 14-digit volume id", field="volume_id",
                          value=self.volume_id)
        shape = self.shape_zyx
        if (not isinstance(shape, tuple) or len(shape) != 3
                or not all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in shape)):
            raise Refusal("IO_FRAME_INVALID", "shape_zyx must be three positive ints", field="shape_zyx",
                          value=shape)
        if any(n > MAX_EXACT_INT for n in shape):
            raise Refusal("IO_FRAME_INVALID", "shape_zyx holds an extent beyond 2**53, which the float64 frame "
                          "comparison cannot hold exactly", field="shape_zyx", value=shape)
        v = self.voxel_size_um
        if isinstance(v, int) and not isinstance(v, bool) and abs(v) > MAX_EXACT_INT:
            raise Refusal("IO_FRAME_INVALID", "voxel_size_um is an integer beyond 2**53, which a float cannot hold "
                          "exactly", field="voxel_size_um", value=v)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise Refusal("IO_FRAME_INVALID", "voxel_size_um must be a finite positive number",
                          field="voxel_size_um", value=v)
        if not isinstance(self.source, str) or not self.source.strip():
            raise Refusal("IO_FRAME_INVALID", "a frame must say where its declaration came from",
                          field="source", value=self.source)

    def as_dict(self) -> dict:
        return {"volume_id": self.volume_id, "shape_zyx": list(self.shape_zyx),
                "voxel_size_um": float(self.voxel_size_um), "axes": list(AXES), "source": self.source}

    @classmethod
    def from_dict(cls, document: Any, *, where: str, source: str | None = None) -> VolumeFrame:
        """The frame a JSON declaration states (the frame-file schema; see the module docstring).

        ``where`` names the declaration in refusals. ``source`` is for a declaration that does not carry
        its own (a run configuration's ``frame`` block): it is then required not to.
        """
        if not isinstance(document, dict):
            raise Refusal("IO_FRAME_INVALID", f"{where} holds a JSON {type(document).__name__}, not an object",
                          declaration=where, field="frame", problem="not an object")
        keys = FRAME_KEYS if source is None else FRAME_KEYS[:-1]
        missing = [key for key in keys if key not in document]
        unknown = sorted(set(document) - set(keys) - {"axes"})
        if missing or unknown:
            raise Refusal("IO_FRAME_INVALID", f"{where} must hold exactly {list(keys)} (and optionally axes); "
                          f"missing {missing}, unknown {unknown}", declaration=where, field="keys",
                          missing=missing, unknown=unknown)
        if "axes" in document and document["axes"] != list(AXES):
            raise Refusal("IO_FRAME_INVALID", f"{where} declares axes {document['axes']!r}; a frame's axes are "
                          f"always {list(AXES)}", declaration=where, field="axes", value=document["axes"])
        shape = document["shape_zyx"]
        return cls(document["volume_id"], tuple(shape) if isinstance(shape, list) else shape,
                   document["voxel_size_um"], document["source"] if source is None else source)


def stated_identity(frame: VolumeFrame, stated: Any, what: str) -> dict[str, str]:
    """Compare what an artefact states about its volume (a mapping such as a tifxyz ``meta.json``) with ``frame``;
    refuse a disagreement, and return ``{fact: "agrees" | "not stated by the artefact" | "not comparable: ..."}``."""
    stated = stated if isinstance(stated, dict) else {}
    facts: dict[str, str] = {}
    for key in IDENTITY_KEYS:
        value = stated.get(key)
        match = _STATED_VOLUME_ID.match(value) if isinstance(value, str) else None
        if value is None:
            facts[key] = "not stated by the artefact"
        elif match is None:
            facts[key] = f"not comparable: {value!r:.80} names no 14-digit volume id"
        elif match.group(1) != frame.volume_id:
            raise Refusal("IO_FRAME_VOLUME_MISMATCH", f"{what}: {key} {value!r} names volume {match.group(1)}, not the "
                          f"declared frame's {frame.volume_id} ({frame.source})", what=what, field=key, stated=value,
                          declared=frame.volume_id)
        else:
            facts[key] = "agrees"
    shape = stated.get("base_shape_zyx")
    if shape is None:
        facts["base_shape_zyx"] = "not stated by the artefact"
    elif not (isinstance(shape, list) and len(shape) == 3
              and all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in shape)):
        facts["base_shape_zyx"] = f"not comparable: {shape!r:.80} is not three positive ints"
    elif tuple(shape) != frame.shape_zyx:
        raise Refusal("IO_FRAME_VOLUME_MISMATCH", f"{what}: base_shape_zyx {shape} is not the declared frame's level-0 "
                      f"shape {list(frame.shape_zyx)} (volume {frame.volume_id})", what=what, field="base_shape_zyx",
                      stated=shape, declared=list(frame.shape_zyx))
    else:
        facts["base_shape_zyx"] = "agrees"
    return facts


def check_points_within(frame: VolumeFrame, zyx: np.ndarray, what: str) -> None:
    """Refuse unless every row of ``zyx`` (N, 3) lies in [0, shape) on each axis.

    ``zyx`` must already contain only the points the caller's declared validity
    policy counts as valid; this function does not decide validity.
    """
    pts = np.asarray(zyx)
    numeric = np.issubdtype(pts.dtype, np.floating) or np.issubdtype(pts.dtype, np.integer)
    if pts.ndim != 2 or pts.shape[1] != 3 or not numeric:
        raise Refusal("IO_FRAME_POINTS_MALFORMED", f"{what}: expected an (N, 3) numeric ZYX array",
                      what=what, shape=list(pts.shape), dtype=str(pts.dtype))
    if not np.isfinite(pts).all():
        raise Refusal("IO_FRAME_POINTS_MALFORMED", f"{what}: non-finite coordinates among valid points",
                      what=what, count=int((~np.isfinite(pts)).any(axis=1).sum()))
    upper = np.asarray(frame.shape_zyx, dtype=np.float64)
    outside = ((pts < 0) | (pts >= upper)).any(axis=1)
    if outside.any():
        first = pts[np.argmax(outside)]
        raise Refusal(
            "IO_FRAME_POINTS_OUTSIDE",
            f"{what}: {int(outside.sum())} of {len(pts)} points lie outside volume {frame.volume_id} "
            f"(level-0 shape zyx {list(frame.shape_zyx)})",
            what=what, volume_id=frame.volume_id, count=int(outside.sum()), total=int(len(pts)),
            example_zyx=[float(c) for c in first],
        )
