"""Synthetic spiral worlds for the demo and the tests: fits, fibres, an umbilicus and a layered CT.

Nothing here is scroll data. The world is an Archimedean spiral about a vertical umbilicus: winding ω sits at
radius ``radius(ω)`` (``spacing * ω`` by default) and angle sense * 2π (ω - phase). Segments are tifxyz grids
named ``<id>-wLLL-HHH`` whose rows are z levels ``step`` apart and whose columns sample ω at arc steps of at
most ``step`` voxels, from LLL to HHH + 1 inclusive. Fibres are written as VC3D ``vc3d_fiber`` documents, the
umbilicus in the spiral fitter's ``control_points`` form, and the CT as an OME-Zarr v2 volume whose bright
layers follow the spiral's sheets.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np

from apok_eval.io.tifxyz import write_tifxyz
from apok_eval.io.umbilicus import write_umbilicus

CENTRE = (500.0, 500.0)  # (y, x)
FRAME = (1000, 1000, 1000)  # (z, y, x) of the default synthetic frame
STEP = 20.0


def spiral_point(omega, z, *, sense=1, phase=0.0, radius=None, spacing=16.0, centre=CENTRE):
    """(z, y, x) of the spiral at winding coordinate ``omega`` and height ``z`` (broadcast)."""
    omega = np.asarray(omega, dtype=np.float64)
    r = spacing * omega if radius is None else radius(omega)
    phi = sense * 2 * np.pi * (omega - phase)
    z = np.broadcast_to(np.asarray(z, dtype=np.float64), omega.shape)
    return np.stack([z, centre[0] + r * np.sin(phi), centre[1] + r * np.cos(phi)], axis=-1)


def omega_columns(lo, hi_end, *, radius, step=STEP):
    """Winding coordinates from ``lo`` to ``hi_end`` at arc steps of at most ``step`` voxels."""
    cols = [float(lo)]
    while cols[-1] < hi_end:
        cols.append(min(hi_end, cols[-1] + step / (2 * np.pi * max(radius(cols[-1]), 1.0))))
    return np.array(cols)


def meta_for(name, x, y, z):
    """A tifxyz ``meta.json`` for a synthetic grid (the valid vertices' bounding box)."""
    ok = z > 0
    return {"uuid": name, "type": "seg", "format": "tifxyz", "scale": [1 / STEP, 1 / STEP],
            "bbox": [[float(x[ok].min()), float(y[ok].min()), float(z[ok].min())],
                     [float(x[ok].max()), float(y[ok].max()), float(z[ok].max())]]}


def write_segment(root: Path, name: str, lo: int, hi: int, *, z_levels, sense=1, phase=0.0, radius=None,
                  spacing=16.0, holes=None, centre=CENTRE, sheet_radius=None) -> Path:
    """One fit segment ``root/name`` holding windings ``lo`` to ``hi``.

    ``radius(ω)`` places the sheets (default ``spacing * ω``); ``sheet_radius(ω, z)``, when given, places them
    at each height instead (a breathing scroll), with ``radius`` still setting the column spacing.
    """
    rad = radius or (lambda w: spacing * np.asarray(w))
    omegas = omega_columns(lo, hi + 1, radius=rad)
    zz, ww = np.meshgrid(np.asarray(z_levels, dtype=np.float64), omegas, indexing="ij")
    by_height = rad if sheet_radius is None else (lambda w: sheet_radius(w, zz))
    zyx = spiral_point(ww, zz, sense=sense, phase=phase, radius=by_height, centre=centre)
    x, y, z = zyx[..., 2].astype(np.float32), zyx[..., 1].astype(np.float32), zyx[..., 0].astype(np.float32)
    if holes is not None:
        for rows, cols in holes:
            for plane in (x, y, z):
                plane[rows, cols] = -1.0
    directory = root / name
    write_tifxyz(directory, x=x, y=y, z=z, meta=meta_for(name, x, y, z))
    return directory


def write_fit(root: Path, ranges, *, z_levels=None, prefix="20990101000000", **kw) -> list[Path]:
    """One segment per (lo, hi) winding range, named ``<prefix>-wLLL-HHH``."""
    z_levels = np.arange(0.0, 401.0, STEP) if z_levels is None else z_levels
    return [write_segment(root, f"{prefix}-w{lo:03d}-{hi:03d}", lo, hi, z_levels=z_levels, **kw)
            for lo, hi in ranges]


def write_shell(root: Path, radius: float, *, z_levels=None, centre=CENTRE) -> Path:
    """A closed cylindrical outer shell of the given radius, as a tifxyz grid ``root/outer_shell``."""
    z_levels = np.arange(0.0, 401.0, STEP) if z_levels is None else z_levels
    n = int(math.ceil(2 * np.pi * radius / STEP))
    phi = np.linspace(-np.pi, np.pi, n + 1)  # closed: the last column repeats the first angle
    zz, pp = np.meshgrid(np.asarray(z_levels, dtype=np.float64), phi, indexing="ij")
    y = (centre[0] + radius * np.sin(pp)).astype(np.float32)
    x = (centre[1] + radius * np.cos(pp)).astype(np.float32)
    z = zz.astype(np.float32)
    directory = root / "outer_shell"
    write_tifxyz(directory, x=x, y=y, z=z, meta=meta_for("outer_shell", x, y, z))
    return directory


def write_umbilicus_file(root: Path, centre=CENTRE, *, z_top: float = 1000.0) -> Path:
    """A vertical umbilicus through ``centre`` from z = 0 to ``z_top``, as ``root/umbilicus.json``."""
    path = root / "umbilicus.json"
    write_umbilicus(path, np.array([[0.0, centre[0], centre[1]], [float(z_top), centre[0], centre[1]]]))
    return path


def fibre_points(omega_start, omega_end, z, *, offset=0.0, sense=1, phase=0.0, spacing=16.0, radius=None, step=1.0,
                 centre=CENTRE):
    """Dense (z, y, x) points along a horizontal fibre at radius(ω) + offset, about ``step`` voxels apart."""
    rad = radius or (lambda w: spacing * np.asarray(w))
    cols = [float(omega_start)]
    while cols[-1] < omega_end:
        cols.append(cols[-1] + step / (2 * np.pi * float(rad(cols[-1]))))
    w = np.array(cols)
    return spiral_point(w, z, sense=sense, phase=phase, radius=lambda v: rad(v) + offset, centre=centre)


def fibre_document(points_zyx, *, scale=4.0, version=1, domain=None, controls_every=50, extra=None) -> dict:
    """A vc3d_fiber document: points stored (x, y, z) times ``scale``, controls every ``controls_every``."""
    xyz = (np.asarray(points_zyx)[:, ::-1] * scale).tolist()
    idx = list(range(0, len(xyz), controls_every))
    if idx[-1] != len(xyz) - 1:
        idx.append(len(xyz) - 1)
    pad = [[xyz[0][0] - 1.0, xyz[0][1], xyz[0][2]]]  # the tracer runs past the first control point
    line = pad + xyz + [[xyz[-1][0] + 1.0, xyz[-1][1], xyz[-1][2]]]
    doc = {"type": "vc3d_fiber", "version": version, "line_points": line}
    if version == 1:
        doc["control_points"] = [xyz[i] for i in idx]
    else:
        doc["optimization_mode"] = "lasagna"
        doc["control_points"] = [{"position": xyz[i], **({} if k == len(idx) - 1 else {"segment_to_next": {}})}
                                 for k, i in enumerate(idx)]
    if domain is not None:
        doc["coordinate_base_shape_zyx"] = list(domain)
    doc.update(extra or {})
    return doc


def fibre_bytes(points_zyx, **kw) -> bytes:
    return json.dumps(fibre_document(points_zyx, **kw)).encode()


# --- a layered CT -----------------------------------------------------------------------------------------


def layered_block(origin, shape, *, sheet_radius: Callable, lo: float, hi: float, centre=CENTRE, phase=0.0,
                  sense=1, spacing=16.0, width=1.5, bright=200.0, dark=20.0, noise=6.0, seed=0) -> np.ndarray:
    """A uint8 CT block at ``origin`` (z, y, x) of ``shape``: bright at the spiral's sheets, dark between them.

    The sheet of winding ω at height z lies at radius ``sheet_radius(ω, z)``, which must be ``spacing * ω`` plus
    a function of z alone (a breathing spiral). A voxel's value is ``dark + (bright - dark) * exp(-d² / 2w²)``,
    d its radial distance to the nearest sheet, plus N(0, ``noise``) seeded by ``seed`` and the block's origin;
    voxels inside winding ``lo`` or outside winding ``hi + 1`` (half a spacing of margin) are air (0).
    """
    origin = np.asarray(origin, dtype=np.int64)
    z, y, x = np.meshgrid(*[np.arange(o, o + n, dtype=np.float64) for o, n in zip(origin, shape)], indexing="ij")
    dy, dx = y - centre[0], x - centre[1]
    r = np.hypot(dy, dx)
    f = np.mod(sense * np.arctan2(dy, dx) / (2 * np.pi) + phase, 1.0)
    base = sheet_radius(np.zeros_like(z), z)  # the z-dependent part of every sheet's radius
    u = (r - base) / spacing - f  # gap coordinate: an integer on a sheet
    d = spacing * np.abs(u - np.round(u))
    value = dark + (bright - dark) * np.exp(-d * d / (2 * width * width))
    rng = np.random.default_rng([int(seed), *(int(v) for v in origin)])
    value = value + rng.normal(0.0, noise, value.shape)
    inside = (r - base >= spacing * (lo - 0.5)) & (r - base <= spacing * (hi + 1.5))
    return np.clip(np.where(inside, value, 0.0), 0, 255).astype(np.uint8)


def write_ct(root: Path, *, frame_shape, chunks=(32, 32, 32), z_range=None, block: Callable,
             compressor: dict | None = None) -> Path:
    """Write a one-level OME-Zarr v2 volume of ``frame_shape`` at ``root``; level 0 is the frame.

    Only chunks overlapping ``z_range`` (default: all) are written, each from ``block(origin, shape)``; chunks
    elsewhere are absent and read as the fill value 0, as a masked volume's are. ``compressor`` is a numcodecs
    configuration from the reader's allow-list (default ``{"id": "zlib", "level": 1}``).
    """
    compressor = {"id": "zlib", "level": 1} if compressor is None else compressor
    root.mkdir(parents=True, exist_ok=True)
    (root / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    ms = {"version": "0.4", "name": "apok-eval synthetic CT",
          "axes": [{"name": a, "type": "space", "unit": "micrometer"} for a in ("z", "y", "x")],
          "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 1.0]}]}]}
    (root / ".zattrs").write_text(json.dumps({"multiscales": [ms]}))
    level = root / "0"
    level.mkdir(exist_ok=True)
    meta = {"zarr_format": 2, "shape": [int(n) for n in frame_shape], "chunks": [int(c) for c in chunks],
            "dtype": "|u1", "compressor": compressor, "fill_value": 0, "order": "C", "filters": None,
            "dimension_separator": "."}
    (level / ".zarray").write_text(json.dumps(meta, indent=1))
    codec = None
    if compressor is not None:
        import numcodecs

        codec = numcodecs.get_codec(dict(compressor))
    z0, z1 = (0, frame_shape[0]) if z_range is None else z_range
    grid = [range(-(-int(n) // int(c))) for n, c in zip(frame_shape, chunks)]
    for index in itertools.product(*grid):
        origin = [i * c for i, c in zip(index, chunks)]
        if origin[0] + chunks[0] <= z0 or origin[0] >= z1:
            continue
        data = np.ascontiguousarray(block(origin, tuple(chunks))).astype(np.uint8)
        if not data.any():
            continue  # all air: left absent, as a masked writer does
        raw = data.tobytes()
        (level / ".".join(str(i) for i in index)).write_bytes(raw if codec is None else codec.encode(raw))
    return root
