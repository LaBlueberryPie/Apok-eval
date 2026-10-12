"""Geometric leakage audit: which fibre points lie on or beside a fit's own input surfaces.

A fit that read verified patches has seen those surfaces. A fibre point within ``radius`` voxels of one of them
is not held out from the fit geometrically, whatever the file names say. The idea is spiralcheck's
(``github.com/Nicodol/spiralcheck``, MIT; its notice is in ``THIRD_PARTY_NOTICES.md``); this is apok-eval's own
implementation, for fibre points against tifxyz patches.

- **Distance.** Exact Euclidean distance from a point to a patch's surface: each quad whose four corners are
  valid is split into two triangles along the (0, 0)-(1, 1) diagonal, and the distance to the nearer triangle
  is taken. Quads with an invalid corner are holes.
- **Search.** A k-d tree over the points; a quad is examined only when some point lies within its
  circumradius plus ``radius`` of its centre, and then only those points.
- **Output.** A boolean per point: within ``radius`` (inclusive) of any input surface. ``far_fibres`` drops the
  near points from each fibre, so the evaluation can be run on all points and on the far subset.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable

import numpy as np
from scipy.spatial import cKDTree

from apok_eval.eval.fibres import Fibre


def _dot(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", u, v)


def point_triangle_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Distance from each point p[i] to the triangle (a[i], b[i], c[i]) (Ericson, Real-Time Collision Detection,
    5.1.5), vectorised. A degenerate triangle gives the distance to its nearest vertex."""
    p, a, b, c = (np.asarray(x, dtype=np.float64) for x in (p, a, b, c))
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = _dot(ab, ap), _dot(ac, ap)
    bp, cp = p - b, p - c
    d3, d4 = _dot(ab, bp), _dot(ac, bp)
    d5, d6 = _dot(ab, cp), _dot(ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    out = np.empty_like(p)
    done = np.zeros(len(p), bool)

    def take(mask, value):
        m = mask & ~done
        out[m] = value[m]
        done[m] = True

    with np.errstate(divide="ignore", invalid="ignore"):
        take((d1 <= 0) & (d2 <= 0), a)
        take((d3 >= 0) & (d4 <= d3), b)
        take((d6 >= 0) & (d5 <= d6), c)
        take((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + (d1 / (d1 - d3))[:, None] * ab)
        take((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + (d2 / (d2 - d6))[:, None] * ac)
        take((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0),
             b + ((d4 - d3) / ((d4 - d3) + (d5 - d6)))[:, None] * (c - b))
        denom = va + vb + vc
        take(np.ones(len(p), bool), a + (vb / denom)[:, None] * ab + (vc / denom)[:, None] * ac)
        dist = np.linalg.norm(p - out, axis=1)
    vertex = np.min([np.linalg.norm(p - v, axis=1) for v in (a, b, c)], axis=0)
    return np.where(np.isfinite(dist), np.minimum(dist, vertex), vertex)


def near_surfaces(points_zyx: np.ndarray, surfaces: Iterable[tuple[np.ndarray, np.ndarray]], *,
                  radius: float = 2.0) -> np.ndarray:
    """For each point, whether it lies within ``radius`` voxels of any surface; ``surfaces`` yields
    (zyx (H, W, 3), valid (H, W)) grids."""
    points = np.asarray(points_zyx, dtype=np.float64).reshape(-1, 3)
    near = np.zeros(len(points), bool)
    if not len(points):
        return near
    tree = cKDTree(points)
    for zyx, valid in surfaces:
        zyx = np.asarray(zyx, dtype=np.float64)
        valid = np.asarray(valid, bool)
        q = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
        if not q.any():
            continue
        a, b, c, d = zyx[:-1, :-1][q], zyx[:-1, 1:][q], zyx[1:, 1:][q], zyx[1:, :-1][q]
        centre = (a + b + c + d) / 4
        reach = np.max([np.linalg.norm(v - centre, axis=1) for v in (a, b, c, d)], axis=0) + radius
        first, _ = tree.query(centre, k=1, distance_upper_bound=float(reach.max()) + 1e-9)
        cand = np.nonzero(first <= reach)[0]
        if not len(cand):
            continue
        lists = tree.query_ball_point(centre[cand], reach[cand])
        quad = np.repeat(cand, [len(x) for x in lists])
        point = np.fromiter((i for x in lists for i in x), dtype=np.int64, count=len(quad))
        todo = ~near[point]
        quad, point = quad[todo], point[todo]
        if not len(point):
            continue
        p = points[point]
        dist = np.minimum(point_triangle_distance(p, a[quad], b[quad], c[quad]),
                          point_triangle_distance(p, a[quad], c[quad], d[quad]))
        near[point[dist <= radius]] = True
    return near


def audit(fibres: list[Fibre], surfaces: Iterable[tuple[np.ndarray, np.ndarray]], *, z_range: tuple[float, float],
          radius: float = 2.0) -> tuple[dict, list[np.ndarray]]:
    """The share of each fibre's in-slab points near an input surface, and per-fibre near masks over all of a
    fibre's points (points outside the slab are never near). ``surfaces`` may be a generator: each surface is
    read once, so a fit's patches need not fit in memory together."""
    counted = []

    def each():
        for surface in surfaces:
            counted.append(1)
            yield surface

    z0, z1 = z_range
    masks, rows = [], []
    slab = [(f.points_zyx[:, 0] >= z0) & (f.points_zyx[:, 0] < z1) for f in fibres]
    pts = np.concatenate([f.points_zyx[m] for f, m in zip(fibres, slab)]) if fibres else np.zeros((0, 3))
    near = near_surfaces(pts, each(), radius=radius)
    start = 0
    for f, m in zip(fibres, slab):
        n = int(m.sum())
        mask = np.zeros(len(f.points_zyx), bool)
        mask[np.nonzero(m)[0]] = near[start:start + n]
        start += n
        masks.append(mask)
        if n:
            rows.append({"name": f.name, "points": n, "near": int(mask.sum())})
    total = int(sum(r["points"] for r in rows))
    report = {"radius": radius, "surfaces": len(counted), "fibres": len(rows), "points": total,
              "near": int(near.sum()), "share_near": float(near.sum() / total) if total else None,
              "fibres_touching": int(sum(r["near"] > 0 for r in rows)), "per_fibre": rows,
              "credit": "after spiralcheck's geometric leakage audit (github.com/Nicodol/spiralcheck, MIT)"}
    return report, masks


def far_fibres(fibres: list[Fibre], masks: list[np.ndarray]) -> list[Fibre]:
    """Each fibre without its near points (a fibre left with none is dropped)."""
    out = []
    for f, m in zip(fibres, masks):
        keep = f.points_zyx[~m]
        if len(keep):
            keep = keep.copy()
            keep.flags.writeable = False
            out.append(dataclasses.replace(f, points_zyx=keep))
    return out
