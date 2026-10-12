"""Geometry for the evaluator: exact distances to bilinear tifxyz surfaces, and 2-D charts.

Every surface here is a set of quads taken from tifxyz grids, each quad the bilinear patch through its four
corners (the interpolation the evaluator fixes). Coordinates are (z, y, x) in the spiral frame.

- ``QuadSurface`` holds the quads and answers "how far is each point from the surface, and where is the
  closest point" exactly up to a stated bound: a k-d tree over corners gives a radius that provably contains
  the closest quad, bounding spheres and then a two-triangle pass with the Hausdorff bound |T|/4 (T the
  quad's twist) prune the candidates, and each survivor is subdivided s² times, s the least that brings
  |T| / (4 s²) under ``tolerance`` (at most ``subdivisions``). Every answer carries its bound.
- ``Chart`` interpolates a payload over quads laid out in two chart coordinates: (winding, z) for a fit,
  (angle, z) for the outer shell. A query that falls in no quad returns NaN; nothing is extrapolated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from apok_eval.refusal import Refusal, register_codes

register_codes(
    "apok_eval.eval.geometry",
    {
        "EVAL_GEOMETRY_INVALID": "an array given to the evaluator's geometry is not finite or has the wrong shape",
        "EVAL_GEOMETRY_EMPTY": "a surface or chart has no valid quad to measure against",
        "EVAL_GEOMETRY_PARAMETER": "a geometry parameter (cap, subdivisions, neighbour count, scale) is out of range",
    },
)

#: Triangulation of the unit parameter square; corner order is P00, P01 (next column), P10 (next row), P11.
_UNIT_TRIANGLES = ((0, 1, 3), (0, 3, 2))
_CORNER_UV = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # (u along columns, v along rows)


def quads_from_grid(zyx: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Corners (Q, 4, 3) and grid indices (Q, 2) of every quad whose four corners are valid and finite."""
    zyx = np.asarray(zyx, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if zyx.ndim != 3 or zyx.shape[2] != 3 or valid.shape != zyx.shape[:2]:
        raise Refusal("EVAL_GEOMETRY_INVALID", f"a grid must be (H, W, 3) with an (H, W) validity mask, not "
                      f"{list(zyx.shape)} and {list(valid.shape)}", grid_shape=list(zyx.shape),
                      valid_shape=list(valid.shape))
    ok = valid & np.isfinite(zyx).all(axis=2)
    q = ok[:-1, :-1] & ok[:-1, 1:] & ok[1:, :-1] & ok[1:, 1:]
    rows, cols = np.nonzero(q)
    corners = np.stack([zyx[rows, cols], zyx[rows, cols + 1], zyx[rows + 1, cols], zyx[rows + 1, cols + 1]], axis=1)
    return corners, np.stack([rows, cols], axis=1)


def twist(corners: np.ndarray) -> np.ndarray:
    """|P00 - P01 - P10 + P11| per quad: the bilinear term's size, which bounds every triangulation error."""
    return np.linalg.norm(corners[:, 0] - corners[:, 1] - corners[:, 2] + corners[:, 3], axis=1)


def diameter(corners: np.ndarray) -> np.ndarray:
    """Largest distance between two corners of each quad (the patch lies in their convex hull)."""
    best = np.zeros(len(corners))
    for a in range(4):
        for b in range(a + 1, 4):
            best = np.maximum(best, np.linalg.norm(corners[:, a] - corners[:, b], axis=1))
    return best


def closest_on_triangles(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Squared distance from ``p`` to triangle (a, b, c) and the closest point's barycentric weights.

    Vectorised over any leading shape (Ericson, *Real-Time Collision Detection*, 5.1.5). Regions are tested
    in Ericson's order, so a degenerate triangle resolves to a vertex or an edge, never to a division by zero.
    """
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    bp = p - b
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    cp = p - c
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    with np.errstate(divide="ignore", invalid="ignore"):
        v_ab = d1 / (d1 - d3)
        w_ac = d2 / (d2 - d6)
        w_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        denom = va + vb + vc
        v_in, w_in = vb / denom, vc / denom
    zero = np.zeros_like(d1)
    one = np.ones_like(d1)
    conditions = [
        (d1 <= 0) & (d2 <= 0),
        (d3 >= 0) & (d4 <= d3),
        (vc <= 0) & (d1 >= 0) & (d3 <= 0),
        (d6 >= 0) & (d5 <= d6),
        (vb <= 0) & (d2 >= 0) & (d6 <= 0),
        (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0),
    ]
    v = np.select(conditions, [zero, one, v_ab, zero, zero, 1 - w_bc], v_in)
    w = np.select(conditions, [zero, zero, zero, one, w_ac, w_bc], w_in)
    u = 1 - v - w
    closest = u[..., None] * a + v[..., None] * b + w[..., None] * c
    d2_out = ((p - closest) ** 2).sum(-1)
    return d2_out, np.stack([u, v, w], axis=-1)


def _subdivision(s: int) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear weights (V, 4) of the (s+1)² sub-grid and its triangles (2s², 3) as sub-grid vertex indices."""
    t = np.linspace(0.0, 1.0, s + 1)
    vv, uu = np.meshgrid(t, t, indexing="ij")  # rows (v) then columns (u)
    u, v = uu.ravel(), vv.ravel()
    weights = np.stack([(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v], axis=1)
    tris = []
    for i in range(s):
        for j in range(s):
            k00, k01, k10, k11 = i * (s + 1) + j, i * (s + 1) + j + 1, (i + 1) * (s + 1) + j, (i + 1) * (s + 1) + j + 1
            tris.append((k00, k01, k11))
            tris.append((k00, k11, k10))
    return weights, np.array(tris, dtype=np.int64)


def _uv(weights: np.ndarray) -> np.ndarray:
    return weights @ _CORNER_UV


@dataclass(frozen=True)
class Nearest:
    """Per query point: distance (inf beyond the cap), quad index (-1 beyond), (u, v) in that quad, error bound."""

    distance: np.ndarray
    quad: np.ndarray
    uv: np.ndarray
    bound: np.ndarray
    cap: float

    @property
    def within_cap(self) -> np.ndarray:
        return self.quad >= 0


class QuadSurface:
    """A union of bilinear quads with exact (bounded) point-to-surface distances up to a cap."""

    def __init__(self, corners: np.ndarray, *, subdivisions: int = 8, tolerance: float = 0.01) -> None:
        corners = np.asarray(corners, dtype=np.float64)
        if corners.ndim != 3 or corners.shape[1:] != (4, 3) or not np.isfinite(corners).all():
            raise Refusal("EVAL_GEOMETRY_INVALID", f"quad corners must be a finite (Q, 4, 3) array, not shape "
                          f"{list(corners.shape)}", shape=list(corners.shape))
        if len(corners) == 0:
            raise Refusal("EVAL_GEOMETRY_EMPTY", "the surface has no valid quad", quads=0)
        if not isinstance(subdivisions, int) or isinstance(subdivisions, bool) or not 1 <= subdivisions <= 32:
            raise Refusal("EVAL_GEOMETRY_PARAMETER", f"subdivisions must be an integer in [1, 32], not "
                          f"{subdivisions!r}", parameter="subdivisions", value=repr(subdivisions))
        if not (isinstance(tolerance, (int, float)) and not isinstance(tolerance, bool) and tolerance > 0):
            raise Refusal("EVAL_GEOMETRY_PARAMETER", f"tolerance must be a positive number, not {tolerance!r}",
                          parameter="tolerance", value=repr(tolerance))
        self.corners = corners
        self.subdivisions = subdivisions
        self.tolerance = float(tolerance)
        self.twist = twist(corners)
        self.diameter = diameter(corners)
        self.max_diameter = float(self.diameter.max())
        self.centre = corners.mean(axis=1)
        self.sphere = np.linalg.norm(corners - self.centre[:, None, :], axis=2).max(axis=1)
        # The subdivision each quad needs for its bound |T| / (4 s²) to meet the tolerance.
        self.steps = np.clip(np.ceil(np.sqrt(self.twist / (4.0 * self.tolerance))), 1, subdivisions).astype(np.int64)
        # Each corner is stored once per quad: duplicates cost memory, not correctness.
        self._tree = cKDTree(corners.reshape(-1, 3))
        self._tables: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.corners)

    def nearest(self, points: np.ndarray, *, cap: float, chunk: int = 4096) -> Nearest:
        """Distance from each point to the surface, exact to within ``bound``, or inf when it exceeds ``cap``."""
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise Refusal("EVAL_GEOMETRY_INVALID", f"query points must be a finite (N, 3) array, not shape "
                          f"{list(points.shape)}", shape=list(points.shape))
        if not (isinstance(cap, (int, float)) and not isinstance(cap, bool) and math.isfinite(cap) and cap > 0):
            raise Refusal("EVAL_GEOMETRY_PARAMETER", f"cap must be a finite positive number, not {cap!r}",
                          parameter="cap", value=repr(cap))
        n = len(points)
        distance = np.full(n, np.inf)
        quad = np.full(n, -1, dtype=np.int64)
        uv = np.full((n, 2), np.nan)
        bound = np.full(n, np.nan)
        d_vertex, _ = self._tree.query(points, k=1, workers=-1)
        # The closest point of a quad lies within that quad's diameter of each of its corners, so a quad that
        # could be within min(d_vertex, cap) has every corner within that radius plus the largest diameter.
        reachable = d_vertex - self.max_diameter <= cap
        for start in range(0, n, chunk):
            idx = np.nonzero(reachable[start:start + chunk])[0] + start
            if len(idx):
                self._nearest_chunk(points, idx, np.minimum(d_vertex[idx], cap), cap, distance, quad, uv, bound)
        return Nearest(distance=distance, quad=quad, uv=uv, bound=bound, cap=float(cap))

    def _table(self, s: int) -> tuple[np.ndarray, np.ndarray]:
        if s not in self._tables:
            self._tables[s] = _subdivision(s)
        return self._tables[s]

    def _nearest_chunk(self, points, idx, reach, cap, distance, quad, uv, bound) -> None:
        radius = reach + self.max_diameter
        hits = self._tree.query_ball_point(points[idx], radius, workers=-1)
        pair_point, pair_quad = [], []
        for k, found in enumerate(hits):
            if found:
                qs = np.unique(np.asarray(found, dtype=np.int64) // 4)
                pair_point.append(np.full(len(qs), k, dtype=np.int64))
                pair_quad.append(qs)
        if not pair_point:
            return
        pp = np.concatenate(pair_point)
        pq = np.concatenate(pair_quad)
        p = points[idx][pp]
        # Spheres: the patch lies in its corners' hull, inside the sphere about their mean; every corner is
        # on the patch. So |p - centre| - sphere bounds a quad's distance below and its nearest corner above.
        lower = np.linalg.norm(p - self.centre[pq], axis=1) - self.sphere[pq]
        nearest_corner = np.linalg.norm(p[:, None, :] - self.corners[pq], axis=2).min(axis=1)
        upper = np.full(len(idx), np.inf)
        np.minimum.at(upper, pp, nearest_corner)
        keep = lower <= np.minimum(upper[pp], cap)
        pp, pq, p = pp[keep], pq[keep], p[keep]
        d_coarse, uv_coarse = self._patch_distance(p, pq, 1)
        err = self.twist[pq] / 4.0
        upper = np.full(len(idx), np.inf)
        np.minimum.at(upper, pp, d_coarse + err)
        keep = d_coarse - err <= np.minimum(upper[pp], cap)
        pp, pq, p, d_best, uv_best = pp[keep], pq[keep], p[keep], d_coarse[keep], uv_coarse[keep]
        steps = self.steps[pq]
        for s in np.unique(steps[steps > 1]):
            sel = steps == s
            d_best[sel], uv_best[sel] = self._patch_distance(p[sel], pq[sel], int(s))
        if not len(pp):
            return
        order = np.lexsort((d_best, pp))
        pp_s, first = np.unique(pp[order], return_index=True)
        best = order[first]
        inside = d_best[best] <= cap
        target = idx[pp_s[inside]]
        chosen = pq[best[inside]]
        distance[target] = d_best[best[inside]]
        quad[target] = chosen
        uv[target] = uv_best[best[inside]]
        bound[target] = self.twist[chosen] / (4.0 * self.steps[chosen] ** 2)

    def _patch_distance(self, p: np.ndarray, pq: np.ndarray, s: int, step: int = 2048) -> tuple[np.ndarray, np.ndarray]:
        weights, tris = self._table(s)
        step = max(16, step * 2 // (s * s))  # about the same memory per block for every s
        out_d = np.empty(len(p))
        out_uv = np.empty((len(p), 2))
        uv_vertices = _uv(weights)
        for start in range(0, len(p), step):
            c = self.corners[pq[start:start + step]]  # (m, 4, 3)
            grid = np.einsum("vk,mkd->mvd", weights, c)  # (m, V, 3)
            a, b, cc = grid[:, tris[:, 0]], grid[:, tris[:, 1]], grid[:, tris[:, 2]]
            d2, bary = closest_on_triangles(p[start:start + step, None, :], a, b, cc)
            t = d2.argmin(axis=1)
            rows = np.arange(len(t))
            out_d[start:start + step] = np.sqrt(d2[rows, t])
            tri = tris[t]
            w = bary[rows, t]
            out_uv[start:start + step] = (w[:, :, None] * uv_vertices[tri]).sum(axis=1)
        return out_d, out_uv

    def evaluate(self, quad: np.ndarray, uv: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Bilinear interpolation of per-corner ``values`` (Q, 4) at (quad, uv); NaN where quad is -1."""
        out = np.full(len(quad), np.nan)
        ok = quad >= 0
        u, v = uv[ok, 0], uv[ok, 1]
        c = values[quad[ok]]
        out[ok] = c[:, 0] * (1 - u) * (1 - v) + c[:, 1] * u * (1 - v) + c[:, 2] * (1 - u) * v + c[:, 3] * u * v
        return out


class Chart:
    """Piecewise-linear interpolation of a payload over quads placed in 2-D chart coordinates.

    ``chart`` (Q, 4, 2) gives each quad's corners in chart coordinates and ``payload`` (Q, 4, P) the values
    there. ``scale`` multiplies the first chart axis before the nearest-quad search, so that quads are roughly
    square to the tree. A query is answered by the first of its ``neighbours`` nearest quads (by centre)
    whose two triangles contain it; the number of queries that hit more than one quad is reported, since that
    means the chart folds over itself there.
    """

    def __init__(self, chart: np.ndarray, payload: np.ndarray, *, scale: float, neighbours: int = 16,
                 tag: np.ndarray | None = None) -> None:
        chart = np.asarray(chart, dtype=np.float64)
        payload = np.asarray(payload, dtype=np.float64)
        if chart.ndim != 3 or chart.shape[1:] != (4, 2) or payload.ndim != 3 or payload.shape[:2] != chart.shape[:2]:
            raise Refusal("EVAL_GEOMETRY_INVALID", f"a chart needs (Q, 4, 2) coordinates and a (Q, 4, P) payload, not "
                          f"{list(chart.shape)} and {list(payload.shape)}", chart_shape=list(chart.shape),
                          payload_shape=list(payload.shape))
        if len(chart) == 0:
            raise Refusal("EVAL_GEOMETRY_EMPTY", "the chart has no valid quad", quads=0)
        if not np.isfinite(chart).all() or not np.isfinite(payload).all():
            raise Refusal("EVAL_GEOMETRY_INVALID", "chart coordinates and payload must be finite",
                          chart_finite=bool(np.isfinite(chart).all()), payload_finite=bool(np.isfinite(payload).all()))
        if not (math.isfinite(scale) and scale > 0) or not (isinstance(neighbours, int) and 1 <= neighbours <= 256):
            raise Refusal("EVAL_GEOMETRY_PARAMETER", f"scale must be finite and positive and neighbours an integer in "
                          f"[1, 256], not {scale!r} and {neighbours!r}", parameter="scale/neighbours",
                          value=repr((scale, neighbours)))
        self.chart = chart
        self.payload = payload
        self.scale = float(scale)
        self.neighbours = min(neighbours, len(chart))
        self.tag = None if tag is None else np.asarray(tag)
        centres = chart.mean(axis=1) * np.array([self.scale, 1.0])
        self._tree = cKDTree(centres)

    def query(self, a: np.ndarray, b: np.ndarray, *, prefer_tag: np.ndarray | None = None,
              chunk: int = 65536) -> tuple[np.ndarray, np.ndarray]:
        """Payload (N, P) at chart points (a, b), NaN where no quad contains the point; and the hit count (N,).

        With several containing quads, one whose ``tag`` equals ``prefer_tag`` for that query wins, else the
        nearest by centre.
        """
        a = np.asarray(a, dtype=np.float64).ravel()
        b = np.asarray(b, dtype=np.float64).ravel()
        n = len(a)
        out = np.full((n, self.payload.shape[2]), np.nan)
        hits = np.zeros(n, dtype=np.int64)
        finite = np.isfinite(a) & np.isfinite(b)
        for s in range(0, n, chunk):
            sl = np.arange(s, min(n, s + chunk))
            sl = sl[finite[sl]]
            if not len(sl):
                continue
            _, nb = self._tree.query(np.stack([a[sl] * self.scale, b[sl]], axis=1), k=self.neighbours, workers=-1)
            nb = nb.reshape(len(sl), -1)
            q = np.stack([a[sl], b[sl]], axis=1)[:, None, :]
            corners = self.chart[nb]  # (m, k, 4, 2)
            inside = np.zeros(nb.shape, dtype=bool)
            weights = np.zeros((*nb.shape, 4))
            for tri in _UNIT_TRIANGLES:
                w = _barycentric_2d(q, corners[:, :, tri[0]], corners[:, :, tri[1]], corners[:, :, tri[2]])
                here = (w >= -1e-9).all(axis=-1) & ~inside  # a point on the shared diagonal takes the first
                for slot, corner in enumerate(tri):
                    weights[:, :, corner] = np.where(here, w[:, :, slot], weights[:, :, corner])
                inside |= here
            hits[sl] = inside.sum(axis=1)
            chosen = np.full(len(sl), -1, dtype=np.int64)
            chosen_w = np.zeros((len(sl), 4))
            for col in range(nb.shape[1]):  # nearest centre first
                take = inside[:, col] & (chosen < 0)
                if prefer_tag is not None and self.tag is not None:
                    want = prefer_tag[sl]
                    take |= (inside[:, col] & (chosen >= 0) & (self.tag[nb[:, col]] == want)
                             & (self.tag[np.maximum(chosen, 0)] != want))
                chosen[take] = nb[take, col]
                chosen_w[take] = weights[take, col]
            ok = chosen >= 0
            out[sl[ok]] = np.einsum("nk,nkp->np", chosen_w[ok], self.payload[chosen[ok]])
        return out, hits


def _barycentric_2d(q: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Barycentric weights (..., 3) of q in 2-D triangles (a, b, c); a degenerate triangle gives NaN."""
    v0, v1, v2 = b - a, c - a, q - a
    d00 = (v0 * v0).sum(-1)
    d01 = (v0 * v1).sum(-1)
    d11 = (v1 * v1).sum(-1)
    d20 = (v2 * v0).sum(-1)
    d21 = (v2 * v1).sum(-1)
    denom = d00 * d11 - d01 * d01
    with np.errstate(divide="ignore", invalid="ignore"):
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
    return np.stack([1 - v - w, v, w], axis=-1)


def cylindrical(zyx: np.ndarray, centre_yx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Radius and angle atan2(dy, dx) of each point about the axis point ``centre_yx`` at its own z."""
    d = zyx[..., 1:] - centre_yx
    return np.hypot(d[..., 0], d[..., 1]), np.arctan2(d[..., 0], d[..., 1])
