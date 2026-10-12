"""A spiral fit as winding-labelled surfaces, and the outer shell as a radius chart.

**Windings.** Each segment is named for its winding range (``...-w003-007`` holds windings 3 to 7). A
vertex's winding coordinate ω comes from that range and its unwrapped angle about the umbilicus:

1. angles φ = atan2(y - u_y(z), x - u_x(z)) are unwrapped along each row, in runs of valid columns split
   where a gap's arc (its column count times the median column step) reaches π × the radius, so that the
   angle cannot have moved by half a turn unseen;
2. runs are tied to one another by whole turns, from the longest run,
   by the median difference over shared columns with an already-tied run in a nearby row (a run that
   shares no column is dropped and counted);
3. the sense (+1 or -1) is the one that makes ω grow outward, read from the radius at the same angle one
   turn apart (an off-centre ring cannot flip it); every segment must agree;
4. ω = sense · φ_unwrapped / 2π + c_s + δ, where the common phase δ is the circular mean, over segments,
   of the fraction that puts each segment's midrange (0.1st to 99.9th percentile) on its range's middle,
   and c_s is the whole number of turns left over. Each segment's residual and span are reported.

So ω's integer part is the fit's winding index, and ω's fraction is fixed by the angle alone, which lets a
ray at angle φ meet the fit at ω = k + frac(sense · φ / 2π + δ) for whole k.

**Charts.** Quads are placed at (ω, z); the point of the fit at (ω, z) is the chart's interpolation of the
corners. Adjacent windings, the ±½ shifts, the local spacing and K-dup all read the fit through it.

**Radius.** The outer shell is charted at (angle, z) with its radius about the umbilicus as the payload; a
query looks at the angle and at the angle ± 2π, and takes the largest radius when the shell folds.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from apok_eval.eval.geometry import Chart, QuadSurface, cylindrical, diameter, quads_from_grid
from apok_eval.refusal import Refusal, register_codes

register_codes(
    "apok_eval.eval.fit",
    {
        "EVAL_SEGMENT_NAME": "a fit segment's name does not end in a winding range -wLLL-HHH with LLL <= HHH",
        "EVAL_SEGMENT_RANGES": "two segments of one fit claim the same winding",
        "EVAL_FIT_SENSE": "segments of one fit disagree on the spiral's sense, or a segment shows none",
        "EVAL_FIT_EMPTY": "no segment of the fit has a valid quad near the slab",
        "EVAL_FIT_PARAMETER": "a fit parameter (slab, margin, shift direction) is out of range",
    },
)

TWO_PI = 2.0 * math.pi
#: A run is tied to tied runs in rows up to this far away.
TIE_ROWS = 3
_RANGE = re.compile(r"-w(\d{3})-(\d{3})\Z")


def winding_range(name: str) -> tuple[int, int]:
    """(first, last) winding from a segment name such as ``myfit-w003-007``."""
    m = _RANGE.search(name)
    if not m or int(m.group(1)) > int(m.group(2)):
        raise Refusal("EVAL_SEGMENT_NAME", f"segment {name!r} does not end in -wLLL-HHH with LLL <= HHH",
                      segment=name)
    return int(m.group(1)), int(m.group(2))


@dataclass(frozen=True, eq=False)
class SegmentGrid:
    name: str
    zyx: np.ndarray  # (H, W, 3) spiral-frame coordinates
    valid: np.ndarray  # (H, W) under the declared validity policy


@dataclass(eq=False)
class _Part:
    name: str
    index: int
    lo: int
    hi: int
    row0: int
    zyx: np.ndarray
    valid: np.ndarray
    r: np.ndarray
    phi: np.ndarray
    centre: np.ndarray
    turns: np.ndarray  # unwrapped, tied angle in turns (sign as measured)
    omega: np.ndarray | None = None


class WindingFit:
    """The fit's segments near a slab, with ω per vertex, a distance surface and an (ω, z) chart."""

    def __init__(self, segments: list[SegmentGrid], umbilicus, *, z_range: tuple[float, float], margin: float = 128.0,
                 max_quad_diameter: float = 80.0, subdivisions: int = 8, neighbours: int = 8) -> None:
        z0, z1 = z_range
        if not (math.isfinite(z0) and math.isfinite(z1) and z0 < z1 and margin >= 0 and max_quad_diameter > 0):
            raise Refusal("EVAL_FIT_PARAMETER", f"z_range must be finite and increasing and margin, diameter limit "
                          f"positive, not {z_range}, {margin}, {max_quad_diameter}", z_range=list(z_range),
                          margin=margin, max_quad_diameter=max_quad_diameter)
        self.z_range = (float(z0), float(z1))
        self.umbilicus = umbilicus
        self.report: dict = {"segments": [], "margin": margin, "max_quad_diameter": max_quad_diameter,
                             "tie_rows": TIE_ROWS}
        ranges = sorted((winding_range(s.name) + (s.name,) for s in segments))
        for (lo_a, hi_a, a), (lo_b, hi_b, b) in zip(ranges, ranges[1:]):
            if lo_b <= hi_a:
                raise Refusal("EVAL_SEGMENT_RANGES", f"segments {a} and {b} overlap in windings "
                              f"{lo_b}-{min(hi_a, hi_b)}", segments=[a, b])
        self.parts = [p for p in (self._crop(i, s, z0 - margin, z1 + margin) for i, s in enumerate(segments)) if p]
        if not self.parts:
            raise Refusal("EVAL_FIT_EMPTY", f"no segment has a valid vertex within z {z0 - margin}..{z1 + margin}",
                          segments=[s.name for s in segments])
        self.sense = self._sense()
        self._label()
        self._build(max_quad_diameter, subdivisions, neighbours)

    # -- construction ---------------------------------------------------------------------------------

    def _crop(self, index: int, seg: SegmentGrid, lo_z: float, hi_z: float) -> _Part | None:
        lo, hi = winding_range(seg.name)
        zyx = np.asarray(seg.zyx, dtype=np.float64)
        valid = np.asarray(seg.valid, dtype=bool) & np.isfinite(zyx).all(axis=2)
        near = valid & (zyx[..., 0] >= lo_z) & (zyx[..., 0] < hi_z)
        rows = np.nonzero(near.any(axis=1))[0]
        if not len(rows):
            self.report["segments"].append({"name": seg.name, "windings": [lo, hi], "rows": 0})
            return None
        r0, r1 = int(rows[0]), int(rows[-1]) + 1
        zyx, valid = zyx[r0:r1], valid[r0:r1].copy()
        query = self.umbilicus.yx_at(zyx[..., 0].ravel(), out_of_range="extrapolate")
        centre = query.yx.reshape(*zyx.shape[:2], 2)
        r, phi = cylindrical(zyx, centre)
        valid &= r > 0
        part = _Part(seg.name, index, lo, hi, r0, zyx, valid, r, phi, centre, np.full(valid.shape, np.nan))
        part.umbilicus_outside = int(query.outside[valid.ravel()].sum())  # type: ignore[attr-defined]
        self._tie_runs(part)
        return part

    @staticmethod
    def _runs(cols: np.ndarray, r_row: np.ndarray, step: float) -> list[np.ndarray]:
        """Split a row's valid columns where a gap could hide half a turn: arc ≥ π × the smaller radius."""
        if not len(cols):
            return []
        arc = np.diff(cols) * step
        breaks = np.nonzero(arc >= np.pi * np.minimum(r_row[cols[:-1]], r_row[cols[1:]]))[0] + 1
        return np.split(cols, breaks)

    def _tie_runs(self, part: _Part) -> None:
        runs = []  # (row, cols, unwrapped turns)
        adjacent = part.valid[:, :-1] & part.valid[:, 1:]
        steps = np.linalg.norm(np.diff(part.zyx, axis=1), axis=2)[adjacent]
        step = float(np.median(steps)) if len(steps) else np.inf
        for i in range(part.valid.shape[0]):
            for cols in self._runs(np.nonzero(part.valid[i])[0], part.r[i], step):
                runs.append((i, cols, np.unwrap(part.phi[i, cols]) / TWO_PI))
        dropped = 0
        if runs:
            ref = max(range(len(runs)), key=lambda k: (len(runs[k][1]), -runs[k][0]))
            i, cols, t = runs[ref]
            part.turns[i, cols] = t
            pending = [k for k in range(len(runs)) if k != ref]
            progress = True
            while pending and progress:
                progress = False
                rest = []
                for k in sorted(pending, key=lambda k: abs(runs[k][0] - runs[ref][0])):
                    i, cols, t = runs[k]
                    offset = self._offset(part, i, cols, t)
                    if offset is None:
                        rest.append(k)
                        continue
                    part.turns[i, cols] = t + offset
                    progress = True
                pending = rest
            for k in pending:
                part.valid[runs[k][0], runs[k][1]] = False
                dropped += len(runs[k][1])
        part.dropped_vertices = dropped  # type: ignore[attr-defined]

    @staticmethod
    def _offset(part: _Part, i: int, cols: np.ndarray, t: np.ndarray) -> float | None:
        h = part.turns.shape[0]
        for d in range(1, TIE_ROWS + 1):  # a run never shares a column with its own row's other runs
            for j in (i - d, i + d):
                if 0 <= j < h:
                    other = part.turns[j, cols]
                    shared = np.isfinite(other)
                    if shared.any():
                        return float(np.round(np.median(other[shared] - t[shared])))
        return None

    def _sense(self) -> int:
        """+1 when the radius grows with the unwrapped angle, -1 when it shrinks; one answer for every segment.

        Read at the same angle one turn apart: r(t + 1) - r(t) along each row. A least-squares slope of r on t
        is not used: when rings sit off the umbilicus, the offset's sine term can outweigh the spiral's pitch over
        a few turns, and a slope can then read the opposite sense. The same-angle difference cancels the offset. A segment with no row
        that makes a full turn shows no sense and abstains (``None`` in the report's ``sense_votes``); a fit in
        which every segment abstains is refused.
        """
        votes: dict[str, int | None] = {}
        for p in self.parts:
            steps = []
            for i in range(p.turns.shape[0]):
                ok = np.isfinite(p.turns[i]) & p.valid[i]
                t, r = p.turns[i, ok], p.r[i, ok]
                if len(t) < 2:
                    continue
                order = np.argsort(t)
                t, r = t[order], r[order]
                reach = t + 1.0 <= t[-1] + 1e-6  # a row of exactly one turn reaches its own end
                if reach.any():
                    steps.append(np.interp(t[reach] + 1.0, t, r) - r[reach])
            if not steps:
                votes[p.name] = None
                continue
            step = float(np.median(np.concatenate(steps)))
            votes[p.name] = 1 if step > 0 else -1 if step < 0 else 0
        self.report["sense_votes"] = votes
        senses = {v for v in votes.values() if v is not None}
        if not senses:
            raise Refusal("EVAL_FIT_SENSE", "no segment has a row that makes a full turn, so the fit shows no sense",
                          senses=votes)
        if len(senses) != 1 or 0 in senses:
            raise Refusal("EVAL_FIT_SENSE", f"segments disagree on the spiral's sense: {votes}", senses=votes)
        return senses.pop()

    def _label(self) -> None:
        a = {}
        for p in self.parts:
            ok = np.isfinite(p.turns) & p.valid
            raw = self.sense * p.turns[ok]
            lo_q, hi_q = np.percentile(raw, [0.1, 99.9])
            a[p.name] = (p.lo + p.hi + 1) / 2.0 - (lo_q + hi_q) / 2.0
            p.span = float(hi_q - lo_q)  # type: ignore[attr-defined]
        z = np.mean([np.exp(2j * math.pi * v) for v in a.values()])
        self.phase = float((np.angle(z) / TWO_PI) % 1.0)
        for p in self.parts:
            whole = round(a[p.name] - self.phase)
            p.omega = np.where(p.valid & np.isfinite(p.turns), self.sense * p.turns + whole + self.phase, np.nan)
            self.report["segments"].append({
                "name": p.name, "windings": [p.lo, p.hi], "rows": [p.row0, p.row0 + p.valid.shape[0]],
                "valid_vertices": int(np.isfinite(p.omega).sum()), "dropped_vertices": p.dropped_vertices,
                "span_turns": p.span, "expected_turns": p.hi - p.lo + 1,
                "residual_turns": float(a[p.name] - self.phase - whole),
                "umbilicus_outside": p.umbilicus_outside,
            })
        self.report.update(sense=self.sense, phase=self.phase)

    def _build(self, max_quad_diameter: float, subdivisions: int, neighbours: int) -> None:
        corners, omegas, tags = [], [], []
        stretched = 0
        for p in self.parts:
            ok = np.isfinite(p.omega)
            c, ij = quads_from_grid(p.zyx, ok)
            if not len(c):
                continue
            w = np.stack([p.omega[ij[:, 0] + di, ij[:, 1] + dj] for di, dj in ((0, 0), (0, 1), (1, 0), (1, 1))], axis=1)
            keep = diameter(c) <= max_quad_diameter
            stretched += int((~keep).sum())
            corners.append(c[keep])
            omegas.append(w[keep])
            tags.append(np.full(int(keep.sum()), p.index))
        if not corners or not sum(len(c) for c in corners):
            raise Refusal("EVAL_FIT_EMPTY", "no valid quad near the slab", segments=[p.name for p in self.parts])
        self.corners = np.concatenate(corners)
        self.omega_corners = np.concatenate(omegas)
        self.tags = np.concatenate(tags)
        self.surface = QuadSurface(self.corners, subdivisions=subdivisions)
        dz = np.median(np.abs(self.corners[:, 2, 0] - self.corners[:, 0, 0]))
        dw = np.median(np.abs(self.omega_corners[:, 1] - self.omega_corners[:, 0]))
        self.chart_scale = float(dz / dw) if dw > 0 and dz > 0 else 1.0
        chart = np.stack([self.omega_corners, self.corners[..., 0]], axis=-1)
        self.chart = Chart(chart, self.corners, scale=self.chart_scale, neighbours=neighbours, tag=self.tags)
        self._ranges = sorted((p.lo, p.hi, p.index) for p in self.parts)
        self.report.update(quads=int(len(self.corners)), stretched_quads=stretched, chart_scale=self.chart_scale,
                           max_twist=float(self.surface.twist.max()))

    # -- queries --------------------------------------------------------------------------------------

    @property
    def windings(self) -> tuple[int, int]:
        """The lowest and highest winding labels of the fit's segments."""
        return self._ranges[0][0], self._ranges[-1][1]

    def phase_at(self, phi: np.ndarray) -> np.ndarray:
        """Fraction of ω at which a ray at angle ``phi`` meets the fit."""
        return np.mod(self.sense * np.asarray(phi) / TWO_PI + self.phase, 1.0)

    def _owner(self, omega: np.ndarray) -> np.ndarray:
        owner = np.full(len(omega), -1)
        k = np.floor(omega)
        for lo, hi, index in self._ranges:
            owner[(k >= lo) & (k <= hi)] = index
        return owner

    def point_at(self, omega: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The fit's (z, y, x) at chart point (ω, z), NaN outside it; and the number of quads that hold it."""
        omega = np.asarray(omega, dtype=np.float64).ravel()
        return self.chart.query(omega, np.asarray(z, dtype=np.float64).ravel(), prefer_tag=self._owner(omega))

    def radius_at(self, omega: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Radius about the umbilicus of the fit at (ω, z); NaN where the fit has no surface there."""
        zyx, _ = self.point_at(omega, z)
        centre = self.umbilicus.yx_at(np.nan_to_num(zyx[:, 0]), out_of_range="extrapolate").yx
        r, _ = cylindrical(zyx, centre)
        return r

    def omega_at(self, quad: np.ndarray, uv: np.ndarray) -> np.ndarray:
        return self.surface.evaluate(quad, uv, self.omega_corners)

    def ray_spacing(self, points: np.ndarray, omega_hint: np.ndarray, *, window: int = 3) -> dict[str, np.ndarray]:
        """Local winding spacing at each point: the gap between the two fit sheets that bracket it on its ray.

        The ray leaves the umbilicus at the point's z through the point. The fit meets it at ω = k + phase;
        the crossings are taken for whole k within ``window`` of the hint (the closest surface point's ω) or,
        without a hint or when the window does not bracket the point, over every winding of the fit. A point
        inside the innermost or outside the outermost crossing takes the nearest gap. Fewer than two
        crossings leave the spacing unknown (NaN).
        """
        points = np.asarray(points, dtype=np.float64)
        n = len(points)
        centre = self.umbilicus.yx_at(points[:, 0], out_of_range="extrapolate").yx
        r_p, phi_p = cylindrical(points, centre)
        f = self.phase_at(phi_p)
        spacing = np.full(n, np.nan)
        edge = np.zeros(n, dtype=bool)
        hint = np.asarray(omega_hint, dtype=np.float64)
        lo_all, hi_all = self._ranges[0][0], self._ranges[-1][1]
        done = np.zeros(n, dtype=bool)
        for stage in (0, 1):  # the window around the hint, then every winding for what is left
            idx = np.nonzero(np.isfinite(hint) & ~done if stage == 0 else ~done)[0]
            if not len(idx):
                continue
            if stage == 0:
                ks = np.floor(hint[idx] - f[idx]).astype(np.int64)[:, None] + np.arange(-window, window + 1)[None, :]
            else:
                ks = np.broadcast_to(np.arange(lo_all - 1, hi_all + 2)[None, :], (len(idx), hi_all - lo_all + 3))
            omega = (ks + f[idx, None]).ravel()
            radii = self.radius_at(omega, np.repeat(points[idx, 0], ks.shape[1])).reshape(ks.shape)
            s, e, ok = _bracket(radii, r_p[idx], need_inside=(stage == 0))
            spacing[idx[ok]] = s[ok]
            edge[idx[ok]] = e[ok]
            done[idx[ok]] = True
        return {"spacing": spacing, "edge": edge, "radius": r_p, "phi": phi_p}

    def adjacent_radii(self, offsets: tuple[int, ...]) -> list[dict]:
        """Per part: the radius of the fit at (ω + k, z) of every vertex, for each k in ``offsets`` (cached)."""
        cache = self.__dict__.setdefault("_adjacent", {})
        out = []
        for p in self.parts:
            ok = np.isfinite(p.omega)
            om, zz = p.omega[ok], p.zyx[..., 0][ok]
            entry = {"mask": ok}
            for k in offsets:
                if (p.index, k) not in cache:
                    cache[(p.index, k)] = self.radius_at(om + k, zz)
                entry[k] = cache[(p.index, k)]
            out.append(entry)
        return out

    def shifted(self, direction: int, *, subdivisions: int | None = None) -> tuple[QuadSurface | None, dict]:
        """The fit with every vertex moved along its umbilicus ray by half the gap to the adjacent winding.

        ``direction`` +1 moves toward winding ω + 1, -1 toward ω - 1. A vertex without that neighbour uses
        the gap on its other side; a vertex with neither is dropped. Quads are re-formed from the moved grids
        under the same diameter limit. When no quad is left the surface is None and ``counts["empty"]`` says
        so: every point is then off that copy.
        """
        if direction not in (1, -1):
            raise Refusal("EVAL_FIT_PARAMETER", f"shift direction must be +1 or -1, not {direction!r}",
                          direction=repr(direction))
        radii = self.adjacent_radii((1, -1))
        corners = []
        counts = {"vertices": 0, "fallback": 0, "dropped": 0, "negative_gap": 0, "empty": False}
        for p, rr in zip(self.parts, radii):
            ok = rr["mask"]
            r = p.r[ok]
            out_gap, in_gap = rr[1] - r, r - rr[-1]
            want, other = (out_gap, in_gap) if direction == 1 else (in_gap, out_gap)
            gap = np.where(np.isfinite(want), want, other)
            counts["vertices"] += int(ok.sum())
            counts["fallback"] += int((~np.isfinite(want) & np.isfinite(other)).sum())
            counts["negative_gap"] += int((gap < 0).sum())
            dr = direction * 0.5 * gap
            moved = p.zyx.copy()
            unit = (p.zyx[..., 1:][ok] - p.centre[ok]) / r[:, None]
            moved_yx = p.centre[ok] + (r + dr)[:, None] * unit
            keep = np.isfinite(dr)
            counts["dropped"] += int((~keep).sum())
            sub = moved[ok]
            sub[:, 1:] = moved_yx
            moved[ok] = sub
            valid = ok.copy()
            valid[ok] = keep
            c, _ = quads_from_grid(moved, valid)
            corners.append(c[diameter(c) <= self.report["max_quad_diameter"]])
        allc = np.concatenate(corners)
        if not len(allc):
            counts["empty"] = True
            return None, counts
        return QuadSurface(allc, subdivisions=subdivisions or self.surface.subdivisions), counts


def _bracket(radii: np.ndarray, r_p: np.ndarray, *, need_inside: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gap between the sorted crossings that bracket each r_p; ``edge`` marks a nearest-gap answer."""
    s = np.sort(np.where(np.isfinite(radii), radii, np.inf), axis=1)
    count = np.isfinite(s).sum(axis=1)
    j = (s <= r_p[:, None]).sum(axis=1)  # crossings at or inside the point
    s = np.where(np.isfinite(s), s, np.nan)
    rows = np.arange(len(r_p))

    def gap(k: np.ndarray) -> np.ndarray:
        k = np.clip(k, 1, s.shape[1] - 1)
        return s[rows, k] - s[rows, k - 1]

    usable = count >= 2
    inside = usable & (j > 0) & (j < count)
    edge = usable & ~inside & (not need_inside)
    outer_edge = np.where(edge, gap(count - 1), np.nan)
    spacing = np.where(inside, gap(j), np.where(edge & (j == 0), gap(np.ones_like(j)), outer_edge))
    return spacing, edge, inside | edge


class OuterShell:
    """Radius R(z, angle) of the outer shell about the umbilicus, from its tifxyz grid."""

    def __init__(self, zyx: np.ndarray, valid: np.ndarray, umbilicus, *, neighbours: int = 8) -> None:
        self.umbilicus = umbilicus
        corners, _ = quads_from_grid(zyx, valid)
        if not len(corners):
            raise Refusal("EVAL_FIT_EMPTY", "the outer shell has no valid quad", surface="outer_shell")
        centre = umbilicus.yx_at(corners[..., 0].ravel(), out_of_range="extrapolate").yx.reshape(*corners.shape[:2], 2)
        r, phi = cylindrical(corners, centre)
        phi = phi[:, :1] + np.angle(np.exp(1j * (phi - phi[:, :1])))  # unwrap each quad about its first corner
        wide = np.ptp(phi, axis=1) > math.pi / 2  # a quad around the axis has no single angle
        self.excluded_wide = int(wide.sum())
        chart = np.stack([phi[~wide], corners[~wide][..., 0]], axis=-1)
        dz = np.median(np.abs(chart[:, 2, 1] - chart[:, 0, 1]))
        dphi = np.median(np.abs(chart[:, 1, 0] - chart[:, 0, 0]))
        self.chart = Chart(chart, r[~wide][..., None], scale=float(dz / dphi) if dphi > 0 and dz > 0 else 1.0,
                           neighbours=neighbours)

    def radius(self, z: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """R at each (z, angle), NaN where the shell has no quad; and how many quads answered (folds > 1)."""
        z = np.asarray(z, dtype=np.float64).ravel()
        phi = np.asarray(phi, dtype=np.float64).ravel()
        best = np.full(len(z), np.nan)
        hits = np.zeros(len(z), dtype=np.int64)
        for turn in (0.0, TWO_PI, -TWO_PI):
            r, h = self.chart.query(phi + turn, z)
            best = np.fmax(best, r[:, 0])
            hits += h
        return best, hits


def segment_from_tifxyz(path, *, policy: str) -> SegmentGrid:
    """A fit segment read from a tifxyz directory, valid under the declared validity ``policy``."""
    from apok_eval.io.tifxyz import read_tifxyz, validity

    surface = read_tifxyz(path)
    return SegmentGrid(name=surface.dir_name, zyx=surface.zyx.astype(np.float64), valid=validity(surface, policy).valid)
