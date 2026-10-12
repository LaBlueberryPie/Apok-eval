"""Where fibre points sit within the fit's sheet gaps: the registration diagnostic behind E7 v3.

A distance tolerance around each sheet that is wide enough to forgive tracing noise covers most of every gap,
so a copy of the fit shifted by half a gap still catches most points. This module asks the narrower question:
do the fibre points sit on the fit's sheets at all, and more so than when the fibres are moved? Everything
below is fixed before a run scores anything.

- **Gap position.** A point's ray leaves the umbilicus at its z through the point and meets the fit's sheets
  at ω = k + f(φ). The two adjacent crossings that bracket the point give u ∈ [0, 1) (0 on the inner sheet) and
  the signed position s = u − round(u) ∈ [−½, ½), 0 on a sheet, in units of the local spacing. Crossings are
  read from ``RadiusTable``, the fit's radius on a regular (ω, z) grid interpolated bilinearly. A point
  without two adjacent crossings (inside the innermost sheet, outside the outermost, or beside a hole) is
  unknown and counted.
- **Exact check.** For the observed points the two bracketing crossings are also computed exactly
  (``WindingFit.radius_at``). If the median circular |s_table − s_exact| exceeds ``table_tolerance`` every
  verdict is INCONCLUSIVE.
- **Points.** Each fibre's in-slab points, thinned to at most ``points_per_fibre`` evenly spaced along the
  trace; the same points are moved for the nulls. A fibre counts in a pass when at least ``min_points`` of its
  points have a known gap position there.
- **Per fibre:** Q_f, the fraction of its points with |s| < ``band`` (1/8: a band a quarter of the spacing
  wide; 0.25 if points fall uniformly in the gap); R_f and c_f, the length and angle / 2π of the mean of
  exp(2πiη) over its points, with η = ω − (sense · θ / 2π + phase), θ unwrapped along the fibre (η keeps one
  value along a fibre that keeps to one sheet, and its integer part names the sheet); the majority sheet n* =
  the most common round(η), its share, and Q*_f, the fraction with |η − n*| < ``band``.
- **Groups:** all fibres; by how each fibre's coordinate scale was set (a declared domain or the 0.25
  default); by radial tercile of the median r / R (reported only). A group's Q and R are means over fibres.
- **Nulls.** Each fibre is moved as a whole and its points re-read. The z-offset null: ``null_draws`` draws,
  each fibre moved by ±U(``z_offset``) voxels in z, its sign flipped when the drawn one keeps fewer than half
  of its points in the slab; points leaving the slab are dropped. Smaller offsets (``bands``,
  ``band_draws`` each) show how fast agreement decays with z. The rotation null: ``rotation_draws`` draws,
  each fibre turned about the umbilicus by U(``rotation``) radians. Numpy PCG64 seeded ``seed``, the draws in
  that order.
- **Verdict per group** (all fibres and each scale group): the null in use is the z-offset null, unless its
  median Q is at least ``uninformative_null_q`` (moving in z did not move the fibres off the sheets), when it is
  the rotation null. REGISTERED: Q ≥ ``registered_q`` and above every draw of the null in use. NOT
  REGISTERED: Q < ``not_registered_q``, or Q not above that null's 95th percentile. Otherwise INCONCLUSIVE.
  NOT MEASURED under ``min_fibres`` fibres. OFFSET-CONSISTENT is reported beside a verdict that is not
  REGISTERED when R exceeds every draw of the null in use and its median by ``offset_margin``: the fibres keep
  a steady place in the gap, but not on the sheets.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from apok_eval.eval.fibres import Fibre
from apok_eval.eval.fit import TWO_PI, OuterShell, WindingFit
from apok_eval.eval.geometry import cylindrical
from apok_eval.refusal import Refusal, register_codes

register_codes(
    "apok_eval.eval.registration",
    {
        "EVAL_REG_PARAMETER": "a registration-diagnostic setting is out of range",
    },
)

TERCILES = ("inner", "middle", "outer")
VERDICTS = ("REGISTERED", "NOT REGISTERED", "INCONCLUSIVE", "NOT MEASURED")
HISTOGRAM_BINS = 20


@dataclass(frozen=True)
class RegConfig:
    z_range: tuple[float, float]
    band: float = 0.125
    points_per_fibre: int = 256
    min_points: int = 20
    min_fibres: int = 10
    z_step: float = 16.0
    omega_bins: int = 1024
    null_draws: int = 100
    z_offset: tuple[float, float] = (128.0, 384.0)
    bands: tuple[tuple[float, float], ...] = ((8.0, 16.0), (32.0, 64.0))
    band_draws: int = 20
    rotation: tuple[float, float] = (math.pi / 2, 3 * math.pi / 2)
    rotation_draws: int = 100
    seed: int = 1
    registered_q: float = 0.40
    not_registered_q: float = 0.30
    uninformative_null_q: float = 0.35
    offset_margin: float = 0.10
    table_tolerance: float = 0.05

    def check(self) -> None:
        z0, z1 = self.z_range
        problems = []

        def need(ok: bool, what: str) -> None:
            if not ok:
                problems.append(what)

        need(math.isfinite(z0) and math.isfinite(z1) and z0 < z1, "z_range must be finite and increasing")
        need(0 < self.band < 0.5, "band must lie in (0, 0.5)")
        need(self.points_per_fibre >= 2 and self.min_points >= 2, "points_per_fibre and min_points must be >= 2")
        need(self.min_fibres >= 1, "min_fibres must be >= 1")
        need(0 < self.z_step <= z1 - z0, "z_step must be positive and fit in the slab")
        need(self.omega_bins >= 8, "omega_bins must be >= 8")
        need(min(self.null_draws, self.band_draws, self.rotation_draws) >= 20, "every null needs >= 20 draws")
        for lo, hi in (self.z_offset, *self.bands):
            need(0 <= lo < hi, f"z offsets ({lo}, {hi}) must satisfy 0 <= lo < hi")
        need(0 <= self.rotation[0] < self.rotation[1] <= TWO_PI, "rotation must lie within [0, 2π] and increase")
        need(0 < self.not_registered_q <= self.registered_q < 1, "need 0 < not_registered_q <= registered_q < 1")
        need(0 < self.uninformative_null_q < 1 and self.offset_margin > 0 and self.table_tolerance > 0,
             "uninformative_null_q in (0, 1), offset_margin and table_tolerance positive")
        if problems:
            raise Refusal("EVAL_REG_PARAMETER", f"registration configuration is out of range: {problems}",
                          problems=problems)


class RadiusTable:
    """The fit's radius about the umbilicus on a regular grid of ω (``omega_bins`` per winding) and z."""

    def __init__(self, fit: WindingFit, z_range: tuple[float, float], *, z_step: float, omega_bins: int,
                 chunk: int = 1 << 18) -> None:
        lo, hi = fit.windings
        self.fit = fit
        self.lo, self.hi, self.bins = lo, hi, int(omega_bins)
        self.z = np.arange(z_range[0], z_range[1] + z_step / 2, z_step)
        omega = lo + np.arange((hi + 1 - lo) * self.bins + 1) / self.bins
        ww, zz = np.meshgrid(omega, self.z, indexing="ij")
        ww, zz = ww.ravel(), zz.ravel()
        radius = np.empty(len(ww))
        for s in range(0, len(ww), chunk):
            radius[s:s + chunk] = fit.radius_at(ww[s:s + chunk], zz[s:s + chunk])
        self.radius = radius.reshape(len(omega), len(self.z)).astype(np.float32)
        count = hi + 1 - lo
        blocks = self.radius[:count * self.bins].reshape(count, self.bins, len(self.z))
        self._search = np.fmax.accumulate(np.where(np.isnan(blocks), -np.inf, blocks), axis=0)

    @property
    def windings(self) -> np.ndarray:
        return np.arange(self.lo, self.hi + 1)

    def _corners(self, f: np.ndarray, z: np.ndarray):
        a = np.clip(np.asarray(f, dtype=np.float64) * self.bins, 0, self.bins - 1e-9)
        b = np.clip((np.asarray(z, dtype=np.float64) - self.z[0]) / (self.z[1] - self.z[0]), 0, len(self.z) - 1 - 1e-9)
        i0, j0 = np.floor(a).astype(np.int64), np.floor(b).astype(np.int64)
        return i0, a - i0, j0, b - j0, np.minimum(j0 + 1, len(self.z) - 1)

    def values(self, f: np.ndarray, z: np.ndarray, ks: np.ndarray) -> np.ndarray:
        """(n, m): the radius of sheets ``ks`` (indices from the lowest winding) on each point's ray, ω = k + f,
        interpolated bilinearly; NaN where any grid corner is a hole."""
        i0, ta, j0, tb, j1 = self._corners(f, z)
        rows = i0[:, None] + ks * self.bins
        out = np.zeros(ks.shape)
        for dr, wa in ((0, 1 - ta), (1, ta)):
            for jj, wb in ((j0, 1 - tb), (j1, tb)):
                out += (wa * wb)[:, None] * self.radius[rows + dr, jj[:, None]]
        return out

    def sheets(self, f: np.ndarray, z: np.ndarray) -> np.ndarray:
        """(n, K): every sheet's radius on each point's ray."""
        n = len(np.asarray(f))
        return self.values(f, z, np.broadcast_to(np.arange(self.hi + 1 - self.lo), (n, self.hi + 1 - self.lo)))

    def bracket(self, f: np.ndarray, z: np.ndarray, r: np.ndarray, *, window: int = 3):
        """Index (from the lowest winding) of the sheet just inside each point and the two bracketing radii.

        A binary search over the nearest grid node's running maximum gives a hint; the sheets from hint − 2 to
        hint + 3 are interpolated and the outermost adjacent pair with r_in ≤ r < r_out is taken. Points with
        no such pair in that window are bracketed over every sheet.
        """
        f, z, r = (np.asarray(x, dtype=np.float64) for x in (f, z, r))
        n, count = len(r), self.hi + 1 - self.lo
        i0, _, j0, tb, _ = self._corners(f, z)
        jz = np.where(tb < 0.5, j0, np.minimum(j0 + 1, len(self.z) - 1))
        lo, hi = np.zeros(n, np.int64), np.full(n, count, np.int64)
        while (lo < hi).any():
            mid = (lo + hi) // 2
            go = (lo < hi) & (self._search[np.minimum(mid, count - 1), i0, jz] <= r)
            lo, hi = np.where(go, mid + 1, lo), np.where(go | (lo >= hi), hi, mid)
        ks = np.clip(lo[:, None] - 1 + np.arange(1 - window, window + 1)[None, :], 0, count - 1)
        k, r_in, r_out, ok = _pair(ks, self.values(f, z, ks), r)
        if (~ok).any():
            miss = np.nonzero(~ok)[0]
            all_k = np.broadcast_to(np.arange(count), (len(miss), count))
            k[miss], r_in[miss], r_out[miss], ok[miss] = _pair(all_k, self.values(f[miss], z[miss], all_k), r[miss])
        return k, r_in, r_out, ok


def _pair(ks: np.ndarray, v: np.ndarray, r: np.ndarray):
    """The outermost adjacent pair of sheets (k, k + 1) in each row of ``ks`` with v_k ≤ r < v_k+1."""
    good = np.isfinite(v[:, :-1]) & np.isfinite(v[:, 1:]) & (v[:, :-1] <= r[:, None]) & (r[:, None] < v[:, 1:])
    good &= ks[:, 1:] == ks[:, :-1] + 1
    ok = good.any(axis=1)
    j = good.shape[1] - 1 - np.argmax(good[:, ::-1], axis=1)
    rows = np.arange(len(r))
    k = np.where(ok, ks[rows, j], -1)
    return k, np.where(ok, v[rows, j], np.nan), np.where(ok, v[rows, j + 1], np.nan), ok


def gap_positions(table: RadiusTable, points: np.ndarray) -> dict[str, np.ndarray]:
    """Gap position of each (z, y, x) point on its umbilicus ray, from the table; NaN where unknown."""
    fit = table.fit
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    centre = fit.umbilicus.yx_at(points[:, 0], out_of_range="extrapolate").yx
    r, phi = cylindrical(points, centre)
    f = fit.phase_at(phi)
    j, r_in, r_out, ok = table.bracket(f, points[:, 0], r)
    u = np.where(ok, (r - r_in) / np.where(ok, r_out - r_in, 1.0), np.nan)
    k = np.where(ok, table.lo + j, np.nan)
    return {"u": u, "s": u - np.round(u), "omega": k + f + u, "k": k, "f": f, "r": r, "phi": phi,
            "spacing": np.where(ok, r_out - r_in, np.nan)}


def exact_positions(fit: WindingFit, points: np.ndarray, k: np.ndarray, f: np.ndarray, r: np.ndarray) -> np.ndarray:
    """s from the exact crossings of sheets k and k + 1 (NaN where they do not bracket the point)."""
    ok = np.isfinite(k)
    s = np.full(len(r), np.nan)
    if not ok.any():
        return s
    z = points[ok, 0]
    inner = fit.radius_at(k[ok] + f[ok], z)
    outer = fit.radius_at(k[ok] + 1 + f[ok], z)
    good = np.isfinite(inner) & np.isfinite(outer) & (inner <= r[ok]) & (r[ok] < outer)
    u = np.where(good, (r[ok] - inner) / np.where(good, outer - inner, 1.0), np.nan)
    s[ok] = u - np.round(u)
    return s


def circular_gap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|a − b| on the unit circle of gap positions (in [0, ½])."""
    d = np.mod(np.asarray(a) - np.asarray(b), 1.0)
    return np.minimum(d, 1.0 - d)


def fibre_stats(s: np.ndarray, eta: np.ndarray, band: float, min_points: int) -> dict:
    ok = np.isfinite(s) & np.isfinite(eta)
    n = int(ok.sum())
    if n < min_points:
        return {"points_used": n, "Q": None, "R": None, "c": None, "majority_share": None, "majority_Q": None}
    s, eta = s[ok], eta[ok]
    z = np.mean(np.exp(2j * np.pi * eta))
    sheet = np.round(eta)
    values, counts = np.unique(sheet, return_counts=True)
    major = values[np.argmax(counts)]
    return {"points_used": n, "Q": float(np.mean(np.abs(s) < band)), "R": float(abs(z)),
            "c": float(np.angle(z) / TWO_PI), "majority_share": float(counts.max() / n),
            "majority_Q": float(np.mean(np.abs(eta - major) < band))}


class _Pass:
    """Gap positions and per-fibre statistics for one placement of every fibre's points."""

    def __init__(self, table: RadiusTable, fibres: list[np.ndarray], config: RegConfig) -> None:
        self.table, self.fibres, self.config = table, fibres, config

    def run(self, moved: list[np.ndarray]) -> tuple[list[dict], dict]:
        z0, z1 = self.config.z_range
        keep = [(p[:, 0] >= z0) & (p[:, 0] < z1) for p in moved]
        pts = np.concatenate([p[k] for p, k in zip(moved, keep)]) if moved else np.zeros((0, 3))
        who = np.concatenate([np.full(int(k.sum()), i) for i, k in enumerate(keep)]) if moved else np.zeros(0, int)
        g = gap_positions(self.table, pts) if len(pts) else {"s": np.zeros(0), "omega": np.zeros(0),
                                                             "phi": np.zeros(0)}
        fit = self.table.fit
        stats = []
        for i in range(len(moved)):
            sel = who == i
            theta = np.unwrap(g["phi"][sel])
            eta = g["omega"][sel] - (fit.sense * theta / TWO_PI + fit.phase)
            stats.append(fibre_stats(g["s"][sel], eta, self.config.band, self.config.min_points))
        return stats, g


def _group_means(stats: list[dict], members: np.ndarray) -> dict:
    q = [stats[i]["Q"] for i in np.nonzero(members)[0] if stats[i]["Q"] is not None]
    r = [stats[i]["R"] for i in np.nonzero(members)[0] if stats[i]["R"] is not None]
    return {"fibres": len(q), "Q": float(np.mean(q)) if q else None, "R": float(np.mean(r)) if r else None}


def verdict(observed: dict, z_null: list[dict], rotation_null: list[dict], config: RegConfig,
            table_ok: bool = True) -> dict:
    """The verdict for one group, by the rule fixed in the module docstring."""
    q, r = observed["Q"], observed["R"]
    if observed["fibres"] < config.min_fibres or q is None:
        return {"verdict": "NOT MEASURED", "reason": f"{observed['fibres']} fibres, under {config.min_fibres}"}

    def values(null, key):
        v = np.array([d[key] for d in null if d[key] is not None], dtype=np.float64)
        return v if len(v) else np.array([np.nan])

    zq = values(z_null, "Q")
    uninformative = bool(np.nanmedian(zq) >= config.uninformative_null_q)
    name, null = ("rotation", rotation_null) if uninformative else ("z-offset", z_null)
    nq, nr = values(null, "Q"), values(null, "R")
    q95, qmax = float(np.nanpercentile(nq, 95)), float(np.nanmax(nq))
    if not table_ok:
        v = "INCONCLUSIVE"
    elif q < config.not_registered_q or not q > q95:
        v = "NOT REGISTERED"
    elif q >= config.registered_q and q > qmax:
        v = "REGISTERED"
    else:
        v = "INCONCLUSIVE"
    offset = (v != "REGISTERED" and r is not None and bool(r > np.nanmax(nr))
              and bool(r - np.nanmedian(nr) >= config.offset_margin))
    return {"verdict": v, "null": name, "z_null_uninformative": uninformative, "table_ok": table_ok,
            "Q": q, "R": r, "fibres": observed["fibres"],
            "null_Q": {"median": float(np.nanmedian(nq)), "p95": q95, "max": qmax},
            "null_R": {"median": float(np.nanmedian(nr)), "max": float(np.nanmax(nr))},
            "offset_consistent": offset}


def _plain(value):
    """Tuples to lists, recursively, for a JSON report."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def _histogram(s: np.ndarray) -> list[int]:
    s = s[np.isfinite(s)]
    return np.histogram(s, bins=HISTOGRAM_BINS, range=(-0.5, 0.5))[0].astype(int).tolist()


def _thin(points: np.ndarray, at_most: int) -> np.ndarray:
    if len(points) <= at_most:
        return points
    return points[np.unique(np.round(np.linspace(0, len(points) - 1, at_most)).astype(np.int64))]


def _move_z(points: np.ndarray, dz: float, z_range: tuple[float, float]) -> np.ndarray:
    z0, z1 = z_range
    out = points.copy()
    out[:, 0] += dz
    inside = (out[:, 0] >= z0) & (out[:, 0] < z1)
    if 2 * inside.sum() < len(points):
        out[:, 0] = points[:, 0] - dz
    return out


def _rotate(points: np.ndarray, angle: float, umbilicus) -> np.ndarray:
    centre = umbilicus.yx_at(points[:, 0], out_of_range="extrapolate").yx
    d = points[:, 1:] - centre
    c, s = math.cos(angle), math.sin(angle)
    out = points.copy()
    out[:, 1] = centre[:, 0] + c * d[:, 0] + s * d[:, 1]
    out[:, 2] = centre[:, 1] - s * d[:, 0] + c * d[:, 1]
    return out


def diagnose(fit: WindingFit, fibres: list[Fibre], *, config: RegConfig,
             shell: OuterShell | None = None, table: RadiusTable | None = None) -> dict:
    """The registration diagnostic on ``fit`` for ``fibres``; a JSON-ready report of metrics only."""
    config.check()
    z0, z1 = config.z_range
    rows, thinned, full = [], [], []
    for f in fibres:
        z = f.points_zyx[:, 0]
        mask = (z >= z0) & (z < z1)
        if not mask.any():
            continue
        full.append(f.points_zyx[mask])
        thinned.append(_thin(f.points_zyx[mask], config.points_per_fibre))
        rows.append({"name": f.name, "digest": f.digest, "scale_source": f.scale_source,
                     "points_in_slab": int(mask.sum()), "points_thinned": len(thinned[-1])})
    report: dict = {"config": _plain(asdict(config)), "fibres_given": len(fibres), "fibres_in_slab": len(rows)}
    if not rows:
        report["verdicts"] = {"all": {"verdict": "NOT MEASURED", "reason": "no fibre in the slab"}}
        return report
    table = table or RadiusTable(fit, config.z_range, z_step=config.z_step, omega_bins=config.omega_bins)
    runner = _Pass(table, thinned, config)

    observed, g_obs = runner.run(thinned)
    # Exact check on the observed points
    pts = np.concatenate(thinned)
    exact = exact_positions(fit, pts, g_obs["k"], g_obs["f"], g_obs["r"])
    both = np.isfinite(exact) & np.isfinite(g_obs["s"])
    known_table = np.isfinite(g_obs["s"])
    gap_error = circular_gap(g_obs["s"][both], exact[both])
    median_error = float(np.median(gap_error)) if len(gap_error) else math.inf
    table_ok = median_error <= config.table_tolerance
    full_g = [gap_positions(table, p)["s"] for p in full]

    # Strata
    tercile = [None] * len(rows)
    if shell is not None:
        for i, p in enumerate(thinned):
            centre = fit.umbilicus.yx_at(p[:, 0], out_of_range="extrapolate").yx
            r, phi = cylindrical(p, centre)
            big_r, _ = shell.radius(p[:, 0], phi)
            rr = r / big_r
            if np.isfinite(rr).any():
                tercile[i] = TERCILES[min(2, max(0, int(np.nanmedian(rr) * 3)))]
    groups = {"all": np.ones(len(rows), bool)}
    for source in sorted({row["scale_source"] for row in rows}):
        groups[f"scale:{source}"] = np.array([row["scale_source"] == source for row in rows])
    for t in TERCILES:
        groups[f"tercile:{t}"] = np.array([x == t for x in tercile])

    rng = np.random.Generator(np.random.PCG64(config.seed))

    def null_series(mover, draws):
        series = []
        for _ in range(draws):
            stats, _ = runner.run([mover(p) for p in thinned])
            series.append({name: _group_means(stats, sel) for name, sel in groups.items()})
        return series

    def z_mover(lo, hi):
        return lambda p: _move_z(p, float(rng.choice((-1.0, 1.0)) * rng.uniform(lo, hi)), config.z_range)

    z_null = null_series(z_mover(*config.z_offset), config.null_draws)
    band_nulls = {f"{lo:g}-{hi:g}": null_series(z_mover(lo, hi), config.band_draws) for lo, hi in config.bands}
    rotation_null = null_series(lambda p: _rotate(p, float(rng.uniform(*config.rotation)), fit.umbilicus),
                                config.rotation_draws)
    _, g_rot = runner.run([_rotate(p, float(rng.uniform(*config.rotation)), fit.umbilicus) for p in thinned])
    _, g_z = runner.run([z_mover(*config.z_offset)(p) for p in thinned])

    obs_groups = {name: _group_means(observed, sel) for name, sel in groups.items()}
    verdicts = {name: verdict(obs_groups[name], [d[name] for d in z_null], [d[name] for d in rotation_null],
                              config, table_ok)
                for name in groups if not name.startswith("tercile:")}

    def curve(series, name):
        q = [d[name]["Q"] for d in series if d[name]["Q"] is not None]
        return {"draws": len(series), "Q_median": float(np.median(q)) if q else None,
                "Q_p95": float(np.percentile(q, 95)) if q else None}

    for i, row in enumerate(rows):
        row.update(tercile=tercile[i], **observed[i])
    report.update(
        points_in_slab=int(sum(len(p) for p in full)), points_thinned=int(len(pts)),
        points_unknown=int((~known_table).sum()),
        table={"z_step": config.z_step, "omega_bins": config.omega_bins, "points_checked": int(both.sum()),
               "median_error": median_error, "p95_error": float(np.percentile(gap_error, 95)) if len(gap_error)
               else None, "exact_disagrees": int((known_table & ~np.isfinite(exact)).sum()), "ok": table_ok},
        histograms={"edges": np.linspace(-0.5, 0.5, HISTOGRAM_BINS + 1).tolist(),
                    "observed_all_points": _histogram(np.concatenate(full_g)),
                    "observed": _histogram(g_obs["s"]), "z_offset_null_one_draw": _histogram(g_z["s"]),
                    "rotation_null_one_draw": _histogram(g_rot["s"])},
        spacing_median=float(np.nanmedian(g_obs["spacing"])) if np.isfinite(g_obs["spacing"]).any() else None,
        groups={name: {"observed": obs_groups[name],
                       "z_offset_null": curve(z_null, name), "rotation_null": curve(rotation_null, name),
                       "z_bands": {k: curve(v, name) for k, v in band_nulls.items()}}
                for name in groups},
        null_series={"z_offset": [d["all"] for d in z_null], "rotation": [d["all"] for d in rotation_null]},
        verdicts=verdicts, per_fibre=rows,
    )
    return report
