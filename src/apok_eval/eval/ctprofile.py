"""Do the fibres and the fit sit on the papyrus in the CT? The CT positive control for E7 v3.

If fibre points spread evenly through a fit's gaps, either the fibres are in another frame than the fit
(H1) or the fit misses the sheets locally (H2). The CT decides between them: whatever lies on papyrus is locked
to the layers the scan shows. Fixed before a run:

- **Profiles.** For each sampled point, the CT (one level of the volume, in the frame's voxels) along its
  umbilicus ray: the direction in (y, x) from the umbilicus at the point's z, offsets δ = −``reach`` ..
  ``reach`` voxels, nearest voxel. A profile that touches an absent chunk, or is flat (the mask), is dropped and
  counted.
- **Points.** Fibres: each in-slab fibre's points thinned to ``points_per_fibre``, halved (down to 4) until the
  chunks their profiles need fit in ``max_chunks`` and ``memory_bytes``. Fit: up to ``fit_points``
  vertices of the fit in the slab whose profiles lie in chunks the fibres already needed. Null: up to
  ``null_points`` random positions in those chunks, within the fit's radial range. Numpy PCG64 seeded ``seed``.
- **Period.** T is ``period_fixed`` when it is set. Otherwise it is the first maximum, at lags
  ``period_range``, of the mean autocorrelation of every profile (each centred); with none there, every verdict
  is INCONCLUSIVE. The autocorrelation is reported either way.
- **Phase.** Per point, φ = arg Σ (v(δ) − v̄) · hann(δ) · exp(−2πiδ / T) over |δ| ≤ ``window``: where in the layer
  pattern the point sits. Points on papyrus share a phase, at a layer's middle or its surface alike.
- **Locking.** Fibres vote once each: R = the length of the mean of each fibre's unit mean phase vector. The fit
  and the null vote per point. p is Rayleigh's. LOCKED: R ≥ ``locked_r`` and p < ``p_max``. NOT LOCKED: R <
  ``unlocked_r`` or p ≥ 0.05. Otherwise UNCLEAR. The null must be NOT LOCKED, or every verdict is INCONCLUSIVE.
- **Reading.** Fibres locked, fit not: the fit misses the sheets locally (H2). Fit locked, fibres not: the
  fibres are in another frame (H1). Both locked: the frames agree and the mismatch is in the fit's sheet
  positions or count (the period against the fit's own sheet spacing says which). Neither: the CT reading is in
  question.
- **Curvature.** Fit-free and CT-free: a circle fitted to each fibre's (y, x) in the slab. ρ / d (its radius over
  its distance from the umbilicus) and |centre − umbilicus| / d, for fibres sweeping at least ``min_sweep`` turns.
  Sheets wrap the umbilicus roughly concentrically, so a fibre in the frame gives ρ / d near 1. Reported only.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np

from apok_eval.eval.fibres import Fibre
from apok_eval.eval.fit import TWO_PI, WindingFit
from apok_eval.eval.geometry import cylindrical
from apok_eval.refusal import Refusal, register_codes

register_codes(
    "apok_eval.eval.ctprofile",
    {
        "EVAL_CT_PARAMETER": "a CT-profile setting is out of range",
        "EVAL_CT_TOO_MANY_CHUNKS": "the fibres' profiles need more chunks than the run allows",
    },
)


@dataclass(frozen=True)
class CTConfig:
    z_range: tuple[float, float]
    reach: int = 32
    window: int = 16
    points_per_fibre: int = 32
    fit_points: int = 10000
    null_points: int = 10000
    period_range: tuple[int, int] = (6, 30)
    min_sweep: float = 0.05
    seed: int = 1
    locked_r: float = 0.30
    unlocked_r: float = 0.15
    p_max: float = 1e-3
    max_chunks: int = 4000
    period_fixed: int = 0
    memory_bytes: int = 6 * 2 ** 30

    def check(self) -> None:
        z0, z1 = self.z_range
        lo, hi = self.period_range
        ok = (math.isfinite(z0) and math.isfinite(z1) and z0 < z1 and 4 <= self.window <= self.reach
              and 2 <= lo < hi < self.reach and self.points_per_fibre >= 2 and self.fit_points >= 10
              and self.null_points >= 10 and 0 < self.min_sweep < 1 and 0 < self.unlocked_r <= self.locked_r < 1
              and 0 < self.p_max < 0.05 and self.max_chunks >= 1 and 0 <= self.period_fixed < self.reach)
        if not ok:
            raise Refusal("EVAL_CT_PARAMETER", f"CT-profile configuration is out of range: {asdict(self)}",
                          config=str(asdict(self)))


class ChunkCache:
    """Level voxels by chunk index, fetched once through ``loader(index) -> array | None``."""

    def __init__(self, chunks: tuple[int, int, int], shape: tuple[int, int, int],
                 loader: Callable[[tuple[int, int, int]], np.ndarray | None]) -> None:
        self.chunks = np.asarray(chunks, dtype=np.int64)
        self.shape = np.asarray(shape, dtype=np.int64)
        self.loader = loader
        self.arrays: dict[tuple[int, int, int], np.ndarray | None] = {}

    def index(self, voxels: np.ndarray) -> np.ndarray:
        return voxels // self.chunks

    def needed(self, voxels: np.ndarray) -> set[tuple[int, int, int]]:
        inside = np.all((voxels >= 0) & (voxels < self.shape), axis=-1)
        idx = self.index(voxels[inside])
        return {tuple(int(v) for v in row) for row in np.unique(idx.reshape(-1, 3), axis=0)}

    def load(self, keys, mapper=map) -> None:
        keys = [k for k in sorted(keys) if k not in self.arrays]
        for k, a in zip(keys, mapper(self.loader, keys)):
            self.arrays[k] = a

    def covered(self, voxels: np.ndarray) -> np.ndarray:
        """(n,) whether every voxel of each row of ``voxels`` (n, m, 3) is inside a loaded, present chunk."""
        inside = np.all((voxels >= 0) & (voxels < self.shape), axis=-1)
        idx = self.index(voxels)
        present = np.array([self.arrays.get(tuple(int(v) for v in row)) is not None
                            for row in idx.reshape(-1, 3)]).reshape(idx.shape[:-1])
        return np.all(inside & present, axis=1)

    def sample(self, voxels: np.ndarray) -> np.ndarray:
        """Values at integer ``voxels`` (..., 3); NaN where the chunk is not loaded or absent."""
        flat = voxels.reshape(-1, 3)
        out = np.full(len(flat), np.nan)
        idx = self.index(flat)
        keys = idx[:, 0] * 1_000_003 ** 2 + idx[:, 1] * 1_000_003 + idx[:, 2]
        for key in np.unique(keys):
            sel = np.nonzero(keys == key)[0]
            k = tuple(int(v) for v in idx[sel[0]])
            arr = self.arrays.get(k)
            if arr is None:
                continue
            local = flat[sel] - np.asarray(k) * self.chunks
            ok = np.all((local >= 0) & (local < np.asarray(arr.shape)), axis=1)
            out[sel[ok]] = arr[local[ok, 0], local[ok, 1], local[ok, 2]]
        return out.reshape(voxels.shape[:-1])


def ray_voxels(points: np.ndarray, umbilicus, reach: int) -> np.ndarray:
    """(n, 2·reach + 1, 3) nearest voxels along each point's outward umbilicus ray."""
    points = np.asarray(points, dtype=np.float64)
    centre = umbilicus.yx_at(points[:, 0], out_of_range="extrapolate").yx
    d = points[:, 1:] - centre
    norm = np.hypot(d[:, 0], d[:, 1])
    unit = d / np.where(norm > 0, norm, 1.0)[:, None]
    offsets = np.arange(-reach, reach + 1, dtype=np.float64)
    pos = np.repeat(points[:, None, :], len(offsets), axis=1)
    pos[:, :, 1:] += offsets[None, :, None] * unit[:, None, :]
    return np.round(pos).astype(np.int64)


def profiles(cache: ChunkCache, voxels: np.ndarray) -> tuple[np.ndarray, dict]:
    """Profiles (n, m) with dropped rows NaN, and the counts of each reason for dropping."""
    v = cache.sample(voxels)
    missing = ~np.isfinite(v).all(axis=1)
    flat = ~missing & (np.nanstd(np.where(np.isfinite(v), v, 0.0), axis=1) == 0)
    v[missing | flat] = np.nan
    return v, {"missing_chunk": int(missing.sum()), "flat": int(flat.sum()), "kept": int((~missing & ~flat).sum())}


def period(prof: np.ndarray, lags: tuple[int, int]) -> tuple[int | None, list[float]]:
    """The first local maximum of the mean autocorrelation within ``lags``, and the curve from lag 0."""
    p = prof[np.isfinite(prof).all(axis=1)]
    if not len(p):
        return None, []
    p = p - p.mean(axis=1, keepdims=True)
    m = p.shape[1]
    var = (p * p).sum(axis=1)
    keep = var > 0
    p, var = p[keep], var[keep]
    ac = [float(np.mean((p[:, : m - k] * p[:, k:]).sum(axis=1) / var)) for k in range(0, lags[1] + 2)]
    for k in range(lags[0], lags[1] + 1):
        if ac[k] > ac[k - 1] and ac[k] >= ac[k + 1]:
            return k, ac
    return None, ac


def phases(prof: np.ndarray, T: float, window: int) -> np.ndarray:
    """Per-profile phase at δ = 0 of the layer pattern of period T (NaN for dropped profiles)."""
    m = prof.shape[1]
    reach = (m - 1) // 2
    delta = np.arange(-reach, reach + 1)
    w = np.where(np.abs(delta) <= window, 0.5 * (1 + np.cos(np.pi * delta / (window + 1))), 0.0)
    good = np.isfinite(prof).all(axis=1)
    centred = np.where(good[:, None], prof - np.where(good, prof.mean(axis=1), 0.0)[:, None], np.nan)
    z = np.nansum(centred * (w * np.exp(-2j * np.pi * delta / T))[None, :], axis=1)
    out = np.angle(z)
    out[~np.isfinite(prof).all(axis=1)] = np.nan
    return out


def rayleigh(angles: np.ndarray) -> dict:
    """Resultant length, mean angle and Rayleigh's p (Zar's approximation) of ``angles``."""
    a = angles[np.isfinite(angles)]
    n = len(a)
    if n == 0:
        return {"n": 0, "R": None, "mean_angle": None, "p": None}
    z = np.mean(np.exp(1j * a))
    R = float(abs(z))
    nr = n * R
    p = math.exp(math.sqrt(1 + 4 * n + 4 * (n * n - nr * nr)) - (1 + 2 * n))
    return {"n": n, "R": R, "mean_angle": float(np.angle(z)), "p": float(min(1.0, p))}


def locked(stats: dict, config: CTConfig) -> str:
    if not stats["n"]:
        return "INCONCLUSIVE"
    if stats["R"] < config.unlocked_r or stats["p"] >= 0.05:
        return "NOT LOCKED"
    if stats["R"] >= config.locked_r and stats["p"] < config.p_max:
        return "LOCKED"
    return "UNCLEAR"


def circle_fit(yx: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Least-squares (Kåsa) circle through 2-D points; None when they are too few or collinear."""
    if len(yx) < 3:
        return None
    y, x = yx[:, 0], yx[:, 1]
    A = np.stack([y, x, np.ones_like(y)], axis=1)
    b = y * y + x * x
    sol, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    if rank < 3:
        return None
    cy, cx = sol[0] / 2, sol[1] / 2
    r2 = sol[2] + cy * cy + cx * cx
    if not (np.isfinite(r2) and r2 > 0):
        return None
    return np.array([cy, cx]), float(math.sqrt(r2))


def _thin(points: np.ndarray, at_most: int) -> np.ndarray:
    if len(points) <= at_most:
        return points
    return points[np.unique(np.round(np.linspace(0, len(points) - 1, at_most)).astype(np.int64))]


def _mean_profile(prof: np.ndarray) -> list[float] | None:
    p = prof[np.isfinite(prof).all(axis=1)]
    if not len(p):
        return None
    z = (p - p.mean(axis=1, keepdims=True)) / p.std(axis=1, keepdims=True)
    return [float(v) for v in z.mean(axis=0)]


def _quantiles(values) -> dict | None:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if not len(v):
        return None
    q = np.percentile(v, [10, 25, 50, 75, 90])
    return {"n": int(len(v)), **{f"q{k}": float(x) for k, x in zip((10, 25, 50, 75, 90), q)}}


def curvature(fibres: list[Fibre], umbilicus, config: CTConfig) -> dict:
    """ρ / d and |centre − umbilicus| / d of each fibre's slab arc (see the module docstring)."""
    z0, z1 = config.z_range
    ratio, offset, used = [], [], 0
    for f in fibres:
        p = f.points_zyx[(f.points_zyx[:, 0] >= z0) & (f.points_zyx[:, 0] < z1)]
        if len(p) < 3:
            continue
        centre = umbilicus.yx_at(p[:, 0], out_of_range="extrapolate").yx
        r, phi = cylindrical(p, centre)
        if np.ptp(np.unwrap(phi)) / TWO_PI < config.min_sweep:
            continue
        fitted = circle_fit(p[:, 1:])
        if fitted is None:
            continue
        c, rho = fitted
        u = centre.mean(axis=0)
        d = float(np.linalg.norm(p[:, 1:].mean(axis=0) - u))
        if d <= 0:
            continue
        used += 1
        ratio.append(rho / d)
        offset.append(float(np.linalg.norm(c - u)) / d)
    return {"fibres": used, "rho_over_d": _quantiles(ratio), "centre_offset_over_d": _quantiles(offset),
            "within_20pct": int(sum(abs(x - 1) <= 0.2 for x in ratio))}


def diagnose(fit: WindingFit, fibres: list[Fibre], cache: ChunkCache, *, config: CTConfig,
             mapper=map, families: dict[str, str] | None = None) -> dict:
    """The CT-profile diagnostic on ``fit`` and ``fibres`` with the CT behind ``cache``; a JSON-ready report of
    metrics only.

    Beside the per-fibre vote the verdicts use, the fibres' locking is also reported per point (every fibre
    point's phase counted once, like the fit's vertices), by scale group and, when ``families`` maps fibre
    names to H / V / U, by family. These figures never change a verdict."""
    config.check()
    rng = np.random.Generator(np.random.PCG64(config.seed))
    z0, z1 = config.z_range
    umb = fit.umbilicus
    in_slab = []
    for f in fibres:
        p = f.points_zyx[(f.points_zyx[:, 0] >= z0) & (f.points_zyx[:, 0] < z1)]
        if len(p):
            in_slab.append((f, p))
    report: dict = {"config": {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(config).items()},
                    "fibres_in_slab": len(in_slab), "curvature": curvature(fibres, umb, config)}
    if not in_slab:
        report["verdicts"] = {"fibres": "INCONCLUSIVE", "fit": "INCONCLUSIVE", "reason": "no fibre in the slab"}
        return report
    limit = min(config.max_chunks, config.memory_bytes // max(1, int(np.prod(cache.chunks))))
    per = config.points_per_fibre
    while True:
        rows = [{"name": f.name, "scale_source": f.scale_source} for f, _ in in_slab]
        thinned = [_thin(p, per) for _, p in in_slab]
        fib_pts = np.concatenate(thinned)
        who = np.concatenate([np.full(len(t), i) for i, t in enumerate(thinned)])
        fib_vox = ray_voxels(fib_pts, umb, config.reach)
        need = cache.needed(fib_vox)
        if len(need) <= limit or per <= 4:
            break
        per = max(4, per // 2)
    if len(need) > limit:
        raise Refusal("EVAL_CT_TOO_MANY_CHUNKS", f"the fibres' profiles need {len(need)} chunks, over the run's "
                      f"{limit}", needed=len(need), allowed=limit)
    report["points_per_fibre_used"] = per
    cache.load(need, mapper)
    fib_prof, fib_counts = profiles(cache, fib_vox)

    # Fit vertices in the slab whose profiles lie in the loaded chunks
    fit_list = []
    for p in fit.parts:
        v = p.zyx[p.valid]
        fit_list.append(v[(v[:, 0] >= z0) & (v[:, 0] < z1)])
    fit_all = np.concatenate(fit_list) if fit_list else np.zeros((0, 3))
    r_all, _ = cylindrical(fit_all, umb.yx_at(fit_all[:, 0], out_of_range="extrapolate").yx) if len(fit_all) \
        else (np.zeros(0), None)
    loaded_idx = np.array(sorted(k for k, a in cache.arrays.items() if a is not None), dtype=np.int64)
    if len(fit_all) and len(loaded_idx):
        home = cache.index(np.round(fit_all).astype(np.int64))
        loaded_set = {tuple(r) for r in loaded_idx.tolist()}
        near = np.array([tuple(h) in loaded_set for h in home.tolist()])
        cand = fit_all[near]
        cand = cand[rng.permutation(len(cand))[: 4 * config.fit_points]]
        vox = ray_voxels(cand, umb, config.reach)
        cand = cand[cache.covered(vox)][: config.fit_points]
    else:
        cand = np.zeros((0, 3))
    fit_prof, fit_counts = profiles(cache, ray_voxels(cand, umb, config.reach)) if len(cand) else \
        (np.zeros((0, 2 * config.reach + 1)), {"kept": 0})

    # Null: random positions in the loaded chunks, within the fit's radial range
    null = np.zeros((0, 3))
    if len(loaded_idx):
        r_lo, r_hi = (float(np.min(r_all)), float(np.max(r_all))) if len(r_all) else (0.0, np.inf)
        k = loaded_idx[rng.integers(0, len(loaded_idx), 4 * config.null_points)]
        q = (k + rng.uniform(0, 1, (len(k), 3))) * cache.chunks
        r_q, _ = cylindrical(q, umb.yx_at(q[:, 0], out_of_range="extrapolate").yx)
        q = q[(r_q >= r_lo) & (r_q <= r_hi) & (q[:, 0] >= z0) & (q[:, 0] < z1)]
        null = q[cache.covered(ray_voxels(q, umb, config.reach))][: config.null_points]
    null_prof, null_counts = profiles(cache, ray_voxels(null, umb, config.reach)) if len(null) else \
        (np.zeros((0, 2 * config.reach + 1)), {"kept": 0})

    T, ac = period(np.concatenate([fib_prof, fit_prof, null_prof]), config.period_range)
    if config.period_fixed:
        T = config.period_fixed
    report.update(points={"fibres": fib_counts, "fit": fit_counts, "null": null_counts},
                  chunks={"needed": len(need), "loaded": int(sum(a is not None for a in cache.arrays.values())),
                          "absent": int(sum(a is None for a in cache.arrays.values()))},
                  period=T, autocorrelation=ac,
                  mean_profiles={"fibres": _mean_profile(fib_prof), "fit": _mean_profile(fit_prof),
                                 "null": _mean_profile(null_prof)})
    if T is None:
        report["verdicts"] = {"fibres": "INCONCLUSIVE", "fit": "INCONCLUSIVE", "reason": "no layer period found"}
        return report

    ph_f = phases(fib_prof, T, config.window)
    per_fibre = []
    for i, row in enumerate(rows):
        a = ph_f[who == i]
        a = a[np.isfinite(a)]
        if len(a):
            z = np.mean(np.exp(1j * a))
            per_fibre.append(float(np.angle(z)))
            row.update(points=int(len(a)), R=float(abs(z)), phase=float(np.angle(z)))
        else:
            row.update(points=0, R=None, phase=None)
    fib_stats = rayleigh(np.array(per_fibre))
    fit_stats = rayleigh(phases(fit_prof, T, config.window)) if len(fit_prof) else rayleigh(np.zeros(0))
    null_stats = rayleigh(phases(null_prof, T, config.window)) if len(null_prof) else rayleigh(np.zeros(0))
    groups = {}
    for source in sorted({r["scale_source"] for r in rows}):
        groups[source] = rayleigh(np.array([r["phase"] for r in rows if r["scale_source"] == source
                                            and r["phase"] is not None]))
    finite = np.isfinite(ph_f)

    def per_point(member):
        keep = finite & np.isin(who, [i for i, r in enumerate(rows) if member(r)])
        return rayleigh(ph_f[keep])

    point_stats = {"all": rayleigh(ph_f[finite])}
    point_stats.update({f"scale:{k}": per_point(lambda r, k=k: r["scale_source"] == k) for k in groups})
    by_family = {}
    if families is not None:
        for row in rows:
            row["family"] = families.get(row["name"], "U")
        for fam in sorted({r["family"] for r in rows}):
            by_family[fam] = rayleigh(np.array([r["phase"] for r in rows if r["family"] == fam
                                                and r["phase"] is not None]))
            point_stats[f"family:{fam}"] = per_point(lambda r, fam=fam: r["family"] == fam)
    null_verdict = locked(null_stats, config)
    sound = null_verdict == "NOT LOCKED"
    verdicts = {"null": null_verdict,
                "fibres": locked(fib_stats, config) if sound else "INCONCLUSIVE",
                "fit": locked(fit_stats, config) if sound else "INCONCLUSIVE"}
    verdicts.update({f"fibres:{k}": (locked(v, config) if sound else "INCONCLUSIVE") for k, v in groups.items()})
    offset = None
    if fib_stats["mean_angle"] is not None and fit_stats["mean_angle"] is not None:
        d = (fib_stats["mean_angle"] - fit_stats["mean_angle"] + math.pi) % TWO_PI - math.pi
        offset = float(d / TWO_PI * T)
    report.update(locking={"fibres": fib_stats, "fit": fit_stats, "null": null_stats, "fibres_by_scale": groups,
                           "fibres_by_family": by_family, "fibres_per_point": point_stats},
                  fibre_minus_fit_offset_voxels=offset, verdicts=verdicts,
                  per_fibre_R=_quantiles(r["R"] for r in rows), per_fibre=rows)
    return report
