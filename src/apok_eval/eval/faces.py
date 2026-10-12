"""E7 v3: face-aware held-out registration of a spiral fit to human fibre traces.

villa documents that horizontal (H) fibres lie on a sheet's front face and vertical (V) fibres on its back face,
a few voxels apart, and that a fit can target only one face (``spiral-fitting/README.md``, "Vertical-fiber
radial offset"). Scoring every fibre against the fit's sheet itself (:mod:`apok_eval.eval.registration`) would
fail a fit on one face for the other family alone. E7 v3 lets each family sit at its own place in the gap,
learnt on half the fibres and scored on the other half. Every rule below is fixed before a run scores anything.

- **Gap position.** As :mod:`apok_eval.eval.registration`: each fibre point's signed position s ∈ [−½, ½) in
  the fit's gap on its umbilicus ray, 0 on a sheet, in units of the local spacing; the same thinning
  (``points_per_fibre``), the same table and its exact check (a failed check makes every verdict
  INCONCLUSIVE). A fibre is scored in a pass when at least ``min_points`` of its points have a known s.
- **Families.** Each fibre is H, V or untagged (U) by villa's ``classify_fiber_hv`` precedence
  (``apok_eval.eval.fibres.hv_family``).
- **Split.** A fibre is in the calibration half when the first byte of sha256("<split_seed>:<name>") is even,
  otherwise in the evaluation half. The split depends on the name alone, never on the data.
- **Calibration.** For a group of calibration fibres, the share at offset D is the mean over its fibres of the
  fraction of their points within ``band`` of D, distance taken around the gap's circle. D runs over the grid
  −½, −½ + ``offset_step``, …, ½ − ``offset_step``; the D with the largest share is chosen. Tied D form runs of
  neighbours on the circle (a precise family scores the same anywhere within ``band`` of its true offset): the
  centre of the longest run is taken, ties between runs going to the smallest |D|, then the lower D. Each
  family (H, V, U) gets its own D from its own calibration fibres;
  a family with no scored calibration fibre takes the D of all calibration fibres pooled (reported).
- **Score.** An evaluation fibre's Q_f is the fraction of its points within ``band`` of its family's D. A
  group's Q is the mean of Q_f over its scored evaluation fibres (fibre-weighted); the groups are all
  evaluation fibres ("all"), each family, and any named subset of fibres (its evaluation fibres, at their
  families' D). R is the mean over the same fibres of the registration diagnostic's per-fibre R.
- **Nulls.** The registration diagnostic's z-offset null (``null_draws``, each fibre moved by ±U(``z_offset``)
  in z) and rotation null (``rotation_draws``, each fibre turned about the umbilicus by U(``rotation``)), numpy
  PCG64 seeded ``seed``, in that order. Each draw goes through the identical procedure, calibration included.
- **Verdict** per group, for "all", H, V and each subset (U is reported only): the registration rule
  (``registration.verdict``) on the evaluation fibres. PASS when it gives REGISTERED (Q ≥ ``registered_q`` and
  above every draw of the null in use), FAIL when NOT REGISTERED (Q < ``not_registered_q`` or not above that
  null's 95th percentile), INCONCLUSIVE otherwise, NOT MEASURED under ``min_fibres`` scored evaluation fibres.
  The null in use is the z-offset null unless its median Q is at least ``uninformative_null_q``. The CT
  positive control that gates scoring is the caller's (:mod:`apok_eval.eval.cli` runs
  :mod:`apok_eval.eval.ctprofile` first).
- **Ceiling (reported, never a threshold).** For each σ in ``sigmas``, every observed point with a known gap
  position is replaced by a perfect fit's: s = wrap((face + N(0, σ)) / spacing), with the point's own local
  spacing, face 0 for H, ``face_gap`` voxels for V, and one of the two drawn per fibre (p = ½) for U, seeded
  ``ceiling_seed``. The same split, calibration and score give the Q a perfect front-face fit could reach
  under fibre noise σ.
- **Also reported:** per-family gap histograms (all scored points, both halves), circular-mean offsets, the
  calibrated D in spacing units and in voxels, and per-fibre winding consistency (the majority share: the
  fraction of a fibre's points on its most common sheet of the fit) against the z-offset null.
- **Winding identity at the family offset.** The majority share rounds η at the fit's sheet, so a family
  whose points sit well out from the fit's sheet lies near the rounding boundary and its switches are overcounted. Beside
  it, each fibre's share of its points on its most common sheet after shifting η by its family's calibrated D:
  sheet = round(η − D), so the boundaries sit at D ± ½. The same points as the majority share (finite s and η,
  at least ``min_points``); every fibre, both halves; a fibre whose family has no D has none. Each null draw
  uses the D it calibrated itself. Reported as ``winding_consistency_at_offset`` and per fibre as
  ``majority_share_at_D``; no verdict reads it.
- **For a fit that was given the calibration half.** Three reports, each off unless asked for, read off the
  same passes (no draw changes): the calibration fibres' own Q at their family's D (a manipulation check);
  every pass scored again with every family's D fixed at a given value (the disclosed bias: D is calibrated on
  the fibres the fit was given); and each pass scored per stratum of points (``stratum_scores``), a point
  keeping the stratum of its unmoved position in every null draw, D recalibrated per draw as for the groups.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass

import numpy as np

from apok_eval.eval.fibres import Fibre, hv_family
from apok_eval.eval.fit import TWO_PI, WindingFit
from apok_eval.eval.registration import (HISTOGRAM_BINS, RadiusTable, RegConfig, _move_z, _plain, _rotate, _thin,
                                    circular_gap, exact_positions, fibre_stats, gap_positions, verdict)
from apok_eval.refusal import Refusal, register_codes

register_codes(
    "apok_eval.eval.faces",
    {
        "EVAL_FACE_PARAMETER": "an E7 v3 setting is out of range",
        "EVAL_FACE_FAMILIES": "the fibre families given do not match the fibres",
    },
)

FAMILIES = ("H", "V", "U")
SCORED_FAMILIES = ("H", "V")
E7V3 = {"REGISTERED": "PASS", "NOT REGISTERED": "FAIL", "INCONCLUSIVE": "INCONCLUSIVE",
        "NOT MEASURED": "NOT MEASURED"}


@dataclass(frozen=True)
class FaceConfig:
    offset_step: float = 0.01
    split_seed: int = 2
    sigmas: tuple[float, ...] = (1.0, 2.0, 3.0)
    face_gap: float = 4.0
    ceiling_seed: int = 3

    def check(self) -> None:
        n = 1.0 / self.offset_step if self.offset_step > 0 else math.inf
        problems = []
        if not (0 < self.offset_step <= 0.1 and abs(n - round(n)) < 1e-9):
            problems.append("offset_step must divide 1 and lie in (0, 0.1]")
        if not self.sigmas or any(not (0 < s < 50) for s in self.sigmas):
            problems.append("sigmas must be positive and under 50 voxels")
        if not (0 <= self.face_gap < 50):
            problems.append("face_gap must lie in [0, 50) voxels")
        if problems:
            raise Refusal("EVAL_FACE_PARAMETER", f"E7 v3 configuration is out of range: {problems}",
                          problems=problems)

    def grid(self) -> np.ndarray:
        n = int(round(1.0 / self.offset_step))
        return (np.arange(n) - n // 2) / n


def in_calibration(name: str, seed: int) -> bool:
    """The split: calibration when the first byte of sha256("<seed>:<name>") is even."""
    return hashlib.sha256(f"{seed}:{name}".encode()).digest()[0] % 2 == 0


def fractions(s_per_fibre: list[np.ndarray], band: float, grid: np.ndarray, min_points: int) -> np.ndarray:
    """(len(grid), fibres): each fibre's fraction of known points within ``band`` of each D (around the
    circle); NaN for a fibre with fewer than ``min_points`` known points."""
    out = np.full((len(grid), len(s_per_fibre)), np.nan)
    known = [s[np.isfinite(s)] for s in s_per_fibre]
    keep = [i for i, s in enumerate(known) if len(s) >= min_points]
    if not keep:
        return out
    lens = np.array([len(known[i]) for i in keep])
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    d = np.mod(np.concatenate([known[i] for i in keep])[None, :] - grid[:, None], 1.0)
    hits = (np.minimum(d, 1.0 - d) < band).astype(np.float64)
    out[:, keep] = np.add.reduceat(hits, starts, axis=1) / lens
    return out


def best_offset(frac: np.ndarray, grid: np.ndarray) -> int | None:
    """Index of the D with the largest mean share over the columns of ``frac``. When several D tie, the tied D
    form runs of neighbours on the circle; the centre of the longest run is taken (the lower middle of an
    even run), ties between runs going to the centre with the smallest |D|, then the lower D. None without a
    column."""
    if frac.shape[1] == 0:
        return None
    share = frac.mean(axis=1)
    top = share >= share.max() - 1e-12
    n = len(grid)
    if top.all():
        return int(np.argmin(np.abs(grid)))
    start = int(np.argmin(top))  # a D off the maximum: runs are read from there around the circle
    runs, run = [], []
    for k in range(1, n + 1):
        i = (start + k) % n
        if top[i]:
            run.append(i)
        elif run:
            runs.append(run)
            run = []
    if run:
        runs.append(run)
    centres = [(r[(len(r) - 1) // 2], len(r)) for r in runs]
    return int(min(centres, key=lambda c: (-c[1], abs(grid[c[0]]), grid[c[0]]))[0])


def score(frac: np.ndarray, family: np.ndarray, calibration: np.ndarray, grid: np.ndarray,
          r: np.ndarray, groups: dict[str, np.ndarray], fixed: float | None = None) -> dict:
    """Calibrate each family's D on the calibration fibres and score the evaluation fibres (module docstring).

    ``fixed``: every family's D is the grid value nearest ``fixed`` instead, with no calibration. Also
    returned: each family's grid index and each calibration fibre's own share at its family's D (``q_calibration``,
    with its fibre-weighted mean in ``calibration``)."""
    known = np.isfinite(frac[0])
    cal, ev = calibration & known, ~calibration & known
    pooled = best_offset(frac[:, cal], grid)
    offsets, index = {}, {}
    for g in FAMILIES:
        i = best_offset(frac[:, cal & (family == g)], grid)
        fallback = i is None
        i = pooled if fallback else i
        if fixed is not None:
            i, fallback = int(np.argmin(np.abs(grid - fixed))), False
        index[g] = i
        offsets[g] = {"D": None if i is None else float(grid[i]), "fallback": fallback,
                      "calibration_fibres": int((cal & (family == g)).sum()),
                      "calibration_share": None if i is None or fallback or not (cal & (family == g)).any()
                      else float(frac[i, cal & (family == g)].mean())}
    q = np.full(frac.shape[1], np.nan)
    for g in FAMILIES:
        m = ev & (family == g)
        if index[g] is not None and m.any():
            q[m] = frac[index[g], m]
    out = {}
    for name, members in groups.items():
        m = ev & members & np.isfinite(q)
        rr = r[m & np.isfinite(r)]
        out[name] = {"fibres": int(m.sum()), "Q": float(q[m].mean()) if m.any() else None,
                     "R": float(rr.mean()) if len(rr) else None}
    q_cal = np.full(frac.shape[1], np.nan)
    for g in FAMILIES:
        own = cal & (family == g)
        if index[g] is not None and own.any():
            q_cal[own] = frac[index[g], own]
    return {"offsets": offsets, "groups": out, "q": q, "index": index, "q_calibration": q_cal,
            "calibration": {"all": _fibre_mean(q_cal), **{g: _fibre_mean(q_cal[family == g]) for g in FAMILIES}}}


def _fibre_mean(values: np.ndarray) -> dict:
    """The fibre-weighted mean of the finite values, with their count."""
    v = values[np.isfinite(values)]
    return {"fibres": int(len(v)), "Q": float(v.mean()) if len(v) else None}


def stratum_scores(s_list: list[np.ndarray], labels: list[np.ndarray], names, family: np.ndarray,
                   index: dict, q: np.ndarray, groups: dict[str, np.ndarray], band: float, grid: np.ndarray,
                   min_points: int) -> dict:
    """Strata of points for one pass: per stratum and group, the mean over the group's scored evaluation
    fibres (finite ``q``) with at least ``min_points`` known points in the stratum of their fraction of those
    points within ``band`` of their family's D (``index``, the pass's own calibration). ``labels[i]`` gives the
    stratum (an index into ``names``) of each of fibre i's points in ``s_list[i]``."""
    scored = np.isfinite(q)
    out = {}
    for k, name in enumerate(names):
        sub = [s[np.asarray(lab) == k] for s, lab in zip(s_list, labels)]
        known = np.array([int(np.isfinite(s).sum()) for s in sub])
        qk = np.full(len(sub), np.nan)
        for g in FAMILIES:
            m = np.nonzero(family == g)[0]
            if index[g] is not None and len(m):
                qk[m] = fractions([sub[i] for i in m], band, grid[index[g]:index[g] + 1], min_points)[0]
        out[name] = {}
        for gname, members in groups.items():
            m = members & scored
            ok = m & np.isfinite(qk)
            out[name][gname] = {"fibres": int(ok.sum()), "Q": float(qk[ok].mean()) if ok.any() else None,
                                "points": int(known[m].sum())}
    return out


def _pass(table: RadiusTable, moved: list[np.ndarray], config: RegConfig):
    """Gap positions, local spacings, η and the per-fibre registration statistics for one placement of every
    fibre."""
    z0, z1 = config.z_range
    keep = [(p[:, 0] >= z0) & (p[:, 0] < z1) for p in moved]
    pts = np.concatenate([p[k] for p, k in zip(moved, keep)])
    who = np.concatenate([np.full(int(k.sum()), i) for i, k in enumerate(keep)])
    g = gap_positions(table, pts) if len(pts) else {k: np.zeros(0) for k in ("s", "omega", "phi", "spacing")}
    fit = table.fit
    s_list, spacing, stats, etas = [], [], [], []
    for i in range(len(moved)):
        sel = who == i
        eta = g["omega"][sel] - (fit.sense * np.unwrap(g["phi"][sel]) / TWO_PI + fit.phase)
        s_list.append(g["s"][sel])
        spacing.append(g["spacing"][sel])
        etas.append(eta)
        stats.append(fibre_stats(g["s"][sel], eta, config.band, config.min_points))
    return s_list, spacing, stats, g, pts, etas


def share_at_offset(s: np.ndarray, eta: np.ndarray, d: float | None, min_points: int) -> float | None:
    """The fraction of a fibre's points on its most common sheet after shifting η by its family's D
    (sheet = round(η − D)); the points the majority share uses, None under ``min_points`` or without D."""
    if d is None:
        return None
    ok = np.isfinite(s) & np.isfinite(eta)
    n = int(ok.sum())
    if n < min_points:
        return None
    _, counts = np.unique(np.round(eta[ok] - d), return_counts=True)
    return float(counts.max() / n)


def shares_at_offsets(s_list, etas, family: np.ndarray, offsets: dict, min_points: int) -> list[float | None]:
    """``share_at_offset`` for every fibre of a pass, at the D that pass calibrated for its family."""
    return [share_at_offset(s, e, offsets[str(fam)]["D"], min_points) for s, e, fam in zip(s_list, etas, family)]


def _consistency(shares: list) -> dict:
    share = np.array([v for v in shares if v is not None], dtype=np.float64)
    return {"fibres": int(len(share)), "median": float(np.median(share)) if len(share) else None,
            "mean": float(share.mean()) if len(share) else None,
            "at_least_0.9": float(np.mean(share >= 0.9)) if len(share) else None}


def _against_null(observed: dict, draws: list[dict]) -> dict:
    """An observed consistency summary with its z-offset null's quantiles and the draws at or above it."""
    out = dict(observed)
    keys = ("median", "mean", "at_least_0.9")
    out["z_offset_null"] = {k: _quantiles(d[k] for d in draws) for k in keys}
    out["null_draws_at_or_above"] = {k: int(sum(d[k] is not None and observed[k] is not None and d[k] >= observed[k]
                                                for d in draws)) for k in keys}
    return out


def _quantiles(values) -> dict:
    v = np.array([x for x in values if x is not None], dtype=np.float64)
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)), "max": float(v.max())}


def _circular_mean(s: np.ndarray) -> dict:
    s = s[np.isfinite(s)]
    if not len(s):
        return {"points": 0, "mean": None, "resultant": None}
    z = np.mean(np.exp(2j * np.pi * s))
    return {"points": int(len(s)), "mean": float(np.angle(z) / TWO_PI), "resultant": float(abs(z))}


def diagnose(fit: WindingFit, fibres: list[Fibre], *, config: RegConfig,
             face: FaceConfig = FaceConfig(), families: list[tuple[str, str]] | None = None,
             subsets: dict[str, set[str]] | None = None, table: RadiusTable | None = None,
             calibration_q: bool = False, fixed_d: float | None = None, strata=None,
             strata_min_points: int = 8) -> dict:
    """E7 v3 on ``fit`` for ``fibres``; a JSON-ready report of metrics only (module docstring).

    ``families`` gives each fibre's (family, source) in order; by default ``hv_family``. ``subsets`` names
    sets of fibre names scored as their own groups.

    Three optional reports, each off by default so that E7 v3's own report is unchanged without them:
    ``calibration_q`` reports the calibration fibres' own Q at their family's D (``calibration_q``);
    ``fixed_d`` scores every pass a second time with every family's D fixed at that value (``fixed_d``: groups,
    their nulls and the registration verdict, reported only; per fibre ``Q_at_fixed_D``); ``strata`` (an object
    with ``names`` and ``labels(points_zyx)``) labels each thinned point by its unmoved position, and every pass,
    the nulls included, is scored per stratum by ``stratum_scores`` (``strata``). None of them changes a draw:
    they are read off the same passes.
    """
    config.check()
    face.check()
    if families is None:
        families = [hv_family(f) for f in fibres]
    if len(families) != len(fibres) or any(fam not in FAMILIES for fam, _ in families):
        raise Refusal("EVAL_FACE_FAMILIES", f"{len(families)} families for {len(fibres)} fibres, or a family "
                      f"outside {list(FAMILIES)}", fibres=len(fibres), families=len(families))
    subsets = dict(subsets or {})
    z0, z1 = config.z_range
    grid = face.grid()
    rows, thinned = [], []
    for f, (fam, source) in zip(fibres, families):
        mask = (f.points_zyx[:, 0] >= z0) & (f.points_zyx[:, 0] < z1)
        if not mask.any():
            continue
        thinned.append(_thin(f.points_zyx[mask], config.points_per_fibre))
        rows.append({"name": f.name, "digest": f.digest, "scale_source": f.scale_source, "family": fam,
                     "family_source": source, "half": "calibration" if in_calibration(f.name, face.split_seed)
                     else "evaluation", "points_in_slab": int(mask.sum()), "points_thinned": len(thinned[-1])})
    report: dict = {"config": {"registration": _plain(asdict(config)), "face": _plain(asdict(face))},
                    "fibres_given": len(fibres), "fibres_in_slab": len(rows),
                    "families_in_slab": {g: sum(r["family"] == g for r in rows) for g in FAMILIES},
                    "family_sources": {s: sum(r["family_source"] == s for r in rows)
                                       for s in sorted({r["family_source"] for r in rows})}}
    if not rows:
        report["verdicts"] = {"all": {"verdict": "NOT MEASURED", "e7v3": "NOT MEASURED",
                                      "reason": "no fibre in the slab"}}
        return report
    family = np.array([r["family"] for r in rows])
    calibration = np.array([r["half"] == "calibration" for r in rows])
    groups = {"all": np.ones(len(rows), bool)}
    groups.update({g: family == g for g in FAMILIES})
    groups.update({f"subset:{k}": np.array([r["name"] in v for r in rows]) for k, v in sorted(subsets.items())})
    table = table or RadiusTable(fit, config.z_range, z_step=config.z_step, omega_bins=config.omega_bins)
    labels = None if strata is None else [np.asarray(strata.labels(p)) for p in thinned]

    def run(moved):
        s_list, spacing, stats, g, pts, etas = _pass(table, moved, config)
        r = np.array([np.nan if st["R"] is None else st["R"] for st in stats])
        frac = fractions(s_list, config.band, grid, config.min_points)
        sc = score(frac, family, calibration, grid, r, groups)
        sc["at_D"] = shares_at_offsets(s_list, etas, family, sc["offsets"], config.min_points)
        if fixed_d is not None:
            sc["fixed"] = score(frac, family, calibration, grid, r, groups, fixed=fixed_d)
        if labels is not None:  # each point keeps the stratum of its unmoved position
            kept = [lab[(p[:, 0] >= z0) & (p[:, 0] < z1)] for lab, p in zip(labels, moved)]
            sc["strata"] = stratum_scores(s_list, kept, strata.names, family, sc["index"], sc["q"], groups,
                                          config.band, grid, strata_min_points)
        return sc, s_list, spacing, stats, g, pts

    def extras(sc):
        return {"fixed": sc["fixed"]["groups"] if "fixed" in sc else None, "strata": sc.get("strata")}

    observed, s_obs, spacing_obs, stats_obs, g_obs, pts = run(thinned)
    exact = exact_positions(fit, pts, g_obs["k"], g_obs["f"], g_obs["r"])
    both = np.isfinite(exact) & np.isfinite(g_obs["s"])
    gap_error = circular_gap(g_obs["s"][both], exact[both])
    median_error = float(np.median(gap_error)) if len(gap_error) else math.inf
    table_ok = median_error <= config.table_tolerance

    rng = np.random.Generator(np.random.PCG64(config.seed))
    z_null, rot_null, z_consistency, z_at_d, z_extra, rot_extra = [], [], [], [], [], []
    for _ in range(config.null_draws):
        moved = [_move_z(p, float(rng.choice((-1.0, 1.0)) * rng.uniform(*config.z_offset)), config.z_range)
                 for p in thinned]
        sc, _, _, st, _, _ = run(moved)
        z_null.append(sc["groups"])
        z_consistency.append(_consistency([x["majority_share"] for x in st]))
        z_at_d.append(_consistency(sc["at_D"]))
        z_extra.append(extras(sc))
    for _ in range(config.rotation_draws):
        sc, *_ = run([_rotate(p, float(rng.uniform(*config.rotation)), fit.umbilicus) for p in thinned])
        rot_null.append(sc["groups"])
        rot_extra.append(extras(sc))

    verdicts = {}
    for name in groups:
        if name == "U":
            continue
        v = verdict(observed["groups"][name], [d[name] for d in z_null], [d[name] for d in rot_null], config,
                    table_ok)
        v["e7v3"] = E7V3[v["verdict"]]
        verdicts[name] = v

    # Ceiling: a perfect front-face fit under fibre noise σ, on the observed points' own spacings
    crng = np.random.Generator(np.random.PCG64(face.ceiling_seed))
    faces = np.where(family == "V", face.face_gap,
                     np.where(family == "H", 0.0, face.face_gap * (crng.uniform(size=len(rows)) < 0.5)))
    r_obs = np.array([np.nan if st["R"] is None else st["R"] for st in stats_obs])
    ceiling = {}
    for sigma in face.sigmas:
        sim = []
        for i, (s, sp) in enumerate(zip(s_obs, spacing_obs)):
            x = (faces[i] + crng.normal(0.0, sigma, len(s))) / sp
            sim.append(np.where(np.isfinite(s), x - np.round(x), np.nan))
        sc = score(fractions(sim, config.band, grid, config.min_points), family, calibration, grid, r_obs, groups)
        ceiling[f"{sigma:g}"] = {k: sc["groups"][k]["Q"] for k in ("all", *SCORED_FAMILIES)}

    spacing_all = np.concatenate(spacing_obs)
    spacing_median = float(np.nanmedian(spacing_all)) if np.isfinite(spacing_all).any() else None
    per_family = {}
    for g in FAMILIES:
        s = np.concatenate([s_obs[i] for i in np.nonzero(family == g)[0]]) if (family == g).any() else np.zeros(0)
        sp = np.concatenate([spacing_obs[i] for i in np.nonzero(family == g)[0]]) if (family == g).any() \
            else np.zeros(0)
        d = observed["offsets"][g]["D"]
        sp_med = float(np.nanmedian(sp)) if np.isfinite(sp).any() else None
        per_family[g] = {"histogram": np.histogram(s[np.isfinite(s)], bins=HISTOGRAM_BINS,
                                                   range=(-0.5, 0.5))[0].astype(int).tolist(),
                         "circular_mean": _circular_mean(s), "spacing_median": sp_med,
                         "D": observed["offsets"][g],
                         "D_voxels": None if d is None or sp_med is None else float(d * sp_med)}

    consistency = _against_null(_consistency([st["majority_share"] for st in stats_obs]), z_consistency)
    at_offset = _against_null(_consistency(observed["at_D"]), z_at_d)
    at_offset["by_family"] = {g: _consistency([v for v, fam in zip(observed["at_D"], family) if fam == g])
                              for g in FAMILIES}

    for i, row in enumerate(rows):
        st = stats_obs[i]
        row.update(points_used=st["points_used"], Q_at_D=None if not np.isfinite(observed["q"][i])
                   else float(observed["q"][i]), R=st["R"], majority_share=st["majority_share"],
                   majority_share_at_D=observed["at_D"][i])
        if fixed_d is not None:
            q0 = observed["fixed"]["q"][i]
            row["Q_at_fixed_D"] = float(q0) if np.isfinite(q0) else None
    if calibration_q:
        report["calibration_q"] = observed["calibration"]
    if fixed_d is not None:
        fx = observed["fixed"]
        fixed_verdicts = {}
        for name in groups:
            if name == "U":
                continue
            v = verdict(fx["groups"][name], [d["fixed"][name] for d in z_extra],
                        [d["fixed"][name] for d in rot_extra], config, table_ok)
            fixed_verdicts[name] = {**v, "e7v3": E7V3[v["verdict"]]}
        report["fixed_d"] = {
            "D": float(fixed_d), "offsets": fx["offsets"], "calibration_q": fx["calibration"],
            "groups": {name: {"observed": fx["groups"][name],
                              "z_offset_null": _quantiles(d["fixed"][name]["Q"] for d in z_extra),
                              "rotation_null": _quantiles(d["fixed"][name]["Q"] for d in rot_extra)}
                       for name in groups},
            "verdicts": fixed_verdicts}
    if labels is not None:
        counts = np.bincount(np.concatenate(labels).astype(np.int64), minlength=len(strata.names))
        report["strata"] = {
            "names": list(strata.names), "min_points": int(strata_min_points),
            "points_thinned": {n: int(counts[k]) for k, n in enumerate(strata.names)},
            "groups": {n: {name: {"observed": observed["strata"][n][name],
                                  "z_offset_null": _quantiles(d["strata"][n][name]["Q"] for d in z_extra),
                                  "rotation_null": _quantiles(d["strata"][n][name]["Q"] for d in rot_extra)}
                           for name in groups} for n in strata.names}}
    report.update(
        split={h: {g: int(((calibration if h == "calibration" else ~calibration) & (family == g)).sum())
                   for g in FAMILIES} for h in ("calibration", "evaluation")},
        points_thinned=int(len(pts)), points_unknown=int((~np.isfinite(g_obs["s"])).sum()),
        table={"points_checked": int(both.sum()), "median_error": median_error,
               "p95_error": float(np.percentile(gap_error, 95)) if len(gap_error) else None, "ok": table_ok},
        spacing_median=spacing_median, offsets=observed["offsets"], families=per_family,
        groups={name: {"observed": observed["groups"][name],
                       "z_offset_null": _quantiles(d[name]["Q"] for d in z_null),
                       "rotation_null": _quantiles(d[name]["Q"] for d in rot_null)} for name in groups},
        verdicts=verdicts, ceiling=ceiling, winding_consistency=consistency,
        winding_consistency_at_offset=at_offset, per_fibre=rows,
        null_series={"z_offset": [d["all"] for d in z_null], "rotation": [d["all"] for d in rot_null]},
    )
    return report


def paired_difference(a: list[dict], b: list[dict], *, draws: int = 10000, seed: int = 4,
                      margin: float = 0.05, field: str = "Q_at_D", half: str | None = "evaluation") -> dict:
    """Fit A's E7 v3 Q minus fit B's, over the evaluation fibres both scored (``per_fibre`` rows
    of two reports on one slab, matched by name and digest), with a fibre-bootstrap one-sided 95% interval.

    CONFIRMED when the difference is at least ``margin`` and its lower bound (5th percentile) is above 0;
    REFUTED when the upper bound (95th percentile) is below 0; INCONCLUSIVE otherwise, and with no shared fibre.
    ``field`` names the per-fibre value compared (e.g. ``majority_share_at_D``); ``half`` the fibres it is read
    on (None: both halves).
    """
    def scored(rows):
        return {(r["name"], r["digest"]): r[field] for r in rows
                if (half is None or r.get("half") == half) and r.get(field) is not None}

    qa, qb = scored(a), scored(b)
    shared = sorted(set(qa) & set(qb))
    out = {"field": field, "half": half, "fibres": len(shared), "only_a": len(set(qa) - set(qb)),
           "only_b": len(set(qb) - set(qa)), "draws": draws, "seed": seed, "margin": margin}
    if not shared:
        return {**out, "difference": None, "lower_95": None, "upper_95": None, "verdict": "INCONCLUSIVE"}
    d = np.array([qa[k] - qb[k] for k in shared])
    rng = np.random.Generator(np.random.PCG64(seed))
    boot = d[rng.integers(0, len(d), (draws, len(d)))].mean(axis=1)
    lo, hi = float(np.percentile(boot, 5)), float(np.percentile(boot, 95))
    diff = float(d.mean())
    verdict = "CONFIRMED" if diff >= margin and lo > 0 else "REFUTED" if hi < 0 else "INCONCLUSIVE"
    return {**out, "difference": diff, "lower_95": lo, "upper_95": hi, "verdict": verdict}


def equivalent(lower: float | None, upper: float | None, margin: float) -> str:
    """An equivalence reading of one 90% interval: CONFIRMED when it lies inside (−margin, margin), REFUTED when it lies
    entirely outside it, INCONCLUSIVE otherwise (and without an interval)."""
    if lower is None or upper is None:
        return "INCONCLUSIVE"
    if -margin < lower and upper < margin:
        return "CONFIRMED"
    if upper <= -margin or lower >= margin:
        return "REFUTED"
    return "INCONCLUSIVE"
