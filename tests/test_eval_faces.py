"""E7 v3 (apok_eval.eval.faces) and the fibre families (apok_eval.eval.fibres.hv_family) on synthetic spirals.

The synthetic scroll breathes: every sheet's radius is 16 ω + A sin(2π z / L), so a fibre moved 128-384 voxels
in z lands at an unrelated place in the gap and the z-offset null is informative.
H fibres are arcs at one z; V fibres run 200 voxels in z at one angle. Nothing here is scroll data.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from apok_eval import jsonstrict
from apok_eval.eval import faces
from apok_eval.eval import fibres as fb
from apok_eval.eval.fibres import Fibre, hv_family, read_fibre
from apok_eval.eval.fit import WindingFit, segment_from_tifxyz
from apok_eval.eval.registration import RadiusTable, RegConfig
from apok_eval.io.tifxyz import write_tifxyz
from apok_eval.io.umbilicus import read_umbilicus
from apok_eval.refusal import Refusal
from eval_fixtures import builders as b

A, L, PHASE, SPACING = 24.0, 512.0, 0.3, 16.0
Z = np.arange(0.0, 1001.0, 20.0)
SLAB = (100.0, 900.0)
CFG = RegConfig(z_range=SLAB, z_step=16.0, omega_bins=256, null_draws=20, band_draws=20, rotation_draws=20,
                points_per_fibre=64)
FACE = faces.FaceConfig()


def radius(w, z):
    return SPACING * np.asarray(w) + A * np.sin(2 * np.pi * np.asarray(z) / L)


@pytest.fixture(scope="module")
def umbilicus(tmp_path_factory):
    return read_umbilicus(b.write_umbilicus_file(tmp_path_factory.mktemp("umb")))


@pytest.fixture(scope="module")
def fit(tmp_path_factory, umbilicus):
    root = tmp_path_factory.mktemp("fit")
    dirs = []
    for lo, hi in ((10, 14), (15, 20)):
        omegas = b.omega_columns(lo, hi + 1, radius=lambda w: SPACING * np.asarray(w))
        zz, ww = np.meshgrid(Z, omegas, indexing="ij")
        zyx = b.spiral_point(ww, zz, phase=PHASE, radius=lambda w: radius(w, zz))
        x, y, z = (zyx[..., i].astype(np.float32) for i in (2, 1, 0))
        name = f"20990101000000-w{lo:03d}-{hi:03d}"
        write_tifxyz(root / name, x=x, y=y, z=z, meta=b._meta(name, x, y, z))
        dirs.append(root / name)
    return WindingFit([segment_from_tifxyz(d, policy="spiral") for d in dirs], umbilicus, z_range=SLAB)


@pytest.fixture(scope="module")
def table(fit):
    return RadiusTable(fit, SLAB, z_step=CFG.z_step, omega_bins=CFG.omega_bins)


def h_points(w, z, offset=0.0, rng=None, noise=0.0):
    p = b.fibre_points(w, w + 0.15, z, offset=offset, phase=PHASE, radius=lambda v: radius(v, z))
    return _jitter(p, rng, noise)


def v_points(w, z0, offset=0.0, rng=None, noise=0.0):
    z = np.arange(z0, z0 + 200.0)
    p = b.spiral_point(np.full(len(z), float(w)), z, phase=PHASE, radius=lambda v: radius(v, z) + offset)
    return _jitter(p, rng, noise)


def _jitter(p, rng, noise):
    """Move each point radially by N(0, noise) voxels."""
    if not noise:
        return p
    d = p[:, 1:] - np.asarray(b.CENTRE)
    unit = d / np.linalg.norm(d, axis=1, keepdims=True)
    out = p.copy()
    out[:, 1:] += unit * rng.normal(0.0, noise, len(p))[:, None]
    return out


def make(points, name, **meta):
    pts = np.asarray(points, dtype=np.float64)
    pts.flags.writeable = False
    return Fibre(name=name, digest=f"sha256:{name}", version=1, optimization_mode="lasagna", points_zyx=pts,
                 control_count=2, line_count=len(pts), trimmed=True, coordinate_scale=0.25,
                 scale_source="default-0.25", coordinate_base_shape_zyx=None, meta=meta)


def planted(h_offset=0.0, v_offset=4.0, noise=0.0, n=30, seed=1):
    """``n`` H and ``n`` V fibres at the given radial offsets (voxels) from the fit's sheets."""
    rng = np.random.default_rng(seed)
    out, fam = [], []
    for k in range(n):
        w = 11.0 + 7.5 * k / n
        out.append(make(h_points(w, 120.0 + 25 * k, h_offset, rng, noise), f"h{k:02d}"))
        fam.append(("H", "test"))
        out.append(make(v_points(w + 0.37, 110.0 + 19 * k, v_offset, rng, noise), f"v{k:02d}"))
        fam.append(("V", "test"))
    return out, fam


def run(fit, table, fibres, fam, **kw):
    return faces.diagnose(fit, fibres, config=CFG, face=FACE, families=fam, table=table, **kw)


# -- families ---------------------------------------------------------------------------------------------------

def test_hv_family_follows_villas_precedence():
    v = make(v_points(12.0, 200.0), "v")
    h = make(h_points(12.0, 300.0), "h")
    assert hv_family(v) == ("V", "geometry") and hv_family(h) == ("H", "geometry")
    tagged = dataclasses.replace(v, meta={"hv_classification": {"manual_tag": " h ", "automatic_tag": "V",
                                                                "automatic_certainty": 1.0}})
    assert hv_family(tagged) == ("H", "manual")
    auto = dataclasses.replace(v, meta={"hv_classification": {"automatic_tag": "h", "automatic_certainty": 0.5}})
    assert hv_family(auto) == ("H", "automatic")
    weak = dataclasses.replace(v, meta={"hv_classification": {"automatic_tag": "H", "automatic_certainty": 0.49}})
    assert hv_family(weak) == ("V", "geometry")
    junk = dataclasses.replace(v, meta={"hv_classification": {"automatic_tag": "H", "automatic_certainty": "x"}})
    assert hv_family(junk) == ("V", "geometry")
    assert hv_family(dataclasses.replace(v, meta={"hv_classification": "V"})) == ("V", "geometry")
    assert hv_family(dataclasses.replace(v, meta={"hv_classification": {"manual_tag": "X"}})) == ("V", "geometry")


def test_hv_family_geometry_thresholds_are_inclusive():
    # 3-4-5 triangles: the z extent is exactly 0.8 (V) or 0.2 (H) of the path length
    assert hv_family(make([[0, 0, 0], [80, 60, 0]], "v")) == ("V", "geometry")
    # villa compares with 1 - 0.8, which is just under 0.2 in floating point: exactly 0.2 stays untagged
    assert hv_family(make([[0, 0, 0], [20, 0, 0], [20, 80, 0]], "u"), spacing=0.0) == ("U", "geometry")
    assert hv_family(make([[0, 0, 0], [19, 0, 0], [19, 81, 0]], "h"), spacing=0.0) == ("H", "geometry")
    assert hv_family(make([[0, 0, 0], [50, 50, 50]], "u")) == ("U", "geometry")
    assert hv_family(make([[0, 0, 0], [10, 0, 0]], "short")) == ("U", "none")  # decimated to one point
    assert hv_family(make([[5, 5, 5], [5, 5, 5], [5, 5, 5]], "still"), spacing=0.0) == ("U", "none")


def test_decimation_is_villas_and_read_fibre_keeps_the_tags():
    p = np.cumsum(np.random.default_rng(0).normal(0, 3, (500, 3)), axis=0)
    keep, last = [0], p[0]
    for i in range(1, len(p)):
        if np.linalg.norm(p[i] - last) >= 7.0:
            keep.append(i)
            last = p[i]
    assert np.array_equal(fb.decimate(p, 7.0), p[keep]) and fb.decimate(p, 0.0) is p
    hv = {"manual_tag": "V", "automatic_tag": "H", "automatic_certainty": 0.9}
    f = read_fibre(b.fibre_bytes(h_points(12.0, 300.0), extra={"hv_classification": hv}), name="t",
                   base_shape_zyx=b.FRAME)
    assert f.meta["hv_classification"] == hv and hv_family(f) == ("V", "manual")


# -- the procedure ------------------------------------------------------------------------------------------------

def test_split_is_a_pinned_hash_of_the_name():
    names = [f"fibre-{i:03d}.json" for i in range(400)]
    cal = [faces.in_calibration(n, FACE.split_seed) for n in names]
    assert FACE.split_seed == 2 and 160 < sum(cal) < 240
    assert [faces.in_calibration(n, 2) for n in ("a", "b", "c", "d")] == [True, True, False, True]
    assert cal != [faces.in_calibration(n, 1) for n in names]


def test_fractions_wrap_around_the_gap_and_need_min_points():
    grid = FACE.grid()
    assert len(grid) == 100 and grid[0] == -0.5 and grid[50] == 0.0
    s = [np.array([0.49, -0.49, 0.45, 0.0]), np.array([0.1, np.nan]), np.array([np.nan] * 5)]
    f = faces.fractions(s, 0.125, grid, 2)
    assert f[0, 0] == pytest.approx(0.75) and f[50, 0] == pytest.approx(0.25)
    assert np.isnan(f[:, 1]).all() and np.isnan(f[:, 2]).all()


def test_best_offset_breaks_ties_toward_zero_then_lower():
    grid = FACE.grid()
    frac = np.zeros((100, 1))
    frac[[40, 60], 0] = 1.0
    assert faces.best_offset(frac, grid) == 40
    frac[50, 0] = 1.0
    assert faces.best_offset(frac, grid) == 50
    assert faces.best_offset(np.zeros((100, 0)), grid) is None
    frac = np.zeros((100, 1))
    frac[60:64, 0] = 1.0  # a flat top: its centre (the lower middle), not its edge nearest 0
    frac[10, 0] = 1.0
    assert faces.best_offset(frac, grid) == 61
    frac = np.zeros((100, 1))
    frac[[98, 99, 0, 1, 2], 0] = 1.0  # a run across the circle's seam
    assert faces.best_offset(frac, grid) == 0
    assert faces.best_offset(np.ones((100, 1)), grid) == 50


def test_settings_and_families_refuse():
    for bad in ({"offset_step": 0.03}, {"offset_step": 0.0}, {"sigmas": ()}, {"sigmas": (0.0,)},
                {"face_gap": -1.0}):
        with pytest.raises(Refusal) as r:
            faces.FaceConfig(**bad).check()
        assert r.value.code == "EVAL_FACE_PARAMETER"
    fibres, fam = planted(n=2)
    with pytest.raises(Refusal) as r:
        faces.diagnose(None, fibres, config=CFG, families=fam[:-1])
    assert r.value.code == "EVAL_FACE_FAMILIES"
    with pytest.raises(Refusal) as r:
        faces.diagnose(None, fibres, config=CFG, families=[("X", "t")] * len(fibres))
    assert r.value.code == "EVAL_FACE_FAMILIES"


# -- the method's fixtures ---------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def front_face(fit, table):
    fibres, fam = planted()
    config = dataclasses.replace(CFG, null_draws=100)  # the default count: the null's median is stable
    return faces.diagnose(fit, fibres, config=config, face=FACE, families=fam, table=table,
                          subsets={"first": {f.name for f in fibres[:40]}})


def test_a_front_face_fit_passes_in_both_families(front_face):
    rep = front_face
    v = rep["verdicts"]
    assert {k: v[k]["e7v3"] for k in ("all", "H", "V")} == {"all": "PASS", "H": "PASS", "V": "PASS"}
    assert v["all"]["null"] == "z-offset" and not v["all"]["z_null_uninformative"]
    assert rep["offsets"]["H"]["D"] == pytest.approx(0.0, abs=0.02)
    assert rep["offsets"]["V"]["D"] == pytest.approx(0.25, abs=0.02)  # 4 voxels out of 16
    assert rep["families"]["V"]["D_voxels"] == pytest.approx(4.0, abs=0.5)
    assert rep["groups"]["all"]["observed"]["Q"] > 0.95 and rep["table"]["ok"]
    assert "U" not in v and v["subset:first"]["e7v3"] == "PASS"
    assert rep["groups"]["subset:first"]["observed"]["fibres"] < rep["groups"]["all"]["observed"]["fibres"]
    assert rep["winding_consistency"]["median"] == pytest.approx(1.0)
    jsonstrict.dumps(rep)


def test_the_null_stays_near_a_quarter(front_face):
    """Calibration on the null's own calibration half does not lift the evaluation half's Q (under 0.27).

    A precise fibre's null Q is nearly 0 or 1, so one family's draws scatter widely; their mean is the stable
    figure. The calibrated rotation null sits near ½ on a perfect spiral (every fibre moves ¼-¾ of a gap) and
    is reported, not used, while the z-offset null is informative."""
    groups = front_face["groups"]
    assert 0.20 < groups["all"]["z_offset_null"]["median"] < 0.27
    for name in ("all", "H", "V"):
        assert 0.20 < groups[name]["z_offset_null"]["mean"] < 0.27
    assert groups["all"]["rotation_null"]["median"] > 0.35


@pytest.mark.parametrize("shift", [3.0, -3.0])
def test_a_fit_shifted_three_voxels_still_passes(fit, table, shift):
    fibres, fam = planted(h_offset=shift, v_offset=4.0 + shift)
    rep = run(fit, table, fibres, fam)
    assert {k: rep["verdicts"][k]["e7v3"] for k in ("all", "H", "V")} == {"all": "PASS", "H": "PASS", "V": "PASS"}
    assert rep["offsets"]["H"]["D"] == pytest.approx(shift / SPACING, abs=0.02)


def test_scatter_of_half_the_gap_fails(fit, table):
    fibres, fam = planted(noise=SPACING / 2)
    rep = run(fit, table, fibres, fam)
    assert {k: rep["verdicts"][k]["e7v3"] for k in ("all", "H", "V")} == {"all": "FAIL", "H": "FAIL", "V": "FAIL"}
    assert rep["groups"]["all"]["observed"]["Q"] < 0.30


def test_one_offset_for_both_families_would_fail_v(fit, table):
    """One offset for everyone: the same fibres scored at D = 0 put V fibres outside the band."""
    fibres, fam = planted()
    rep = run(fit, table, fibres, [("H", "t")] * len(fibres))
    assert rep["groups"]["all"]["observed"]["Q"] < 0.75


def test_calibration_never_sees_the_evaluation_half(fit, table):
    """Calibration fibres sit on the sheets, evaluation fibres half a gap out: the learnt D cannot reach them."""
    fibres, fam = planted(v_offset=0.0)
    moved = [make(v_points(11.0 + 7.5 * k / 30 + 0.37, 110.0 + 19 * k, 0.0 if faces.in_calibration(
        f"v{k:02d}", FACE.split_seed) else SPACING / 2), f"v{k:02d}") for k in range(30)]
    fibres = [f for f in fibres if f.name.startswith("h")] + moved
    fam = [("H", "t")] * 30 + [("V", "t")] * 30
    rep = run(fit, table, fibres, fam)
    assert rep["offsets"]["V"]["D"] == pytest.approx(0.0, abs=0.02)
    assert rep["groups"]["V"]["observed"]["Q"] < 0.05 and rep["verdicts"]["V"]["e7v3"] == "FAIL"


def test_a_family_without_calibration_fibres_takes_the_pooled_offset(fit, table):
    fibres, fam = planted()
    lone = [i for i, f in enumerate(fibres) if fam[i][0] == "V" and not faces.in_calibration(f.name, FACE.split_seed)]
    fam = [("U", "t") if i in lone[:5] else x for i, x in enumerate(fam)]
    rep = run(fit, table, fibres, fam)
    assert rep["offsets"]["U"]["fallback"] and rep["offsets"]["U"]["calibration_fibres"] == 0
    assert rep["offsets"]["U"]["D"] is not None and rep["groups"]["U"]["observed"]["fibres"] == 5


def test_the_ceiling_matches_gaussian_noise(front_face):
    expect = {s: math.erf(SPACING / 8 / (s * math.sqrt(2))) for s in (1.0, 2.0, 3.0)}
    for s, q in expect.items():
        got = front_face["ceiling"][f"{s:g}"]
        assert got["all"] == pytest.approx(q, abs=0.04) and got["V"] == pytest.approx(q, abs=0.06)


def test_no_fibre_in_the_slab_is_not_measured(fit, table):
    rep = run(fit, table, [make(h_points(12.0, 950.0), "far")], [("H", "t")])
    assert rep["fibres_in_slab"] == 0 and rep["verdicts"]["all"]["e7v3"] == "NOT MEASURED"


def test_default_families_come_from_hv_family(fit, table):
    fibres, _ = planted(n=12)
    rep = faces.diagnose(fit, fibres, config=CFG, face=FACE, table=table)
    assert rep["families_in_slab"] == {"H": 12, "V": 12, "U": 0} and rep["family_sources"] == {"geometry": 24}


def test_paired_difference_pairs_fibres_by_name_and_digest():
    def rows(qs, half="evaluation"):
        return [{"name": f"f{i}", "digest": f"d{i}", "half": half, "Q_at_D": q} for i, q in enumerate(qs)]

    better = faces.paired_difference(rows([0.5] * 30), rows([0.3] * 30))
    assert better["verdict"] == "CONFIRMED" and better["difference"] == pytest.approx(0.2) and better["fibres"] == 30
    worse = faces.paired_difference(rows([0.2] * 30), rows([0.3] * 30))
    assert worse["verdict"] == "REFUTED" and worse["upper_95"] < 0
    small = faces.paired_difference(rows([0.33] * 30), rows([0.3] * 30))
    assert small["verdict"] == "INCONCLUSIVE" and small["lower_95"] > 0  # above 0 but under the 0.05 margin
    noisy = faces.paired_difference(rows([0.9, 0.0] * 15), rows([0.4] * 30))
    assert noisy["verdict"] == "INCONCLUSIVE" and noisy["lower_95"] < 0 < noisy["upper_95"]
    renamed = rows([0.5] * 30)
    renamed[0]["digest"] = "other"
    part = faces.paired_difference(renamed + rows([None], "evaluation"), rows([0.3] * 30) + rows([0.9], "calibration"))
    assert part["fibres"] == 29 and part["only_a"] == 1 and part["only_b"] == 1
    assert faces.paired_difference([], rows([0.3]))["verdict"] == "INCONCLUSIVE"


# -- winding identity at the family offset -----------------------------------------------------------------

def test_share_at_offset_rounds_at_d_plus_or_minus_a_half():
    eta = np.array([0.4, 0.45, 0.6, 0.55, 0.3, np.nan])
    s = np.array([0.4, 0.45, -0.4, -0.45, 0.3, 0.1])
    assert faces.share_at_offset(s, eta, 0.0, 3) == pytest.approx(0.6)  # boundary at 0.5 splits 3 / 2
    assert faces.share_at_offset(s, eta, 0.2, 3) == pytest.approx(1.0)  # boundaries at -0.3 and 0.7
    assert faces.share_at_offset(s, eta + 7.0, 0.2, 3) == pytest.approx(1.0)
    assert faces.share_at_offset(s, eta, -0.12, 3) == pytest.approx(0.8)  # boundary at 0.38: only 0.3 below
    assert faces.share_at_offset(s, eta, 0.2, 6) is None  # five usable points
    assert faces.share_at_offset(s, eta, None, 1) is None
    s_unknown = s.copy()
    s_unknown[0] = np.nan  # a point with an unknown gap position is not used, as in the majority share
    assert faces.share_at_offset(s_unknown, eta, 0.0, 3) == pytest.approx(0.5)


def test_winding_identity_at_the_offset_does_not_count_boundary_switches(fit, table):
    # V fibres half a gap out (7.5 of 16 voxels) with 1.5 voxels of noise straddle the fit-sheet rounding boundary
    fibres, fam = planted(h_offset=0.0, v_offset=7.5, noise=1.5, n=24, seed=4)
    rep = run(fit, table, fibres, fam)
    at_d = rep["winding_consistency_at_offset"]
    assert rep["offsets"]["V"]["D"] == pytest.approx(0.47, abs=0.05)
    assert at_d["by_family"]["V"]["median"] > 0.95 and at_d["by_family"]["H"]["median"] > 0.95
    v_rows = [r for r in rep["per_fibre"] if r["family"] == "V" and r["majority_share"] is not None]
    assert len(v_rows) >= 20
    assert np.median([r["majority_share"] for r in v_rows]) < 0.85  # the fit-sheet share overcounts switches
    assert all(r["majority_share_at_D"] >= r["majority_share"] - 1e-12 for r in v_rows)
    assert at_d["fibres"] == sum(r["majority_share_at_D"] is not None for r in rep["per_fibre"])
    assert at_d["mean"] == pytest.approx(np.mean([r["majority_share_at_D"] for r in rep["per_fibre"]
                                                  if r["majority_share_at_D"] is not None]))
    assert at_d["median"] > rep["winding_consistency"]["median"]
    null = at_d["z_offset_null"]
    assert null["mean"]["n"] == CFG.null_draws and null["mean"]["max"] < at_d["mean"]
    assert at_d["null_draws_at_or_above"] == {"median": 0, "mean": 0, "at_least_0.9": 0}
    assert {"median", "mean", "at_least_0.9"} <= set(rep["winding_consistency"]["z_offset_null"])
    jsonstrict.dumps(rep)


def test_each_null_draw_rounds_at_the_offsets_it_calibrated(fit, table, monkeypatch):
    calibrated, used = [], []
    score, shares = faces.score, faces.shares_at_offsets

    def spy_score(*a, **k):
        out = score(*a, **k)
        calibrated.append(out["offsets"])
        return out

    def spy_shares(s_list, etas, family, offsets, min_points):
        used.append(offsets)
        return shares(s_list, etas, family, offsets, min_points)

    monkeypatch.setattr(faces, "score", spy_score)
    monkeypatch.setattr(faces, "shares_at_offsets", spy_shares)
    fibres, fam = planted(v_offset=6.0, noise=1.0, n=12, seed=5)
    run(fit, table, fibres, fam)
    passes = 1 + CFG.null_draws + CFG.rotation_draws
    assert len(used) == passes and len(calibrated) >= passes
    assert all(u is c for u, c in zip(used, calibrated[:passes]))
    assert len({round(u["V"]["D"], 6) for u in used}) > 1  # the draws did recalibrate


def test_a_family_without_d_has_no_share_at_offset():
    offsets = {"H": {"D": 0.0}, "V": {"D": None}, "U": {"D": 0.2}}
    s = [np.zeros(5), np.zeros(5), np.full(5, 0.3)]
    eta = [np.zeros(5), np.zeros(5), np.full(5, 0.3)]
    assert faces.shares_at_offsets(s, eta, np.array(["H", "V", "U"]), offsets, 3) == [1.0, None, 1.0]


def test_paired_difference_compares_shares_at_the_offset_on_either_half():
    def rows(vals, half="evaluation"):
        return [{"name": f"f{i}", "digest": f"d{i}", "half": half, "majority_share_at_D": v, "Q_at_D": 0.1}
                for i, v in enumerate(vals)]

    a = rows([0.8] * 20) + [{**r, "name": "c" + r["name"]} for r in rows([0.9] * 10, "calibration")]
    b = rows([0.7] * 20) + [{**r, "name": "c" + r["name"]} for r in rows([0.2] * 10, "calibration")]
    ev = faces.paired_difference(a, b, field="majority_share_at_D")
    assert ev["fibres"] == 20 and ev["difference"] == pytest.approx(0.1) and ev["field"] == "majority_share_at_D"
    both = faces.paired_difference(a, b, field="majority_share_at_D", half=None)
    assert both["fibres"] == 30 and both["difference"] == pytest.approx((20 * 0.1 + 10 * 0.7) / 30)
    assert faces.paired_difference(a, b)["difference"] == pytest.approx(0.0)  # Q_at_D, the default field


@pytest.mark.parametrize(("lo", "hi", "verdict"), [
    (-0.04, 0.04, "CONFIRMED"), (-0.05, 0.0, "INCONCLUSIVE"), (0.0, 0.05, "INCONCLUSIVE"),
    (0.05, 0.08, "REFUTED"), (-0.09, -0.05, "REFUTED"), (0.049, 0.2, "INCONCLUSIVE"), (-0.2, 0.2, "INCONCLUSIVE"),
    (None, 0.0, "INCONCLUSIVE"), (0.0, None, "INCONCLUSIVE")])
def test_equivalent_reads_an_interval_against_the_margin(lo, hi, verdict):
    assert faces.equivalent(lo, hi, 0.05) == verdict


def test_consistency_summary_reports_mean_median_and_high_share():
    c = faces._consistency([0.5, None, 0.6, 1.0, 0.95])
    assert c == {"fibres": 4, "median": pytest.approx(0.775), "mean": pytest.approx(0.7625),
                 "at_least_0.9": pytest.approx(0.5)}
    assert faces._consistency([None]) == {"fibres": 0, "median": None, "mean": None, "at_least_0.9": None}
