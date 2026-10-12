"""The registration diagnostic (apok_eval.eval.registration) on synthetic spirals with planted fibres."""

from __future__ import annotations

import math

import numpy as np
import pytest

from apok_eval import jsonstrict
from apok_eval.eval import registration as reg
from apok_eval.eval.fibres import read_fibre
from apok_eval.eval.fit import OuterShell, WindingFit, segment_from_tifxyz
from apok_eval.eval.geometry import cylindrical
from apok_eval.io.umbilicus import read_umbilicus
from apok_eval.refusal import Refusal
from eval_fixtures import builders as b

Z = np.arange(20.0, 401.0, 20.0)
SLAB = (150.0, 250.0)
CFG = reg.RegConfig(z_range=SLAB, z_step=10.0, omega_bins=256, null_draws=20, band_draws=20, rotation_draws=20,
                    z_offset=(20.0, 40.0), bands=((2.0, 4.0), (8.0, 16.0)), points_per_fibre=64)


@pytest.fixture(scope="module")
def umbilicus(tmp_path_factory):
    return read_umbilicus(b.write_umbilicus_file(tmp_path_factory.mktemp("umb")))


@pytest.fixture(scope="module")
def fit(tmp_path_factory, umbilicus):
    dirs = b.write_fit(tmp_path_factory.mktemp("fit"), [(10, 14), (15, 20)], z_levels=Z, phase=0.3)
    return WindingFit([segment_from_tifxyz(d, policy="spiral") for d in dirs], umbilicus, z_range=SLAB)


@pytest.fixture(scope="module")
def table(fit):
    return reg.RadiusTable(fit, SLAB, z_step=CFG.z_step, omega_bins=CFG.omega_bins)


def fibre(points, name, **kw):
    return read_fibre(b.fibre_bytes(points, **kw), name=name, base_shape_zyx=b.FRAME)


def fibres_at(offset, *, n=12, jitter=None, domain_every=0):
    """n fibres a fifth of a turn long, at ``offset`` voxels outside a sheet (or jittered per point)."""
    rng = np.random.default_rng(7)
    out = []
    for k in range(n):
        w = 11.0 + 0.6 * k
        if jitter is None:
            pts = b.fibre_points(w, w + 0.2, 170.0 + 6 * k, offset=offset, phase=0.3)
        else:
            pts = b.fibre_points(w, w + 0.2, 170.0 + 6 * k, phase=0.3,
                                 radius=lambda v: 16.0 * np.asarray(v) + rng.uniform(-jitter, jitter, np.shape(v)))
        kw = {"domain": [4000, 4000, 4000]} if domain_every and k % domain_every == 0 else {}
        out.append(fibre(pts, f"f{k:02d}", **kw))
    return out


def refuses(code, fn, *a, **kw):
    with pytest.raises(Refusal) as info:
        fn(*a, **kw)
    assert info.value.code == code, info.value.as_dict()
    return info.value


# --- configuration -----------------------------------------------------------------------------------

@pytest.mark.parametrize("change", [
    {"z_range": (250.0, 150.0)}, {"band": 0.5}, {"band": 0.0}, {"points_per_fibre": 1}, {"min_points": 1},
    {"min_fibres": 0}, {"z_step": 0.0}, {"z_step": 1000.0}, {"omega_bins": 4}, {"null_draws": 10},
    {"band_draws": 19}, {"rotation_draws": 5}, {"z_offset": (40.0, 20.0)}, {"bands": ((-1.0, 2.0),)},
    {"rotation": (2.0, 1.0)}, {"rotation": (0.0, 7.0)}, {"registered_q": 0.2}, {"not_registered_q": 0.0},
    {"registered_q": 1.0}, {"uninformative_null_q": 1.0}, {"offset_margin": 0.0}, {"table_tolerance": 0.0},
])
def test_settings_out_of_range_refuse(change):
    bad = reg.RegConfig(**{**{k: getattr(CFG, k) for k in ("z_range", "z_step")}, **change})
    r = refuses("EVAL_REG_PARAMETER", bad.check)
    assert r.details["problems"]


def test_defaults_are_the_fixed_rule_and_the_slab_is_always_given():
    c = reg.RegConfig(z_range=SLAB)
    c.check()
    assert (c.band, c.registered_q, c.not_registered_q, c.uninformative_null_q) == (0.125, 0.40, 0.30, 0.35)
    assert (c.null_draws, c.z_offset, c.rotation_draws, c.seed) == (100, (128.0, 384.0), 100, 1)
    with pytest.raises(TypeError):
        reg.RegConfig()  # no default slab: the slab is a property of the data, never assumed


# --- gap positions -----------------------------------------------------------------------------------

def test_points_on_a_sheet_sit_at_zero_and_offsets_read_as_fractions_of_the_gap(table):
    on = reg.gap_positions(table, b.fibre_points(12.0, 12.4, 200.0, phase=0.3))
    assert np.nanmax(np.abs(on["s"])) < 0.03 and np.isfinite(on["s"]).all()
    quarter = reg.gap_positions(table, b.fibre_points(12.0, 12.4, 200.0, phase=0.3, offset=4.0))
    assert np.allclose(quarter["u"], 0.25, atol=0.03) and np.allclose(quarter["spacing"], 16.0, atol=0.4)
    assert np.allclose(quarter["omega"], quarter["k"] + quarter["f"] + quarter["u"])
    inside_out = reg.gap_positions(table, b.fibre_points(12.0, 12.4, 200.0, phase=0.3, offset=-6.0))
    assert np.allclose(inside_out["s"], -6.0 / 16.0, atol=0.03)  # just inside a sheet: negative


def test_points_beyond_the_fit_or_beside_a_hole_are_unknown(fit, table, tmp_path, umbilicus):
    inner = reg.gap_positions(table, b.fibre_points(10.0, 10.3, 200.0, phase=0.3, offset=-40.0))
    outer = reg.gap_positions(table, b.fibre_points(20.0, 20.3, 200.0, phase=0.3, offset=40.0))
    assert np.isnan(inner["s"]).all() and np.isnan(outer["s"]).all()
    holes = [(slice(None), slice(40, 80))]
    dirs = b.write_fit(tmp_path, [(10, 14)], z_levels=Z, phase=0.3, holes=holes)
    holed = WindingFit([segment_from_tifxyz(d, policy="spiral") for d in dirs], umbilicus, z_range=SLAB)
    t = reg.RadiusTable(holed, SLAB, z_step=10.0, omega_bins=256)
    cols = b.omega_columns(10, 15, radius=lambda w: 16.0 * np.asarray(w))
    lo, hi = cols[40], cols[79]
    g = reg.gap_positions(t, b.fibre_points(lo + 0.02, hi - 0.02, 200.0, phase=0.3, offset=3.0))
    assert np.isnan(g["s"]).mean() > 0.9


def test_the_table_agrees_with_the_exact_crossings(fit, table):
    pts = np.concatenate([b.fibre_points(w, w + 0.3, z, phase=0.3, offset=o)
                          for w, z, o in ((11.0, 160.0, 3.0), (13.5, 240.0, -5.0), (16.2, 200.0, 7.0))])
    g = reg.gap_positions(table, pts)
    exact = reg.exact_positions(fit, pts, g["k"], g["f"], g["r"])
    assert np.isfinite(exact).all()
    assert np.median(reg.circular_gap(g["s"], exact)) < 0.01
    none = reg.exact_positions(fit, pts[:3], np.full(3, np.nan), g["f"][:3], g["r"][:3])
    assert np.isnan(none).all()


def test_circular_gap_wraps():
    assert np.allclose(reg.circular_gap(np.array([0.45, -0.45, 0.1]), np.array([-0.45, 0.45, 0.3])), [0.1, 0.1, 0.2])


# --- per-fibre statistics ------------------------------------------------------------------------------

def test_fibre_stats_majority_and_offset():
    eta = np.full(50, 12.3)
    st = reg.fibre_stats(eta - np.round(eta), eta, 0.125, 20)
    assert st["R"] == pytest.approx(1.0) and st["c"] == pytest.approx(0.3) and st["Q"] == 0.0
    assert st["majority_share"] == 1.0 and st["majority_Q"] == 0.0
    mixed = np.concatenate([np.full(30, 12.02), np.full(20, 13.01)])
    st = reg.fibre_stats(mixed - np.round(mixed), mixed, 0.125, 20)
    assert st["Q"] == 1.0 and st["majority_share"] == 0.6 and st["majority_Q"] == 0.6
    few = reg.fibre_stats(np.zeros(5), np.zeros(5), 0.125, 20)
    assert few["Q"] is None and few["points_used"] == 5


# --- verdict -----------------------------------------------------------------------------------------

def null(q, r=0.3):
    return [{"fibres": 12, "Q": v, "R": r} for v in q]


@pytest.mark.parametrize("q, z_null, rot_null, expected, which", [
    (0.80, null(np.linspace(0.20, 0.30, 20)), null([0.9] * 20), "REGISTERED", "z-offset"),
    (0.80, null(np.linspace(0.70, 0.90, 20)), null(np.linspace(0.2, 0.3, 20)), "REGISTERED", "rotation"),
    (0.80, null(np.linspace(0.70, 0.90, 20)), null(np.linspace(0.7, 0.9, 20)), "NOT REGISTERED", "rotation"),
    (0.29, null([0.10] * 20), null([0.1] * 20), "NOT REGISTERED", "z-offset"),
    (0.38, null(np.linspace(0.20, 0.30, 20)), null([0.1] * 20), "INCONCLUSIVE", "z-offset"),
    (0.45, null(list(np.linspace(0.20, 0.30, 19)) + [0.5]), null([0.1] * 20), "INCONCLUSIVE", "z-offset"),
    (0.31, null(list(np.linspace(0.20, 0.30, 18)) + [0.32, 0.33]), null([0.1] * 20), "NOT REGISTERED",
     "z-offset"),
])
def test_the_verdict_follows_the_rule(q, z_null, rot_null, expected, which):
    v = reg.verdict({"fibres": 12, "Q": q, "R": 0.5}, z_null, rot_null, reg.RegConfig(z_range=SLAB))
    assert (v["verdict"], v["null"]) == (expected, which)
    assert v["z_null_uninformative"] == (which == "rotation")


def test_offset_consistency_needs_r_above_the_null():
    z = null(np.linspace(0.20, 0.30, 20), r=0.4)
    v = reg.verdict({"fibres": 12, "Q": 0.1, "R": 0.9}, z, null([0.1] * 20), reg.RegConfig(z_range=SLAB))
    assert v["verdict"] == "NOT REGISTERED" and v["offset_consistent"] is True
    v = reg.verdict({"fibres": 12, "Q": 0.1, "R": 0.45}, z, null([0.1] * 20), reg.RegConfig(z_range=SLAB))
    assert v["offset_consistent"] is False  # above the null, but not by the margin
    zr = [{"fibres": 12, "Q": 0.25, "R": 0.4 if k else 0.95} for k in range(20)]
    v = reg.verdict({"fibres": 12, "Q": 0.1, "R": 0.9}, zr, null([0.1] * 20), reg.RegConfig(z_range=SLAB))
    assert v["offset_consistent"] is False  # one null draw holds as steady a place


def test_too_few_fibres_or_a_bad_table():
    assert reg.verdict({"fibres": 9, "Q": 0.9, "R": 1.0}, null([0.2] * 20), null([0.2] * 20),
                       reg.RegConfig(z_range=SLAB))["verdict"] == "NOT MEASURED"
    v = reg.verdict({"fibres": 12, "Q": 0.9, "R": 1.0}, null([0.2] * 20), null([0.2] * 20), reg.RegConfig(z_range=SLAB),
                    table_ok=False)
    assert v["verdict"] == "INCONCLUSIVE" and v["table_ok"] is False


# --- movers ------------------------------------------------------------------------------------------

def test_a_z_move_keeps_the_fibre_in_the_slab_and_a_rotation_keeps_its_radius(umbilicus):
    pts = b.fibre_points(12.0, 12.2, 240.0, phase=0.3)
    moved = reg._move_z(pts, 30.0, SLAB)  # +30 would leave the slab: the sign flips
    assert np.allclose(moved[:, 0], 210.0) and np.array_equal(moved[:, 1:], pts[:, 1:])
    kept = reg._move_z(pts, -30.0, SLAB)
    assert np.allclose(kept[:, 0], 210.0)
    turned = reg._rotate(pts, math.pi / 2, umbilicus)
    centre = umbilicus.yx_at(pts[:, 0], out_of_range="extrapolate").yx
    r0, p0 = cylindrical(pts, centre)
    r1, p1 = cylindrical(turned, centre)
    assert np.allclose(r0, r1) and np.allclose(np.mod(p1 - p0, 2 * np.pi), math.pi / 2)


def test_thinning_keeps_the_ends_and_at_most_the_limit():
    pts = np.arange(300.0).reshape(100, 3)
    thin = reg._thin(pts, 10)
    assert len(thin) == 10 and np.array_equal(thin[0], pts[0]) and np.array_equal(thin[-1], pts[-1])
    assert reg._thin(pts[:5], 10) is not None and len(reg._thin(pts[:5], 10)) == 5


# --- end to end --------------------------------------------------------------------------------------

def test_fibres_on_the_sheets_are_registered_against_the_rotation_null(fit, table):
    rep = reg.diagnose(fit, fibres_at(0.0, domain_every=3), config=CFG, table=table)
    v = rep["verdicts"]["all"]
    assert v["verdict"] == "REGISTERED" and v["null"] == "rotation" and v["z_null_uninformative"]
    assert rep["groups"]["all"]["observed"]["Q"] > 0.95 and rep["table"]["ok"]
    assert set(rep["verdicts"]) == {"all", "scale:coordinate_base_shape_zyx", "scale:default-0.25"}
    assert rep["verdicts"]["scale:coordinate_base_shape_zyx"]["verdict"] == "NOT MEASURED"  # 4 fibres
    row = rep["per_fibre"][0]
    assert row["majority_share"] == 1.0 and row["R"] > 0.99 and row["scale_source"] == "coordinate_base_shape_zyx"
    assert len(rep["null_series"]["z_offset"]) == CFG.null_draws
    assert set(rep["groups"]["all"]["z_bands"]) == {"2-4", "8-16"}
    assert sum(rep["histograms"]["observed"][9:11]) == rep["points_thinned"] - rep["points_unknown"]


def test_fibres_spread_through_the_gap_are_not_registered(fit, table):
    rep = reg.diagnose(fit, fibres_at(0.0, jitter=8.0), config=CFG, table=table)
    v = rep["verdicts"]["all"]
    assert v["verdict"] == "NOT REGISTERED" and 0.18 < v["Q"] < 0.32 and not v["offset_consistent"]


def test_fibres_halfway_between_sheets_are_not_registered(fit, table):
    rep = reg.diagnose(fit, fibres_at(8.0), config=CFG, table=table)
    v = rep["verdicts"]["all"]
    assert v["verdict"] == "NOT REGISTERED" and v["Q"] < 0.05
    assert all(abs(abs(row["c"]) - 0.5) < 0.05 for row in rep["per_fibre"])


def test_the_report_is_json_deterministic_and_names_strata(fit, table, umbilicus, tmp_path):
    shell_dir = b.write_shell(tmp_path, 21 * 16.0 + 30, z_levels=Z)
    s = segment_from_tifxyz(shell_dir, policy="spiral")
    shell = OuterShell(s.zyx, s.valid, umbilicus)
    fibres = fibres_at(0.0)
    one = reg.diagnose(fit, fibres, config=CFG, table=table, shell=shell)
    two = reg.diagnose(fit, fibres, config=CFG, table=table, shell=shell)
    assert jsonstrict.dumps(one) == jsonstrict.dumps(two)
    assert {r["tercile"] for r in one["per_fibre"]} <= {"inner", "middle", "outer"}
    assert "tercile:inner" in one["groups"] and "tercile:inner" not in one["verdicts"]
    assert one["config"]["bands"] == [[2.0, 4.0], [8.0, 16.0]]


def test_no_fibre_in_the_slab_is_not_measured(fit, table):
    far = [fibre(b.fibre_points(12.0, 12.2, 60.0, phase=0.3), "low")]
    rep = reg.diagnose(fit, far, config=CFG, table=table)
    assert rep["fibres_in_slab"] == 0 and rep["verdicts"]["all"]["verdict"] == "NOT MEASURED"


def test_the_windowed_bracket_matches_the_full_search_and_falls_back(table):
    import copy

    rng = np.random.default_rng(3)
    n = 400
    w, z, off = rng.uniform(10.2, 20.8, n), rng.uniform(150, 250, n), rng.uniform(-7.5, 7.5, n)
    pts = np.stack([b.spiral_point(w[i], z[i], phase=0.3, radius=lambda v, o=off[i]: 16.0 * v + o)
                    for i in range(n)])
    g = reg.gap_positions(table, pts)
    count = table.hi + 1 - table.lo
    every = np.broadcast_to(np.arange(count), (n, count))
    k, r_in, r_out, ok = reg._pair(every, table.sheets(g["f"], pts[:, 0]), g["r"])
    assert ok.mean() > 0.9 and np.array_equal(np.where(ok, table.lo + k, np.nan), g["k"], equal_nan=True)
    broken = copy.copy(table)
    broken._search = np.zeros_like(table._search)  # every hint lands on the outermost sheet: all fall back
    assert np.array_equal(broken.bracket(g["f"], pts[:, 0], g["r"])[0], table.bracket(g["f"], pts[:, 0], g["r"])[0])
    assert np.array_equal(table.windings, np.arange(10, 21))
