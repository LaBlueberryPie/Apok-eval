"""The CT-profile diagnostic (apok_eval.eval.ctprofile) on a synthetic CT whose bright layers follow the spiral."""

from __future__ import annotations

import math

import numpy as np
import pytest

from apok_eval import jsonstrict
from apok_eval.eval import ctprofile as ct
from apok_eval.eval.fibres import read_fibre
from apok_eval.eval.fit import WindingFit, segment_from_tifxyz
from apok_eval.io.umbilicus import read_umbilicus
from apok_eval.refusal import Refusal
from eval_fixtures import builders as b

Z = np.arange(20.0, 401.0, 20.0)
SLAB = (150.0, 250.0)
PHASE = 0.3
CFG = ct.CTConfig(z_range=SLAB, reach=24, window=12, points_per_fibre=16, fit_points=400, null_points=400,
                  period_range=(6, 22), max_chunks=400)
CHUNK = (32, 32, 32)


def layers(index):
    """A synthetic CT chunk: bright (200) at the spiral's sheets r = 16(k + f), dark (20) between them."""
    origin = np.asarray(index) * np.asarray(CHUNK)
    z, y, x = np.meshgrid(*[np.arange(o, o + c) for o, c in zip(origin, CHUNK)], indexing="ij")
    dy, dx = y - b.CENTRE[0], x - b.CENTRE[1]
    r = np.hypot(dy, dx)
    f = np.mod(np.arctan2(dy, dx) / (2 * np.pi) + PHASE, 1.0)
    frac = r / 16.0 - f
    d = 16.0 * np.abs(frac - np.round(frac))
    return (20 + 180 * np.exp(-d * d / (2 * 1.5 ** 2))).astype(np.uint8)


@pytest.fixture(scope="module")
def umbilicus(tmp_path_factory):
    return read_umbilicus(b.write_umbilicus_file(tmp_path_factory.mktemp("umb")))


@pytest.fixture(scope="module")
def fit(tmp_path_factory, umbilicus):
    dirs = b.write_fit(tmp_path_factory.mktemp("fit"), [(10, 14), (15, 20)], z_levels=Z, phase=PHASE)
    return WindingFit([segment_from_tifxyz(d, policy="spiral") for d in dirs], umbilicus, z_range=SLAB)


def cache():
    return ct.ChunkCache(CHUNK, b.FRAME, layers)


def fibres(offsets):
    out = []
    for k, o in enumerate(offsets):
        w = 11.0 + 0.45 * k
        pts = b.fibre_points(w, w + 0.15, 160.0 + 5 * k, offset=o, phase=PHASE)
        out.append(read_fibre(b.fibre_bytes(pts, **({"domain": [4000] * 3} if k % 2 else {})), name=f"f{k:02d}",
                              base_shape_zyx=b.FRAME))
    return out


def test_settings_out_of_range_refuse():
    for change in ({"window": 40}, {"period_range": (6, 40)}, {"locked_r": 0.1}, {"p_max": 0.1},
                   {"z_range": (2.0, 1.0)}, {"min_sweep": 0.0}, {"max_chunks": 0}):
        with pytest.raises(Refusal) as r:
            ct.CTConfig(**{**{"z_range": SLAB}, **change}).check()
        assert r.value.code == "EVAL_CT_PARAMETER"
    ct.CTConfig(z_range=SLAB).check()


def test_rays_point_outward_and_the_cache_samples_the_layers(umbilicus):
    pts = b.fibre_points(12.0, 12.05, 200.0, phase=PHASE)
    vox = ct.ray_voxels(pts, umbilicus, 8)
    r = np.hypot(vox[..., 1] - b.CENTRE[0], vox[..., 2] - b.CENTRE[1])
    assert np.all(np.diff(r, axis=1) > -0.5) and np.all(r[:, -1] - r[:, 0] > 14)  # nearest voxels
    c = cache()
    c.load(c.needed(vox))
    prof, counts = ct.profiles(c, vox)
    assert counts["kept"] == len(pts) and np.all(prof[:, 8] > 150) and np.all(prof[:, 0] < 60)
    empty = ct.ChunkCache(CHUNK, b.FRAME, lambda k: None)
    empty.load(empty.needed(vox))
    _, counts = ct.profiles(empty, vox)
    assert counts["missing_chunk"] == len(pts) and not empty.covered(vox).any()
    flat = ct.ChunkCache(CHUNK, b.FRAME, lambda k: np.zeros(CHUNK, np.uint8))
    flat.load(flat.needed(vox))
    _, counts = ct.profiles(flat, vox)
    assert counts["flat"] == len(pts)


def test_period_and_phase_find_the_layers(umbilicus):
    pts = np.concatenate([b.fibre_points(w, w + 0.1, 200.0, phase=PHASE, offset=o)
                          for w, o in ((12.0, 0.0), (14.0, 4.0))])
    c = cache()
    vox = ct.ray_voxels(pts, umbilicus, 24)
    c.load(c.needed(vox))
    prof, _ = ct.profiles(c, vox)
    T, ac = ct.period(prof, (6, 22))
    assert T in (15, 16, 17) and ac[0] == pytest.approx(1.0)
    ph = ct.phases(prof, T, 12)
    n_on = len(b.fibre_points(12.0, 12.1, 200.0, phase=PHASE))
    on, off = ph[:n_on], ph[n_on:]
    assert ct.rayleigh(on)["R"] > 0.95 and abs(ct.rayleigh(on)["mean_angle"]) < 0.3
    assert abs(ct.rayleigh(off)["mean_angle"] - 2 * math.pi * 4 / T) < 0.4  # 4 voxels outside: phase 2π·4/T


def test_rayleigh_and_locking_rule():
    assert ct.rayleigh(np.zeros(0))["R"] is None
    s = ct.rayleigh(np.zeros(50))
    assert s["R"] == pytest.approx(1.0) and s["p"] < 1e-10 and ct.locked(s, ct.CTConfig(z_range=SLAB)) == "LOCKED"
    spread = ct.rayleigh(np.linspace(-math.pi, math.pi, 64, endpoint=False))
    assert spread["R"] < 1e-9 and ct.locked(spread, ct.CTConfig(z_range=SLAB)) == "NOT LOCKED"
    mid = {"n": 400, "R": 0.2, "p": 1e-6, "mean_angle": 0.0}
    assert ct.locked(mid, ct.CTConfig(z_range=SLAB)) == "UNCLEAR"
    assert ct.locked({"n": 0, "R": None, "p": None}, ct.CTConfig(z_range=SLAB)) == "INCONCLUSIVE"


def test_circle_fit_and_degenerate_points():
    t = np.linspace(0, 1, 30)
    c, r = ct.circle_fit(np.stack([5 + 40 * np.sin(t), -3 + 40 * np.cos(t)], axis=1))
    assert np.allclose(c, [5, -3]) and r == pytest.approx(40)
    assert ct.circle_fit(np.zeros((2, 2))) is None
    assert ct.circle_fit(np.stack([np.arange(5.0), np.arange(5.0)], axis=1)) is None


def test_fibres_on_the_sheets_and_the_fit_both_lock(fit):
    rep = ct.diagnose(fit, fibres([0.0] * 12), cache(), config=CFG)
    v = rep["verdicts"]
    assert (v["null"], v["fibres"], v["fit"]) == ("NOT LOCKED", "LOCKED", "LOCKED")
    assert rep["period"] in (15, 16, 17) and abs(rep["fibre_minus_fit_offset_voxels"]) < 1.5
    assert set(k for k in v if k.startswith("fibres:")) == {"fibres:coordinate_base_shape_zyx",
                                                             "fibres:default-0.25"}
    assert rep["curvature"]["fibres"] == 12 and abs(rep["curvature"]["rho_over_d"]["q50"] - 1) < 0.1
    assert rep["mean_profiles"]["fibres"][CFG.reach] > 1.0
    jsonstrict.dumps(rep)


def test_per_point_locking_sits_beside_the_vote(fit):
    """Every fibre point counted once, by scale group and by family; the verdicts do not change."""
    fs = fibres([0.0] * 6 + [16.0 * k / 6 for k in range(6)])  # six on the sheets, six spread through the gap
    fam = {f.name: ("H" if i < 6 else "V") for i, f in enumerate(fs)}
    rep = ct.diagnose(fit, fs, cache(), config=CFG, families=fam)
    pp = rep["locking"]["fibres_per_point"]
    assert pp["all"]["n"] == sum(r["points"] for r in rep["per_fibre"])
    assert pp["family:H"]["R"] > 0.9 and pp["family:V"]["R"] < 0.3 and pp["family:H"]["n"] + pp["family:V"]["n"] \
        == pp["all"]["n"]
    assert rep["locking"]["fibres_by_family"]["H"]["R"] > 0.9
    assert set(pp) == {"all", "scale:coordinate_base_shape_zyx", "scale:default-0.25", "family:H", "family:V"}
    plain = ct.diagnose(fit, fs, cache(), config=CFG)
    assert plain["verdicts"] == rep["verdicts"] and plain["locking"]["fibres_by_family"] == {}


def test_fibres_spread_through_the_layers_do_not_lock(fit):
    rep = ct.diagnose(fit, fibres([16.0 * k / 12 for k in range(12)]), cache(), config=CFG)
    v = rep["verdicts"]
    assert v["fibres"] == "NOT LOCKED" and v["fit"] == "LOCKED"
    shifted = ct.diagnose(fit, fibres([5.0] * 12), cache(), config=CFG)  # one steady offset: locked, offset
    assert shifted["verdicts"]["fibres"] == "LOCKED"
    assert shifted["fibre_minus_fit_offset_voxels"] == pytest.approx(5.0, abs=1.5)


def test_too_many_chunks_refuses_and_no_slab_fibre_is_inconclusive(fit):
    with pytest.raises(Refusal) as r:
        ct.diagnose(fit, fibres([0.0] * 4), cache(), config=ct.CTConfig(**{**CFG.__dict__, "max_chunks": 1}))
    assert r.value.code == "EVAL_CT_TOO_MANY_CHUNKS"
    far = [read_fibre(b.fibre_bytes(b.fibre_points(12.0, 12.2, 60.0, phase=PHASE)), name="low",
                      base_shape_zyx=b.FRAME)]
    rep = ct.diagnose(fit, far, cache(), config=CFG)
    assert rep["fibres_in_slab"] == 0 and rep["verdicts"]["fibres"] == "INCONCLUSIVE"


def test_points_are_halved_to_fit_the_chunk_budget(fit):
    full = ct.diagnose(fit, fibres([0.0] * 12), cache(), config=CFG)
    tight = ct.diagnose(fit, fibres([0.0] * 12), cache(),
                        config=ct.CTConfig(**{**CFG.__dict__, "max_chunks": full["chunks"]["needed"] - 1}))
    assert full["points_per_fibre_used"] == 16 and tight["points_per_fibre_used"] < 16
    assert tight["chunks"]["needed"] < full["chunks"]["needed"] and tight["verdicts"]["fibres"] == "LOCKED"
