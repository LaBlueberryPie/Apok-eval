"""The geometric leakage audit (apok_eval.eval.leakage): exact point-to-surface distances on synthetic patches."""

from __future__ import annotations

import numpy as np
import pytest

from apok_eval.eval import leakage as lk
from apok_eval.eval.fibres import Fibre


def brute(p, a, b, c, n=400):
    """Distance to a dense sample of the triangle (an upper bound within the sample's spacing)."""
    u, v = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    keep = u + v <= 1
    u, v = u[keep], v[keep]
    pts = a + u[:, None] * (b - a) + v[:, None] * (c - a)
    return np.linalg.norm(pts - p, axis=1).min()


def test_point_triangle_distance_in_every_region():
    rng = np.random.default_rng(3)
    a, b, c = np.array([0.0, 0, 0]), np.array([10.0, 0, 0]), np.array([2.0, 8, 0])
    p = rng.uniform(-6, 16, (300, 3))
    got = lk.point_triangle_distance(p, *(np.broadcast_to(v, p.shape) for v in (a, b, c)))
    want = np.array([brute(x, a, b, c) for x in p])
    assert np.all(got <= want + 1e-9) and np.all(want - got < 0.05)
    flat = np.broadcast_to(np.array([1.0, 1, 1]), (2, 3))
    assert lk.point_triangle_distance(np.array([[1.0, 1, 3], [4.0, 1, 1]]), flat, flat, flat) == pytest.approx([2, 3])


def plane(z0, n=12, step=5.0):
    """A flat patch at z = z0 over y, x in [0, (n - 1) step], with one hole quad's corner."""
    y, x = np.meshgrid(np.arange(n) * step, np.arange(n) * step, indexing="ij")
    zyx = np.stack([np.full_like(y, z0), y, x], axis=-1)
    valid = np.ones((n, n), bool)
    valid[0, 0] = False
    return zyx, valid


def fibre(points, name="f"):
    pts = np.asarray(points, dtype=np.float64)
    pts.flags.writeable = False
    return Fibre(name=name, digest="sha256:x", version=1, optimization_mode="lasagna", points_zyx=pts,
                 control_count=2, line_count=len(pts), trimmed=True, coordinate_scale=0.25,
                 scale_source="default-0.25", coordinate_base_shape_zyx=None, meta={})


def test_near_points_are_found_and_holes_and_edges_respected():
    surf = [plane(100.0)]
    pts = np.array([[102.0, 20, 20],   # 2 voxels above: near (inclusive)
                     [102.1, 20, 20],  # just beyond
                     [99.0, 30, 54],   # inside the far edge
                     [100.0, 30, 57],  # 2 voxels past the far edge, in plane: near
                     [100.0, 30, 57.5],
                     [100.0, 1, 1]])   # inside the hole quad: nearest valid surface is > 2 away
    near = lk.near_surfaces(pts, surf)
    assert near.tolist() == [True, False, True, True, False, False]
    assert lk.near_surfaces(pts, surf, radius=3.0)[1] and not lk.near_surfaces(np.zeros((0, 3)), surf).any()


def test_audit_reports_shares_and_far_fibres_drop_near_points():
    near_f = fibre([[101.0, 10, 10 + k] for k in range(10)] + [[150.0, 10, 10]], "near")
    far_f = fibre([[120.0, 10, 10 + k] for k in range(10)], "far")
    out_f = fibre([[500.0, 10, 10]], "outside")
    rep, masks = lk.audit([near_f, far_f, out_f], [plane(100.0)], z_range=(0.0, 200.0))
    assert rep["points"] == 21 and rep["near"] == 10 and rep["share_near"] == pytest.approx(10 / 21)
    assert rep["fibres"] == 2 and rep["fibres_touching"] == 1 and "spiralcheck" in rep["credit"]
    far = lk.far_fibres([near_f, far_f, out_f], masks)
    assert [f.name for f in far] == ["near", "far", "outside"] and len(far[0].points_zyx) == 1
    assert not far[0].points_zyx.flags.writeable
    gone = lk.far_fibres([near_f], [np.ones(11, bool)])
    assert gone == []
