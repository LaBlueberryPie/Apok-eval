"""``apok-eval demo``: the whole chain on a synthetic scroll, with no data to download.

The demo writes, in a fresh directory, a breathing spiral scroll and everything the evaluator reads from a
real one: a layered CT (OME-Zarr v2), a fit (tifxyz segments), an umbilicus, and human-style fibre traces
(VC3D fibre JSON), H fibres on the sheets' front faces and V fibres a few voxels behind them, on their back
faces. It declares the directory in a run-local licence registry, then runs ``apok-eval e7v3`` three times,
seals each run in a record and verifies the record:

1. **front-face fit** - the fit follows the sheets. Expected: control CONFIRMED, E7 v3 PASS.
2. **fit without the breathing** - the fit ignores how the sheets move with z. The fibres and the CT still
   agree, so the control is CONFIRMED; the fibres do not sit at a steady place in the fit's gaps. Expected:
   E7 v3 FAIL.
3. **fibres off the papyrus** - each fibre is moved to its own place between the sheets, so the fibres do not
   sit on the CT's layers. Expected: control REFUTED, and E7 v3 NOT MEASURED: the evaluator does not score a
   fit against fibres the CT does not support.

Everything is synthetic; nothing here is scroll data. Every setting is the evaluator's default.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from apok_eval.eval import synthetic as syn
from apok_eval.eval.local import Inputs, Settings, evaluate, seal_run
from apok_eval.licence.registry import SHIPPED_REGISTRY, read_registry

FRAME = (896, 448, 448)  # (z, y, x): a multiple of the CT chunk on every axis
CENTRE = (224.0, 224.0)
CHUNKS = (32, 32, 32)
SPACING, AMPLITUDE, PERIOD, PHASE = 16.0, 24.0, 512.0, 0.3
SEGMENTS = ((4, 6), (7, 10))
Z_LEVELS = np.arange(0.0, FRAME[0] + 1.0, 16.0)
SLAB = (96.0, 800.0)
FIBRE_BAND = (448.0, 576.0)  # the fibres' heights, where the sheets move fastest with z
CT_Z = (416, 608)  # the CT is written around them
N_FIBRES = 30
V_BEHIND = 4.0  # voxels: the V fibres' back face

#: The demo's settings beyond its slab: none, the default rule.
DEMO_SETTINGS: dict = {}


def sheet_radius(w, z):
    """The true sheets: radius 16 ω, breathing by 24 voxels over a 512-voxel period in z."""
    return SPACING * np.asarray(w) + AMPLITUDE * np.sin(2 * np.pi * np.asarray(z) / PERIOD)


def flat_radius(w, z):
    """A fit that misses the breathing: radius 16 ω at every height."""
    return SPACING * np.asarray(w) + 0.0 * np.asarray(z)


def _plain(w):
    return SPACING * np.asarray(w)


def write_fit(root: Path, radius) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for lo, hi in SEGMENTS:
        syn.write_segment(root, f"demo-w{lo:03d}-{hi:03d}", lo, hi, z_levels=Z_LEVELS, phase=PHASE, radius=_plain,
                          sheet_radius=radius, centre=CENTRE)
    return root


def write_fibres(root: Path, *, spread: bool) -> Path:
    """``N_FIBRES`` H and ``N_FIBRES`` V fibres in ``FIBRE_BAND``, tagged by hand as VC3D does. With ``spread``,
    fibre k sits ``16 * ((7 k) mod N) / N`` voxels out from its sheet instead of on its face."""
    root.mkdir(parents=True, exist_ok=True)
    z0, z1 = FIBRE_BAND
    for k in range(N_FIBRES):
        w = 5.0 + 4.0 * k / N_FIBRES
        shift = SPACING * ((7 * k) % N_FIBRES) / N_FIBRES if spread else 0.0
        zh = z0 + 6.0 + 4.0 * k
        h = syn.fibre_points(w, w + 0.2, zh, phase=PHASE, centre=CENTRE, offset=shift,
                             radius=lambda v, zh=zh: sheet_radius(v, zh))
        zv = np.arange(z0 + k, z0 + k + 96.0)
        v = syn.spiral_point(np.full(len(zv), w + 0.37), zv, phase=PHASE, centre=CENTRE,
                             radius=lambda q, zv=zv: sheet_radius(q, zv) + V_BEHIND + shift)
        for tag, pts in (("H", h), ("V", v)):
            assert z0 <= pts[:, 0].min() and pts[:, 0].max() < z1
            doc = syn.fibre_document(pts, version=4, extra={"hv_classification": {"manual_tag": tag}})
            (root / f"{tag.lower()}{k:02d}.json").write_text(json.dumps(doc))
    return root


def write_world(root: Path) -> dict[str, Path]:
    """Write the synthetic scroll under ``root``; return the path of each input."""
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "ct": syn.write_ct(root / "ct.zarr", frame_shape=FRAME, chunks=CHUNKS, z_range=CT_Z,
                           block=lambda origin, shape: syn.layered_block(
                               origin, shape, sheet_radius=sheet_radius, lo=SEGMENTS[0][0], hi=SEGMENTS[-1][1],
                               centre=CENTRE, phase=PHASE, spacing=SPACING)),
        "umbilicus": syn.write_umbilicus_file(root, CENTRE, z_top=float(FRAME[0])),
        "fit": write_fit(root / "fit", sheet_radius),
        "fit_flat": write_fit(root / "fit-without-breathing", flat_radius),
        "fibres": write_fibres(root / "fibres", spread=False),
        "fibres_off": write_fibres(root / "fibres-off-papyrus", spread=True),
    }
    paths["registry"] = write_registry(root)
    return paths


def write_registry(root: Path) -> Path:
    """A run-local registry: the shipped one, plus the demo directory declared as synthetic MIT data."""
    doc = read_registry(SHIPPED_REGISTRY)
    uri = Path(root).resolve().as_uri()
    doc["hosts"][uri] = {"allowed_licences": ["MIT"], "requires_terms": [],
                         "note": "the demo's directory, written by apok-eval demo"}
    doc["assets"]["apok-eval-demo"] = {
        "kind": "synthetic", "locations": [uri], "licence": "MIT", "additional_terms": [], "aliases": [],
        "evidence": "Written by apok_eval.eval.demo (MIT); no scroll data.",
        "note": "The synthetic scroll of apok-eval demo."}
    path = root / "registry.json"
    path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return path


@dataclass(frozen=True)
class Scenario:
    name: str
    title: str
    fit: str
    fibres: str
    expected: dict


SCENARIOS = (
    Scenario("front-face-fit", "a fit that follows the sheets", "fit", "fibres",
             {"control": "CONFIRMED", "all": "PASS", "H": "PASS", "V": "PASS"}),
    Scenario("fit-without-breathing", "a fit that misses how the sheets move with z", "fit_flat", "fibres",
             {"control": "CONFIRMED", "all": "FAIL", "H": "FAIL", "V": "FAIL"}),
    Scenario("fibres-off-papyrus", "fibres that do not sit on the CT's layers", "fit", "fibres_off",
             {"control": "REFUTED", "all": "NOT MEASURED"}),
)


def run(root: Path | None = None, *, progress=None, scenarios=SCENARIOS) -> dict:
    """Build the world (in ``root``, or a temporary directory removed afterwards) and run every scenario."""
    from apok_eval.ledger.verify import run_verification

    say = progress or (lambda text: None)
    keep = root is not None
    base = Path(root) if keep else Path(tempfile.mkdtemp(prefix="apok-eval-demo-"))
    try:
        say(f"writing the synthetic scroll in {base}")
        paths = write_world(base)
        settings = Settings(z_range=SLAB, **DEMO_SETTINGS)
        out = {"root": str(base), "kept": keep, "scenarios": []}
        for sc in scenarios:
            say(f"scenario {sc.name}: {sc.title}")
            inputs = Inputs(segments=str(paths[sc.fit]), fibres=str(paths[sc.fibres]),
                            umbilicus=str(paths["umbilicus"]), volume=str(paths["ct"]), level=0,
                            frame_shape=FRAME)
            result, listing = evaluate(inputs, settings, registry=str(paths["registry"]),
                                       progress=lambda t: say(f"  {t}"))
            run_dir = base / "runs" / sc.name
            seal = seal_run(result, listing, run_dir, run_id=f"demo-{sc.name}",
                            purpose=f"apok-eval demo: {sc.title} (synthetic data)", allow_dirty=True)
            report = run_verification(run_dir / "record.json")
            fig = result["figures"]
            got = {"control": fig["control"], **fig["e7v3"]}
            out["scenarios"].append({
                "name": sc.name, "title": sc.title, "figures": fig, "seal": seal,
                "record": str(run_dir / "record.json"),
                "verify": {"ok": report["ok"], "counts": report["counts"]},
                "expected": sc.expected,
                "as_expected": all(got.get(k) == v for k, v in sc.expected.items()),
            })
            say(f"  sealed {seal}; verify {'ok' if report['ok'] else 'FAILED'} {report['counts']}")
        out["ok"] = all(s["as_expected"] and s["verify"]["ok"] for s in out["scenarios"])
        return out
    finally:
        if not keep:
            shutil.rmtree(base, ignore_errors=True)
