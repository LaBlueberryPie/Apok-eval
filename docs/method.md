# The method: E7 v3, its CT positive control, and what a verdict means

This page states every rule `apok-eval e7v3` applies, in the order it applies them. The module docstrings
(`src/apok_eval/eval/*.py`) are the normative text; this page follows them. Everything here is fixed before a
run scores anything: a run's record keeps every setting in force, and changing one is a different run.

## Inputs and the frame

- **The frame.** All coordinates are (z, y, x) voxels of one declared grid, the *spiral frame*: usually one
  level of the CT volume. Its shape is declared (`--frame-shape`, or the shape of the CT level given with
  `--volume` and `--level`, which must then agree). Nothing is inferred from file names or units.
- **Fibres** (`apok_eval.eval.fibres`): VC3D `vc3d_fiber` JSON, read by villa's convention, ported with credit
  from villa `e0bbb8b` (MIT; [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)): the dense `line_points` trimmed to the span between the line points nearest
  the first and last control points; scaled by the unique dyadic 2^k (k in [-5, 5]) that maps
  `coordinate_base_shape_zyx` onto the frame, or by 0.25 when the fibre does not declare its domain;
  coordinates rounded to float32 first, as villa does. apok-eval is stricter than villa: strict JSON; a fibre
  without control points is refused (villa skips it with a warning); every scaled point must lie in the frame
  within one voxel. A refused fibre is counted by its refusal code and left out.
- **The fit** (`apok_eval.eval.fit.WindingFit`): tifxyz segments named for their winding range
  (`...-wLLL-HHH`; `--windings` names segments that are not), valid under the declared validity policy
  (default `spiral`, the spiral fitter's own), interpolated bilinearly. A vertex's winding coordinate ω comes
  from its segment's range and its unwrapped angle about the umbilicus: unwrap per row in runs, tie runs by whole
  turns through shared columns, one sense for all segments, and a common phase fitted by circular mean. The
  sense is read at the same angle one turn apart, r(t + 1) − r(t) along each row, which cancels a ring's offset
  from the umbilicus. Each segment's residual and span in turns are reported.
- **The umbilicus** (`apok_eval.io.umbilicus`): the axis the sheets wind around, in a declared format
  (`fitter`, `vc3d-json`, `vc3d-text`, `thaumato-text`).

## Where a fibre point sits: the gap position

For a point, the ray leaves the umbilicus at the point's height and passes through the point. The fit's sheets
cross that ray at ω = k + f(φ). The two adjacent crossings that bracket the point give u ∈ [0, 1) (0 on the
inner sheet), and the point's **gap position** is s = u − round(u) ∈ [−½, ½): 0 on a sheet, in units of the
local sheet spacing. Crossings are read from a table of the fit's radius on a regular (ω, z) grid; the exact
crossings are computed for every observed point too, and if their median disagreement with the table exceeds
0.05 of a gap, every verdict is INCONCLUSIVE. A point with no bracketing pair (inside the innermost sheet,
outside the outermost, beside a hole) is unknown, and counted. Each fibre is thinned to at most 256 evenly
spaced points; a fibre is scored in a pass when at least 20 of its points have a known gap position.

## Families, offsets and the held-out split

villa documents that horizontal (H) fibres lie on a sheet's front face and vertical (V) fibres on its back face,
a few voxels apart, and that a fit can target only one face (`spiral-fitting/README.md`, "Vertical-fiber radial
offset"). So E7 v3 does not ask "are the fibres on the fit's sheet?" but "does each family sit at one steady
place in the fit's gaps?":

- **Families.** Each fibre is H, V or untagged (U) by villa's `classify_fiber_hv` precedence: VC3D's manual
  tag; else its automatic tag when its certainty is at least 0.5; else V when the z extent is at least 0.8 of
  the path length, H when at most 0.2, untagged otherwise.
- **Split.** A fibre is in the *calibration* half when the first byte of sha256("<split_seed>:<name>") is even,
  otherwise in the *evaluation* half. The name alone decides, never the data.
- **Offsets.** For each family, the offset D (a grid of 0.01 of the gap) that maximises the calibration fibres'
  mean share of points within ⅛ of a gap of D, distance taken around the gap's circle. Tied D form a flat top
  (a precise family scores the same anywhere within ⅛ of its true offset); its centre is taken. A family with no
  scored calibration fibre takes the pooled D, and the report says so.
- **Score.** An evaluation fibre's Q_f is the fraction of its points within ⅛ of a gap of its family's D. A
  group's Q is the mean of Q_f over its scored evaluation fibres: "all", H, V. Points spread evenly through
  the gap give Q ≈ 0.25; a family sitting precisely at one place gives Q near 1.

## Nulls

A good Q is only evidence if moved fibres do not score as well. Each fibre is moved as a whole, its points read
again, and the whole procedure, calibration included, run on the moved fibres:

- **z-offset null** (100 draws): each fibre moved up or down by 128–384 voxels (its sign flipped when the drawn
  move keeps fewer than half its points in the slab). The null assumes the sheets do not stay at one radius
  over that distance, so that a moved fibre lands at an unrelated place in the gap; where they do, it is
  uninformative (below).
- **rotation null** (100 draws): each fibre turned about the umbilicus by ¼–¾ of a turn. On a near-perfect
  spiral every rotated fibre lands ¼–¾ of a gap away and a re-calibrated D finds it, so this null is lenient;
  it is used only when the z-offset null is uninformative (its median Q is at least 0.35: moving in z did not
  move the fibres off their places).

Draws use numpy's PCG64 seeded `seed`, in that order; the seeds are in the record.

## Verdict

Per group ("all", H, V), on the evaluation fibres, against the null in use:

- **PASS**: Q ≥ 0.40 and Q above every draw of the null;
- **FAIL**: Q < 0.30, or Q not above the null's 95th percentile;
- **INCONCLUSIVE**: otherwise, or when the table check failed;
- **NOT MEASURED**: fewer than 10 scored evaluation fibres in the group.

## The CT positive control

Fibres that sit nowhere in particular in the fit's gaps could mean the fit is wrong, or that the fibres are in
another frame than the fit (wrong scale, wrong volume, wrong level). The CT tells these apart: whatever lies on
papyrus is locked to the layers the scan shows. With `--volume`, before E7 v3 (`apok_eval.eval.ctprofile`):

- **Profiles.** For each sampled point, the CT along its umbilicus ray, 32 voxels each way. Fibres: up to 32
  points each, fewer if the chunks they need exceed the run's budget. Fit: up to 10,000 of its vertices in
  chunks the fibres already needed. Null: up to 10,000 random positions in those chunks, within the fit's
  radial range.
- **Period and phase.** The layer period T is the first maximum of the profiles' mean autocorrelation between 6
  and 30 voxels (or a fixed period, when one is set). Each point's phase is where in that layer pattern it sits.
- **Locking.** Fibres vote once each (the length R of the mean of each fibre's unit mean phase vector); the fit
  and the null vote per point; p is Rayleigh's. LOCKED: R ≥ 0.30 and p < 0.001; NOT LOCKED: R < 0.15 or
  p ≥ 0.05; otherwise UNCLEAR.
- **The control**: CONFIRMED when the fibres are LOCKED and the null NOT LOCKED; REFUTED when the fibres are NOT
  LOCKED (with a NOT LOCKED null); INCONCLUSIVE otherwise. **E7 v3 is NOT MEASURED unless the control is
  CONFIRMED.** Without a volume the control is NOT RUN, E7 v3 is still scored, and the output says so.
- **Reading.** The fit's own locking is reported beside it: fibres locked and fit not means the fit misses the
  sheets locally; both locked means the frames agree and any mismatch is in the fit's sheet positions or count.

## The ceiling

What a perfect front-face fit could score with these very fibres: every observed point with a known gap
position is replaced by a perfect fit's, with fibre noise σ = 1, 2 and 3 voxels (H on the sheet, V 4 voxels
behind it, untagged fibres one or the other at random), on each point's own local spacing; the same split,
calibration and score give the Q such a fit would reach. It is reported, never a threshold: it says how high Q
can be expected to go for these fibres, so a modest Q is not misread as a bad fit.

## Also reported

Per-family gap histograms and circular-mean offsets; the calibrated D in gap units and in voxels; each fibre's
winding consistency (the share of its points on its most common sheet of the fit, and the same after shifting
by its family's D) against the z-offset null; every evaluation fibre's Q at its family's offset. Two runs on
one slab can be compared fibre by fibre with `apok_eval.eval.faces.paired_difference` (a bootstrap over the
fibres both scored).

## The leakage audit (optional)

A fit that read verified patches has seen those surfaces: a fibre point within 2 voxels of one of them is not
held out from the fit geometrically, whatever the file names say. With `--patches DIR` (the fit's input
patches, tifxyz), `apok_eval.eval.leakage` marks each fibre point near a patch (exact point-to-triangle
distance over the patches' valid quads) and E7 v3 is scored again on the far points only. The idea is
spiralcheck's ([github.com/Nicodol/spiralcheck](https://github.com/Nicodol/spiralcheck), MIT; its notice is in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)); this is apok-eval's own implementation, for fibre points
against tifxyz patches.

## What a verdict does not say

- PASS says the families sit at steady places in the fit's gaps, better than moved fibres do, in this slab. It
  does not say the fit's winding numbers are right, that the fit is right outside the slab, or that a fit that
  read these fibres did not learn their places (run the leakage audit, and use fibres the fit did not read).
- FAIL says they do not, in this slab. With a CONFIRMED control the fibres are on papyrus, so the fit is the
  likely culprit; without the control (NOT RUN), the fibres' frame is not checked.
- The z-offset null assumes the sheets move with z over a few hundred voxels. Where they do not (a nearly
  cylindrical stretch), it becomes uninformative and the lenient rotation null is used; the report says which.
- The rule's numbers (⅛ band, 0.40 / 0.30, 100 draws, 128–384 voxels) are fixed choices, not fitted to any
  data set; `--settings` can change them, and the record then says so.

## Defaults

| setting | default | where |
|---|---|---|
| slab | required (`--z0`, `--z1`) | registration, CT |
| band | ⅛ of a gap | `registration.band` |
| points per fibre / minimum | 256 / 20 | `registration.points_per_fibre`, `min_points` |
| gap-position table | a z row every 16 voxels, 1024 ω bins per winding | `registration.z_step`, `omega_bins` |
| minimum fibres per group | 10 | `registration.min_fibres` |
| z-offset null | 100 draws, 128–384 voxels | `registration.null_draws`, `z_offset` |
| rotation null | 100 draws, ¼–¾ turn (in radians: [π/2, 3π/2]) | `registration.rotation_draws`, `rotation` |
| PASS / FAIL thresholds | 0.40 / 0.30 | `registration.registered_q`, `not_registered_q` |
| uninformative z-null | median Q ≥ 0.35 | `registration.uninformative_null_q` |
| table check | median error ≤ 0.05 of a gap | `registration.table_tolerance` |
| seeds | nulls 1, split 2, ceiling 3, CT sampling 1 | `registration.seed`, `face.split_seed`, `face.ceiling_seed`, `ct.seed` |
| offset grid | 0.01 of a gap | `face.offset_step` |
| ceiling | σ = 1, 2, 3 voxels; V 4 voxels behind | `face.sigmas`, `face.face_gap` |
| CT rays | 32 voxels each way, window 16 | `ct.reach`, `ct.window` |
| CT points | 32 per fibre; up to 10,000 fit vertices and 10,000 null positions | `ct.points_per_fibre`, `ct.fit_points`, `ct.null_points` |
| CT layer period | the first autocorrelation maximum between 6 and 30 voxels; 0 searches, a positive value fixes it | `ct.period_range`, `ct.period_fixed` |
| CT chunk budget | at most 4000 chunks in memory, and at most 6 GiB (6 × 2³⁰) divided by a chunk's voxel count | `ct.max_chunks`, `ct.memory_bytes` |
| CT curvature report | fibre arcs that sweep at least 0.05 of a turn | `ct.min_sweep` |
| CT locking | LOCKED R ≥ 0.30, p < 0.001; NOT LOCKED R < 0.15 | `ct.locked_r`, `ct.p_max`, `ct.unlocked_r` |
| fit margin around the slab | 128 voxels | `margin` |
| leakage radius | 2 voxels | `leakage_radius` |
| registration diagnostic only | z bands 8–16 and 32–64 voxels, 20 draws each; offset margin 0.10 | `registration.bands`, `band_draws`, `offset_margin` |

A `--settings` file overrides any of these, by section: `{"registration": {...}, "face": {...}, "ct": {...},
"margin": ..., "leakage_radius": ...}`. Unknown names are refused. Values are in the units of the table: voxels,
shares of a gap, and **radians** for `registration.rotation` (a quarter turn is π/2, not 0.25).

`registration.bands`, `band_draws` and `offset_margin` are checked and recorded with every run, but they decide
no E7 v3 verdict and change none of the figures a run quotes: `bands` and `band_draws` are read only by the
registration diagnostic (`apok_eval.eval.registration.diagnose`, which `e7v3` does not run), and
`offset_margin` sets only the `offset_consistent` flag reported beside each verdict.
