"""``apok-eval demo``, ``apok-eval e7v3``, ``apok-eval verify`` and ``apok-eval inspect``.

* ``apok-eval demo [--keep DIR] [--json]`` - build a synthetic scroll (CT, fit, umbilicus, fibres) and run
  three E7 v3 scenarios on it, each sealed and verified (:mod:`apok_eval.eval.demo`). Exit 0 when every
  verdict is the expected one and every record verifies, else 1.
* ``apok-eval e7v3 --segments DIR --fibres DIR --umbilicus FILE --z0 Z0 --z1 Z1 ...`` - E7 v3 on local files
  (:mod:`apok_eval.eval.local`), with the CT positive control when a volume is given, sealed in a run record
  unless ``--no-record``. Prints the verdicts (``--json``: the figures as JSON).
* ``apok-eval verify RECORD`` - ``apok-eval ledger verify``: recompute every digest, root, figure and the seal.
* ``apok-eval inspect PATH`` - ``apok-eval io inspect``: say what a local input is and summarise it.

A refusal exits 3 with the refusal printed as JSON (:func:`apok_eval.cli.main`).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys

from apok_eval import jsonstrict
from apok_eval.cli import print_json
from apok_eval.fileio import read_regular
from apok_eval.io.tifxyz import POLICIES
from apok_eval.io.umbilicus import VARIANTS
from apok_eval.refusal import Refusal, register_codes

register_codes(
    "apok_eval.eval.cli",
    {"EVAL_CLI_FILE_INVALID": "a JSON file named on the apok-eval command line is not readable strict JSON"},
)

GROUP_ORDER = ("all", "H", "V")
TERMS_KEY = "terms_accepted_by"


def _json_file(path: str, what: str):
    try:
        return jsonstrict.loads(read_regular(path))
    except (OSError, jsonstrict.StrictJsonError) as exc:
        raise Refusal("EVAL_CLI_FILE_INVALID", f"{what} {path!r} is not readable strict JSON: {exc}", file=what,
                      path=path, problem=str(exc)) from None


def _progress(quiet: bool):
    if quiet:
        return None
    return lambda text: print(f"apok-eval: {text}", file=sys.stderr, flush=True)


def _fmt(value, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def render_figures(fig: dict) -> list[str]:
    """Plain lines for one run's figures."""
    lines = [f"  CT positive control: {fig['control']}"]
    ct = fig.get("ct")
    if ct:
        lines.append(f"    fibres R {_fmt(ct['fibres_R'])}, fit R {_fmt(ct['fit_R'])}, null R {_fmt(ct['null_R'])}, "
                     f"layer period {ct['period']} voxels")
    lines.append(f"  fibres: {fig['fibres']['read']} read of {fig['fibres']['files']} files"
                 + (f", refused {fig['fibres']['refused']}" if fig["fibres"]["refused"] else ""))
    names = [g for g in GROUP_ORDER if g in fig["e7v3"]] + sorted(g for g in fig["e7v3"] if g not in GROUP_ORDER)
    for g in names:
        q, p95 = fig["Q"].get(g), fig["null_Q_p95"].get(g)
        extra = "" if q is None else f"  Q {_fmt(q)} vs {fig['null'].get(g)} null p95 {_fmt(p95)}"
        lines.append(f"  E7 v3 {g:<4} {fig['e7v3'][g]}{extra}")
    off = {g: v for g, v in (fig.get("offsets_voxels") or {}).items() if v is not None}
    if off:
        lines.append("  family offsets (voxels from the fit's sheet): "
                     + ", ".join(f"{g} {v:+.1f}" for g, v in off.items()))
    if fig.get("ceiling"):
        lines.append("  ceiling (a perfect front-face fit, Q for all): "
                     + ", ".join(f"sigma {s}: {_fmt(v.get('all'))}" for s, v in fig["ceiling"].items()))
    if fig.get("leakage"):
        lk = fig["leakage"]
        lines.append(f"  leakage: {_fmt(lk['share_near'])} of in-slab fibre points near an input patch; "
                     f"E7 v3 on the far points: {lk['e7v3_far']}")
    return lines


# --- demo --------------------------------------------------------------------------------------------------


def _demo(args: argparse.Namespace) -> int:
    from apok_eval.eval import demo

    if args.keep is not None and os.path.exists(args.keep) and os.listdir(args.keep):
        raise Refusal("EVAL_RUN_INPUT", f"--keep {args.keep!r} exists and is not empty; the demo writes a fresh "
                      "directory", input="keep", path=args.keep)
    out = demo.run(args.keep, progress=_progress(args.quiet or args.json))
    if args.json:
        print_json(out)
        return 0 if out["ok"] else 1
    print("apok-eval demo: E7 v3 on a synthetic scroll (no scroll data)")
    for sc in out["scenarios"]:
        print(f"\n{sc['name']}: {sc['title']}")
        for line in render_figures(sc["figures"]):
            print(line)
        v = sc["verify"]
        print(f"  record: sealed {sc['seal']}; verify {'ok' if v['ok'] else 'FAILED'} "
              f"({v['counts']['PASS']} passed, {v['counts']['FAIL']} failed, {v['counts']['NOT CHECKED']} not "
              f"checked)")
        print(f"  expected {sc['expected']}: {'yes' if sc['as_expected'] else 'NO'}")
    where = f"kept in {out['root']}" if out["kept"] else "written to a temporary directory, now removed"
    print(f"\nthe synthetic scroll and the run records were {where}")
    print("demo " + ("passed: every verdict as expected and every record verifies" if out["ok"] else "FAILED"))
    return 0 if out["ok"] else 1


# --- e7v3 --------------------------------------------------------------------------------------------------


def _default_run_id() -> str:
    return "e7v3-" + _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dt%H%M%Sz")


def _e7v3(args: argparse.Namespace) -> int:
    from apok_eval.eval.local import Inputs, evaluate, load_settings, seal_run

    windings = _json_file(args.windings, "--windings") if args.windings else None
    terms = _json_file(args.terms, "--terms") if args.terms else None
    if terms is not None and (not isinstance(terms, dict) or set(terms) != {TERMS_KEY}):
        raise Refusal("EVAL_CLI_FILE_INVALID", f"--terms must hold one JSON object with the single key {TERMS_KEY} "
                      "(your own acceptance, docs/licence.md)", file="--terms", path=args.terms,
                      problem="not a terms fragment")
    if terms is not None and args.registry is None:
        raise Refusal("EVAL_CLI_FILE_INVALID", "--terms records an acceptance for the licence gate; give the "
                      "registry it is checked against with --registry", file="--terms", path=args.terms,
                      problem="no --registry")
    settings = load_settings(args.settings, (args.z0, args.z1))
    inputs = Inputs(segments=args.segments, fibres=args.fibres, umbilicus=args.umbilicus,
                    umbilicus_variant=args.umbilicus_variant, policy=args.policy,
                    frame_shape=tuple(args.frame_shape) if args.frame_shape else None, windings=windings,
                    volume=args.volume, level=args.level, patches=args.patches)
    run_id = args.run_id or _default_run_id()
    out_dir = None if args.no_record else (args.out or os.path.join("apok-eval-runs", run_id))
    if out_dir is not None and os.path.isdir(out_dir) and os.listdir(out_dir):
        raise Refusal("LEDGER_RUN_DIR_NOT_EMPTY", f"run directory {out_dir} is not empty; a run is written to a "
                      "new directory", run_dir=out_dir)
    result, listing = evaluate(inputs, settings, registry=args.registry, terms=terms,
                               threads=args.threads, progress=_progress(args.quiet or args.json))
    seal = None
    if out_dir is not None:
        seal = seal_run(result, listing, out_dir, run_id=run_id, purpose=args.purpose or "E7 v3 on a local fit",
                        allow_dirty=args.allow_dirty)
    if args.json:
        print_json({"figures": result["figures"], "seal": seal,
                    "record": None if out_dir is None else os.path.join(out_dir, "record.json")})
        return 0
    print(f"apok-eval e7v3: slab z [{args.z0:g}, {args.z1:g}), frame {result['frame']['shape_zyx']}")
    for line in render_figures(result["figures"]):
        print(line)
    if not result["licence"]["checked"]:
        print("  licence: NOT CHECKED (no --registry)")
    if seal is not None:
        print(f"  record: {os.path.join(out_dir, 'record.json')} sealed {seal}")
        print(f"  check it with: apok-eval verify {os.path.join(out_dir, 'record.json')}")
    return 0


# --- verify and inspect ------------------------------------------------------------------------------------


def _verify(args: argparse.Namespace) -> int:
    from apok_eval.ledger.cli import _verify as ledger_verify

    return ledger_verify(args)


def _inspect(args: argparse.Namespace) -> int:
    from apok_eval.io.cli import _inspect as io_inspect

    return io_inspect(args)


def register(subparsers) -> None:
    demo = subparsers.add_parser("demo", help="run E7 v3 end to end on a synthetic scroll (no data needed)",
                                 description="Build a synthetic scroll (CT, fit, umbilicus, fibres), run three E7 v3 "
                                 "scenarios on it with the CT positive control, seal and verify each run.")
    demo.add_argument("--keep", metavar="DIR", help="write the synthetic scroll and the records to DIR and keep them "
                      "(default: a temporary directory, removed at the end)")
    demo.add_argument("--json", action="store_true", help="print the whole demo report as JSON")
    demo.add_argument("--quiet", action="store_true", help="no progress lines on stderr")
    demo.set_defaults(func=_demo)

    e7 = subparsers.add_parser("e7v3", help="E7 v3 on a local fit, fibres and umbilicus (and CT)",
                               description="Score a spiral fit against held-out human fibre traces (E7 v3), after "
                               "the CT positive control when a volume is given; seal the run in a record.")
    need = e7.add_argument_group("inputs")
    need.add_argument("--segments", required=True, metavar="DIR",
                      help="directory of the fit's tifxyz segments, one subdirectory each, named ...-wLLL-HHH")
    need.add_argument("--fibres", required=True, metavar="DIR", help="directory of VC3D fibre .json files")
    need.add_argument("--umbilicus", required=True, metavar="FILE", help="the umbilicus file")
    need.add_argument("--z0", required=True, type=float, help="the slab's lower z (inclusive), frame voxels")
    need.add_argument("--z1", required=True, type=float, help="the slab's upper z (exclusive), frame voxels")
    need.add_argument("--frame-shape", nargs=3, type=int, metavar=("Z", "Y", "X"),
                      help="the spiral frame's shape in voxels (required without --volume)")
    need.add_argument("--umbilicus-variant", choices=VARIANTS, default="fitter",
                      help="the umbilicus file's format (default: fitter, the spiral fitter's JSON)")
    need.add_argument("--policy", choices=POLICIES, default="spiral",
                      help="tifxyz validity policy for the segments (default: spiral)")
    need.add_argument("--windings", metavar="MAP.json",
                      help="JSON {segment directory: [first, last]} for segments not named ...-wLLL-HHH")
    ct = e7.add_argument_group("CT positive control and leakage audit")
    ct.add_argument("--volume", metavar="ZARR", help="the CT as a local OME-Zarr v2 volume")
    ct.add_argument("--level", type=int, default=0,
                    help="the volume level whose voxels are the frame's (default 0)")
    ct.add_argument("--patches", metavar="DIR", help="the fit's input patches (tifxyz), for the leakage audit")
    rec = e7.add_argument_group("settings, licence and record")
    rec.add_argument("--settings", metavar="FILE", help="JSON overriding the rule's settings (registration, face, "
                     "ct, margin, leakage_radius); the record keeps every value in force")
    rec.add_argument("--registry", metavar="FILE", help="asset registry declaring the inputs: turns the licence "
                     "gate on (docs/licence.md)")
    rec.add_argument("--terms", metavar="TERMS.json", help="your own acceptance, {\"terms_accepted_by\": {...}}, for "
                     "inputs whose licence requires terms")
    rec.add_argument("--out", metavar="DIR", help="new run directory for the record (default "
                     "apok-eval-runs/RUN_ID)")
    rec.add_argument("--run-id", help="run id, [a-z0-9][a-z0-9._-]* (default e7v3-<UTC time>)")
    rec.add_argument("--purpose", help="one line saying why the run was made (kept in the record)")
    rec.add_argument("--no-record", action="store_true", help="print the verdicts only; seal nothing")
    rec.add_argument("--allow-dirty", action="store_true", help="record uncommitted changes to apok-eval's own "
                     "code as a diff instead of refusing")
    rec.add_argument("--threads", type=int, default=8, help="threads decoding CT chunks (default 8)")
    rec.add_argument("--json", action="store_true", help="print the figures, seal and record path as JSON")
    rec.add_argument("--quiet", action="store_true", help="no progress lines on stderr")
    e7.set_defaults(func=_e7v3)

    verify = subparsers.add_parser("verify", help="verify a sealed run record (same as: ledger verify)",
                                   description="Recompute every digest, root, figure and the seal of a record.")
    verify.add_argument("record", help="path of a record.json")
    verify.add_argument("--expect-seal", metavar="SEAL", help="the seal anchored elsewhere (tamper evidence)")
    verify.add_argument("--allow-missing-inputs", action="store_true",
                        help="report missing input files as NOT CHECKED instead of failing")
    verify.add_argument("--no-code-check", action="store_true",
                        help="do not compare the source manifest with the recorded commit")
    verify.add_argument("--repo", metavar="PATH", help="git repository holding the recorded commit")
    verify.add_argument("--json", action="store_true", help="print the report as JSON")
    verify.set_defaults(func=_verify)

    inspect = subparsers.add_parser("inspect", help="say what a local input is and summarise it (same as: io "
                                    "inspect)", description="Detect what a local input is (tifxyz, OME-Zarr v2, "
                                    "mesh, point collections, umbilicus), read it and print a summary.")
    inspect.add_argument("path", metavar="PATH", help="a tifxyz or OME-Zarr v2 directory, a point-collections or "
                         "umbilicus .json, or an .obj/.ply mesh")
    frame = inspect.add_mutually_exclusive_group()
    frame.add_argument("--frame", metavar="FRAME.json", default=None,
                       help="the declared volume frame (volume_id, shape_zyx, voxel_size_um, source)")
    frame.add_argument("--no-frame", action="store_true",
                       help="inspect an OME-Zarr volume without a frame; the summary says NOT CHECKED")
    inspect.add_argument("--policy", choices=POLICIES, default=None,
                         help="tifxyz only: the one validity policy to apply (required with --frame)")
    inspect.add_argument("--json", action="store_true", help="print the whole inspect record as JSON")
    inspect.set_defaults(func=_inspect)
