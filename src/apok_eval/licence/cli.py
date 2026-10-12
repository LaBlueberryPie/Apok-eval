"""``apok-eval licence-check RUN_CONFIG [--registry PATH] [--for-release] [--json]``.

Clears a run configuration's inputs against the asset registry (default: the
registry shipped with apok-eval). When every input clears it prints, per input, the
asset, its licence, obligations, additional terms, required terms and the
decision, then the acceptance used (``--json``: the clearance record,
:meth:`LicenceClearance.as_dict`), and exits 0.

When the gate refuses, the refusal propagates to :func:`apok_eval.cli.main`, which
prints it as JSON on stdout and exits 3. Before that, each input's own
decision (:func:`apok_eval.licence.gate.input_decisions`: asset, licence,
obligations, additional terms, cleared or refused with code and reason) is
printed on stderr - as text, or with ``--json`` as a JSON list - so the inputs
that would clear are shown too. Nothing per input is printed when the refusal
concerns the whole run (the registry, the run's shape, the acceptance).
"""

from __future__ import annotations

import argparse
import sys

from apok_eval.cli import print_json
from apok_eval.licence.gate import InputDecision, LicenceClearance, input_decisions, licence_check, load_run_config
from apok_eval.refusal import Refusal


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "licence-check",
        help="clear a run's inputs against the asset registry and its licences",
        description="apok-eval licence-check: clear a run configuration's inputs against the asset registry "
                    "and its licences",
    )
    parser.add_argument("run_config", metavar="RUN_CONFIG", help="run configuration JSON (inputs, terms_accepted_by)")
    parser.add_argument("--registry", metavar="PATH", default=None,
                        help="asset registry JSON (default: the registry shipped with apok-eval)")
    parser.add_argument("--for-release", action="store_true",
                        help="also refuse inputs whose licence forbids publication or shipping in the pipeline")
    parser.add_argument("--json", action="store_true", help="print the clearance record as JSON")
    parser.set_defaults(func=_licence_check)


def _licence_check(args: argparse.Namespace) -> int:
    try:
        clearance = licence_check(args.run_config, args.registry, for_release=args.for_release)
    except Refusal:
        _report_inputs(args)
        raise
    if args.json:
        print_json(clearance.as_dict())
    else:
        print(render(clearance))
    return 0


def _report_inputs(args: argparse.Namespace) -> None:
    # A second, independent reading of both files: it reports, the refusal already raised decides.
    try:
        decisions = input_decisions(load_run_config(args.run_config), args.registry, for_release=args.for_release)
    except Refusal:
        return  # the refusal concerns the whole run; there is no per-input decision to show
    if args.json:
        print_json([d.as_dict() for d in decisions], sys.stderr)
    else:
        print(render_decisions(decisions), file=sys.stderr)


def render(clearance: LicenceClearance) -> str:
    """Human-readable report of a clearance."""
    record = clearance.as_dict()
    mode = "for release" if record["for_release"] else "for a run"
    lines = [f"licence-check: CLEARED {len(record['inputs'])} input(s) {mode}",
             f"registry: {record['registry']['source']} ({record['registry']['digest']})"]
    for item in record["inputs"]:
        lines += [
            f"input {item['index']}: asset {item['asset']} ({item['kind']})",
            f"  uri:              {item['uri']}",
            f"  host:             {item['host']}",
            f"  licence:          {item['licence']} - {item['licence_name']} <{item['licence_url']}>",
            f"  publication:      {item['publication']}; commercial use: {'yes' if item['commercial_use'] else 'no'}; "
            f"shippable in pipeline: {'yes' if item['shippable_in_pipeline'] else 'no'}",
            f"  additional terms: {', '.join(item['additional_terms']) or 'none'}",
            f"  required terms:   {', '.join(item['required_terms']) or 'none'}",
            *_obligation_lines(item["obligations"]),
            f"  decision:         {item['decision']}",
        ]
    acceptance = record["acceptance"]
    if acceptance is None:
        lines.append("acceptance: none (no input requires terms)")
    else:
        lines.append(f"acceptance: {acceptance['person']} on {acceptance['date']}")
        for entry in acceptance["accepted"]:
            pin = "text pinned and matched" if entry["pinned"] else "text not pinned by the registry"
            scope = "binds this person only" if entry["per_person"] else "not per-person"
            lines.append(f"  - {entry['terms']}: {entry['title']} ({pin}; {scope})")
    return "\n".join(lines)


def render_decisions(decisions: tuple[InputDecision, ...]) -> str:
    """Human-readable report of each input judged on its own, printed when the gate refuses the run."""
    refused = sum(d.refusal is not None for d in decisions)
    lines = [f"licence-check: REFUSED; judged on its own, {refused} of {len(decisions)} input(s) would be refused"]
    for d in decisions:
        decision = "cleared" if d.refusal is None else f"refused {d.refusal.code}: {d.refusal.reason}"
        lines += [
            f"input {d.index}: asset {d.asset}",
            f"  uri:              {d.given_uri}",
            f"  licence:          {d.licence or 'none (unknown asset or licence null)'}",
            f"  additional terms: {', '.join(d.additional_terms) or 'none'}",
            *_obligation_lines(d.obligations),
            f"  decision:         {decision}",
        ]
    return "\n".join(lines)


def _obligation_lines(obligations: list[str] | tuple[str, ...]) -> list[str]:
    return ["  obligations:" + ("" if obligations else " none"), *(f"    - {text}" for text in obligations)]
