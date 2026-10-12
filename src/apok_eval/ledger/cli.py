"""``apok-eval ledger`` subcommands.

* ``apok-eval ledger verify RECORD [--expect-seal SEAL] [--allow-missing-inputs]
  [--no-code-check] [--repo PATH] [--json]`` - run every check; print one
  line per check (or the JSON report with ``--json``); exit 0 when nothing
  failed, else refuse with ``LEDGER_VERIFY_FAILED`` (exit 3). On a refusal the
  per-check lines (with ``--json``, the JSON report) go to stderr, so stdout
  holds only the refusal document, as for every other ``apok-eval`` command.
* ``apok-eval ledger verify-index INDEX [--anchor ENTRY_DIGEST]
  [--no-record-check]`` - verify the hash chain (and, by default, that every
  listed record exists and has the listed seal); print a JSON summary.
* ``apok-eval ledger show RECORD`` - print the record as indented JSON. It does
  not verify anything; a note on stderr says so.
"""

from __future__ import annotations

import argparse
import sys

from apok_eval.cli import print_json
from apok_eval.ledger.index import verify_index
from apok_eval.ledger.record import load_record
from apok_eval.ledger.verify import FAIL, NOT_CHECKED, run_verification, summarise


def _print_report(report: dict, out) -> None:
    for check in report["checks"]:
        line = f"{check['status']:<12} {check['check']}"
        if check["status"] == FAIL:
            line += f"  {check['code']}: {check['reason']}"
        elif check["status"] == NOT_CHECKED:
            line += f"  {check['reason']}"
        print(line, file=out)
    counts = report["counts"]
    print(f"record {report['record']}: {counts['PASS']} passed, {counts[FAIL]} failed, "
          f"{counts[NOT_CHECKED]} not checked; seal {report['seal']}", file=out)


def _verify(args: argparse.Namespace) -> int:
    report = run_verification(args.record, expect_seal=args.expect_seal, check_code=not args.no_code_check,
                              allow_missing_inputs=args.allow_missing_inputs, repo=args.repo)
    if args.json:
        print_json(report, sys.stdout if report["ok"] else sys.stderr)
    else:
        _print_report(report, sys.stdout if report["ok"] else sys.stderr)
    summarise(report, f"record {args.record}")
    return 0


def _verify_index(args: argparse.Namespace) -> int:
    summary = verify_index(args.index, anchor=args.anchor, check_records=not args.no_record_check)
    print_json(summary)
    return 0


def _show(args: argparse.Namespace) -> int:
    record = load_record(args.record)
    print("note: `show` does not verify; run `apok-eval verify` for that", file=sys.stderr)
    print_json(record)
    return 0


def register(subparsers) -> None:
    parser = subparsers.add_parser("ledger", help="sealed run records: verify, verify-index, show",
                                   description="apok-eval ledger: verify a sealed run record or an index, "
                                               "or print a record")
    sub = parser.add_subparsers(dest="ledger_command", metavar="LEDGER_COMMAND", required=True)

    verify = sub.add_parser("verify", help="recompute every digest, root, figure and the seal of a record")
    verify.add_argument("record", help="path of a record.json")
    verify.add_argument("--expect-seal", metavar="SEAL", help="the seal anchored elsewhere (tamper evidence)")
    verify.add_argument("--allow-missing-inputs", action="store_true",
                        help="report missing input and model files as NOT CHECKED instead of failing")
    verify.add_argument("--no-code-check", action="store_true",
                        help="do not compare the source manifest with the recorded commit")
    verify.add_argument("--repo", metavar="PATH", help="git repository holding the recorded commit; the commit "
                        "missing from it fails (default: the one holding the running apok_eval package, where a "
                        "missing commit is NOT CHECKED)")
    verify.add_argument("--json", action="store_true", help="print the report as JSON")
    verify.set_defaults(func=_verify)

    index = sub.add_parser("verify-index", help="verify the hash chain of an index")
    index.add_argument("index", help="path of an index.jsonl")
    index.add_argument("--anchor", metavar="ENTRY_DIGEST", help="an entry digest anchored elsewhere")
    index.add_argument("--no-record-check", action="store_true",
                       help="do not open the listed records (e.g. when they are not on this machine)")
    index.set_defaults(func=_verify_index)

    show = sub.add_parser("show", help="print a record (unverified)")
    show.add_argument("record", help="path of a record.json")
    show.set_defaults(func=_show)
