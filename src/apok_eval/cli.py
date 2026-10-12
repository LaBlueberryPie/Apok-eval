"""``apok-eval`` command line.

Exit codes: 0 success; 1 unexpected error; 2 usage error (a command group
without its subcommand included); 3 a guard refused (the refusal is printed as
JSON on stdout, with its code and reason).

Every JSON document a subcommand prints goes through :func:`print_json`: two-space
indent, sorted keys, non-ASCII written as itself, and strict JSON only, so
``apok_eval.jsonstrict.loads`` accepts apok-eval's own output (:func:`json_safe` says how a
value strict JSON cannot hold is written).
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import numbers
import os
import sys
from typing import Any, Sequence, TextIO

from apok_eval import __version__
from apok_eval.fileio import portable_name
from apok_eval.refusal import Refusal

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_REFUSED = 0, 1, 2, 3

# Each module contributes its subcommands through ``register(subparsers)``.
COMMAND_MODULES = (
    "apok_eval.eval.cli",
    "apok_eval.licence.cli",
    "apok_eval.ledger.cli",
    "apok_eval.io.cli",
)


def json_safe(value: Any) -> Any:
    """``value`` as strict JSON can hold it, for printing.

    A string (or a ``PathLike``, or a key) holding lone surrogates, which is how Python spells a file name
    whose bytes are not UTF-8, is written with those bytes as ``\\xNN`` (:func:`apok_eval.fileio.portable_name`);
    ``bytes`` likewise; a non-finite number as its ``repr`` (``"nan"``, ``"inf"``, ``"-inf"``: a refusal's
    details may echo the bad value it refused); a tuple as a list; numpy and other ``numbers`` integers and
    reals as ``int`` and ``float``. Anything else (a set, an object, a non-string key) is a ``TypeError``:
    a printer that coerced it silently could print two different values alike.
    """
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return portable_name(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        as_float = float(value)
        return as_float if math.isfinite(as_float) else repr(as_float)
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", "backslashreplace")
    if isinstance(value, os.PathLike):
        return json_safe(os.fspath(value))
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"a JSON object key must be a string, not {type(key).__name__} ({key!r})")
            out[portable_name(key)] = json_safe(item)
        return out
    raise TypeError(f"a {type(value).__name__} cannot be printed as JSON: {value!r}")


def _dumps(obj: Any) -> str:
    return json.dumps(json_safe(obj), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)


def print_json(obj: Any, file: TextIO | None = None) -> None:
    """Print ``obj`` as one JSON document (see the module docstring) on ``file`` (default: stdout)."""
    print(_dumps(obj), file=sys.stdout if file is None else file)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="apok-eval", description="Verification-first evaluation of spiral fits for the Vesuvius Challenge.")
    parser.add_argument("--version", action="version", version=f"apok-eval {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    for name in COMMAND_MODULES:
        importlib.import_module(name).register(sub)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help(sys.stderr)
        return EXIT_USAGE
    try:
        return int(args.func(args) or EXIT_OK)
    except Refusal as refusal:
        try:
            text = _dumps(refusal.as_dict())
        except (TypeError, ValueError) as exc:  # details json_safe cannot print: say so, never crash a refusal
            text = _dumps({"refused": True, "code": refusal.code, "reason": refusal.reason,
                           "details": {"unprintable_details": repr(refusal.details), "printer_error": str(exc)}})
        print(text)
        return EXIT_REFUSED


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
