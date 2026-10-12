"""Strict JSON: the one parser for every JSON document apok-eval reads, and its deterministic writer.

Python's :mod:`json` keeps the last of two equal keys without a word, so
``{"licence": "EDUCELAB-SCROLLS", "licence": "MIT"}`` reads as MIT. It also
accepts ``NaN``, ``Infinity`` and ``-Infinity``, which are not JSON; turns
``1e400`` into infinity; decodes a ``"\\ud800"`` escape to a lone surrogate,
which is not Unicode text and can be neither encoded, digested nor written
back; and raises ``RecursionError``, not ``ValueError``, on deep nesting. Two
different documents could therefore read alike, and a document could read but
not be re-serialised.

:func:`loads` parses UTF-8 **bytes** and raises :class:`StrictJsonError` (a
``ValueError`` whose message is the precise problem) for:

* bytes that are not UTF-8 (naming the byte offset), or that start with a
  byte-order mark (RFC 8259 section 8.1 forbids one);
* a syntax error (naming line and column);
* a key that appears twice in one object, at any depth;
* a number that overflows a double (``1e400``, ``-1e999``);
* an integer with more digits than ``int()`` converts
  (``sys.get_int_max_str_digits()``: 4300 unless the interpreter is configured
  otherwise; the limit guards ``int()`` against quadratic input). Shorter
  integers stay exact Python ints, for the caller to judge;
* arrays and objects nested deeper than ``max_depth`` levels, the outermost
  array or object being level 1 (default :data:`MAX_DEPTH`). Nesting deep
  enough to exhaust the parser's recursion is refused the same way: no
  ``RecursionError`` escapes;
* a string or object key holding a lone surrogate (an unpaired escape in
  ``\\ud800``-``\\udfff``), naming where by a path such as ``$.a[2]`` or
  ``$["a.b"]`` (a key that is not a plain ASCII identifier is bracketed, so
  the path names one place); a valid pair such as
  ``"\\ud83d\\ude00"`` is one character and is accepted;
* ``NaN``, ``Infinity`` and ``-Infinity`` with ``constants="reject"`` (the
  default). With ``constants="collect"`` they parse to ``nan``, ``inf`` and
  ``-inf`` and :func:`loads` returns ``(value, tokens)``, the tokens in document
  order, so a caller that must read such files (tifxyz ``meta.json``) can report
  them instead.

The parse runs first (encoding, byte-order mark, syntax, keys, constants,
numbers, recursion), then the depth check, then the lone-surrogate check; the
first problem found is the one raised.

Why :data:`MAX_DEPTH` is 256: whatever :func:`loads` returns under it can be
handed to :func:`apok_eval.canonical.canonical_dumps` and to :func:`dumps`, which
recurse once per level (CPython's default recursion limit is 1000), and no
format apok-eval reads nests beyond a handful of levels.

:func:`dumps` writes plain JSON data (:mod:`apok_eval.canonical` rules: dicts with
str keys, lists, str, int, finite float, bool, None) as sorted-key JSON
indented by 4 spaces, in raw UTF-8 (no ``\\u`` escapes), with a trailing
newline: deterministic bytes for files that C++ tools also read. It refuses,
with :class:`StrictJsonError`, anything else and anything :func:`loads` would
refuse to read back (nesting deeper than :data:`MAX_DEPTH`, an integer too long
for ``str()``, a lone surrogate), so a writer never produces bytes its own
reader refuses.
"""

from __future__ import annotations

import json
import math
import re
import sys
from typing import Any, Literal, overload

from apok_eval.canonical import canonical_dumps

MAX_DEPTH = 256

_CONSTANT_MODES = ("reject", "collect")
_NONSTANDARD = {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}
_BOM = "\N{BYTE ORDER MARK}"
# Text decoded from valid UTF-8 holds no surrogate, so only a \uD800-\uDFFF escape can produce one.
_SURROGATE_ESCAPE = re.compile(r"\\u[dD][89a-fA-F]")
_SURROGATE = re.compile("[\ud800-\udfff]")
_PLAIN_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class StrictJsonError(ValueError):
    """A document is not strict JSON (or a value cannot be written as such); ``str(exc)`` names the problem."""


def problem_of(exc: BaseException) -> str:
    """The ``problem`` a refusal records when reading a strict-JSON file failed: the parser's problem
    (``str(exc)``) for a :class:`StrictJsonError`; for anything else (an ``OSError``: missing, a FIFO, no
    permission; the ``ValueError`` of a path holding NUL) its type and message, ``"<Type>: <message>"``."""
    return str(exc) if isinstance(exc, StrictJsonError) else f"{type(exc).__name__}: {exc}"


def _excerpt(token: str) -> str:
    return token if len(token) <= 24 else f"{token[:20]}... ({len(token)} characters)"


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise StrictJsonError(f"duplicate key {key!r}")
        obj[key] = value
    return obj


def _finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise StrictJsonError(f"number {_excerpt(token)} overflows a double")
    return value


def _bounded_int(token: str) -> int:
    try:
        return int(token)
    except ValueError:  # the token is a valid integer: only the digit-count limit makes int() refuse it
        raise StrictJsonError(f"integer {_excerpt(token)} has {len(token.lstrip('-'))} digits, more than int() "
                              f"converts ({sys.get_int_max_str_digits()})") from None


def _nested_deeper_than(value: Any, limit: int) -> bool:
    """Whether arrays/objects nest more than ``limit`` levels; iterative, one level at a time."""
    level = [value] if isinstance(value, (dict, list)) else []
    depth = 0
    while level:
        depth += 1
        if depth > limit:
            return True
        level = [child for node in level for child in (node.values() if isinstance(node, dict) else node)
                 if isinstance(child, (dict, list))]
    return False


def _member(where: str, key: str) -> str:
    """The path of member ``key`` of the object at ``where``: ``.key`` for a plain identifier, else ``["key"]``.

    A key holding ``.``, ``[`` or a quote would otherwise name another place (``{"a.b": ...}`` and
    ``{"a": {"b": ...}}`` would both be ``$.a.b``), and an all-digit key would read as an array index.
    """
    return f"{where}.{key}" if _PLAIN_KEY.fullmatch(key) else f"{where}[{json.dumps(key, ensure_ascii=False)}]"


def _lone_surrogate(value: Any) -> str | None:
    """Where the first string or key, in document order, that holds a lone surrogate is; None if none does."""
    pending: list[tuple[bool, Any, str]] = [(False, value, "$")]
    while pending:
        is_key, node, where = pending.pop()
        if isinstance(node, str):
            found = _SURROGATE.search(node)
            if found:
                what = "a key of the object" if is_key else "the string"
                return f"lone surrogate {found.group()!r} in {what} at {where}"
        elif isinstance(node, dict):
            for key, child in reversed(node.items()):
                pending += [(False, child, _member(where, key)), (True, key, where)]
        elif isinstance(node, list):
            pending += [(False, child, f"{where}[{i}]") for i, child in reversed(list(enumerate(node)))]
    return None


@overload
def loads(data: bytes | bytearray, *, constants: Literal["reject"] = ..., max_depth: int = ...) -> Any: ...


@overload
def loads(data: bytes | bytearray, *, constants: Literal["collect"],
          max_depth: int = ...) -> tuple[Any, tuple[str, ...]]: ...


def loads(data: bytes | bytearray, *, constants: str = "reject", max_depth: int = MAX_DEPTH) -> Any:
    """Parse ``data`` as strict UTF-8 JSON (see the module docstring); :class:`StrictJsonError` if it is not.

    Returns the value, or ``(value, tokens)`` with ``constants="collect"``.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError(f"strict JSON is parsed from bytes, not {type(data).__name__}")
    if constants not in _CONSTANT_MODES:
        raise ValueError(f"constants must be one of {_CONSTANT_MODES}, not {constants!r}")
    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 1:
        raise ValueError(f"max_depth must be a positive int, not {max_depth!r}")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise StrictJsonError(f"not UTF-8 ({exc.reason} at byte {exc.start})") from None
    if text.startswith(_BOM):
        raise StrictJsonError("starts with a byte-order mark (U+FEFF), which JSON text must not carry")
    tokens: list[str] = []

    def constant(name: str) -> float:
        if constants == "reject":
            raise StrictJsonError(f"non-standard constant {name} (NaN, Infinity and -Infinity are not JSON)")
        tokens.append(name)
        return _NONSTANDARD[name]

    try:
        value = json.loads(text, object_pairs_hook=_unique_keys, parse_constant=constant,
                           parse_float=_finite_float, parse_int=_bounded_int)
    except json.JSONDecodeError as exc:
        raise StrictJsonError(f"invalid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}") from None
    except RecursionError:
        raise StrictJsonError(f"arrays and objects nested deeper than the parser can recurse (the limit is "
                              f"{max_depth} levels)") from None
    if _nested_deeper_than(value, max_depth):
        raise StrictJsonError(f"arrays and objects nested deeper than {max_depth} levels")
    problem = _lone_surrogate(value) if _SURROGATE_ESCAPE.search(text) else None
    if problem is not None:
        raise StrictJsonError(f"{problem}: not Unicode text (it cannot be encoded as UTF-8)")
    return (value, tuple(tokens)) if constants == "collect" else value


def dumps(value: Any) -> bytes:
    """``value`` as sorted-key, 4-space-indented, raw UTF-8 JSON with a trailing newline.

    :class:`StrictJsonError` names why ``value`` is not plain finite JSON, or why
    :func:`loads` would refuse the bytes.
    """
    try:
        canonical_dumps(value)  # the type discipline: no tuples, non-str keys, NaN or foreign scalars
        data = (json.dumps(value, indent=4, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()
    except (TypeError, ValueError, RecursionError) as exc:
        raise StrictJsonError(f"not plain finite JSON: {type(exc).__name__}: {exc}") from None
    try:
        loads(data)
    except StrictJsonError as exc:
        raise StrictJsonError(f"would not read back as strict JSON: {exc}") from None
    return data
