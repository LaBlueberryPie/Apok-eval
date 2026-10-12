"""The release guard's normaliser: every number written in a text file, reduced to one canonical string.

``tests/test_no_paris4_figures.py`` tokenises every line of every tracked text file with :func:`tokens`, hashes
each canonical form with :func:`figure_hash`, and looks the hash up in a private list (``figure_hashes.json``,
never committed: see ``release/README.md``). This module holds no figure and no hash.

The canonical form of a number:

- the absolute value (the sign is dropped, so ``-0.25`` and ``0.25`` are one form);
- written in plain positional notation: no exponent, no thousands separator, no trailing fractional zero, no
  trailing ``.``, a leading ``0`` before ``.`` for |x| < 1; an integer-valued number is written as an integer
  (``2048.0`` becomes ``2048``);
- a percentage is divided by 100 first (``12.5%`` becomes ``0.125``).

A number as written in a file: an optional sign, digits with optional thousands commas or underscores, an
optional fraction, an optional exponent, an optional ``%``; not part of an identifier or a hex digest. A hyphen
after a letter (as in ``w003``) starts no number; one after a digit (as in the range ``0.25-0.75``) does, so a
range gives two numbers. ``(4,096)`` in code is read both as ``4096`` and as ``4`` and ``96``.
"""

from __future__ import annotations

import hashlib
import re
from decimal import Decimal, InvalidOperation

NUMBER = re.compile(
    r"(?<![\w.])(?<![A-Za-z_]-)"
    r"[+\-−]?(?:\d{1,3}(?:,\d{3})+|\d+(?:_\d+)*|(?=\.\d))(?:\.\d+)?(?:[eE][+\-]?\d+)?%?"
    r"(?![\w.]\w|\w)"
)


def _canon_decimal(d: Decimal) -> str:
    d = abs(d)
    if d == d.to_integral_value():
        return str(int(d))
    return format(d.normalize(), "f")


def canonical(token: str) -> str | None:
    """The canonical string of a number written as text, or None if it is not one."""
    t = token.strip()
    pct = t.endswith("%")
    if pct:
        t = t[:-1]
    t = t.replace("−", "-").lstrip("+-").replace(",", "").replace("_", "").replace(" ", "")
    if not t:
        return None
    try:
        d = Decimal(t)
    except InvalidOperation:
        return None
    if not d.is_finite():
        return None
    if pct:
        d = d / Decimal(100)
    return _canon_decimal(d)


def sig_digits(canon: str) -> int:
    """Significant digits of a canonical string (an integer's trailing zeros do not count)."""
    if "." in canon:
        return len(canon.replace(".", "").lstrip("0"))
    return len(canon.rstrip("0")) or 1


def figure_hash(canon: str) -> str:
    """sha256 of the canonical string's UTF-8 bytes, as lowercase hex."""
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def tokens(line: str):
    """Yield ``(token, canonical)`` for each number written in ``line``."""
    for m in NUMBER.finditer(line):
        tok = m.group(0)
        c = canonical(tok)
        if c is None:
            continue
        yield tok, c
        if "," in tok:  # "(4,096)" in code is two numbers; check the parts too
            for part in tok.split(","):
                pc = canonical(part)
                if pc is not None:
                    yield part, pc
