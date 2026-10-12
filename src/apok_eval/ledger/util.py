"""Small, refusal-free building blocks shared by the ledger.

* **Strict JSON.** Everything the ledger reads back (records, sidecar lines,
  index lines, figures files) is parsed by :func:`apok_eval.jsonstrict.loads`
  with its default limits: a duplicate object key, ``NaN``/``Infinity``, a
  literal such as ``1e400`` that overflows to infinity, a lone surrogate or
  nesting deeper than ``apok_eval.jsonstrict.MAX_DEPTH`` (256) levels is a
  ``ValueError``. Python's :mod:`json` accepts all of them, and each would let
  two different files parse alike or make a later canonical serialisation
  (which recurses once per level) fail.
* **Regular files only.** The ledger reads recorded files through
  :func:`apok_eval.fileio.read_regular` and :func:`apok_eval.fileio.digest_and_size`
  (re-exported here), which open the path without blocking and refuse
  anything that is not a regular file once open (:class:`NotRegularFile`): a
  FIFO would block a reader forever and a device such as ``/dev/zero`` never
  ends.
* **Names.** Run ids and the names of inputs, models, outputs, chunk
  manifests and seeds match ``[a-z0-9][a-z0-9._-]{0,127}`` in full (a
  trailing newline does not match).
* **JSON pointers** follow RFC 6901: ``""`` is the whole document, ``/a/0``
  the first element of member ``a``; ``~1`` and ``~0`` escape ``/`` and ``~``;
  array indices are canonical decimals.
* **Safe relative paths** are POSIX, relative, with no empty, ``.`` or ``..``
  component, no backslash and no NUL.
* **Atomic publication** is :func:`apok_eval.fileio.write_atomic` (re-exported
  here): a temporary file in the target directory, fsynced, published with
  ``os.replace`` (overwrite allowed) or ``os.link`` (no-clobber), then the
  directory fsynced.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from apok_eval import jsonstrict
from apok_eval.canonical import is_text, is_utf8  # noqa: F401  (re-exported for the ledger's modules)
from apok_eval.fileio import (  # noqa: F401  (re-exported for the ledger's modules)
    TMP_PREFIX,
    NotRegularFile,
    TargetExists,
    digest_and_size,
    probe_writable,
    read_regular,
    write_atomic,
)

NAME_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_INDEX_RE = re.compile(r"0|[1-9][0-9]*")
_BAD_ESCAPE = re.compile(r"~(?![01])")


def is_name(value: Any) -> bool:
    return isinstance(value, str) and NAME_RE.fullmatch(value) is not None


def is_size(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def is_safe_relpath(value: Any) -> bool:
    if not isinstance(value, str) or "\\" in value or "\x00" in value or not is_utf8(value):
        return False
    # An empty component also rules out "", an absolute path and a trailing slash.
    return all(part not in ("", ".", "..") for part in value.split("/"))


def resolve_pointer(document: Any, pointer: str) -> Any:
    """The value ``pointer`` (RFC 6901) names in ``document``.

    Raises ``ValueError`` for a malformed pointer and ``LookupError`` when it
    does not resolve.
    """
    if not isinstance(pointer, str):
        raise ValueError(f"a JSON pointer is a string, got {type(pointer).__name__}")
    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise ValueError(f"JSON pointer {pointer!r} must be empty or start with '/'")
    current = document
    for raw in pointer[1:].split("/"):
        if _BAD_ESCAPE.search(raw):
            raise ValueError(f"JSON pointer token {raw!r} has a '~' not followed by 0 or 1")
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif isinstance(current, list) and _INDEX_RE.fullmatch(token) and int(token) < len(current):
            current = current[int(token)]
        else:
            raise LookupError(f"JSON pointer {pointer!r}: no member {token!r}")
    return current


def read_json_at(path: Path, pointer: str) -> Any:
    """The value at ``pointer`` in the strict-JSON regular file ``path``.

    The value serialises canonically: :func:`apok_eval.jsonstrict.loads` returns
    nothing else. Raises ``OSError`` (:class:`NotRegularFile` included),
    ``ValueError`` (a :class:`~apok_eval.jsonstrict.StrictJsonError` included) or
    ``LookupError``.
    """
    return resolve_pointer(jsonstrict.loads(read_regular(path)), pointer)
