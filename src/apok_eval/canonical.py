"""Canonical JSON and content digests.

Every digest apok-eval records is a string ``"sha256:<64 lowercase hex>"``.

Canonical JSON (used for seals and for any digest of structured data):
UTF-8; object keys sorted by code point; no insignificant whitespace;
non-ASCII characters written as themselves; floats written by Python's
shortest round-trip ``repr``. Only ``dict`` (with ``str`` keys), ``list``,
``str``, ``int``, finite ``float``, ``bool`` and ``None`` are accepted:
anything else (tuples, NaN, infinities, non-string keys, numpy scalars)
raises ``TypeError`` or ``ValueError`` rather than being coerced, because a
silent coercion would make two different records hash alike.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from typing import Any, BinaryIO

DIGEST_PREFIX = "sha256:"
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_CHUNK = 1 << 20


def _check(obj: Any, path: str) -> None:
    if obj is None or isinstance(obj, (bool, str)):
        return
    if isinstance(obj, int):
        return
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"non-finite float at {path or '$'}")
        return
    if isinstance(obj, list):
        for i, item in enumerate(obj):
            _check(item, f"{path}[{i}]")
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            if not isinstance(key, str):
                raise TypeError(f"non-string key {key!r} at {path or '$'}")
            _check(value, f"{path}.{key}")
        return
    raise TypeError(f"{type(obj).__name__} is not canonical JSON (at {path or '$'})")


def canonical_dumps(obj: Any) -> bytes:
    """Serialise ``obj`` to canonical JSON bytes."""
    _check(obj, "")
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return DIGEST_PREFIX + hashlib.sha256(data).hexdigest()


def sha256_stream(fh: BinaryIO) -> str:
    h = hashlib.sha256()
    for block in iter(lambda: fh.read(_CHUNK), b""):
        h.update(block)
    return DIGEST_PREFIX + h.hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    with open(path, "rb") as fh:
        return sha256_stream(fh)


def digest_json(obj: Any) -> str:
    """Digest of the canonical JSON serialisation of ``obj``."""
    return sha256_bytes(canonical_dumps(obj))


def is_utf8(value: str) -> bool:
    """False for strings holding lone surrogates (e.g. undecodable file names), which canonical JSON cannot encode."""
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def is_text(value: Any) -> bool:
    """A UTF-8-encodable string that is not empty or only whitespace."""
    return isinstance(value, str) and bool(value.strip()) and is_utf8(value)


def is_digest(value: Any) -> bool:
    return isinstance(value, str) and bool(_DIGEST_RE.match(value))


def digest_raw(value: str) -> bytes:
    """The 32 raw bytes of a ``sha256:`` digest string."""
    if not is_digest(value):
        raise ValueError(f"not a sha256 digest string: {value!r}")
    return bytes.fromhex(value[len(DIGEST_PREFIX):])
