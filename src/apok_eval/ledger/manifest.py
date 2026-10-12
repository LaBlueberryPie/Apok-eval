"""Chunk-manifest sidecars: the entries behind a recorded Merkle root.

A chunk manifest lists every object a run asked a store for (chunks and the
metadata objects ``.zgroup``, ``.zattrs``, each level's ``.zarray`` and any
sidecar ``meta.json``), in the convention of :mod:`apok_eval.merkle`:

* present: ``{"key": str, "digest": "sha256:...", "size": int}``
* absent:  ``{"key": str, "absent": true}``

The rules are :mod:`apok_eval.merkle`'s (an absent entry carries *exactly*
``key`` and ``absent``; ``size`` is a non-negative ``int`` and never a
``bool``; keys are unique); :func:`entry_problem` states them as a reason so
a refusal can name the bad entry, and a test keeps it in agreement with
``apok_eval.merkle.manifest_leaves``. The sidecar adds one byte form per set of
entries:

Sidecar format (``manifests/<name>.jsonl`` in the run directory): one entry
per line, each line the canonical JSON of the entry followed by ``\\n``, lines
sorted strictly by the UTF-8 bytes of ``key`` (the Merkle order), no blank
lines. An empty manifest is the empty file. The Merkle root is
``apok_eval.merkle.manifest_root`` of the entries.
"""

from __future__ import annotations

from typing import Any, Iterable

from apok_eval import jsonstrict
from apok_eval.canonical import canonical_dumps, is_digest
from apok_eval.ledger.util import is_size

PRESENT_KEYS = frozenset({"key", "digest", "size"})
ABSENT_KEYS = frozenset({"key", "absent"})


def entry_problem(entry: Any) -> str | None:
    """Why ``entry`` is not a valid manifest entry, or ``None``."""
    if not isinstance(entry, dict):
        return f"entry is a {type(entry).__name__}, not an object"
    key = entry.get("key")
    if not isinstance(key, str) or not key or "#" in key:
        return f"key {key!r} must be a non-empty string without '#'"
    if "absent" in entry:
        if entry["absent"] is not True or set(entry) != ABSENT_KEYS:
            return f"absent chunk {key!r} must be exactly {{key, absent: true}}"
        return None
    if set(entry) != PRESENT_KEYS:
        return f"present chunk {key!r} must carry exactly key, digest and size"
    if not is_digest(entry["digest"]):
        return f"present chunk {key!r} has digest {entry['digest']!r}, not a sha256 digest"
    if not is_size(entry["size"]):
        return f"present chunk {key!r} has size {entry['size']!r}, not a non-negative int"
    return None


def _order_key(entry: dict) -> bytes:
    return entry["key"].encode("utf-8")


def sorted_entries(entries: Iterable[Any]) -> list[dict]:
    """Validated copies of ``entries`` in Merkle order; ``ValueError`` names the first bad entry."""
    out = []
    for position, entry in enumerate(entries):
        problem = entry_problem(entry)
        if problem:
            raise ValueError(f"entry {position}: {problem}")
        out.append(dict(entry))
    out.sort(key=_order_key)
    for before, after in zip(out, out[1:]):
        if before["key"] == after["key"]:
            raise ValueError(f"key {after['key']!r} appears more than once")
    return out


def to_jsonl(entries: list[dict]) -> bytes:
    """Sidecar bytes for entries already in Merkle order."""
    return b"".join(canonical_dumps(entry) + b"\n" for entry in entries)


def parse_jsonl(data: bytes) -> list[dict]:
    """Entries of a sidecar; ``ValueError`` names the first offending line."""
    if not data:
        return []
    if not data.endswith(b"\n"):
        raise ValueError("the last line is not terminated by a newline")
    entries: list[dict] = []
    for number, line in enumerate(data[:-1].split(b"\n"), start=1):
        try:
            entry = jsonstrict.loads(line)
        except ValueError as exc:
            raise ValueError(f"line {number} is not JSON ({exc})") from exc
        problem = entry_problem(entry)
        if problem:
            raise ValueError(f"line {number}: {problem}")
        if canonical_dumps(entry) != line:
            raise ValueError(f"line {number} is not the canonical JSON of its entry")
        if entries and _order_key(entry) <= _order_key(entries[-1]):
            raise ValueError(f"line {number}: key {entry['key']!r} is out of order or repeated")
        entries.append(entry)
    return entries


def counts(entries: list[dict]) -> tuple[int, int]:
    """``(present, absent)`` entry counts."""
    absent = sum(1 for entry in entries if entry.get("absent") is True)
    return len(entries) - absent, absent
