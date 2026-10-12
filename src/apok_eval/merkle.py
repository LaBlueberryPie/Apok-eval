"""Merkle root over a set of named digests (a run record's root over every
chunk read).

Scheme ``apok-merkle-v1``, specified here so it can be recomputed by anyone:

* Input: a set of ``(key, digest)`` pairs, ``key`` a non-empty ``str``
  (e.g. a Zarr chunk key ``"0/12.3.4"``), ``digest`` a ``sha256:`` string.
  Keys must be unique. Pairs are sorted by the UTF-8 bytes of ``key``, so the
  root does not depend on the order chunks were read in.
* Leaf hash: ``SHA256(0x00 || u32be(len(key_utf8)) || key_utf8 || digest32)``.
  The key is inside the leaf, so moving a digest to another key changes
  the root.
* Node hash: ``SHA256(0x01 || left32 || right32)``. The 0x00/0x01 prefixes
  keep a leaf from being passed off as an interior node.
* An odd node at the end of a level is promoted unchanged to the next level
  (never paired with a copy of itself, so ``[a, b, c]`` and ``[a, b, c, c]``
  cannot share a root).
* The root of the empty set is ``SHA256(0x02)``.
"""

from __future__ import annotations

import hashlib
import struct
from collections import Counter
from typing import Iterable

from apok_eval.canonical import DIGEST_PREFIX, digest_raw

SCHEME = "apok-merkle-v1"
_LEAF, _NODE, _EMPTY = b"\x00", b"\x01", b"\x02"


def leaf_hash(key: str, digest: str) -> bytes:
    if not isinstance(key, str) or not key:
        raise ValueError(f"Merkle leaf key must be a non-empty str, got {key!r}")
    k = key.encode("utf-8")
    return hashlib.sha256(_LEAF + struct.pack(">I", len(k)) + k + digest_raw(digest)).digest()


def node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(_NODE + left + right).digest()


def merkle_root(pairs: Iterable[tuple[str, str]]) -> str:
    """Root of ``pairs`` as a ``sha256:`` string; see the module docstring."""
    items = list(pairs)
    dupes = sorted(k for k, n in Counter(k for k, _ in items).items() if n > 1)
    if dupes:
        raise ValueError(f"duplicate Merkle keys: {dupes[:5]}")
    items.sort(key=lambda kv: kv[0].encode("utf-8"))
    level = [leaf_hash(k, d) for k, d in items]
    if not level:
        return DIGEST_PREFIX + hashlib.sha256(_EMPTY).hexdigest()
    while len(level) > 1:
        nxt = [node_hash(level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    return DIGEST_PREFIX + level[0].hex()


# --- chunk manifests --------------------------------------------------------------
#
# A chunk manifest lists every chunk a run asked a store for, one entry each:
#   present: {"key": "<array path>/<chunk key>", "digest": "sha256:...", "size": <bytes>}
#   absent:  {"key": "<array path>/<chunk key>", "absent": true}
# Digests are of the chunk's bytes exactly as stored (before decompression).
# An absent chunk is recorded as absent, never as its fill value, so a manifest
# cannot be satisfied by deleting chunks. Absent entries enter the Merkle tree
# under the key "<key>#absent" with the digest of the empty byte string; keys
# may not contain "#", so the two kinds of leaf cannot collide. A key appears
# once: a chunk recorded both present and absent is refused.

ABSENT_SUFFIX = "#absent"
_EMPTY_DIGEST = DIGEST_PREFIX + hashlib.sha256(b"").hexdigest()


def manifest_leaves(entries: Iterable[dict]) -> list[tuple[str, str]]:
    leaves = []
    seen: set[str] = set()
    for entry in entries:
        key = entry.get("key")
        if not isinstance(key, str) or not key or "#" in key:
            raise ValueError(f"chunk manifest key must be a non-empty str without '#': {key!r}")
        if key in seen:
            raise ValueError(f"chunk {key!r} appears more than once in the manifest")
        seen.add(key)
        if "absent" in entry:
            if entry["absent"] is not True or set(entry) != {"key", "absent"}:
                raise ValueError(f"absent chunk {key!r} must be exactly {{key, absent: true}}")
            leaves.append((key + ABSENT_SUFFIX, _EMPTY_DIGEST))
        else:
            size = entry.get("size")
            if set(entry) != {"key", "digest", "size"} or isinstance(size, bool) or not isinstance(size, int) \
                    or size < 0:
                raise ValueError(f"present chunk {key!r} needs exactly key, digest and a non-negative int size")
            digest_raw(entry["digest"])
            leaves.append((key, entry["digest"]))
    return leaves


def manifest_root(entries: Iterable[dict]) -> str:
    """Merkle root of a chunk manifest (see the comment above)."""
    return merkle_root(manifest_leaves(entries))
