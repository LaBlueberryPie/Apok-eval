"""The read log: every object a volume's reader fetched, as an apok_eval.merkle manifest.

:class:`LoggedStore` wraps a :class:`apok_eval.io.store.Store` so that every
``get`` (chunks *and* metadata objects such as ``.zgroup``, ``.zattrs``,
``<level>/.zarray`` and sidecars) is recorded in a :class:`ReadLog` before the
bytes reach the caller. Entries follow the chunk-manifest convention of
:mod:`apok_eval.merkle` exactly:

* present: ``{"key": <store key>, "digest": "sha256:<hex>", "size": <bytes>}``,
  the digest and size of the bytes exactly as stored (before any
  decompression);
* absent: ``{"key": <store key>, "absent": true}``.

:meth:`ReadLog.entries` lists them sorted by the UTF-8 bytes of the key, and
:meth:`ReadLog.root` is :func:`apok_eval.merkle.manifest_root` of that list
(scheme ``apok-merkle-v1``). Keys are store keys relative to the volume root,
so the same objects read from a local copy and from a remote store give the
same root.

A key read again must come back identical: different bytes, or present one
time and absent another, is refused (``IO_OBJECT_CHANGED``) and the log keeps
the first record.
"""

from __future__ import annotations

from apok_eval.canonical import sha256_bytes
from apok_eval.io.store import Store, check_key
from apok_eval.merkle import SCHEME, manifest_root
from apok_eval.refusal import Refusal, register_codes

register_codes(
    "apok_eval.io.readlog",
    {"IO_OBJECT_CHANGED": "an object read twice through one read log came back different"},
)


def _entry(key: str, data: bytes | None) -> dict:
    if data is None:
        return {"key": key, "absent": True}
    return {"key": key, "digest": sha256_bytes(data), "size": len(data)}


class ReadLog:
    """Manifest of every object fetched for one volume; see the module docstring."""

    def __init__(self) -> None:
        self._entries: dict[str, dict] = {}

    def record(self, key: str, data: bytes | None) -> None:
        entry = _entry(check_key(key), None if data is None else bytes(data))
        first = self._entries.setdefault(key, entry)
        if first != entry:
            raise Refusal("IO_OBJECT_CHANGED", f"{key!r} was read before as {_describe(first)} and is now "
                          f"{_describe(entry)}", key=key, before=dict(first), now=entry)

    def entries(self) -> list[dict]:
        return [dict(self._entries[k]) for k in sorted(self._entries, key=lambda k: k.encode("utf-8"))]

    def root(self) -> str:
        return manifest_root(self.entries())

    def __len__(self) -> int:
        return len(self._entries)

    def as_dict(self) -> dict:
        return {"scheme": SCHEME, "root": self.root(), "entries": self.entries()}


def _describe(entry: dict) -> str:
    return "absent" if entry.get("absent") else f"{entry['size']} bytes {entry['digest']}"


class LoggedStore:
    """A store whose every ``get`` is recorded in ``log``."""

    def __init__(self, store: Store, log: ReadLog) -> None:
        self.store = store
        self.log = log

    def get(self, key: str) -> bytes | None:
        data = self.store.get(key)
        self.log.record(key, data)
        return data

    def describe(self) -> dict:
        return self.store.describe()
