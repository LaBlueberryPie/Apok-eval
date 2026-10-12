"""Synthetic fixtures for the store, read-log and OME-Zarr v2 tests.

Everything is generated in the test's tmp_path; nothing here touches scroll
data or any host but 127.0.0.1. Two independent writers make the volumes:
zarr-python (a real third-party v2 writer) and :func:`write_array` below, a
minimal writer that can also emit what zarr-python never does (edge chunks
cropped to the array edge, private codec ids, malformed metadata). Remote
stores are given real clearances, issued by the licence gate
(:func:`clearance_for`).
"""

from __future__ import annotations

import http.server
import itertools
import json
import threading
import urllib.parse
from pathlib import Path

import numpy as np

SHAPE0 = (37, 29, 23)  # odd extents: every level has edge chunks
N_LEVELS = 4


def planted(dtype: str) -> tuple[list, list]:
    """Distinct values planted at known level-0 and level-2 (z, y, x) positions."""
    hi = 60000 if dtype == "<u2" else 240
    level0 = [((0, 0, 0), hi + 10), ((7, 7, 7), hi + 11), ((8, 8, 8), hi + 12), ((36, 28, 22), hi + 13),
              ((20, 13, 9), hi + 14)]
    level2 = [((3, 4, 2), hi + 1), ((9, 7, 5), hi + 2), ((4, 3, 3), hi + 3)]
    return level0, level2


def pyramid(dtype: str = "|u1") -> list[np.ndarray]:
    """Expected arrays per level: a smooth background, planted values, strided 2x downsampling."""
    z, y, x = np.indices(SHAPE0)
    base = (z * 31 + y * 7 + x * 3) % 200 + 20
    if dtype == "<u2":
        base = base * 257 + 1  # exercises the high byte; stays below the planted values
    level0 = base.astype(dtype)
    plant0, plant2 = planted(dtype)
    for pos, value in plant0:
        level0[pos] = value
    levels = [level0]
    for k in range(1, N_LEVELS):
        levels.append(np.ascontiguousarray(levels[-1][::2, ::2, ::2]))
        if k == 2:
            for pos, value in plant2:
                levels[2][pos] = value
    return levels


def encode(raw: bytes, compressor: dict | None) -> bytes:
    if compressor is None:
        return raw
    import numcodecs

    return numcodecs.get_codec(dict(compressor)).encode(raw)


def chunk_path(root: Path, path: str, index: tuple[int, ...], separator: str) -> Path:
    return root.joinpath(path, *separator.join(str(i) for i in index).split("/"))


def write_array(root: Path, path: str, data: np.ndarray, chunks: tuple[int, int, int], *,
                compressor: dict | None = None, separator: str = ".", declare_separator: bool = True,
                edge: str = "full", fill_value: int | None = 0, absent: tuple = (), **meta_overrides) -> None:
    """Write one v2 array. ``edge`` is "full", "cropped" or "mixed" (alternating per edge chunk)."""
    meta = {"zarr_format": 2, "shape": list(data.shape), "chunks": list(chunks), "dtype": data.dtype.str,
            "compressor": compressor, "fill_value": fill_value, "order": "C", "filters": None}
    if declare_separator:
        meta["dimension_separator"] = separator
    meta.update(meta_overrides)
    (root / path).mkdir(parents=True, exist_ok=True)
    (root / path / ".zarray").write_text(json.dumps(meta, indent=4))
    grid = [range(-(-n // c)) for n, c in zip(data.shape, chunks)]
    for index in itertools.product(*grid):
        if index in absent:
            continue
        block = data[tuple(slice(i * c, (i + 1) * c) for i, c in zip(index, chunks))]
        full = edge == "full" or (edge == "mixed" and sum(index) % 2 == 0)
        if full and block.shape != tuple(chunks):
            padded = np.full(chunks, fill_value or 0, dtype=data.dtype)
            padded[tuple(slice(0, n) for n in block.shape)] = block
            block = padded
        target = chunk_path(root, path, index, separator)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(encode(np.ascontiguousarray(block).tobytes(), compressor))


def multiscales(n_levels: int, *, axes: tuple | None = ("z", "y", "x")) -> dict:
    ms: dict = {"version": "0.4", "name": "apok-fixture", "datasets": [
        {"path": str(k), "coordinateTransformations": [{"type": "scale", "scale": [2.0 ** k] * 3}]}
        for k in range(n_levels)]}
    if axes is not None:
        ms["axes"] = [{"name": a, "type": "space", "unit": "micrometer"} for a in axes]
    return ms


def write_group(root: Path, n_levels: int, **ms_kwargs) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (root / ".zattrs").write_text(json.dumps({"multiscales": [multiscales(n_levels, **ms_kwargs)]}))


def write_volume(root: Path, levels: list[np.ndarray], specs: list[dict], **ms_kwargs) -> Path:
    write_group(root, len(levels), **ms_kwargs)
    for k, (data, spec) in enumerate(zip(levels, specs)):
        write_array(root, str(k), data, **spec)
    return root


def edit_json(path: Path, change) -> None:
    """Apply ``change(obj)`` (in place, or returning a replacement) to the JSON file at ``path``."""
    obj = json.loads(path.read_text())
    result = change(obj)
    path.write_text(json.dumps(obj if result is None else result))


def small_levels() -> list[np.ndarray]:
    """A 3-level uint8 pyramid (9, 7, 5) -> (5, 4, 3) -> (3, 2, 2) for the negative fixtures."""
    data = (np.arange(9 * 7 * 5).reshape(9, 7, 5) % 251 + 1).astype("|u1")
    return [data, np.ascontiguousarray(data[::2, ::2, ::2]), np.ascontiguousarray(data[::4, ::4, ::4])]


def write_small(root: Path, **ms_kwargs) -> Path:
    levels = small_levels()
    specs = [{"chunks": (4, 4, 4)}, {"chunks": (4, 4, 4), "separator": "/"}, {"chunks": (2, 2, 2)}]
    return write_volume(root, levels, specs, **ms_kwargs)


# --- licence clearances ----------------------------------------------------------------------

MIT = {"name": "MIT License", "url": "https://opensource.org/license/mit", "requires_terms": [],
       "publication": "allowed", "commercial_use": True, "shippable_in_pipeline": True,
       "obligations": ["keep the notice"], "evidence": "test fixture"}


def clearance_for(tmp_path: Path, *uris: str):
    """A REAL clearance for ``uris``, issued by :func:`apok_eval.licence.check_run`.

    The registry is written to ``tmp_path``: each URI's ``scheme://host[:port]`` is a declared host
    allowing MIT with no host terms (``requires_terms: []``), and each URI is an MIT asset of its own:
    ``synthetic`` when served by the loopback http server (the registry allows synthetic assets only
    on local storage), ``dataset`` elsewhere (fictional buckets and hosts that no test contacts). MIT
    requires no terms, so the run configuration carries no ``terms_accepted_by``.
    """
    from apok_eval.licence import check_run

    hosts, assets = {}, {}
    for i, uri in enumerate(uris):
        parts = urllib.parse.urlsplit(uri)
        hosts[f"{parts.scheme}://{parts.netloc}"] = {"allowed_licences": ["MIT"], "requires_terms": [],
                                                     "note": "test host"}
        assets[f"io-test-{i}"] = {"kind": "synthetic" if parts.scheme == "http" else "dataset",
                                  "locations": [uri], "licence": "MIT", "additional_terms": [], "aliases": [],
                                  "evidence": "test fixture"}
    registry = {"schema": "apok-asset-registry-v1", "licences": {"MIT": MIT}, "terms": {}, "hosts": hosts,
                "assets": assets}
    path = tmp_path / f"registry-{len(list(tmp_path.glob('registry-*.json')))}.json"
    path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return check_run({"inputs": [{"asset": a, "uri": u} for a, u in zip(assets, uris)]}, path)


class DuckClearance:
    """NOT a clearance: an object whose covers() always answers True, recording every question."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    def covers(self, uri: str) -> bool:
        self.asked.append(uri)
        return True


# --- loopback HTTP --------------------------------------------------------------------------


class LoopbackServer:
    """An HTTP/1.0 server on 127.0.0.1 serving files under ``root`` plus scripted routes.

    ``route(path, *responses)`` scripts a path: each response is ``(status, body)``
    or ``(status, body, headers)``, served in order with the last one repeating.
    A response whose headers include ``Content-Length`` is sent with that value
    verbatim (to simulate a truncated body). Every request is appended to
    ``requests`` as ``{"path", "headers"}``.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = root
        self.routes: dict[str, list] = {}
        self.requests: list[dict] = []
        self._lock = threading.Lock()

    def route(self, path: str, *responses) -> None:
        self.routes[path] = list(responses)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def paths(self) -> list[str]:
        return [r["path"] for r in self.requests]

    def _respond(self, path: str):
        with self._lock:
            script = self.routes.get(path)
            if script:
                return script.pop(0) if len(script) > 1 else script[0]
        target = self.root.joinpath(*path.lstrip("/").split("/")) if self.root and path != "/" else None
        if target is not None and target.is_file():
            return 200, target.read_bytes()
        return 404, b"not found"

    def __enter__(self) -> "LoopbackServer":
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - http.server's naming
                with outer._lock:
                    outer.requests.append({"path": self.path, "headers": dict(self.headers.items())})
                status, body, *rest = outer._respond(self.path)
                headers = rest[0] if rest else {}
                self.send_response(status)
                self.send_header("Content-Length", headers.get("Content-Length", str(len(body))))
                for name, value in headers.items():
                    if name != "Content-Length":
                        self.send_header(name, value)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args) -> None:
                pass

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.02},
                                        daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
