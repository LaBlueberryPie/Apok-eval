"""OME-Zarr v2 volumes, read by apok-eval's own reader (zarr-python is only a test oracle).

A volume is a Zarr v2 group used as OME-NGFF 0.4 multiscales: a root
``.zgroup`` and ``.zattrs`` and one array per pyramid level. Villa writes and
reads this layout (facts and file:line citations at villa ``fb8c2c4`` are in
``docs/io-zarr.md``). :func:`open_volume` validates everything below before any
chunk is read, and refuses on the first violation; nothing is inferred or
defaulted except where stated and recorded.

Group
    ``.zgroup`` must exist with ``zarr_format`` the integer 2. ``.zattrs`` must
    exist with ``multiscales`` a non-empty list; only ``multiscales[0]`` is read.
    Metadata objects must be strict JSON objects (:func:`apok_eval.jsonstrict.loads`:
    UTF-8 without BOM, no duplicate keys, no NaN/Infinity, no number that
    overflows a double, no integer past ``int()``'s digit limit, at most 256
    levels of nesting, no lone-surrogate escape such as ``"\\ud800"``, which
    parses but is not Unicode and could not be written back as UTF-8)
    (``IO_OBJECT_NOT_JSON``).
    The sidecars ``meta.json`` and ``metadata.json`` are fetched (present or
    absent) so they enter the read log, and are not interpreted.
Axes
    When ``multiscales[0].axes`` is present it must be exactly z, y, x (strings
    or objects with a ``name``). When absent, the caller must declare the order
    (``axes_declared``) and who declares it (``axes_declared_by``), else
    ``IO_ZARR_AXES_UNDECLARED``; a declared order other than exactly
    ``("z", "y", "x")`` is ``IO_ZARR_AXES`` whether or not the store has axes,
    and the declaration is recorded. A level's own ``.zattrs``, when present, must
    also name z, y, x in any of ``_ARRAY_DIMENSIONS``, ``dimension_names`` and
    ``axes`` it carries (villa ZarrChunkFetcher.cpp validateArrayAxes).
Datasets and transforms (villa's strict rules, ZarrChunkFetcher.cpp:607-772)
    Dataset paths must be exactly ``"0", "1", ... "N"`` in order (canonical
    numerals, no gaps, duplicates or reordering), with ``N <= 31`` as in villa
    (ZarrChunkFetcher.cpp:695-696), so ``2**k`` stays small. Multiscales-level
    ``coordinateTransformations`` are refused. Per dataset, transforms are
    either absent or a non-empty list holding exactly one ``scale`` and at most
    one ``translation``, each three finite numbers; either every level declares
    transforms or none does. Scales must be positive, isotropic and dyadic:
    ``scale_k = scale_0 * 2**k``, compared to a relative 1e-9 at every
    magnitude. Translations must be 0: below 1e-9 of the level's own voxel
    (``|t| <= 1e-9 * scale``) on every axis. Both tolerances are unit-free, so
    the verdict does not depend on the unit a writer chose; villa's tolerance
    (ZarrChunkFetcher.cpp:601-605) is absolute below 1. Scale values are never
    read as physical units: the voxel size is the declared
    :class:`~apok_eval.io.frame.VolumeFrame`'s.
Arrays (``<k>/.zarray``)
    All eight v2 keys must be present; ``zarr_format`` 2; ``shape`` and
    ``chunks`` three positive integers; ``dtype`` ``"|u1"`` or ``"<u2"`` and the
    same on every level; ``order`` ``"C"``; ``filters`` null or empty;
    ``dimension_separator`` absent (-> ``"."``, recorded) or ``"."``/``"/"``;
    ``fill_value`` an integral number within the dtype (``0`` or ``0.0``; read
    as the integer, the declared value recorded), or null (-> 0, recorded);
    ``compressor`` null or one of blosc (cname blosclz/lz4/lz4hc/zlib/zstd),
    zstd, zlib, gzip, lz4, decoded through numcodecs (``IO_CODEC_UNAVAILABLE``
    names the codec when numcodecs cannot provide it). Villa's repo-private
    codecs (vc-delta3d, vcz1, c3d) and every other id are refused.
Frame and pyramid
    Level 0's shape must equal the declared frame's ``shape_zyx``. Level k must
    have shape ``ceil(shape_0 / 2**k)`` per axis, which is the same as
    ``ceil(shape_{k-1} / 2)`` at every level; with
    ``pyramid_tolerance="chunk-padding"`` it may instead exceed that by less
    than one of its own storage chunks per axis (villa's tolerance,
    ZarrChunkFetcher.cpp:590-593, 936-948), and the padding is recorded.
Chunks
    The chunk with grid index ``(a, b, c)`` of level ``k`` is stored under
    ``"<k>/" + sep.join((a, b, c))``. An absent chunk reads as the level's fill
    value. A stored chunk decodes either to the full chunk shape (whose
    padding beyond the array edge is never read) or, at an array edge, to
    exactly the cropped shape; any other byte count is ``IO_ZARR_CHUNK_SIZE``.
    Bytes are C order.
    A Blosc frame's 16-byte header is checked before decoding: its ``cbytes``
    must equal the stored object's length (and ``nbytes + 16`` when the frame
    is stored uncompressed), else ``IO_CODEC_DECODE``; its ``nbytes`` must be
    one of the two allowed sizes, else ``IO_ZARR_CHUNK_SIZE``. (numcodecs
    would otherwise decode a truncated uncompressed frame from memory past
    its end.)
    No chunk decodes to more than the full chunk's byte count plus one, so a
    small hostile chunk (a decompression bomb) is refused, not allocated: a
    zstd frame must declare its content size and an lz4 object its 4-byte size
    prefix, and that size must be one of the two allowed sizes, before
    decoding (``IO_CODEC_DECODE`` / ``IO_ZARR_CHUNK_SIZE`` with ``stage:
    "header"``); zlib and gzip streams are inflated with an output bound and
    refused (``IO_ZARR_CHUNK_SIZE``, ``stage: "decoded"``, ``size_at_least``)
    when more remains.
    :func:`chunk_codec` (the allow-list) and :func:`decode_chunk` (the checks
    of this section) work without a :class:`Volume`; the CT positive control
    reads its chunks through them (:mod:`apok_eval.eval.cli`).
Coordinates
    :meth:`Volume.read` takes a half-open box ``[start, stop)`` in LEVEL-0
    voxels. A box reaching outside the level-0 shape is refused unless
    ``clip=True``, which is recorded. Level-0 index ``i`` maps to level-k
    index ``floor(i / 2**k)``, and level-k index ``j`` covers level-0
    ``[j * 2**k, (j + 1) * 2**k)``; the level-k box read is therefore
    ``[floor(start / 2**k), ceil(stop / 2**k))``. A box is an ordered (z, y, x)
    triple: a tuple, list or 1-D array of integers, never a set, dict or other
    iterable. Every read returns a :class:`TransformRecord` stating the level,
    both boxes, the level-0 extent the returned voxels cover, the mapping,
    :data:`VOXEL_CONVENTION`, how many chunks it covered and which of them were
    absent and read as the (recorded) fill value: a read made wholly of fill,
    as when the stored chunk layout and the separator disagree, shows there.

Every object fetched (metadata and chunks) is recorded in the volume's
:class:`apok_eval.io.readlog.ReadLog`.
"""

from __future__ import annotations

import copy
import dataclasses
import gzip
import io
import itertools
import math
import struct
import zlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from apok_eval import jsonstrict
from apok_eval.io.frame import AXES, VolumeFrame, finite_number
from apok_eval.io.readlog import LoggedStore, ReadLog
from apok_eval.io.store import Store
from apok_eval.merkle import SCHEME
from apok_eval.refusal import Refusal, register_codes

register_codes(
    "apok_eval.io.omezarr",
    {
        "IO_OBJECT_NOT_JSON": "a metadata object is not a strict UTF-8 JSON object",
        "IO_ZARR_ARGUMENT_INVALID": "an argument to the OME-Zarr reader is not of the required kind",
        "IO_ZARR_GROUP": "the store root is not a Zarr v2 group",
        "IO_ZARR_MULTISCALES": "the group's multiscales metadata is missing or malformed",
        "IO_ZARR_AXES": "declared axes are not exactly z, y, x",
        "IO_ZARR_AXES_UNDECLARED": "the store declares no axes and the caller declared none",
        "IO_ZARR_DATASET_PATHS": "multiscales dataset paths are not exactly '0'..'N' (N <= 31) in order",
        "IO_ZARR_TRANSFORM": "coordinate transformations are malformed or unsupported",
        "IO_ZARR_TRANSLATION": "a coordinate translation is not zero",
        "IO_ZARR_SCALE": "a coordinate scale is not positive and isotropic",
        "IO_ZARR_SCALE_NOT_DYADIC": "a level's scale is not 2**k times level 0's",
        "IO_ZARR_ARRAY_META": "a level's .zarray is missing or malformed",
        "IO_ZARR_DTYPE": "a level's dtype is not |u1 or <u2, or differs between levels",
        "IO_ZARR_ORDER": "a level's memory order is not C",
        "IO_ZARR_FILTERS": "a level declares filters",
        "IO_ZARR_SEPARATOR": "a level's dimension_separator is not '.' or '/'",
        "IO_ZARR_FILL": "a level's fill_value is neither null nor an integral number within its dtype",
        "IO_CODEC_UNSUPPORTED": "a level's compressor is not on the allow-list",
        "IO_CODEC_UNAVAILABLE": "an allow-listed compressor cannot be loaded (numcodecs missing)",
        "IO_CODEC_DECODE": "a stored chunk failed to decode",
        "IO_ZARR_CHUNK_SIZE": "a decoded chunk is neither the full nor the edge-cropped chunk size, or, in the CT "
                              "control's reads (full-size chunks only), not the full size",
        "IO_ZARR_SHAPE_MISMATCH": "level 0's shape differs from the declared frame",
        "IO_ZARR_PYRAMID_SHAPE": "a level's shape is not the ceil-halving of level 0 (within tolerance)",
        "IO_ZARR_LEVEL_INVALID": "a read asked for a level the volume does not have",
        "IO_ZARR_BOX_INVALID": "a read box is not two (z, y, x) integer triples with start < stop",
        "IO_ZARR_BOX_OUTSIDE": "a read box reaches outside the level-0 shape",
    },
)

DTYPES = {"|u1": np.dtype("|u1"), "<u2": np.dtype("<u2")}
CODECS = {"blosc": "Blosc", "zstd": "Zstd", "zlib": "Zlib", "gzip": "GZip", "lz4": "LZ4"}  # id -> numcodecs class
BLOSC_CNAMES = ("blosclz", "lz4", "lz4hc", "zlib", "zstd")
PRIVATE_CODECS = ("vc-delta3d", "vcz1", "c3d")
PYRAMID_TOLERANCES = ("exact", "chunk-padding")
SIDECARS = ("meta.json", "metadata.json")
MAX_LEVELS = 32  # dataset paths "0".."31", villa's range (ZarrChunkFetcher.cpp:695-696 @fb8c2c4)
TOLERANCE = 1e-9
ARRAY_KEYS = ("zarr_format", "shape", "chunks", "dtype", "compressor", "fill_value", "order", "filters")
ARRAY_AXIS_ATTRS = ("_ARRAY_DIMENSIONS", "dimension_names", "axes")
_BLOSC_HEADER = struct.Struct("<BBBBIII")
_BLOSC_MEMCPYED = 0x02
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"  # RFC 8878 section 3.1.1, little-endian 0xFD2FB528

MAPPING = ("level-k index j = floor(i / 2**k) for level-0 index i; "
           "level-k index j covers level-0 indices [j * 2**k, (j + 1) * 2**k)")
VOXEL_CONVENTION = (
    "apok-voxel-v1: index i on any level names the voxel occupying [i, i + 1) of that level's grid, and a "
    "level-k voxel stands for the level-0 block it covers; whether its value is the level-0 sample at "
    "j * 2**k (strided) or an aggregate of the block (mean, max) is UNKNOWN, because the downsampling "
    "method is neither read nor checked")
ASSUMPTIONS = (
    VOXEL_CONVENTION,
    "units: .zattrs scales are checked only for positivity, isotropy and dyadic ratios; no unit or voxel size "
    "is read from the store (writers label relative factors 'micrometer'); the voxel size is the declared frame's",
    "pyramid: level k is taken to be a 2**k-per-axis downsample of level 0, as its shape (and its scale, when "
    "declared) is checked to be",
    "absent chunks read as their level's fill value; by value, absent, masked and true-fill voxels cannot be "
    "told apart, so the read log records every absent chunk as absent",
)


# --- strict JSON ----------------------------------------------------------------------


def _json_object(data: bytes, key: str) -> dict:
    try:
        obj = jsonstrict.loads(data)
    except jsonstrict.StrictJsonError as exc:
        raise Refusal("IO_OBJECT_NOT_JSON", f"{key} is not strict UTF-8 JSON: {exc}", key=key,
                      problem=str(exc)) from None
    if not isinstance(obj, dict):
        raise Refusal("IO_OBJECT_NOT_JSON", f"{key} holds a JSON {type(obj).__name__}, not an object", key=key)
    return obj


# --- small value checks ---------------------------------------------------------------


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_triple(value: Any) -> tuple[int, int, int] | None:
    if isinstance(value, list) and len(value) == 3 and all(_is_int(v) and v > 0 for v in value):
        return (value[0], value[1], value[2])
    return None


def _close(a: float, b: float) -> bool:
    """Both finite and equal to a relative TOLERANCE. villa's finiteMetadataEqual (ZarrChunkFetcher.cpp:601-605
    @fb8c2c4) bounds by max(1, |a|, |b|), which is absolute below 1 and passed 1e-10, 3e-10, 9e-10 as dyadic."""
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= TOLERANCE * max(abs(a), abs(b))


def _index_triple(value: Any) -> tuple[int, int, int] | None:
    """``value`` as three ints if it is an ordered triple: a tuple, list or 1-D array of Python or NumPy
    integers (not booleans). A set, dict, generator or other iterable has no (z, y, x) order of its own."""
    if not (isinstance(value, (tuple, list)) or (isinstance(value, np.ndarray) and value.ndim == 1)):
        return None
    items = list(value)
    if len(items) == 3 and all(isinstance(i, (int, np.integer)) and not isinstance(i, (bool, np.bool_))
                               for i in items):
        return (int(items[0]), int(items[1]), int(items[2]))
    return None


# --- group and multiscales ------------------------------------------------------------


def _check_group(data: bytes | None) -> None:
    if data is None:
        raise Refusal("IO_ZARR_GROUP", ".zgroup is absent: the store root is not a Zarr v2 group", key=".zgroup")
    fmt = _json_object(data, ".zgroup").get("zarr_format")
    if not (_is_int(fmt) and fmt == 2):
        raise Refusal("IO_ZARR_GROUP", f".zgroup zarr_format is {fmt!r}, not 2", key=".zgroup",
                      field="zarr_format", value=fmt)


def _multiscales(data: bytes | None) -> tuple[dict, int]:
    """``multiscales[0]`` (with a checked dataset list) and the number of descriptors."""
    if data is None:
        raise Refusal("IO_ZARR_MULTISCALES", ".zattrs is absent: no multiscales metadata", key=".zattrs",
                      field=".zattrs")
    descriptors = _json_object(data, ".zattrs").get("multiscales")
    if not isinstance(descriptors, list) or not descriptors or not isinstance(descriptors[0], dict):
        raise Refusal("IO_ZARR_MULTISCALES", ".zattrs multiscales is not a non-empty list of objects",
                      key=".zattrs", field="multiscales")
    ms = descriptors[0]
    if "coordinateTransformations" in ms:
        raise Refusal("IO_ZARR_TRANSFORM", "multiscales-level coordinateTransformations are not supported",
                      key=".zattrs", field="multiscales[0].coordinateTransformations", rule="per-dataset-only")
    datasets = ms.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise Refusal("IO_ZARR_MULTISCALES", "multiscales[0].datasets is not a non-empty list", key=".zattrs",
                      field="multiscales[0].datasets")
    if len(datasets) > MAX_LEVELS:
        raise Refusal("IO_ZARR_DATASET_PATHS", f"multiscales[0] lists {len(datasets)} datasets; at most {MAX_LEVELS} "
                      f"(paths '0'..'{MAX_LEVELS - 1}', villa's range) are read", key=".zattrs",
                      field="multiscales[0].datasets", count=len(datasets), limit=MAX_LEVELS)
    for i, dataset in enumerate(datasets):
        path = dataset.get("path") if isinstance(dataset, dict) else None
        if not isinstance(path, str):
            raise Refusal("IO_ZARR_MULTISCALES", f"datasets[{i}] is not an object with a string path",
                          key=".zattrs", field=f"multiscales[0].datasets[{i}].path")
        if path != str(i):
            raise Refusal("IO_ZARR_DATASET_PATHS",
                          f"datasets[{i}].path is {path!r}; paths must be '0'..'{len(datasets) - 1}' in order "
                          "(canonical numerals, no gaps, duplicates or reordering)", key=".zattrs", index=i,
                          path=path, expected=str(i))
    return ms, len(descriptors)


def _check_axes(axes: Any, field: str, key: str) -> None:
    names = None
    if isinstance(axes, list):
        names = [a if isinstance(a, str) else a.get("name") if isinstance(a, dict) else None for a in axes]
    if names != list(AXES):
        raise Refusal("IO_ZARR_AXES", f"{key} {field} is {axes!r}; only axes exactly (z, y, x) are read",
                      key=key, field=field, axes=axes)


def _axes_record(ms: dict, declared: tuple[str, ...] | None, declared_by: str | None) -> dict:
    """Where the (z, y, x) order comes from, with the caller's declaration (already checked), if any."""
    record = {"order": list(AXES), "source": ".zattrs", "declared": None if declared is None else list(declared),
              "declared_by": declared_by}
    if "axes" in ms:
        _check_axes(ms["axes"], "multiscales[0].axes", ".zattrs")
        return record
    if declared is None:
        raise Refusal("IO_ZARR_AXES_UNDECLARED", "multiscales[0] has no axes and the caller declared none "
                      "(pass axes_declared=('z', 'y', 'x') and axes_declared_by naming who declares it)",
                      key=".zattrs", field="multiscales[0].axes")
    return {**record, "source": "declared"}


def _transforms(dataset: dict, level: int) -> tuple[tuple[float, ...] | None, tuple[float, ...] | None]:
    """(scale, translation) declared for one dataset; (None, None) when it declares none."""
    if "coordinateTransformations" not in dataset:
        return None, None
    field = f"multiscales[0].datasets[{level}].coordinateTransformations"
    entries = dataset["coordinateTransformations"]
    if not isinstance(entries, list) or not entries:
        raise Refusal("IO_ZARR_TRANSFORM", f"{field} is not a non-empty list", key=".zattrs", field=field,
                      rule="non-empty-list")
    found: dict[str, tuple[float, ...]] = {}
    for entry in entries:
        kind = entry.get("type") if isinstance(entry, dict) else None
        if kind not in ("scale", "translation") or kind in found:
            raise Refusal("IO_ZARR_TRANSFORM", f"{field} holds an unsupported or repeated transform {kind!r}; "
                          "only one scale and at most one translation are read", key=".zattrs", field=field,
                          rule="type", type=kind)
        values = entry.get(kind)
        numbers = [finite_number(v) for v in values] if isinstance(values, list) and len(values) == 3 else [None]
        if None in numbers:
            raise Refusal("IO_ZARR_TRANSFORM", f"{field} {kind} is {values!r}, not three finite numbers",
                          key=".zattrs", field=field, rule="three-finite-numbers", type=kind)
        found[kind] = tuple(numbers)
    if "scale" not in found:
        raise Refusal("IO_ZARR_TRANSFORM", f"{field} has no scale", key=".zattrs", field=field, rule="one-scale")
    return found["scale"], found.get("translation")


def _check_transforms(transforms: list[tuple[tuple[float, ...] | None, tuple[float, ...] | None]]) -> None:
    declared = [scale is not None for scale, _ in transforms]
    if any(declared) and not all(declared):
        level = declared.index(False)
        raise Refusal("IO_ZARR_TRANSFORM", f"level {level} declares no coordinateTransformations while others do",
                      key=".zattrs", field=f"multiscales[0].datasets[{level}].coordinateTransformations",
                      rule="all-or-none", level=level)
    for level, (scale, translation) in enumerate(transforms):
        if scale is None:  # and so no translation: _transforms requires a scale beside one
            continue
        if min(scale) <= 0:
            raise Refusal("IO_ZARR_SCALE", f"level {level} scale {list(scale)} is not positive", key=".zattrs",
                          level=level, rule="positive", scale=list(scale))
        if translation is not None and not all(abs(t) <= TOLERANCE * s for t, s in zip(translation, scale)):
            raise Refusal("IO_ZARR_TRANSLATION", f"level {level} translation is {list(translation)}, not zero "
                          f"(within {TOLERANCE} of its voxel, scale {list(scale)})", key=".zattrs", level=level,
                          translation=list(translation))
        if not (_close(scale[0], scale[1]) and _close(scale[0], scale[2])):
            raise Refusal("IO_ZARR_SCALE", f"level {level} scale {list(scale)} is not isotropic", key=".zattrs",
                          level=level, rule="isotropic", scale=list(scale))
        expected = transforms[0][0][0] * 2 ** level
        if not _close(scale[0], expected):
            raise Refusal("IO_ZARR_SCALE_NOT_DYADIC", f"level {level} scale {scale[0]!r} is not 2**{level} x "
                          f"level 0's ({expected!r})", key=".zattrs", level=level, scale=list(scale),
                          expected=expected)


# --- arrays ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Level:
    """One validated pyramid level."""

    path: str
    factor: int  # 2**k
    shape: tuple[int, int, int]
    chunks: tuple[int, int, int]
    dtype: str
    separator: str
    separator_declared: bool
    fill_value: int
    fill_value_declared: int | float | None
    compressor: dict | None
    scale: tuple[float, ...] | None
    translation: tuple[float, ...] | None
    codec: Any = dataclasses.field(repr=False, compare=False)
    padding: tuple[int, int, int] = (0, 0, 0)

    def as_dict(self) -> dict:
        return {
            "path": self.path, "factor": self.factor, "shape": list(self.shape), "chunks": list(self.chunks),
            "dtype": self.dtype, "separator": self.separator, "separator_declared": self.separator_declared,
            "fill_value": self.fill_value, "fill_value_declared": self.fill_value_declared,
            "compressor": copy.deepcopy(self.compressor), "padding": list(self.padding),
            "scale": None if self.scale is None else list(self.scale),
            "translation": None if self.translation is None else list(self.translation),
        }


def chunk_codec(config: Any, key: str) -> Any:
    """A decoder for an allow-listed compressor config; None for no compressor. ``key`` names the ``.zarray`` the
    config comes from. Refused unless allow-listed (``IO_CODEC_UNSUPPORTED``) or when numcodecs cannot provide it
    (``IO_CODEC_UNAVAILABLE``)."""
    if config is None:
        return None
    cid = config.get("id") if isinstance(config, dict) else None
    if not isinstance(cid, str) or cid not in CODECS:
        why = "is a villa repo-private codec" if cid in PRIVATE_CODECS else "is not on the allow-list"
        raise Refusal("IO_CODEC_UNSUPPORTED", f"{key} compressor {config!r} {why} (null, "
                      f"{', '.join(CODECS)})", key=key, codec=cid)
    if cid == "blosc" and config.get("cname") not in BLOSC_CNAMES:
        raise Refusal("IO_CODEC_UNSUPPORTED", f"{key} blosc cname {config.get('cname')!r} is not one of "
                      f"{', '.join(BLOSC_CNAMES)}", key=key, codec=cid, cname=config.get("cname"))
    try:
        import numcodecs

        return getattr(numcodecs, CODECS[cid])()
    except (ImportError, AttributeError):
        raise Refusal("IO_CODEC_UNAVAILABLE", f"{key} needs codec {cid!r}, which numcodecs cannot provide here "
                      "(install numcodecs)", key=key, codec=cid) from None


def _check_array_axes(data: bytes | None, key: str) -> None:
    if data is None:
        return
    attrs = _json_object(data, key)
    for name in ARRAY_AXIS_ATTRS:
        if name in attrs:
            _check_axes(attrs[name], name, key)


def _open_level(store: LoggedStore, level: int, transform: tuple) -> Level:
    path = str(level)
    key = f"{path}/.zarray"
    data = store.get(key)
    if data is None:
        raise Refusal("IO_ZARR_ARRAY_META", f"{key} is absent", key=key, field=".zarray")
    meta = _json_object(data, key)
    missing = [name for name in ARRAY_KEYS if name not in meta]
    if missing:
        raise Refusal("IO_ZARR_ARRAY_META", f"{key} lacks {missing}", key=key, field=missing[0])
    if not (_is_int(meta["zarr_format"]) and meta["zarr_format"] == 2):
        raise Refusal("IO_ZARR_ARRAY_META", f"{key} zarr_format is {meta['zarr_format']!r}, not 2", key=key,
                      field="zarr_format", value=meta["zarr_format"])
    shape, chunks = _positive_triple(meta["shape"]), _positive_triple(meta["chunks"])
    for name, value in (("shape", shape), ("chunks", chunks)):
        if value is None:
            raise Refusal("IO_ZARR_ARRAY_META", f"{key} {name} is {meta[name]!r}, not three positive integers",
                          key=key, field=name, value=meta[name])
    dtype = meta["dtype"]
    if not isinstance(dtype, str) or dtype not in DTYPES:
        raise Refusal("IO_ZARR_DTYPE", f"{key} dtype {dtype!r} is not one of {sorted(DTYPES)}", key=key,
                      field="dtype", value=dtype)
    if meta["order"] != "C":
        raise Refusal("IO_ZARR_ORDER", f"{key} order is {meta['order']!r}; only C order is read", key=key,
                      field="order", value=meta["order"])
    if meta["filters"] not in (None, []):
        raise Refusal("IO_ZARR_FILTERS", f"{key} declares filters {meta['filters']!r}; none are applied, so "
                      "none may be declared", key=key, field="filters", value=meta["filters"])
    separator = meta.get("dimension_separator", ".")
    if separator not in (".", "/"):
        raise Refusal("IO_ZARR_SEPARATOR", f"{key} dimension_separator is {separator!r}, not '.' or '/'",
                      key=key, field="dimension_separator", value=separator)
    fill = meta["fill_value"]
    integral = _is_int(fill) or (isinstance(fill, float) and fill.is_integer())
    if fill is not None and not (integral and 0 <= fill <= np.iinfo(DTYPES[dtype]).max):
        raise Refusal("IO_ZARR_FILL", f"{key} fill_value {fill!r} is neither null nor an integral number within "
                      f"{dtype}", key=key, field="fill_value", value=fill)
    codec = chunk_codec(meta["compressor"], key)
    _check_array_axes(store.get(f"{path}/.zattrs"), f"{path}/.zattrs")
    return Level(path=path, factor=2 ** level, shape=shape, chunks=chunks, dtype=dtype, separator=separator,
                 separator_declared="dimension_separator" in meta, fill_value=0 if fill is None else int(fill),
                 fill_value_declared=fill, compressor=meta["compressor"], scale=transform[0],
                 translation=transform[1], codec=codec)


def _with_padding(levels: list[Level], tolerance: str) -> list[Level]:
    """``levels`` with their padding over ``ceil(shape_0 / 2**k)`` recorded, or a refusal."""
    out = []
    for lv in levels:
        expected = tuple(-(-n // lv.factor) for n in levels[0].shape)
        padding = tuple(s - e for s, e in zip(lv.shape, expected))
        limit = lv.chunks if tolerance == "chunk-padding" else (1, 1, 1)
        for axis, (pad, lim) in enumerate(zip(padding, limit)):
            if not 0 <= pad < lim:
                raise Refusal("IO_ZARR_PYRAMID_SHAPE",
                              f"level {lv.path} shape {list(lv.shape)} is not ceil(level-0 shape / {lv.factor}) = "
                              f"{list(expected)} on axis {AXES[axis]} (tolerance {tolerance!r})",
                              key=f"{lv.path}/.zarray", level=int(lv.path), axis=AXES[axis], shape=list(lv.shape),
                              expected=list(expected), tolerance=tolerance)
        out.append(dataclasses.replace(lv, padding=padding))
    return out


# --- the volume -----------------------------------------------------------------------


@dataclass(frozen=True)
class TransformRecord:
    """What one :meth:`Volume.read` returned, in both grids. Boxes are half-open [start, stop)."""

    volume_id: str
    level: int
    factor: int
    requested_start: tuple[int, int, int]
    requested_stop: tuple[int, int, int]
    clip_requested: bool
    clipped: bool
    level0_start: tuple[int, int, int]
    level0_stop: tuple[int, int, int]
    level_start: tuple[int, int, int]
    level_stop: tuple[int, int, int]
    covered_start: tuple[int, int, int]
    covered_stop: tuple[int, int, int]
    chunks: int  # chunk keys the level-k box covers, each fetched
    absent_chunks: tuple[str, ...]  # those found absent, in grid order, read as fill_value
    fill_value: int
    mapping: str = MAPPING
    voxel_convention: str = VOXEL_CONVENTION

    def as_dict(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in dataclasses.asdict(self).items()}


class Volume:
    """A validated OME-Zarr v2 volume; open it with :func:`open_volume`."""

    def __init__(self, store: LoggedStore, frame: VolumeFrame, levels: list[Level], axes: dict,
                 multiscales: dict, pyramid_tolerance: str) -> None:
        self._store = store
        self.frame = frame
        self.levels = levels
        self.axes = axes
        self.multiscales = multiscales
        self.pyramid_tolerance = pyramid_tolerance

    @property
    def log(self) -> ReadLog:
        return self._store.log

    def read(self, zyx_start: Any, zyx_stop: Any, level: int = 0, *,
             clip: bool = False) -> tuple[np.ndarray, TransformRecord]:
        """Voxels of ``level`` covering the level-0 box ``[zyx_start, zyx_stop)``; see the module docstring."""
        start, stop = _index_triple(zyx_start), _index_triple(zyx_stop)
        if start is None or stop is None or any(a >= b for a, b in zip(start, stop)):
            raise Refusal("IO_ZARR_BOX_INVALID", f"box {zyx_start!r}..{zyx_stop!r} is not two (z, y, x) integer "
                          "triples with start < stop on every axis", start=repr(zyx_start), stop=repr(zyx_stop))
        if not (_is_int(level) and 0 <= level < len(self.levels)):
            raise Refusal("IO_ZARR_LEVEL_INVALID", f"level {level!r} is not one of 0..{len(self.levels) - 1}",
                          level=repr(level), levels=len(self.levels))
        if not isinstance(clip, bool):
            raise Refusal("IO_ZARR_ARGUMENT_INVALID", f"clip must be True or False, not {clip!r}", field="clip")
        shape = self.frame.shape_zyx
        lo = tuple(max(a, 0) for a in start)
        hi = tuple(min(b, n) for b, n in zip(stop, shape))
        clipped = (lo, hi) != (start, stop)
        if clipped and not clip:
            raise Refusal("IO_ZARR_BOX_OUTSIDE", f"box {list(start)}..{list(stop)} reaches outside the level-0 "
                          f"shape {list(shape)} of volume {self.frame.volume_id} (pass clip=True to clip)",
                          start=list(start), stop=list(stop), shape=list(shape))
        if any(a >= b for a, b in zip(lo, hi)):
            raise Refusal("IO_ZARR_BOX_OUTSIDE", f"box {list(start)}..{list(stop)} lies wholly outside the level-0 "
                          f"shape {list(shape)} of volume {self.frame.volume_id}", start=list(start),
                          stop=list(stop), shape=list(shape))
        lv = self.levels[level]
        level_start = tuple(a // lv.factor for a in lo)
        level_stop = tuple(-(-b // lv.factor) for b in hi)
        array, chunks, absent = self._read_box(lv, level_start, level_stop)
        record = TransformRecord(
            volume_id=self.frame.volume_id, level=level, factor=lv.factor, requested_start=start,
            requested_stop=stop, clip_requested=clip, clipped=clipped, level0_start=lo, level0_stop=hi,
            level_start=level_start, level_stop=level_stop,
            covered_start=tuple(j * lv.factor for j in level_start),
            covered_stop=tuple(min(j * lv.factor, n) for j, n in zip(level_stop, shape)),
            chunks=chunks, absent_chunks=tuple(absent), fill_value=lv.fill_value,
        )
        return array, record

    def _read_box(self, lv: Level, start: tuple[int, ...],
                  stop: tuple[int, ...]) -> tuple[np.ndarray, int, list[str]]:
        """Level-k voxels ``[start, stop)``, the number of chunks covered and the keys of the absent ones.
        ``stop`` never exceeds the level's shape, so the padding of a full-size stored edge chunk is never
        copied, and each chunk's part of the box ends at ``min(stop, origin + chunks)`` whichever form it has."""
        out = np.empty(tuple(b - a for a, b in zip(start, stop)), dtype=DTYPES[lv.dtype])
        grid = [range(a // c, (b - 1) // c + 1) for a, b, c in zip(start, stop, lv.chunks)]
        indices = list(itertools.product(*grid))
        absent = []
        for index in indices:
            origin = tuple(i * c for i, c in zip(index, lv.chunks))
            lo = tuple(max(a, o) for a, o in zip(start, origin))
            hi = tuple(min(b, o + c) for b, o, c in zip(stop, origin, lv.chunks))
            target = tuple(slice(s - a, e - a) for s, e, a in zip(lo, hi, start))
            key = f"{lv.path}/{lv.separator.join(str(i) for i in index)}"
            chunk = self._chunk(lv, key, origin)
            if chunk is None:
                absent.append(key)
                out[target] = lv.fill_value
            else:
                out[target] = chunk[tuple(slice(s - o, e - o) for s, e, o in zip(lo, hi, origin))]
        return out, len(indices), absent

    def _chunk(self, lv: Level, key: str, origin: tuple[int, ...]) -> np.ndarray | None:
        """The chunk stored at ``key`` (origin ``origin``) in its stored shape, full or cropped at the array
        edge; None when it is absent."""
        edge = tuple(min(c, n - o) for c, n, o in zip(lv.chunks, lv.shape, origin))
        data = self._store.get(key)
        if data is None:
            return None
        return decode_chunk(key, data, compressor=lv.compressor, codec=lv.codec, dtype=DTYPES[lv.dtype],
                            chunks=lv.chunks, edge=edge)

    def describe(self) -> dict:
        """The open volume as canonical-JSON data: store, frame, axes, levels, assumptions, read-log root."""
        return {
            "format": "ome-zarr-v2",
            "store": self._store.describe(),
            "frame": self.frame.as_dict(),
            "axes": dict(self.axes),
            "multiscales": dict(self.multiscales),
            "pyramid_tolerance": self.pyramid_tolerance,
            "levels": [lv.as_dict() for lv in self.levels],
            "assumptions": list(ASSUMPTIONS),
            "read_log": {"scheme": SCHEME, "root": self.log.root(), "objects": len(self.log)},
        }


def _check_blosc_frame(key: str, data: bytes, sizes: tuple[int, int]) -> None:
    """Refuse a Blosc frame whose header disagrees with the stored object or the chunk geometry.

    numcodecs (0.15) decodes a frame flagged MEMCPYED by copying the header's
    ``nbytes`` without comparing them with the bytes present, so a truncated or
    forged frame decodes to whatever memory follows it, at exactly the expected
    size; and ``nbytes`` sizes the output allocation. The 16-byte header is
    version, versionlz, flags, typesize (u8 each), nbytes, blocksize, cbytes
    (u32 little-endian), per c-blosc's README_HEADER.rst.
    """
    _, _, flags, _, nbytes, _, cbytes = (_BLOSC_HEADER.unpack_from(data) if len(data) >= _BLOSC_HEADER.size
                                         else (0, 0, 0, 0, None, 0, None))
    if cbytes != len(data) or (flags & _BLOSC_MEMCPYED and cbytes != nbytes + _BLOSC_HEADER.size):
        raise Refusal("IO_CODEC_DECODE", f"{key}: blosc header (cbytes {cbytes}, nbytes {nbytes}, flags {flags:#x}) "
                      f"does not describe this {len(data)}-byte object; it is truncated, padded or forged",
                      key=key, codec="blosc", stage="header", size=len(data), cbytes=cbytes)
    if nbytes not in sizes:
        raise Refusal("IO_ZARR_CHUNK_SIZE", f"{key}: blosc header declares {nbytes} decoded bytes; expected "
                      f"{_expected_sizes(sizes)}", key=key, stage="header", size=nbytes, expected=sorted(set(sizes)))


def _expected_sizes(sizes: tuple[int, int]) -> str:
    """The allowed decoded sizes ``(full, edge)``, as a reason names them: the edge-cropped size only when it differs
    from the full one (an interior chunk, or a caller that takes only full-size chunks, has one allowed size)."""
    return f"{sizes[0]} (full chunk)" + (f" or {sizes[1]} (edge-cropped)" if sizes[1] != sizes[0] else "")


def _zstd_content_size(data: bytes) -> int | None:
    """The Frame_Content_Size of the zstd frame starting ``data`` (RFC 8878 section 3.1.1.1), or None when
    ``data`` does not start with a zstd frame header that declares one."""
    if len(data) < 5 or data[:4] != _ZSTD_MAGIC:
        return None
    descriptor = data[4]
    single_segment = descriptor >> 5 & 1
    start = 5 + (1 - single_segment) + (0, 1, 2, 4)[descriptor & 3]  # window descriptor, dictionary id
    width = (single_segment, 2, 4, 8)[descriptor >> 6]
    if width == 0 or len(data) < start + width:
        return None
    value = int.from_bytes(data[start:start + width], "little")
    return value + 256 if width == 2 else value


def _declared_size(codec: str, data: bytes) -> int | None:
    """The decoded size a zstd frame or an lz4 object (numcodecs' 4-byte little-endian prefix) declares."""
    if codec == "zstd":
        return _zstd_content_size(data)
    return int.from_bytes(data[:4], "little") if len(data) >= 4 else None


def _check_declared_size(key: str, codec: str, data: bytes, sizes: tuple[int, int]) -> None:
    """Refuse a zstd or lz4 chunk unless its header declares one of the two allowed decoded sizes: the
    decoder allocates what the header declares, so the check bounds memory before decoding."""
    declared = _declared_size(codec, data)
    if declared is None:
        raise Refusal("IO_CODEC_DECODE", f"{key}: the {len(data)}-byte object has no {codec} header declaring its "
                      "decoded size", key=key, codec=codec, stage="header", size=len(data))
    if declared not in sizes:
        raise Refusal("IO_ZARR_CHUNK_SIZE", f"{key}: {codec} header declares {declared} decoded bytes; expected "
                      f"{_expected_sizes(sizes)}", key=key, stage="header", size=declared, expected=sorted(set(sizes)))


def _inflate(codec: str, data: bytes, limit: int) -> bytes:
    """A zlib or gzip stream inflated to at most ``limit + 1`` bytes, as numcodecs would decode it otherwise
    (gzip members concatenated; bytes after a zlib stream ignored)."""
    if codec == "gzip":
        with gzip.GzipFile(fileobj=io.BytesIO(data), mode="rb") as stream:
            return stream.read(limit + 1)
    stream = zlib.decompressobj()
    out = stream.decompress(data, limit + 1)
    if len(out) <= limit and not stream.eof:  # every input byte consumed, yet the stream has not ended
        raise zlib.error("incomplete or truncated stream")
    return out


def _decode(compressor: dict | None, decoder: Any, key: str, data: bytes, sizes: tuple[int, int]) -> bytes:
    """The chunk object decoded, never to more than ``max(sizes) + 1`` bytes."""
    if decoder is None:
        return data
    codec, limit = compressor["id"], max(sizes)
    if codec == "blosc":
        _check_blosc_frame(key, data, sizes)
    elif codec in ("zstd", "lz4"):
        _check_declared_size(key, codec, data, sizes)
    try:
        raw = _inflate(codec, data, limit) if codec in ("zlib", "gzip") else decoder.decode(data)
    except Exception as exc:  # third-party decoders raise assorted types; each is a refusal here
        raise Refusal("IO_CODEC_DECODE", f"{key} failed to decode with {compressor!r}: {exc}", key=key,
                      codec=codec, error=str(exc)) from None
    if len(raw) > limit:
        raise Refusal("IO_ZARR_CHUNK_SIZE", f"{key} decodes to more than {limit} bytes (the full chunk); decoding "
                      f"stopped at {len(raw)}", key=key, stage="decoded", size_at_least=len(raw),
                      expected=sorted(set(sizes)))
    return raw


def decode_chunk(key: str, data: bytes, *, compressor: dict | None, codec: Any, dtype: np.dtype,
                 chunks: tuple[int, int, int], edge: tuple[int, int, int]) -> np.ndarray:
    """The stored chunk object ``data`` (at ``key``) decoded under every check of the module docstring's Chunks
    section, without a :class:`Volume`: ``compressor`` is the level's allow-listed config and ``codec`` its
    decoder from :func:`chunk_codec`; ``dtype`` the level's dtype; ``chunks`` its chunk shape; ``edge`` this chunk's
    shape cropped at the array edge (``chunks`` itself for an interior chunk, or for a caller that takes only
    full-size chunks). Returns the chunk, C order, in the stored shape: ``chunks`` or ``edge``. Nothing is decoded
    to more than the full chunk's bytes plus one, and any other size is ``IO_ZARR_CHUNK_SIZE``."""
    full_bytes, edge_bytes = math.prod(chunks) * dtype.itemsize, math.prod(edge) * dtype.itemsize
    raw = _decode(compressor, codec, key, data, (full_bytes, edge_bytes))
    if len(raw) == full_bytes:
        return np.frombuffer(raw, dtype=dtype).reshape(chunks)
    if len(raw) == edge_bytes:
        return np.frombuffer(raw, dtype=dtype).reshape(edge)
    raise Refusal("IO_ZARR_CHUNK_SIZE", f"{key} decodes to {len(raw)} bytes; expected "
                  f"{_expected_sizes((full_bytes, edge_bytes))}", key=key, stage="decoded", size=len(raw),
                  expected=sorted({full_bytes, edge_bytes}))


def open_volume(store: Store, declared: VolumeFrame, *, axes_declared: tuple[str, ...] | None = None,
                axes_declared_by: str | None = None, pyramid_tolerance: str = "exact") -> Volume:
    """Open and validate the OME-Zarr v2 volume at ``store``'s root against ``declared``.

    ``axes_declared`` (an order, which must be ``("z", "y", "x")``) and
    ``axes_declared_by`` (who declares it) come together or not at all; they are
    required when the store declares no axes. ``pyramid_tolerance`` is
    ``"exact"`` or ``"chunk-padding"``. No chunk is read: every check in the
    module docstring under Group, Axes, Datasets, Arrays and Frame runs here.
    """
    if not isinstance(store, Store):
        raise Refusal("IO_ZARR_ARGUMENT_INVALID", f"store {store!r} has no get/describe", field="store")
    if not isinstance(declared, VolumeFrame):
        raise Refusal("IO_ZARR_ARGUMENT_INVALID", f"declared must be a VolumeFrame, not {declared!r}",
                      field="declared")
    if pyramid_tolerance not in PYRAMID_TOLERANCES:
        raise Refusal("IO_ZARR_ARGUMENT_INVALID", f"pyramid_tolerance {pyramid_tolerance!r} is not one of "
                      f"{PYRAMID_TOLERANCES}", field="pyramid_tolerance")
    if axes_declared_by is not None and not (isinstance(axes_declared_by, str) and axes_declared_by.strip()):
        raise Refusal("IO_ZARR_ARGUMENT_INVALID", "axes_declared_by must be None or a non-blank source",
                      field="axes_declared_by")
    if (axes_declared is None) != (axes_declared_by is None):
        missing = "axes_declared" if axes_declared is None else "axes_declared_by"
        raise Refusal("IO_ZARR_ARGUMENT_INVALID", f"an axes declaration is an order (axes_declared) and who declares "
                      f"it (axes_declared_by); {missing} is missing", field=missing)
    zyx = isinstance(axes_declared, (tuple, list)) and list(axes_declared) == list(AXES)
    if axes_declared is not None and not zyx:
        raise Refusal("IO_ZARR_AXES", f"{axes_declared_by} declares axes {axes_declared!r}; only axes exactly "
                      "(z, y, x) are read", field="axes_declared", axes=repr(axes_declared),
                      declared_by=axes_declared_by)
    logged = LoggedStore(store, ReadLog())
    _check_group(logged.get(".zgroup"))
    ms, descriptors = _multiscales(logged.get(".zattrs"))
    axes = _axes_record(ms, None if axes_declared is None else tuple(axes_declared), axes_declared_by)
    transforms = [_transforms(dataset, k) for k, dataset in enumerate(ms["datasets"])]
    _check_transforms(transforms)
    for name in SIDECARS:
        logged.get(name)
    levels: list[Level] = []
    for k, transform in enumerate(transforms):
        lv = _open_level(logged, k, transform)
        if k == 0 and lv.shape != declared.shape_zyx:
            raise Refusal("IO_ZARR_SHAPE_MISMATCH", f"level 0 shape {list(lv.shape)} differs from the declared "
                          f"shape {list(declared.shape_zyx)} of volume {declared.volume_id} ({declared.source})",
                          key="0/.zarray", field="shape", stored=list(lv.shape), declared=list(declared.shape_zyx))
        if levels and lv.dtype != levels[0].dtype:
            raise Refusal("IO_ZARR_DTYPE", f"level {k} dtype {lv.dtype!r} differs from level 0's "
                          f"{levels[0].dtype!r}", key=f"{k}/.zarray", field="dtype", value=lv.dtype)
        levels.append(lv)
    levels = _with_padding(levels, pyramid_tolerance)
    return Volume(logged, declared, levels, axes,
                  {"version": ms.get("version"), "name": ms.get("name"), "descriptors": descriptors},
                  pyramid_tolerance)
