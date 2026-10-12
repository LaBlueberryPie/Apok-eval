"""tifxyz surfaces: a strict reader, a canonical writer and named validity policies.

A tifxyz surface is a directory holding ``x.tif``, ``y.tif``, ``z.tif`` and
``meta.json``; ``docs/io-surfaces.md`` cites the villa source of every rule
used here. Grid vertex (row r, column c) is TIFF pixel (image line r, sample
c) and its point is (X, Y, Z) = (x.tif[r, c], y.tif[r, c], z.tif[r, c]).
apok-eval exposes the grid as an (H, W, 3) array in **(z, y, x)** order:
``zyx[..., 0]`` comes from ``z.tif`` and indexes OME-Zarr axis 0, ``y.tif``
axis 1, ``x.tif`` axis 2. Coordinates are level-0 voxels of a volume the
surface does not name; :func:`check_within_frame` compares them with a
declared :class:`~apok_eval.io.frame.VolumeFrame`.

Reading (:func:`read_tifxyz`) never writes to the directory and never decides
validity. Each file is read once into memory and digested; tifffile decodes
those bytes. x/y/z must be single-IFD, single-sample IEEE float32 images of one
height and width, and the returned arrays keep the raw float32 bit patterns
(-0.0, NaN payloads, signalling NaNs) in little-endian order. A file with no
first IFD (a truncated download whose IFD followed the pixels), a cut-off IFD,
a zero image/strip/tile size, a strip/tile list shorter or longer than the
geometry needs, a sparse, short or truncated strip/tile and any decoder error
is a refusal, never a zero-fill. ``meta.json`` is parsed by
:func:`apok_eval.jsonstrict.loads` (duplicate keys, lone surrogates and deep nesting
refused; NaN/Infinity collected and reported as a finding) and must declare
``format`` "tifxyz", ``type`` "seg", a non-empty string ``uuid`` and a
``scale`` [sx, sy] of two finite positive numbers that stay positive and
finite as float32. The stored ``bbox`` is never used. ``mask.tif`` and every
other file are inventoried (name, kind, size, digest; a name that is not valid
UTF-8 is recorded escaped, with its bytes in hex); the mask is decoded only when
a validity policy asks for it.

Validity is a separate, named policy (:data:`POLICIES`), because the villa
readers disagree; :func:`validity_report` evaluates several and counts where
each pair disagrees. Writing (:func:`write_tifxyz`) produces canonical bytes
that depend on no library version: the TIFF layout is fixed by
:func:`canonical_tiff_header` and ``meta.json`` is sorted-key, 4-space JSON
with a trailing newline.
"""

from __future__ import annotations

import io
import math
import os
import struct
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

import numpy as np
import tifffile

from apok_eval import jsonstrict
from apok_eval.canonical import sha256_bytes
from apok_eval.fileio import NotRegularFile, digest_and_size, portable_name, read_regular, write_atomic
from apok_eval.io.frame import VolumeFrame, check_points_within, stated_identity
from apok_eval.refusal import Refusal, register_codes

register_codes(
    "apok_eval.io.tifxyz",
    {
        "IO_TIFXYZ_NOT_DIRECTORY": "a tifxyz path is not a directory",
        "IO_TIFXYZ_FILE_MISSING": "a required tifxyz file (x.tif, y.tif, z.tif, meta.json) is absent",
        "IO_TIFXYZ_FILE_KIND": "a tifxyz file name exists but is not a regular file",
        "IO_TIFXYZ_META_INVALID": "meta.json is not strict JSON holding an object",
        "IO_TIFXYZ_META_FIELD": "a required meta.json field (format, type, uuid, scale) is missing or wrong",
        "IO_TIFXYZ_TIFF_INVALID": "a file is not a TIFF with a complete first IFD and non-zero image, strip and "
                                  "tile sizes",
        "IO_TIFXYZ_TIFF_PAGES": "a coordinate TIFF holds more than one IFD",
        "IO_TIFXYZ_TIFF_SAMPLE": "a TIFF is not single-sample IEEE float32, or is volumetric",
        "IO_TIFXYZ_TIFF_COMPRESSION": "a TIFF uses a compression or predictor outside the supported set",
        "IO_TIFXYZ_CODEC_MISSING": "a supported compression or predictor needs the missing imagecodecs package",
        "IO_TIFXYZ_TIFF_SEGMENTS": "a strip/tile list does not match the geometry, or a strip or tile is missing, "
                                   "short, or runs past the end of the file",
        "IO_TIFXYZ_TIFF_DECODE": "tifffile failed to decode a TIFF's image data",
        "IO_TIFXYZ_SHAPE_MISMATCH": "x.tif, y.tif and z.tif differ in height or width",
        "IO_TIFXYZ_POLICY_UNKNOWN": "a validity policy name is not a declared policy",
        "IO_TIFXYZ_MASK_UNSUPPORTED": "mask.tif has a layout the chosen policy defines no rule for",
        "IO_TIFXYZ_MASK_SHAPE": "mask.tif's shape cannot be applied to the grid under the chosen policy",
        "IO_TIFXYZ_MASK_VALUES": "under apok-strict, mask.tif holds values other than 0 and 255",
        "IO_TIFXYZ_SCALE_INVALID": "a scale is not a positive finite float32",
        "IO_TIFXYZ_EXTENT_INVALID": "a stored grid extent given to the canvas rule is not an integer in [0, 2**53]",
        "IO_TIFXYZ_WRITE_EXISTS": "the output directory exists and overwrite was not requested",
        "IO_TIFXYZ_WRITE_SYMLINK": "the output directory is a symbolic link, which the writer does not follow",
        "IO_TIFXYZ_WRITE_INVALID": "arrays or meta given to the writer cannot be written as canonical tifxyz",
        "IO_TIFXYZ_WRITE_TOO_LARGE": "a plane's pixel bytes exceed 0xffff0000 (BigTIFF), which the writer refuses",
    },
)

META_FILE = "meta.json"
MASK_FILE = "mask.tif"
COORDINATE_FILES = ("x.tif", "y.tif", "z.tif")
POLICIES = ("vc3d", "spiral", "vesuvius", "lasagna", "apok-strict")

#: TileWidth, TileLength, TileOffsets, TileByteCounts.
_TILE_TAGS = (322, 323, 324, 325)
#: TIFF compression values apok-eval decodes (all lossless); others are refused.
COMPRESSIONS = {1: "none", 5: "LZW", 8: "Adobe Deflate", 32946: "Deflate"}
#: TIFF predictor values apok-eval decodes.
PREDICTORS = {1: "none", 2: "horizontal", 3: "floating point"}
#: C++ writeTiff switches to BigTIFF above this many pixel bytes (villa Tiff.cpp:129-131).
BIGTIFF_THRESHOLD = 0xFFFF0000
#: (SampleFormat, BitsPerSample) pairs for which the C++ mask rule ``>= 255`` is defined.
VC3D_MASK_FORMATS = frozenset({(1, 8), (1, 16), (1, 32), (2, 16), (2, 32), (3, 32), (3, 64)})


# --------------------------------------------------------------------------------------------
# Surface


@dataclass(frozen=True, eq=False)
class TifxyzSurface:
    """One decoded tifxyz directory. Arrays are read-only little-endian float32 (H, W)."""

    path: str
    dir_name: str
    meta: dict[str, Any]
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    tiffs: dict[str, dict[str, Any]]  # x.tif / y.tif / z.tif -> layout facts
    mask_tiff: bytes | None  # mask.tif bytes as read, decoded on demand
    files: tuple[dict[str, Any], ...]  # every directory entry, sorted by name
    findings: tuple[dict[str, Any], ...]

    @property
    def uuid(self) -> str:
        return self.meta["uuid"]

    @property
    def shape(self) -> tuple[int, int]:
        return self.z.shape

    @property
    def scale(self) -> list[Any]:
        """``meta.json`` ``scale`` verbatim: [sx, sy], samples per base voxel along columns, rows."""
        return list(self.meta["scale"])

    @property
    def scale_float32(self) -> tuple[float, float]:
        """``scale`` as the C++ loader holds it (float32), widened exactly to double."""
        return tuple(float(np.float32(v)) for v in self.meta["scale"])  # type: ignore[return-value]

    @property
    def zyx(self) -> np.ndarray:
        """(H, W, 3) grid in (z, y, x) order; bit-identical copies of the decoded planes."""
        return np.stack([self.z, self.y, self.x], axis=-1)

    @property
    def full_resolution_shape(self) -> tuple[int, int]:
        """(rows, cols) of the full-resolution canvas the renderers size for this grid."""
        sx, sy = self.meta["scale"]
        rows, cols = self.shape
        return full_resolution_extent(rows, sy), full_resolution_extent(cols, sx)

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready summary (no pixel data; ``meta`` is described by its keys and file digest)."""
        return {
            "path": portable_name(self.path),
            "dir_name": portable_name(self.dir_name),
            "uuid": self.uuid,
            "shape_hw": list(self.shape),
            "axes": ["z", "y", "x"],
            "scale": self.scale,
            "scale_float32": list(self.scale_float32),
            "full_resolution_shape_hw": list(self.full_resolution_shape),
            "meta_keys": sorted(self.meta),
            "tiffs": self.tiffs,
            "mask_present": self.mask_tiff is not None,
            "files": list(self.files),
            "findings": list(self.findings),
        }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def read_tifxyz(path: str | os.PathLike[str]) -> TifxyzSurface:
    """Read and check one tifxyz directory; see the module docstring for every rule."""
    directory = os.fspath(path)
    if not os.path.isdir(directory):
        raise Refusal("IO_TIFXYZ_NOT_DIRECTORY", f"{directory} is not a directory", path=directory)
    raw: dict[str, bytes] = {}
    for name in (META_FILE, *COORDINATE_FILES):
        raw[name] = _read_file(directory, name, required=True)  # type: ignore[assignment]
    mask = _read_file(directory, MASK_FILE, required=False)
    if mask is not None:
        raw[MASK_FILE] = mask

    meta_path = os.path.join(directory, META_FILE)
    try:
        meta, constants = jsonstrict.loads(raw[META_FILE], constants="collect")
    except jsonstrict.StrictJsonError as exc:
        raise Refusal("IO_TIFXYZ_META_INVALID", f"{meta_path}: {exc}", file=META_FILE, path=meta_path,
                      problem=str(exc)) from None
    check_meta(meta, meta_path)

    planes: dict[str, np.ndarray] = {}
    tiffs: dict[str, dict[str, Any]] = {}
    for name in COORDINATE_FILES:
        planes[name], tiffs[name] = _decode_coordinate_plane(name, raw[name])
    shapes = {name: list(arr.shape) for name, arr in planes.items()}
    if len({tuple(s) for s in shapes.values()}) != 1:
        raise Refusal("IO_TIFXYZ_SHAPE_MISMATCH", f"{directory}: x/y/z.tif differ in height x width {shapes}",
                      path=directory, shapes=shapes)

    dir_name = os.path.basename(os.path.normpath(os.path.abspath(directory)))
    findings = _meta_findings(meta, dir_name, constants)
    files, link_findings = _inventory(directory, raw)
    return TifxyzSurface(
        path=directory, dir_name=dir_name, meta=meta,
        x=planes["x.tif"], y=planes["y.tif"], z=planes["z.tif"],
        tiffs=tiffs, mask_tiff=mask, files=tuple(files), findings=tuple(findings + link_findings),
    )


def _read_file(directory: str, name: str, *, required: bool) -> bytes | None:
    path = os.path.join(directory, name)
    try:
        return read_regular(path)
    except NotRegularFile as exc:
        absent = exc.missing and not os.path.lexists(path)  # a dangling link has an entry: it is a kind, not absent
        if absent and not required:
            return None
        if absent:
            raise Refusal("IO_TIFXYZ_FILE_MISSING", f"{name} is missing from {directory}", file=name,
                          path=directory) from None
        kind = exc.kind or "a dangling link"
        raise Refusal("IO_TIFXYZ_FILE_KIND", f"{path} is {kind}, not a regular file", file=name, path=directory,
                      kind=kind) from None


def check_meta(meta: Any, where: str) -> None:
    """Refuse unless ``meta`` declares format, type, uuid and scale as a tifxyz reader needs."""
    if not isinstance(meta, dict):
        raise Refusal("IO_TIFXYZ_META_INVALID", f"{where}: meta.json must hold a JSON object, not "
                      f"{type(meta).__name__}", file=META_FILE, path=where)
    if meta.get("format") != "tifxyz":
        raise Refusal("IO_TIFXYZ_META_FIELD", f"{where}: format must be \"tifxyz\"", field="format",
                      value=meta.get("format"), path=where)
    if meta.get("type") != "seg":
        raise Refusal("IO_TIFXYZ_META_FIELD", f"{where}: type must be \"seg\"", field="type",
                      value=meta.get("type"), path=where)
    uuid = meta.get("uuid")
    if not isinstance(uuid, str) or not uuid:
        raise Refusal("IO_TIFXYZ_META_FIELD", f"{where}: uuid must be a non-empty string", field="uuid",
                      value=uuid, path=where)
    problem = _scale_problem(meta.get("scale"))
    if problem is not None:
        raise Refusal("IO_TIFXYZ_META_FIELD", f"{where}: scale {problem}", field="scale",
                      value=_json_safe(meta.get("scale")), path=where)


def _scale_problem(scale: Any) -> str | None:
    if not isinstance(scale, list) or len(scale) != 2:
        return "must be a list of two numbers [sx, sy]"
    for i, value in enumerate(scale):
        if _as_positive_float32(value) is None:
            return (f"[{i}] = {_json_safe(value)!r} is not a positive finite number that stays positive and "
                    "finite as float32 (the C++ loader stores scale as float32)")
    return None


def _as_positive_float32(value: Any) -> float | None:
    """``value`` narrowed to float32 (as a double), or None unless a positive finite non-bool number."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        return None
    try:
        as_double = float(value)
    except OverflowError:
        return None
    with np.errstate(over="ignore"):
        narrowed = float(np.float32(as_double))
    if not math.isfinite(narrowed) or narrowed <= 0:
        return None
    return narrowed


def _meta_findings(meta: dict[str, Any], dir_name: str, constants: tuple[str, ...]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if meta["uuid"] != dir_name:
        findings.append({"finding": "uuid-differs-from-directory", "uuid": meta["uuid"],
                         "directory": portable_name(dir_name)})
    scale = [float(v) for v in meta["scale"]]
    if any(v > 1 for v in scale):
        findings.append({"finding": "scale-above-one", "scale": scale,
                         "detail": "scale is samples per voxel (the reciprocal of the step); a value above 1 "
                                   "looks like a step size"})
    if scale == [20.0, 20.0]:
        findings.append({"finding": "scale-vesuvius-default", "scale": scale,
                         "detail": "(20, 20) is the vesuvius reader's silent default for a missing scale"})
    if constants:
        findings.append({"finding": "meta-nonstandard-json", "tokens": sorted(set(constants)),
                         "detail": "NaN/Infinity are not JSON; nlohmann::json (C++) cannot parse this file"})
    return findings


def _inventory(directory: str, raw: dict[str, bytes]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Every directory entry, not descending and not following links except the files read."""
    files: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    with os.scandir(directory) as entries:
        for entry in sorted(entries, key=lambda e: e.name):
            record: dict[str, Any] = {"name": portable_name(entry.name)}
            if record["name"] != entry.name:
                record["name_hex"] = os.fsencode(entry.name).hex()
                findings.append({"finding": "name-not-utf8", "name": record["name"], "name_hex": record["name_hex"]})
            if entry.is_symlink():
                target = os.readlink(entry.path)
                record.update(kind="symlink", target=portable_name(target))
                if record["target"] != target:
                    record["target_hex"] = os.fsencode(target).hex()
                if entry.name in raw:
                    findings.append({"finding": "symlink-followed", "file": entry.name,
                                     "target": portable_name(os.path.realpath(entry.path))})
            elif entry.is_file(follow_symlinks=False):
                record["kind"] = "file"
            elif entry.is_dir(follow_symlinks=False):
                record["kind"] = "directory"
            else:
                record["kind"] = "other"
            if entry.name in raw:
                record.update(size=len(raw[entry.name]), digest=sha256_bytes(raw[entry.name]))
            elif record["kind"] == "file":
                try:
                    digest, size = digest_and_size(entry.path)
                except OSError as exc:  # an entry the surface does not need: recorded, not fatal to the read
                    record["unreadable"] = f"{type(exc).__name__}: {exc.strerror or exc}"
                    findings.append({"finding": "entry-unreadable", "name": record["name"],
                                     "error": record["unreadable"]})
                else:
                    record.update(size=size, digest=digest)
            files.append(record)
    return files, findings


# --------------------------------------------------------------------------------------------
# TIFF decoding


def _decode_coordinate_plane(name: str, data: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    array, layout = decode_first_ifd(name, data, _require_float32_single_ifd)
    return _little_endian_float32(array), layout


def _require_float32_single_ifd(name: str, layout: dict[str, Any]) -> None:
    if not layout["single_ifd"]:
        raise Refusal("IO_TIFXYZ_TIFF_PAGES", f"{name}: more than one IFD; only one image is meaningful",
                      file=name, next_ifd_offset=layout["next_ifd_offset"])
    if layout["samples_per_pixel"] != 1:
        raise Refusal("IO_TIFXYZ_TIFF_SAMPLE", f"{name}: SamplesPerPixel is {layout['samples_per_pixel']}, "
                      "not 1", file=name, field="samples_per_pixel", value=layout["samples_per_pixel"])
    if layout["bits_per_sample"] != 32:
        raise Refusal("IO_TIFXYZ_TIFF_SAMPLE", f"{name}: BitsPerSample is {layout['bits_per_sample']}, not 32",
                      file=name, field="bits_per_sample", value=layout["bits_per_sample"])
    if layout["sample_format"] != 3:
        raise Refusal("IO_TIFXYZ_TIFF_SAMPLE", f"{name}: SampleFormat is {layout['sample_format']}, not 3 "
                      "(IEEE float)", file=name, field="sample_format", value=layout["sample_format"])


def decode_first_ifd(name: str, data: bytes,
                     check_layout: Callable[[str, dict[str, Any]], None] | None = None
                     ) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode the first IFD of the TIFF bytes ``data`` with tifffile.

    ``check_layout(name, layout)`` runs on the IFD's facts before any pixel is
    decoded. The first IFD must exist and be complete, its image, strip and
    tile sizes non-zero, and its strip/tile lists exactly as long as the
    geometry needs (tifffile would zero-fill missing ones); every strip/tile
    must be present, lie inside the file and, uncompressed, hold its full
    size. tifffile errors become ``IO_TIFXYZ_TIFF_DECODE``.

    Every IFD value is read as one integer (a tag that is missing, empty, of
    another type or holding several values is ``IO_TIFXYZ_TIFF_INVALID`` naming
    it); SamplesPerPixel and BitsPerSample must be at least 1; the number of
    strips or tiles the geometry needs is computed arithmetically and compared
    with the declared counts before any per-segment list is built, so a
    196-byte file declaring four billion strips is refused at once. Any other
    exception the structure checks meet in tifffile's derived values
    (``TypeError``, ``ValueError``, ``ZeroDivisionError``, ``OverflowError``,
    ``MemoryError``, ...) is ``IO_TIFXYZ_TIFF_INVALID`` (``field: "structure"``),
    never an escaping exception.
    """
    try:
        tif = tifffile.TiffFile(io.BytesIO(data))
    except Exception as exc:  # tifffile reports malformed input through several exception types
        raise Refusal("IO_TIFXYZ_TIFF_INVALID", f"{name}: not a readable TIFF ({type(exc).__name__}: {exc})",
                      file=name) from None
    with tif:
        try:
            page, layout = _checked_structure(name, data, tif, check_layout)
        except Refusal:
            raise
        except Exception as exc:  # tifffile derives values lazily from raw tags; damage surfaces as assorted types
            raise Refusal("IO_TIFXYZ_TIFF_INVALID", f"{name}: the first IFD's structure cannot be read "
                          f"({type(exc).__name__}: {exc})", file=name, field="structure",
                          error=type(exc).__name__) from None
        try:
            array = page.asarray()
        except Exception as exc:  # decoder failures surface as TiffFileError, zlib.error, ValueError, ...
            raise Refusal("IO_TIFXYZ_TIFF_DECODE", f"{name}: decoding failed ({type(exc).__name__}: {exc})",
                          file=name) from None
    return array, layout


def _checked_structure(name: str, data: bytes, tif: tifffile.TiffFile,
                       check_layout: Callable[[str, dict[str, Any]], None] | None
                       ) -> tuple[tifffile.TiffPage, dict[str, Any]]:
    """The first page and its layout, after every structure check that runs before decoding."""
    page = _first_page(name, data, tif)
    layout = _layout(name, data, tif, page)
    depth = _page_int(name, page, "image_depth", "imagedepth")
    if depth != 1:
        raise Refusal("IO_TIFXYZ_TIFF_SAMPLE", f"{name}: ImageDepth is {depth}; volumetric TIFFs "
                      "are not surfaces", file=name, field="image_depth", value=depth)
    _require_sizes(name, page, layout)
    if check_layout is not None:
        check_layout(name, layout)
    _require_codecs(name, layout)
    _require_segments(name, data, page, layout)
    return page, layout


def _page_int(name: str, page: tifffile.TiffPage, field: str, attr: str) -> int:
    """``page.<attr>`` as one int; tifffile derives it from a raw tag that may be missing, empty, of another
    type or hold several values, each of which is refused naming ``field``."""
    try:
        value = getattr(page, attr)
        if isinstance(value, (tuple, list, bytes, str)):
            raise TypeError(f"{type(value).__name__} {value!r:.40}")
        return int(value)
    except Exception as exc:  # TypeError, ValueError, OverflowError from tifffile or int()
        raise Refusal("IO_TIFXYZ_TIFF_INVALID", f"{name}: {field} is not one integer ({type(exc).__name__}: "
                      f"{exc})", file=name, field=field, value=None) from None


def _first_page(name: str, data: bytes, tif: tifffile.TiffFile) -> tifffile.TiffPage:
    # tifffile opens a file whose first-IFD offset is 0 or past the end with no pages and a log line.
    try:
        return tif.pages[0]
    except IndexError:
        fmt = tif.tiff
        (first,) = struct.unpack(fmt.offsetformat, data[fmt.offsetsize: 2 * fmt.offsetsize])
        raise Refusal("IO_TIFXYZ_TIFF_INVALID", f"{name}: the header's first-IFD offset {first} leads to no image "
                      f"file directory in the {len(data)}-byte file (truncated, e.g. a download cut short of an "
                      "IFD written after the pixels)", file=name, field="first_ifd_offset", value=int(first)) from None


def _layout(name: str, data: bytes, tif: tifffile.TiffFile, page: tifffile.TiffPage) -> dict[str, Any]:
    next_ifd = _next_ifd_offset(name, tif, page, data)
    return {
        "file": name,
        "size": len(data),
        "digest": sha256_bytes(data),
        "byteorder": "II" if tif.byteorder == "<" else "MM",
        "bigtiff": bool(tif.is_bigtiff),
        "single_ifd": next_ifd == 0,
        "next_ifd_offset": next_ifd,
        "height": _page_int(name, page, "image_length", "imagelength"),
        "width": _page_int(name, page, "image_width", "imagewidth"),
        "samples_per_pixel": _page_int(name, page, "samples_per_pixel", "samplesperpixel"),
        "bits_per_sample": _page_int(name, page, "bits_per_sample", "bitspersample"),
        "sample_format": _page_int(name, page, "sample_format", "sampleformat"),
        "planar_config": _page_int(name, page, "planar_config", "planarconfig"),
        "photometric": _page_int(name, page, "photometric", "photometric"),
        "compression": _page_int(name, page, "compression", "compression"),
        "predictor": _page_int(name, page, "predictor", "predictor"),
        "tiled": bool(page.is_tiled),
        "tile_hw": ([_page_int(name, page, "tile_length", "tilelength"),
                     _page_int(name, page, "tile_width", "tilewidth")] if page.is_tiled else None),
        "rows_per_strip": None if page.is_tiled else _page_int(name, page, "rows_per_strip", "rowsperstrip"),
        "segments": len(page.dataoffsets),
    }


def _next_ifd_offset(name: str, tif: tifffile.TiffFile, page: tifffile.TiffPage, data: bytes) -> int:
    # Read the first IFD's next-IFD pointer directly: tifffile truncates a damaged IFD chain with only
    # a log warning, which would make a multi-page file look single-page.
    fmt = tif.tiff
    (count,) = struct.unpack(fmt.tagnoformat, data[page.offset: page.offset + fmt.tagnosize])
    pos = page.offset + fmt.tagnosize + count * fmt.tagsize
    if pos + fmt.offsetsize > len(data):
        raise Refusal("IO_TIFXYZ_TIFF_INVALID", f"{name}: the first IFD's next-IFD pointer (bytes {pos} to "
                      f"{pos + fmt.offsetsize}) is cut off by the end of the {len(data)}-byte file", file=name,
                      field="next_ifd_offset", value=None)
    (offset,) = struct.unpack(fmt.offsetformat, data[pos: pos + fmt.offsetsize])
    return int(offset)


def _require_sizes(name: str, page: tifffile.TiffPage, layout: dict[str, Any]) -> None:
    # Zero sizes decode to empty arrays or divide by zero; none of them is an image. Any tile tag makes
    # the page tiled here, because tifffile calls a page with TileWidth 0 striped.
    # SamplesPerPixel and BitsPerSample 0 divide by zero in the segment arithmetic.
    sizes = {"image_length": layout["height"], "image_width": layout["width"],
             "samples_per_pixel": layout["samples_per_pixel"], "bits_per_sample": layout["bits_per_sample"]}
    if page.is_tiled or any(tag in page.tags for tag in _TILE_TAGS):
        sizes["tile_length"] = _page_int(name, page, "tile_length", "tilelength")
        sizes["tile_width"] = _page_int(name, page, "tile_width", "tilewidth")
    else:
        sizes["rows_per_strip"] = layout["rows_per_strip"]
    for field, value in sizes.items():
        if value < 1:
            raise Refusal("IO_TIFXYZ_TIFF_INVALID", f"{name}: {field} is {value}; a TIFF image, strip or tile has "
                          "at least one row and column", file=name, field=field, value=value)


def _require_codecs(name: str, layout: dict[str, Any]) -> None:
    compression, predictor = layout["compression"], layout["predictor"]
    if compression not in COMPRESSIONS:
        raise Refusal("IO_TIFXYZ_TIFF_COMPRESSION", f"{name}: compression {compression} is not one of "
                      f"{sorted(COMPRESSIONS)}", file=name, field="compression", value=compression)
    if predictor not in PREDICTORS:
        raise Refusal("IO_TIFXYZ_TIFF_COMPRESSION", f"{name}: predictor {predictor} is not one of "
                      f"{sorted(PREDICTORS)}", file=name, field="predictor", value=predictor)
    try:
        tifffile.TIFF.DECOMPRESSORS[compression]
    except KeyError:
        raise Refusal("IO_TIFXYZ_CODEC_MISSING", f"{name}: {COMPRESSIONS[compression]} decompression needs the "
                      "imagecodecs package, which is not installed", file=name,
                      codec=COMPRESSIONS[compression]) from None
    if predictor != 1:
        try:
            tifffile.TIFF.UNPREDICTORS[predictor]
        except KeyError:
            raise Refusal("IO_TIFXYZ_CODEC_MISSING", f"{name}: the {PREDICTORS[predictor]} predictor needs the "
                          "imagecodecs package, which is not installed", file=name,
                          codec=f"predictor {PREDICTORS[predictor]}") from None


def _require_segments(name: str, data: bytes, page: tifffile.TiffPage, layout: dict[str, Any]) -> None:
    needed = _segment_count(layout)
    # The declared counts, not page.dataoffsets: tifffile cuts a long list to the expected length and
    # zero-fills the segments a short list lacks, logging only a warning. They are compared before any
    # per-segment list is built: the geometry may claim billions of segments the file cannot hold.
    tags = (324, 325) if page.is_tiled else (273, 279)
    declared = [page.tags[tag].count if tag in page.tags else 0 for tag in tags]
    if declared != [needed] * 2:
        kind = "Tile" if page.is_tiled else "Strip"
        raise Refusal("IO_TIFXYZ_TIFF_SEGMENTS", f"{name}: {kind}Offsets has {declared[0]} and {kind}ByteCounts "
                      f"{declared[1]} entries; the geometry needs {needed}", file=name,
                      expected=needed, offsets=declared[0], byte_counts=declared[1])
    expected = _uncompressed_segment_sizes(layout)
    for index, (offset, count) in enumerate(zip(page.dataoffsets, page.databytecounts, strict=True)):
        if offset == 0 or count == 0:
            raise Refusal("IO_TIFXYZ_TIFF_SEGMENTS", f"{name}: segment {index} is absent (offset {offset}, "
                          f"{count} bytes); tifffile would fill it with zeros", file=name, segment=index)
        if offset + count > len(data):
            raise Refusal("IO_TIFXYZ_TIFF_SEGMENTS", f"{name}: segment {index} ends at byte {offset + count}, "
                          f"past the end of the {len(data)}-byte file", file=name, segment=index)
        if layout["compression"] == 1 and count < expected[index]:
            raise Refusal("IO_TIFXYZ_TIFF_SEGMENTS", f"{name}: uncompressed segment {index} holds {count} "
                          f"bytes, {expected[index]} needed", file=name, segment=index)


def _samples_and_planes(layout: dict[str, Any]) -> tuple[int, int]:
    """(samples interleaved per segment, separate planes): chunky (planar 1) or one plane per sample."""
    samples = layout["samples_per_pixel"] if layout["planar_config"] == 1 else 1
    return samples, layout["samples_per_pixel"] // samples


def _segment_count(layout: dict[str, Any]) -> int:
    """How many strips or tiles the geometry needs, by integer arithmetic (no list is built)."""
    _, planes = _samples_and_planes(layout)
    height, width = layout["height"], layout["width"]
    if layout["tiled"]:
        rows, cols = layout["tile_hw"]
        return -(-height // rows) * -(-width // cols) * planes
    return -(-height // min(layout["rows_per_strip"], height)) * planes


def _uncompressed_segment_sizes(layout: dict[str, Any]) -> list[int]:
    """Each segment's uncompressed byte count; called only once the declared counts equal
    :func:`_segment_count`, so the list is no longer than the file's own offset list."""
    samples, planes = _samples_and_planes(layout)
    bits, height, width = layout["bits_per_sample"], layout["height"], layout["width"]
    if layout["tiled"]:
        rows, cols = layout["tile_hw"]
        return [rows * -(-cols * samples * bits // 8)] * _segment_count(layout)
    rows_per_strip = min(layout["rows_per_strip"], height)
    strips = -(-height // rows_per_strip)
    row_bytes = -(-width * samples * bits // 8)
    sizes = [min(rows_per_strip, height - i * rows_per_strip) * row_bytes for i in range(strips)]
    return sizes * planes


def _little_endian_float32(array: np.ndarray) -> np.ndarray:
    # Byte-swap through uint32 so no float conversion can touch a NaN payload.
    words = array.view(array.dtype.str[0] + "u4").astype("<u4")
    out = np.ascontiguousarray(words).view("<f4")
    out.flags.writeable = False
    return out


# --------------------------------------------------------------------------------------------
# Validity policies


@dataclass(frozen=True, eq=False)
class _Mask:
    array: np.ndarray  # first IFD as decoded: (H', W') or (H', W', S) / (S, H', W')
    channel0: np.ndarray  # (H', W')
    layout: dict[str, Any]


@dataclass(frozen=True, eq=False)
class PolicyResult:
    """One policy's verdict: ``valid`` is an (H, W) bool array; findings say what the policy ignored."""

    policy: str
    valid: np.ndarray
    findings: tuple[dict[str, Any], ...]


def validity(surface: TifxyzSurface, policy: str) -> PolicyResult:
    """Evaluate one named validity policy on ``surface`` (see ``docs/io-surfaces.md``).

    - ``vc3d``: invalid iff z <= 0 (so -0.0; NaN z stays valid), then mask.tif
      channel 0 below 255 invalidates; the mask may be the grid size or an exact
      integer multiple per axis (a vertex survives only if its whole block is
      >= 255) and is ignored otherwise, as the C++ loader does.
    - ``spiral``: mask.tif channel 0 == 0 invalidates; valid iff ANY of x, y, z != -1.
    - ``vesuvius``: a mask whose shape equals the grid decides alone (!= 0);
      otherwise (absent, or a shape mismatch, which vesuvius ignores) valid iff z > 0 and z finite.
    - ``lasagna``: valid iff ALL of x, y, z != -1; mask.tif ignored.
    - ``apok-strict``: valid iff x, y, z finite and z > 0 (which also excludes the
      (-1, -1, -1) sentinel) and, with a mask, mask == 255; a mask value outside
      {0, 255}, a mask of another shape, or a multi-page mask is refused.
    """
    if policy not in POLICIES:
        raise Refusal("IO_TIFXYZ_POLICY_UNKNOWN", f"validity policy {policy!r} is not one of {list(POLICIES)}",
                      policy=policy)
    rule, reads_mask = _POLICY_RULES[policy]
    mask = _decode_mask(surface.mask_tiff) if reads_mask and surface.mask_tiff is not None else None
    valid, findings = rule(surface, mask)
    valid = np.ascontiguousarray(valid, dtype=bool)
    valid.flags.writeable = False
    return PolicyResult(policy=policy, valid=valid, findings=tuple(findings))


def _decode_mask(data: bytes) -> _Mask:
    array, layout = decode_first_ifd(MASK_FILE, data)
    if array.ndim == 2:
        channel0 = array
    elif layout["planar_config"] == 2:
        channel0 = array[0]
    else:
        channel0 = array[..., 0]
    return _Mask(array=array, channel0=channel0, layout=layout)


def _mask_ignored(policy: str, why: str, mask: _Mask, surface: TifxyzSurface) -> dict[str, Any]:
    return {"finding": "mask-ignored", "policy": policy, "reason": why, "mask_shape": list(mask.array.shape),
            "grid_shape": list(surface.shape)}


def _rule_vc3d(s: TifxyzSurface, mask: _Mask | None) -> tuple[np.ndarray, list[dict[str, Any]]]:
    valid = ~(s.z <= 0)
    if mask is None:
        return valid, []
    fmt = (mask.layout["sample_format"], mask.layout["bits_per_sample"])
    if fmt not in VC3D_MASK_FORMATS:
        raise Refusal("IO_TIFXYZ_MASK_UNSUPPORTED", f"vc3d policy: the C++ mask rule is undefined for "
                      f"SampleFormat {fmt[0]} with {fmt[1]} bits", policy="vc3d", sample_format=fmt[0],
                      bits_per_sample=fmt[1])
    factors = _integer_factors(mask.channel0.shape, s.shape)
    if factors is None:
        return valid, [_mask_ignored("vc3d", "not the grid size or an integer multiple of it", mask, s)]
    (rows, cols), (ky, kx) = s.shape, factors
    keep = (mask.channel0 >= 255).reshape(rows, ky, cols, kx).all(axis=(1, 3))
    return valid & keep, []


def _integer_factors(mask_shape: tuple[int, ...], grid_shape: tuple[int, int]) -> tuple[int, int] | None:
    if any(m % g for m, g in zip(mask_shape, grid_shape, strict=True)):
        return None
    return mask_shape[0] // grid_shape[0], mask_shape[1] // grid_shape[1]


def _rule_spiral(s: TifxyzSurface, mask: _Mask | None) -> tuple[np.ndarray, list[dict[str, Any]]]:
    valid = (s.x != -1) | (s.y != -1) | (s.z != -1)
    if mask is None:
        return valid, []
    if mask.channel0.shape != s.shape:
        raise Refusal("IO_TIFXYZ_MASK_SHAPE", f"spiral policy: mask.tif channel 0 is {list(mask.channel0.shape)}"
                      f", grid is {list(s.shape)}; the spiral loader's boolean index fails",
                      policy="spiral", mask_shape=list(mask.channel0.shape), grid_shape=list(s.shape))
    return valid & (mask.channel0 != 0), []


def _rule_vesuvius(s: TifxyzSurface, mask: _Mask | None) -> tuple[np.ndarray, list[dict[str, Any]]]:
    fallback = (s.z > 0) & np.isfinite(s.z)
    if mask is None:
        return fallback, []
    if not mask.layout["single_ifd"]:
        raise Refusal("IO_TIFXYZ_MASK_UNSUPPORTED", "vesuvius policy: a multi-page mask.tif is read through "
                      "tifffile series grouping, which apok-eval does not model", policy="vesuvius")
    if mask.array.shape != s.shape:
        return fallback, [_mask_ignored("vesuvius", "shape differs from the grid", mask, s)]
    return mask.array != 0, []


def _rule_lasagna(s: TifxyzSurface, mask: None) -> tuple[np.ndarray, list[dict[str, Any]]]:
    valid = (s.x != -1) & (s.y != -1) & (s.z != -1)
    if s.mask_tiff is None:
        return valid, []
    return valid, [{"finding": "mask-ignored", "policy": "lasagna", "reason": "lasagna never reads mask.tif"}]


def _rule_apok_strict(s: TifxyzSurface, mask: _Mask | None) -> tuple[np.ndarray, list[dict[str, Any]]]:
    valid = np.isfinite(s.x) & np.isfinite(s.y) & np.isfinite(s.z) & (s.z > 0)
    if mask is None:
        return valid, []
    if not mask.layout["single_ifd"]:
        raise Refusal("IO_TIFXYZ_MASK_UNSUPPORTED", "apok-strict policy: mask.tif holds more than one IFD",
                      policy="apok-strict")
    if mask.array.shape != s.shape:
        raise Refusal("IO_TIFXYZ_MASK_SHAPE", f"apok-strict policy: mask.tif is {list(mask.array.shape)}, grid is "
                      f"{list(s.shape)}", policy="apok-strict", mask_shape=list(mask.array.shape),
                      grid_shape=list(s.shape))
    bad = (mask.array != 0) & (mask.array != 255)
    if bad.any():
        row, col = (int(i) for i in np.argwhere(bad)[0])
        raise Refusal("IO_TIFXYZ_MASK_VALUES", f"apok-strict policy: {int(bad.sum())} mask.tif values are neither "
                      f"0 nor 255 (first at row {row}, col {col}: {mask.array[row, col]!r})", policy="apok-strict",
                      count=int(bad.sum()), row=row, col=col, value=_json_safe(mask.array[row, col].item()))
    return valid & (mask.array == 255), []


#: policy -> (rule, whether the rule reads mask.tif). lasagna never decodes the mask, so an
#: undecodable mask cannot stop it.
_POLICY_RULES: dict[str, tuple[Callable[..., tuple[np.ndarray, list[dict[str, Any]]]], bool]] = {
    "vc3d": (_rule_vc3d, True),
    "spiral": (_rule_spiral, True),
    "vesuvius": (_rule_vesuvius, True),
    "lasagna": (_rule_lasagna, False),
    "apok-strict": (_rule_apok_strict, True),
}


@dataclass(frozen=True, eq=False)
class ValidityReport:
    """Several policies on one surface, the grid census, and pairwise disagreement counts."""

    shape: tuple[int, int]
    results: dict[str, PolicyResult]
    census: dict[str, int]
    disagreements: tuple[dict[str, Any], ...]
    bounds: dict[str, dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "shape_hw": list(self.shape),
            "census": dict(self.census),
            "policies": {
                name: {"valid": int(r.valid.sum()), "invalid": int(r.valid.size - r.valid.sum()),
                       **self.bounds[name], "findings": list(r.findings)}
                for name, r in self.results.items()
            },
            "disagreements": list(self.disagreements),
        }


def validity_report(surface: TifxyzSurface, policies: tuple[str, ...] | list[str] = POLICIES) -> ValidityReport:
    """Evaluate ``policies`` (refusing if any refuses) and compare every pair."""
    return _report(surface, {name: validity(surface, name) for name in policies})


def _report(surface: TifxyzSurface, results: dict[str, PolicyResult]) -> ValidityReport:
    names = list(results)
    disagreements = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            va, vb = results[a].valid, results[b].valid
            only_a, only_b = int((va & ~vb).sum()), int((vb & ~va).sum())
            disagreements.append({"a": a, "b": b, "only_a": only_a, "only_b": only_b, "differ": only_a + only_b})
    return ValidityReport(shape=surface.shape, results=results, census=grid_census(surface),
                          disagreements=tuple(disagreements),
                          bounds={name: _bounds(surface, r.valid) for name, r in results.items()})


def grid_census(surface: TifxyzSurface) -> dict[str, int]:
    """Policy-independent counts of the vertex kinds the policies disagree about."""
    x, y, z = surface.x, surface.y, surface.z
    marked = (x == -1) | (y == -1) | (z == -1)
    sentinel = (x == -1) & (y == -1) & (z == -1)
    return {
        "vertices": int(z.size),
        "full_sentinel": int(sentinel.sum()),
        "partial_sentinel": int((marked & ~sentinel).sum()),
        "nonfinite": int((~(np.isfinite(x) & np.isfinite(y) & np.isfinite(z))).sum()),
        "nan_z": int(np.isnan(z).sum()),
        "z_nonpositive_unmarked": int(((z <= 0) & ~marked).sum()),
        "negative_zero_z": int(((z == 0) & np.signbit(z)).sum()),
    }


def _bounds(surface: TifxyzSurface, valid: np.ndarray) -> dict[str, Any]:
    """Bounds of valid vertices with finite coordinates (the stored bbox is never trusted)."""
    points = _widen(surface.zyx[valid])
    finite = np.isfinite(points).all(axis=1)
    out: dict[str, Any] = {"valid_nonfinite": int((~finite).sum()), "min_zyx": None, "max_zyx": None}
    if finite.any():
        out["min_zyx"] = [float(v) for v in points[finite].min(axis=0)]
        out["max_zyx"] = [float(v) for v in points[finite].max(axis=0)]
    return out


def _widen(points: np.ndarray) -> np.ndarray:
    # Widening a signalling NaN raises the FP "invalid" flag; the value is still NaN, so silence it.
    with np.errstate(invalid="ignore"):
        return points.astype(np.float64)


def check_within_frame(surface: TifxyzSurface, frame: VolumeFrame, policy: str) -> int:
    """Refuse unless what ``meta.json`` states about the surface's volume (``volume``, ``target_volume``,
    ``base_shape_zyx``; :func:`apok_eval.io.frame.stated_identity`) agrees with ``frame`` and every vertex ``policy``
    counts as valid lies inside it; return how many vertices."""
    stated_identity(frame, surface.meta, f"tifxyz {surface.dir_name}")
    result = validity(surface, policy)
    points = _widen(surface.zyx[result.valid])
    check_points_within(frame, points, f"tifxyz {surface.dir_name} ({policy} policy)")
    return int(len(points))


def full_resolution_extent(stored: int, scale: float) -> int:
    """Full-resolution canvas extent of ``stored`` grid cells: max(1, lround(stored * (1 / float32(scale)))).

    The reciprocal of the float32 scale is taken in double and rounded half away
    from zero, as vc_render_tifxyz does (villa vesuvius/src/vesuvius/tifxyz_canvas.py:15-41);
    152 cells at scale 0.05 give 3040, where truncation would give 3039.
    ``stored`` must be a non-bool integer in [0, 2**53] (exact as a double).
    """
    if isinstance(stored, (bool, np.bool_)) or not isinstance(stored, (int, np.integer)) \
            or not 0 <= stored <= 2 ** 53:
        raise Refusal("IO_TIFXYZ_EXTENT_INVALID", f"stored extent {stored!r} is not an integer in [0, 2**53]",
                      value=repr(stored))
    scale32 = _as_positive_float32(scale)
    if scale32 is None:
        raise Refusal("IO_TIFXYZ_SCALE_INVALID", f"scale {scale!r} is not a positive finite float32",
                      value=repr(scale))
    quotient = float(stored) * (1.0 / scale32)
    whole = math.floor(quotient)
    return max(1, whole + 1 if quotient - whole >= 0.5 else whole)


def inspect_tifxyz(path: str | os.PathLike[str], policies: tuple[str, ...] | list[str] = POLICIES) -> dict[str, Any]:
    """Summary for ``apok-eval io inspect``: the surface plus each policy's verdict.

    A policy that refuses (e.g. apok-strict on a 0/1 mask) is reported under
    ``refused`` with its code and reason; the others are compared pairwise.
    """
    surface = read_tifxyz(path)
    results: dict[str, PolicyResult] = {}
    refused: dict[str, Any] = {}
    for name in policies:
        try:
            results[name] = validity(surface, name)
        except Refusal as refusal:
            refused[name] = refusal.as_dict()
    return {"surface": surface.as_dict(), "validity": _report(surface, results).as_dict(),
            "refused_policies": refused}


# --------------------------------------------------------------------------------------------
# Canonical writer

_SHORT, _LONG = 3, 4
_IFD_OFFSET = 8
_TAG_COUNT = 11
# Header (8) + entry count (2) + entries (12 each) + next-IFD pointer (4), rounded up to 4 bytes so
# float32 samples are aligned and the file can be memory-mapped.
_DATA_OFFSET = -(-(_IFD_OFFSET + 2 + 12 * _TAG_COUNT + 4) // 4) * 4


def canonical_tiff_header(height: int, width: int) -> bytes:
    """The bytes before the pixel data of apok-eval's canonical float32 TIFF (``_DATA_OFFSET`` long).

    Layout: "II", 42, IFD at offset 8; one IFD of 11 entries in tag order
    (ImageWidth, ImageLength, BitsPerSample 32, Compression 1, Photometric 1,
    StripOffsets, SamplesPerPixel 1, RowsPerStrip H, StripByteCounts 4HW,
    PlanarConfig 1, SampleFormat 3), next-IFD pointer 0, zero padding to
    offset 148, then H*W little-endian float32 samples in one strip.
    """
    if height < 1 or width < 1:
        raise Refusal("IO_TIFXYZ_WRITE_INVALID", f"a plane must be at least 1 x 1, not {height} x {width}",
                      field="shape", shape=[height, width])
    nbytes = height * width * 4
    if nbytes > BIGTIFF_THRESHOLD:
        raise Refusal("IO_TIFXYZ_WRITE_TOO_LARGE", f"{height} x {width} float32 is {nbytes} bytes, above "
                      f"{BIGTIFF_THRESHOLD:#x}; C++ would switch to BigTIFF, which this writer refuses",
                      shape=[height, width], nbytes=nbytes)
    entries = (
        (256, _LONG, width), (257, _LONG, height), (258, _SHORT, 32), (259, _SHORT, 1), (262, _SHORT, 1),
        (273, _LONG, _DATA_OFFSET), (277, _SHORT, 1), (278, _LONG, height), (279, _LONG, nbytes),
        (284, _SHORT, 1), (339, _SHORT, 3),
    )
    out = bytearray(b"II" + struct.pack("<HI", 42, _IFD_OFFSET) + struct.pack("<H", len(entries)))
    for tag, kind, value in entries:
        value_field = struct.pack("<HH", value, 0) if kind == _SHORT else struct.pack("<I", value)
        out += struct.pack("<HHI", tag, kind, 1) + value_field
    out += struct.pack("<I", 0)
    out += bytes(_DATA_OFFSET - len(out))
    return bytes(out)


def encode_float32_tiff(plane: np.ndarray, name: str = "plane") -> bytes:
    """Canonical TIFF bytes for one (H, W) float32 plane, bit patterns preserved."""
    _check_plane(name, plane)
    header = canonical_tiff_header(*plane.shape)
    return header + plane.view(plane.dtype.str[0] + "u4").astype("<u4").tobytes()


def _check_plane(name: str, plane: Any) -> None:
    if not isinstance(plane, np.ndarray) or plane.ndim != 2 or plane.dtype.kind != "f" or plane.dtype.itemsize != 4:
        raise Refusal("IO_TIFXYZ_WRITE_INVALID", f"{name} must be a 2-D float32 numpy array, not "
                      f"{getattr(plane, 'dtype', type(plane).__name__)} {list(getattr(plane, 'shape', []))}",
                      field=name)


def encode_meta_json(meta: dict[str, Any]) -> bytes:
    """``meta.json`` bytes as C++ writes them: sorted keys, indent 4, trailing newline; values as given."""
    try:
        return jsonstrict.dumps(meta)
    except jsonstrict.StrictJsonError as exc:
        raise Refusal("IO_TIFXYZ_WRITE_INVALID", f"meta must hold only JSON the reader accepts: {exc}",
                      field="meta", problem=str(exc)) from None


def write_tifxyz(directory: str | os.PathLike[str], *, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                 meta: dict[str, Any], overwrite: bool = False) -> dict[str, str]:
    """Write a canonical tifxyz directory; return {file name: digest} for the four files written.

    Refuses an existing ``directory`` unless ``overwrite``; with ``overwrite``
    it replaces x.tif, y.tif, z.tif and meta.json (each atomically, meta.json
    last) and touches no other file. A ``directory`` that is a symbolic link
    is refused rather than followed out of the declared path. mask.tif and
    other channels are never written: the caller owns them.
    """
    target = os.fspath(directory)
    planes = {"x.tif": x, "y.tif": y, "z.tif": z}
    for name, plane in planes.items():
        _check_plane(name, plane)
    shapes = {name: list(plane.shape) for name, plane in planes.items()}
    if len({tuple(s) for s in shapes.values()}) != 1:
        raise Refusal("IO_TIFXYZ_WRITE_INVALID", f"x, y and z must share one shape, got {shapes}", field="shape",
                      shapes=shapes)
    check_meta(meta, "meta given to write_tifxyz")
    files = {name: encode_float32_tiff(plane, name) for name, plane in planes.items()}
    files[META_FILE] = encode_meta_json(meta)

    if os.path.lexists(target):
        if not overwrite:
            raise Refusal("IO_TIFXYZ_WRITE_EXISTS", f"{target} exists; pass overwrite=True to replace its "
                          "x/y/z.tif and meta.json", path=target)
        if os.path.islink(target):
            raise Refusal("IO_TIFXYZ_WRITE_SYMLINK", f"{target} is a symbolic link to "
                          f"{portable_name(os.path.realpath(target))}; the writer does not follow it", path=target)
        if not os.path.isdir(target):
            raise Refusal("IO_TIFXYZ_NOT_DIRECTORY", f"{target} exists and is not a directory", path=target)
    else:
        os.makedirs(target)
    for name, data in files.items():
        write_atomic(os.path.join(target, name), data, overwrite=True)
    return {name: sha256_bytes(data) for name, data in files.items()}
