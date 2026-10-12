# `apok_eval.io` surfaces — tifxyz, triangle meshes, point collections, umbilicus

Tested on synthetic fixtures. `apok-eval e7v3` reads a fit's segments and patches with the tifxyz reader and
its umbilicus with the umbilicus reader; the mesh and point-collection readers are used by `apok-eval inspect`.
Modules: `src/apok_eval/io/tifxyz.py`, `mesh.py`, `pointcollection.py`, `umbilicus.py`. External formats are cited as
`file:line` in ScrollPrize/villa at commit `fb8c2c4`. villa's root is MIT and its `volume-cartographer/` (VC3D)
sources are GPL-3.0; both were read to learn the formats, and no code is copied
([THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)).

## 1. Rules shared by all four readers

- **Axes.** apok-eval's coordinate order is always (z, y, x) in level-0 voxels: z indexes OME-Zarr axis 0. Every format
  here stores x first (tifxyz bands, OBJ/PLY `x y z`, PCL `p = [x, y, z]`) except the umbilicus (z, y, x keys);
  each reader exposes the (z, y, x) view explicitly (`zyx`, `vertices_zyx`, `points_zyx()`, `control_zyx`) and
  says so in its summary (`"axes": ["z", "y", "x"]`).
- **Frames are declared, never inferred, and checked against what the artefact states.** No artefact here names
  its volume reliably enough to stand in for a declaration; where one does state a volume id or level-0 shape
  (tifxyz `meta.json`, umbilicus metadata), it must agree with the declared frame (`IO_FRAME_VOLUME_MISMATCH`,
  §2.5). Each module has a helper
  that runs `apok_eval.io.frame.check_points_within` on the points it would hand on (`check_within_frame`,
  `check_mesh_within`, `check_point_collections_within`, and `read_umbilicus(..., frame=)`), which refuses with
  `IO_FRAME_POINTS_OUTSIDE` / `IO_FRAME_POINTS_MALFORMED` (codes owned by `frame.py`).
- **Strict JSON** (`apok_eval.jsonstrict.loads`, [docs/jsonstrict.md](jsonstrict.md), used by the tifxyz, PCL and umbilicus
  readers): UTF-8 without BOM; a duplicate object key at any depth is refused (Python's `json` silently keeps the
  last); a number that overflows a double is refused (`1e400` would become infinity); an integer too long for
  Python's `int()` (more than `sys.get_int_max_str_digits()` digits, 4300 by default, where `json` raises a bare
  `ValueError`), nesting deeper than 256 levels (and, deeper still, nesting that would exhaust the parser's
  recursion) and a string or key holding a lone surrogate (an unpaired `\ud800`-style escape: not Unicode, it
  could not be digested or written back) are refused. Shorter integers stay exact Python ints and each field check
  judges them (a 400-digit coordinate is refused by the field that holds it, naming it). `NaN`/`Infinity`/
  `-Infinity`: the PCL and umbilicus readers refuse them (`non-standard constant NaN`); the tifxyz reader parses
  `meta.json` with `constants="collect"`, keeps them and records the finding `meta-nonstandard-json` (spiral can
  write a NaN bbox with `json.dump`, and nlohmann::json, which VC3D uses, cannot parse such a file). JSON integers
  and floats keep their Python types (`1` stays int, `1.0` float). The problem string names the cause (e.g.
  `duplicate key 'format'`) in the refusal's reason.
- **Findings vs refusals.** A refusal (`apok_eval.refusal.Refusal`) stops the read and names the offending file,
  field, id, line or index in `details`. A finding is a dict `{"finding": "<slug>", ...}` attached to the result:
  something a consumer must know but that is not malformed.
- **Writers** are deterministic (no timestamps, comments or library tags) and refuse an existing target unless
  `overwrite=True`. Files are written by `apok_eval.fileio.write_atomic`, the one atomic writer the ledger uses too:
  a temporary file (`.apok-tmp-<hex>`) in the same directory, created with `O_EXCL` and mode 0o666 (umask
  applies) and fsynced, then `os.replace`d (overwrite) or hard-linked (no overwrite, so a file that appeared
  meanwhile is never clobbered: `apok_eval.fileio.TargetExists`, a `FileExistsError`), then the directory fsynced,
  so the tifxyz writer's "meta.json last" order holds on disk after a crash. Nothing other than the target and
  its temporary file is touched; an
  existing target file that is a symlink is replaced, not followed, and the tifxyz writer refuses a target
  directory that is a symlink (2.7). Every writer refuses data its own reader would refuse: JSON goes through
  `apok_eval.jsonstrict.dumps` (sorted keys, indent 4, raw UTF-8, `"\n"`), which refuses what is not plain finite
  JSON (`apok_eval.canonical` rules), an integer too long for `str()`, nesting too deep to encode and anything its own
  bytes would fail to read back as (nesting past 256 levels), naming the problem in the refusal's `problem`
  detail; the PCL writer takes the reader's `role` and applies the absolute
  rule (4).
- **Digests** are `sha256:` of the exact bytes read (`apok_eval.canonical.sha256_bytes`); bytes are read once and the
  same bytes are digested and decoded.
- Every file is read through `apok_eval.fileio.read_regular`: opened without blocking and refused unless it is a
  regular file, so a FIFO never blocks a reader and a device such as `/dev/zero` is never read; the refusal
  names what was found (`problem`: `missing`, `not a regular file (a FIFO)` and so on, or `not a valid path`
  for a path holding NUL).
- I/O errors that are not format questions (permission denied, a vanished file during a read) propagate as
  `OSError`; they are loud, never defaulted. `apok-eval inspect` turns them into `IO_INSPECT_UNKNOWN` with
  `problem: "unreadable"`, the exception type and the errno. Malformed content is meant never to escape as a bare Python
  exception: every such path found so far (`IndexError`, `struct.error`, `ValueError`, `ZeroDivisionError`,
  `RecursionError`, `MemoryError` from damaged TIFF structure, huge integers, deep nesting or hostile counts) is a
  refusal with its own code and a test.

## 2. tifxyz (`apok_eval.io.tifxyz`)

### 2.1 Container and grid
A surface is a directory with `x.tif`, `y.tif`, `z.tif`, `meta.json` (all required, as the vesuvius reader
requires, `vesuvius/src/vesuvius/tifxyz/reader.py:381-388`). Grid vertex (row r, col c) is TIFF image line r,
sample c; its point is (X, Y, Z) = (x.tif[r,c], y.tif[r,c], z.tif[r,c]) (`lasagna/tifxyz_format.md:25-31`). The
C++ sampler maps `vz=p[2], vy=p[1], vx=p[0]` (`volume-cartographer/core/src/Slicing.cpp:447,495`), so z.tif feeds
OME-Zarr axis 0. `TifxyzSurface.zyx` is the (H, W, 3) stack `[z, y, x]` (as spiral and vesuvius stack it,
`spiral-fitting/tifxyz.py:402`, `vesuvius/src/vesuvius/tifxyz/types.py:765-766`).

`read_tifxyz(path)` never writes to the directory. It reads `meta.json`, x/y/z.tif and (if present) `mask.tif`
into memory once, then inventories every directory entry (sorted by name, not descending): `kind` (`file`,
`directory`, `symlink` with its `target`, `other`), and for regular files `size` and `digest`. A symlinked core
file is read through and recorded as the finding `symlink-followed` with its resolved target; other links are
recorded but not followed. A required name that exists but is not a regular file (directory, FIFO, device,
dangling link) is refused (`IO_TIFXYZ_FILE_KIND`, naming the `kind`), as is a `mask.tif` of that kind. A regular
file the surface does not need that cannot be read (`EACCES`) is recorded with `unreadable` (the error) instead
of a digest, plus the finding `entry-unreadable`; it does not stop the read. An entry name or link target that is not
valid UTF-8 (Python decodes it to lone surrogates, which no canonical JSON can hold) is recorded with its
undecodable bytes written `\xNN` (`apok_eval.fileio.portable_name`) plus `name_hex` / `target_hex`, the exact bytes in hex,
and the finding `name-not-utf8`; `as_dict()` writes `path` and `dir_name` the same way.

### 2.2 TIFF decoding (x/y/z and mask)
tifffile decodes the bytes; apok-eval checks the structure first. The first IFD is the image; for x/y/z the next-IFD
pointer, read directly from the file (tifffile truncates a damaged IFD chain with only a log warning), must be 0,
otherwise `IO_TIFXYZ_TIFF_PAGES`. A mask may have more IFDs (C++ `writeValidMask` writes `[mask, img]`,
`QuadSurface.cpp:1597-1611`); its first IFD is the mask, and its layout record holds that IFD's next-IFD pointer
(`single_ifd`, `next_ifd_offset`), not a page count.

tifffile tolerates several damaged structures with only a log line, so apok-eval checks them itself, before decoding,
for the bands and the mask alike: libtiff writes the IFD after the pixel data, so a download cut short leaves a
header whose first-IFD offset points past the end, which tifffile opens as a file with no pages; a short
StripOffsets/TileOffsets list makes tifffile zero-fill the missing strips or tiles (`read_segments: expected 2
segments, got 1`), and a long one is silently cut.

| check (x/y/z) | refusal |
|---|---|
| bytes parse as TIFF (classic or BigTIFF, `II` or `MM`) | `IO_TIFXYZ_TIFF_INVALID` (`file`) |
| the header's first-IFD offset leads to an IFD (not 0, not past the end) | `IO_TIFXYZ_TIFF_INVALID` (`field` = `first_ifd_offset`, `value` = the offset) |
| every IFD value read (sizes, samples, bits, sample format, planar configuration, photometric, compression, predictor) is one integer: a tag that is missing where tifffile needs it, empty (`count` 0), ASCII or holding several values is refused (also for the mask) | `IO_TIFXYZ_TIFF_INVALID` (`field` names the tag, `value` null) |
| any other exception tifffile's lazily derived values raise during these checks (`TypeError`, `ValueError`, `struct.error`, `ZeroDivisionError`, `OverflowError`, `MemoryError`) | `IO_TIFXYZ_TIFF_INVALID` (`field` = `structure`, `error` = the exception type) |
| the first IFD's next-IFD pointer lies inside the file | `IO_TIFXYZ_TIFF_INVALID` (`field` = `next_ifd_offset`) |
| exactly one IFD | `IO_TIFXYZ_TIFF_PAGES` |
| SamplesPerPixel 1, BitsPerSample 32, SampleFormat 3 (IEEE float) | `IO_TIFXYZ_TIFF_SAMPLE` (`field` names which) |
| ImageDepth 1 (also for the mask) | `IO_TIFXYZ_TIFF_SAMPLE` (`field` = `image_depth`) |
| ImageLength, ImageWidth, SamplesPerPixel, BitsPerSample and RowsPerStrip, or TileLength and TileWidth (any tile tag makes the page tiled here; tifffile calls a page with TileWidth 0 striped), are at least 1 (also for the mask: SamplesPerPixel 0 once divided by zero) | `IO_TIFXYZ_TIFF_INVALID` (`field` = `image_length`, `image_width`, `samples_per_pixel`, `bits_per_sample`, `rows_per_strip`, `tile_length` or `tile_width`) |
| Compression in {1 none, 5 LZW, 8 Adobe Deflate, 32946 Deflate}; Predictor in {1, 2, 3} | `IO_TIFXYZ_TIFF_COMPRESSION` |
| the codec is available: LZW and the floating-point predictor (3) need `imagecodecs`; Deflate uses zlib and the horizontal predictor (2) has tifffile's numpy fallback (`TIFF.UNPREDICTORS[2]`, integer differences of the 32-bit words, so float32 bits are kept exactly; tested) | `IO_TIFXYZ_CODEC_MISSING` (`codec` = `LZW`, `predictor floating point`, or, where tifffile lacks the fallback, `predictor horizontal`) |
| the StripOffsets/StripByteCounts (or TileOffsets/TileByteCounts) tags each declare exactly as many entries as the geometry needs (ceil(H / RowsPerStrip), or the tile grid, times the planes); a missing tag declares 0. The count is computed arithmetically and compared before any per-segment list is built, so a 196-byte file declaring ImageLength 0xFFFFFF00 and RowsPerStrip 1 is refused instead of making the reader build a 4.3-billion-entry list | `IO_TIFXYZ_TIFF_SEGMENTS` (`expected`, `offsets`, `byte_counts`) |
| every strip/tile has offset > 0 and byte count > 0 (tifffile would zero-fill a sparse one) | `IO_TIFXYZ_TIFF_SEGMENTS` |
| every strip/tile lies inside the file (a truncated download) | `IO_TIFXYZ_TIFF_SEGMENTS` |
| an uncompressed strip/tile holds its full size (last strip: remaining rows only) | `IO_TIFXYZ_TIFF_SEGMENTS` |
| tifffile decodes without error (any exception, e.g. corrupt zlib data) | `IO_TIFXYZ_TIFF_DECODE` |
| x, y and z have one height x width | `IO_TIFXYZ_SHAPE_MISMATCH` |

C++ instead accepts 8/16/32/64-bit int/uint/float bands and zero-fills undecodable tiles or scanlines
(`QuadSurface.cpp:2750-2761, 2790-2814, 2846-2850, 2873-2875`); apok-eval refuses both, because a converted or
zero-filled band is not the file's data and a truncated file must not look like a holey surface. Arrays are
returned little-endian float32, read-only, with every bit pattern kept (-0.0, quiet and signalling NaN payloads):
big-endian files are byte-swapped through a uint32 view, never through float arithmetic. Each band's layout
record (`TifxyzSurface.tiffs`) holds digest, size, byte order, BigTIFF flag, IFD pointer, sample layout,
photometric, compression, predictor, tiling/strip geometry and segment count.

A zero-size mask is refused like a band (`field` = `image_length` or `image_width`): vc3d would otherwise have
taken a 0 x 0 mask as a factor-0 "integer multiple" of the grid and kept every vertex without a finding.

Supported producer layouts: the C++ and spiral (Pillow) single uncompressed strip
(`QuadSurface.cpp:2269-2278`, `spiral-fitting/tifxyz.py:435-437`; Pillow omits SamplesPerPixel, which TIFF
defaults to 1), lasagna/merge tifffile uncompressed strips (`lasagna/fit2tifxyz.py:956-961`), and the vesuvius
tiled layout (`vesuvius/src/vesuvius/tifxyz/writer.py:126-145`); vesuvius' LZW needs `imagecodecs`, which is not
installed in this environment, so its LZW output is refused with `IO_TIFXYZ_CODEC_MISSING` (tests cover the
uncompressed and Deflate tiled variants).

### 2.3 meta.json
Parsed strictly (section 1) and kept losslessly in `TifxyzSurface.meta`. Required (`check_meta`, also run by the
writer):

| field | rule | refusal |
|---|---|---|
| (document) | a JSON object | `IO_TIFXYZ_META_INVALID` |
| `format` | the string `"tifxyz"` | `IO_TIFXYZ_META_FIELD`, `field=format` |
| `type` | the string `"seg"` (VC3D requires it, `Segmentation.cpp:31-37`) | `field=type` |
| `uuid` | a non-empty string (C++ throws without it, `QuadSurface.cpp:3059`) | `field=uuid` |
| `scale` | a list of exactly 2 numbers [sx, sy], not bool, finite, > 0, and still finite and > 0 after narrowing to float32 (C++ holds it as float32, `QuadSurface.cpp:2890-2891`) | `field=scale` |

`scale` = [sx, sy] = grid samples per base voxel along columns and rows, the reciprocal of the vertex spacing
(`lasagna/tifxyz_format.md:111-115`); spiral writes [1/step, 1/step] = [0.05, 0.05] for its default step 20
(`spiral-fitting/config.py:736`). It is exposed verbatim (`scale`) and as float32 (`scale_float32`, e.g.
0.05000000074505806). The stored `bbox` is never used (producers emitted several axis orders,
`spiral-fitting/tifxyz.py:357-363`); bounds come from decoded valid vertices (2.6). Findings:
`uuid-differs-from-directory` (both recorded; C++ ids can follow either, `QuadSurface.cpp:1029,2208,3059`),
`scale-above-one` (a component > 1 looks like a step size), `scale-vesuvius-default` (exactly (20, 20), the
vesuvius reader's silent default for a missing scale, `vesuvius/src/vesuvius/tifxyz/reader.py:412-419`),
`meta-nonstandard-json`.

### 2.4 Validity policies
Validity is never applied while reading. `validity(surface, policy)` returns a read-only (H, W) bool array and
policy findings; `validity_report(surface, policies)` evaluates several (refusing if one refuses) and counts, for
every pair (a, b), the vertices valid only under a and only under b. Policies (`POLICIES`):

| policy | rule | source |
|---|---|---|
| `vc3d` | invalid iff z <= 0 (so -0.0 and +0.0; NaN z stays valid); then mask.tif channel 0 < 255 invalidates, for SampleFormat/bits uint 8/16/32, int 16/32, float 32/64; the mask may be the grid size or an exact integer multiple per axis (the vertex survives only if every mask pixel of its block is >= 255); a non-multiple mask is ignored (finding `mask-ignored`); any other mask format (e.g. int8, whose C++ threshold is undefined) is refused | `QuadSurface.cpp:2899-2903, 2926-2971` |
| `spiral` | mask.tif channel 0 == 0 sets the vertex to -1; valid iff ANY of x, y, z != -1 (so z <= 0, NaN and partial sentinels are valid); a channel-0 shape other than the grid is refused (the spiral loader's boolean index fails) | `spiral-fitting/tifxyz.py:41-43, 382-404` |
| `vesuvius` | a mask whose whole array shape equals the grid decides alone: valid iff mask != 0 (z is not checked); otherwise (no mask, or a mismatched one, which vesuvius silently drops: finding `mask-ignored`) valid iff z > 0 and z finite; a multi-page mask is refused (tifffile series grouping, which apok-eval does not model) | `vesuvius/src/vesuvius/tifxyz/reader.py:541-549, 666-685` |
| `lasagna` | valid iff ALL of x, y, z != -1; mask.tif never read (finding `mask-ignored` when present; an undecodable mask does not stop it) | `lasagna/tifxyz_io.py:78-87` |
| `apok-strict` | valid iff x, y, z all finite and z > 0 (which also excludes (-1, -1, -1)) and, with a mask, mask == 255; a mask value outside {0, 255}, a mask whose shape is not the grid's, or a multi-page mask is refused | apok-eval's own policy |

Mask channel 0 is the first sample of a contiguous multi-sample mask or the first plane of a planar-separate one.
A mask is decoded only when a policy reads it, from the bytes read with the surface, with the same TIFF checks as
the bands (except the float32/single-IFD rules). `grid_census(surface)` counts, independently of any policy:
`vertices`, `full_sentinel` (-1, -1, -1), `partial_sentinel` (some but not all components -1, e.g. (5, -1, -1)),
`nonfinite`, `nan_z`, `z_nonpositive_unmarked` (z <= 0 with no -1 component) and `negative_zero_z`.

The test fixture plants one of each disputed vertex; the policies reject exactly these cells (no mask):

| vertex (row, col) | (x, y, z) | vc3d | spiral | vesuvius | lasagna | apok-strict |
|---|---|---|---|---|---|---|
| (0,0) | z = -0.0 | invalid | valid | invalid | valid | invalid |
| (0,1) | z = NaN (0x7fc00001) | valid | valid | invalid | valid | invalid |
| (0,2) | x = sNaN (0x7f800001) | valid | valid | valid | valid | invalid |
| (1,0) | z = -3.5 | invalid | valid | invalid | valid | invalid |
| (1,1) | (5, -1, -1) | invalid | valid | invalid | invalid | invalid |
| (1,2) | (-1, -1, -1) | invalid | invalid | invalid | invalid | invalid |
| (2,0) | (-1, 7, 9) | valid | valid | valid | invalid | valid |
| (2,1) | z = +0.0 | invalid | valid | invalid | valid | invalid |

A 0/255 mask with zeros at two ordinary cells adds those two cells to vc3d, spiral and apok-strict and makes
vesuvius reject only those two (every disputed cell above becomes valid under vesuvius). A 0/1 mask invalidates
every vertex under vc3d (1 < 255), behaves like 0/255 under spiral and vesuvius, and is refused by apok-strict.
A z = +inf vertex is valid under vc3d and invalid under vesuvius and apok-strict.

### 2.5 Coordinate contract and canvas
`check_within_frame(surface, frame, policy)` first reads off what `meta.json` itself states about the volume
(`apok_eval.io.frame.stated_identity`, the rule "read off the artefact and checked before any read"): `volume`
(VC3D's `Segmentation::ensureScrollSource`, the volume uuid) and `target_volume` (`vc_grow_seg_from_seed`, the
volume directory name, compared by its leading 14-digit id) must name the declared frame's `volume_id`, and
`base_shape_zyx` (spiral combined previews, lasagna exports) must be its level-0 shape, else
`IO_FRAME_VOLUME_MISMATCH` (`field`, `stated`, `declared`): a surface traced on another volume is refused even
when its coordinates would fit. A value that names no 14-digit id, or a shape that is not three positive ints,
is not comparable and is reported so, never guessed; an absent key is "not stated by the artefact" (most
spiral per-winding meshes state nothing). `apok-eval io inspect --frame` records the outcome per key in the
summary's `stated_identity`. It then passes the vertices `policy` counts as valid to `check_points_within`; it
returns how many were checked. NaN-bearing vertices counted valid by vc3d or lasagna are refused as malformed,
and apok-strict's partial sentinel (-1, 7, 9) is refused as outside.

`full_resolution_extent(stored, scale)` = max(1, lround(stored * (1 / float32(scale)))), reciprocal in double,
halves away from zero (`vesuvius/src/vesuvius/tifxyz_canvas.py:15-41`, pinned by
`vesuvius/tests/tifxyz/test_full_resolution_shape.py:20,47-49`): 152 cells at 0.05 give 3040 (truncation would
give 3039), 7 x 5 at scale 2 give 4 x 3. A scale that is not a positive finite float32 is refused
(`IO_TIFXYZ_SCALE_INVALID`) instead of falling back to the stored extent as vesuvius does; `stored` must be a
non-bool integer (Python or numpy) in [0, 2**53], exact as a double, else `IO_TIFXYZ_EXTENT_INVALID` (a negative
count was silently raised to 1, a float or a string coerced).
`TifxyzSurface.full_resolution_shape` = (extent(H, sy), extent(W, sx)).

### 2.6 Bounds and inspection
`ValidityReport.as_dict()` gives per policy the valid/invalid counts, `valid_nonfinite`, and `min_zyx`/`max_zyx`
over valid vertices with all-finite coordinates (None if there are none). `inspect_tifxyz(path, policies)` returns
`{"surface", "validity", "refused_policies"}`: a policy that refuses (e.g. apok-strict on a 0/1 mask) is listed
with its refusal code and reason, the others are compared; a mask that cannot be decoded (no first IFD, a short
strip list, a missing codec) is such a policy refusal, not an exception. The summary is canonical JSON (sealable as
is): non-UTF-8 names are escaped as 2.1 describes.

### 2.7 Canonical writer
`write_tifxyz(directory, x=, y=, z=, meta=, overwrite=False)` writes x.tif, y.tif, z.tif, then meta.json, and
returns `{name: digest}`. Planes must be 2-D float32 numpy arrays of one shape, at least 1 x 1
(`IO_TIFXYZ_WRITE_INVALID`, `field` names the plane or `shape`); `meta` must pass `check_meta` and hold only plain
finite JSON (`IO_TIFXYZ_WRITE_INVALID`, `field=meta`; NaN is refused because nlohmann::json cannot read it). An
existing directory is refused (`IO_TIFXYZ_WRITE_EXISTS`) unless `overwrite=True`; then only the four files are
replaced and nothing else in the directory is touched or deleted (vesuvius' `rmtree` of the whole directory,
`vesuvius/src/vesuvius/tifxyz/writer.py:294-297`, is the anti-pattern). A target that is a symbolic link (even to a
directory) is refused (`IO_TIFXYZ_WRITE_SYMLINK`, `path`; the reason names the link's target) rather than followed
out of the declared path; an existing non-directory is refused (`IO_TIFXYZ_NOT_DIRECTORY`, `path`). `meta` is
encoded by `apok_eval.jsonstrict.dumps` (section 1); a refusal names the problem (`problem`, e.g. `non-finite float at
.area`). mask.tif and other channels are never written; the caller owns them.

TIFF bytes (`canonical_tiff_header` + samples), independent of any library version:

| offset | content |
|---|---|
| 0 | `II`, 42, first IFD offset 8 (classic little-endian TIFF) |
| 8 | entry count 11, then 12-byte entries in tag order: 256 ImageWidth LONG W; 257 ImageLength LONG H; 258 BitsPerSample SHORT 32; 259 Compression SHORT 1; 262 Photometric SHORT 1 (MinIsBlack); 273 StripOffsets LONG 148; 277 SamplesPerPixel SHORT 1; 278 RowsPerStrip LONG H; 279 StripByteCounts LONG 4HW; 284 PlanarConfig SHORT 1; 339 SampleFormat SHORT 3. SHORT values sit in the first two bytes of the value field, the other two are 0 |
| 142 | next-IFD offset 0 |
| 146 | two zero bytes, so samples are 4-byte aligned (tifffile then reports the page memory-mappable) |
| 148 | H x W little-endian float32 samples, row-major, one strip; nothing after |

No Software, DateTime, ImageDescription or resolution tags. BigTIFF: C++ switches to BigTIFF when pixel bytes
exceed 0xffff0000 (`volume-cartographer/core/src/Tiff.cpp:129-131`); this writer refuses such planes
(`IO_TIFXYZ_WRITE_TOO_LARGE`) before reading any sample, and writes exactly 0xffff0000 bytes classically. The
layout cannot be byte-identical to both producers: Pillow omits SamplesPerPixel and libtiff writes the IFD after
the data.

meta.json bytes: `json.dumps(meta, indent=4, sort_keys=True, ensure_ascii=False)` + `"\n"`, the form C++ writes
with nlohmann's sorted `dump(4)` (`QuadSurface.cpp:2553-2558`); values exactly as given (float formatting is
Python's shortest round-trip repr, which may differ from nlohmann's text for the same double).

Round trip (tested): read -> write -> read gives identical uint32 views of x/y/z (including -0.0 and the NaN
payloads 0x7f800001, 0x7fc00001) and equal parsed meta.json; writing twice gives byte-identical files; the
writer's files decode identically in tifffile; tifffile-written strip, multi-strip, tiled, Deflate-tiled and
big-endian planes, and hand-built Deflate strips with the horizontal predictor (`II` and `MM`), read to the same
bits.

## 3. Triangle meshes (`apok_eval.io.mesh`)

`TriangleMesh`: `vertices_xyz` (N, 3) float32 or float64 as stored, `faces` (F, 3) int64 0-based, optional OBJ
`texcoords` (T, 2|3), `normals` (M, 3), `face_texcoords`/`face_normals` (F, 3), optional PLY `vertex_properties`
((name, PLY type, (N,) array), ...), and `source` (format, digest, path, comments, ignored statements).
`vertices_zyx` is `vertices_xyz[:, ::-1]`. `read_mesh(path)` dispatches on the extension `.obj`/`.ply` (any
case), else `IO_MESH_FORMAT_UNKNOWN`. Faces with any vertex count other than 3 are refused
(`IO_MESH_FACE_NOT_TRIANGLE`), never triangulated.

**OBJ** (UTF-8 text, lines split on `\n`, surrounding whitespace and `\r` stripped, `#` lines counted):
`v x y z` (exactly 3 numbers), `vt u v [w]` (2 or 3, one arity per file), `vn x y z`, `f a b c` with corners
`v`, `v/vt`, `v//vn` or `v/vt/vn`, one form for the whole file (`IO_MESH_FACE_FORM_MIXED`). Numbers are ASCII
decimals (`[+-]digits[.digits][e[+-]digits]`; `nan`, `inf`, `1_0` and non-ASCII digits are syntax errors). Positive
indices are 1-based and checked against the file's totals after reading (forward references allowed); negative
indices count back from the elements defined so far; 0 or anything out of range is `IO_MESH_INDEX_OUT_OF_RANGE`
with the line, kind (v/vt/vn) and index. `o`, `g`, `s`, `usemtl`, `mtllib` carry no geometry and are counted in
`source["ignored"]`; any other statement (`l`, `p`, `vp`, curves, ...) is `IO_MESH_OBJ_UNSUPPORTED`; vertex
colours (extra `v` numbers) are syntax errors, and so is a `#` anywhere in a `v`, `vt`, `vn` or `f` statement (a
trailing inline comment): `IO_MESH_OBJ_SYNTAX`, reason "inline comment", checked before the arguments are counted
so that `f 1 2 3 # tri` is not mistaken for a five-vertex face. A `#` in an ignored statement (`usemtl stone#2`) is
part of a name. A face index with more than 19 significant digits (leading zeros not counted) exceeds any element
count and is `IO_MESH_INDEX_OUT_OF_RANGE` with `index` None and `digits`, before Python's `int()` digit limit
could raise. This covers what
`volume-cartographer/apps/src/vc_tifxyz2obj.cpp:112,124,137,252,301-307` writes (fixed 6-decimal `v`/`vt`/`vn`,
`f a/a/a b/b/b c/c/c`).

**PLY** (header lines split on `\n`, `\r` stripped, ASCII): `ply`, one `format ascii 1.0` or
`format binary_little_endian 1.0` (others: `IO_MESH_PLY_UNSUPPORTED`), `comment`/`obj_info` (kept in `source`),
`element <name> <count>`, `property <type> <name>`, `property list <count type> <item type> <name>`,
`end_header`. Types: char/int8, uchar/uint8, short/int16, ushort/uint16, int/int32, uint/uint32, float/float32,
double/float64; anything else `IO_MESH_PLY_PROPERTY_TYPE`. Layout (`IO_MESH_PLY_LAYOUT`): exactly the elements
`vertex` then `face`; vertex properties distinct scalars including `x`, `y`, `z`, which must be all float or all
double (`IO_MESH_PLY_PROPERTY_TYPE`); the face element holds exactly one property, `list uchar int|uint
vertex_indices`. Extra vertex properties are kept with their types. Data: binary short of the declared counts or
ascii running out of values is `IO_MESH_TRUNCATED` (element named); bytes or values after the last face are
`IO_MESH_TRAILING_DATA`; an ascii value that is not its type (or overflows it, or is out of the integer range) is
`IO_MESH_VALUE_INVALID` (element index and value named); a non-finite coordinate is `IO_MESH_VALUE_INVALID`; an
index < 0 or >= N (uint indices are not wrapped) is `IO_MESH_INDEX_OUT_OF_RANGE`. ASCII data is read as a token
stream (line breaks within the data are not checked) into lists, never into arrays sized by the header, so a header
count beyond the data (`element vertex 10000000000000`) is `IO_MESH_TRUNCATED`, not a 36 TiB allocation; the binary
reader compares the byte length with the counts first. An ASCII integer with more than 19 significant digits is
`IO_MESH_VALUE_INVALID` (no PLY integer type holds it); zero-padded integers read their value.

**Writers.** `encode_obj` / `write_obj`: `v`, `vt`, `vn`, then `f` (1-based, the mesh's corner form), numbers as
Python's shortest round-trip repr (float32 vertices round-trip exactly), `\n` line ends, nothing else.
`encode_ply` / `write_ply`: `binary_little_endian 1.0`, `x y z` as float or double per the vertex dtype, then the
vertex properties in order under their PLY 1.0 names, then `element face F` with `property list uchar int
vertex_indices`; no comments. Each writer refuses (`IO_MESH_WRITE_INVALID`, `field` named) malformed arrays
(non-finite or non-float vertices, out-of-range or non-integer indices, per-corner index arrays without their
target array or with the wrong row count), and what its format cannot carry: OBJ refuses vertex properties, PLY
refuses texcoords/normals and their per-corner indices, and more than 2**31 vertices (int32 indices).
`TriangleMesh.without_attributes()` drops them explicitly. `check_mesh_within(mesh, frame)` checks
`vertices_zyx`; `inspect_mesh(path)` summarises (counts, coordinate type, zyx bounds, attributes, source).

## 4. VC3D point collections (`apok_eval.io.pointcollection`)

Format "PointCollections v1" (`volume-cartographer/core/src/PointCollections.cpp`, `spiral-fitting/point_collection.py`):

| element | rule | refusal |
|---|---|---|
| file | exists and is a regular file (the fitter prints a warning and continues with nothing, `point_collection.py:174-181`) | `IO_PCL_FILE_MISSING` |
| bytes | strict JSON; NaN/Infinity tokens refused (C++ writes null for NaN, `PointCollections.cpp:36-45`) | `IO_PCL_JSON_INVALID` |
| document | a JSON object | `IO_PCL_STRUCTURE` (`field=document`) |
| `vc_pointcollections_json_version` | exactly the string `"1"` (both readers compare to the string, `point_collection.py:137-139`, `PointCollections.cpp:608-609`) | `IO_PCL_VERSION` |
| `collections` | required object (C++ `at()`; Python's `.get` default is not followed) | `IO_PCL_STRUCTURE` |
| collection / point keys | canonical decimal uint64: `0` or `[1-9][0-9]{0,19}` (at most 20 digits, so `int()` never meets its digit limit), at most 2**64-1; `-1`, ` 7`, `1_0`, `07`, `+1`, non-ASCII digits, 21 or more digits refused (Python `int()` and C++ `stoull` accept some of these) | `IO_PCL_ID_NONCANONICAL` (`kind`, `key`, `collection_id`) |
| collection | an object with string `name`, object `points`, and `metadata` an object when present | `IO_PCL_STRUCTURE` (`field`, `collection_id`) |
| point | an object; `p` three finite non-bool numbers [x, y, z]; `wind_a` absent/null (unannotated, NaN) or a finite number; `creation_time` absent (0) or an int64 | `IO_PCL_POINT_INVALID` (`collection_id`, `point_id`, `field`) |

Everything else (links, fiber_dir, tags, windings_linked, anchor2d, autoFillMode, autoFillConstant, color, any
`vc_open_data_*` stamp, unknown keys) is kept verbatim in `document` and not interpreted. `PointCollection`
gives collections in ascending numeric id (2 before 10, whatever the file order) and `point_ids` ascending (the
fitter's chain order), `zyx` (from p reversed), `wind_a`, `creation_time`,
`metadata`. Identity of a point is (file digest, collection id, point id) (`identities()`): ids are not
file-global in practice (`spiral-fitting/connect_overlapping_patches.py:654-657`).

**Role.** `read_point_collections(path, role=...)`: `role` is required, a fitter role (`absolute`, `relative`,
`same_winding`, `drawn_control_points`, `fit_session.py:163-167`) or None for a role-less file; anything else is
`IO_PCL_ROLE_UNKNOWN`. Derivation as `spiral-fitting/fit_spiral.py:759-769`: absolute iff the declared role is
`absolute`, or, role-less, iff the basename is exactly `abs_winding.json`; a role-less file of another name gets
role `legacy` (relative). `metadata.winding_is_absolute` is not authoritative. Findings:
`winding-flag-disagrees` (VC3D's reading of the flag, true when missing or non-boolean,
`PointCollections.cpp:94-99`, differs from the derived role; the flag is reported as written or `"absent"`),
`basename-near-miss` (role-less, not exactly `abs_winding.json`, but lower-cased alphanumerics start with `abs` and
contain `winding`, e.g. `abs_windings.json`, which the fitter treats as relative), `role-conflicts-with-basename`
(a declared role differs from the conventional role of a conventional file name, `fit_session.py:418-423`),
`fiber-dir-dropped-by-vc3d` (a fiber_dir other than h/v), `point-ids-not-file-global`.

**Validators.** Absolute: every point must have a finite `wind_a` > 0, else `IO_PCL_ABSOLUTE_WINDING` naming
collection, point and value (null as None). The fitter checks only points attached to a verified patch
(`fit_spiral.py:1779-1803`); apok-eval cannot know attachment and checks all. Non-absolute: `annotations` reports per
collection the state the fitter's normalisation (`point_collection.py:878-900`) would act on: `all` (kept),
`none` (every wind_a set to 0.0: a same-winding collection), `mixed` (the unannotated points are deleted),
`empty`, plus `has_winding_annotations` and `unannotated_point_ids`. Nothing is applied. The fitter's z-range
trimming and decimation of unattached strips are decisions for whatever consumes the collections, not checked here.

**Writer.** `encode_point_collections(document)` / `write_point_collections(path, document, role=...)`: the
document must be JSON `apok_eval.jsonstrict.dumps` accepts (`IO_PCL_WRITE_INVALID`, `problem` naming what and where, e.g.
`tuple is not canonical JSON (at .extra)`) and pass the structural checks above (same codes). `role` is required, as
for the reader (no default); the role is derived from it and the target's basename exactly as the reader derives
it, and an absolute file whose point lacks a finite `wind_a` > 0 is refused (`IO_PCL_ABSOLUTE_WINDING`, as when
reading) before anything is written. `encode_point_collections` alone applies everything but the role rule, which
needs the file name. Bytes are
`json.dumps(sort_keys=True, indent=4, ensure_ascii=False)` + `"\n"` (ids sort as strings, `"10"` before `"2"`, as
nlohmann's std::map does). Read -> write -> read preserves every key and value (floats as parsed; ints stay
ints). A load-and-save through VC3D would instead quantise p and wind_a to float32 (`PointCollections.hpp:40-42`).
`check_point_collections_within(pcl, frame)`; `inspect_point_collections(path, role=)`.

## 5. Umbilicus (`apok_eval.io.umbilicus`)

`read_umbilicus(path, variant="fitter", coordinate_scale=None, frame=None)`; `variant` is one of
(`IO_UMBILICUS_VARIANT_UNKNOWN` otherwise):

| variant | accepted | source |
|---|---|---|
| `fitter` | JSON object; `control_points` a list of objects with `z`, `y`, `x` (other point keys kept in the file, not read) | `spiral-fitting/umbilicus.py:25-33` |
| `vc3d-json` | root array, else `points`, else `control_points`; entries `[z, y, x]` (exactly 3; VC3D accepts >= 3) or `{z, y, x}` | `volume-cartographer/core/src/Umbilicus.cpp:276-324` |
| `vc3d-text` | lines `z, y, x` (exactly three non-empty ASCII decimals; VC3D skips empty tokens and accepts trailing junk via `stod`); blank and `#` lines skipped | `Umbilicus.cpp:219-266` |
| `thaumato-text` | every line `a, b, c` integers (no blank or comment lines, as the Python reader would crash on them); y = a - 500, z = b - 500, x = c - 500 | `spiral-fitting/umbilicus.py:6-13` |

Refusals: missing file `IO_UMBILICUS_FILE_MISSING`; invalid strict JSON `IO_UMBILICUS_JSON_INVALID`, except
that a JSON variant given comma text (every non-comment line three decimals) is `IO_UMBILICUS_TEXT_UNDECLARED`,
because VC3D z,y,x and Thaumato (y,z,x)+500 share that syntax; both `points` and `control_points` present under any
variant `IO_UMBILICUS_AMBIGUOUS` (VC3D reads `points`, the fitter `control_points`); a root or entry of the wrong
shape `IO_UMBILICUS_STRUCTURE` (`index` of the entry); a malformed text line `IO_UMBILICUS_TEXT_SYNTAX` (`line`);
a coordinate that is not a finite non-bool number `IO_UMBILICUS_VALUE_INVALID` (`index` or `line`, and `axis`;
a Thaumato integer too long for `int()` is read as a double, so it is this refusal, not a crash), and a finite one
that `coordinate_scale` makes overflow a double, the same code with `scale` added (unchecked, two z values became
+inf, passed the duplicate test because inf - inf is NaN, and gave NaN centres);
fewer than two control points `IO_UMBILICUS_TOO_FEW`; two equal z (after scaling; -0.0 equals 0.0)
`IO_UMBILICUS_DUPLICATE_Z` (`z`, `sources`, `source_kind`).

`coordinate_scale` (finite > 0, else `IO_UMBILICUS_SCALE_INVALID`) multiplies z, y and x only when given and is
recorded; apok-eval never applies spiral-scroll's default or a metadata `coordinate_scale` (the latter, written by
vc_gen_umbilicus with another meaning, is the finding `metadata-coordinate-scale-not-applied`). Control points
are sorted by z (stable) and held as float64; the fitter casts to float32 first (`umbilicus.py:31-32`), so its
values can differ from apok-eval's by float32 rounding.

**Metadata** (`metadata`, JSON variants, object root only; null means none; as VC3D's reader and resolver check
it, `Umbilicus.cpp:331-421`, `ScrollUmbilicus.cpp:204-208,248-252`): an object; `voxelsize_um` a positive finite
number; `volume` a non-empty string; `volume_width`, `volume_height`, `volume_slices` positive ints (JSON `64.0` is
not an int), all three or none (a partial triplet is refused naming the missing keys): `IO_UMBILICUS_METADATA_INVALID`
(`field`). With a declared `frame`: with s = coordinate_scale (1 when absent) and N the frame's extent on the
dimension's axis (slices -> z, height -> y, width -> x), the stamped d must be what VC3D's rescale test
(`uniformRescaleFactor`, `ScrollUmbilicus.cpp:262-315`; the factor multiplies the points,
`LineAnnotationController.cpp:13400`) accepts for that factor: for s >= 1 the stamped grid is the frame downsampled
by s, so d is floor(N / s) or ceil(N / s); for s < 1 the frame is the stamped grid downsampled by m = 1 / s, so N is
floor(d / m) or ceil(d / m) (s = 1 means d = N). Dividing by m, not multiplying by s, keeps an inexact s such as
1/3 exact (1 / float(1/3) is 3.0). VC3D searches the factors 1..64 and 1/2..1/64 and takes a unique one; apok-eval checks
the declared one. `voxelsize_um` must equal
frame.voxel_size_um * s within relative 1e-6; otherwise `IO_UMBILICUS_FRAME_MISMATCH` (`field`). The scaled
control points must lie inside the frame (`IO_FRAME_POINTS_OUTSIDE`). `volume` is a provenance string
(vc_gen_umbilicus: "Volume store the grids were built from"): when it names a 14-digit volume id (or begins with
one), that id must be the frame's (`IO_FRAME_VOLUME_MISMATCH`, `apok_eval.io.frame.stated_identity`); any other value
other than the frame's volume id is the finding `metadata-volume-differs` (not comparable as an identity). Finding `extension-disagrees-with-variant`: VC3D picks JSON
for `.json` and text otherwise (`Umbilicus.cpp:206`).

**Interpolation.** `Umbilicus.yx_at(z, out_of_range=...)`: `out_of_range` is required, `"extrapolate"` (the
fitter's `interp1d(..., fill_value='extrapolate')`, `umbilicus.py:33`: the first/last segment continued) or
`"clamp"` (the first/last control point held, returned exactly at and beyond the ends, as VC3D's
`interpolate_center` does, `Umbilicus.cpp:452-460`); else `IO_UMBILICUS_POLICY_UNKNOWN`. `"clamp"` models only
that end-value hold: VC3D's consumers never evaluate at z itself but read a dense table built at the integer slices
0 .. volume_slices - 1 (`Umbilicus.cpp:428-435`) at index lround(z) clamped to that range
(`Umbilicus.cpp:496-503`, `vc_project_tifxyz.cpp:823`), so at a fractional z, and beyond the volume's slice range,
VC3D's centre differs from apok-eval's linear one. The 1e-5 snapping to a control point is not modelled either. z is a
finite number or 1-D array of them (`IO_UMBILICUS_QUERY_INVALID`, `problem` = `not numeric`, `not a scalar or 1-D`
or `not finite`, with `dtype` and `shape`). Inside the span: y = slope * (z - z_lo) + y_lo on the bracketing
segment (the interp1d form). The result carries `outside` per query (strictly below the first or above the last
control z) and `outside_count`.

**Writer.** `encode_umbilicus(control_zyx, metadata=None)` / `write_umbilicus(...)`: fitter form,
`{"control_points": [{"x", "y", "z"}...] ascending z, "metadata"?}`, sorted keys, indent 4, `"\n"`;
`IO_UMBILICUS_WRITE_INVALID` for a non-finite or non-(N, 3) numeric array (`field=control_zyx`) or a document
`apok_eval.jsonstrict.dumps` refuses (`field=metadata`, `problem`: by then the control points are finite floats, so only
the metadata can be the cause). The whole document is checked, not the metadata alone: it nests the metadata one
level deeper, so metadata exactly 256 levels deep is refused (`would not read back as strict JSON: arrays and
objects nested deeper than 256 levels`). The read checks (too few, duplicate z, metadata) apply. Per-point extra keys
(estimates, inside, score) are not written. `inspect_umbilicus(path, variant=, coordinate_scale=)`.

## 6. Refusal codes

| code | trigger |
|---|---|
| `IO_TIFXYZ_NOT_DIRECTORY` | read path is not a directory; write target exists and is not a directory |
| `IO_TIFXYZ_FILE_MISSING` | x.tif, y.tif, z.tif or meta.json absent |
| `IO_TIFXYZ_FILE_KIND` | a core file or mask.tif exists but is a directory, FIFO, device or dangling link (`kind`) |
| `IO_TIFXYZ_META_INVALID` | meta.json not strict JSON (UTF-8, BOM, syntax, duplicate key, overflow, integer past int()'s digit limit, nesting past 256 levels, a lone surrogate) or not an object |
| `IO_TIFXYZ_META_FIELD` | format != "tifxyz", type != "seg", uuid not a non-empty string, scale not two positive finite float32-safe numbers |
| `IO_TIFXYZ_TIFF_INVALID` | bytes are not a TIFF tifffile can open; no first IFD (offset 0 or past the end); next-IFD pointer cut off by the end of file; a zero image length/width, rows per strip, or tile length/width |
| `IO_TIFXYZ_TIFF_PAGES` | x/y/z.tif has a non-zero next-IFD pointer |
| `IO_TIFXYZ_TIFF_SAMPLE` | x/y/z not SamplesPerPixel 1 / 32 bits / IEEE float; any TIFF with ImageDepth != 1 |
| `IO_TIFXYZ_TIFF_COMPRESSION` | compression not in {1, 5, 8, 32946} or predictor not in {1, 2, 3} |
| `IO_TIFXYZ_CODEC_MISSING` | LZW or the floating-point predictor without imagecodecs installed (or a predictor tifffile has no fallback for) |
| `IO_TIFXYZ_TIFF_SEGMENTS` | the offsets/byte-counts lists are not exactly as long as the geometry needs; a strip/tile is absent, runs past end of file, or (uncompressed) is short |
| `IO_TIFXYZ_TIFF_DECODE` | tifffile raised while decoding |
| `IO_TIFXYZ_SHAPE_MISMATCH` | x/y/z heights or widths differ |
| `IO_TIFXYZ_POLICY_UNKNOWN` | validity policy not in POLICIES |
| `IO_TIFXYZ_MASK_UNSUPPORTED` | vc3d: mask format without a C++ rule (e.g. int8); vesuvius or apok-strict: multi-page mask |
| `IO_TIFXYZ_MASK_SHAPE` | spiral: channel 0 not the grid shape; apok-strict: mask not the grid shape |
| `IO_TIFXYZ_MASK_VALUES` | apok-strict: a mask value outside {0, 255} (NaN included) |
| `IO_TIFXYZ_SCALE_INVALID` | canvas extent asked for with a scale that is not a positive finite float32 |
| `IO_FRAME_VOLUME_MISMATCH` (owned by `frame.py`) | a tifxyz `meta.json` `volume`/`target_volume` naming another 14-digit volume id, or a `base_shape_zyx` other than the frame's shape; umbilicus metadata `volume` naming another volume id (`what`, `field`, `stated`, `declared`) |
| `IO_TIFXYZ_EXTENT_INVALID` | canvas extent asked for with a stored extent that is not a non-bool integer in [0, 2**53] |
| `IO_TIFXYZ_WRITE_EXISTS` | output directory exists without overwrite |
| `IO_TIFXYZ_WRITE_SYMLINK` | with overwrite, the output directory is a symbolic link |
| `IO_TIFXYZ_WRITE_INVALID` | plane not 2-D float32, shapes differ, empty plane, meta refused by apok_eval.jsonstrict.dumps |
| `IO_TIFXYZ_WRITE_TOO_LARGE` | a plane exceeds 0xffff0000 pixel bytes (BigTIFF) |
| `IO_MESH_FORMAT_UNKNOWN` | extension not .obj/.ply |
| `IO_MESH_FILE_MISSING` | the path holds NUL, names nothing, or names no regular file (`problem`: `not a valid path`, `missing`, `not a regular file (<kind>)`) |
| `IO_MESH_OBJ_SYNTAX` | not UTF-8; a `#` inside a v/vt/vn/f statement; wrong number count; non-decimal token; vt arity change; malformed face corner |
| `IO_MESH_OBJ_UNSUPPORTED` | a statement other than v/vt/vn/f/o/g/s/usemtl/mtllib |
| `IO_MESH_FACE_NOT_TRIANGLE` | an OBJ or PLY face with other than 3 vertices |
| `IO_MESH_FACE_FORM_MIXED` | OBJ corners mix v, v/vt, v//vn, v/vt/vn |
| `IO_MESH_INDEX_OUT_OF_RANGE` | OBJ index 0, negative beyond the defined, beyond the total, or of more than 19 significant digits; PLY index < 0 or >= N |
| `IO_MESH_VALUE_INVALID` | OBJ number overflow; PLY non-finite coordinate; PLY ascii value not its type (an integer of more than 19 significant digits included) |
| `IO_MESH_PLY_HEADER` | no `ply`/`end_header`, missing or repeated format, non-ASCII or unknown header line |
| `IO_MESH_PLY_UNSUPPORTED` | format other than ascii 1.0 / binary_little_endian 1.0 |
| `IO_MESH_PLY_PROPERTY_TYPE` | unknown type; x/y/z not all float or all double; vertex_indices not list uchar int/uint |
| `IO_MESH_PLY_LAYOUT` | elements not [vertex, face]; vertex properties not distinct scalars with x,y,z; face not exactly vertex_indices |
| `IO_MESH_TRUNCATED` | data ends before the declared vertices or faces (however large the declared count) |
| `IO_MESH_TRAILING_DATA` | bytes or values after the last face |
| `IO_MESH_WRITE_INVALID` | malformed mesh arrays, or data the target format cannot carry |
| `IO_MESH_WRITE_EXISTS` | output exists without overwrite |
| `IO_PCL_FILE_MISSING` | the file is absent, not a regular file, or the path is not valid (`problem` names which and the kind found) |
| `IO_PCL_JSON_INVALID` | not strict JSON (an integer past int()'s digit limit, nesting past 256 levels, a lone surrogate, NaN/Infinity tokens included) |
| `IO_PCL_VERSION` | version key absent or not the string "1" |
| `IO_PCL_STRUCTURE` | document, collections, a collection, name, points or metadata of the wrong JSON type |
| `IO_PCL_ID_NONCANONICAL` | collection or point key not a canonical uint64 (more than 20 digits included) |
| `IO_PCL_POINT_INVALID` | point not an object; bad p, wind_a or creation_time |
| `IO_PCL_ROLE_UNKNOWN` | declared role not a fitter role |
| `IO_PCL_ABSOLUTE_WINDING` | absolute file (read or written) with a point whose wind_a is not finite and > 0 |
| `IO_PCL_WRITE_INVALID` | document refused by apok_eval.jsonstrict.dumps (tuples, NaN, non-string keys, an integer past str()'s limit, nesting past 256 levels; `problem` names it) |
| `IO_PCL_WRITE_EXISTS` | output exists without overwrite |
| `IO_UMBILICUS_FILE_MISSING` | the file is absent, not a regular file, or the path is not valid (`problem` names which and the kind found) |
| `IO_UMBILICUS_VARIANT_UNKNOWN` | variant not in VARIANTS |
| `IO_UMBILICUS_JSON_INVALID` | not strict JSON (an integer past int()'s digit limit, nesting past 256 levels, a lone surrogate, NaN/Infinity tokens included) and not comma text |
| `IO_UMBILICUS_TEXT_UNDECLARED` | comma text read under a JSON variant |
| `IO_UMBILICUS_TEXT_SYNTAX` | text not UTF-8 or a line not three numbers of the variant's kind |
| `IO_UMBILICUS_STRUCTURE` | root or entry not of the variant's shape |
| `IO_UMBILICUS_AMBIGUOUS` | both points and control_points |
| `IO_UMBILICUS_VALUE_INVALID` | coordinate not a finite non-bool number, or one coordinate_scale makes overflow (`scale` named) |
| `IO_UMBILICUS_TOO_FEW` | fewer than two control points |
| `IO_UMBILICUS_DUPLICATE_Z` | two control points with equal z |
| `IO_UMBILICUS_SCALE_INVALID` | coordinate_scale not finite and > 0 |
| `IO_UMBILICUS_METADATA_INVALID` | metadata not an object; bad voxelsize_um, volume or dimension; partial triplet |
| `IO_UMBILICUS_FRAME_MISMATCH` | stamped dimensions are not what VC3D's rescale test accepts at the scale, or the voxel size does not match |
| `IO_UMBILICUS_POLICY_UNKNOWN` | out_of_range not "extrapolate"/"clamp" |
| `IO_UMBILICUS_QUERY_INVALID` | query z not finite numbers in a scalar or 1-D array (`problem` names which) |
| `IO_UMBILICUS_WRITE_INVALID` | control_zyx not a finite (N, 3) numeric array, or metadata refused by apok_eval.jsonstrict.dumps |
| `IO_UMBILICUS_WRITE_EXISTS` | output exists without overwrite |

Every code is triggered by a test asserting the code and the named cause. Guards that are comparisons or
recomputations rather than a single raise (policy rules, census, thresholds, rounding, index bases, id canon,
role derivation, interpolation ends) have tests of their own that fail when the comparison is changed.

## 7. Summaries (`inspect_*`) and other public functions
`inspect_tifxyz(path, policies=POLICIES)`, `inspect_mesh(path)`, `inspect_point_collections(path, role=)`,
`inspect_umbilicus(path, variant="fitter", coordinate_scale=None)`; each returns a canonical-JSON dict and raises
`Refusal` for malformed input (`OSError` only for I/O failures, section 1). `apok-eval inspect` prints these same
summaries (its tests compare them), but for a mesh, point collections or an umbilicus it calls the reader itself
and summarises the object it read (`read_mesh(...).as_dict()`, which is all `inspect_mesh` does), so the file is
read once and a `--frame` check judges the very object summarised; a tifxyz without `--frame` goes through
`inspect_tifxyz`. Also public for callers:
`full_resolution_extent`, `check_within_frame`, `check_mesh_within`, `check_point_collections_within`, the
writers (`write_tifxyz`, `write_obj`, `write_ply`, `write_point_collections(..., role=)`, `write_umbilicus`).

## 8. Declared assumptions and deviations from villa readers
- All checks are stricter than at least one villa reader; each is listed with its refusal above.
- tifxyz meta.json may carry NaN tokens (finding); the writer will not reproduce them.
- The vc3d policy models `load_quad_from_tifxyz` (z <= 0 then mask), not the later C++ predicates
  (`isPointValid` tests x != -1, `validMask` NaN or all -1); those disagree among themselves
  (`QuadSurface.hpp:429-443`, `QuadSurface.cpp:1459-1463, 276-280`).
- Multi-page masks are read through their first IFD for vc3d and spiral (libtiff and Pillow read the first page);
  vesuvius' handling is not modelled and refused.
- The tifxyz writer writes only x/y/z/meta.json; the surface's mask and channels must be copied by the caller.
- PCL absolute checks apply to every point, not only attached ones, when reading and when writing.
- A strip/tile list must have exactly the geometry's length; a missing StripByteCounts tag (which tifffile estimates
  for uncompressed data) is refused rather than estimated.
- Umbilicus control points are float64 (the fitter's are float32).
- Whole files are read into memory (a 3040 x 20000 band is ~243 MB).

## 9. Open questions
- Whether C++-written TIFFs are ever big-endian (libtiff host order) is unverified; apok-eval reads both.
- The compression of cv::imwritemulti masks is not in villa source; an LZW one needs imagecodecs here.
- Which frame a published dataset's PCLs and umbilicus use is a property of that dataset; the readers only check
  a frame the caller declares.
- Whether `QuadSurface::resample` should divide or multiply scale is contested in villa; apok-eval only reads scale.
- Whether VC3D-written umbilicus metadata dimensions use ceil or floor for downsampled grids is not pinned by the
  writer; the rescale test VC3D reads them with accepts both, and so does apok-eval's (section 5).
- The integer-digit limit of `apok_eval.jsonstrict.loads` follows the interpreter (`sys.get_int_max_str_digits()`, set by
  `PYTHONINTMAXSTRDIGITS`); under a non-default limit the threshold between a field-level refusal and
  `*_JSON_INVALID` moves. Both are refusals.

## 10. Inspect (`apok-eval inspect`)

`apok-eval inspect PATH [--frame FRAME.json | --no-frame] [--policy NAME] [--json]`, also spelled
`apok-eval io inspect` (`src/apok_eval/io/cli.py`).
Detection, the frame file, the record, the `IO_INSPECT_*` codes and the OME-Zarr reading are specified in
[docs/io-zarr.md](io-zarr.md), "Inspect"; this section gives what the command does with the four formats here. A tifxyz
surface is a directory with an entry `x.tif`, `y.tif` or `z.tif`; a mesh is an `.obj` or `.ply` file (any
case); a `.json` file holds point collections when its top-level object has `vc_pointcollections_json_version`
or `collections`, an umbilicus when its top level is an array or an object with `control_points` or `points`;
markers of both kinds are `IO_INSPECT_AMBIGUOUS`. Every choice inspect makes is in the record's `read_as`.

| kind | read with | `read_as` | `--frame` (optional) |
|---|---|---|---|
| tifxyz | `inspect_tifxyz(PATH, policies)`: every policy in `POLICIES`, or only `--policy NAME`; a policy that refuses is listed under `refused_policies` | `{"policies": [...]}` | needs `--policy` (`IO_INSPECT_OPTION_INVALID`, `option: "--policy"`: the policy decides which vertices are points; none is assumed). The surface is read once: `read_tifxyz`, `validity_report(surface, [policy])` (a refusing policy refuses the inspect: no frame check can be made under it), `check_within_frame(surface, frame, policy)`; `count` is the vertices checked |
| mesh | `read_mesh(PATH)`; the summary is `inspect_mesh`'s | `{}` | `check_mesh_within`; `count` is the vertices |
| point collections | `read_point_collections(PATH, role=None)`: role-less, so the role is derived from the file name as the fitter derives it for an undeclared role, and the summary (`inspect_point_collections(PATH, role=None)`'s) shows `declared_role: null` and the derived `role` | `{"role": null}` | `check_point_collections_within`; `count` is the points |
| umbilicus | `read_umbilicus(PATH, variant=...)`: `fitter` when the object's only marker is `control_points`, else `vc3d-json` (a top-level array or `points` is read only by VC3D); no `coordinate_scale`; the summary is `inspect_umbilicus`'s. Both `points` and `control_points` reach the reader, which refuses them (`IO_UMBILICUS_AMBIGUOUS`) | `{"variant": ..., "coordinate_scale": null}` | passed to `read_umbilicus(..., frame=)`, which checks the control points and the metadata against it (`IO_FRAME_POINTS_OUTSIDE`, `IO_UMBILICUS_FRAME_MISMATCH`); `count` is the control points |

`--policy` with any other kind is `IO_INSPECT_OPTION_INVALID`, and so is a name not in `POLICIES` given to
`inspect_path` (on the command line, `choices` make it a usage error), rather than a summary listing it under
`refused_policies`. Without `--frame` (or with `--no-frame`) the record's `frame` is `NOT CHECKED`, with the
reason. A marker only picks the reader: a directory holding `x.tif` but no `meta.json` is a tifxyz surface
that `read_tifxyz` refuses (`IO_TIFXYZ_FILE_MISSING`).

Not detected: umbilicus text (`vc3d-text`, `thaumato-text`, any extension but `.json`) and a tifxyz
`meta.json` given alone (no marker: give the directory); both are `IO_INSPECT_UNKNOWN`. A `control_points`
file whose entries are `[z, y, x]` arrays is read under `fitter`, which refuses it, although VC3D would read
it; inspect has no `--variant` (an open question in [docs/io-zarr.md](io-zarr.md)).
