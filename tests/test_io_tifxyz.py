"""tifxyz: bit-exact round trip, canonical bytes, known-answer axes, every refusal, and the validity
policies disagreeing exactly where the villa readers do (docs/io-surfaces.md)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import struct
import zlib

import numpy as np
import pytest
import tifffile

from apok_eval.canonical import canonical_dumps
from apok_eval.io import tifxyz as T
from apok_eval.io.frame import VolumeFrame
from apok_eval.refusal import Refusal

META = {"format": "tifxyz", "type": "seg", "uuid": "seg01", "scale": [0.05, 0.05]}
H, W = 4, 5


def f32(bits: int) -> np.float32:
    return np.array([bits], dtype="<u4").view("<f4")[0]


def u32(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a).view(np.uint32)


def fixture_planes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """4 x 5 grid; every special vertex the policies disagree about is planted at a known cell."""
    rows, cols = np.mgrid[0:H, 0:W]
    x = (10 + cols).astype(np.float32)
    y = (20 + rows).astype(np.float32)
    z = (30 + rows * W + cols).astype(np.float32)
    z[0, 0] = f32(0x80000000)  # -0.0
    z[0, 1] = f32(0x7FC00001)  # quiet NaN, non-default payload
    x[0, 2] = f32(0x7F800001)  # signalling NaN
    z[1, 0] = -3.5  # non-sentinel z <= 0
    x[1, 1], y[1, 1], z[1, 1] = 5, -1, -1  # partial sentinel (5, -1, -1)
    x[1, 2], y[1, 2], z[1, 2] = -1, -1, -1  # full sentinel
    x[2, 0], y[2, 0], z[2, 0] = -1, 7, 9  # partial sentinel with z > 0
    z[2, 1] = 0.0
    return x, y, z


def write_fixture(path, mask=None, meta=None, **mask_kwargs) -> str:
    x, y, z = fixture_planes()
    T.write_tifxyz(path, x=x, y=y, z=z, meta=dict(meta or META))
    if mask is not None:
        tifffile.imwrite(os.path.join(path, "mask.tif"), mask, **mask_kwargs)
    return str(path)


def mask_255(dtype=np.uint8, on=255) -> np.ndarray:
    m = np.full((H, W), on, dtype=dtype)
    m[3, 0] = m[3, 1] = 0
    return m


def build_tiff(width, height, strips, *, bits=32, sample_format=3, spp=1, compression=1, rows_per_strip=None,
               extra=(), offsets=None, counts=None, next_ifd=0, tail=b"", byteorder="<", drop=()) -> bytes:
    """A minimal classic TIFF assembled by hand (independent of the writer under test)."""
    rows_per_strip = height if rows_per_strip is None else rows_per_strip
    entries = [(256, 4, [width]), (257, 4, [height]), (258, 3, [bits]), (259, 3, [compression]),
               (262, 3, [1]), (273, 4, None), (277, 3, [spp]), (278, 4, [rows_per_strip]), (279, 4, None),
               (339, 3, [sample_format]), *extra]
    entries = sorted((e for e in entries if e[0] not in drop), key=lambda e: e[0])
    ifd_size = 2 + 12 * len(entries) + 4
    n = len(strips)
    arrays_at = 8 + ifd_size
    data_at = arrays_at + (8 * n if n > 1 else 0)
    real_offsets, pos = [], data_at
    for s in strips:
        real_offsets.append(pos)
        pos += len(s)
    values = {273: offsets if offsets is not None else real_offsets,
              279: counts if counts is not None else [len(s) for s in strips]}
    bo = byteorder
    out = bytearray((b"II" if bo == "<" else b"MM") + struct.pack(bo + "HI", 42, 8))
    out += struct.pack(bo + "H", len(entries))
    for tag, kind, vals in entries:
        vals = values.get(tag, vals)
        if tag in (273, 279) and len(vals) > 1:
            where = arrays_at + (0 if tag == 273 else 4 * n)
            out += struct.pack(bo + "HHII", tag, 4, len(vals), where)
        elif kind == 3:
            out += struct.pack(bo + "HHIHH", tag, 3, 1, vals[0], 0)
        else:
            out += struct.pack(bo + "HHII", tag, 4, 1, vals[0])
    out += struct.pack(bo + "I", next_ifd)
    if n > 1:
        out += struct.pack(bo + f"{n}I", *values[273]) + struct.pack(bo + f"{n}I", *values[279])
    for s in strips:
        out += s
    return bytes(out) + tail


def build_ifd_last(width, height, strip: bytes) -> bytes:
    """libtiff's layout: header, the pixel strip, then the IFD (a download cut short loses the IFD)."""
    entries = [(256, 4, width), (257, 4, height), (258, 3, 32), (259, 3, 1), (262, 3, 1), (273, 4, 8), (277, 3, 1),
               (278, 4, height), (279, 4, len(strip)), (284, 3, 1), (339, 3, 3)]
    ifd = struct.pack("<H", len(entries))
    for tag, kind, value in entries:
        ifd += struct.pack("<HHI", tag, kind, 1)
        ifd += struct.pack("<HH", value, 0) if kind == 3 else struct.pack("<I", value)
    return b"II" + struct.pack("<HI", 42, 8 + len(strip)) + strip + ifd + struct.pack("<I", 0)


def build_tiled(width, height, tiles, *, tile_hw=(16, 16), declared=None, compression=8) -> bytes:
    """A tiled float32 TIFF by hand; ``declared`` limits how many TileOffsets/TileByteCounts entries it lists."""
    n = len(tiles) if declared is None else declared
    entries = [(256, 4, width), (257, 4, height), (258, 3, 32), (259, 3, compression), (262, 3, 1), (277, 3, 1),
               (284, 3, 1), (322, 4, tile_hw[1]), (323, 4, tile_hw[0]), (324, 4, None), (325, 4, None), (339, 3, 3)]
    arrays_at = 8 + 2 + 12 * len(entries) + 4
    offsets, pos = [], arrays_at + (8 * n if n > 1 else 0)
    for t in tiles:
        offsets.append(pos)
        pos += len(t)
    lists = {324: offsets[:n], 325: [len(t) for t in tiles][:n]}
    out = bytearray(b"II" + struct.pack("<HI", 42, 8) + struct.pack("<H", len(entries)))
    for tag, kind, value in entries:
        if tag in lists and n > 1:
            out += struct.pack("<HHII", tag, 4, n, arrays_at + (0 if tag == 324 else 4 * n))
        elif kind == 3:
            out += struct.pack("<HHIHH", tag, 3, 1, value, 0)
        else:
            out += struct.pack("<HHII", tag, 4, 1, lists[tag][0] if tag in lists else value)
    out += struct.pack("<I", 0)
    if n > 1:
        out += struct.pack(f"<{n}I", *lists[324]) + struct.pack(f"<{n}I", *lists[325])
    return bytes(out) + b"".join(tiles)


def float_strip(values) -> bytes:
    return np.asarray(values, dtype="<f4").tobytes()


def replace_file(directory, name, data: bytes) -> None:
    with open(os.path.join(directory, name), "wb") as fh:
        fh.write(data)


def refusal(fn, *args, **kwargs) -> Refusal:
    with pytest.raises(Refusal) as exc:
        fn(*args, **kwargs)
    return exc.value


# ---------------------------------------------------------------------------------------------- round trip


def test_round_trip_is_bit_exact_and_meta_equal(tmp_path):
    src = write_fixture(tmp_path / "a", meta={**META, "bbox": [[9, 9, 9], [0, 0, 0]], "extra": {"k": [1, 2.5]}})
    first = T.read_tifxyz(src)
    T.write_tifxyz(tmp_path / "b", x=first.x, y=first.y, z=first.z, meta=first.meta)
    second = T.read_tifxyz(tmp_path / "b")
    x, y, z = fixture_planes()
    for plane, original, a, b in (("x", x, first.x, second.x), ("y", y, first.y, second.y),
                                  ("z", z, first.z, second.z)):
        assert np.array_equal(u32(a), u32(original)), plane
        assert np.array_equal(u32(a), u32(b)), plane
    assert u32(second.z)[0, 0] == 0x80000000 and u32(second.z)[0, 1] == 0x7FC00001
    assert u32(second.x)[0, 2] == 0x7F800001
    assert first.meta == second.meta == {**META, "bbox": [[9, 9, 9], [0, 0, 0]], "extra": {"k": [1, 2.5]}}
    assert isinstance(second.meta["extra"]["k"][0], int) and isinstance(second.meta["extra"]["k"][1], float)


def test_writing_twice_gives_identical_bytes(tmp_path):
    a, b = write_fixture(tmp_path / "a"), write_fixture(tmp_path / "b")
    for name in ("x.tif", "y.tif", "z.tif", "meta.json"):
        with open(os.path.join(a, name), "rb") as fa, open(os.path.join(b, name), "rb") as fb:
            assert fa.read() == fb.read(), name


def test_writer_returns_digests_of_the_files_it_wrote(tmp_path):
    x, y, z = fixture_planes()
    digests = T.write_tifxyz(tmp_path / "s", x=x, y=y, z=z, meta=META)
    assert sorted(digests) == ["meta.json", "x.tif", "y.tif", "z.tif"]
    for name, digest in digests.items():
        with open(tmp_path / "s" / name, "rb") as fh:
            assert digest == "sha256:" + hashlib.sha256(fh.read()).hexdigest()


def test_canonical_tiff_bytes_are_pinned():
    header = T.canonical_tiff_header(3, 4)
    assert len(header) == 148
    expected = bytearray(b"II*\x00\x08\x00\x00\x00\x0b\x00")
    for tag, kind, value in ((256, 4, 4), (257, 4, 3), (258, 3, 32), (259, 3, 1), (262, 3, 1), (273, 4, 148),
                             (277, 3, 1), (278, 4, 3), (279, 4, 48), (284, 3, 1), (339, 3, 3)):
        expected += struct.pack("<HHI", tag, kind, 1)
        expected += struct.pack("<HH", value, 0) if kind == 3 else struct.pack("<I", value)
    expected += b"\x00" * 4 + b"\x00" * 2
    assert header == bytes(expected)
    plane = np.arange(12, dtype=">f4").reshape(3, 4)  # big-endian input still writes "II" bytes
    assert T.encode_float32_tiff(plane) == header + np.arange(12, dtype="<f4").tobytes()


def test_writer_output_decodes_identically_in_tifffile(tmp_path):
    src = write_fixture(tmp_path / "s")
    for name, plane in zip(("x.tif", "y.tif", "z.tif"), fixture_planes(), strict=True):
        with tifffile.TiffFile(os.path.join(src, name)) as tif:
            page = tif.pages[0]
            assert len(tif.pages) == 1 and page.is_memmappable and page.compression == 1
            assert page.samplesperpixel == 1 and page.bitspersample == 32 and page.sampleformat == 3
            assert not any(t in page.tags for t in (270, 305, 306, 282, 283, 296))
            assert np.array_equal(u32(page.asarray()), u32(plane))


@pytest.mark.parametrize("kwargs", [
    {},  # one uncompressed strip (lasagna / merge_concat_runs layout)
    {"tile": (16, 16)},  # vesuvius' tiling, uncompressed because imagecodecs (LZW) is not required
    {"tile": (16, 16), "compression": "zlib"},
    {"rowsperstrip": 1},
    {"rowsperstrip": 10},  # a short last strip (33 = 3 * 10 + 3)
    {"byteorder": ">"},
])
def test_reading_tifffile_written_planes_gives_the_same_bits(tmp_path, kwargs):
    rng = np.random.default_rng(7)
    planes = {n: rng.normal(size=(33, 21)).astype(np.float32) for n in ("x", "y", "z")}
    planes["z"][3, 4] = f32(0x7F800001)
    planes["z"][5, 6] = f32(0x80000000)
    d = tmp_path / "t"
    d.mkdir()
    for n, plane in planes.items():
        tifffile.imwrite(d / f"{n}.tif", plane, photometric="minisblack", **kwargs)
    (d / "meta.json").write_text(json.dumps(META))
    s = T.read_tifxyz(d)
    for n in ("x", "y", "z"):
        assert np.array_equal(u32(getattr(s, n)), u32(planes[n])), n
        assert getattr(s, n).dtype == np.dtype("<f4") and not getattr(s, n).flags.writeable
    assert s.tiffs["z.tif"]["byteorder"] == ("MM" if kwargs.get("byteorder") == ">" else "II")
    assert s.tiffs["z.tif"]["tiled"] == ("tile" in kwargs)


def test_overwrite_replaces_only_the_four_files(tmp_path):
    src = write_fixture(tmp_path / "s", mask=mask_255())
    (tmp_path / "s" / "notes.txt").write_text("keep me")
    with open(os.path.join(src, "mask.tif"), "rb") as fh:
        mask_bytes = fh.read()
    with np.errstate(invalid="ignore"):  # the planted signalling NaN
        x, y, z = (p + 1 for p in fixture_planes())
    T.write_tifxyz(src, x=x, y=y, z=z, meta={**META, "uuid": "s"}, overwrite=True)
    assert sorted(os.listdir(src)) == ["mask.tif", "meta.json", "notes.txt", "x.tif", "y.tif", "z.tif"]
    assert (tmp_path / "s" / "notes.txt").read_text() == "keep me"
    with open(os.path.join(src, "mask.tif"), "rb") as fh:
        assert fh.read() == mask_bytes
    s = T.read_tifxyz(src)
    assert np.array_equal(u32(s.y), u32(y)) and s.uuid == "s"


def test_meta_json_is_sorted_indent_four_with_trailing_newline(tmp_path):
    src = write_fixture(tmp_path / "s", meta={**META, "Zeta": "é", "alpha": [1.5, {"b": None, "a": True}]})
    text = (tmp_path / "s" / "meta.json").read_text(encoding="utf-8")
    assert text == json.dumps({**META, "Zeta": "é", "alpha": [1.5, {"b": None, "a": True}]}, indent=4,
                              sort_keys=True, ensure_ascii=False) + "\n"
    assert text.index('"Zeta"') < text.index('"alpha"') < text.index('"format"')
    assert T.read_tifxyz(src).meta["Zeta"] == "é"


def test_reading_does_not_touch_the_directory(tmp_path):
    src = write_fixture(tmp_path / "s", mask=mask_255())

    def snapshot():
        return {e.name: (e.stat().st_mtime_ns, e.stat().st_size) for e in os.scandir(src)}

    before = snapshot()
    s = T.read_tifxyz(src)
    T.validity_report(s)
    T.inspect_tifxyz(src)
    assert snapshot() == before


# ---------------------------------------------------------------------------------- known answer and frame


def test_planted_values_map_z_tif_to_axis_0(tmp_path):
    d = tmp_path / "k"
    d.mkdir()
    planes = {n: np.full((6, 7), 1.0, dtype=np.float32) for n in "xyz"}
    planes["x"][2, 5], planes["y"][2, 5], planes["z"][2, 5] = 31.0, 41.0, 51.0
    planes["x"][4, 1], planes["y"][4, 1], planes["z"][4, 1] = 3.0, 4.0, 99.0
    for n, plane in planes.items():
        tifffile.imwrite(d / f"{n}.tif", plane)  # an independent writer
    (d / "meta.json").write_text(json.dumps(META))
    s = T.read_tifxyz(d)
    assert s.shape == (6, 7)
    assert list(s.zyx[2, 5]) == [51.0, 41.0, 31.0]
    assert list(s.zyx[4, 1]) == [99.0, 4.0, 3.0]
    # Axis 0 of the frame is z: a 100-slice, 50-row, 40-column volume holds z=99 but not x=45.
    frame = VolumeFrame(volume_id="20990101000004", shape_zyx=(100, 50, 40), voxel_size_um=2.4, source="test")
    assert T.check_within_frame(s, frame, "apok-strict") == 42
    narrow = VolumeFrame(volume_id="20990101000004", shape_zyx=(99, 50, 40), voxel_size_um=2.4, source="test")
    exc = refusal(T.check_within_frame, s, narrow, "apok-strict")
    assert exc.code == "IO_FRAME_POINTS_OUTSIDE" and exc.details["example_zyx"] == [99.0, 4.0, 3.0]


def test_frame_check_counts_only_valid_vertices_of_the_named_policy(tmp_path):
    s = T.read_tifxyz(write_fixture(tmp_path / "s"))
    frame = VolumeFrame(volume_id="20990101000004", shape_zyx=(64, 32, 32), voxel_size_um=2.4, source="test")
    # apok-strict keeps the partial sentinel (-1, 7, 9) (finite, z > 0); the frame contract then rejects x = -1.
    exc = refusal(T.check_within_frame, s, frame, "apok-strict")
    assert exc.code == "IO_FRAME_POINTS_OUTSIDE"
    assert (exc.details["count"], exc.details["total"], exc.details["example_zyx"]) == (1, 13, [9.0, 7.0, -1.0])
    # lasagna and vc3d count NaN-bearing vertices valid, and the contract refuses non-finite points.
    for policy in ("lasagna", "vc3d"):
        exc = refusal(T.check_within_frame, s, frame, policy)
        assert exc.code == "IO_FRAME_POINTS_MALFORMED" and policy in exc.details["what"]


@pytest.mark.parametrize("stored,scale,expected", [
    (152, 0.05, 3040), (152, 0.05000000074505806, 3040), (7, 2.0, 4), (5, 2.0, 3), (1, 20.0, 1), (0, 0.5, 1),
    (3, 1, 3),
])
def test_canvas_extent_rule(stored, scale, expected):
    assert T.full_resolution_extent(stored, scale) == expected


def test_canvas_extent_is_not_truncation():
    assert int(152 / float(np.float32(0.05))) == 3039
    assert T.full_resolution_extent(152, 0.05) == 3040


def test_surface_exposes_scale_verbatim_and_as_float32(tmp_path):
    s = T.read_tifxyz(write_fixture(tmp_path / "s", meta={**META, "scale": [0.05, 0.1]}))
    assert s.scale == [0.05, 0.1]
    assert s.scale_float32 == (0.05000000074505806, 0.10000000149011612)
    assert s.full_resolution_shape == (T.full_resolution_extent(H, 0.1), T.full_resolution_extent(W, 0.05))
    assert s.full_resolution_shape == (40, 100)


@pytest.mark.parametrize("stored", [-5, "152", True, np.bool_(True), 152.7, float("nan"), None, 2 ** 53 + 1,
                                    pytest.param(10 ** 400, id="10**400")])
def test_canvas_extent_refuses_a_stored_extent_that_is_not_a_count(stored):
    exc = refusal(T.full_resolution_extent, stored, 0.05)
    assert exc.code == "IO_TIFXYZ_EXTENT_INVALID" and exc.details["value"] == repr(stored)


def test_canvas_extent_accepts_numpy_integers_and_the_largest_exact_count():
    assert T.full_resolution_extent(np.int64(152), 0.05) == 3040
    assert T.full_resolution_extent(2 ** 53, 1.0) == 2 ** 53


@pytest.mark.parametrize("scale", [0, -1.0, float("nan"), float("inf"), True, 1e-50, 1e39, "0.05", None])
def test_canvas_extent_refuses_a_scale_that_is_not_positive_float32(scale):
    exc = refusal(T.full_resolution_extent, 10, scale)
    assert exc.code == "IO_TIFXYZ_SCALE_INVALID" and exc.details["value"] == repr(scale)


# ------------------------------------------------------------------------------------------- findings


def test_findings_name_uuid_mismatch_step_like_scale_and_vesuvius_default(tmp_path):
    s = T.read_tifxyz(write_fixture(tmp_path / "w001", meta={**META, "uuid": "other", "scale": [20, 20]}))
    kinds = {f["finding"]: f for f in s.findings}
    assert kinds["uuid-differs-from-directory"]["uuid"] == "other"
    assert kinds["uuid-differs-from-directory"]["directory"] == "w001"
    assert kinds["scale-above-one"]["scale"] == [20.0, 20.0]
    assert "scale-vesuvius-default" in kinds
    assert s.uuid == "other" and s.dir_name == "w001"


def test_a_single_large_scale_component_is_flagged_but_not_the_default(tmp_path):
    s = T.read_tifxyz(write_fixture(tmp_path / "seg01", meta={**META, "scale": [0.05, 2]}))
    assert [f["finding"] for f in s.findings] == ["scale-above-one"]


def test_clean_surface_has_no_findings(tmp_path):
    assert T.read_tifxyz(write_fixture(tmp_path / "seg01")).findings == ()


def test_nonstandard_json_tokens_in_meta_are_kept_and_flagged(tmp_path):
    src = write_fixture(tmp_path / "seg01")
    (tmp_path / "seg01" / "meta.json").write_text(
        '{"format": "tifxyz", "type": "seg", "uuid": "seg01", "scale": [0.05, 0.05], '
        '"bbox": [[NaN, 1, 2], [Infinity, -Infinity, 3]]}')
    s = T.read_tifxyz(src)
    finding = [f for f in s.findings if f["finding"] == "meta-nonstandard-json"]
    assert finding and finding[0]["tokens"] == ["-Infinity", "Infinity", "NaN"]
    assert np.isnan(s.meta["bbox"][0][0])
    # ...and the writer will not emit them, because nlohmann::json cannot parse them.
    exc = refusal(T.write_tifxyz, tmp_path / "out", x=s.x, y=s.y, z=s.z, meta=s.meta)
    assert exc.code == "IO_TIFXYZ_WRITE_INVALID" and exc.details["field"] == "meta"


def test_inventory_records_every_entry_and_symlinked_core_files(tmp_path):
    src = write_fixture(tmp_path / "seg01", mask=mask_255())
    (tmp_path / "seg01" / "winding.tif").write_bytes(b"abc")
    (tmp_path / "seg01" / "overlapping").mkdir()
    os.replace(tmp_path / "seg01" / "x.tif", tmp_path / "x_elsewhere.tif")
    os.symlink(tmp_path / "x_elsewhere.tif", tmp_path / "seg01" / "x.tif")
    os.symlink("/nonexistent/target", tmp_path / "seg01" / "dangling")
    s = T.read_tifxyz(src)
    files = {f["name"]: f for f in s.files}
    assert sorted(files) == ["dangling", "mask.tif", "meta.json", "overlapping", "winding.tif", "x.tif", "y.tif",
                             "z.tif"]
    assert files["winding.tif"]["digest"] == "sha256:" + hashlib.sha256(b"abc").hexdigest()
    assert files["overlapping"] == {"name": "overlapping", "kind": "directory"}
    assert files["dangling"] == {"name": "dangling", "kind": "symlink", "target": "/nonexistent/target"}
    assert files["x.tif"]["kind"] == "symlink" and files["x.tif"]["digest"] == s.tiffs["x.tif"]["digest"]
    followed = [f for f in s.findings if f["finding"] == "symlink-followed"]
    assert followed == [{"finding": "symlink-followed", "file": "x.tif",
                         "target": os.path.realpath(tmp_path / "x_elsewhere.tif")}]
    assert s.as_dict()["mask_present"] is True


# -------------------------------------------------------------------------------------- reader refusals


def test_not_a_directory(tmp_path):
    (tmp_path / "f").write_text("x")
    exc = refusal(T.read_tifxyz, tmp_path / "f")
    assert exc.code == "IO_TIFXYZ_NOT_DIRECTORY" and exc.details["path"] == str(tmp_path / "f")


@pytest.mark.parametrize("name", ["meta.json", "x.tif", "y.tif", "z.tif"])
def test_missing_core_file(tmp_path, name):
    src = write_fixture(tmp_path / "s")
    os.remove(os.path.join(src, name))
    exc = refusal(T.read_tifxyz, src)
    assert exc.code == "IO_TIFXYZ_FILE_MISSING" and exc.details["file"] == name


def test_core_file_that_is_a_directory_or_dangling_link(tmp_path):
    src = write_fixture(tmp_path / "s")
    os.remove(os.path.join(src, "y.tif"))
    os.mkdir(os.path.join(src, "y.tif"))
    exc = refusal(T.read_tifxyz, src)
    assert exc.code == "IO_TIFXYZ_FILE_KIND" and exc.details["file"] == "y.tif"
    src2 = write_fixture(tmp_path / "t")
    os.symlink(tmp_path / "missing.tif", os.path.join(src2, "mask.tif"))
    exc = refusal(T.read_tifxyz, src2)
    assert exc.code == "IO_TIFXYZ_FILE_KIND" and exc.details["file"] == "mask.tif"


@pytest.mark.parametrize("text,why", [
    (b"{not json", "invalid JSON"),
    (b'{"format": "tifxyz", "format": "tifxyz"}', "duplicate key 'format'"),
    (b'\xef\xbb\xbf{"format": "tifxyz"}', "byte-order mark"),
    (b'{"format": "tifxyz", "scale": [1e400, 1]}', "overflows a double"),
    (b'{"uuid": "\xff"}', "not UTF-8"),
    (b'[1, 2]', "must hold a JSON object"),
])
def test_malformed_meta_json(tmp_path, text, why):
    src = write_fixture(tmp_path / "s")
    replace_file(src, "meta.json", text)
    exc = refusal(T.read_tifxyz, src)
    assert exc.code == "IO_TIFXYZ_META_INVALID" and exc.details["file"] == "meta.json"
    assert why in exc.reason


@pytest.mark.parametrize("field,value", [
    ("format", None), ("format", "TIFXYZ"), ("type", None), ("type", "mesh"), ("uuid", None), ("uuid", ""),
    ("uuid", 7), ("scale", None), ("scale", [0.05]), ("scale", [0.05, 0.05, 0.05]), ("scale", (0.05, 0.05)),
    ("scale", ["0.05", 0.05]), ("scale", [True, 0.05]), ("scale", [0.05, 0]), ("scale", [-0.05, 0.05]),
    ("scale", [0.05, 1e-50]), ("scale", [1e39, 0.05]), ("scale", [10 ** 400, 0.05]),
    ("scale", [0.05, float("nan")]), ("scale", [float("inf"), 0.05]),
])
def test_meta_fields_are_checked_and_named(field, value):
    meta = dict(META)
    if value is None:
        del meta[field]
    else:
        meta[field] = value
    exc = refusal(T.check_meta, meta, "fixture")
    assert exc.code == "IO_TIFXYZ_META_FIELD" and exc.details["field"] == field


def test_scale_refusal_names_the_offending_component():
    exc = refusal(T.check_meta, {**META, "scale": [0.05, 1e-50]}, "fixture")
    assert "[1] = 1e-50" in exc.reason and "float32" in exc.reason


def test_meta_check_reads_the_file_on_disk(tmp_path):
    src = write_fixture(tmp_path / "s")
    replace_file(src, "meta.json", json.dumps({**META, "type": "mesh"}).encode())
    exc = refusal(T.read_tifxyz, src)
    assert exc.code == "IO_TIFXYZ_META_FIELD" and exc.details["field"] == "type"
    assert exc.details["path"] == os.path.join(src, "meta.json")


def test_not_a_tiff(tmp_path):
    src = write_fixture(tmp_path / "s")
    replace_file(src, "y.tif", b"GIF89a not a tiff at all")
    exc = refusal(T.read_tifxyz, src)
    assert exc.code == "IO_TIFXYZ_TIFF_INVALID" and exc.details["file"] == "y.tif"


def test_more_than_one_ifd(tmp_path):
    src = write_fixture(tmp_path / "s")
    tifffile.imwrite(os.path.join(src, "z.tif"), np.ones((2, H, W), dtype=np.float32))
    exc = refusal(T.read_tifxyz, src)
    assert exc.code == "IO_TIFXYZ_TIFF_PAGES" and exc.details["file"] == "z.tif"
    assert exc.details["next_ifd_offset"] > 0


def test_a_dangling_next_ifd_pointer_is_still_a_second_ifd(tmp_path):
    src = write_fixture(tmp_path / "s")
    replace_file(src, "x.tif", build_tiff(W, H, [float_strip(np.ones((H, W)))], next_ifd=10 ** 6))
    exc = refusal(T.read_tifxyz, src)
    assert exc.code == "IO_TIFXYZ_TIFF_PAGES" and exc.details["next_ifd_offset"] == 10 ** 6


@pytest.mark.parametrize("plane,kwargs,field", [
    (np.ones((H, W, 2), dtype=np.float32), {"photometric": "minisblack", "planarconfig": "contig"},
     "samples_per_pixel"),
    (np.ones((H, W), dtype=np.float64), {}, "bits_per_sample"),
    (np.ones((H, W), dtype=np.uint32), {}, "sample_format"),
    (np.ones((H, W), dtype=np.int32), {}, "sample_format"),
])
def test_coordinates_must_be_single_sample_float32(tmp_path, plane, kwargs, field):
    src = write_fixture(tmp_path / "s")
    tifffile.imwrite(os.path.join(src, "x.tif"), plane, **kwargs)
    exc = refusal(T.read_tifxyz, src)
    assert exc.code == "IO_TIFXYZ_TIFF_SAMPLE" and exc.details["field"] == field
    assert exc.details["file"] == "x.tif"


def test_volumetric_tiff_is_refused(tmp_path):
    src = write_fixture(tmp_path / "s")
    replace_file(src, "z.tif", build_tiff(W, H, [float_strip(np.ones((H, W)))], extra=[(32997, 4, [2])]))
    exc = refusal(T.read_tifxyz, src)
    assert exc.code == "IO_TIFXYZ_TIFF_SAMPLE" and exc.details["field"] == "image_depth"


def test_unsupported_compression_and_predictor(tmp_path):
    src = write_fixture(tmp_path / "s")
    tifffile.imwrite(os.path.join(src, "x.tif"), np.ones((H, W), dtype=np.float32), compression="lzma")
    exc = refusal(T.read_tifxyz, src)
    assert exc.code == "IO_TIFXYZ_TIFF_COMPRESSION" and exc.details["field"] == "compression"
    assert exc.details["value"] == 34925
    replace_file(src, "x.tif", build_tiff(W, H, [float_strip(np.ones((H, W)))], extra=[(317, 3, [4])]))
    exc = refusal(T.read_tifxyz, src)
    assert exc.code == "IO_TIFXYZ_TIFF_COMPRESSION" and exc.details["field"] == "predictor"
    assert exc.details["value"] == 4


class _NoCodecs:
    """Stands in for tifffile's codec tables when imagecodecs is absent."""

    def __getitem__(self, key):
        if key == 1:
            return lambda data, *a, **k: data
        raise KeyError(f"{key} requires the 'imagecodecs' package")


def test_lzw_without_imagecodecs_names_the_codec(tmp_path, monkeypatch):
    monkeypatch.setattr(tifffile.TIFF, "DECOMPRESSORS", _NoCodecs())
    src = write_fixture(tmp_path / "s")
    replace_file(src, "y.tif", build_tiff(W, H, [b"\x80" * 40], compression=5))
    exc = refusal(T.read_tifxyz, src)
    assert exc.code == "IO_TIFXYZ_CODEC_MISSING" and exc.details == {"file": "y.tif", "codec": "LZW"}


@pytest.mark.skipif(importlib.util.find_spec("imagecodecs") is not None, reason="imagecodecs is installed here")
def test_lzw_in_this_environment_is_refused_not_zero_filled(tmp_path):
    src = write_fixture(tmp_path / "s")
    replace_file(src, "y.tif", build_tiff(W, H, [b"\x80" * 40], compression=5))
    exc = refusal(T.read_tifxyz, src)
    assert exc.code == "IO_TIFXYZ_CODEC_MISSING" and exc.details["codec"] == "LZW"


def test_predictor_without_imagecodecs_names_the_predictor(tmp_path, monkeypatch):
    monkeypatch.setattr(tifffile.TIFF, "UNPREDICTORS", _NoCodecs())
    src = write_fixture(tmp_path / "s")
    replace_file(src, "z.tif", build_tiff(W, H, [float_strip(np.ones((H, W)))], extra=[(317, 3, [2])]))
    exc = refusal(T.read_tifxyz, src)
    assert exc.code == "IO_TIFXYZ_CODEC_MISSING"
    assert exc.details == {"file": "z.tif", "codec": "predictor horizontal"}


def test_sparse_segment_is_refused_not_zero_filled(tmp_path):
    src = write_fixture(tmp_path / "s")
    strips = [float_strip(np.ones((2, W))), float_strip(np.ones((2, W)))]
    good = build_tiff(W, H, strips, rows_per_strip=2)
    replace_file(src, "x.tif", good)
    assert T.read_tifxyz(src).x.shape == (H, W)  # the two-strip builder output itself is fine
    for offsets, counts in (([build_tiff(W, H, strips, rows_per_strip=2).find(strips[0]), 0], None),
                            (None, [40, 0])):
        replace_file(src, "x.tif", build_tiff(W, H, strips, rows_per_strip=2, offsets=offsets, counts=counts))
        exc = refusal(T.read_tifxyz, src)
        assert exc.code == "IO_TIFXYZ_TIFF_SEGMENTS" and exc.details == {"file": "x.tif", "segment": 1}
        assert "absent" in exc.reason


def test_truncated_file_is_refused(tmp_path):
    src = write_fixture(tmp_path / "s")
    with open(os.path.join(src, "z.tif"), "rb") as fh:
        data = fh.read()
    replace_file(src, "z.tif", data[:-4])
    exc = refusal(T.read_tifxyz, src)
    assert exc.code == "IO_TIFXYZ_TIFF_SEGMENTS" and exc.details == {"file": "z.tif", "segment": 0}
    assert "past the end" in exc.reason


def test_short_uncompressed_strip_is_refused(tmp_path):
    src = write_fixture(tmp_path / "s")
    replace_file(src, "z.tif", build_tiff(W, H, [float_strip(np.ones((H, W)))], counts=[H * W * 4 - 4]))
    exc = refusal(T.read_tifxyz, src)
    assert exc.code == "IO_TIFXYZ_TIFF_SEGMENTS" and exc.details == {"file": "z.tif", "segment": 0}
    assert f"{H * W * 4 - 4} bytes, {H * W * 4} needed" in exc.reason


def test_uncompressed_tiles_must_hold_whole_tiles(tmp_path):
    src = write_fixture(tmp_path / "s")
    tifffile.imwrite(os.path.join(src, "z.tif"), np.ones((20, 20), dtype=np.float32), tile=(16, 16))
    for name in ("x.tif", "y.tif"):
        tifffile.imwrite(os.path.join(src, name), np.ones((20, 20), dtype=np.float32))
    assert T.read_tifxyz(src).shape == (20, 20)
    with tifffile.TiffFile(os.path.join(src, "z.tif")) as tif:
        tag = tif.pages[0].tags[325]  # TileByteCounts
        fmt = {3: "<H", 4: "<I"}[int(tag.dtype)]
        count_at = tag.valueoffset + struct.calcsize(fmt) * 3
    with open(os.path.join(src, "z.tif"), "r+b") as fh:
        fh.seek(count_at)
        fh.write(struct.pack(fmt, 16 * 16 * 4 - 8))
    exc = refusal(T.read_tifxyz, src)
    assert exc.code == "IO_TIFXYZ_TIFF_SEGMENTS" and exc.details["segment"] == 3


def test_decoder_error_is_a_refusal(tmp_path):
    src = write_fixture(tmp_path / "s")
    path = os.path.join(src, "y.tif")
    tifffile.imwrite(path, np.ones((H, W), dtype=np.float32), compression="zlib")
    with tifffile.TiffFile(path) as tif:
        offset, count = tif.pages[0].dataoffsets[0], tif.pages[0].databytecounts[0]
    with open(path, "r+b") as fh:
        fh.seek(offset)
        fh.write(b"\x00" * count)
    exc = refusal(T.read_tifxyz, src)
    assert exc.code == "IO_TIFXYZ_TIFF_DECODE" and exc.details["file"] == "y.tif"


def test_band_shapes_must_agree(tmp_path):
    src = write_fixture(tmp_path / "s")
    tifffile.imwrite(os.path.join(src, "y.tif"), np.ones((H, W + 1), dtype=np.float32))
    exc = refusal(T.read_tifxyz, src)
    assert exc.code == "IO_TIFXYZ_SHAPE_MISMATCH"
    assert exc.details["shapes"] == {"x.tif": [H, W], "y.tif": [H, W + 1], "z.tif": [H, W]}


# ------------------------------------------------------------- damaged structure: a refusal, never a crash


def test_a_download_cut_short_of_an_ifd_written_after_the_pixels(tmp_path):
    src = write_fixture(tmp_path / "s")
    strip = float_strip(np.arange(H * W).reshape(H, W))
    whole = build_ifd_last(W, H, strip)
    replace_file(src, "x.tif", whole)
    assert np.array_equal(T.read_tifxyz(src).x, np.arange(H * W, dtype=np.float32).reshape(H, W))
    cut = 8 + len(strip) // 2  # inside the pixels: the IFD offset now points past the end
    replace_file(src, "x.tif", whole[:cut])
    exc = refusal(T.read_tifxyz, src)
    assert exc.code == "IO_TIFXYZ_TIFF_INVALID"
    assert exc.details == {"file": "x.tif", "field": "first_ifd_offset", "value": 8 + len(strip)}
    assert f"in the {cut}-byte file" in exc.reason
    replace_file(src, "x.tif", whole[: 8 + len(strip) + 20])  # inside the IFD entries: tifffile cannot open it
    exc = refusal(T.read_tifxyz, src)
    assert exc.code == "IO_TIFXYZ_TIFF_INVALID" and exc.details == {"file": "x.tif"}


def test_a_header_whose_first_ifd_offset_is_zero(tmp_path):
    src = write_fixture(tmp_path / "s")
    replace_file(src, "z.tif", b"II*\x00\x00\x00\x00\x00")
    exc = refusal(T.read_tifxyz, src)
    assert exc.code == "IO_TIFXYZ_TIFF_INVALID"
    assert exc.details == {"file": "z.tif", "field": "first_ifd_offset", "value": 0}


def test_a_mask_without_a_first_ifd_is_refused_by_the_policies_that_read_it(tmp_path):
    src = write_fixture(tmp_path / "s")
    replace_file(src, "mask.tif", b"II*\x00" + struct.pack("<I", 4096))
    s = T.read_tifxyz(src)  # recorded, not interpreted
    exc = refusal(T.validity, s, "vc3d")
    assert exc.code == "IO_TIFXYZ_TIFF_INVALID"
    assert exc.details == {"file": "mask.tif", "field": "first_ifd_offset", "value": 4096}
    summary = T.inspect_tifxyz(src)
    assert sorted(summary["refused_policies"]) == ["apok-strict", "spiral", "vc3d", "vesuvius"]
    assert {r["code"] for r in summary["refused_policies"].values()} == {"IO_TIFXYZ_TIFF_INVALID"}
    assert list(summary["validity"]["policies"]) == ["lasagna"]
    canonical_dumps(summary)


@pytest.mark.parametrize("cut", [1, 2, 4])
def test_a_next_ifd_pointer_cut_off_by_the_end_of_file(tmp_path, cut):
    whole = build_ifd_last(W, H, float_strip(np.ones((H, W))))
    T.decode_first_ifd("y.tif", whole)  # ending exactly after the pointer is fine
    exc = refusal(T.decode_first_ifd, "y.tif", whole[:-cut])
    assert exc.code == "IO_TIFXYZ_TIFF_INVALID"
    assert exc.details == {"file": "y.tif", "field": "next_ifd_offset", "value": None}
    assert f"the {len(whole) - cut}-byte file" in exc.reason


def test_a_short_strip_list_is_refused_not_zero_filled(tmp_path):
    # Deflate, RowsPerStrip 2: four rows need two strips, the file lists one; tifffile zero-filled rows 2-3.
    src = write_fixture(tmp_path / "s")
    plane = np.arange(1, H * W + 1, dtype=np.float32).reshape(H, W)
    one = build_tiff(W, H, [zlib.compress(plane[:2].tobytes())], compression=8, rows_per_strip=2)
    both = build_tiff(W, H, [zlib.compress(plane[:2].tobytes()), zlib.compress(plane[2:].tobytes())],
                      compression=8, rows_per_strip=2)
    replace_file(src, "z.tif", both)
    assert np.array_equal(T.read_tifxyz(src).z, plane)
    replace_file(src, "z.tif", one)
    exc = refusal(T.read_tifxyz, src)
    assert exc.code == "IO_TIFXYZ_TIFF_SEGMENTS"
    assert exc.details == {"file": "z.tif", "expected": 2, "offsets": 1, "byte_counts": 1}
    assert "StripOffsets has 1 and StripByteCounts 1 entries; the geometry needs 2" in exc.reason


def test_a_short_tile_list_is_refused_not_zero_filled():
    plane = np.arange(16 * 32, dtype=np.float32).reshape(16, 32) + 1
    tiles = [zlib.compress(np.ascontiguousarray(plane[:, :16]).tobytes()),
             zlib.compress(np.ascontiguousarray(plane[:, 16:]).tobytes())]
    array, layout = T.decode_first_ifd("z.tif", build_tiled(32, 16, tiles))
    assert np.array_equal(array, plane) and layout["tiled"] and layout["segments"] == 2
    exc = refusal(T.decode_first_ifd, "z.tif", build_tiled(32, 16, tiles, declared=1))
    assert exc.code == "IO_TIFXYZ_TIFF_SEGMENTS"
    assert exc.details == {"file": "z.tif", "expected": 2, "offsets": 1, "byte_counts": 1}
    assert "TileOffsets has 1 and TileByteCounts 1 entries" in exc.reason


@pytest.mark.parametrize("strips,kwargs,declared", [
    (3, {}, (3, 3)),  # a long list: tifffile silently drops the third entry
    (1, {"drop": (273,)}, (0, 1)),  # no StripOffsets
    (1, {"drop": (279,)}, (1, 0)),  # no StripByteCounts (tifffile would estimate one)
])
def test_strip_lists_must_be_exactly_as_long_as_the_geometry_needs(strips, kwargs, declared):
    rows = 2 if strips == 3 else H
    data = build_tiff(W, H, [float_strip(np.ones((rows, W)))] * strips, rows_per_strip=rows, **kwargs)
    exc = refusal(T.decode_first_ifd, "x.tif", data)
    assert exc.code == "IO_TIFXYZ_TIFF_SEGMENTS"
    assert exc.details == {"file": "x.tif", "expected": 2 if strips == 3 else 1, "offsets": declared[0],
                           "byte_counts": declared[1]}


@pytest.mark.parametrize("field,data", [
    ("image_length", build_tiff(W, 0, [float_strip(np.ones((H, W)))])),
    ("image_width", build_tiff(0, H, [float_strip(np.ones((H, W)))])),
    ("rows_per_strip", build_tiff(W, H, [float_strip(np.ones((H, W)))], rows_per_strip=0)),
    ("tile_width", build_tiled(32, 16, [b"\x00" * 1024] * 2, tile_hw=(16, 0), compression=1)),
    ("tile_length", build_tiled(32, 16, [b"\x00" * 1024] * 2, tile_hw=(0, 16), compression=1)),
], ids=lambda v: v if isinstance(v, str) else "tiff")
def test_zero_image_strip_and_tile_sizes_are_refused(field, data):
    exc = refusal(T.decode_first_ifd, "x.tif", data)
    assert exc.code == "IO_TIFXYZ_TIFF_INVALID"
    assert exc.details == {"file": "x.tif", "field": field, "value": 0}


def test_a_zero_size_mask_is_refused_not_taken_as_a_factor_zero_multiple(tmp_path):
    src = write_fixture(tmp_path / "s")
    replace_file(src, "mask.tif", build_tiff(0, 0, [b"\xff"], bits=8, sample_format=1))
    s = T.read_tifxyz(src)
    for policy in ("vc3d", "spiral", "vesuvius", "apok-strict"):
        exc = refusal(T.validity, s, policy)
        assert exc.code == "IO_TIFXYZ_TIFF_INVALID", policy
        assert exc.details == {"file": "mask.tif", "field": "image_length", "value": 0}, policy


def test_a_zero_width_band_is_refused_by_the_reader(tmp_path):
    src = write_fixture(tmp_path / "s")
    replace_file(src, "x.tif", build_tiff(0, H, [float_strip(np.ones((H, W)))]))
    exc = refusal(T.read_tifxyz, src)
    assert exc.code == "IO_TIFXYZ_TIFF_INVALID" and exc.details["field"] == "image_width"


@pytest.mark.parametrize("byteorder", ["<", ">"])
def test_deflate_with_the_horizontal_predictor_reads_bit_exactly(tmp_path, byteorder):
    # tifffile undoes predictor 2 without imagecodecs (integer differences of the 32-bit words).
    x, y, z = fixture_planes()
    d = tmp_path / "p"
    d.mkdir()
    for n, plane in (("x", x), ("y", y), ("z", z)):
        words = plane.view("<u4").astype(np.uint32)
        words[:, 1:] = words[:, 1:] - words[:, :-1]
        strip = zlib.compress(words.astype(byteorder + "u4").tobytes())
        (d / f"{n}.tif").write_bytes(build_tiff(W, H, [strip], compression=8, extra=[(317, 3, [2])],
                                                byteorder=byteorder))
    (d / "meta.json").write_text(json.dumps(META))
    s = T.read_tifxyz(d)
    for n, plane in (("x", x), ("y", y), ("z", z)):
        assert np.array_equal(u32(getattr(s, n)), u32(plane)), n
    assert s.tiffs["z.tif"]["predictor"] == 2 and s.tiffs["z.tif"]["compression"] == 8


@pytest.mark.skipif(importlib.util.find_spec("imagecodecs") is not None, reason="imagecodecs is installed here")
def test_floating_point_predictor_in_this_environment_needs_imagecodecs(tmp_path):
    src = write_fixture(tmp_path / "s")
    replace_file(src, "z.tif", build_tiff(W, H, [float_strip(np.ones((H, W)))], extra=[(317, 3, [3])]))
    exc = refusal(T.read_tifxyz, src)
    assert exc.code == "IO_TIFXYZ_CODEC_MISSING"
    assert exc.details == {"file": "z.tif", "codec": "predictor floating point"}


def test_names_that_are_not_utf8_are_recorded_escaped(tmp_path):
    src = write_fixture(tmp_path / "s")
    with open(os.path.join(os.fsencode(src), b"\xff.tif"), "wb") as fh:
        fh.write(b"abc")
    os.symlink(b"/nonexistent/\xfe", os.path.join(os.fsencode(src), b"link"))
    summary = T.inspect_tifxyz(src)
    canonical_dumps(summary)  # sealable: no lone surrogates
    files = {f["name"]: f for f in summary["surface"]["files"]}
    assert files["\\xff.tif"] == {"name": "\\xff.tif", "name_hex": "ff2e746966", "kind": "file", "size": 3,
                                   "digest": "sha256:" + hashlib.sha256(b"abc").hexdigest()}
    assert files["link"] == {"name": "link", "kind": "symlink", "target": "/nonexistent/\\xfe",
                             "target_hex": b"/nonexistent/\xfe".hex()}
    assert {"finding": "name-not-utf8", "name": "\\xff.tif", "name_hex": "ff2e746966"} in \
        summary["surface"]["findings"]
    assert "\\xff.tif" not in [f["name"] for f in T.read_tifxyz(write_fixture(tmp_path / "t")).files]


# ------------------------------------------------------------------------------------------ validity


INVALID = {  # without a mask, the vertices each policy rejects (docs/io-surfaces.md, "Validity policies")
    "vc3d": {(0, 0), (1, 0), (1, 1), (1, 2), (2, 1)},
    "spiral": {(1, 2)},
    "vesuvius": {(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 1)},
    "lasagna": {(1, 1), (1, 2), (2, 0)},
    "apok-strict": {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 1)},
}
ALL = {(r, c) for r in range(H) for c in range(W)}


def invalid_cells(result) -> set:
    return {(int(r), int(c)) for r, c in np.argwhere(~result.valid)}


def disagreements(expected: dict) -> list[dict]:
    names = list(expected)
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            only_a = len((ALL - expected[a]) - (ALL - expected[b]))
            only_b = len((ALL - expected[b]) - (ALL - expected[a]))
            out.append({"a": a, "b": b, "only_a": only_a, "only_b": only_b, "differ": only_a + only_b})
    return out


def test_policies_without_mask(tmp_path):
    s = T.read_tifxyz(write_fixture(tmp_path / "s"))
    report = T.validity_report(s)
    for policy, expected in INVALID.items():
        assert invalid_cells(report.results[policy]) == expected, policy
        assert report.results[policy].findings == ()
    assert list(report.disagreements) == disagreements(INVALID)
    assert report.census == {"vertices": 20, "full_sentinel": 1, "partial_sentinel": 2, "nonfinite": 2,
                             "nan_z": 1, "z_nonpositive_unmarked": 3, "negative_zero_z": 1}


def test_positive_infinite_z(tmp_path):
    x, y, z = fixture_planes()
    z[3, 4] = np.inf  # passes "z > 0" but not "finite": only the finiteness clause tells them apart
    T.write_tifxyz(tmp_path / "s", x=x, y=y, z=z, meta=META)
    s = T.read_tifxyz(tmp_path / "s")
    assert invalid_cells(T.validity(s, "vesuvius")) == INVALID["vesuvius"] | {(3, 4)}
    assert invalid_cells(T.validity(s, "apok-strict")) == INVALID["apok-strict"] | {(3, 4)}
    assert invalid_cells(T.validity(s, "vc3d")) == INVALID["vc3d"]
    assert T.grid_census(s)["nonfinite"] == 3 and T.grid_census(s)["nan_z"] == 1


def test_policies_with_a_0_255_mask(tmp_path):
    s = T.read_tifxyz(write_fixture(tmp_path / "s", mask=mask_255()))
    masked = {(3, 0), (3, 1)}
    expected = {
        "vc3d": INVALID["vc3d"] | masked,
        "spiral": INVALID["spiral"] | masked,
        "vesuvius": masked,  # a matching mask decides alone: the sentinel and z <= 0 cells come back
        "lasagna": INVALID["lasagna"],
        "apok-strict": INVALID["apok-strict"] | masked,
    }
    report = T.validity_report(s)
    for policy, cells in expected.items():
        assert invalid_cells(report.results[policy]) == cells, policy
    assert [f["finding"] for f in report.results["lasagna"].findings] == ["mask-ignored"]
    assert list(report.disagreements) == disagreements(expected)


def test_a_0_1_mask_splits_the_readers(tmp_path):
    s = T.read_tifxyz(write_fixture(tmp_path / "s", mask=mask_255(on=1)))
    assert invalid_cells(T.validity(s, "vc3d")) == ALL  # 1 < 255 invalidates every vertex in C++
    assert invalid_cells(T.validity(s, "spiral")) == INVALID["spiral"] | {(3, 0), (3, 1)}
    assert invalid_cells(T.validity(s, "vesuvius")) == {(3, 0), (3, 1)}
    exc = refusal(T.validity, s, "apok-strict")
    assert exc.code == "IO_TIFXYZ_MASK_VALUES" and exc.details["count"] == 18
    assert (exc.details["row"], exc.details["col"], exc.details["value"]) == (0, 0, 1)
    summary = T.inspect_tifxyz(s.path)
    assert summary["refused_policies"]["apok-strict"]["code"] == "IO_TIFXYZ_MASK_VALUES"
    assert sorted(summary["validity"]["policies"]) == ["lasagna", "spiral", "vc3d", "vesuvius"]
    canonical_dumps(summary)  # sealable as it stands: no NaN, no numpy scalars, no tuples


def test_inspect_summary_of_the_full_fixture_is_canonical_json(tmp_path):
    summary = T.inspect_tifxyz(write_fixture(tmp_path / "seg01", mask=mask_255()))
    canonical_dumps(summary)
    assert summary["surface"]["shape_hw"] == [H, W] and summary["refused_policies"] == {}
    assert summary["validity"]["census"]["partial_sentinel"] == 2


@pytest.mark.parametrize("dtype,on", [(np.uint16, 255), (np.uint16, 65535), (np.float32, 255.0),
                                      (np.int16, 255), (np.float64, 300.0)])
def test_vc3d_mask_threshold_is_255_for_listed_formats(tmp_path, dtype, on):
    s = T.read_tifxyz(write_fixture(tmp_path / "s", mask=mask_255(dtype, on)))
    assert invalid_cells(T.validity(s, "vc3d")) == INVALID["vc3d"] | {(3, 0), (3, 1)}


def test_vc3d_mask_value_254_invalidates(tmp_path):
    s = T.read_tifxyz(write_fixture(tmp_path / "s", mask=mask_255(np.float32, 254.9)))
    assert invalid_cells(T.validity(s, "vc3d")) == ALL


def test_vc3d_refuses_a_mask_format_with_no_defined_rule(tmp_path):
    s = T.read_tifxyz(write_fixture(tmp_path / "s", mask=np.full((H, W), -1, dtype=np.int8)))
    exc = refusal(T.validity, s, "vc3d")
    assert exc.code == "IO_TIFXYZ_MASK_UNSUPPORTED"
    assert exc.details == {"policy": "vc3d", "sample_format": 2, "bits_per_sample": 8}
    assert invalid_cells(T.validity(s, "spiral")) == INVALID["spiral"]  # int8 -1 != 0 keeps everything


def test_vc3d_integer_multiple_mask_invalidates_whole_blocks(tmp_path):
    big = np.full((2 * H, 2 * W), 255, dtype=np.uint8)
    big[7, 9] = 0  # -> vertex (3, 4)
    big[4, 0] = 254  # -> vertex (2, 0)
    s = T.read_tifxyz(write_fixture(tmp_path / "s", mask=big))
    assert invalid_cells(T.validity(s, "vc3d")) == INVALID["vc3d"] | {(3, 4), (2, 0)}
    assert T.validity(s, "vc3d").findings == ()
    exc = refusal(T.validity, s, "spiral")
    assert exc.code == "IO_TIFXYZ_MASK_SHAPE" and exc.details["mask_shape"] == [2 * H, 2 * W]
    vesuvius = T.validity(s, "vesuvius")
    assert invalid_cells(vesuvius) == INVALID["vesuvius"]
    assert vesuvius.findings[0]["finding"] == "mask-ignored" and vesuvius.findings[0]["policy"] == "vesuvius"
    exc = refusal(T.validity, s, "apok-strict")
    assert exc.code == "IO_TIFXYZ_MASK_SHAPE" and exc.details["policy"] == "apok-strict"


def test_vc3d_ignores_a_non_multiple_mask_and_says_so(tmp_path):
    s = T.read_tifxyz(write_fixture(tmp_path / "s", mask=np.zeros((H + 1, W), dtype=np.uint8)))
    result = T.validity(s, "vc3d")
    assert invalid_cells(result) == INVALID["vc3d"]
    assert result.findings[0]["finding"] == "mask-ignored" and result.findings[0]["mask_shape"] == [H + 1, W]


@pytest.mark.parametrize("planar", ["contig", "separate"])
def test_mask_channel_0_is_the_validity_channel(tmp_path, planar):
    channel0 = mask_255()
    other = np.zeros((H, W), dtype=np.uint8)
    stacked = np.stack([channel0, other], axis=-1 if planar == "contig" else 0)
    s = T.read_tifxyz(write_fixture(tmp_path / "s", mask=stacked, photometric="minisblack", planarconfig=planar))
    assert invalid_cells(T.validity(s, "vc3d")) == INVALID["vc3d"] | {(3, 0), (3, 1)}
    assert invalid_cells(T.validity(s, "spiral")) == INVALID["spiral"] | {(3, 0), (3, 1)}
    vesuvius = T.validity(s, "vesuvius")  # vesuvius compares the whole array's shape: ignored
    assert invalid_cells(vesuvius) == INVALID["vesuvius"] and vesuvius.findings[0]["finding"] == "mask-ignored"


def test_multi_page_mask(tmp_path):
    s = T.read_tifxyz(write_fixture(tmp_path / "s", mask=np.stack([mask_255(), np.zeros((H, W), np.uint8)])))
    assert invalid_cells(T.validity(s, "vc3d")) == INVALID["vc3d"] | {(3, 0), (3, 1)}  # first IFD only
    assert invalid_cells(T.validity(s, "spiral")) == INVALID["spiral"] | {(3, 0), (3, 1)}
    for policy in ("vesuvius", "apok-strict"):
        exc = refusal(T.validity, s, policy)
        assert exc.code == "IO_TIFXYZ_MASK_UNSUPPORTED" and exc.details == {"policy": policy}


def test_apok_strict_refuses_nan_in_a_float_mask(tmp_path):
    m = mask_255(np.float32)
    m[2, 3] = np.nan
    s = T.read_tifxyz(write_fixture(tmp_path / "s", mask=m))
    exc = refusal(T.validity, s, "apok-strict")
    assert exc.code == "IO_TIFXYZ_MASK_VALUES" and (exc.details["row"], exc.details["col"]) == (2, 3)
    assert exc.details["value"] == "nan"


def test_undecodable_mask_is_refused_when_a_policy_needs_it(tmp_path, monkeypatch):
    monkeypatch.setattr(tifffile.TIFF, "DECOMPRESSORS", _NoCodecs())
    src = write_fixture(tmp_path / "s")
    replace_file(src, "mask.tif", build_tiff(W, H, [b"\x80" * 20], bits=8, sample_format=1, compression=5))
    s = T.read_tifxyz(src)  # recorded, not interpreted
    exc = refusal(T.validity, s, "vc3d")
    assert exc.code == "IO_TIFXYZ_CODEC_MISSING" and exc.details["file"] == "mask.tif"
    assert invalid_cells(T.validity(s, "lasagna")) == INVALID["lasagna"]


def test_unknown_policy(tmp_path):
    s = T.read_tifxyz(write_fixture(tmp_path / "s"))
    exc = refusal(T.validity, s, "VC3D")
    assert exc.code == "IO_TIFXYZ_POLICY_UNKNOWN" and exc.details["policy"] == "VC3D"


def test_report_bounds_use_decoded_valid_coordinates(tmp_path):
    s = T.read_tifxyz(write_fixture(tmp_path / "s", meta={**META, "bbox": [[0, 0, 0], [1, 1, 1]]}))
    strict = T.validity_report(s, ["apok-strict"]).as_dict()["policies"]["apok-strict"]
    assert strict["valid"] == 13 and strict["valid_nonfinite"] == 0
    assert strict["min_zyx"] == [9.0, 7.0, -1.0] and strict["max_zyx"] == [49.0, 23.0, 14.0]
    vc3d = T.validity_report(s, ["vc3d"]).as_dict()["policies"]["vc3d"]
    assert vc3d["valid_nonfinite"] == 2


def test_validity_masks_are_read_only(tmp_path):
    s = T.read_tifxyz(write_fixture(tmp_path / "s"))
    with pytest.raises(ValueError):
        T.validity(s, "vc3d").valid[0, 0] = True


# ---------------------------------------------------------------------------------------- writer refusals


def test_existing_directory_needs_overwrite(tmp_path):
    src = write_fixture(tmp_path / "s")
    x, y, z = fixture_planes()
    exc = refusal(T.write_tifxyz, src, x=x, y=y, z=z, meta=META)
    assert exc.code == "IO_TIFXYZ_WRITE_EXISTS" and exc.details["path"] == src
    (tmp_path / "file").write_text("x")
    exc = refusal(T.write_tifxyz, tmp_path / "file", x=x, y=y, z=z, meta=META, overwrite=True)
    assert exc.code == "IO_TIFXYZ_NOT_DIRECTORY" and exc.details["path"] == str(tmp_path / "file")
    assert (tmp_path / "file").read_text() == "x"


def test_writer_does_not_follow_a_symlinked_target_directory(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "link"
    os.symlink(outside, link)
    x, y, z = fixture_planes()
    exc = refusal(T.write_tifxyz, link, x=x, y=y, z=z, meta=META, overwrite=True)
    assert exc.code == "IO_TIFXYZ_WRITE_SYMLINK" and exc.details == {"path": str(link)}
    assert str(outside) in exc.reason
    assert os.listdir(outside) == [] and os.path.islink(link)
    exc = refusal(T.write_tifxyz, link, x=x, y=y, z=z, meta=META)
    assert exc.code == "IO_TIFXYZ_WRITE_EXISTS"


@pytest.mark.parametrize("field,bad", [
    ("x.tif", np.ones((H, W), dtype=np.float64)),
    ("y.tif", np.ones((H, W, 1), dtype=np.float32)),
    ("z.tif", [[1.0]]),
    ("z.tif", np.ones((H, W), dtype=np.int32)),
])
def test_writer_refuses_non_float32_planes(tmp_path, field, bad):
    x, y, z = fixture_planes()
    planes = {"x": x, "y": y, "z": z, field[0]: bad}
    exc = refusal(T.write_tifxyz, tmp_path / "o", meta=META, **planes)
    assert exc.code == "IO_TIFXYZ_WRITE_INVALID" and exc.details["field"] == field
    assert not (tmp_path / "o").exists()


def test_writer_refuses_mismatched_shapes_and_empty_planes(tmp_path):
    x, y, z = fixture_planes()
    exc = refusal(T.write_tifxyz, tmp_path / "o", x=x, y=y[:, :-1], z=z, meta=META)
    assert exc.code == "IO_TIFXYZ_WRITE_INVALID" and exc.details["field"] == "shape"
    empty = np.zeros((0, 3), dtype=np.float32)
    exc = refusal(T.write_tifxyz, tmp_path / "o", x=empty, y=empty, z=empty, meta=META)
    assert exc.code == "IO_TIFXYZ_WRITE_INVALID" and exc.details["shape"] == [0, 3]
    assert not (tmp_path / "o").exists()


@pytest.mark.parametrize("meta,code,field", [
    ({**META, "bbox": (1, 2)}, "IO_TIFXYZ_WRITE_INVALID", "meta"),
    ({**META, "area": float("nan")}, "IO_TIFXYZ_WRITE_INVALID", "meta"),
    ({**META, "scale": [20]}, "IO_TIFXYZ_META_FIELD", "scale"),
    ({**META, "format": "obj"}, "IO_TIFXYZ_META_FIELD", "format"),
])
def test_writer_refuses_meta_it_could_not_read_back(tmp_path, meta, code, field):
    x, y, z = fixture_planes()
    exc = refusal(T.write_tifxyz, tmp_path / "o", x=x, y=y, z=z, meta=meta)
    assert exc.code == code and exc.details["field"] == field
    assert not (tmp_path / "o").exists()


def test_bigtiff_threshold_is_refused_above_and_written_at(tmp_path):
    at_limit = T.canonical_tiff_header(16384, 65535)  # exactly 0xffff0000 pixel bytes
    assert struct.unpack("<I", at_limit[8 + 2 + 12 * 8 + 8: 8 + 2 + 12 * 8 + 12])[0] == 0xFFFF0000
    exc = refusal(T.canonical_tiff_header, 16384, 65536)
    assert exc.code == "IO_TIFXYZ_WRITE_TOO_LARGE" and exc.details["nbytes"] == 0x100000000
    huge = np.broadcast_to(np.float32(1), (16384, 65536))  # no 4 GiB allocation: the guard runs first
    exc = refusal(T.write_tifxyz, tmp_path / "o", x=huge, y=huge, z=huge, meta=META)
    assert exc.code == "IO_TIFXYZ_WRITE_TOO_LARGE" and exc.details["shape"] == [16384, 65536]
    assert not (tmp_path / "o").exists()


# --------------------------------------------------------------------------------------- shared helpers


@pytest.mark.parametrize("value, why", [
    (b'"\\ud800"', "lone surrogate '\\ud800' in the string at $.area"),
    (b"[" * 300 + b"]" * 300, "arrays and objects nested deeper than 256 levels"),
], ids=["lone-surrogate", "deeper-than-256"])
def test_meta_json_with_a_lone_surrogate_or_deeper_than_the_strict_limit_is_refused(tmp_path, value, why):
    # Both read before apok_eval.jsonstrict: the surrogate as a str that cannot be encoded, the nesting as data.
    src = write_fixture(tmp_path / "s")
    replace_file(src, "meta.json", b'{"format": "tifxyz", "type": "seg", "uuid": "s", "scale": [0.05, 0.05], '
                                   b'"area": ' + value + b"}")
    exc = refusal(T.read_tifxyz, src)
    assert exc.code == "IO_TIFXYZ_META_INVALID" and exc.details["file"] == "meta.json"
    assert why in exc.reason


@pytest.mark.parametrize("value", [b"1" * 5000, b"[" * 100000 + b"]" * 100000], ids=["huge-int", "deep-nesting"])
def test_meta_json_with_an_unconvertible_integer_or_deep_nesting_is_refused(tmp_path, value):
    src = write_fixture(tmp_path / "s")
    replace_file(src, "meta.json", b'{"format": "tifxyz", "type": "seg", "uuid": "s", "scale": [0.05, 0.05], '
                                   b'"area": ' + value + b"}")
    exc = refusal(T.read_tifxyz, src)
    assert exc.code == "IO_TIFXYZ_META_INVALID" and exc.details["file"] == "meta.json"
    assert "more than int() converts" in exc.reason or "nested deeper" in exc.reason


def nested(depth: int) -> list:
    value: list = []
    for _ in range(depth):
        value = [value]
    return value


@pytest.mark.parametrize("extra,why", [(10 ** 5000, "ValueError"), (nested(100000), "RecursionError"),
                                       (nested(300), "would not read back as strict JSON: arrays and objects nested "
                                                     "deeper than 256 levels")],
                         ids=["huge-int", "deep-nesting", "deeper-than-the-reader-reads"])
def test_writer_refuses_meta_json_the_reader_would_refuse(tmp_path, extra, why):
    x, y, z = fixture_planes()
    exc = refusal(T.write_tifxyz, tmp_path / "o", x=x, y=y, z=z, meta={**META, "extra": extra})
    assert exc.code == "IO_TIFXYZ_WRITE_INVALID" and exc.details["field"] == "meta"
    assert why in exc.details["problem"]
    assert not (tmp_path / "o").exists()


# --- hostile IFD values are refusals, not crashes or allocations ------------------------------------------

_ENTRY = {256: 0, 257: 1, 258: 2, 259: 3, 262: 4, 273: 5, 277: 6, 278: 7, 279: 8, 284: 9, 339: 10}


def _patched(data: bytes, tag: int, *, typ=None, count=None, value=None) -> bytes:
    """One IFD entry of apok-eval's own canonical TIFF (11 entries from offset 10, in tag order) rewritten."""
    out = bytearray(data)
    at = 10 + 12 * _ENTRY[tag]
    assert struct.unpack_from("<H", out, at)[0] == tag
    for offset, fmt, new in ((2, "<H", typ), (4, "<I", count), (8, "<I", value)):
        if new is not None:
            struct.pack_into(fmt, out, at + offset, new)
    return bytes(out)


def _mask_tiff(**entries) -> bytes:
    """An uncompressed (H, W) uint8 all-255 mask in the same 11-entry layout, with ``entries`` patched."""
    header = T.encode_float32_tiff(np.zeros((H, W), np.float32), "mask.tif")[:148]
    data = header + np.full((H, W), 255, np.uint8).tobytes()
    data = _patched(_patched(_patched(data, 258, value=8), 339, value=1), 279, value=H * W)
    for tag, value in entries.items():
        data = _patched(data, int(tag[1:]), value=value)
    return data


def test_a_geometry_claiming_billions_of_strips_is_refused_before_any_list_is_built(tmp_path):
    import time

    src = write_fixture(tmp_path / "s")
    x = open(os.path.join(src, "x.tif"), "rb").read()
    with open(os.path.join(src, "x.tif"), "wb") as fh:
        fh.write(_patched(_patched(x, 257, value=0xFFFFFF00), 278, value=1))
    started = time.monotonic()
    exc = refusal(T.read_tifxyz, src)
    assert time.monotonic() - started < 5  # once: a list of 4.3e9 entries, then MemoryError
    assert exc.code == "IO_TIFXYZ_TIFF_SEGMENTS"
    assert exc.details == {"file": "x.tif", "expected": 0xFFFFFF00, "offsets": 1, "byte_counts": 1}


@pytest.mark.parametrize("tag, field", [("t277", "samples_per_pixel"), ("t258", "bits_per_sample")])
def test_a_mask_with_zero_samples_or_bits_is_refused_not_a_division_by_zero(tmp_path, tag, field):
    src = write_fixture(tmp_path / "s")
    with open(os.path.join(src, "mask.tif"), "wb") as fh:
        fh.write(_mask_tiff(**{tag: 0}))
    surface = T.read_tifxyz(src)
    exc = refusal(T.validity, surface, "vc3d")
    assert exc.code == "IO_TIFXYZ_TIFF_INVALID" and exc.details == {"file": "mask.tif", "field": field, "value": 0}


@pytest.mark.parametrize("tag, edit, field", [
    (262, {"count": 0}, "photometric"),            # no value: tifffile gives an empty tuple
    (257, {"typ": 1, "count": 0}, "image_length"),  # an empty BYTE value
    (256, {"typ": 2, "count": 4, "value": 0x00313233}, "image_width"),  # ASCII "321"
])
def test_an_ifd_value_that_is_not_one_integer_is_refused_naming_its_tag(tmp_path, tag, edit, field):
    src = write_fixture(tmp_path / "s")
    x = open(os.path.join(src, "x.tif"), "rb").read()
    with open(os.path.join(src, "x.tif"), "wb") as fh:
        fh.write(_patched(x, tag, **edit))
    exc = refusal(T.read_tifxyz, src)
    assert exc.code == "IO_TIFXYZ_TIFF_INVALID" and exc.details["file"] == "x.tif"
    # tifffile itself refuses some damage while opening the file ("not a readable TIFF", no field)
    assert exc.details.get("field", field) in (field, "structure") and "TypeError" in exc.reason


def test_any_other_exception_in_the_structure_checks_is_a_refusal_naming_it(tmp_path, monkeypatch):
    src = write_fixture(tmp_path / "s")

    def damaged(*args):  # tifffile computes some values lazily; damage there raises whatever it raises
        raise struct.error("unpack requires a buffer of 4 bytes")

    monkeypatch.setattr(T, "_next_ifd_offset", damaged)
    exc = refusal(T.read_tifxyz, src)
    assert exc.code == "IO_TIFXYZ_TIFF_INVALID"
    assert exc.details == {"file": "x.tif", "field": "structure", "error": "error"}  # struct.error


def test_check_within_frame_compares_the_volume_meta_json_states(tmp_path):
    plane = np.full((2, 2), 5, np.float32)
    frame = VolumeFrame("20231117143551", (100, 100, 100), 7.91, "test")

    def surface(name, **stated):
        T.write_tifxyz(tmp_path / name, x=plane, y=plane, z=plane, meta={**META, **stated})
        return T.read_tifxyz(tmp_path / name)

    assert T.check_within_frame(surface("s", volume="20231117143551", base_shape_zyx=[100, 100, 100]), frame,
                                "vc3d") == 4
    exc = refusal(T.check_within_frame, surface("o", volume="20990101000004", base_shape_zyx=[50, 60, 70]), frame,
                  "vc3d")  # its coordinates would fit the frame
    assert exc.code == "IO_FRAME_VOLUME_MISMATCH" and exc.details["field"] == "volume"
