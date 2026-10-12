"""Known-answer OME-Zarr v2 volumes: planted values on level 0 and level 2 read back exactly.

Three stores are built per test in tmp_path, by two independent writers:

* ``zarr-python``: zarr-python v2 (a real third-party writer), uint8, levels coded
  blosc-lz4 '.', blosc-zstd '/', zlib '.', raw '/'; full-size edge chunks (zarr
  always writes them so); one level-0 chunk deleted (absent -> fill 0).
* ``own-cropped``: the tests' own writer, uint8, zlib '/', raw with no declared
  separator, blosc-lz4 '/', blosc-zstd '.'; edge chunks cropped (and on level 1
  alternately full and cropped); level 0 has fill_value null and an absent edge chunk.
* ``own-full-u2``: the tests' own writer, little-endian uint16, gzip '.', zstd '/',
  lz4 '.', blosc-zlib '/'; full-size edge chunks.

zarr-python reads ``zarr-python`` and ``own-full-u2`` as an oracle (it cannot read
cropped edge chunks, so ``own-cropped`` is checked against the arrays written).
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
import zarr
from io_zarr_fixtures import (
    SHAPE0, LoopbackServer, chunk_path, clearance_for, edit_json, multiscales, planted, pyramid, small_levels,
    write_small, write_volume,
)
from numcodecs import Blosc, Zlib

from apok_eval.canonical import canonical_dumps, sha256_file
from apok_eval.io.frame import VolumeFrame
from apok_eval.io.omezarr import ASSUMPTIONS, MAPPING, VOXEL_CONVENTION, open_volume
from apok_eval.io.store import HttpStore, LocalStore
from apok_eval.licence import LicenceClearance
from apok_eval.merkle import SCHEME, manifest_root

FRAME = VolumeFrame("20990101000004", SHAPE0, 2.4, "test:known-answer")
SMALL = VolumeFrame("20241024131838", (9, 7, 5), 7.91, "test:small")


def blosc(cname: str, shuffle: int = 1) -> dict:
    return {"id": "blosc", "cname": cname, "clevel": 5, "shuffle": shuffle, "blocksize": 0}


def build_zarr_python(root):
    levels = pyramid("|u1")
    group = zarr.open_group(str(root), mode="w")
    group.attrs["multiscales"] = [multiscales(4)]
    # separator None: zarr writes no dimension_separator key, so those levels exercise the '.' default.
    plan = [(Blosc(cname="lz4", shuffle=1), None, (8, 8, 8)), (Blosc(cname="zstd", shuffle=2), "/", (8, 8, 8)),
            (Zlib(level=1), None, (4, 4, 4)), (None, "/", (2, 2, 2))]
    for k, (compressor, separator, chunks) in enumerate(plan):
        array = group.create_dataset(str(k), shape=levels[k].shape, chunks=chunks, dtype="u1",
                                     compressor=compressor, fill_value=0, dimension_separator=separator)
        array[...] = levels[k]
    (root / "0" / "2.1.0").unlink()  # level-0 z 16:24, y 8:16, x 0:8 now absent
    levels[0][16:24, 8:16, 0:8] = 0
    return levels, {"0/2.1.0"}


def build_own_cropped(root):
    levels = pyramid("|u1")
    specs = [dict(chunks=(8, 8, 8), compressor={"id": "zlib", "level": 1}, separator="/", edge="cropped",
                  fill_value=None, absent=((0, 3, 2),)),
             dict(chunks=(8, 8, 8), declare_separator=False, edge="mixed"),
             dict(chunks=(4, 4, 4), compressor=blosc("lz4"), separator="/", edge="cropped"),
             dict(chunks=(4, 4, 4), compressor=blosc("zstd", 2), edge="cropped")]
    write_volume(root, levels, specs)
    levels[0][0:8, 24:29, 16:23] = 0  # fill_value null reads as 0
    return levels, {"0/0/3/2"}


def build_own_full_u2(root):
    levels = pyramid("<u2")
    specs = [dict(chunks=(16, 16, 16), compressor={"id": "gzip", "level": 1}),
             dict(chunks=(8, 8, 8), compressor={"id": "zstd", "level": 1}, separator="/"),
             dict(chunks=(4, 4, 4), compressor={"id": "lz4", "acceleration": 1}),
             dict(chunks=(4, 4, 4), compressor=blosc("zlib"), separator="/")]
    write_volume(root, levels, specs)
    return levels, set()


BUILDERS = {"zarr-python": build_zarr_python, "own-cropped": build_own_cropped, "own-full-u2": build_own_full_u2}
ORACLE_READABLE = {"zarr-python", "own-full-u2"}


@pytest.fixture(params=sorted(BUILDERS))
def known(request, tmp_path):
    root = tmp_path / f"{request.param}.zarr"
    root.mkdir()
    expected, absent = BUILDERS[request.param](root)
    oracle = zarr.open_group(str(root), mode="r") if request.param in ORACLE_READABLE else None
    return request.param, root, expected, absent, oracle


def level_box(start, stop, k):
    f = 2 ** k
    return tuple(a // f for a in start), tuple(-(-b // f) for b in stop)


BOXES = [((0, 0, 0), (1, 1, 1)), ((7, 7, 7), (9, 9, 9)), ((5, 6, 7), (19, 21, 17)), ((30, 25, 20), (37, 29, 23)),
         ((15, 0, 15), (17, 29, 16)), ((1, 2, 3), (36, 27, 22)), ((16, 8, 0), (24, 16, 8))]


def test_every_level_reads_back_whole_equal_to_the_written_arrays_and_to_zarr_python(known):
    _, root, expected, _, oracle = known
    volume = open_volume(LocalStore(root), FRAME)
    for k in range(4):
        array, record = volume.read((0, 0, 0), SHAPE0, k)
        assert array.dtype == expected[k].dtype and array.shape == expected[k].shape
        assert np.array_equal(array, expected[k])
        assert record.level_start == (0, 0, 0) and record.level_stop == expected[k].shape
        if oracle is not None:
            assert np.array_equal(array, oracle[str(k)][...])


def test_planted_level0_values_read_back_exactly(known):
    _, root, expected, _, oracle = known
    volume = open_volume(LocalStore(root), FRAME)
    whole, _ = volume.read((0, 0, 0), SHAPE0, 0)
    for pos, value in planted(expected[0].dtype.str)[0]:
        voxel, record = volume.read(pos, tuple(c + 1 for c in pos), 0)
        assert voxel.shape == (1, 1, 1) and voxel[0, 0, 0] == value
        assert whole[pos] == value
        assert record.level_start == pos and record.covered_start == pos
        if oracle is not None:
            assert oracle["0"][pos] == value


def test_planted_level2_values_read_back_through_level0_coordinates(known):
    _, root, expected, _, oracle = known
    volume = open_volume(LocalStore(root), FRAME)
    for pos, value in planted(expected[0].dtype.str)[1]:
        corner = tuple(4 * c for c in pos)
        far = tuple(min(4 * c + 3, n - 1) for c, n in zip(pos, SHAPE0))  # any level-0 voxel of the 4^3 block
        for start in (corner, far):
            voxel, record = volume.read(start, tuple(c + 1 for c in start), 2)
            assert voxel.shape == (1, 1, 1) and voxel[0, 0, 0] == value
            assert record.level == 2 and record.factor == 4
            assert record.level_start == pos and record.level_stop == tuple(c + 1 for c in pos)
            assert record.covered_start == corner
            assert record.covered_stop == tuple(min(4 * c + 4, n) for c, n in zip(pos, SHAPE0))
        if oracle is not None:
            assert oracle["2"][pos] == value


@pytest.mark.parametrize("start,stop", BOXES)
def test_boxes_across_chunk_boundaries_read_exactly_on_every_level(known, start, stop):
    _, root, expected, _, oracle = known
    volume = open_volume(LocalStore(root), FRAME)
    for k in range(4):
        lo, hi = level_box(start, stop, k)
        array, record = volume.read(start, stop, k)
        window = tuple(slice(a, b) for a, b in zip(lo, hi))
        assert np.array_equal(array, expected[k][window])
        assert (record.level_start, record.level_stop) == (lo, hi)
        assert (record.level0_start, record.level0_stop) == (start, stop) and not record.clipped
        if oracle is not None:
            assert np.array_equal(array, oracle[str(k)][window])


def test_a_read_record_states_both_boxes_the_mapping_and_the_voxel_convention(tmp_path):
    expected, _ = build_own_full_u2(tmp_path)
    volume = open_volume(LocalStore(tmp_path), FRAME)
    array, record = volume.read((5, 5, 5), (9, 9, 9), 2)
    assert array.shape == (2, 2, 2)
    assert record.as_dict() == {
        "volume_id": "20990101000004", "level": 2, "factor": 4,
        "requested_start": [5, 5, 5], "requested_stop": [9, 9, 9], "clip_requested": False, "clipped": False,
        "level0_start": [5, 5, 5], "level0_stop": [9, 9, 9], "level_start": [1, 1, 1], "level_stop": [3, 3, 3],
        "covered_start": [4, 4, 4], "covered_stop": [12, 12, 12], "chunks": 1, "absent_chunks": [],
        "fill_value": 0, "mapping": MAPPING, "voxel_convention": VOXEL_CONVENTION}
    _, edge = volume.read((30, 25, 20), (37, 29, 23), 2)
    assert (edge.level_start, edge.level_stop) == ((7, 6, 5), (10, 8, 6))
    assert edge.chunks == 2 and edge.absent_chunks == ()  # level-2 chunks of 4: z 1..2, y 1, x 1
    assert (edge.covered_start, edge.covered_stop) == ((28, 24, 20), (37, 29, 23))
    canonical_dumps(record.as_dict())


def test_clip_is_explicit_and_recorded(tmp_path):
    expected, _ = build_own_full_u2(tmp_path)
    volume = open_volume(LocalStore(tmp_path), FRAME)
    array, record = volume.read((-3, 0, 0), (5, 40, 23), 0, clip=True)
    assert np.array_equal(array, expected[0][0:5, 0:29, 0:23])
    assert record.clip_requested and record.clipped
    assert (record.requested_start, record.requested_stop) == ((-3, 0, 0), (5, 40, 23))
    assert (record.level0_start, record.level0_stop) == ((0, 0, 0), (5, 29, 23))
    _, inside = volume.read((0, 0, 0), (2, 2, 2), 1, clip=True)
    assert inside.clip_requested and not inside.clipped


def test_absent_chunks_read_as_fill_and_are_logged_absent(known):
    name, root, expected, absent, _ = known
    volume = open_volume(LocalStore(root), FRAME)
    whole, record = volume.read((0, 0, 0), SHAPE0, 0)
    assert np.array_equal(whole, expected[0])
    assert set(record.absent_chunks) == absent and record.fill_value == 0
    entries = {e["key"]: e for e in volume.log.entries()}
    for key in absent:
        assert entries[key] == {"key": key, "absent": True}
    if name == "zarr-python":
        assert not whole[16:24, 8:16, 0:8].any()
    if name == "own-cropped":
        assert not whole[0:8, 24:29, 16:23].any()
        assert volume.describe()["levels"][0]["fill_value_declared"] is None
        assert volume.describe()["levels"][0]["fill_value"] == 0


def test_the_read_log_digests_every_object_exactly_as_stored(known):
    _, root, _, absent, _ = known
    volume = open_volume(LocalStore(root), FRAME)
    opened = {e["key"] for e in volume.log.entries()}
    assert opened == {".zgroup", ".zattrs", "meta.json", "metadata.json"} | {
        f"{k}/{name}" for k in range(4) for name in (".zarray", ".zattrs")}
    for k in range(4):
        volume.read((0, 0, 0), SHAPE0, k)
    entries = volume.log.entries()
    assert entries == sorted(entries, key=lambda e: e["key"].encode())
    assert volume.log.root() == manifest_root(entries)
    for entry in entries:
        path = root.joinpath(*entry["key"].split("/"))
        if entry.get("absent"):
            assert not path.exists()
        else:  # the digest is of the compressed bytes on disk, not of decoded voxels
            assert entry["digest"] == sha256_file(path) and entry["size"] == path.stat().st_size
    assert absent <= {e["key"] for e in entries if e.get("absent")}
    chunk_entries = [e for e in entries if e["key"][0].isdigit() and "/." not in e["key"]]
    assert len(chunk_entries) == sum(
        np.prod([-(-n // c) for n, c in zip(lv["shape"], lv["chunks"])]) for lv in volume.describe()["levels"])


class CountingStore(LocalStore):
    def __init__(self, root):
        super().__init__(root)
        self.fetched: list[str] = []

    def get(self, key):
        self.fetched.append(key)
        return super().get(key)


def test_overlapping_reads_refetch_shared_chunks_and_check_them_against_the_log(tmp_path):
    build_own_full_u2(tmp_path)
    store = CountingStore(tmp_path)
    volume = open_volume(store, FRAME)
    first, _ = volume.read((0, 0, 0), (20, 20, 20), 0)
    second, _ = volume.read((10, 10, 10), (30, 29, 23), 0)
    assert np.array_equal(first[10:20, 10:20, 10:20], second[0:10, 0:10, 0:10])
    assert store.fetched.count("0/1.1.1") == 2  # fetched by both reads, recorded once
    assert [e["key"] for e in volume.log.entries()].count("0/1.1.1") == 1


def test_sidecars_enter_the_log_uninterpreted(tmp_path):
    write_small(tmp_path)
    (tmp_path / "meta.json").write_bytes(b"not json, and not read as such")
    volume = open_volume(LocalStore(tmp_path), SMALL)
    entries = {e["key"]: e for e in volume.log.entries()}
    assert entries["meta.json"]["digest"] == sha256_file(tmp_path / "meta.json")
    assert entries["metadata.json"] == {"key": "metadata.json", "absent": True}


def test_a_remote_read_matches_the_local_read_and_its_root(tmp_path, monkeypatch):
    root = tmp_path / "vol.zarr"
    root.mkdir()
    expected, _ = build_zarr_python(root)
    local = open_volume(LocalStore(root), FRAME)
    with LoopbackServer(tmp_path) as server:
        clearance = clearance_for(tmp_path, server.url + "/vol.zarr")  # issued by the licence gate
        asked: list[tuple[str, int]] = []  # each question put to the gate, with the requests served before it
        gate_covers = LicenceClearance.covers

        def covers(self, uri):
            asked.append((uri, len(server.requests)))
            return gate_covers(self, uri)

        monkeypatch.setattr(LicenceClearance, "covers", covers)
        remote = open_volume(HttpStore(server.url + "/vol.zarr", clearance), FRAME)
        for start, stop in BOXES:
            for k in (0, 2):
                assert np.array_equal(remote.read(start, stop, k)[0], local.read(start, stop, k)[0])
        assert np.array_equal(remote.read((0, 0, 0), SHAPE0, 2)[0], expected[2])
        assert np.array_equal(local.read((0, 0, 0), SHAPE0, 2)[0], expected[2])
        assert remote.log.entries() == local.log.entries()
        assert remote.log.root() == local.log.root()
        assert set(server.paths()) == {"/vol.zarr/" + e["key"] for e in remote.log.entries()}
        for i, path in enumerate(server.paths()):  # the gate was asked about each object before it was requested
            assert any(uri == server.url + path and served <= i for uri, served in asked), path
        assert remote.describe()["store"]["base_uri"] == server.url + "/vol.zarr"


def test_describe_is_canonical_and_records_every_declaration(known):
    name, root, _, _, _ = known
    volume = open_volume(LocalStore(root), FRAME)
    volume.read((0, 0, 0), (9, 9, 9), 1)
    described = volume.describe()
    canonical_dumps(described)
    assert described["frame"] == FRAME.as_dict()
    assert described["axes"] == {"order": ["z", "y", "x"], "source": ".zattrs", "declared": None,
                                 "declared_by": None}
    assert described["multiscales"] == {"version": "0.4", "name": "apok-fixture", "descriptors": 1}
    assert described["pyramid_tolerance"] == "exact"
    assert described["assumptions"] == list(ASSUMPTIONS) and VOXEL_CONVENTION in described["assumptions"]
    assert described["read_log"] == {"scheme": SCHEME, "root": volume.log.root(), "objects": len(volume.log)}
    levels = described["levels"]
    assert [lv["factor"] for lv in levels] == [1, 2, 4, 8]
    assert [lv["scale"] for lv in levels] == [[1.0] * 3, [2.0] * 3, [4.0] * 3, [8.0] * 3]
    assert [lv["shape"] for lv in levels] == [[37, 29, 23], [19, 15, 12], [10, 8, 6], [5, 4, 3]]
    assert all(lv["padding"] == [0, 0, 0] and lv["translation"] is None for lv in levels)
    if name == "zarr-python":
        assert [(lv["separator"], lv["separator_declared"]) for lv in levels] == [
            (".", False), ("/", True), (".", False), ("/", True)]
        assert levels[0]["compressor"]["cname"] == "lz4" and levels[3]["compressor"] is None
    if name == "own-cropped":
        assert (levels[1]["separator"], levels[1]["separator_declared"]) == (".", False)


@pytest.mark.parametrize("order", [("z", "y", "x"), ["z", "y", "x"]])
def test_axes_may_be_declared_by_the_caller_when_the_store_has_none(tmp_path, order):
    write_small(tmp_path, axes=None)
    volume = open_volume(LocalStore(tmp_path), SMALL, axes_declared=order, axes_declared_by="registry:test-volume")
    assert volume.axes == {"order": ["z", "y", "x"], "source": "declared", "declared": ["z", "y", "x"],
                           "declared_by": "registry:test-volume"}
    assert np.array_equal(volume.read((0, 0, 0), (9, 7, 5))[0], small_levels()[0])


def test_a_declaration_beside_the_stores_own_axes_is_recorded_with_the_store_as_source(tmp_path):
    write_small(tmp_path)
    volume = open_volume(LocalStore(tmp_path), SMALL, axes_declared=("z", "y", "x"), axes_declared_by="registry:v")
    assert volume.axes == {"order": ["z", "y", "x"], "source": ".zattrs", "declared": ["z", "y", "x"],
                           "declared_by": "registry:v"}


def test_a_surrogate_pair_escape_is_one_character_and_accepted(tmp_path):
    write_small(tmp_path)
    edit_json(tmp_path / ".zattrs", lambda o: o["multiscales"][0].update(name="vol \U0001F600"))
    assert b"\\ud83d\\ude00" in (tmp_path / ".zattrs").read_bytes()  # written as an escaped pair
    volume = open_volume(LocalStore(tmp_path), SMALL)
    assert volume.describe()["multiscales"]["name"] == "vol \U0001F600"
    canonical_dumps(volume.describe())


def test_axes_given_as_strings_and_matching_array_dimension_attrs_are_accepted(tmp_path):
    write_small(tmp_path)
    edit_json(tmp_path / ".zattrs", lambda o: o["multiscales"][0].update(axes=["z", "y", "x"]))
    (tmp_path / "1" / ".zattrs").write_text('{"_ARRAY_DIMENSIONS": ["z", "y", "x"], "dimension_names": '
                                            '["z", "y", "x"], "axes": [{"name": "z"}, {"name": "y"}, {"name": "x"}]}')
    volume = open_volume(LocalStore(tmp_path), SMALL)
    assert volume.axes["source"] == ".zattrs"


@pytest.mark.parametrize("scales", [None, [[7.91] * 3, [15.82] * 3, [31.64] * 3], [[1, 1, 1], [2, 2, 2], [4, 4, 4]]])
def test_scales_may_be_relative_physical_or_absent_and_are_recorded_not_interpreted(tmp_path, scales):
    write_small(tmp_path)

    def change(obj):
        for k, dataset in enumerate(obj["multiscales"][0]["datasets"]):
            if scales is None:
                del dataset["coordinateTransformations"]
            else:
                dataset["coordinateTransformations"] = [{"type": "scale", "scale": scales[k]},
                                                        {"type": "translation", "translation": [0, 1e-12, 0.0]}]
    edit_json(tmp_path / ".zattrs", change)
    volume = open_volume(LocalStore(tmp_path), SMALL)
    recorded = [lv["scale"] for lv in volume.describe()["levels"]]
    assert recorded == ([None] * 3 if scales is None else [[float(v) for v in s] for s in scales])


@pytest.mark.parametrize("compressor", [None, blosc("lz4"), blosc("lz4hc"), blosc("zstd", 2), blosc("zlib", 0),
                                        blosc("blosclz"), {"id": "zstd", "level": 3}, {"id": "zlib", "level": 9},
                                        {"id": "gzip", "level": 1}, {"id": "lz4", "acceleration": 1}])
def test_every_allow_listed_compressor_decodes(tmp_path, compressor):
    data = (np.arange(6 * 5 * 3).reshape(6, 5, 3) * 1031 % 65536).astype("<u2")
    write_volume(tmp_path, [data], [dict(chunks=(4, 4, 2), compressor=compressor, edge="cropped")])
    frame = VolumeFrame("20241024131838", (6, 5, 3), 7.91, "test")
    array, _ = open_volume(LocalStore(tmp_path), frame).read((0, 0, 0), (6, 5, 3))
    assert array.dtype == np.dtype("<u2") and np.array_equal(array, data)


@pytest.mark.parametrize("scale0", [1e-10, 7.91e-6, 7.91, 3e5])
def test_dyadic_scales_and_zero_translations_pass_at_any_magnitude(tmp_path, scale0):
    write_small(tmp_path)

    def change(obj):
        for k, ds in enumerate(obj["multiscales"][0]["datasets"]):
            ds["coordinateTransformations"] = [{"type": "scale", "scale": [scale0 * 2 ** k] * 3},
                                               {"type": "translation", "translation": [0, scale0 * 1e-12, -0.0]}]
    edit_json(tmp_path / ".zattrs", change)
    volume = open_volume(LocalStore(tmp_path), SMALL)
    assert [lv.scale for lv in volume.levels] == [(scale0 * 2 ** k,) * 3 for k in range(3)]


@pytest.mark.parametrize("fill", [0.0, 7.0, 255.0])
def test_an_integral_float_fill_value_is_read_as_the_integer_and_recorded_as_declared(tmp_path, fill):
    levels = small_levels()
    write_volume(tmp_path, levels, [dict(chunks=(4, 4, 4), fill_value=fill, absent=((1, 1, 1),)),
                                    dict(chunks=(4, 4, 4)), dict(chunks=(2, 2, 2))])
    volume = open_volume(LocalStore(tmp_path), SMALL)
    described = volume.describe()["levels"][0]
    assert described["fill_value"] == int(fill) and type(described["fill_value"]) is int
    assert described["fill_value_declared"] == fill and type(described["fill_value_declared"]) is float
    array, record = volume.read((4, 4, 4), (8, 7, 5))
    assert (array == int(fill)).all() and record.absent_chunks == ("0/1.1.1",) and record.fill_value == int(fill)


def test_a_separator_that_disagrees_with_the_stored_layout_shows_in_the_record(tmp_path):
    # Nested chunks whose .zarray omits dimension_separator (zarr's NestedDirectoryStore before 2.8): the '.'
    # default names keys that do not exist, so the read is all fill, and the record says which chunks.
    levels = small_levels()
    write_volume(tmp_path, levels, [dict(chunks=(4, 4, 4), separator="/", declare_separator=False),
                                    dict(chunks=(4, 4, 4)), dict(chunks=(2, 2, 2))])
    volume = open_volume(LocalStore(tmp_path), SMALL)
    array, record = volume.read((0, 0, 0), (9, 7, 5), 0)
    assert not array.any() and levels[0].all()
    assert record.chunks == 12 and record.fill_value == 0
    assert record.absent_chunks == tuple(f"0/{z}.{y}.{x}" for z in range(3) for y in range(2) for x in range(2))
    assert volume.describe()["levels"][0]["separator_declared"] is False
    stored, ok = volume.read((0, 0, 0), (9, 7, 5), 1)
    assert np.array_equal(stored, levels[1]) and ok.absent_chunks == () and ok.chunks == 2


def test_raw_chunks_need_no_codec_library(tmp_path, monkeypatch):
    write_small(tmp_path)
    monkeypatch.setitem(sys.modules, "numcodecs", None)
    array, _ = open_volume(LocalStore(tmp_path), SMALL).read((0, 0, 0), (9, 7, 5), 2)
    assert np.array_equal(array, small_levels()[2])


def test_chunk_padding_tolerance_accepts_padding_within_one_chunk_and_records_it(tmp_path):
    levels = small_levels()
    padded = np.zeros((6, 4, 3), dtype="|u1")
    padded[:5] = levels[1]
    write_volume(tmp_path, [levels[0], padded, levels[2]],
                 [dict(chunks=(4, 4, 4)), dict(chunks=(4, 4, 4)), dict(chunks=(2, 2, 2))])
    volume = open_volume(LocalStore(tmp_path), SMALL, pyramid_tolerance="chunk-padding")
    assert [lv["padding"] for lv in volume.describe()["levels"]] == [[0, 0, 0], [1, 0, 0], [0, 0, 0]]
    assert volume.describe()["pyramid_tolerance"] == "chunk-padding"
    array, record = volume.read((0, 0, 0), (9, 7, 5), 1)
    assert np.array_equal(array, levels[1]) and record.level_stop == (5, 4, 3)


def test_fill_values_and_separators_are_read_per_level(tmp_path):
    levels = small_levels()
    write_volume(tmp_path, levels, [dict(chunks=(4, 4, 4), fill_value=7, absent=((1, 1, 1),)),
                                    dict(chunks=(4, 4, 4), separator="/", fill_value=255, absent=((0, 0, 0),)),
                                    dict(chunks=(2, 2, 2), fill_value=None)])
    volume = open_volume(LocalStore(tmp_path), SMALL)
    level0, _ = volume.read((0, 0, 0), (9, 7, 5), 0)
    assert (level0[4:8, 4:7, 4:5] == 7).all() and np.array_equal(level0[:4], levels[0][:4])
    level1, _ = volume.read((0, 0, 0), (9, 7, 5), 1)
    assert (level1[0:4, 0:4, 0:3] == 255).all() and np.array_equal(level1[4:], levels[1][4:])
    assert chunk_path(tmp_path, "1", (1, 0, 0), "/").exists()


def test_a_full_size_edge_chunk_is_cropped_to_the_array_edge(tmp_path):
    levels = small_levels()
    write_volume(tmp_path / "v", levels, [dict(chunks=(4, 4, 4), edge="full", fill_value=99),
                                          dict(chunks=(4, 4, 4), edge="cropped"), dict(chunks=(2, 2, 2))])
    assert (tmp_path / "v" / "0" / "2.1.1").stat().st_size == 64  # stored padded with 99s
    array, _ = open_volume(LocalStore(tmp_path / "v"), SMALL).read((8, 4, 4), (9, 7, 5))
    assert np.array_equal(array, levels[0][8:9, 4:7, 4:5]) and 99 not in array
