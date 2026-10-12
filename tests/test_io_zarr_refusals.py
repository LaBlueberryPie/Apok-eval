"""Every OME-Zarr refusal fires on its own fixture, with its own code, naming its cause.

Each case starts from a valid 3-level uint8 volume (9, 7, 5) -> (5, 4, 3) -> (3, 2, 2)
written by the tests' own writer, damages one thing, and asserts the refusal code
and the details that name the offending object, field, level or value.
"""

from __future__ import annotations

import json
import os
import shutil
import sys

import numpy as np
import pytest
from io_zarr_fixtures import edit_json, small_levels, write_small, write_volume

from apok_eval.io.frame import VolumeFrame
from apok_eval.io.omezarr import open_volume
from apok_eval.io.store import LocalStore
from apok_eval.refusal import Refusal

SMALL = VolumeFrame("20241024131838", (9, 7, 5), 7.91, "test:small")


@pytest.fixture
def root(tmp_path):
    return write_small(tmp_path / "v.zarr")


def refusal(fn, *args, **kwargs) -> Refusal:
    with pytest.raises(Refusal) as exc:
        fn(*args, **kwargs)
    return exc.value


def open_small(root, **kwargs):
    return open_volume(LocalStore(root), SMALL, **kwargs)


def multiscale(change):
    """A damage: apply ``change`` to multiscales[0] in .zattrs (its return value is ignored)."""
    def apply(obj):
        change(obj["multiscales"][0])
    return lambda root: edit_json(root / ".zattrs", apply)


def dataset(level, transforms):
    def change(ms):
        ms["datasets"][level]["coordinateTransformations"] = transforms
    return multiscale(change)


def scales_and_translation(scale0, translation):
    """A damage: dyadic scales from ``scale0``, and ``translation`` on level 0."""
    def change(ms):
        for k, ds in enumerate(ms["datasets"]):
            ds["coordinateTransformations"] = [{"type": "scale", "scale": [scale0 * 2 ** k] * 3}]
        ms["datasets"][0]["coordinateTransformations"].append({"type": "translation", "translation": translation})
    return multiscale(change)


def scales(*values):
    """A damage: give level k the isotropic scale values[k]."""
    def change(ms):
        for k, value in enumerate(values):
            ms["datasets"][k]["coordinateTransformations"] = [{"type": "scale", "scale": [value] * 3}]
    return multiscale(change)


def array(level, **fields):
    """A damage: set (or, for the value ``DELETE``, remove) fields of ``<level>/.zarray``."""
    def change(obj):
        for name, value in fields.items():
            if value is DELETE:
                del obj[name]
            else:
                obj[name] = value
    return lambda root: edit_json(root / str(level) / ".zarray", change)


def raw(key, data: bytes):
    """A damage: overwrite the object at ``key`` with ``data``."""
    return lambda root: root.joinpath(*key.split("/")).write_bytes(data)


def remove(key):
    return lambda root: root.joinpath(*key.split("/")).unlink()


DELETE = object()


def nested_list(depth: int) -> list:
    value: list = []
    for _ in range(depth - 1):
        value = [value]
    return value

SCALE = lambda s: {"type": "scale", "scale": s}  # noqa: E731
TRANSLATION = lambda t: {"type": "translation", "translation": t}  # noqa: E731


# --- the named negative fixtures ----------------------------------------------------------------

NAMED = {
    "non-dyadic scale": (dataset(1, [SCALE([3.0, 3.0, 3.0])]), "IO_ZARR_SCALE_NOT_DYADIC",
                         {"level": 1, "scale": [3.0, 3.0, 3.0], "expected": 2.0}),
    "non-zero translation": (dataset(2, [SCALE([4.0] * 3), TRANSLATION([0, 0.5, 0])]), "IO_ZARR_TRANSLATION",
                             {"level": 2, "translation": [0.0, 0.5, 0.0]}),
    "wrong axis order": (multiscale(lambda ms: ms.update(axes=["x", "y", "z"])), "IO_ZARR_AXES",
                         {"key": ".zattrs", "field": "multiscales[0].axes", "axes": ["x", "y", "z"]}),
    "missing axes without declaration": (multiscale(lambda ms: ms.pop("axes")), "IO_ZARR_AXES_UNDECLARED",
                                         {"field": "multiscales[0].axes"}),
    "vcz1 compressor": (array(1, compressor={"id": "vcz1"}), "IO_CODEC_UNSUPPORTED",
                        {"key": "1/.zarray", "codec": "vcz1"}),
    "a filters entry": (array(0, filters=[{"id": "delta", "dtype": "|u1"}]), "IO_ZARR_FILTERS",
                        {"key": "0/.zarray", "value": [{"id": "delta", "dtype": "|u1"}]}),
    "F order": (array(2, order="F"), "IO_ZARR_ORDER", {"key": "2/.zarray", "value": "F"}),
    "big-endian dtype": (array(0, dtype=">u2"), "IO_ZARR_DTYPE", {"key": "0/.zarray", "value": ">u2"}),
    "bad chunk size": (raw("0/0.0.0", bytes(63)), "IO_ZARR_CHUNK_SIZE",
                       {"key": "0/0.0.0", "size": 63, "expected": [64]}),
    "shape mismatch with the declared frame": (array(0, shape=[9, 7, 6]), "IO_ZARR_SHAPE_MISMATCH",
                                               {"stored": [9, 7, 6], "declared": [9, 7, 5]}),
    "non-ceil-halving level shape": (array(1, shape=[5, 4, 2]), "IO_ZARR_PYRAMID_SHAPE",
                                     {"level": 1, "axis": "x", "expected": [5, 4, 3]}),
}


# A table-consistency check, not a test of the reader: each named fixture has its own code.
assert len({code for _, code, _ in NAMED.values()}) == len(NAMED)


@pytest.mark.parametrize("name", sorted(NAMED))
def test_named_negative_fixture_is_refused_with_its_code_and_cause(root, name):
    damage, code, details = NAMED[name]
    open_small(root).read((0, 0, 0), (9, 7, 5))  # undamaged, the fixture reads
    damage(root)
    exc = refusal(lambda: open_small(root).read((0, 0, 0), (9, 7, 5)))
    assert exc.code == code, exc.as_dict()
    assert {k: exc.details.get(k) for k in details} == details


def test_the_vcz1_refusal_says_it_is_a_repo_private_codec(root):
    array(1, compressor={"id": "vcz1"})(root)
    assert "repo-private" in refusal(open_small, root).reason


@pytest.mark.parametrize("data, details, reason", [
    (b"{", {"key": ".zgroup", "problem": "invalid JSON: Expecting property name enclosed in double quotes at line 1 "
                                       "column 2"},
     ".zgroup is not strict UTF-8 JSON: invalid JSON: Expecting property name enclosed in double quotes at line 1 "
     "column 2"),
    (b"[2]", {"key": ".zgroup"}, ".zgroup holds a JSON list, not an object"),
], ids=["parse-failure", "not-an-object"])
def test_object_not_json_carries_the_problem_only_for_a_parse_failure(root, data, details, reason):
    # docs/io-zarr.md lists `problem` as present for a parse failure only; a top level that is not an object is
    # named in the reason.
    raw(".zgroup", data)(root)
    exc = refusal(open_small, root)
    assert exc.code == "IO_OBJECT_NOT_JSON" and exc.details == details and exc.reason == reason


# --- every other refusal -----------------------------------------------------------------

CASES = {
    # strict JSON
    "zgroup not json": (raw(".zgroup", b"{"), "IO_OBJECT_NOT_JSON", {"key": ".zgroup"}),
    "zgroup duplicate key": (raw(".zgroup", b'{"zarr_format": 2, "zarr_format": 3}'), "IO_OBJECT_NOT_JSON",
                             {"key": ".zgroup", "problem": "duplicate key 'zarr_format'"}),
    "zattrs NaN": (raw(".zattrs", b'{"multiscales": NaN}'), "IO_OBJECT_NOT_JSON", {"key": ".zattrs"}),
    "zarray overflowing float": (raw("0/.zarray", b'{"fill_value": 1e400}'), "IO_OBJECT_NOT_JSON",
                                 {"key": "0/.zarray"}),
    "zarray not utf-8": (raw("0/.zarray", b'{"dtype": "\xff"}'), "IO_OBJECT_NOT_JSON", {"key": "0/.zarray"}),
    "zarray with a BOM": (raw("0/.zarray", b'\xef\xbb\xbf{}'), "IO_OBJECT_NOT_JSON", {"key": "0/.zarray"}),
    "zgroup too deeply nested": (raw(".zgroup", b"[" * 100_000), "IO_OBJECT_NOT_JSON", {"key": ".zgroup"}),
    # Python's json read this (an ignored member 300 levels deep) and the volume opened.
    "zattrs nested deeper than 256 levels": (multiscale(lambda ms: ms.update(extra=nested_list(300))),
                                             "IO_OBJECT_NOT_JSON",
                                             {"key": ".zattrs",
                                              "problem": "arrays and objects nested deeper than 256 levels"}),
    "zarray integer past int()'s digit limit": (raw("0/.zarray", b'{"fill_value": ' + b"1" * 5000 + b"}"),
                                                "IO_OBJECT_NOT_JSON", {"key": "0/.zarray"}),
    "zgroup a list": (raw(".zgroup", b"[2]"), "IO_OBJECT_NOT_JSON", {"key": ".zgroup"}),
    "array zattrs not json": (raw("2/.zattrs", b"{'a': 1}"), "IO_OBJECT_NOT_JSON", {"key": "2/.zattrs"}),
    # a lone-surrogate escape parses, but is not Unicode (describe() could not be written as UTF-8)
    "zattrs lone surrogate in a value": (multiscale(lambda ms: ms.update(name="\ud800")), "IO_OBJECT_NOT_JSON",
                                         {"key": ".zattrs"}),
    "zarray lone surrogate in a key": (array(0, compressor={"id": "zlib", "level": 1, "\udfff": 0}),
                                       "IO_OBJECT_NOT_JSON", {"key": "0/.zarray"}),
    "array zattrs lone surrogate in a list": (raw("1/.zattrs", b'{"_ARRAY_DIMENSIONS": ["z", "y", "\\udc00"]}'),
                                              "IO_OBJECT_NOT_JSON", {"key": "1/.zattrs"}),
    # group
    "no zgroup": (remove(".zgroup"), "IO_ZARR_GROUP", {"key": ".zgroup"}),
    "zgroup format 3": (raw(".zgroup", b'{"zarr_format": 3}'), "IO_ZARR_GROUP", {"field": "zarr_format", "value": 3}),
    "zgroup format true": (raw(".zgroup", b'{"zarr_format": true}'), "IO_ZARR_GROUP", {"value": True}),
    "zgroup format 2.0": (raw(".zgroup", b'{"zarr_format": 2.0}'), "IO_ZARR_GROUP", {"value": 2.0}),
    "zgroup format missing": (raw(".zgroup", b"{}"), "IO_ZARR_GROUP", {"value": None}),
    # multiscales
    "no zattrs": (remove(".zattrs"), "IO_ZARR_MULTISCALES", {"key": ".zattrs", "field": ".zattrs"}),
    "no multiscales": (raw(".zattrs", b"{}"), "IO_ZARR_MULTISCALES", {"field": "multiscales"}),
    "empty multiscales": (raw(".zattrs", b'{"multiscales": []}'), "IO_ZARR_MULTISCALES", {"field": "multiscales"}),
    "multiscales of numbers": (raw(".zattrs", b'{"multiscales": [1]}'), "IO_ZARR_MULTISCALES",
                               {"field": "multiscales"}),
    "multiscales an object": (raw(".zattrs", b'{"multiscales": {"datasets": []}}'), "IO_ZARR_MULTISCALES",
                              {"field": "multiscales"}),
    "no datasets": (multiscale(lambda ms: ms.pop("datasets")), "IO_ZARR_MULTISCALES",
                    {"field": "multiscales[0].datasets"}),
    "empty datasets": (multiscale(lambda ms: ms.update(datasets=[])), "IO_ZARR_MULTISCALES",
                       {"field": "multiscales[0].datasets"}),
    "dataset path a number": (multiscale(lambda ms: ms["datasets"][0].update(path=0)), "IO_ZARR_MULTISCALES",
                              {"field": "multiscales[0].datasets[0].path"}),
    "datasets as strings": (multiscale(lambda ms: ms.update(datasets=["0", "1", "2"])), "IO_ZARR_MULTISCALES",
                            {"field": "multiscales[0].datasets[0].path"}),
    # dataset paths
    "path with leading zero": (multiscale(lambda ms: ms["datasets"][1].update(path="01")),
                               "IO_ZARR_DATASET_PATHS", {"index": 1, "path": "01", "expected": "1"}),
    "path gap": (multiscale(lambda ms: ms["datasets"][2].update(path="3")), "IO_ZARR_DATASET_PATHS",
                 {"index": 2, "path": "3"}),
    "path duplicate": (multiscale(lambda ms: ms["datasets"][1].update(path="0")), "IO_ZARR_DATASET_PATHS",
                       {"index": 1, "path": "0"}),
    "paths reordered": (multiscale(lambda ms: ms["datasets"].reverse()), "IO_ZARR_DATASET_PATHS",
                        {"index": 0, "path": "2"}),
    "33 datasets": (multiscale(lambda ms: ms.update(datasets=[{"path": str(k)} for k in range(33)])),
                    "IO_ZARR_DATASET_PATHS", {"field": "multiscales[0].datasets", "count": 33, "limit": 32}),
    # 2**1024 does not fit a float: before the bound, the dyadic check crashed with OverflowError
    "1025 dyadic datasets": (multiscale(lambda ms: ms.update(datasets=[
        {"path": str(k), "coordinateTransformations": [SCALE([1e-300 * 2.0 ** min(k, 1023)] * 3)]}
        for k in range(1025)])), "IO_ZARR_DATASET_PATHS", {"count": 1025, "limit": 32}),
    # transforms
    "multiscales-level transforms": (multiscale(lambda ms: ms.update(coordinateTransformations=[SCALE([1, 1, 1])])),
                                     "IO_ZARR_TRANSFORM", {"field": "multiscales[0].coordinateTransformations",
                                                           "rule": "per-dataset-only"}),
    "transforms an object": (dataset(1, {"type": "scale"}), "IO_ZARR_TRANSFORM",
                             {"field": "multiscales[0].datasets[1].coordinateTransformations",
                              "rule": "non-empty-list"}),
    "transforms empty": (dataset(1, []), "IO_ZARR_TRANSFORM",
                         {"field": "multiscales[0].datasets[1].coordinateTransformations", "rule": "non-empty-list"}),
    "affine transform": (dataset(1, [{"type": "affine", "affine": [1] * 12}]), "IO_ZARR_TRANSFORM",
                         {"rule": "type", "type": "affine"}),
    "repeated scale": (dataset(1, [SCALE([2, 2, 2]), SCALE([2, 2, 2])]), "IO_ZARR_TRANSFORM",
                       {"rule": "type", "type": "scale"}),
    "transform not an object": (dataset(1, ["scale"]), "IO_ZARR_TRANSFORM", {"rule": "type", "type": None}),
    "scale of two": (dataset(1, [SCALE([2, 2])]), "IO_ZARR_TRANSFORM",
                     {"rule": "three-finite-numbers", "type": "scale"}),
    "scale with a string": (dataset(1, [SCALE([2, 2, "2"])]), "IO_ZARR_TRANSFORM",
                            {"rule": "three-finite-numbers", "type": "scale"}),
    "scale with a bool": (dataset(1, [SCALE([2, 2, True])]), "IO_ZARR_TRANSFORM",
                          {"rule": "three-finite-numbers", "type": "scale"}),
    "scale a number": (dataset(1, [SCALE(2)]), "IO_ZARR_TRANSFORM", {"rule": "three-finite-numbers", "type": "scale"}),
    "scale overflowing int": (dataset(1, [SCALE([10 ** 400] * 3)]), "IO_ZARR_TRANSFORM",
                              {"rule": "three-finite-numbers", "type": "scale"}),
    "translation of two": (dataset(1, [SCALE([2, 2, 2]), TRANSLATION([0, 0])]), "IO_ZARR_TRANSFORM",
                           {"rule": "three-finite-numbers", "type": "translation"}),
    "translation only": (dataset(1, [TRANSLATION([0, 0, 0])]), "IO_ZARR_TRANSFORM",
                         {"field": "multiscales[0].datasets[1].coordinateTransformations", "rule": "one-scale"}),
    "transforms on some levels only": (multiscale(lambda ms: ms["datasets"][2].pop("coordinateTransformations")),
                                       "IO_ZARR_TRANSFORM", {"rule": "all-or-none", "level": 2}),
    "translation tiny but not zero": (dataset(0, [SCALE([1, 1, 1]), TRANSLATION([0, 0, 1e-6])]),
                                      "IO_ZARR_TRANSLATION", {"level": 0}),
    # a hundredth of a voxel at a tiny scale: zero by an absolute 1e-9, not by the voxel-relative rule
    "translation a hundredth voxel at a tiny scale": (scales_and_translation(1e-10, [0, 1e-12, 0]),
                                                      "IO_ZARR_TRANSLATION",
                                                      {"level": 0, "translation": [0.0, 1e-12, 0.0]}),
    # scales
    "zero scale": (dataset(0, [SCALE([0, 0, 0])]), "IO_ZARR_SCALE", {"level": 0, "rule": "positive"}),
    "negative scale": (dataset(1, [SCALE([2, -2, 2])]), "IO_ZARR_SCALE", {"level": 1, "rule": "positive"}),
    "anisotropic scale": (dataset(1, [SCALE([2, 2, 2.2])]), "IO_ZARR_SCALE", {"level": 1, "rule": "isotropic"}),
    "slightly anisotropic scale": (dataset(0, [SCALE([1, 1.05, 1])]), "IO_ZARR_SCALE",
                                   {"level": 0, "rule": "isotropic"}),
    "scale doubling overflows": (scales(1e308, 1.7e308, 1.7e308), "IO_ZARR_SCALE_NOT_DYADIC", {"level": 1}),
    "level 2 not 4x": (dataset(2, [SCALE([3.0] * 3)]), "IO_ZARR_SCALE_NOT_DYADIC", {"level": 2, "expected": 4.0}),
    # below magnitude 1 villa's tolerance is absolute; apok-eval's is relative at every magnitude
    "tiny scales tripling": (scales(1e-10, 3e-10, 9e-10), "IO_ZARR_SCALE_NOT_DYADIC",
                             {"level": 1, "scale": [3e-10] * 3, "expected": 2e-10}),
    "metre scales off by 1e-6": (scales(7.91e-6, 2 * 7.91e-6 * (1 + 1e-6), 4 * 7.91e-6), "IO_ZARR_SCALE_NOT_DYADIC",
                                 {"level": 1}),
    "tiny anisotropic scale": (dataset(0, [SCALE([1e-10, 5e-10, 1e-10])]), "IO_ZARR_SCALE",
                               {"level": 0, "rule": "isotropic"}),
    # axes
    "two axes": (multiscale(lambda ms: ms.update(axes=["z", "y"])), "IO_ZARR_AXES", {"axes": ["z", "y"]}),
    "four axes": (multiscale(lambda ms: ms.update(axes=["c", "z", "y", "x"])), "IO_ZARR_AXES",
                  {"axes": ["c", "z", "y", "x"]}),
    "axes a string": (multiscale(lambda ms: ms.update(axes="zyx")), "IO_ZARR_AXES", {"axes": "zyx"}),
    "axis name not a string": (multiscale(lambda ms: ms.update(axes=[{"name": 0}, {"name": "y"}, {"name": "x"}])),
                               "IO_ZARR_AXES", {"field": "multiscales[0].axes"}),
    "array _ARRAY_DIMENSIONS": (raw("1/.zattrs", b'{"_ARRAY_DIMENSIONS": ["x", "y", "z"]}'), "IO_ZARR_AXES",
                                {"key": "1/.zattrs", "field": "_ARRAY_DIMENSIONS"}),
    "array dimension_names": (raw("2/.zattrs", b'{"dimension_names": ["z", "x", "y"]}'), "IO_ZARR_AXES",
                              {"key": "2/.zattrs", "field": "dimension_names"}),
    "array axes": (raw("0/.zattrs", b'{"axes": ["y", "z", "x"]}'), "IO_ZARR_AXES",
                   {"key": "0/.zattrs", "field": "axes"}),
    # arrays
    "no zarray": (remove("2/.zarray"), "IO_ZARR_ARRAY_META", {"key": "2/.zarray", "field": ".zarray"}),
    "zarray lacks order": (array(1, order=DELETE), "IO_ZARR_ARRAY_META", {"key": "1/.zarray", "field": "order"}),
    "zarray lacks filters": (array(0, filters=DELETE), "IO_ZARR_ARRAY_META", {"field": "filters"}),
    "zarray lacks fill_value": (array(0, fill_value=DELETE), "IO_ZARR_ARRAY_META", {"field": "fill_value"}),
    "zarray format 3": (array(1, zarr_format=3), "IO_ZARR_ARRAY_META", {"field": "zarr_format", "value": 3}),
    "zarray format true": (array(1, zarr_format=True), "IO_ZARR_ARRAY_META", {"field": "zarr_format"}),
    "shape of two": (array(0, shape=[9, 7]), "IO_ZARR_ARRAY_META", {"field": "shape", "value": [9, 7]}),
    "shape with zero": (array(0, shape=[9, 7, 0]), "IO_ZARR_ARRAY_META", {"field": "shape"}),
    "shape with float": (array(0, shape=[9, 7, 5.0]), "IO_ZARR_ARRAY_META", {"field": "shape"}),
    "chunks with zero": (array(1, chunks=[4, 0, 4]), "IO_ZARR_ARRAY_META", {"field": "chunks", "key": "1/.zarray"}),
    "chunks with bool": (array(1, chunks=[4, 4, True]), "IO_ZARR_ARRAY_META", {"field": "chunks"}),
    "float dtype": (array(0, dtype="<f4"), "IO_ZARR_DTYPE", {"value": "<f4"}),
    "signed dtype": (array(0, dtype="|i1"), "IO_ZARR_DTYPE", {"value": "|i1"}),
    "uint8 spelled <u1": (array(0, dtype="<u1"), "IO_ZARR_DTYPE", {"value": "<u1"}),
    "dtype a list": (array(0, dtype=["|u1"]), "IO_ZARR_DTYPE", {"value": ["|u1"]}),
    "dtype differs between levels": (array(1, dtype="<u2"), "IO_ZARR_DTYPE",
                                     {"key": "1/.zarray", "field": "dtype", "value": "<u2"}),
    "order null": (array(0, order=None), "IO_ZARR_ORDER", {"value": None}),
    "filters an object": (array(0, filters={}), "IO_ZARR_FILTERS", {"value": {}}),
    "separator dash": (array(0, dimension_separator="-"), "IO_ZARR_SEPARATOR", {"key": "0/.zarray", "value": "-"}),
    "separator null": (array(0, dimension_separator=None), "IO_ZARR_SEPARATOR", {"value": None}),
    "fill negative": (array(0, fill_value=-1), "IO_ZARR_FILL", {"key": "0/.zarray", "value": -1}),
    "fill beyond uint8": (array(0, fill_value=256), "IO_ZARR_FILL", {"value": 256}),
    "fill a fraction": (array(0, fill_value=1.5), "IO_ZARR_FILL", {"value": 1.5}),
    "fill a float beyond uint8": (array(0, fill_value=256.0), "IO_ZARR_FILL", {"value": 256.0}),
    "fill a negative float": (array(0, fill_value=-1.0), "IO_ZARR_FILL", {"value": -1.0}),
    "fill a string": (array(0, fill_value="0"), "IO_ZARR_FILL", {"value": "0"}),
    "fill a bool": (array(0, fill_value=False), "IO_ZARR_FILL", {"value": False}),
    # codecs
    "vc-delta3d": (array(0, compressor={"id": "vc-delta3d"}), "IO_CODEC_UNSUPPORTED", {"codec": "vc-delta3d"}),
    "c3d": (array(0, compressor={"id": "c3d"}), "IO_CODEC_UNSUPPORTED", {"codec": "c3d"}),
    "bz2": (array(0, compressor={"id": "bz2", "level": 1}), "IO_CODEC_UNSUPPORTED", {"codec": "bz2"}),
    "compressor a string": (array(0, compressor="blosc"), "IO_CODEC_UNSUPPORTED", {"codec": None}),
    "compressor without id": (array(0, compressor={"cname": "lz4"}), "IO_CODEC_UNSUPPORTED", {"codec": None}),
    "compressor id a list": (array(0, compressor={"id": ["blosc"]}), "IO_CODEC_UNSUPPORTED", {"codec": ["blosc"]}),
    "blosc snappy": (array(2, compressor={"id": "blosc", "cname": "snappy"}), "IO_CODEC_UNSUPPORTED",
                     {"key": "2/.zarray", "codec": "blosc", "cname": "snappy"}),
    "blosc without cname": (array(2, compressor={"id": "blosc"}), "IO_CODEC_UNSUPPORTED", {"cname": None}),
    # pyramid
    "padded level in exact mode": (array(1, shape=[6, 4, 3]), "IO_ZARR_PYRAMID_SHAPE",
                                   {"level": 1, "axis": "z", "tolerance": "exact"}),
    "short level": (array(2, shape=[3, 1, 2]), "IO_ZARR_PYRAMID_SHAPE", {"level": 2, "axis": "y"}),
    # chunks, at read time
    "edge chunk of a third size": (raw("0/2.1.1", bytes(10)), "IO_ZARR_CHUNK_SIZE",
                                   {"key": "0/2.1.1", "size": 10, "expected": [3, 64]}),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_each_damage_is_refused_with_its_code_and_cause(root, name):
    damage, code, details = CASES[name]
    damage(root)
    exc = refusal(lambda: open_small(root).read((0, 0, 0), (9, 7, 5), 0))
    assert exc.code == code, exc.as_dict()
    assert {k: exc.details.get(k) for k in details} == details


def test_a_level_beyond_one_chunk_of_padding_is_refused_even_with_the_tolerance(root):
    array(1, shape=[9, 4, 3])(root)  # 4 over ceil(9/2)=5 with chunks of 4
    exc = refusal(open_small, root, pyramid_tolerance="chunk-padding")
    assert exc.code == "IO_ZARR_PYRAMID_SHAPE"
    assert exc.details["level"] == 1 and exc.details["axis"] == "z" and exc.details["tolerance"] == "chunk-padding"
    array(1, shape=[4, 4, 3])(root)  # padding cannot be negative
    assert refusal(open_small, root, pyramid_tolerance="chunk-padding").details["axis"] == "z"


def test_a_missing_codec_library_is_refused_naming_the_codec(tmp_path, monkeypatch):
    write_volume(tmp_path, small_levels(), [dict(chunks=(4, 4, 4)), dict(chunks=(4, 4, 4), compressor={
        "id": "zstd", "level": 1}), dict(chunks=(2, 2, 2))])
    monkeypatch.setitem(sys.modules, "numcodecs", None)
    exc = refusal(open_small, tmp_path)
    assert exc.code == "IO_CODEC_UNAVAILABLE" and exc.details == {"key": "1/.zarray", "codec": "zstd"}


BLOSC_LZ4 = {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1, "blocksize": 0}


@pytest.mark.parametrize("compressor", [BLOSC_LZ4, {"id": "zlib", "level": 1}, {"id": "gzip", "level": 1},
                                        {"id": "zstd", "level": 1}, {"id": "lz4", "acceleration": 1}])
def test_a_chunk_that_fails_to_decode_is_refused_naming_it(tmp_path, compressor):
    write_volume(tmp_path, [np.zeros((16, 8, 8), "|u1")], [dict(chunks=(8, 8, 8), compressor=compressor)])
    frame = (tmp_path / "0" / "1.0.0").read_bytes()
    if compressor["id"] == "blosc":  # a consistent header over a corrupted compressed payload
        assert not frame[2] & 0x02
        damaged = frame[:16] + bytes(b ^ 0x5A for b in frame[16:])
    else:  # zstd and lz4 keep the header that declares the decoded size, so the decoder itself fails
        damaged = frame[:{"zstd": 7, "lz4": 4}.get(compressor["id"], 0)] + b"\x02\x01garbage, not a stream"
    (tmp_path / "0" / "1.0.0").write_bytes(damaged)
    volume = open_volume(LocalStore(tmp_path), VolumeFrame("20241024131838", (16, 8, 8), 7.91, "test"))
    assert not volume.read((0, 0, 0), (8, 8, 8))[0].any()  # the neighbouring chunk still reads
    exc = refusal(volume.read, (8, 0, 0), (9, 1, 1))
    assert exc.code == "IO_CODEC_DECODE"
    assert exc.details["key"] == "0/1.0.0" and exc.details["codec"] == compressor["id"]
    assert "stage" not in exc.details


@pytest.fixture
def blosc_root(tmp_path):
    """The small volume with level 0 in blosc-lz4: its 64-byte chunks are stored uncompressed (MEMCPYED)."""
    write_volume(tmp_path, small_levels(), [dict(chunks=(4, 4, 4), compressor=BLOSC_LZ4),
                                            dict(chunks=(4, 4, 4)), dict(chunks=(2, 2, 2))])
    frame = (tmp_path / "0" / "0.0.0").read_bytes()
    assert frame[2] & 0x02 and len(frame) == 80
    return tmp_path, frame


def forged(frame: bytes, cbytes: int, length: int) -> bytes:
    return frame[:12] + cbytes.to_bytes(4, "little") + frame[16:length]


@pytest.mark.parametrize("damage,size,cbytes", [
    (lambda f: f[:40], 40, 80),                 # truncated: cbytes no longer matches
    (lambda f: f[:10], 10, None),               # shorter than the header
    (lambda f: f + b"\x00", 81, 80),            # padded
    (lambda f: forged(f, 40, 40), 40, 40),      # consistent cbytes, but MEMCPYED needs nbytes + 16
])
def test_a_blosc_frame_whose_header_disagrees_with_the_object_is_refused_before_decoding(blosc_root, damage, size,
                                                                                         cbytes):
    root, frame = blosc_root
    (root / "0" / "0.0.0").write_bytes(damage(frame))
    exc = refusal(open_small(root).read, (0, 0, 0), (1, 1, 1))
    assert exc.code == "IO_CODEC_DECODE"
    assert exc.details == {"key": "0/0.0.0", "codec": "blosc", "stage": "header", "size": size, "cbytes": cbytes}


def test_a_blosc_header_declaring_the_wrong_decoded_size_is_refused_before_decoding(blosc_root):
    import numcodecs

    root, _ = blosc_root
    (root / "0" / "0.0.0").write_bytes(numcodecs.Blosc(cname="lz4").encode(bytes(32)))
    exc = refusal(open_small(root).read, (0, 0, 0), (1, 1, 1))
    assert exc.code == "IO_ZARR_CHUNK_SIZE"
    assert exc.details == {"key": "0/0.0.0", "stage": "header", "size": 32, "expected": [64]}
    assert np.array_equal(open_small(root).read((4, 0, 0), (9, 7, 5))[0], small_levels()[0][4:])


def test_a_chunk_directory_replaced_by_a_dangling_link_is_refused_not_read_as_fill(root):
    shutil.rmtree(root / "1" / "1")  # level 1 is nested ('/'): 1/1 holds the chunks of z-row 1
    os.symlink(root / "1" / "moved-away", root / "1" / "1")
    volume = open_small(root)
    assert np.array_equal(volume.read((0, 0, 0), (8, 7, 5), 1)[0], small_levels()[1][:4])  # z-row 0 reads
    exc = refusal(volume.read, (0, 0, 0), (9, 7, 5), 1)
    assert exc.code == "IO_STORE_NOT_A_FILE"
    assert exc.details["key"] == "1/1/0/0" and exc.details["component"] == "1/1"


def test_up_to_32_levels_open_and_read(tmp_path):
    data = small_levels()[0]
    levels = [data[tuple(slice(None, None, 2 ** k) for _ in range(3))] for k in range(32)]
    write_volume(tmp_path, levels, [dict(chunks=(4, 4, 4))] * 32)
    volume = open_small(tmp_path)
    assert len(volume.levels) == 32 and volume.levels[31].factor == 2 ** 31
    assert volume.describe()["levels"][31]["scale"] == [2.0 ** 31] * 3
    array, record = volume.read((0, 0, 0), (9, 7, 5), 31)
    assert array.shape == (1, 1, 1) and array[0, 0, 0] == data[0, 0, 0] and record.level_stop == (1, 1, 1)


def test_a_chunk_changed_between_reads_is_refused(root):
    volume = open_small(root)
    volume.read((0, 0, 0), (2, 2, 2))
    data = bytearray((root / "0" / "0.0.0").read_bytes())
    data[0] ^= 0xFF
    (root / "0" / "0.0.0").write_bytes(bytes(data))
    exc = refusal(volume.read, (1, 1, 1), (3, 3, 3))
    assert exc.code == "IO_OBJECT_CHANGED" and exc.details["key"] == "0/0.0.0"


# --- arguments ------------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs,field", [
    ({"store": object()}, "store"), ({"store": None}, "store"),
    ({"declared": {"volume_id": "20241024131838", "shape_zyx": (9, 7, 5)}}, "declared"),
    ({"pyramid_tolerance": "loose"}, "pyramid_tolerance"), ({"pyramid_tolerance": None}, "pyramid_tolerance"),
    ({"axes_declared_by": "", "axes_declared": ("z", "y", "x")}, "axes_declared_by"),
    ({"axes_declared_by": "  ", "axes_declared": ("z", "y", "x")}, "axes_declared_by"),
    ({"axes_declared_by": 5, "axes_declared": ("z", "y", "x")}, "axes_declared_by"),
    ({"axes_declared_by": "registry:x"}, "axes_declared"),  # who, without an order: not a declaration
    ({"axes_declared": ("z", "y", "x")}, "axes_declared_by"),  # an order, without who declares it
])
def test_open_arguments_of_the_wrong_kind_are_refused(root, kwargs, field):
    args = {"store": LocalStore(root), "declared": SMALL, **kwargs}
    exc = refusal(open_volume, args.pop("store"), args.pop("declared"), **args)
    assert exc.code == "IO_ZARR_ARGUMENT_INVALID" and exc.details["field"] == field


def test_nothing_is_read_when_an_argument_is_refused(tmp_path):
    exc = refusal(open_volume, LocalStore(tmp_path), SMALL, pyramid_tolerance="loose")
    assert exc.code == "IO_ZARR_ARGUMENT_INVALID"  # the empty directory was never asked for .zgroup


@pytest.mark.parametrize("order", [("x", "y", "z"), ["z", "x", "y"], ("z", "y"), ("z", "y", "x", "c"), "zyx",
                                   ({"name": "z"}, {"name": "y"}, {"name": "x"})])
@pytest.mark.parametrize("store_axes", [None, ("z", "y", "x")])
def test_a_declared_axis_order_other_than_zyx_is_refused(tmp_path, order, store_axes):
    write_small(tmp_path, axes=store_axes)
    exc = refusal(open_small, tmp_path, axes_declared=order, axes_declared_by="registry:test-volume")
    assert exc.code == "IO_ZARR_AXES"
    assert exc.details == {"field": "axes_declared", "axes": repr(order), "declared_by": "registry:test-volume"}


def test_a_declared_order_is_refused_before_the_store_is_read(tmp_path):
    exc = refusal(open_volume, LocalStore(tmp_path), SMALL, axes_declared=("x", "y", "z"), axes_declared_by="r:v")
    assert exc.code == "IO_ZARR_AXES" and exc.details["field"] == "axes_declared"  # tmp_path holds no .zgroup


@pytest.mark.parametrize("start,stop", [
    ((0, 0, 0), (0, 1, 1)), ((2, 2, 2), (1, 3, 3)), ((0, 0), (1, 1)), ((0, 0, 0, 0), (1, 1, 1, 1)),
    ((0.0, 0, 0), (1, 1, 1)), ((True, 0, 0), (2, 2, 2)), (None, (1, 1, 1)), ("abc", "def"),
    ((0, 0, 0), np.array([1.0, 1.0, 1.0])), ((np.bool_(False), 0, 0), (1, 1, 1)),
    # iterables without a (z, y, x) order of their own
    ({4, 0, 2}, (9, 7, 5)), ({2: "a", 0: "b", 1: "c"}, (9, 7, 5)), (range(3), (9, 7, 5)),
    ((0, 0, 0), iter((9, 7, 5))), (np.array(0), (9, 7, 5)), (np.array([[0, 0, 0]]), (9, 7, 5)),
])
def test_malformed_boxes_are_refused(root, start, stop):
    exc = refusal(open_small(root).read, start, stop)
    assert exc.code == "IO_ZARR_BOX_INVALID"
    assert exc.details == {"start": repr(start), "stop": repr(stop)}


def test_numpy_integer_boxes_are_accepted(root):
    array_, record = open_small(root).read(np.array([1, 1, 1]), (np.int64(3), np.int32(3), np.uint8(3)))
    assert array_.shape == (2, 2, 2) and record.level0_start == (1, 1, 1)
    assert all(type(c) is int for c in record.level0_start + record.level0_stop)


@pytest.mark.parametrize("level", [3, -1, "0", True, 1.0, None])
def test_levels_the_volume_lacks_are_refused(root, level):
    exc = refusal(open_small(root).read, (0, 0, 0), (1, 1, 1), level)
    assert exc.code == "IO_ZARR_LEVEL_INVALID" and exc.details == {"level": repr(level), "levels": 3}


@pytest.mark.parametrize("clip", [1, "yes", None])
def test_clip_must_be_a_bool(root, clip):
    exc = refusal(open_small(root).read, (0, 0, 0), (1, 1, 1), clip=clip)
    assert exc.code == "IO_ZARR_ARGUMENT_INVALID" and exc.details["field"] == "clip"


@pytest.mark.parametrize("start,stop", [((-1, 0, 0), (1, 1, 1)), ((0, 0, 0), (10, 7, 5)), ((0, 0, 0), (9, 7, 6)),
                                        ((0, -3, 0), (9, 8, 5))])
def test_reads_outside_the_level0_shape_are_refused_not_clipped(root, start, stop):
    exc = refusal(open_small(root).read, start, stop, 1)
    assert exc.code == "IO_ZARR_BOX_OUTSIDE"
    assert exc.details == {"start": list(start), "stop": list(stop), "shape": [9, 7, 5]}


@pytest.mark.parametrize("start,stop", [((9, 0, 0), (12, 7, 5)), ((-5, 0, 0), (0, 7, 5)), ((0, 0, 5), (9, 7, 9))])
def test_a_box_wholly_outside_is_refused_even_when_clipping(root, start, stop):
    exc = refusal(open_small(root).read, start, stop, clip=True)
    assert exc.code == "IO_ZARR_BOX_OUTSIDE" and exc.details["start"] == list(start)
    assert "wholly outside" in exc.reason


def test_the_fixture_store_is_left_untouched_by_reading(root, tmp_path):
    before = {p.relative_to(root): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}
    copy = tmp_path / "copy"
    shutil.copytree(root, copy)
    volume = open_small(root)
    for level in range(3):
        volume.read((0, 0, 0), (9, 7, 5), level)
    after = {p.relative_to(root): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}
    assert before == after
    assert json.loads((root / ".zattrs").read_text()) == json.loads((copy / ".zattrs").read_text())


# --- decompression bombs: no chunk decodes to more than the full chunk's bytes + 1 --------------------------


def _bomb_volume(tmp_path, compressor, payload: bytes):
    """A 16x8x8 |u1 volume of two 512-byte chunks whose chunk 0/1.0.0 is ``payload`` (already encoded)."""
    write_volume(tmp_path, [np.zeros((16, 8, 8), "|u1")], [dict(chunks=(8, 8, 8), compressor=compressor)])
    (tmp_path / "0" / "1.0.0").write_bytes(payload)
    return open_volume(LocalStore(tmp_path), VolumeFrame("20241024131838", (16, 8, 8), 7.91, "test"))


@pytest.mark.parametrize("codec", ["zlib", "gzip"])
def test_a_zlib_or_gzip_bomb_is_refused_without_inflating_it(tmp_path, codec):
    import gzip
    import tracemalloc
    import zlib

    zeros = bytes(16 << 20)  # 16 MiB of zeros compresses to a few kilobytes
    payload = zlib.compress(zeros, 9) if codec == "zlib" else gzip.compress(zeros, 9)
    del zeros
    volume = _bomb_volume(tmp_path, {"id": codec, "level": 1}, payload)
    tracemalloc.start()
    try:
        exc = refusal(volume.read, (8, 0, 0), (9, 1, 1))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert exc.code == "IO_ZARR_CHUNK_SIZE"
    assert exc.details == {"key": "0/1.0.0", "stage": "decoded", "size_at_least": 513, "expected": [512]}
    assert peak < 1 << 20  # nothing near the 16 MiB the stream would inflate to


@pytest.mark.parametrize("codec", ["zlib", "gzip"])
def test_a_truncated_zlib_or_gzip_stream_is_refused_as_undecodable(tmp_path, codec):
    import gzip
    import zlib

    whole = zlib.compress(bytes(range(256)) * 2) if codec == "zlib" else gzip.compress(bytes(range(256)) * 2)
    exc = refusal(_bomb_volume(tmp_path, {"id": codec, "level": 1}, whole[: len(whole) // 2]).read, (8, 0, 0),
                  (9, 1, 1))
    assert exc.code == "IO_CODEC_DECODE" and exc.details["key"] == "0/1.0.0" and exc.details["codec"] == codec


def _zstd_rle_frame(content_size: int, regenerated: int, descriptor: int = 0xE0) -> bytes:
    """A single-segment zstd frame (8-byte content size by default) of one RLE block of ``regenerated`` zeros."""
    header = _ZSTD_MAGIC_BYTES + bytes([descriptor]) + content_size.to_bytes(8 if descriptor >> 6 == 3 else 4,
                                                                             "little")
    return header + ((regenerated << 3) | (1 << 1) | 1).to_bytes(3, "little") + b"\x00"


_ZSTD_MAGIC_BYTES = b"\x28\xb5\x2f\xfd"


@pytest.mark.parametrize("declared", [1 << 40, 513, 0])
def test_a_zstd_frame_declaring_another_content_size_is_refused_before_decoding(tmp_path, declared):
    volume = _bomb_volume(tmp_path, {"id": "zstd", "level": 1}, _zstd_rle_frame(declared, 512))
    exc = refusal(volume.read, (8, 0, 0), (9, 1, 1))
    assert exc.code == "IO_ZARR_CHUNK_SIZE"
    assert exc.details == {"key": "0/1.0.0", "stage": "header", "size": declared, "expected": [512]}


def test_a_zstd_frame_header_is_read_in_each_of_its_layouts(tmp_path):
    import numcodecs

    # numcodecs' own frames (2-byte content size, + 256) and hand-made 4- and 8-byte ones all read.
    assert numcodecs.Zstd(level=1).encode(bytes(512))[4] >> 6 == 1
    for payload in (numcodecs.Zstd(level=1).encode(bytes(512)), _zstd_rle_frame(512, 512, 0xA0),
                    _zstd_rle_frame(512, 512)):
        assert not _bomb_volume(tmp_path, {"id": "zstd", "level": 1}, payload).read((8, 0, 0), (16, 8, 8))[0].any()
    # A frame whose window descriptor and dictionary id precede the content size: 4-byte size at offset 7.
    framed = _ZSTD_MAGIC_BYTES + bytes([0x81, 0x58, 0x07]) + (512).to_bytes(4, "little")
    exc = refusal(_bomb_volume(tmp_path, {"id": "zstd", "level": 1}, framed + b"\x00" * 4).read, (8, 0, 0),
                  (9, 1, 1))
    assert exc.code == "IO_CODEC_DECODE" and "stage" not in exc.details  # the size was found; the decoder failed


@pytest.mark.parametrize("payload, size", [(b"\x28\xb5\x2f\xfd\x00\x58", 6),  # no content size declared
                                           (b"not zstd at all", 15), (b"", 0)])
def test_a_zstd_object_without_a_declared_content_size_is_refused(tmp_path, payload, size):
    exc = refusal(_bomb_volume(tmp_path, {"id": "zstd", "level": 1}, payload).read, (8, 0, 0), (9, 1, 1))
    assert exc.code == "IO_CODEC_DECODE"
    assert exc.details == {"key": "0/1.0.0", "codec": "zstd", "stage": "header", "size": size}


def test_an_lz4_object_whose_size_prefix_is_not_an_allowed_size_is_refused_before_decoding(tmp_path):
    import numcodecs

    good = numcodecs.LZ4().encode(bytes(512))
    assert int.from_bytes(good[:4], "little") == 512
    for declared in (1 << 31, 1000):
        exc = refusal(_bomb_volume(tmp_path, {"id": "lz4"}, declared.to_bytes(4, "little") + good[4:]).read,
                      (8, 0, 0), (9, 1, 1))
        assert exc.code == "IO_ZARR_CHUNK_SIZE"
        assert exc.details == {"key": "0/1.0.0", "stage": "header", "size": declared, "expected": [512]}
    exc = refusal(_bomb_volume(tmp_path, {"id": "lz4"}, b"\x00\x02").read, (8, 0, 0), (9, 1, 1))
    assert exc.code == "IO_CODEC_DECODE"
    assert exc.details == {"key": "0/1.0.0", "codec": "lz4", "stage": "header", "size": 2}
    assert not _bomb_volume(tmp_path, {"id": "lz4"}, good).read((8, 0, 0), (16, 8, 8))[0].any()


@pytest.mark.parametrize("header, size", [
    (bytes([0x20, 200]), 200),                                     # single segment, 1-byte size
    (bytes([0x60]) + (512 - 256).to_bytes(2, "little"), 512),      # 2-byte size, stored minus 256
    (bytes([0xA0]) + (70000).to_bytes(4, "little"), 70000),        # 4-byte size
    (bytes([0xE0]) + (1 << 40).to_bytes(8, "little"), 1 << 40),    # 8-byte size
    (bytes([0x40, 0x58]) + (300 - 256).to_bytes(2, "little"), 300),  # window descriptor first
    (bytes([0x62, 0x01, 0x02]) + (512 - 256).to_bytes(2, "little"), 512),  # single segment, 2-byte dictionary id
    (bytes([0x83, 0x58, 1, 2, 3, 4]) + (9).to_bytes(4, "little"), 9),  # window, 4-byte dictionary id, size
    (bytes([0x00, 0x58]), None),                                   # no content size declared
    (bytes([0xA0, 1, 2]), None),                                   # header cut short
])
def test_the_zstd_content_size_is_read_from_every_header_layout(header, size):
    from apok_eval.io.omezarr import _zstd_content_size

    assert _zstd_content_size(_ZSTD_MAGIC_BYTES + header) == size
    assert _zstd_content_size(b"\x28\xb5\x2f\xfe" + header) is None  # not the zstd magic
