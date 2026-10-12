"""URI canonicalisation: one form per object, whole-segment prefixes, and a
named refusal for every spelling the gate will not interpret."""

from __future__ import annotations

import os

import pytest

from apok_eval.licence.uri import CanonicalURI, canonical_uri
from apok_eval.refusal import Refusal

KEY = "PHercParis4/volumes/20990101000001-test.zarr"
OPEN_CANON = f"s3://vesuvius-challenge-open-data/{KEY}"


def _refused(uri, what="uri"):
    with pytest.raises(Refusal) as info:
        canonical_uri(uri, what=what)
    return info.value


@pytest.mark.parametrize(
    "spelling",
    [
        f"s3://vesuvius-challenge-open-data/{KEY}",
        f"s3://vesuvius-challenge-open-data/{KEY}/",
        f"S3://vesuvius-challenge-open-data/{KEY}",
        f"https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/{KEY}",
        f"https://vesuvius-challenge-open-data.s3.amazonaws.com/{KEY}",
        f"HTTPS://Vesuvius-Challenge-Open-Data.S3.Amazonaws.com/{KEY}/",
        f"https://s3.amazonaws.com/vesuvius-challenge-open-data/{KEY}",
        f"https://s3.us-east-1.amazonaws.com/vesuvius-challenge-open-data/{KEY}",
        "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/./volumes/x/../20990101000001-test.zarr",
    ],
)
def test_every_open_data_bucket_spelling_has_one_canonical_form(spelling):
    assert str(canonical_uri(spelling)) == OPEN_CANON


@pytest.mark.parametrize(
    "spelling",
    [
        f"https://data.aws.ash2txt.org/samples/{KEY}",
        f"s3://vesuvius-challenge/{KEY}",
        f"https://vesuvius-challenge.s3.amazonaws.com/{KEY}",
        f"https://data.aws.ash2txt.org/samples/x/../{KEY}/",
    ],
)
def test_the_gateway_is_the_vesuvius_challenge_bucket(spelling):
    assert str(canonical_uri(spelling)) == f"s3://vesuvius-challenge/{KEY}"


def test_gateway_mapping_applies_only_under_samples_after_dot_resolution():
    assert str(canonical_uri("https://data.aws.ash2txt.org/samples")) == "s3://vesuvius-challenge"
    assert str(canonical_uri("https://data.aws.ash2txt.org/other/x")) == "https://data.aws.ash2txt.org/other/x"
    # '..' is resolved on the URL path first, as an HTTP client would: this leaves /samples.
    assert str(canonical_uri("https://data.aws.ash2txt.org/samples/../x")) == "https://data.aws.ash2txt.org/x"
    assert str(canonical_uri("https://data.aws.ash2txt.org/samplesX/a")) == "https://data.aws.ash2txt.org/samplesX/a"


def test_data_server_and_other_https_hosts_stay_https():
    c = canonical_uri("https://DL.ash2txt.org/datasets/spiral_datasets/PHercParis4/a/../b/./")
    assert c == CanonicalURI("https", "dl.ash2txt.org", ("datasets", "spiral_datasets", "PHercParis4", "b"))
    assert str(c) == "https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4/b"
    assert str(canonical_uri("https://dl.ash2txt.org/")) == "https://dl.ash2txt.org"
    assert str(canonical_uri("https://wk.aws.ash2txt.org/datasets/x")) == "https://wk.aws.ash2txt.org/datasets/x"
    assert str(canonical_uri("https://amazonaws.com.example.org/x")) == "https://amazonaws.com.example.org/x"
    assert str(canonical_uri("https://host1.example.org/x")) == "https://host1.example.org/x"


@pytest.mark.parametrize(
    "uri, host",
    [
        # Other spellings of S3 objects: refused, never left as a plain https host a registry could label anew.
        ("https://vesuvius-challenge-open-data.s3.dualstack.us-east-1.amazonaws.com/k", None),
        ("https://vesuvius-challenge.s3-accelerate.amazonaws.com/k", None),
        ("https://vesuvius-challenge.s3-accelerate.dualstack.amazonaws.com/k", None),
        ("https://vesuvius-challenge-open-data.s3-external-1.amazonaws.com/k", None),
        ("https://vesuvius-challenge-open-data.s3-us-east-1.amazonaws.com/k", None),
        ("https://vesuvius-challenge.s3-fips.us-east-1.amazonaws.com/k", None),
        ("https://vesuvius-challenge-open-data.s3-website-us-east-1.amazonaws.com/k", None),
        ("https://s3.dualstack.us-east-1.amazonaws.com/vesuvius-challenge-open-data/k", None),
        ("https://s3-external-1.amazonaws.com/vesuvius-challenge-open-data/k", None),
        ("https://s3.amazonaws.com/", "s3.amazonaws.com"),  # a path-style endpoint without a bucket
        ("https://amazonaws.com/vesuvius-challenge/k", "amazonaws.com"),
        # The zarr proxy re-serves both buckets under its own paths.
        ("https://zarr.aws.ash2txt.org/data/PHercParis4/volumes/20990101000002-x.zarr/0/0.0.0", None),
        ("https://ZARR.aws.ash2txt.org/open-data/PHercParis4", "zarr.aws.ash2txt.org"),
        # An IP address may be any host (path-style S3 included).
        ("https://52.217.0.1/vesuvius-challenge-open-data/k", None),
        ("https://1.2.3.4", None),
    ],
)
def test_other_names_for_known_objects_are_refused_as_ambiguous(uri, host):
    exc = _refused(uri, what="inputs[0].uri")
    assert exc.code == "LICENCE_URI_AMBIGUOUS" and exc.details["what"] == "inputs[0].uri"
    assert exc.details["host"] == (host or uri.split("/")[2])


def test_prefixes_compare_whole_segments():
    loc = canonical_uri("s3://vesuvius-challenge-open-data/PHercParis4/")
    assert canonical_uri("s3://vesuvius-challenge-open-data/PHercParis4").is_under(loc)
    assert canonical_uri("s3://vesuvius-challenge-open-data/PHercParis4/volumes/x").is_under(loc)
    assert not canonical_uri("s3://vesuvius-challenge-open-data/PHercParis40/volumes/x").is_under(loc)
    assert not canonical_uri("s3://vesuvius-challenge-open-data/PHercParis").is_under(loc)
    assert not canonical_uri("s3://vesuvius-challenge-open-data-2/PHercParis4/x").is_under(loc)
    assert not canonical_uri("s3://vesuvius-challenge/PHercParis4/x").is_under(loc)
    assert not canonical_uri("https://dl.ash2txt.org/PHercParis4").is_under(canonical_uri("synthetic://dl.ash2txt.org"))
    root = canonical_uri("s3://vesuvius-challenge-open-data")
    assert loc.is_under(root) and not root.is_under(loc)


def test_canonical_form_is_a_fixed_point():
    for spelling in [f"https://data.aws.ash2txt.org/samples/{KEY}/", "http://127.0.0.1:80/a/", "/tmp/../tmp/a",
                     "synthetic://Test-Fixtures/x/.", "https://dl.ash2txt.org/a", "/", "/tmp/..", "file:///"]:
        once = canonical_uri(spelling)
        assert canonical_uri(str(once)) == once


def test_the_filesystem_root_is_written_with_its_slash():
    root = canonical_uri("/")
    assert str(root) == "file:///" and root.segments == () and root.is_local
    assert canonical_uri("/tmp").is_under(root) and str(canonical_uri("/tmp/..")) == "file:///"


def test_local_paths_and_file_uris_are_real_absolute_paths(tmp_path):
    root = tmp_path / "fixtures"
    (root / "a").mkdir(parents=True)
    real_root = os.path.realpath(root)
    expected = canonical_uri(f"file://{real_root}/a")
    assert canonical_uri(str(root / "a")) == expected
    assert canonical_uri(f"file://{root}/b/../a/") == expected
    assert expected.scheme == "file" and expected.authority == "" and str(expected) == f"file://{real_root}/a"
    assert expected.is_local


def test_symlinks_are_judged_by_where_they_point(tmp_path):
    root, outside = tmp_path / "root", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)
    resolved = canonical_uri(str(root / "escape" / "data.zarr"))
    assert not resolved.is_under(canonical_uri(str(root)))
    assert resolved.is_under(canonical_uri(str(outside)))


def test_a_symlink_into_a_path_with_forbidden_characters_is_refused(tmp_path):
    spaced = tmp_path / "has space"
    spaced.mkdir()
    (tmp_path / "link").symlink_to(spaced, target_is_directory=True)
    exc = _refused(str(tmp_path / "link" / "x"), what="inputs[0].uri")
    assert exc.code == "LICENCE_URI_MALFORMED"
    assert exc.details["char"] == " " and exc.details["real"].endswith("has space/x")
    assert exc.details["what"] == "inputs[0].uri"


@pytest.mark.parametrize("char", ["%", "#", "?"])
def test_a_symlink_into_a_path_with_uri_markers_is_refused(tmp_path, char):
    marked = tmp_path / f"a{char}b"
    marked.mkdir()
    (tmp_path / "link").symlink_to(marked, target_is_directory=True)
    exc = _refused(str(tmp_path / "link" / "x"), what="inputs[0].uri")
    assert exc.code == "LICENCE_URI_MALFORMED"
    assert exc.details["char"] == char and exc.details["real"].endswith(f"a{char}b/x")


@pytest.mark.parametrize("length", [3, 63])
def test_bucket_names_of_3_to_63_characters_are_accepted(length):
    name = "b" * length
    assert canonical_uri(f"s3://{name}/k").authority == name


def test_loopback_http_is_canonical_with_its_port():
    assert str(canonical_uri("http://127.0.0.1:8123/store.zarr/")) == "http://127.0.0.1:8123/store.zarr"
    assert str(canonical_uri("http://127.0.0.1:80/x")) == "http://127.0.0.1/x"
    assert str(canonical_uri("http://[::1]:9000/x")) == "http://[::1]:9000/x"
    assert str(canonical_uri("http://127.0.0.2/x")) == "http://127.0.0.2/x"
    assert canonical_uri("http://127.0.0.1:8123/x").is_local
    assert not canonical_uri("http://127.0.0.1:8123/x").is_under(canonical_uri("http://127.0.0.1:8124"))


def test_synthetic_names_are_host_like():
    assert str(canonical_uri("synthetic://Test-Fixtures/spiral/seed-7")) == "synthetic://test-fixtures/spiral/seed-7"
    assert canonical_uri("synthetic://apok-fixtures").is_local
    assert not canonical_uri("s3://vesuvius-challenge-open-data/x").is_local
    assert not canonical_uri("https://dl.ash2txt.org/x").is_local


@pytest.mark.parametrize(
    "uri, code, detail",
    [
        ("", "LICENCE_URI_MALFORMED", {}),
        (None, "LICENCE_URI_MALFORMED", {}),
        (["s3://b/k"], "LICENCE_URI_MALFORMED", {}),
        ("s3://bucket/a key", "LICENCE_URI_MALFORMED", {"char": " "}),
        ("s3://bucket/a\tkey", "LICENCE_URI_MALFORMED", {"char": "\t"}),
        ("s3://bucket/a\u200bkey", "LICENCE_URI_MALFORMED", {"char": "\u200b"}),
        ("C:\\data\\x", "LICENCE_URI_MALFORMED", {"char": "\\"}),
        ("s3://Vesuvius-Challenge/x", "LICENCE_URI_MALFORMED", {"bucket": "Vesuvius-Challenge"}),
        ("s3://a..b/x", "LICENCE_URI_MALFORMED", {"bucket": "a..b"}),
        ("s3://ab/x", "LICENCE_URI_MALFORMED", {"bucket": "ab"}),
        ("s3://" + "b" * 64 + "/x", "LICENCE_URI_MALFORMED", {"bucket": "b" * 64}),
        ("https://" + "b" * 64 + ".s3.amazonaws.com/x", "LICENCE_URI_MALFORMED", {"bucket": "b" * 64}),
        ("https://bad..host/x", "LICENCE_URI_MALFORMED", {"host": "bad..host"}),
        ("https://dl.ash2txt.org./x", "LICENCE_URI_MALFORMED", {"host": "dl.ash2txt.org."}),
        ("https://under_score.org/x", "LICENCE_URI_MALFORMED", {"host": "under_score.org"}),
        ("synthetic://-bad/x", "LICENCE_URI_MALFORMED", {"host": "-bad"}),
        ("https://x.s3.amazonaws.com/k", "LICENCE_URI_MALFORMED", {"bucket": "x"}),
        ("https://s3.amazonaws.com/UPPER/k", "LICENCE_URI_MALFORMED", {"bucket": "UPPER"}),
        ("http://127.0.0.1:0/x", "LICENCE_URI_MALFORMED", {"port": "0"}),
        ("http://127.0.0.1:08080/x", "LICENCE_URI_MALFORMED", {"port": "08080"}),
        ("http://127.0.0.1:70000/x", "LICENCE_URI_MALFORMED", {"port": "70000"}),
        ("file://host/x", "LICENCE_URI_MALFORMED", {}),
        ("file://", "LICENCE_URI_MALFORMED", {}),
        ("s3://vesuvius-challenge-open-data/a%2Fb", "LICENCE_URI_AMBIGUOUS", {"char": "%"}),
        ("https://dl.ash2txt.org/x?version=2", "LICENCE_URI_AMBIGUOUS", {"char": "?"}),
        ("https://dl.ash2txt.org/x.zarr#vc-base-scale=2", "LICENCE_URI_AMBIGUOUS", {"char": "#"}),
        ("/data/a#b", "LICENCE_URI_AMBIGUOUS", {"char": "#"}),
        ("https://evil@dl.ash2txt.org/x", "LICENCE_URI_AMBIGUOUS", {}),
        ("http://user@127.0.0.1/x", "LICENCE_URI_AMBIGUOUS", {}),
        ("https://dl.ash2txt.org:443/x", "LICENCE_URI_AMBIGUOUS", {}),
        ("s3://bucket:9000/x", "LICENCE_URI_AMBIGUOUS", {}),
        ("s3://vesuvius-challenge-open-data/a//b", "LICENCE_URI_AMBIGUOUS", {}),
        ("s3://vesuvius-challenge-open-data/../other-bucket/x", "LICENCE_URI_AMBIGUOUS", {}),
        ("https://data.aws.ash2txt.org/samples/../../x", "LICENCE_URI_AMBIGUOUS", {}),
        ("data/scroll.zarr", "LICENCE_URI_RELATIVE", {}),
        ("dl.ash2txt.org/full-scrolls/x", "LICENCE_URI_RELATIVE", {}),
        ("//dl.ash2txt.org/x", "LICENCE_URI_RELATIVE", {}),
        ("https:dl.ash2txt.org/x", "LICENCE_URI_RELATIVE", {}),
        ("s3:/bucket/x", "LICENCE_URI_RELATIVE", {}),
        ("s3:///key", "LICENCE_URI_RELATIVE", {}),
        ("https:///x", "LICENCE_URI_RELATIVE", {}),
        ("ftp://dl.ash2txt.org/x", "LICENCE_URI_SCHEME", {"scheme": "ftp"}),
        ("zarr2://https://dl.ash2txt.org/x", "LICENCE_URI_SCHEME", {"scheme": "zarr2"}),
        ("s3+us-east-1://bucket/x", "LICENCE_URI_SCHEME", {"scheme": "s3+us-east-1"}),
        ("C:/data/x", "LICENCE_URI_SCHEME", {"scheme": "c"}),
        ("http://dl.ash2txt.org/x", "LICENCE_URI_SCHEME", {"host": "dl.ash2txt.org"}),
        ("http://localhost:8000/x", "LICENCE_URI_SCHEME", {"host": "localhost:8000"}),
        ("http://10.0.0.1/x", "LICENCE_URI_SCHEME", {"host": "10.0.0.1"}),
        ("http://[::2]/x", "LICENCE_URI_SCHEME", {"host": "[::2]"}),
    ],
)
def test_each_uri_refusal_names_its_cause(uri, code, detail):
    exc = _refused(uri, what="registry assets['x'].locations[0]")
    assert exc.code == code
    assert exc.details["what"] == "registry assets['x'].locations[0]"
    assert exc.details["uri"] == uri
    for key, value in detail.items():
        assert exc.details[key] == value
