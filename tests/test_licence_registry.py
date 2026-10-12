"""The asset registry: the shipped registry validates, and every way a
registry can be wrong is refused with its own code and the offending entry."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from licence_fixtures import CC_VOLUME_ID, CC_VOLUME_URI, OPEN, SPIRAL_URI, asset, host, registry_doc, write_json

from apok_eval.canonical import digest_json
from apok_eval.licence import registry as registry_module
from apok_eval.licence.registry import (
    SHIPPED_REGISTRY,
    canonical_sample,
    is_gated,
    load_registry,
    parse_registry,
)
from apok_eval.licence.uri import canonical_uri
from apok_eval.refusal import REGISTRY, Refusal

SRC = Path(registry_module.__file__).resolve().parent


def _refused(doc, source="<in-memory>"):
    with pytest.raises(Refusal) as info:
        parse_registry(doc, source)
    return info.value


# --- the shipped registry --------------------------------------------------------


def test_shipped_registry_validates_and_declares_only_generic_entries():
    reg = load_registry()
    assert reg.source == "apok_eval.licence/registry.json"
    assert set(reg.licences) == {"CC-BY-NC-4.0", "EDUCELAB-SCROLLS", "MIT"}
    assert reg.licences["EDUCELAB-SCROLLS"].publication == "forbidden"
    assert reg.licences["CC-BY-NC-4.0"].publication == "ask"
    assert reg.licences["CC-BY-NC-4.0"].requires_terms == ("VESUVIUS-DATA-TERMS",)  # bucket terms are recorded too
    assert reg.licences["MIT"].publication == "allowed" and reg.licences["MIT"].shippable_in_pipeline
    # No scroll data is declared: one synthetic asset, for the tests (the demo adds its own). Users declare theirs.
    assert list(reg.assets) == ["apok-synthetic-fixtures"]
    assert reg.assets["apok-synthetic-fixtures"].kind == "synthetic"
    assert [str(u) for u in reg.assets["apok-synthetic-fixtures"].locations] == ["synthetic://apok-fixtures"]
    # The two public terms texts, pinned by the sha256 of their bytes; no acceptance is recorded here.
    assert {tid: t.text_sha256 for tid, t in reg.terms.items()} == {
        "DATA-SERVER-LICENSE": "sha256:9dc08a92194617d4ae68fb96d4fd0da99d1c7207f821f2a3750b0034eac8d154",
        "VESUVIUS-DATA-TERMS": "sha256:9f7ce6e4621d7add76c11f3428ee61fda94c6ecce833a324441dbe1711b940b2"}
    assert all(t.per_person for t in reg.terms.values())
    gated = [entry for entry in reg.hosts if is_gated(entry.prefix)]
    assert [str(entry.prefix) for entry in gated] == [
        "s3://vesuvius-challenge-open-data", "s3://vesuvius-challenge", "https://dl.ash2txt.org"]
    for entry in gated:
        assert all(reg.licences[lid].requires_terms for lid in entry.allowed_licences)
    data_server = next(entry for entry in reg.hosts if str(entry.prefix) == "https://dl.ash2txt.org")
    assert data_server.requires_terms == ("DATA-SERVER-LICENSE",)  # whatever licence a narrower entry allows
    assert reg.digest == digest_json(json.loads(SHIPPED_REGISTRY.read_text(encoding="utf-8")))


def test_shipped_registry_records_no_acceptance():
    text = SHIPPED_REGISTRY.read_text(encoding="utf-8")
    assert "terms_accepted_by" not in text


def test_registry_is_immutable_and_deterministic(tmp_path):
    a = parse_registry(registry_doc())
    b = load_registry(write_json(tmp_path / "r.json", registry_doc()))
    assert a.digest == b.digest and b.source == str(tmp_path / "r.json")
    with pytest.raises(TypeError):
        a.licences["X"] = a.licences["MIT"]  # type: ignore[index]
    with pytest.raises(AttributeError):
        a.digest = "x"  # type: ignore[misc]
    assert isinstance(a.assets["spiral-input"].locations, tuple)


def test_host_for_picks_the_longest_declared_prefix():
    doc = registry_doc()
    doc["hosts"]["s3://vesuvius-challenge/PHercParis4/volumes/v.zarr"] = host("CC-BY-NC-4.0")
    reg = parse_registry(doc)
    narrow = reg.host_for(canonical_uri("https://data.aws.ash2txt.org/samples/PHercParis4/volumes/v.zarr/0/0.0.0"))
    wide = reg.host_for(canonical_uri("https://data.aws.ash2txt.org/samples/PHercParis4/volumes/v.zarrX"))
    assert str(narrow.prefix) == "s3://vesuvius-challenge/PHercParis4/volumes/v.zarr"
    assert str(wide.prefix) == "s3://vesuvius-challenge"
    assert reg.host_for(canonical_uri("https://example.org/x")) is None


# --- reading -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "content, problem",
    [
        (None, "NotRegularFile"),
        (b"{not json", "invalid JSON: Expecting property name"),
        (b'{"schema": "a", "schema": "b"}', "duplicate key 'schema'"),
        (b'{"x": NaN}', "non-standard constant NaN"),
        (b'{"x": 1e999}', "number 1e999 overflows a double"),
        (b"\xef\xbb\xbf{}", "byte-order mark"),
        (b'{"x": "\xff"}', "not UTF-8"),
        (b'{"x": "\\ud800"}', "lone surrogate '\\ud800'"),
        (b'{"\\udfff": 1}', "lone surrogate '\\udfff'"),
        (b'{"x": ' + b"1" * 5000 + b"}", "more than int() converts"),
        # Python's json raised RecursionError here, which escaped the reader as a crash.
        (b"[" * 100_000 + b"]" * 100_000, "nested deeper than the parser can recurse"),
    ],
)
def test_unreadable_registry_is_refused_with_the_problem(tmp_path, content, problem):
    path = tmp_path / "registry.json"
    if content is not None:
        path.write_bytes(content)
    with pytest.raises(Refusal) as info:
        load_registry(path)
    assert info.value.code == "LICENCE_REGISTRY_UNREADABLE"
    assert info.value.details["source"] == str(path)
    assert problem in info.value.details["problem"]


@pytest.mark.parametrize("name, problem", [
    ("registry\x00.json", "ValueError: "),  # the message itself depends on the Python version
    ("registry\ud800.json", "UnicodeEncodeError: 'utf-8' codec can't encode character '\\ud800'"),
], ids=["nul", "lone-surrogate"])
def test_a_registry_path_open_refuses_is_unreadable(tmp_path, name, problem):
    # open() raises ValueError, not OSError, for these paths; the reader once let it escape as a crash.
    path = str(tmp_path / name)
    with pytest.raises(Refusal) as info:
        load_registry(path)
    assert info.value.code == "LICENCE_REGISTRY_UNREADABLE"
    assert info.value.details["source"] == path and info.value.details["problem"].startswith(problem)


# --- shape -----------------------------------------------------------------------------


def _set(path, value):
    def mutate(doc):
        *parents, last = path
        node = doc
        for key in parents:
            node = node[key]
        node[last] = value
        return doc
    return mutate


def _drop(path):
    def mutate(doc):
        *parents, last = path
        node = doc
        for key in parents:
            node = node[key]
        del node[last]
        return doc
    return mutate


MALFORMED_CASES = [
    ("not an object", lambda d: [d], {"table": "$"}),
    ("missing table", _drop(("terms",)), {"table": "$", "missing": ["terms"]}),
    ("unknown top-level key", _set(("extra",), {}), {"table": "$", "unknown": ["extra"]}),
    ("table not an object", _set(("licences",), []), {"table": "licences"}),
    ("note not text", _set(("note",), ""), {"table": "note"}),
    # Otherwise-valid entries under a bad id: only the id guard can refuse them.
    ("bad licence id", lambda d: _set(("licences", "bad id"), dict(d["licences"]["MIT"]))(d),
     {"table": "licences", "entry": "bad id", "field": "id"}),
    ("bad asset id", _set(("assets", "-x"), asset("synthetic", ["synthetic://apok-fixtures/x"], "MIT")),
     {"table": "assets", "entry": "-x", "field": "id"}),
    ("id with a newline", lambda d: _set(("terms", "TX\n"), dict(d["terms"]["RESERVATION"]))(d),
     {"table": "terms", "entry": "TX\n", "field": "id"}),
    ("entry not an object", _set(("terms", "TX"), "text"), {"table": "terms", "entry": "TX"}),
    ("missing field", _drop(("licences", "MIT", "evidence")), {"entry": "MIT", "missing": ["evidence"]}),
    ("unknown field", _set(("licences", "MIT", "licnece"), 1), {"entry": "MIT", "unknown": ["licnece"]}),
    ("publication enum", _set(("licences", "MIT", "publication"), "maybe"), {"field": "publication"}),
    ("bool not int", _set(("licences", "MIT", "commercial_use"), 1), {"field": "commercial_use"}),
    ("shippable not bool", _set(("licences", "MIT", "shippable_in_pipeline"), None),
     {"field": "shippable_in_pipeline"}),
    ("duplicate requires_terms", _set(("licences", "CC-BY-NC-4.0", "requires_terms"), ["BUCKET-TERMS"] * 2),
     {"field": "requires_terms"}),
    ("requires_terms not ids", _set(("licences", "CC-BY-NC-4.0", "requires_terms"), "BUCKET-TERMS"),
     {"field": "requires_terms"}),
    ("empty obligation", _set(("licences", "MIT", "obligations"), [""]), {"field": "obligations"}),
    ("blank name", _set(("licences", "MIT", "name"), "  "), {"field": "name"}),
    ("text_sha256 not a digest", _set(("terms", "BUCKET-TERMS", "text_sha256"), "sha256:abc"),
     {"field": "text_sha256"}),
    ("per_person not bool", _set(("terms", "BUCKET-TERMS", "per_person"), "yes"), {"field": "per_person"}),
    ("allowed_licences not a list", _set(("hosts", OPEN, "allowed_licences"), "MIT"),
     {"table": "hosts", "field": "allowed_licences"}),
    ("host without note", _drop(("hosts", OPEN, "note")), {"table": "hosts", "missing": ["note"]}),
    ("host without requires_terms", _drop(("hosts", OPEN, "requires_terms")),
     {"table": "hosts", "missing": ["requires_terms"]}),
    ("host requires_terms not ids", _set(("hosts", OPEN, "requires_terms"), "SERVER-TERMS"),
     {"table": "hosts", "field": "requires_terms"}),
    ("text with a lone surrogate", _set(("licences", "MIT", "evidence"), "test \ud800 fixture"),
     {"entry": "MIT", "field": "evidence"}),
    ("obligation with a lone surrogate", _set(("licences", "MIT", "obligations"), ["keep \udfff"]),
     {"entry": "MIT", "field": "obligations"}),
    ("unknown kind", _set(("assets", "synthetic", "kind"), "scroll"), {"field": "kind"}),
    ("no locations", _set(("assets", "synthetic", "locations"), []), {"field": "locations"}),
    ("location not a string", _set(("assets", "synthetic", "locations"), [7]), {"field": "locations"}),
    ("licence not an id", _set(("assets", "synthetic", "licence"), 5), {"field": "licence"}),
    ("aliases repeat case-insensitively", _set(("assets", "spiral-input", "aliases"), ["A", "a"]),
     {"field": "aliases"}),
    ("additional_terms missing", _drop(("assets", "synthetic", "additional_terms")),
     {"missing": ["additional_terms"]}),
    ("volume id not 14 digits", _set(("assets", CC_VOLUME_ID, "canonical_volume"),
                                     {"sample": "PHercParis4", "volume_id": "2099010100000"}),
     {"field": "canonical_volume"}),
    ("volume id with a newline", _set(("assets", CC_VOLUME_ID, "canonical_volume"),
                                      {"sample": "PHercParis4", "volume_id": CC_VOLUME_ID + "\n"}),
     {"field": "canonical_volume"}),
    ("canonical_volume with an extra key", _set(("assets", CC_VOLUME_ID, "canonical_volume"),
                                                {"sample": "PHercParis4", "volume_id": CC_VOLUME_ID, "scan": "x"}),
     {"entry": CC_VOLUME_ID, "field": "canonical_volume"}),
    ("canonical_volume without a sample", _set(("assets", CC_VOLUME_ID, "canonical_volume"),
                                               {"volume_id": CC_VOLUME_ID}),
     {"entry": CC_VOLUME_ID, "field": "canonical_volume"}),
    ("canonical_volume sample not a string", _set(("assets", CC_VOLUME_ID, "canonical_volume"),
                                                  {"sample": 4, "volume_id": CC_VOLUME_ID}),
     {"entry": CC_VOLUME_ID, "field": "canonical_volume"}),
    ("canonical_volume sample blank", _set(("assets", CC_VOLUME_ID, "canonical_volume"),
                                           {"sample": " ", "volume_id": CC_VOLUME_ID}),
     {"entry": CC_VOLUME_ID, "field": "canonical_volume"}),
    ("canonical_volume missing on a volume", _drop(("assets", CC_VOLUME_ID, "canonical_volume")),
     {"entry": CC_VOLUME_ID, "field": "canonical_volume"}),
    ("canonical_volume on code", _set(("assets", "code", "canonical_volume"),
                                      {"sample": "PHercParis4", "volume_id": "20990101000009"}),
     {"entry": "code", "field": "canonical_volume"}),
    ("canonical_volume on a synthetic asset", _set(("assets", "synthetic", "canonical_volume"),
                                                   {"sample": "PHercParis4", "volume_id": CC_VOLUME_ID}),
     {"entry": "synthetic", "field": "canonical_volume"}),
    # A volume asset is keyed by its volume id and its locations name that volume.
    ("volume keyed by a scroll name", lambda d: _set(("assets", "Scroll1"), d["assets"].pop(CC_VOLUME_ID))(d),
     {"entry": "Scroll1", "field": "id", "volume_id": CC_VOLUME_ID}),
    ("volume keyed by another volume id", lambda d: _set(("assets", "20990101000005"),
                                                         d["assets"].pop(CC_VOLUME_ID))(d),
     {"entry": "20990101000005", "field": "id", "volume_id": CC_VOLUME_ID}),
    ("volume located at a whole scroll", _set(("assets", CC_VOLUME_ID, "locations"), [f"{OPEN}/PHercParis4"]),
     {"entry": CC_VOLUME_ID, "field": "locations", "locations": [f"{OPEN}/PHercParis4"]}),
    ("volume located at another volume", _set(("assets", CC_VOLUME_ID, "locations"),
                                              [CC_VOLUME_URI, f"{OPEN}/PHercParis4/volumes/{CC_VOLUME_ID}0-x.zarr"]),
     {"entry": CC_VOLUME_ID, "field": "locations",
      "locations": [f"{OPEN}/PHercParis4/volumes/{CC_VOLUME_ID}0-x.zarr"]}),
    ("volume located at a bucket root", _set(("assets", CC_VOLUME_ID, "locations"), [OPEN]),
     {"entry": CC_VOLUME_ID, "field": "locations", "locations": [OPEN]}),
    ("volume id inside a longer name", _set(("assets", CC_VOLUME_ID, "locations"), [f"{OPEN}/v/x-{CC_VOLUME_ID}.zarr"]),
     {"entry": CC_VOLUME_ID, "field": "locations"}),
    ("one location twice", _set(("assets", CC_VOLUME_ID, "locations"),
                                [CC_VOLUME_URI, CC_VOLUME_URI.replace("s3://vesuvius-challenge-open-data",
                                 "https://vesuvius-challenge-open-data.s3.amazonaws.com")]),
     {"entry": CC_VOLUME_ID, "field": "locations"}),
    ("one host twice", _set(("hosts", "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"),
                            host("CC-BY-NC-4.0")),
     {"table": "hosts", "prefix": OPEN, "other": OPEN}),
    ("alias is another asset's id", _set(("assets", "spiral-input", "aliases"), ["Synthetic"]),
     {"entry": "spiral-input", "name": "Synthetic", "other": "synthetic"}),
    ("two assets share an alias", lambda d: (d["assets"]["synthetic"].update(aliases=["spiral input"]), d)[1],
     {"entry": "synthetic", "name": "spiral input", "other": "spiral-input"}),
]


def _with_code_asset(doc):
    doc["assets"]["code"] = asset("code", ["synthetic://apok-fixtures/code"], "MIT")
    return doc


@pytest.mark.parametrize("label, mutate, detail", MALFORMED_CASES, ids=[c[0] for c in MALFORMED_CASES])
def test_malformed_registries_are_refused_naming_the_entry(label, mutate, detail):
    exc = _refused(mutate(_with_code_asset(registry_doc())), source="reg.json")
    assert exc.code == "LICENCE_REGISTRY_MALFORMED", exc.reason
    assert exc.details["source"] == "reg.json"
    for key, value in detail.items():
        assert exc.details[key] == value, (key, exc.details)


def test_the_code_asset_used_above_is_itself_valid():
    assert "code" in parse_registry(_with_code_asset(registry_doc())).assets


@pytest.mark.parametrize("name", [CC_VOLUME_ID, f"{CC_VOLUME_ID}.zarr", f"{CC_VOLUME_ID}-7.910um-54keV-masked.zarr"])
def test_a_volume_location_names_its_volume_in_each_folder_grammar(name):
    doc = registry_doc()
    doc["assets"][CC_VOLUME_ID]["locations"] = [f"{OPEN}/PHercParis4/volumes/{name}"]
    assert str(parse_registry(doc).assets[CC_VOLUME_ID].locations[0]).endswith(name)


@pytest.mark.parametrize("schema", ["apok-asset-registry-v2", None, 1])
def test_wrong_schema_version_is_its_own_refusal(schema):
    doc = registry_doc()
    if schema is None:
        del doc["schema"]
    else:
        doc["schema"] = schema
    exc = _refused(doc)
    assert exc.code == "LICENCE_REGISTRY_SCHEMA"
    assert exc.details["schema"] == schema


@pytest.mark.parametrize(
    "where, value, code, what",
    [
        ("hosts", "dl.ash2txt.org", "LICENCE_URI_RELATIVE", "registry hosts['dl.ash2txt.org']"),
        ("hosts", "https://zarr.aws.ash2txt.org", "LICENCE_URI_AMBIGUOUS",
         "registry hosts['https://zarr.aws.ash2txt.org']"),
        ("hosts", "https://vesuvius-challenge.s3-accelerate.amazonaws.com", "LICENCE_URI_AMBIGUOUS",
         "registry hosts['https://vesuvius-challenge.s3-accelerate.amazonaws.com']"),
        ("assets", "ftp://example.org/x", "LICENCE_URI_SCHEME", "registry assets['synthetic'].locations[0]"),
    ],
)
def test_unusable_registry_locations_name_the_registry_field(where, value, code, what):
    doc = registry_doc()
    if where == "hosts":
        doc["hosts"][value] = host()
    else:
        doc["assets"]["synthetic"]["locations"] = [value]
    exc = _refused(doc)
    assert exc.code == code and exc.details["what"] == what


# --- references --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutate, table, entry, field, ref",
    [
        (_set(("licences", "MIT", "requires_terms"), ["NOPE"]), "licences", "MIT", "requires_terms", "NOPE"),
        (_set(("hosts", OPEN, "allowed_licences"), ["CC-BY-NC-4.0", "GPL"]), "hosts", OPEN, "allowed_licences", "GPL"),
        (_set(("hosts", OPEN, "requires_terms"), ["PORTAL-TERMS"]), "hosts", OPEN, "requires_terms", "PORTAL-TERMS"),
        (_set(("assets", "synthetic", "licence"), "Apache-2.0"), "assets", "synthetic", "licence", "Apache-2.0"),
        (_set(("assets", "spiral-input", "additional_terms"), ["RESERVATION", "EMBARGO"]), "assets", "spiral-input",
         "additional_terms", "EMBARGO"),
    ],
)
def test_unknown_licence_and_terms_ids_are_refused(mutate, table, entry, field, ref):
    exc = _refused(mutate(registry_doc()))
    assert exc.code == "LICENCE_REGISTRY_UNKNOWN_REF"
    assert (exc.details["table"], exc.details["entry"], exc.details["field"], exc.details["ref"]) == (
        table, entry, field, ref)


# --- hard-wired gating of the Vesuvius Challenge hosts ------------------------------------


def _weak_licence(**overrides):
    entry = {"name": "weak", "url": "https://example.org/weak", "requires_terms": ["BUCKET-TERMS"],
             "publication": "ask", "commercial_use": False, "shippable_in_pipeline": False, "obligations": [],
             "evidence": "test"}
    entry.update(overrides)
    return entry


@pytest.mark.parametrize(
    "host, licence, weakness",
    [
        ("https://dl.ash2txt.org", {"id": "MIT"}, "requires no terms"),
        ("https://dl.ash2txt.org/datasets", {"id": "MIT"}, "requires no terms"),
        ("https://data.aws.ash2txt.org", {"id": "MIT"}, "requires no terms"),
        ("https://data.aws.ash2txt.org/other", {"id": "MIT"}, "requires no terms"),
        ("https://vesuvius-challenge.s3.amazonaws.com/PHercParis4", {"id": "MIT"}, "requires no terms"),
        ("s3://vesuvius-challenge-open-data/PHercParis4", {"id": "MIT"}, "requires no terms"),
        # Every host under ash2txt.org is Vesuvius Challenge's own: WebKnossos, thumbnails, volumes, the domain itself.
        ("https://wk.aws.ash2txt.org", {"id": "MIT"}, "requires no terms"),
        ("https://thumbs-vc.aws.ash2txt.org", {"id": "MIT"}, "requires no terms"),
        ("https://volumes.aws.ash2txt.org/PHercParis4", {"id": "MIT"}, "requires no terms"),
        ("https://ash2txt.org", {"id": "MIT"}, "requires no terms"),
        (OPEN, {"id": "WEAK", "publication": "allowed"}, "allows publication"),
        (OPEN, {"id": "WEAK", "commercial_use": True}, "allows commercial use"),
        (OPEN, {"id": "WEAK", "shippable_in_pipeline": True}, "is shippable in the pipeline"),
    ],
)
def test_vesuvius_challenge_hosts_cannot_allow_an_open_licence(host, licence, weakness):
    doc = registry_doc()
    licence = dict(licence)
    lid = licence.pop("id")
    if lid == "WEAK":
        doc["licences"]["WEAK"] = _weak_licence(**licence)
    doc["hosts"][host] = {"allowed_licences": ["EDUCELAB-SCROLLS", lid], "requires_terms": [], "note": "attempt"}
    exc = _refused(doc)
    assert exc.code == "LICENCE_REGISTRY_GATED_HOST"
    assert exc.details["licence"] == lid and exc.details["weakness"] == weakness
    assert exc.details["host"] == str(canonical_uri(host))


def test_a_terms_gated_licence_is_accepted_on_a_gated_host_and_mit_elsewhere():
    doc = registry_doc()
    doc["licences"]["WEAK"] = _weak_licence()
    doc["hosts"]["s3://vesuvius-challenge/PHercParis4/volumes/v.zarr"] = host("WEAK")
    doc["hosts"]["https://wk.aws.ash2txt.org"] = host("WEAK")
    doc["hosts"]["https://github.com/ScrollPrize"] = host("MIT")
    doc["hosts"]["s3://vesuvius-challenge-open-data-mirror"] = host("MIT", note="another bucket: not gated")
    doc["hosts"]["https://notash2txt.org"] = host("MIT", note="not under ash2txt.org: a whole label must match")
    doc["hosts"]["https://ash2txt.org.example.com"] = host("MIT", note="not under ash2txt.org")
    assert len(parse_registry(doc).hosts) == 10


@pytest.mark.parametrize(
    "spelling",
    [
        "https://vesuvius-challenge-open-data.s3.dualstack.us-east-1.amazonaws.com",
        "https://vesuvius-challenge.s3-accelerate.amazonaws.com",
        "https://vesuvius-challenge-open-data.s3-external-1.amazonaws.com",
        "https://vesuvius-challenge.s3-fips.us-east-1.amazonaws.com",
        "https://s3.dualstack.us-east-1.amazonaws.com/vesuvius-challenge-open-data",
        "https://zarr.aws.ash2txt.org/data",
        "https://zarr.aws.ash2txt.org/open-data",
        "https://52.217.0.1/vesuvius-challenge-open-data",
    ],
)
def test_bucket_data_cannot_be_declared_mit_under_another_spelling(spelling):
    doc = registry_doc()
    doc["hosts"][spelling] = host("MIT")
    exc = _refused(doc)
    assert exc.code == "LICENCE_URI_AMBIGUOUS"
    assert exc.details["what"] == f"registry hosts[{spelling!r}]" and exc.details["host"] in spelling


# --- synthetic assets stay local ------------------------------------------------------------


def test_a_synthetic_asset_on_a_remote_host_is_refused():
    doc = registry_doc()
    doc["assets"]["synthetic"]["locations"] = ["synthetic://apok-fixtures", "https://example.org/fixtures/"]
    exc = _refused(doc)
    assert exc.code == "LICENCE_REGISTRY_SYNTHETIC_REMOTE"
    assert exc.details["asset"] == "synthetic" and exc.details["locations"] == ["https://example.org/fixtures"]


def test_a_synthetic_asset_may_live_on_disk_or_a_loopback_server(tmp_path):
    doc = registry_doc()
    doc["assets"]["synthetic"]["locations"] = ["synthetic://apok-fixtures", str(tmp_path),
                                               "http://127.0.0.1:8123/fixtures"]
    assert parse_registry(doc).assets["synthetic"].locations[1].scheme == "file"


# --- volumes and aliases ------------------------------------------------------------------------


def test_sample_aliases_normalise():
    assert canonical_sample("Scroll1") == canonical_sample("PHercParis4") == canonical_sample("PHerc. Paris 4")
    assert canonical_sample("Scroll 3") == canonical_sample("PHerc332") == canonical_sample("PHerc0332")
    assert canonical_sample("scroll5") == canonical_sample("PHerc172") == canonical_sample("PHerc0172")
    assert canonical_sample("Scroll2") == canonical_sample("phercparis3")
    assert canonical_sample("Scroll4") == canonical_sample("PHerc1667")
    assert canonical_sample("Scroll1") != canonical_sample("PHercParis3")
    assert canonical_sample("PHerc0332") != canonical_sample("PHerc0172")


LEGACY_COPY = "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized/x.zarr"


def _second_volume(sample, licence, additional_terms=None, *, aid="legacy-copy", kind="dataset", uri=LEGACY_COPY):
    # A volume asset is keyed by its volume id, so a second claim on one volume is a copy or derivative of it.
    doc = registry_doc()
    doc["assets"][aid] = asset(kind, [uri], licence, additional_terms=additional_terms, volume=(sample, CC_VOLUME_ID))
    return doc


def test_the_same_volume_under_two_licences_is_refused_through_an_alias():
    exc = _refused(_second_volume("Scroll1", "EDUCELAB-SCROLLS"))
    assert exc.code == "LICENCE_REGISTRY_VOLUME_CONFLICT"
    assert exc.details["volume_id"] == CC_VOLUME_ID
    assert exc.details["assets"] == [CC_VOLUME_ID, "legacy-copy"]
    assert exc.details["licences"] == ["CC-BY-NC-4.0", "EDUCELAB-SCROLLS"]


def test_the_same_volume_unlicensed_in_one_entry_is_a_conflict():
    exc = _refused(_second_volume("PHerc. Paris 4", None))
    assert exc.code == "LICENCE_REGISTRY_VOLUME_CONFLICT" and exc.details["licences"] == ["CC-BY-NC-4.0", None]


def test_the_same_volume_with_different_additional_terms_is_a_conflict():
    doc = _second_volume("PHercParis4", "CC-BY-NC-4.0", ["RESERVATION"],
                         uri=CC_VOLUME_URI.replace("PHercParis4/", "mirror/"))
    exc = _refused(doc)
    assert exc.code == "LICENCE_REGISTRY_VOLUME_CONFLICT"
    assert exc.details["additional_terms"] == [[], ["RESERVATION"]]


def test_the_same_volume_id_under_two_samples_is_refused():
    exc = _refused(_second_volume("PHercParis3", "CC-BY-NC-4.0", uri=CC_VOLUME_URI.replace("PHercParis4/", "m/")))
    assert exc.code == "LICENCE_REGISTRY_VOLUME_SAMPLE"
    assert exc.details["samples"] == ["PHercParis4", "PHercParis3"]
    assert exc.details["assets"] == [CC_VOLUME_ID, "legacy-copy"]


def test_a_segment_of_the_volume_under_another_licence_is_a_conflict():
    exc = _refused(_second_volume("Scroll1", "EDUCELAB-SCROLLS", kind="segment", aid="seg",
                                  uri="https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/paths/s1"))
    assert exc.code == "LICENCE_REGISTRY_VOLUME_CONFLICT" and exc.details["assets"] == [CC_VOLUME_ID, "seg"]


def test_two_copies_of_one_volume_conflict_without_the_volume_itself():
    doc = registry_doc()
    del doc["assets"][CC_VOLUME_ID]
    doc["assets"]["legacy"] = asset("dataset", ["https://dl.ash2txt.org/full-scrolls/Scroll5/PHerc172.volpkg/"
                                                "volumes_zarr/20241024131838.zarr"], "EDUCELAB-SCROLLS",
                                    volume=("Scroll5", "20241024131838"))
    doc["assets"]["bucket"] = asset("dataset", [f"{OPEN}/PHerc0172/volumes/20241024131838-7.910um-53keV-masked.zarr"],
                                    "CC-BY-NC-4.0", volume=("PHerc0172", "20241024131838"))
    exc = _refused(doc)
    assert exc.code == "LICENCE_REGISTRY_VOLUME_CONFLICT"
    assert exc.details["volume_id"] == "20241024131838" and exc.details["assets"] == ["bucket", "legacy"]


def test_two_locations_of_one_volume_under_one_licence_are_accepted():
    doc = _second_volume("Scroll1", "CC-BY-NC-4.0", uri=CC_VOLUME_URI.replace("PHercParis4/", "mirror/"))
    doc["assets"]["seg"] = asset("segment", [f"{OPEN}/PHercParis4/paths/s1"], "CC-BY-NC-4.0",
                                 volume=("PHerc. Paris 4", CC_VOLUME_ID))
    reg = parse_registry(doc)
    assert set(reg.assets) == {CC_VOLUME_ID, "legacy-copy", "seg", "spiral-input", "synthetic"}
    assert reg.assets["seg"].canonical_volume.volume_id == CC_VOLUME_ID


# --- overlapping locations ---------------------------------------------------------------------


def test_overlapping_locations_with_different_licences_are_refused():
    doc = registry_doc()
    doc["assets"]["whole-scroll"] = asset("dataset", [f"{OPEN}/PHercParis4"], "EDUCELAB-SCROLLS")
    exc = _refused(doc)
    assert exc.code == "LICENCE_REGISTRY_LOCATION_OVERLAP"
    assert exc.details["assets"] == [CC_VOLUME_ID, "whole-scroll"]
    assert exc.details["locations"] == [CC_VOLUME_URI, f"{OPEN}/PHercParis4"]


def test_overlapping_locations_with_different_additional_terms_are_refused():
    doc = registry_doc()
    doc["assets"]["spiral-subset"] = asset("dataset", [SPIRAL_URI + "/tracks"], "EDUCELAB-SCROLLS")
    exc = _refused(doc)
    assert exc.code == "LICENCE_REGISTRY_LOCATION_OVERLAP"
    assert exc.details["assets"] == ["spiral-input", "spiral-subset"]


def test_overlap_with_identical_obligations_or_a_sibling_prefix_is_accepted():
    doc = registry_doc()
    doc["assets"]["spiral-subset"] = asset("dataset", [SPIRAL_URI + "/tracks"], "EDUCELAB-SCROLLS",
                                           additional_terms=["RESERVATION"])
    doc["assets"]["sibling"] = asset("dataset", [SPIRAL_URI + "0"], "EDUCELAB-SCROLLS")
    assert "sibling" in parse_registry(doc).assets


# --- every registered code is raised somewhere, and only registered codes are raised ---------------


def test_licence_codes_registered_equal_codes_raised():
    raised = set()
    for path in SRC.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call)
                    and getattr(node.exc.func, "id", None) == "Refusal"):
                raised.add(node.exc.args[0].value)
    registered = {code for code, owner in REGISTRY.items() if owner.startswith("apok_eval.licence")}
    assert raised == registered
    assert all(code.startswith("LICENCE_") for code in registered)
