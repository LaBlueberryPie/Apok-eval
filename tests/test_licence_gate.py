"""The licence gate: a fully covered run clears, synthetic data clears without
terms, and every refusal fires with its own code and names its cause."""

from __future__ import annotations

import copy
import pickle

import pytest
from licence_fixtures import (
    CC_VOLUME_ID,
    CC_VOLUME_URI,
    EDUCELAB_GATEWAY_VOLUME,
    OPEN,
    SPIRAL_URI,
    SYNTHETIC_URI,
    acceptance,
    asset,
    host,
    registry_doc,
    run,
    write_json,
)

from apok_eval.canonical import digest_json, sha256_bytes
from apok_eval.clearance import Clearance
from apok_eval.licence import (
    LicenceClearance,
    canonical_uri,
    check_run,
    input_decisions,
    licence_check,
    load_registry,
    parse_registry,
)
from apok_eval.licence.registry import SHIPPED_REGISTRY, read_registry
from apok_eval.refusal import Refusal

ALL_TERMS = ("BUCKET-TERMS", "SERVER-TERMS", "RESERVATION")


def _refused(config, registry=None, **kwargs):
    with pytest.raises(Refusal) as info:
        check_run(config, registry if registry is not None else registry_doc(), **kwargs)
    return info.value


def _clear(config, registry=None, **kwargs):
    return check_run(config, registry if registry is not None else registry_doc(), **kwargs)


# --- clearing ------------------------------------------------------------------------


def test_a_fully_covered_run_clears_with_its_obligations_and_acceptance():
    config = run((CC_VOLUME_ID, CC_VOLUME_URI + "/0"), ("spiral-input", SPIRAL_URI + "/tracks/"),
                 ("synthetic", SYNTHETIC_URI + "/seed-7"), accepted=acceptance(*ALL_TERMS, person="  Jane Example "))
    clearance = _clear(config)
    assert [item.asset for item in clearance.inputs] == [CC_VOLUME_ID, "spiral-input", "synthetic"]
    assert [item.licence for item in clearance.inputs] == ["CC-BY-NC-4.0", "EDUCELAB-SCROLLS", "MIT"]
    spiral = clearance.inputs[1]
    assert str(spiral.uri) == SPIRAL_URI + "/tracks" and spiral.given_uri == SPIRAL_URI + "/tracks/"
    assert spiral.host == "https://dl.ash2txt.org"
    assert spiral.obligations == ("no redistribution without written approval",)
    assert spiral.additional_terms == ("RESERVATION",)
    assert spiral.required_terms == ("SERVER-TERMS", "RESERVATION")
    assert clearance.required_terms == ("BUCKET-TERMS", "SERVER-TERMS", "RESERVATION")
    assert clearance.acceptance.person == "Jane Example"
    assert [(a.terms, a.pinned, a.per_person) for a in clearance.acceptance.accepted] == [
        ("BUCKET-TERMS", False, True), ("SERVER-TERMS", False, True), ("RESERVATION", False, True)]
    assert clearance.registry_digest == digest_json(registry_doc())
    record = clearance.as_dict()
    assert record["schema"] == "apok-licence-clearance-v1"
    assert record["inputs"][1]["decision"] == "cleared"
    assert record["acceptance"]["person"] == "Jane Example"
    assert digest_json(record)  # canonical-JSON serialisable as it stands


def test_synthetic_data_clears_without_terms():
    clearance = _clear(run(("synthetic", SYNTHETIC_URI + "/spiral/seed-7")))
    assert clearance.acceptance is None and clearance.required_terms == ()
    assert clearance.as_dict()["acceptance"] is None


def test_terms_accepted_by_null_means_absent():
    config = run(("synthetic", SYNTHETIC_URI))
    config["terms_accepted_by"] = None
    assert _clear(config).acceptance is None
    config["inputs"] = [{"asset": "spiral-input", "uri": SPIRAL_URI}]
    assert _refused(config).code == "LICENCE_TERMS_ABSENT"


def test_other_top_level_keys_are_left_to_other_modules():
    config = run(("synthetic", SYNTHETIC_URI))
    config["fold"] = {"blocks": 4}
    assert _clear(config).inputs[0].asset == "synthetic"


def test_local_fixtures_clear_under_a_declared_root_and_symlinks_cannot_escape(tmp_path):
    root, outside = tmp_path / "fixtures", tmp_path / "scroll-copy"
    root.mkdir()
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)
    doc = registry_doc()
    doc["hosts"][str(root)] = host("MIT", note="this test's fixtures")
    doc["assets"]["disk-fixtures"] = asset("synthetic", [f"file://{root}"], "MIT")
    clearance = _clear(run(("disk-fixtures", str(root / "volume.zarr"))), doc)
    assert clearance.covers(f"file://{root}/volume.zarr/0/0.0.0")
    exc = _refused(run(("disk-fixtures", str(root / "escape" / "volume.zarr"))), doc)
    assert exc.code == "LICENCE_URI_OUTSIDE_ASSET" and exc.details["asset"] == "disk-fixtures"


def test_a_loopback_test_server_can_be_cleared_for_reader_tests():
    doc = registry_doc()
    doc["hosts"]["http://127.0.0.1:8123"] = host("MIT", note="reader test server")
    doc["assets"]["served-fixture"] = asset("synthetic", ["http://127.0.0.1:8123/store.zarr"], "MIT")
    clearance = _clear(run(("served-fixture", "http://127.0.0.1:8123/store.zarr/")), doc)
    assert clearance.covers("http://127.0.0.1:8123/store.zarr/.zgroup")
    assert not clearance.covers("http://127.0.0.1:8124/store.zarr/.zgroup")


# --- the clearance object ---------------------------------------------------------------


@pytest.fixture
def cleared():
    return _clear(run((CC_VOLUME_ID, CC_VOLUME_URI), ("synthetic", SYNTHETIC_URI + "/a"),
                      accepted=acceptance("BUCKET-TERMS")))


def test_clearance_implements_the_protocol_and_covers_by_whole_segments(cleared):
    assert isinstance(cleared, Clearance)
    https = CC_VOLUME_URI.replace("s3://vesuvius-challenge-open-data",
                                  "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com")
    assert cleared.covers(CC_VOLUME_URI + "/0/0.0.0")
    assert cleared.covers(https + "/.zattrs")
    assert cleared.covers("synthetic://apok-fixtures/a/b")
    assert not cleared.covers(CC_VOLUME_URI + "X/.zattrs")
    assert not cleared.covers("synthetic://apok-fixtures/b")
    assert not cleared.covers(f"{OPEN}/PHercParis4")  # a parent of what was cleared is not cleared
    assert not cleared.covers("relative/path")  # not canonicalisable: not covered, and no exception
    assert not cleared.covers(CC_VOLUME_URI + "/a%2Fb")
    assert not cleared.covers(None)  # type: ignore[arg-type]


def test_clearance_is_immutable(cleared):
    with pytest.raises(AttributeError, match="immutable"):
        cleared._inputs = ()
    with pytest.raises(AttributeError, match="immutable"):
        cleared.covers = lambda uri: True
    with pytest.raises(AttributeError, match="immutable"):
        del cleared._acceptance
    with pytest.raises(AttributeError):
        cleared.inputs = ()
    assert isinstance(cleared.inputs, tuple) and isinstance(cleared.inputs[0].obligations, tuple)
    with pytest.raises(AttributeError):
        cleared.inputs[0].licence = "MIT"
    for clone in (copy.copy, copy.deepcopy, lambda c: pickle.loads(pickle.dumps(c))):
        with pytest.raises(AttributeError, match="immutable"):
            clone(cleared)
    assert cleared.covers(CC_VOLUME_URI)


@pytest.mark.parametrize("token", [None, object(), "issue"])
def test_a_clearance_cannot_be_constructed_outside_the_gate(token):
    with pytest.raises(Refusal) as info:
        LicenceClearance(token, inputs=(), acceptance=None, for_release=False, registry_digest="sha256:" + "0" * 64,
                         registry_source="forged")
    assert info.value.code == "LICENCE_CLEARANCE_UNISSUED"
    assert info.value.details["token"] == repr(token)


def test_the_clearance_record_never_writes_the_acceptance_key(cleared):
    assert "terms_accepted_by" not in str(cleared.as_dict())


# --- run configuration ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "config, field",
    [
        ([{"asset": "synthetic", "uri": SYNTHETIC_URI}], "$"),
        ({}, "inputs"),
        ({"inputs": []}, "inputs"),
        ({"inputs": {"asset": "synthetic", "uri": SYNTHETIC_URI}}, "inputs"),
        ({"inputs": ["synthetic"]}, "inputs[0]"),
        ({"inputs": [{"asset": "synthetic"}]}, "inputs[0]"),
        ({"inputs": [{"asset": "synthetic", "uri": SYNTHETIC_URI, "licence": "MIT"}]}, "inputs[0]"),
        ({"inputs": [{"asset": "synthetic", "uri": SYNTHETIC_URI}, {"asset": 7, "uri": SYNTHETIC_URI}]}, "inputs[1]"),
        ({"inputs": [{"asset": "synthetic", "uri": [SYNTHETIC_URI]}]}, "inputs[0]"),
    ],
)
def test_malformed_run_configurations_are_refused(config, field):
    exc = _refused(config)
    assert exc.code == "LICENCE_RUN_MALFORMED" and exc.details["field"] == field


@pytest.mark.parametrize("content, problem", [
    (None, "NotRegularFile"), (b"[", "invalid JSON: Expecting value"),
    (b'{"inputs": [], "inputs": []}', "duplicate key 'inputs'"),
    (b'{"inputs": [], "fold": "\\ud800"}', "lone surrogate '\\ud800'"),
    # Python's json raised RecursionError here, which escaped the gate as a crash.
    (b'{"inputs": [], "fold": ' + b"[" * 100_000 + b"]" * 100_000 + b"}", "nested deeper than the parser can recurse"),
])
def test_unreadable_run_configurations_are_refused(tmp_path, content, problem):
    path = tmp_path / "run.json"
    if content is not None:
        path.write_bytes(content)
    with pytest.raises(Refusal) as info:
        licence_check(path, write_json(tmp_path / "registry.json", registry_doc()))
    assert info.value.code == "LICENCE_RUN_UNREADABLE"
    assert info.value.details["path"] == str(path) and problem in info.value.details["problem"]


@pytest.mark.parametrize("name, problem", [
    ("run\x00.json", "ValueError: "),  # the message itself depends on the Python version
    ("run\ud800.json", "UnicodeEncodeError: 'utf-8' codec can't encode character '\\ud800'"),
], ids=["nul", "lone-surrogate"])
def test_a_run_configuration_path_open_refuses_is_unreadable(tmp_path, name, problem):
    # open() raises ValueError, not OSError, for these paths; the reader once let it escape as a crash.
    path = str(tmp_path / name)
    with pytest.raises(Refusal) as info:
        licence_check(path, write_json(tmp_path / "registry.json", registry_doc()))
    assert info.value.code == "LICENCE_RUN_UNREADABLE"
    assert info.value.details["path"] == path and info.value.details["problem"].startswith(problem)


def test_the_registry_is_validated_before_the_run_configuration_is_read(tmp_path):
    with pytest.raises(Refusal) as info:
        licence_check(tmp_path / "missing-run.json", {"schema": "x"})
    assert info.value.code == "LICENCE_REGISTRY_SCHEMA" and info.value.details["schema"] == "x"


@pytest.mark.parametrize("registry", [load_registry(), parse_registry(registry_doc()), 7, b"registry.json"])
def test_a_registry_argument_of_another_type_is_a_clear_type_error(registry):
    with pytest.raises(TypeError, match="registry must be a path, a registry document"):
        check_run(run(("synthetic", SYNTHETIC_URI)), registry)
    with pytest.raises(TypeError, match="registry must be a path, a registry document"):
        input_decisions(run(("synthetic", SYNTHETIC_URI)), registry)


def test_a_run_configuration_nested_deeper_than_the_strict_limit_is_refused(tmp_path):
    # Other modules' keys are ignored by the gate, but the document is still read strictly: Python's json
    # accepted 300 levels, and the run cleared.
    path = tmp_path / "run.json"
    path.write_text('{"inputs": [{"asset": "synthetic", "uri": "synthetic://apok-fixtures"}], "fold": '
                    + "[" * 300 + "]" * 300 + "}", encoding="utf-8")
    with pytest.raises(Refusal) as info:
        licence_check(path, write_json(tmp_path / "registry.json", registry_doc()))
    assert info.value.code == "LICENCE_RUN_UNREADABLE" and info.value.details["path"] == str(path)
    assert info.value.details["problem"] == "arrays and objects nested deeper than 256 levels"


def test_a_duplicated_acceptance_key_cannot_hide_behind_null(tmp_path):
    path = tmp_path / "run.json"
    path.write_text('{"inputs": [{"asset": "synthetic", "uri": "synthetic://apok-fixtures"}], '
                    '"terms_accepted_by": {"person": "X"}, "terms_accepted_by": null}', encoding="utf-8")
    with pytest.raises(Refusal) as info:
        licence_check(path, write_json(tmp_path / "registry.json", registry_doc()))
    assert info.value.code == "LICENCE_RUN_UNREADABLE" and "terms_accepted_by" in info.value.details["problem"]


# --- per-input refusals --------------------------------------------------------------------------


def test_an_asset_not_in_the_registry_is_refused():
    exc = _refused(run(("synthetic", SYNTHETIC_URI), ("nowhere", SYNTHETIC_URI)))
    assert exc.code == "LICENCE_ASSET_UNKNOWN"
    assert (exc.details["input"], exc.details["asset"], exc.details["name_of"]) == (1, "nowhere", None)


@pytest.mark.parametrize("name", ["spiral input", "Spiral-Input"])
def test_an_alias_or_miscased_id_is_not_an_id(name):
    exc = _refused(run((name, SPIRAL_URI), accepted=acceptance(*ALL_TERMS)))
    assert exc.code == "LICENCE_ASSET_UNKNOWN"
    assert exc.details["asset"] == name and exc.details["name_of"] == "spiral-input"


def test_an_unlicensed_asset_is_refused_with_its_note():
    doc = registry_doc()
    doc["assets"]["scroll2"] = asset("dataset", ["https://dl.ash2txt.org/full-scrolls/Scroll2"], None)
    doc["assets"]["scroll2"]["note"] = "sources disagree on the licence"
    exc = _refused(run(("scroll2", "https://dl.ash2txt.org/full-scrolls/Scroll2/x.zarr"),
                       accepted=acceptance(*ALL_TERMS)), doc)
    assert exc.code == "LICENCE_ASSET_UNLICENSED"
    assert exc.details["asset"] == "scroll2" and exc.details["note"] == "sources disagree on the licence"
    assert "sources disagree" in exc.reason


def test_an_unusable_input_uri_names_the_input():
    exc = _refused(run(("synthetic", SYNTHETIC_URI), ("synthetic", "fixtures/x")))
    assert exc.code == "LICENCE_URI_RELATIVE" and exc.details["what"] == "inputs[1].uri"


@pytest.mark.parametrize(
    "name, uri",
    [
        ("synthetic", "https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4/tracks"),
        ("synthetic", "https://data.aws.ash2txt.org/samples/" + EDUCELAB_GATEWAY_VOLUME),
        (CC_VOLUME_ID, "https://data.aws.ash2txt.org/samples/" + EDUCELAB_GATEWAY_VOLUME),
        (CC_VOLUME_ID, CC_VOLUME_URI + "0"),
        ("spiral-input", SPIRAL_URI + "0/tracks"),
    ],
)
def test_data_cannot_be_relabelled_as_another_asset(name, uri):
    exc = _refused(run((name, uri), accepted=acceptance(*ALL_TERMS)))
    assert exc.code == "LICENCE_URI_OUTSIDE_ASSET"
    assert exc.details["asset"] == name and exc.details["input"] == 0
    assert exc.details["uri"] == str(canonical_uri(uri))


def test_an_input_on_an_undeclared_host_is_refused(tmp_path):
    doc = registry_doc()
    doc["assets"]["some-code"] = asset("code", ["https://github.com/ScrollPrize/villa"], "MIT")
    doc["assets"]["local-copy"] = asset("dataset", [str(tmp_path)], "MIT")
    exc = _refused(run(("some-code", "https://github.com/ScrollPrize/villa/tree/fb8c2c4")), doc)
    assert exc.code == "LICENCE_HOST_UNDECLARED"
    assert exc.details["asset"] == "some-code"
    assert exc.details["uri"] == "https://github.com/ScrollPrize/villa/tree/fb8c2c4"
    exc = _refused(run(("local-copy", str(tmp_path / "x"))), doc)  # local roots must be declared too
    assert exc.code == "LICENCE_HOST_UNDECLARED" and exc.details["asset"] == "local-copy"


@pytest.mark.parametrize(
    "spelling",
    [
        "https://data.aws.ash2txt.org/samples/" + EDUCELAB_GATEWAY_VOLUME,
        "s3://vesuvius-challenge/" + EDUCELAB_GATEWAY_VOLUME,
        "https://vesuvius-challenge.s3.us-east-1.amazonaws.com/" + EDUCELAB_GATEWAY_VOLUME + "/",
    ],
)
def test_an_educelab_volume_on_the_gateway_cannot_pass_as_cc_by_nc(spelling):
    doc = registry_doc()
    doc["assets"]["mislabelled"] = asset("dataset", [spelling], "CC-BY-NC-4.0")
    exc = _refused(run(("mislabelled", spelling.rstrip("/") + "/0/0.0.0"), accepted=acceptance(*ALL_TERMS)), doc)
    assert exc.code == "LICENCE_HOST_DISALLOWS"
    assert exc.details["licence"] == "CC-BY-NC-4.0" and exc.details["host"] == "s3://vesuvius-challenge"
    assert exc.details["allowed"] == ["EDUCELAB-SCROLLS"] and exc.details["asset"] == "mislabelled"


def test_a_checked_volume_on_the_gateway_clears_only_under_its_own_host_entry():
    doc = registry_doc()
    checked = "s3://vesuvius-challenge/PHercParis4/volumes/20990101000003-2.400um-0.2m-137keV-masked.zarr"
    doc["hosts"][checked] = host("CC-BY-NC-4.0", note="this volume was checked to be CC BY-NC")
    doc["assets"]["20990101000003"] = asset("volume", [checked], "CC-BY-NC-4.0", volume=("Scroll1", "20990101000003"))
    doc["assets"]["dls"] = asset("dataset", ["s3://vesuvius-challenge/" + EDUCELAB_GATEWAY_VOLUME], "CC-BY-NC-4.0")
    gateway = checked.replace("s3://vesuvius-challenge/", "https://data.aws.ash2txt.org/samples/")
    cleared = _clear(run(("20990101000003", gateway + "/2"), accepted=acceptance("BUCKET-TERMS")), doc)
    assert cleared.inputs[0].host == checked
    exc = _refused(run(("dls", "s3://vesuvius-challenge/" + EDUCELAB_GATEWAY_VOLUME),
                       accepted=acceptance("BUCKET-TERMS")), doc)
    assert exc.code == "LICENCE_HOST_DISALLOWS" and exc.details["host"] == "s3://vesuvius-challenge"


# --- acceptance ----------------------------------------------------------------------------


def _with_acceptance(raw):
    config = run((CC_VOLUME_ID, CC_VOLUME_URI))
    config["terms_accepted_by"] = raw
    return config


def _replace(**changes):
    raw = acceptance("BUCKET-TERMS")
    raw.update(changes)
    return raw


def _drop_key(key):
    raw = acceptance("BUCKET-TERMS")
    del raw[key]
    return raw


ACCEPTANCE_CASES = [
    ("a bare name", "Jane Example", "terms_accepted_by"),
    ("a list", [acceptance("BUCKET-TERMS")], "terms_accepted_by"),
    ("no person", _drop_key("person"), "terms_accepted_by"),
    ("an extra key", _replace(persons=["Jane"]), "terms_accepted_by"),
    ("a list of persons", _replace(person=["Jane Example", "John Example"]), "terms_accepted_by.person"),
    ("a person object", _replace(person={"name": "Jane"}), "terms_accepted_by.person"),
    ("an empty person", _replace(person=""), "terms_accepted_by.person"),
    ("a blank person", _replace(person="   "), "terms_accepted_by.person"),
    ("two people with and", _replace(person="Jane and John"), "terms_accepted_by.person"),
    ("two people with AND", _replace(person="Jane AND John"), "terms_accepted_by.person"),
    ("a comma list", _replace(person="Jane, John"), "terms_accepted_by.person"),
    ("an ampersand", _replace(person="Jane & John"), "terms_accepted_by.person"),
    ("a semicolon", _replace(person="Jane; John"), "terms_accepted_by.person"),
    ("a slash", _replace(person="Jane/John"), "terms_accepted_by.person"),
    ("a plus", _replace(person="Jane+John"), "terms_accepted_by.person"),
    ("a pipe", _replace(person="Jane | John"), "terms_accepted_by.person"),
    ("et al", _replace(person="Jane Example et al."), "terms_accepted_by.person"),
    ("et", _replace(person="Jean et Marie"), "terms_accepted_by.person"),
    ("und", _replace(person="Jane Doe und John Roe"), "terms_accepted_by.person"),
    # NFKC folds fullwidth forms into the ASCII separators.
    ("a fullwidth ampersand", _replace(person="Jane \uff06 John"), "terms_accepted_by.person"),
    ("a fullwidth comma", _replace(person="Jane Doe\uff0cJohn Roe"), "terms_accepted_by.person"),
    ("a fullwidth solidus", _replace(person="Jane\uff0fJohn"), "terms_accepted_by.person"),
    ("an ideographic comma", _replace(person="Jane Doe\u3001John Roe"), "terms_accepted_by.person"),
    ("an Arabic comma", _replace(person="Jane Doe\u060c John Roe"), "terms_accepted_by.person"),
    ("a bullet", _replace(person="Jane Doe \u2022 John Roe"), "terms_accepted_by.person"),
    ("a team", _replace(person="The Example Team"), "terms_accepted_by.person"),
    ("everyone", _replace(person="Everyone"), "terms_accepted_by.person"),
    ("all members", _replace(person="all team members"), "terms_accepted_by.person"),
    ("on behalf of", _replace(person="Jane Doe (on behalf of Example)"), "terms_accepted_by.person"),
    ("a company", _replace(person="Example Ltd"), "terms_accepted_by.person"),
    ("a fullwidth company", _replace(person="Example \uff2c\uff54\uff44"), "terms_accepted_by.person"),
    ("a line break", _replace(person="Jane\nJohn"), "terms_accepted_by.person"),
    ("no letter", _replace(person="1234"), "terms_accepted_by.person"),
    ("a day-first date", _replace(date="01/06/2025"), "terms_accepted_by.date"),
    ("an impossible day", _replace(date="2026-02-30"), "terms_accepted_by.date"),
    ("an unpadded date", _replace(date="2025-06-1"), "terms_accepted_by.date"),
    ("a compact date", _replace(date="20250601"), "terms_accepted_by.date"),
    ("a week date", _replace(date="2025-W23"), "terms_accepted_by.date"),
    ("a numeric date", _replace(date=20250601), "terms_accepted_by.date"),
    ("a date with a newline", _replace(date="2025-06-01\n"), "terms_accepted_by.date"),
    ("no accepted terms", _replace(accepted=[]), "terms_accepted_by.accepted"),
    ("accepted not a list", _replace(accepted={"terms": "BUCKET-TERMS", "text_sha256": None}),
     "terms_accepted_by.accepted"),
    ("an accepted string", _replace(accepted=["BUCKET-TERMS"]), "terms_accepted_by.accepted[0]"),
    ("an accepted entry without digest", _replace(accepted=[{"terms": "BUCKET-TERMS"}]),
     "terms_accepted_by.accepted[0]"),
    ("an accepted entry with an extra key", _replace(accepted=[{"terms": "BUCKET-TERMS", "text_sha256": None,
                                                                "person": "John Example"}]),
     "terms_accepted_by.accepted[0]"),
    ("unknown terms", _replace(accepted=[{"terms": "BUCKET-TERMS", "text_sha256": None},
                                         {"terms": "ALL-OF-THEM", "text_sha256": None}]),
     "terms_accepted_by.accepted[1].terms"),
    ("terms not a string", _replace(accepted=[{"terms": ["BUCKET-TERMS"], "text_sha256": None}]),
     "terms_accepted_by.accepted[0].terms"),
    ("a bad digest", _replace(accepted=[{"terms": "BUCKET-TERMS", "text_sha256": "sha256:XYZ"}]),
     "terms_accepted_by.accepted[0].text_sha256"),
    ("terms twice", _replace(accepted=[{"terms": "BUCKET-TERMS", "text_sha256": None}] * 2),
     "terms_accepted_by.accepted[1]"),
]


@pytest.mark.parametrize("label, raw, field", ACCEPTANCE_CASES, ids=[c[0] for c in ACCEPTANCE_CASES])
def test_malformed_acceptances_are_refused_naming_the_field(label, raw, field):
    exc = _refused(_with_acceptance(copy.deepcopy(raw)))
    assert exc.code == "LICENCE_ACCEPTANCE_MALFORMED", exc.reason
    assert exc.details["field"] == field


def test_an_in_memory_acceptance_with_odd_keys_is_refused_not_crashed():
    exc = _refused(_with_acceptance({1: "Jane", "date": "2025-06-01", "accepted": []}))
    assert exc.code == "LICENCE_ACCEPTANCE_MALFORMED" and exc.details["keys"] == ["1", "accepted", "date"]


def test_a_malformed_acceptance_is_refused_even_when_no_terms_are_needed():
    config = run(("synthetic", SYNTHETIC_URI), accepted=acceptance("BUCKET-TERMS", person="Jane and John"))
    exc = _refused(config)
    assert exc.code == "LICENCE_ACCEPTANCE_MALFORMED" and exc.details["value"] == "Jane and John"


@pytest.mark.parametrize("person", ["Jane Example", "Anderson Sandoval", "Jean-Luc Ørsted", "Q", "jane@example.org",
                                    "Mette Allison", "Etienne Teamey", "Ünal Özdemir", "ジョン・スミス", "Paul Inchbald"])
def test_one_named_person_is_accepted(person):
    clearance = _clear(_with_acceptance(acceptance("BUCKET-TERMS", person=person)))
    assert clearance.acceptance.person == person


def test_the_person_is_recorded_as_written_not_normalised():
    person = "Ｊａｎｅ Example"  # fullwidth letters: NFKC is used to detect separators only
    assert _clear(_with_acceptance(acceptance("BUCKET-TERMS", person=person))).acceptance.person == person


def test_a_pinned_terms_text_must_be_the_one_accepted():
    pinned = sha256_bytes(b"the bucket terms, the version the person read")
    doc = registry_doc()
    doc["terms"]["BUCKET-TERMS"]["text_sha256"] = pinned
    other = sha256_bytes(b"some other text")
    for cited in (other, None):
        exc = _refused(_with_acceptance(acceptance("BUCKET-TERMS", digests={"BUCKET-TERMS": cited})), doc)
        assert exc.code == "LICENCE_TERMS_DIGEST_MISMATCH"
        assert (exc.details["terms"], exc.details["cited"], exc.details["pinned"]) == ("BUCKET-TERMS", cited, pinned)
    good = _clear(_with_acceptance(acceptance("BUCKET-TERMS", digests={"BUCKET-TERMS": pinned})), doc)
    assert good.acceptance.accepted[0].pinned and good.acceptance.accepted[0].text_sha256 == pinned


def test_an_unpinned_terms_text_records_the_cited_digest_as_unverified():
    cited = sha256_bytes(b"what the person read")
    clearance = _clear(_with_acceptance(acceptance("BUCKET-TERMS", digests={"BUCKET-TERMS": cited})))
    entry = clearance.acceptance.accepted[0]
    assert entry.text_sha256 == cited and not entry.pinned


def test_terms_are_required_and_absence_names_them():
    exc = _refused(run((CC_VOLUME_ID, CC_VOLUME_URI), ("spiral-input", SPIRAL_URI)))
    assert exc.code == "LICENCE_TERMS_ABSENT"
    assert exc.details["required"] == ["BUCKET-TERMS", "SERVER-TERMS", "RESERVATION"]
    assert exc.details["required_by"] == {"BUCKET-TERMS": [0], "SERVER-TERMS": [1], "RESERVATION": [1]}


def test_an_acceptance_that_misses_terms_names_the_missing_ids():
    exc = _refused(run((CC_VOLUME_ID, CC_VOLUME_URI), ("spiral-input", SPIRAL_URI),
                       accepted=acceptance("SERVER-TERMS", "BUCKET-TERMS")))
    assert exc.code == "LICENCE_TERMS_UNCOVERED"
    assert exc.details["missing"] == ["RESERVATION"]  # an asset's additional terms count
    assert exc.details["required_by"] == {"RESERVATION": [1]} and exc.details["person"] == "Jane Example"
    exc = _refused(run(("spiral-input", SPIRAL_URI), accepted=acceptance("BUCKET-TERMS")))
    assert exc.code == "LICENCE_TERMS_UNCOVERED" and exc.details["missing"] == ["SERVER-TERMS", "RESERVATION"]


def test_a_hosts_own_terms_are_required_whatever_licence_it_allows():
    doc = registry_doc()
    doc["hosts"]["https://dl.ash2txt.org"]["allowed_licences"] = ["EDUCELAB-SCROLLS", "CC-BY-NC-4.0"]
    doc["assets"]["cc-on-server"] = asset("dataset", ["https://dl.ash2txt.org/other/cc"], "CC-BY-NC-4.0")
    config = run(("cc-on-server", "https://dl.ash2txt.org/other/cc/x"), accepted=acceptance("BUCKET-TERMS"))
    exc = _refused(config, doc)
    assert exc.code == "LICENCE_TERMS_UNCOVERED"
    assert exc.details["missing"] == ["SERVER-TERMS"] and exc.details["required_by"] == {"SERVER-TERMS": [0]}
    config["terms_accepted_by"] = acceptance("BUCKET-TERMS", "SERVER-TERMS")
    assert _clear(config, doc).inputs[0].required_terms == ("BUCKET-TERMS", "SERVER-TERMS")


def test_a_narrower_host_entry_cannot_drop_the_terms_of_the_host_around_it():
    doc = registry_doc()
    doc["hosts"]["https://dl.ash2txt.org/other"] = host("CC-BY-NC-4.0", requires_terms=["RESERVATION"])
    doc["assets"]["cc-on-server"] = asset("dataset", ["https://dl.ash2txt.org/other/cc"], "CC-BY-NC-4.0")
    config = run(("cc-on-server", "https://dl.ash2txt.org/other/cc/x"), accepted=acceptance("BUCKET-TERMS"))
    exc = _refused(config, doc)
    assert exc.code == "LICENCE_TERMS_UNCOVERED" and exc.details["missing"] == ["SERVER-TERMS", "RESERVATION"]
    config["terms_accepted_by"] = acceptance("BUCKET-TERMS", "SERVER-TERMS", "RESERVATION")
    cleared = _clear(config, doc).inputs[0]
    assert cleared.host == "https://dl.ash2txt.org/other"  # the narrow entry decides the licence ...
    assert cleared.required_terms == ("BUCKET-TERMS", "SERVER-TERMS", "RESERVATION")  # ... not the terms


def test_cc_by_nc_bucket_data_also_needs_a_recorded_acceptance():
    exc = _refused(run((CC_VOLUME_ID, CC_VOLUME_URI)))
    assert exc.code == "LICENCE_TERMS_ABSENT" and exc.details["required"] == ["BUCKET-TERMS"]


# --- release mode ----------------------------------------------------------------------------


def _release_registry():
    doc = registry_doc()
    doc["licences"]["OPEN-NOT-SHIPPED"] = dict(doc["licences"]["MIT"], shippable_in_pipeline=False)
    doc["hosts"]["synthetic://release"] = host("MIT", "OPEN-NOT-SHIPPED", note="release tests")
    doc["assets"]["not-shipped"] = asset("code", ["synthetic://release/a"], "OPEN-NOT-SHIPPED")
    doc["assets"]["mit-with-terms"] = asset("code", ["synthetic://release/b"], "MIT", additional_terms=["RESERVATION"])
    return doc


def test_release_mode_clears_open_shippable_inputs():
    clearance = _clear(run(("synthetic", SYNTHETIC_URI)), for_release=True)
    assert clearance.for_release and clearance.as_dict()["for_release"] is True
    assert not _clear(run(("synthetic", SYNTHETIC_URI))).for_release


def test_release_mode_refuses_unpublishable_inputs_naming_input_and_licence():
    config = run(("synthetic", SYNTHETIC_URI), (CC_VOLUME_ID, CC_VOLUME_URI), accepted=acceptance("BUCKET-TERMS"))
    assert _clear(config).inputs[1].licence == "CC-BY-NC-4.0"
    exc = _refused(config, for_release=True)
    assert exc.code == "LICENCE_RELEASE_PUBLICATION"
    assert (exc.details["input"], exc.details["asset"], exc.details["licence"], exc.details["publication"]) == (
        1, CC_VOLUME_ID, "CC-BY-NC-4.0", "ask")
    exc = _refused(run(("spiral-input", SPIRAL_URI), accepted=acceptance(*ALL_TERMS)), for_release=True)
    assert exc.code == "LICENCE_RELEASE_PUBLICATION" and exc.details["publication"] == "forbidden"


def test_release_mode_refuses_unshippable_inputs():
    doc = _release_registry()
    exc = _refused(run(("not-shipped", "synthetic://release/a")), doc, for_release=True)
    assert exc.code == "LICENCE_RELEASE_UNSHIPPABLE"
    assert (exc.details["asset"], exc.details["licence"]) == ("not-shipped", "OPEN-NOT-SHIPPED")


def test_release_mode_refuses_inputs_with_additional_terms():
    doc = _release_registry()
    config = run(("mit-with-terms", "synthetic://release/b"), accepted=acceptance("RESERVATION"))
    assert _clear(config, doc).inputs[0].additional_terms == ("RESERVATION",)
    exc = _refused(config, doc, for_release=True)
    assert exc.code == "LICENCE_RELEASE_ADDITIONAL_TERMS"
    assert exc.details["asset"] == "mit-with-terms" and exc.details["additional_terms"] == ["RESERVATION"]


# --- each input judged on its own (for the refusal report) -----------------------------------


def test_input_decisions_judge_every_input_on_its_own():
    config = run(("synthetic", SYNTHETIC_URI + "/a"), ("spiral-input", SPIRAL_URI + "/x"), ("nowhere", SYNTHETIC_URI),
                 (CC_VOLUME_ID, CC_VOLUME_URI), accepted=acceptance("BUCKET-TERMS"))
    assert _refused(config).code == "LICENCE_ASSET_UNKNOWN"  # the gate stops at the first refusal ...
    decisions = input_decisions(config, registry_doc())  # ... the report shows each input
    assert [d.index for d in decisions] == [0, 1, 2, 3]
    assert [None if d.refusal is None else d.refusal.code for d in decisions] == [
        None, "LICENCE_TERMS_UNCOVERED", "LICENCE_ASSET_UNKNOWN", None]
    assert decisions[1].refusal.details["missing"] == ["SERVER-TERMS", "RESERVATION"]
    assert (decisions[1].licence, decisions[1].additional_terms) == ("EDUCELAB-SCROLLS", ("RESERVATION",))
    assert decisions[1].obligations == ("no redistribution without written approval",)
    assert (decisions[2].asset, decisions[2].licence, decisions[2].additional_terms) == ("nowhere", None, ())
    record = decisions[1].as_dict()
    assert record["decision"] == "refused" and record["refusal"]["code"] == "LICENCE_TERMS_UNCOVERED"
    assert decisions[0].as_dict()["decision"] == "cleared" and decisions[0].as_dict()["refusal"] is None
    assert digest_json([d.as_dict() for d in decisions if d.refusal is None])


def test_input_decisions_apply_terms_and_release_per_input():
    config = run(("synthetic", SYNTHETIC_URI), (CC_VOLUME_ID, CC_VOLUME_URI))
    assert [d.refusal.code if d.refusal else None for d in input_decisions(config, registry_doc())] == [
        None, "LICENCE_TERMS_ABSENT"]
    config["terms_accepted_by"] = acceptance("BUCKET-TERMS")
    assert all(d.refusal is None for d in input_decisions(config, registry_doc()))
    decisions = input_decisions(config, registry_doc(), for_release=True)
    assert [d.refusal.code if d.refusal else None for d in decisions] == [None, "LICENCE_RELEASE_PUBLICATION"]


def test_input_decisions_raise_refusals_that_concern_the_whole_run():
    with pytest.raises(Refusal) as info:
        input_decisions({"inputs": []}, registry_doc())
    assert info.value.code == "LICENCE_RUN_MALFORMED"
    with pytest.raises(Refusal) as info:
        input_decisions(run(("synthetic", SYNTHETIC_URI), accepted=acceptance("BUCKET-TERMS", person="A and B")),
                        registry_doc())
    assert info.value.code == "LICENCE_ACCEPTANCE_MALFORMED"


# --- the shipped registry, end to end -------------------------------------------------------


def _pinned() -> dict[str, str | None]:
    return {tid: t.text_sha256 for tid, t in load_registry().terms.items()}


def _declared(*entries: tuple[str, dict], hosts: dict | None = None) -> dict:
    """The shipped registry with a user's own assets (and hosts) added, as docs/licence.md describes."""
    doc = read_registry(SHIPPED_REGISTRY)
    doc["hosts"].update(hosts or {})
    for aid, entry in entries:
        doc["assets"][aid] = entry
    return doc


def test_shipped_registry_declares_no_scroll_data():
    uri = "https://dl.ash2txt.org/datasets/example/umbilicus.json"
    with pytest.raises(Refusal) as info:
        check_run(run(("example-dataset", uri)))
    assert info.value.code == "LICENCE_ASSET_UNKNOWN"
    assert [a for a in load_registry().assets] == ["apok-synthetic-fixtures"]


def test_shipped_registry_clears_synthetic_fixtures_without_terms():
    clearance = check_run(run(("apok-synthetic-fixtures", "synthetic://apok-fixtures/spiral/seed-7")))
    assert clearance.inputs[0].licence == "MIT" and clearance.acceptance is None
    assert clearance.as_dict()["registry"]["source"] == "apok_eval.licence/registry.json"


def test_data_server_data_a_user_declares_needs_the_pinned_server_terms():
    base = "https://dl.ash2txt.org/datasets/example"
    doc = _declared(("example-dataset", asset("dataset", [base], "EDUCELAB-SCROLLS")))
    config = run(("example-dataset", base + "/eval_fibers/a.json"))
    with pytest.raises(Refusal) as info:
        check_run(config, doc)
    assert info.value.code == "LICENCE_TERMS_ABSENT" and info.value.details["required"] == ["DATA-SERVER-LICENSE"]
    config["terms_accepted_by"] = acceptance("DATA-SERVER-LICENSE")
    with pytest.raises(Refusal) as info:
        check_run(config, doc)  # the text is pinned: an acceptance that cites none does not match it
    assert info.value.code == "LICENCE_TERMS_DIGEST_MISMATCH"
    config["terms_accepted_by"] = acceptance("DATA-SERVER-LICENSE", digests=_pinned())
    clearance = check_run(config, doc)
    assert clearance.inputs[0].licence == "EDUCELAB-SCROLLS" and clearance.inputs[0].host == "https://dl.ash2txt.org"
    with pytest.raises(Refusal) as info:
        check_run(config, doc, for_release=True)
    assert info.value.code == "LICENCE_RELEASE_PUBLICATION"
    with pytest.raises(Refusal) as info:
        check_run(run(("example-dataset", "https://dl.ash2txt.org/datasets/other/x.json"),
                      accepted=config["terms_accepted_by"]), doc)
    assert info.value.code == "LICENCE_URI_OUTSIDE_ASSET"  # only what was declared clears


def test_a_bucket_volume_a_user_declares_is_cc_by_nc_and_needs_the_data_terms():
    volume_id = "20990101000003"
    uri = f"s3://vesuvius-challenge-open-data/Example/volumes/{volume_id}-test.zarr"
    doc = _declared((volume_id, asset("volume", [uri], "CC-BY-NC-4.0", volume=("Example", volume_id))))
    config = run((volume_id, uri + "/0/.zarray"))
    with pytest.raises(Refusal) as info:
        check_run(config, doc)
    assert info.value.details["required"] == ["VESUVIUS-DATA-TERMS"]
    config["terms_accepted_by"] = acceptance("VESUVIUS-DATA-TERMS", digests=_pinned())
    assert check_run(config, doc).inputs[0].licence == "CC-BY-NC-4.0"


def test_no_registry_can_label_bucket_or_data_server_data_mit():
    for prefix in ("s3://vesuvius-challenge-open-data/Example", "https://dl.ash2txt.org/datasets/example"):
        doc = _declared(("relabelled", asset("dataset", [prefix], "MIT")), hosts={prefix: host("MIT")})
        with pytest.raises(Refusal) as info:
            check_run(run(("relabelled", prefix + "/x")), doc)
        assert info.value.code == "LICENCE_REGISTRY_GATED_HOST"


def test_a_local_copy_is_declared_under_a_file_root_with_the_licence_of_its_source(tmp_path):
    root = tmp_path / "data"
    (root / "scroll").mkdir(parents=True)
    copy_uri, root_uri = (root / "scroll").as_uri(), root.as_uri()
    doc = _declared(("my-local-copy", asset("dataset", [copy_uri], "CC-BY-NC-4.0")),
                    hosts={root_uri: host("CC-BY-NC-4.0", note="my local disk")})
    config = run(("my-local-copy", str(root / "scroll" / "fibres")))
    with pytest.raises(Refusal) as info:
        check_run(config, doc)
    assert info.value.code == "LICENCE_TERMS_ABSENT" and info.value.details["required"] == ["VESUVIUS-DATA-TERMS"]
    config["terms_accepted_by"] = acceptance("VESUVIUS-DATA-TERMS", digests=_pinned())
    clearance = check_run(config, doc)
    assert clearance.inputs[0].host == root_uri and clearance.acceptance.person == "Jane Example"
    with pytest.raises(Refusal) as info:
        check_run(run(("my-local-copy", str(tmp_path / "elsewhere")), accepted=config["terms_accepted_by"]), doc)
    assert info.value.code == "LICENCE_URI_OUTSIDE_ASSET"
