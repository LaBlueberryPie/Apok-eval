"""The licence gate: clear a run's inputs against the asset registry.

A run configuration is a JSON object with ``inputs``, a non-empty list of
``{"asset": <registry id>, "uri": <location>}``, and optionally
``terms_accepted_by``: ``{"person": <one individual>, "date": "YYYY-MM-DD",
"accepted": [{"terms": <terms id>, "text_sha256": <digest> | null}, ...]}``.
Other top-level keys are ignored (other modules extend the run
configuration); ``terms_accepted_by: null`` means absent. Only the person
named in ``terms_accepted_by`` writes it: a tool never accepts terms on
anyone's behalf. ``vesuvius.accept_terms`` is not evidence of
acceptance: it writes the three bytes ``yes`` to a package-local
``agreement.txt`` that nothing reads (villa fb8c2c4
``vesuvius/src/vesuvius/install/accept_terms.py:41-48``).

:func:`check_run` refuses, in this order, the first of:

1. the registry is invalid (``LICENCE_REGISTRY_*``, see :mod:`.registry`);
   :func:`licence_check` validates the registry before it reads the run
   configuration file (``LICENCE_RUN_UNREADABLE``);
2. ``inputs`` is not a non-empty list of objects with exactly string
   ``asset`` and ``uri`` (``LICENCE_RUN_MALFORMED``);
3. ``terms_accepted_by`` is present but malformed - not an object with exactly
   person, date and accepted; a person that is empty, not a string (a list of
   persons), names several people (after NFKC normalisation: a list
   separator, including ``、``, ``،`` and ``•``, or the word and, und, et or
   et al), names a group or an organisation (team, everyone, all, members,
   staff, group, on behalf of, Ltd, Inc, ...) or contains no letter; a date
   that is not a real ``YYYY-MM-DD`` day; ``accepted`` not a non-empty list
   (``LICENCE_ACCEPTANCE_MALFORMED``);
4. then entry by entry through ``accepted``: an entry that is not exactly
   ``{terms, text_sha256}``, names terms the registry does not declare, cites
   something that is neither null nor a digest, or repeats an earlier entry's
   terms (``LICENCE_ACCEPTANCE_MALFORMED``), or cites a text digest other than
   the one the registry pins for those terms - null included
   (``LICENCE_TERMS_DIGEST_MISMATCH``);
5. per input, in order: the asset id is not in the registry
   (``LICENCE_ASSET_UNKNOWN``; an alias is named as such), the asset has
   ``licence: null`` (``LICENCE_ASSET_UNLICENSED``), the URI cannot be
   canonicalised (``LICENCE_URI_*``), it is not under one of the asset's
   locations (``LICENCE_URI_OUTSIDE_ASSET``), it is under no declared host or
   local root (``LICENCE_HOST_UNDECLARED``), or that host does not allow the
   asset's licence (``LICENCE_HOST_DISALLOWS``);
6. the inputs require terms (their licences' ``requires_terms``, the
   ``requires_terms`` of every host prefix they are under, and their assets'
   ``additional_terms``) and there is no acceptance (``LICENCE_TERMS_ABSENT``),
   or the acceptance misses some of them (``LICENCE_TERMS_UNCOVERED``, naming
   the missing ids);
7. with ``for_release``, per input: the licence's publication is not
   "allowed" (``LICENCE_RELEASE_PUBLICATION``), it is not shippable in the
   pipeline (``LICENCE_RELEASE_UNSHIPPABLE``), or the asset carries
   additional terms, which the gate cannot evaluate for release
   (``LICENCE_RELEASE_ADDITIONAL_TERMS``).

Otherwise it issues a :class:`LicenceClearance`.

:func:`input_decisions` judges each input on its own, for a report: the
command line prints it when the gate refuses, so every input's decision is
shown, not only the first refusal. It never clears a run.
"""

from __future__ import annotations

import datetime as _dt
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping

from apok_eval.canonical import is_digest
from apok_eval.licence.registry import Registry, load_registry, parse_registry, read_strict_json
from apok_eval.licence.uri import CanonicalURI, canonical_uri
from apok_eval.refusal import Refusal, register_codes

register_codes(
    "apok_eval.licence.gate",
    {
        "LICENCE_RUN_UNREADABLE": "a run configuration file cannot be read as strict JSON",
        "LICENCE_RUN_MALFORMED": "a run configuration's inputs are not a non-empty list of {asset, uri} objects",
        "LICENCE_ASSET_UNKNOWN": "an input names an asset id that is not in the registry",
        "LICENCE_ASSET_UNLICENSED": "an input names an asset declared with licence null",
        "LICENCE_URI_OUTSIDE_ASSET": "an input URI is not under any of its asset's declared locations",
        "LICENCE_HOST_UNDECLARED": "an input URI is not under any declared host or local root",
        "LICENCE_HOST_DISALLOWS": "an input's asset licence is not allowed on the host its URI is on",
        "LICENCE_ACCEPTANCE_MALFORMED": "terms_accepted_by does not name one person, a real date and known terms",
        "LICENCE_TERMS_DIGEST_MISMATCH": "an acceptance cites a terms text digest other than the registry's pin",
        "LICENCE_TERMS_ABSENT": "inputs require terms but the run configuration carries no terms_accepted_by",
        "LICENCE_TERMS_UNCOVERED": "terms_accepted_by does not cover every terms id the inputs require",
        "LICENCE_RELEASE_PUBLICATION": "for release: an input's licence does not allow publication",
        "LICENCE_RELEASE_UNSHIPPABLE": "for release: an input's licence is not shippable in the pipeline",
        "LICENCE_RELEASE_ADDITIONAL_TERMS": "for release: an input's asset carries additional terms",
        "LICENCE_CLEARANCE_UNISSUED": "a LicenceClearance was constructed outside the licence gate",
    },
)

CLEARANCE_SCHEMA = "apok-licence-clearance-v1"
_ACCEPTANCE_FIELDS = {"person", "date", "accepted"}
_ACCEPTED_FIELDS = {"terms", "text_sha256"}
# Matched after NFKC normalisation, which folds fullwidth forms (＆ ， ／ ；) into ASCII. A list separator - ASCII,
# the ideographic comma 、, the Arabic comma and semicolon ، ؛, a bullet • or a fraction or division slash - or a
# joining word ("and", "und", "et", "et al.") means more than one person.
_SEVERAL_PEOPLE = re.compile(r"[,;&/+|\u3001\u060c\u061b\u2022\u2044\u2215]|\b(?:and|und|et)\b", re.IGNORECASE)
# A group, or an organisation, is not one individual (the terms bind each team member separately).
_NOT_AN_INDIVIDUAL = re.compile(r"\b(?:team|everyone|everybody|all|members|staff|group|on\s+behalf\s+of"
                                r"|ltd|limited|inc|llc|gmbh|plc|corp|company)\b", re.IGNORECASE)
_ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_ISSUE = object()  # held only by check_run; see LicenceClearance


@dataclass(frozen=True)
class AcceptedTerms:
    terms: str
    title: str
    text_sha256: str | None
    pinned: bool  # the registry pins this text, and the acceptance cites the pinned digest
    per_person: bool  # the acceptance binds only the named person

    def as_dict(self) -> dict[str, Any]:
        return {"terms": self.terms, "title": self.title, "text_sha256": self.text_sha256, "pinned": self.pinned,
                "per_person": self.per_person}


@dataclass(frozen=True)
class Acceptance:
    person: str
    date: str
    accepted: tuple[AcceptedTerms, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"person": self.person, "date": self.date, "accepted": [a.as_dict() for a in self.accepted]}


@dataclass(frozen=True)
class ClearedInput:
    index: int
    asset: str
    kind: str
    given_uri: str
    uri: CanonicalURI
    host: str
    licence: str
    licence_name: str
    licence_url: str
    obligations: tuple[str, ...]
    additional_terms: tuple[str, ...]
    required_terms: tuple[str, ...]
    publication: str
    commercial_use: bool
    shippable_in_pipeline: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "asset": self.asset, "kind": self.kind, "given_uri": self.given_uri,
            "uri": str(self.uri), "host": self.host, "licence": self.licence, "licence_name": self.licence_name,
            "licence_url": self.licence_url, "obligations": list(self.obligations),
            "additional_terms": list(self.additional_terms), "required_terms": list(self.required_terms),
            "publication": self.publication, "commercial_use": self.commercial_use,
            "shippable_in_pipeline": self.shippable_in_pipeline, "decision": "cleared",
        }


@dataclass(frozen=True)
class InputDecision:
    """One input judged on its own (see :func:`input_decisions`); not a clearance."""

    index: int
    asset: str
    given_uri: str
    licence: str | None  # the asset's licence in the registry; None if the asset is unknown or unlicensed
    obligations: tuple[str, ...]
    additional_terms: tuple[str, ...]
    refusal: Refusal | None  # None: the input would clear as the run's only input

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "asset": self.asset, "given_uri": self.given_uri, "licence": self.licence,
            "obligations": list(self.obligations), "additional_terms": list(self.additional_terms),
            "decision": "cleared" if self.refusal is None else "refused",
            "refusal": None if self.refusal is None else self.refusal.as_dict(),
        }


class LicenceClearance:
    """What the gate cleared for one run; implements :class:`apok_eval.clearance.Clearance`.

    Issued only by :func:`check_run`: the constructor requires a module-private
    token and refuses (``LICENCE_CLEARANCE_UNISSUED``) without it. Immutable:
    attributes cannot be set or deleted, every collection is a tuple, and copy
    and pickle are refused for the same reason (they rebuild the object by
    setting attributes); use :meth:`as_dict` to record one. Python cannot stop
    deliberate forgery (``object.__new__`` plus ``object.__setattr__``); the
    token stops a clearance being made by accident or by a careless caller.
    """

    __slots__ = ("_inputs", "_acceptance", "_for_release", "_registry_digest", "_registry_source")

    def __init__(self, token: object, *, inputs: tuple[ClearedInput, ...], acceptance: Acceptance | None,
                 for_release: bool, registry_digest: str, registry_source: str) -> None:
        if token is not _ISSUE:
            raise Refusal("LICENCE_CLEARANCE_UNISSUED", "a LicenceClearance is issued only by "
                          "apok_eval.licence.check_run", token=repr(token))
        for name, value in (("_inputs", tuple(inputs)), ("_acceptance", acceptance), ("_for_release", for_release),
                            ("_registry_digest", registry_digest), ("_registry_source", registry_source)):
            object.__setattr__(self, name, value)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"LicenceClearance is immutable; cannot set {name!r}")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"LicenceClearance is immutable; cannot delete {name!r}")

    @property
    def inputs(self) -> tuple[ClearedInput, ...]:
        return self._inputs

    @property
    def acceptance(self) -> Acceptance | None:
        return self._acceptance

    @property
    def for_release(self) -> bool:
        return self._for_release

    @property
    def registry_digest(self) -> str:
        return self._registry_digest

    @property
    def required_terms(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(t for item in self._inputs for t in item.required_terms))

    def covers(self, uri: str) -> bool:
        """True only if ``uri``, canonicalised now, is under a cleared input's canonical URI.

        A URI that cannot be canonicalised is not covered (False, never an error).
        """
        try:
            target = canonical_uri(uri, what="covers() uri")
        except Refusal:
            return False
        return any(target.is_under(item.uri) for item in self._inputs)

    def as_dict(self) -> dict[str, Any]:
        """A canonical-JSON-serialisable record of the clearance."""
        return {
            "schema": CLEARANCE_SCHEMA,
            "registry": {"digest": self._registry_digest, "source": self._registry_source},
            "for_release": self._for_release,
            "inputs": [item.as_dict() for item in self._inputs],
            "required_terms": list(self.required_terms),
            "acceptance": None if self._acceptance is None else self._acceptance.as_dict(),
        }

    def __repr__(self) -> str:
        assets = ", ".join(item.asset for item in self._inputs)
        return f"LicenceClearance(inputs=[{assets}], for_release={self._for_release})"


# --- reading ----------------------------------------------------------------------


def load_run_config(path: str | os.PathLike[str]) -> Any:
    """The run configuration at ``path``, read as strict JSON (validated by :func:`check_run`)."""
    doc, problem = read_strict_json(path)
    if problem is not None:
        raise Refusal("LICENCE_RUN_UNREADABLE", f"run configuration {os.fspath(path)!r} is not readable strict "
                      f"JSON: {problem}", path=os.fspath(path), problem=problem)
    return doc


def licence_check(run_config: str | os.PathLike[str], registry: str | os.PathLike[str] | None = None, *,
                  for_release: bool = False) -> LicenceClearance:
    """:func:`check_run` on the run configuration file ``run_config``; the registry is validated first."""
    reg = _registry(registry)
    return _check(load_run_config(run_config), reg, for_release)


def _registry(registry: str | os.PathLike[str] | Mapping[str, Any] | None) -> Registry:
    if isinstance(registry, Mapping):
        return parse_registry(registry)
    if registry is None or isinstance(registry, (str, os.PathLike)):
        return load_registry(registry)
    raise TypeError(f"registry must be a path, a registry document (a mapping) or None, not "
                    f"{type(registry).__name__}: the gate validates the registry itself on every call")


# --- the gate -----------------------------------------------------------------------


def check_run(run: Any, registry: str | os.PathLike[str] | Mapping[str, Any] | None = None, *,
              for_release: bool = False) -> LicenceClearance:
    """Clear ``run`` against ``registry`` (a path, a parsed document, or None for the shipped one).

    Refuses as listed in the module docstring; returns the clearance otherwise.
    """
    return _check(run, _registry(registry), for_release)


def input_decisions(run: Any, registry: str | os.PathLike[str] | Mapping[str, Any] | None = None, *,
                    for_release: bool = False) -> tuple[InputDecision, ...]:
    """Each input of ``run`` judged on its own, for a report; never a clearance.

    An input is cleared here when it would clear as the run's only input: its
    asset, URI and host pass, the acceptance covers the terms it requires and,
    with ``for_release``, it passes the release checks. Otherwise its decision
    carries the refusal that stops it. A refusal that concerns the whole run
    (the registry, the run's shape, a malformed acceptance) is raised as by
    :func:`check_run`.
    """
    reg = _registry(registry)
    inputs = _inputs(run)
    acceptance = _acceptance(run.get("terms_accepted_by"), reg)
    return tuple(_decide(i, item, reg, acceptance, for_release) for i, item in enumerate(inputs))


def _check(run: Any, reg: Registry, for_release: bool) -> LicenceClearance:
    inputs = _inputs(run)
    acceptance = _acceptance(run.get("terms_accepted_by"), reg)
    cleared = tuple(_clear_input(i, item, reg) for i, item in enumerate(inputs))
    _check_terms(cleared, acceptance)
    if for_release:
        _check_release(cleared)
    return LicenceClearance(_ISSUE, inputs=cleared, acceptance=acceptance, for_release=for_release,
                            registry_digest=reg.digest, registry_source=reg.source)


def _decide(i: int, item: dict[str, str], reg: Registry, acceptance: Acceptance | None,
            for_release: bool) -> InputDecision:
    refusal = None
    try:
        cleared = _clear_input(i, item, reg)
        _check_terms((cleared,), acceptance)
        if for_release:
            _check_release((cleared,))
    except Refusal as exc:
        refusal = exc
    asset = reg.assets.get(item["asset"])
    lic = reg.licences.get(asset.licence) if asset is not None and asset.licence is not None else None
    return InputDecision(index=i, asset=item["asset"], given_uri=item["uri"],
                         licence=None if lic is None else lic.id, obligations=() if lic is None else lic.obligations,
                         additional_terms=() if asset is None else asset.additional_terms, refusal=refusal)


def _inputs(run: Any) -> list[dict[str, str]]:
    if not isinstance(run, dict):
        raise Refusal("LICENCE_RUN_MALFORMED", "a run configuration must be a JSON object", field="$")
    inputs = run.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise Refusal("LICENCE_RUN_MALFORMED", "a run configuration needs 'inputs', a non-empty list",
                      field="inputs", value=inputs)
    for i, item in enumerate(inputs):
        if (not isinstance(item, dict) or set(item) != {"asset", "uri"}
                or not isinstance(item["asset"], str) or not isinstance(item["uri"], str)):
            raise Refusal("LICENCE_RUN_MALFORMED", f"inputs[{i}] must be an object with exactly the strings "
                          "'asset' and 'uri'", field=f"inputs[{i}]", value=item)
    return inputs


def _person_problem(person: str) -> str | None:
    name = unicodedata.normalize("NFKC", person).strip()
    if not name:
        return "is empty"
    if not name.isprintable():
        return "contains a control character or line break"
    if _SEVERAL_PEOPLE.search(name):
        return ("names more than one person (a list separator, 'and', 'und', 'et' or 'et al'); each team member "
                "accepts separately")
    if _NOT_AN_INDIVIDUAL.search(name):
        return ("names a group or an organisation ('team', 'everyone', 'on behalf of', 'Ltd', ...); the terms bind "
                "one individual")
    if not any(ch.isalpha() for ch in name):
        return "contains no letter"
    return None


def _is_real_date(value: Any) -> bool:
    if not isinstance(value, str) or not _ISO_DATE.fullmatch(value):
        return False
    try:
        _dt.date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _acceptance(raw: Any, reg: Registry) -> Acceptance | None:
    if raw is None:
        return None
    if not isinstance(raw, dict) or set(raw) != _ACCEPTANCE_FIELDS:
        raise Refusal("LICENCE_ACCEPTANCE_MALFORMED", "terms_accepted_by must be an object with exactly person, "
                      "date and accepted", field="terms_accepted_by",
                      keys=sorted(map(str, raw)) if isinstance(raw, dict) else type(raw).__name__)
    person = raw["person"]
    if not isinstance(person, str):
        raise Refusal("LICENCE_ACCEPTANCE_MALFORMED", "terms_accepted_by.person must be one individual's name as a "
                      f"string, not {type(person).__name__}; each team member accepts separately",
                      field="terms_accepted_by.person", value=person)
    problem = _person_problem(person)
    if problem is not None:
        raise Refusal("LICENCE_ACCEPTANCE_MALFORMED", f"terms_accepted_by.person {problem}",
                      field="terms_accepted_by.person", value=person)
    if not _is_real_date(raw["date"]):
        raise Refusal("LICENCE_ACCEPTANCE_MALFORMED", "terms_accepted_by.date must be a real day written "
                      "YYYY-MM-DD", field="terms_accepted_by.date", value=raw["date"])
    entries = raw["accepted"]
    if not isinstance(entries, list) or not entries:
        raise Refusal("LICENCE_ACCEPTANCE_MALFORMED", "terms_accepted_by.accepted must be a non-empty list",
                      field="terms_accepted_by.accepted", value=entries)
    accepted: dict[str, AcceptedTerms] = {}
    for j, entry in enumerate(entries):
        accepted_entry = _accepted(j, entry, reg)
        if accepted_entry.terms in accepted:
            raise Refusal("LICENCE_ACCEPTANCE_MALFORMED", f"terms_accepted_by.accepted names "
                          f"{accepted_entry.terms!r} twice", field=f"terms_accepted_by.accepted[{j}]",
                          terms=accepted_entry.terms)
        accepted[accepted_entry.terms] = accepted_entry
    return Acceptance(person=person.strip(), date=raw["date"], accepted=tuple(accepted.values()))


def _accepted(j: int, entry: Any, reg: Registry) -> AcceptedTerms:
    field = f"terms_accepted_by.accepted[{j}]"
    if not isinstance(entry, dict) or set(entry) != _ACCEPTED_FIELDS:
        raise Refusal("LICENCE_ACCEPTANCE_MALFORMED", f"{field} must be an object with exactly terms and "
                      "text_sha256", field=field, value=entry)
    tid, cited = entry["terms"], entry["text_sha256"]
    if not isinstance(tid, str) or tid not in reg.terms:
        raise Refusal("LICENCE_ACCEPTANCE_MALFORMED", f"{field}.terms {tid!r} is not a terms id the registry "
                      "declares", field=f"{field}.terms", value=tid, known=sorted(reg.terms))
    if cited is not None and not is_digest(cited):
        raise Refusal("LICENCE_ACCEPTANCE_MALFORMED", f"{field}.text_sha256 must be null or a sha256 digest",
                      field=f"{field}.text_sha256", value=cited)
    terms = reg.terms[tid]
    if terms.text_sha256 is not None and cited != terms.text_sha256:
        raise Refusal("LICENCE_TERMS_DIGEST_MISMATCH", f"the acceptance of {tid!r} cites text {cited}, but the "
                      f"registry pins {terms.text_sha256}: the accepted text is not the registered one",
                      terms=tid, cited=cited, pinned=terms.text_sha256)
    return AcceptedTerms(terms=tid, title=terms.title, text_sha256=cited, pinned=terms.text_sha256 is not None,
                         per_person=terms.per_person)


def _clear_input(i: int, item: dict[str, str], reg: Registry) -> ClearedInput:
    name = item["asset"]
    asset = reg.assets.get(name)
    if asset is None:
        owner = reg.name_owner(name)
        hint = "" if owner is None else f"; it is a name of {owner!r} - inputs must give the registry id"
        raise Refusal("LICENCE_ASSET_UNKNOWN", f"inputs[{i}]: asset {name!r} is not in the registry{hint}",
                      input=i, asset=name, name_of=owner)
    if asset.licence is None:
        raise Refusal("LICENCE_ASSET_UNLICENSED", f"inputs[{i}]: asset {name!r} is declared without a licence "
                      f"(licence: null) and is refused until its licence is declared. Note: {asset.note or '-'}",
                      input=i, asset=name, note=asset.note)
    uri = canonical_uri(item["uri"], what=f"inputs[{i}].uri")
    if not any(uri.is_under(location) for location in asset.locations):
        raise Refusal("LICENCE_URI_OUTSIDE_ASSET", f"inputs[{i}]: {str(uri)!r} is not under any location of "
                      f"asset {name!r}", input=i, asset=name, uri=str(uri),
                      locations=[str(location) for location in asset.locations])
    host = reg.host_for(uri)
    if host is None:
        raise Refusal("LICENCE_HOST_UNDECLARED", f"inputs[{i}]: {str(uri)!r} is not under any host or local root "
                      "the registry declares", input=i, asset=name, uri=str(uri))
    if asset.licence not in host.allowed_licences:
        raise Refusal("LICENCE_HOST_DISALLOWS", f"inputs[{i}]: asset {name!r} carries licence {asset.licence!r}, "
                      f"which host {str(host.prefix)!r} does not allow ({list(host.allowed_licences)})",
                      input=i, asset=name, licence=asset.licence, host=str(host.prefix),
                      allowed=list(host.allowed_licences))
    lic = reg.licences[asset.licence]
    return ClearedInput(
        index=i, asset=name, kind=asset.kind, given_uri=item["uri"], uri=uri, host=str(host.prefix),
        licence=lic.id, licence_name=lic.name, licence_url=lic.url, obligations=lic.obligations,
        additional_terms=asset.additional_terms,
        required_terms=tuple(dict.fromkeys(lic.requires_terms + reg.host_terms_for(uri) + asset.additional_terms)),
        publication=lic.publication, commercial_use=lic.commercial_use,
        shippable_in_pipeline=lic.shippable_in_pipeline,
    )


def _check_terms(cleared: tuple[ClearedInput, ...], acceptance: Acceptance | None) -> None:
    required: dict[str, list[int]] = {}
    for item in cleared:
        for tid in item.required_terms:
            required.setdefault(tid, []).append(item.index)
    if not required:
        return
    if acceptance is None:
        raise Refusal("LICENCE_TERMS_ABSENT", f"the inputs require terms {list(required)}, but the run "
                      "configuration carries no terms_accepted_by, the record of your own acceptance",
                      required=list(required), required_by=required)
    accepted = {entry.terms for entry in acceptance.accepted}
    missing = [tid for tid in required if tid not in accepted]
    if missing:
        raise Refusal("LICENCE_TERMS_UNCOVERED", f"the acceptance by {acceptance.person!r} does not cover terms "
                      f"{missing}", missing=missing, required_by={tid: required[tid] for tid in missing},
                      person=acceptance.person)


def _check_release(cleared: tuple[ClearedInput, ...]) -> None:
    for item in cleared:
        if item.publication != "allowed":
            raise Refusal("LICENCE_RELEASE_PUBLICATION", f"inputs[{item.index}]: asset {item.asset!r} is under "
                          f"{item.licence!r}, whose publication is {item.publication!r}, not 'allowed'",
                          input=item.index, asset=item.asset, licence=item.licence, publication=item.publication)
        if not item.shippable_in_pipeline:
            raise Refusal("LICENCE_RELEASE_UNSHIPPABLE", f"inputs[{item.index}]: asset {item.asset!r} is under "
                          f"{item.licence!r}, which is not shippable in the pipeline", input=item.index,
                          asset=item.asset, licence=item.licence)
        if item.additional_terms:
            raise Refusal("LICENCE_RELEASE_ADDITIONAL_TERMS", f"inputs[{item.index}]: asset {item.asset!r} carries "
                          f"additional terms {list(item.additional_terms)}, which the gate cannot evaluate for "
                          "release; a person must decide", input=item.index, asset=item.asset, licence=item.licence,
                          additional_terms=list(item.additional_terms))
