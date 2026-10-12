"""The asset registry: everything apok-eval may read, with its licence.

Format ``apok-asset-registry-v1`` - one strict JSON object (see
:func:`read_strict_json`) with exactly the keys ``schema``,
``licences``, ``terms``, ``hosts``, ``assets`` and an optional ``note``.
Every entry lists its fields exactly; a missing or unknown field is refused,
never defaulted (field tables: ``docs/licence.md``).

* ``licences``: id -> {name, url, requires_terms, publication
  ("allowed" | "forbidden" | "ask"), commercial_use, shippable_in_pipeline,
  obligations, evidence}.
* ``terms``: id -> {title, url, per_person, text_sha256 (a digest, or null
  while the text is not pinned), evidence}.
* ``hosts``: canonical host or prefix (any spelling; canonicalised here) ->
  {allowed_licences, requires_terms (terms ids every input on or below the
  host needs, whatever its licence: a server's own access terms), note}. A URI
  belongs to the longest declared prefix it is under for its allowed
  licences, and needs the terms of every declared prefix it is under.
* ``assets``: id -> {kind, locations, licence (id, or null = declared but not
  yet licensed), additional_terms (terms ids), aliases, canonical_volume
  ({sample, volume_id}), evidence, note (optional)}. A ``volume`` asset is
  keyed by its 14-digit volume id, declares that id as its canonical volume,
  and each of its locations names the volume (last segment ``<id>``,
  ``<id>-...`` or ``<id>.<ext>``). A ``dataset`` or ``segment`` asset may
  declare the canonical volume it is a copy of or derived from; other kinds
  may not.

Validation, in order; the first failure refuses:

1. strict JSON (``LICENCE_REGISTRY_UNREADABLE``), the schema string
   (``LICENCE_REGISTRY_SCHEMA``), then shape: tables, ids
   (``[A-Za-z0-9][A-Za-z0-9._+-]*``), field sets and field types (text is
   non-blank Unicode: a lone surrogate is refused), host keys and asset
   locations canonicalised (``LICENCE_URI_*`` naming the registry field), no
   host prefix declared twice, no location listed twice in one asset, the
   volume rules above (``LICENCE_REGISTRY_MALFORMED``);
2. every referenced licence and terms id is declared
   (``LICENCE_REGISTRY_UNKNOWN_REF``);
3. a host on Vesuvius Challenge storage (:func:`is_gated`: the buckets in
   :data:`GATED_BUCKETS` and every host under :data:`GATED_DOMAINS`,
   hard-wired so that no registry can label data-server or bucket data as MIT
   or synthetic) allows only licences that require terms, forbid commercial
   use, are not shippable and are not freely publishable
   (``LICENCE_REGISTRY_GATED_HOST``);
4. a ``synthetic`` asset lives only on local storage
   (``LICENCE_REGISTRY_SYNTHETIC_REMOTE``); no name (asset id or alias,
   compared case-insensitively) belongs to two assets
   (``LICENCE_REGISTRY_MALFORMED``);
5. assets declaring the same 14-digit volume id (the volume itself and any
   dataset or segment declaring it) name the same sample after alias
   normalisation (``LICENCE_REGISTRY_VOLUME_SAMPLE``) and carry the same
   licence and additional terms (``LICENCE_REGISTRY_VOLUME_CONFLICT``);
6. assets whose locations overlap (one a whole-segment prefix of the other)
   carry the same licence and additional terms
   (``LICENCE_REGISTRY_LOCATION_OVERLAP``) - otherwise a URI could be claimed
   through the asset with fewer obligations.

The digest of a registry is the ``digest_json`` of the document as read.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping

from apok_eval import jsonstrict
from apok_eval.canonical import digest_json, is_digest, is_text
from apok_eval.fileio import read_regular
from apok_eval.licence.uri import CanonicalURI, canonical_uri
from apok_eval.refusal import Refusal, register_codes
from apok_eval.scrolls import canonical_scroll, scroll_key

register_codes(
    "apok_eval.licence.registry",
    {
        "LICENCE_REGISTRY_UNREADABLE": "a registry file cannot be read as strict JSON",
        "LICENCE_REGISTRY_SCHEMA": "a registry does not declare schema apok-asset-registry-v1",
        "LICENCE_REGISTRY_MALFORMED": "a registry entry has a missing, unknown or ill-typed field, a bad id, a "
                                      "duplicate host, location or name, or breaks the volume keying rules",
        "LICENCE_REGISTRY_UNKNOWN_REF": "a registry entry references a licence or terms id that is not declared",
        "LICENCE_REGISTRY_GATED_HOST": "a Vesuvius Challenge data host allows a licence that is not terms-gated, "
                                       "non-commercial, unshippable and unpublishable",
        "LICENCE_REGISTRY_SYNTHETIC_REMOTE": "a synthetic asset declares a location that is not local",
        "LICENCE_REGISTRY_VOLUME_SAMPLE": "two assets declare one volume id under different samples",
        "LICENCE_REGISTRY_VOLUME_CONFLICT": "two assets declare one volume with different licences or additional "
                                            "terms",
        "LICENCE_REGISTRY_LOCATION_OVERLAP": "two assets with different licences or additional terms claim "
                                             "overlapping locations",
    },
)

SCHEMA = "apok-asset-registry-v1"
SHIPPED_REGISTRY = Path(__file__).with_name("registry.json")
SHIPPED_SOURCE = "apok_eval.licence/registry.json"
PUBLICATION = ("allowed", "forbidden", "ask")
KINDS = ("volume", "dataset", "segment", "model-weights", "code", "synthetic")
#: Kinds that may name the canonical volume they copy or derive from (a volume must name itself).
VOLUME_KINDS = ("volume", "dataset", "segment")
_TABLES = ("licences", "terms", "hosts", "assets")

#: Buckets serving Vesuvius Challenge scroll data: the open-data bucket and the full-dataset bucket
#: behind the data.aws.ash2txt.org/samples gateway (villa fb8c2c4 scrollprize.org/src/data/atlasDataAccess.json:8-15).
GATED_BUCKETS = ("vesuvius-challenge-open-data", "vesuvius-challenge")
#: Domains whose every host is gated. ash2txt.org is Vesuvius Challenge's own: the data server dl. (its LICENSE.txt;
#: scrollprize.org/docs/38_tutorial_spiral.md:109), the gateway data.aws., the zarr proxy zarr.aws. and the
#: thumbnail services thumbs(-vc).aws. (atlasDataAccess.json:12-44), WebKnossos wk.aws.
#: (scrollprize.org/src/data/atlasOverlay.json:215) and volumes.aws. (lasagna/docs/tifxyz_training_integration.md:50).
GATED_DOMAINS = ("ash2txt.org",)

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
_VOLUME_ID = re.compile(r"[0-9]{14}")


def is_gated(location: CanonicalURI) -> bool:
    """True if ``location`` is on Vesuvius Challenge storage: a gated bucket, or a host under a gated domain."""
    name = location.authority
    return name in GATED_BUCKETS or any(name == d or name.endswith("." + d) for d in GATED_DOMAINS)


def canonical_sample(name: str) -> str:
    """Comparison key for a sample id (:mod:`apok_eval.scrolls`): the key of the catalog id a known name resolves
    to, else the name's own key (case and every character but a letter or digit ignored)."""
    return scroll_key(canonical_scroll(name) or name)


# --- typed records ------------------------------------------------------------------


@dataclass(frozen=True)
class Licence:
    id: str
    name: str
    url: str
    requires_terms: tuple[str, ...]
    publication: str
    commercial_use: bool
    shippable_in_pipeline: bool
    obligations: tuple[str, ...]
    evidence: str


@dataclass(frozen=True)
class Terms:
    id: str
    title: str
    url: str
    per_person: bool
    text_sha256: str | None
    evidence: str


@dataclass(frozen=True)
class Host:
    prefix: CanonicalURI
    allowed_licences: tuple[str, ...]
    requires_terms: tuple[str, ...]
    note: str


@dataclass(frozen=True)
class CanonicalVolume:
    sample: str
    volume_id: str


@dataclass(frozen=True)
class Asset:
    id: str
    kind: str
    locations: tuple[CanonicalURI, ...]
    licence: str | None
    additional_terms: tuple[str, ...]
    aliases: tuple[str, ...]
    canonical_volume: CanonicalVolume | None
    evidence: str
    note: str | None

    @property
    def obligations_key(self) -> tuple[str | None, frozenset[str]]:
        """What a URI under this asset obliges: its licence and additional terms."""
        return self.licence, frozenset(self.additional_terms)


@dataclass(frozen=True)
class Registry:
    """A validated registry. Build it with :func:`parse_registry` or :func:`load_registry` only."""

    licences: Mapping[str, Licence]
    terms: Mapping[str, Terms]
    hosts: tuple[Host, ...]
    assets: Mapping[str, Asset]
    names: Mapping[str, str]  # casefolded asset id or alias -> asset id
    digest: str
    source: str

    def host_for(self, uri: CanonicalURI) -> Host | None:
        """The declared host with the longest prefix ``uri`` is under, or None."""
        matches = [host for host in self.hosts if uri.is_under(host.prefix)]
        return max(matches, key=lambda host: len(host.prefix.segments), default=None)

    def host_terms_for(self, uri: CanonicalURI) -> tuple[str, ...]:
        """The ``requires_terms`` of every declared host ``uri`` is under, widest first.

        A server's access terms bind everything below it: a narrower host entry chooses the allowed
        licences, but cannot drop the terms of an entry that encloses it.
        """
        matches = sorted((host for host in self.hosts if uri.is_under(host.prefix)),
                         key=lambda host: len(host.prefix.segments))
        return tuple(dict.fromkeys(t for host in matches for t in host.requires_terms))

    def name_owner(self, name: str) -> str | None:
        """The asset id that ``name`` (an id or alias, any case) belongs to, or None."""
        return self.names.get(name.casefold())


# --- field shapes -------------------------------------------------------------------

Check = Callable[[Any], bool]


def _flag(v: Any) -> bool:
    return isinstance(v, bool)


def _ident(v: Any) -> bool:
    return isinstance(v, str) and bool(_ID.fullmatch(v))


def _idents(v: Any) -> bool:
    return isinstance(v, list) and all(_ident(x) for x in v) and len(set(v)) == len(v)


def _texts(v: Any) -> bool:
    return isinstance(v, list) and all(is_text(x) for x in v)


def _names(v: Any) -> bool:
    return _texts(v) and len({x.casefold() for x in v}) == len(v)


def _digest_or_null(v: Any) -> bool:
    return v is None or is_digest(v)


def _publication(v: Any) -> bool:
    return isinstance(v, str) and v in PUBLICATION


def _kind(v: Any) -> bool:
    return isinstance(v, str) and v in KINDS


def _licence_ref(v: Any) -> bool:
    return v is None or _ident(v)


def _uris(v: Any) -> bool:
    return isinstance(v, list) and len(v) > 0 and all(isinstance(x, str) for x in v)


def _volume(v: Any) -> bool:
    return (isinstance(v, dict) and set(v) == {"sample", "volume_id"} and is_text(v["sample"])
            and isinstance(v["volume_id"], str) and bool(_VOLUME_ID.fullmatch(v["volume_id"])))


_TEXT = (is_text, "a non-empty string")  # a lone surrogate is not text: it could not be digested
_LICENCE_FIELDS: dict[str, tuple[Check, str]] = {
    "name": _TEXT,
    "url": _TEXT,
    "requires_terms": (_idents, "a list of distinct terms ids"),
    "publication": (_publication, 'one of "allowed", "forbidden", "ask"'),
    "commercial_use": (_flag, "true or false"),
    "shippable_in_pipeline": (_flag, "true or false"),
    "obligations": (_texts, "a list of non-empty strings"),
    "evidence": _TEXT,
}
_TERMS_FIELDS: dict[str, tuple[Check, str]] = {
    "title": _TEXT,
    "url": _TEXT,
    "per_person": (_flag, "true or false"),
    "text_sha256": (_digest_or_null, 'null or a "sha256:<64 hex>" digest'),
    "evidence": _TEXT,
}
_HOST_FIELDS: dict[str, tuple[Check, str]] = {
    "allowed_licences": (_idents, "a list of distinct licence ids"),
    "requires_terms": (_idents, "a list of distinct terms ids"),
    "note": _TEXT,
}
_ASSET_FIELDS: dict[str, tuple[Check, str]] = {
    "kind": (_kind, f"one of {', '.join(KINDS)}"),
    "locations": (_uris, "a non-empty list of URI strings"),
    "licence": (_licence_ref, "a licence id or null"),
    "additional_terms": (_idents, "a list of distinct terms ids"),
    "aliases": (_names, "a list of distinct non-empty names"),
    "canonical_volume": (_volume, '{"sample": <sample id>, "volume_id": <14 digits>}'),
    "evidence": _TEXT,
    "note": _TEXT,
}
_OPTIONAL = {"licences": set(), "terms": set(), "hosts": set(), "assets": {"canonical_volume", "note"}}
_FIELDS = {"licences": _LICENCE_FIELDS, "terms": _TERMS_FIELDS, "hosts": _HOST_FIELDS, "assets": _ASSET_FIELDS}


def _entries(doc: dict[str, Any], table: str, source: str) -> Iterator[tuple[str, dict[str, Any]]]:
    fields, optional = _FIELDS[table], _OPTIONAL[table]
    for key, entry in doc[table].items():
        if table != "hosts" and not _ident(key):
            raise Refusal("LICENCE_REGISTRY_MALFORMED", f"{table} id {key!r} is not of the form "
                          "[A-Za-z0-9][A-Za-z0-9._+-]*", source=source, table=table, entry=key, field="id")
        if not isinstance(entry, dict):
            raise Refusal("LICENCE_REGISTRY_MALFORMED", f"{table}[{key!r}] must be a JSON object",
                          source=source, table=table, entry=key)
        missing = sorted(set(fields) - optional - entry.keys())
        unknown = sorted(entry.keys() - set(fields))
        if missing or unknown:
            raise Refusal("LICENCE_REGISTRY_MALFORMED", f"{table}[{key!r}] lacks {missing} or has unknown fields "
                          f"{unknown}", source=source, table=table, entry=key, missing=missing, unknown=unknown)
        for field, (ok, expected) in fields.items():
            if field in entry and not ok(entry[field]):
                raise Refusal("LICENCE_REGISTRY_MALFORMED", f"{table}[{key!r}].{field} must be {expected}",
                              source=source, table=table, entry=key, field=field, value=entry[field])
        yield key, entry


# --- loading ------------------------------------------------------------------------


def read_strict_json(path: str | os.PathLike[str]) -> tuple[Any, str | None]:
    """``(document, None)``, or ``(None, problem)`` if ``path`` cannot be read or is not strict JSON.

    Strict JSON is :func:`apok_eval.jsonstrict.loads`: a duplicated key (``"licence": "EDUCELAB-SCROLLS"`` then
    ``"licence": "MIT"``, or an acceptance followed by ``null``), NaN/Infinity, an overflowing number, a lone
    surrogate, invalid UTF-8 or a BOM is a problem, never resolved silently. So is a path ``open`` refuses
    with a ``ValueError`` rather than an ``OSError`` (an embedded NUL, an unencodable surrogate).
    """
    try:
        return jsonstrict.loads(read_regular(path)), None
    except (OSError, ValueError) as exc:  # StrictJsonError is a ValueError; NotRegularFile (a FIFO, a device) an OSError
        return None, jsonstrict.problem_of(exc)


def read_registry(path: str | os.PathLike[str]) -> Any:
    """The registry document at ``path``, read as strict JSON (not yet validated)."""
    doc, problem = read_strict_json(path)
    if problem is not None:
        raise Refusal("LICENCE_REGISTRY_UNREADABLE", f"registry {os.fspath(path)!r} is not readable strict JSON: "
                      f"{problem}", source=os.fspath(path), problem=problem)
    return doc


def load_registry(path: str | os.PathLike[str] | None = None) -> Registry:
    """Read and validate the registry at ``path`` (default: the one shipped with apok-eval)."""
    if path is None:
        return parse_registry(read_registry(SHIPPED_REGISTRY), SHIPPED_SOURCE)
    return parse_registry(read_registry(path), os.fspath(path))


def parse_registry(doc: Any, source: str = "<in-memory>") -> Registry:
    """Validate a registry document (see the module docstring) and return it typed and immutable."""
    if not isinstance(doc, dict):
        raise Refusal("LICENCE_REGISTRY_MALFORMED", "a registry must be a JSON object", source=source, table="$")
    if doc.get("schema") != SCHEMA:
        raise Refusal("LICENCE_REGISTRY_SCHEMA", f"registry schema is {doc.get('schema')!r}; this gate reads "
                      f"{SCHEMA!r} only", source=source, schema=doc.get("schema"))
    required, present = {"schema", *_TABLES}, set(doc)
    if not required <= present or present - required - {"note"}:
        raise Refusal("LICENCE_REGISTRY_MALFORMED", "a registry has exactly the keys schema, licences, terms, "
                      "hosts, assets and an optional note", source=source, table="$",
                      missing=sorted(required - present), unknown=sorted(present - required - {"note"}))
    bad_table = next((t for t in _TABLES if not isinstance(doc[t], dict)), None)
    if bad_table is not None or ("note" in doc and not is_text(doc["note"])):
        raise Refusal("LICENCE_REGISTRY_MALFORMED", f"registry table {bad_table or 'note'!r} has the wrong type "
                      "(tables are JSON objects, note a non-empty string)", source=source,
                      table=bad_table or "note")

    licences = {key: _licence(key, e) for key, e in _entries(doc, "licences", source)}
    terms = {key: _terms(key, e) for key, e in _entries(doc, "terms", source)}
    hosts = _hosts(doc, source)
    assets = {key: _asset(key, e, source) for key, e in _entries(doc, "assets", source)}

    _check_references(licences, terms, hosts, assets, source)
    _check_gated_hosts(hosts, licences, source)
    names = _check_assets(assets, source)
    _check_volumes(assets, source)
    _check_overlaps(assets, source)
    return Registry(licences=MappingProxyType(licences), terms=MappingProxyType(terms), hosts=hosts,
                    assets=MappingProxyType(assets), names=MappingProxyType(names), digest=digest_json(doc),
                    source=source)


def _licence(key: str, e: dict[str, Any]) -> Licence:
    return Licence(id=key, name=e["name"], url=e["url"], requires_terms=tuple(e["requires_terms"]),
                   publication=e["publication"], commercial_use=e["commercial_use"],
                   shippable_in_pipeline=e["shippable_in_pipeline"], obligations=tuple(e["obligations"]),
                   evidence=e["evidence"])


def _terms(key: str, e: dict[str, Any]) -> Terms:
    return Terms(id=key, title=e["title"], url=e["url"], per_person=e["per_person"], text_sha256=e["text_sha256"],
                 evidence=e["evidence"])


def _hosts(doc: dict[str, Any], source: str) -> tuple[Host, ...]:
    seen: dict[CanonicalURI, str] = {}
    hosts = []
    for key, e in _entries(doc, "hosts", source):
        prefix = canonical_uri(key, what=f"registry hosts[{key!r}]")
        if prefix in seen:
            raise Refusal("LICENCE_REGISTRY_MALFORMED", f"hosts {seen[prefix]!r} and {key!r} are the same prefix "
                          f"{str(prefix)!r}", source=source, table="hosts", entry=key, prefix=str(prefix),
                          other=seen[prefix])
        seen[prefix] = key
        hosts.append(Host(prefix=prefix, allowed_licences=tuple(e["allowed_licences"]),
                          requires_terms=tuple(e["requires_terms"]), note=e["note"]))
    return tuple(hosts)


def _asset(key: str, e: dict[str, Any], source: str) -> Asset:
    locations = tuple(canonical_uri(u, what=f"registry assets[{key!r}].locations[{i}]")
                      for i, u in enumerate(e["locations"]))
    if len(set(locations)) != len(locations):
        raise Refusal("LICENCE_REGISTRY_MALFORMED", f"assets[{key!r}] lists one location twice (after "
                      "canonicalisation)", source=source, table="assets", entry=key, field="locations",
                      locations=[str(u) for u in locations])
    volume = e.get("canonical_volume")
    if volume is not None and e["kind"] not in VOLUME_KINDS:
        raise Refusal("LICENCE_REGISTRY_MALFORMED", f"assets[{key!r}]: a {e['kind']!r} asset cannot name a "
                      f"canonical_volume (only {', '.join(VOLUME_KINDS)})", source=source, table="assets",
                      entry=key, field="canonical_volume")
    if e["kind"] == "volume":
        _check_volume_keying(key, volume, locations, source)
    return Asset(id=key, kind=e["kind"], locations=locations, licence=e["licence"],
                 additional_terms=tuple(e["additional_terms"]), aliases=tuple(e["aliases"]),
                 canonical_volume=None if volume is None else CanonicalVolume(volume["sample"], volume["volume_id"]),
                 evidence=e["evidence"], note=e.get("note"))


def _names_volume(location: CanonicalURI, volume_id: str) -> bool:
    # Folder names (villa fb8c2c4): <volume id>-<voxel>um-...-masked.zarr in the buckets
    # (scrollprize.org/static/data_browser/index.json:186-188), <volume id>.zarr in the legacy volumes_zarr
    # (vesuvius/src/vesuvius/install/configs/scrolls.yaml:32,38), <volume id> for a legacy TIFF stack
    # (scrollprize.org/docs/06_tutorial3.md:256).
    last = location.segments[-1] if location.segments else ""
    return last == volume_id or last.startswith((volume_id + "-", volume_id + "."))


def _check_volume_keying(key: str, volume: dict[str, str] | None, locations: tuple[CanonicalURI, ...],
                         source: str) -> None:
    if volume is None:
        raise Refusal("LICENCE_REGISTRY_MALFORMED", f"assets[{key!r}]: a volume asset needs its canonical_volume",
                      source=source, table="assets", entry=key, field="canonical_volume")
    if key != volume["volume_id"]:
        raise Refusal("LICENCE_REGISTRY_MALFORMED", f"assets[{key!r}]: a volume asset is keyed by its 14-digit volume "
                      f"id, here {volume['volume_id']!r}", source=source, table="assets", entry=key, field="id",
                      volume_id=volume["volume_id"])
    unnamed = [str(u) for u in locations if not _names_volume(u, key)]
    if unnamed:
        raise Refusal("LICENCE_REGISTRY_MALFORMED", f"assets[{key!r}]: locations {unnamed} do not name volume {key} "
                      "(last segment <id>, <id>-... or <id>.<ext>); declare a whole scroll or a derived copy as a "
                      "dataset", source=source, table="assets", entry=key, field="locations", locations=unnamed)


# --- cross-entry checks -------------------------------------------------------------


def _check_references(licences: dict[str, Licence], terms: dict[str, Terms], hosts: tuple[Host, ...],
                      assets: dict[str, Asset], source: str) -> None:
    refs: list[tuple[str, str, str, str, Mapping[str, Any]]] = []
    for lid, lic in licences.items():
        refs += [("licences", lid, "requires_terms", t, terms) for t in lic.requires_terms]
    for host in hosts:
        refs += [("hosts", str(host.prefix), "allowed_licences", lid, licences) for lid in host.allowed_licences]
        refs += [("hosts", str(host.prefix), "requires_terms", t, terms) for t in host.requires_terms]
    for aid, asset in assets.items():
        if asset.licence is not None:
            refs.append(("assets", aid, "licence", asset.licence, licences))
        refs += [("assets", aid, "additional_terms", t, terms) for t in asset.additional_terms]
    for table, entry, field, ref, namespace in refs:
        if ref not in namespace:
            raise Refusal("LICENCE_REGISTRY_UNKNOWN_REF", f"{table}[{entry!r}].{field} names {ref!r}, which is "
                          "not declared", source=source, table=table, entry=entry, field=field, ref=ref)


def _gated_weakness(lic: Licence) -> str | None:
    if not lic.requires_terms:
        return "requires no terms"
    if lic.publication == "allowed":
        return "allows publication"
    if lic.commercial_use:
        return "allows commercial use"
    if lic.shippable_in_pipeline:
        return "is shippable in the pipeline"
    return None


def _check_gated_hosts(hosts: tuple[Host, ...], licences: dict[str, Licence], source: str) -> None:
    for host in hosts:
        if not is_gated(host.prefix):
            continue
        for lid in host.allowed_licences:
            weakness = _gated_weakness(licences[lid])
            if weakness is not None:
                raise Refusal("LICENCE_REGISTRY_GATED_HOST", f"host {str(host.prefix)!r} serves Vesuvius Challenge "
                              f"data, but its allowed licence {lid!r} {weakness}", source=source,
                              host=str(host.prefix), licence=lid, weakness=weakness)


def _check_assets(assets: dict[str, Asset], source: str) -> dict[str, str]:
    for aid, asset in assets.items():
        if asset.kind == "synthetic":
            remote = [str(u) for u in asset.locations if not u.is_local]
            if remote:
                raise Refusal("LICENCE_REGISTRY_SYNTHETIC_REMOTE", f"synthetic asset {aid!r} declares non-local "
                              f"locations {remote}; synthetic data is generated by apok-eval, never fetched",
                              source=source, asset=aid, locations=remote)
    # Ids are claimed before aliases, so an alias that collides with an id is the one named.
    claims = [(aid, aid) for aid in assets] + [(aid, alias) for aid, a in assets.items() for alias in a.aliases]
    owner: dict[str, str] = {}
    for aid, name in claims:
        holder = owner.setdefault(name.casefold(), aid)
        if holder != aid:
            raise Refusal("LICENCE_REGISTRY_MALFORMED", f"name {name!r} belongs to assets {holder!r} and {aid!r} "
                          "(ids and aliases compare case-insensitively)", source=source, table="assets",
                          entry=aid, field="aliases", name=name, other=holder)
    return owner


def _check_volumes(assets: dict[str, Asset], source: str) -> None:
    by_volume: dict[str, list[Asset]] = {}
    for aid in sorted(assets):
        volume = assets[aid].canonical_volume
        if volume is not None:
            by_volume.setdefault(volume.volume_id, []).append(assets[aid])
    for volume_id, group in sorted(by_volume.items()):
        first = group[0]
        for other in group[1:]:
            a, b = first.canonical_volume, other.canonical_volume
            if canonical_sample(a.sample) != canonical_sample(b.sample):
                raise Refusal("LICENCE_REGISTRY_VOLUME_SAMPLE", f"volume {volume_id} is declared under sample "
                              f"{a.sample!r} by {first.id!r} and {b.sample!r} by {other.id!r}", source=source,
                              volume_id=volume_id, assets=[first.id, other.id], samples=[a.sample, b.sample])
            if first.obligations_key != other.obligations_key:
                raise Refusal("LICENCE_REGISTRY_VOLUME_CONFLICT", f"volume {volume_id} ({a.sample}) is declared by "
                              f"{first.id!r} and {other.id!r} with different licences or additional terms; the "
                              "registry's maintainer resolves the conflict", source=source, volume_id=volume_id,
                              assets=[first.id, other.id], licences=[first.licence, other.licence],
                              additional_terms=[sorted(first.additional_terms), sorted(other.additional_terms)])


def _check_overlaps(assets: dict[str, Asset], source: str) -> None:
    ordered = [assets[aid] for aid in sorted(assets)]
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            if a.obligations_key == b.obligations_key:
                continue
            for la in a.locations:
                for lb in b.locations:
                    if la.is_under(lb) or lb.is_under(la):
                        raise Refusal("LICENCE_REGISTRY_LOCATION_OVERLAP", f"assets {a.id!r} ({a.licence}) and "
                                      f"{b.id!r} ({b.licence}) claim overlapping locations {str(la)!r} and "
                                      f"{str(lb)!r} with different licences or additional terms", source=source,
                                      assets=[a.id, b.id], locations=[str(la), str(lb)])
