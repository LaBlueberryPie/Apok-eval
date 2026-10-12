# `apok_eval.licence` — asset registry and licence gate

The gate answers one question before anything is read: may this run read these inputs, under which
licence, and has the person running it accepted the terms those inputs require? It is built and tested
against synthetic fixtures. External facts are cited to ScrollPrize/villa at `fb8c2c4` (paths relative to the
villa root; the shipped registry cites `e0bbb8b`, where every cited file is byte-identical).

`apok-eval e7v3` runs the gate when it is given `--registry` (and `--terms` for inputs whose licence needs an
acceptance). Without a registry nothing is checked, and the run's record says so (`licence.checked: false`).
Remote reads (`apok_eval.io.store.HttpStore`) always need a clearance from this gate.

## What it does

`apok-eval licence-check` reads a **run configuration** and an **asset registry** and either issues a
`LicenceClearance` (exit 0) or refuses with one stable code that names its cause (exit 3). It refuses:

- any input that is not a declared asset with a licence;
- any input URI that is not under one of its asset's declared locations, or not under a declared host or
  local root, or on a host that does not allow the asset's licence;
- any input whose licence or additional terms require terms that `terms_accepted_by` — one individual's own
  acceptance, in their own words — does not cover;
- in `--for-release` mode, any input whose licence does not allow publication or shipping.

Remote readers (`apok_eval.io`) require a clearance issued by this gate, and `covers(uri)` is the check:
`HttpStore` accepts a clearance only if `type(c) is apok_eval.licence.LicenceClearance` (the exact type, which
only `check_run` constructs) and `covers(uri)` returns `True`. `isinstance(c, apok_eval.clearance.Clearance)`
would not do: the protocol is runtime-checkable, so it accepts any object with a `covers` method. Nor would
`isinstance(c, LicenceClearance)`: it consults `__class__`, which a `Mock(spec=LicenceClearance)` sets, and it
accepts a subclass overriding `covers`.

## Declaring your data

The shipped registry (`src/apok_eval/licence/registry.json`) declares licences, terms and the Vesuvius
Challenge hosts, and one synthetic asset. **It declares no scroll data.** Before the gate clears a run on
your data you declare that data yourself, in a copy of the shipped registry:

1. **Copy the registry**: `cp $(python -c "import apok_eval.licence.registry as r; print(r.SHIPPED_REGISTRY)")
   my-registry.json`.
2. **Find the data's licence at its source**, not from this tool: the data catalogue keeps it per volume
   (`volumes[vid].properties.license`, `scrollprize.org/src/components/atlas/buildIndex.js:83-99`), and
   `vesuvius/src/vesuvius/install/accept_terms.py:55` says which datasets are CC BY-NC 4.0 and which are under
   the EduceLab-Scrolls terms. Some samples carry additional terms in the catalogue (for example a notice
   reserving scholarly publication until a date); declare those as a `terms` entry and list it in the asset's
   `additional_terms`.
3. **Declare a local root** for local copies: a `hosts` entry for the directory that holds them, allowing the
   licence of the data's source. A local copy of CC BY-NC bucket data is still CC BY-NC.
4. **Declare the asset**: an `assets` entry whose `locations` are the paths (or bucket URIs) you will read.
   Declare only what you read; everything else stays refused.
5. **Write your acceptance** in a file of your own, outside any repository, as one JSON object holding only
   `terms_accepted_by`, and pass it with `--terms`. Nobody else, and no tool, writes it for you.

For example, a local copy of fibres, an umbilicus and fit segments made from CC BY-NC 4.0 data, under
`/data/scroll`:

```json
"hosts": {
  "file:///data": {"allowed_licences": ["CC-BY-NC-4.0"], "requires_terms": [],
                   "note": "my local disk; copies keep their source's licence"}
},
"assets": {
  "my-scroll-copy": {
    "kind": "dataset", "locations": ["file:///data/scroll"], "licence": "CC-BY-NC-4.0",
    "additional_terms": [], "aliases": [],
    "evidence": "where the licence was read: the catalogue entry of the source volume",
    "note": "fibres, umbilicus and a fit, copied from the open-data bucket"
  }
}
```

(added to the copied registry's `hosts` and `assets`), and `my-terms.json`:

```json
{"terms_accepted_by": {
  "person": "<your name>",
  "date": "YYYY-MM-DD",
  "accepted": [{"terms": "VESUVIUS-DATA-TERMS",
                "text_sha256": "sha256:9f7ce6e4621d7add76c11f3428ee61fda94c6ecce833a324441dbe1711b940b2"}]
}}
```

Then `apok-eval e7v3 ... --registry my-registry.json --terms my-terms.json`. The run finds, for each input
path, the declared asset with the longest location above it (`EVAL_RUN_UNDECLARED` when there is none), builds
the run configuration from those and your terms file, and refuses before reading anything unless
`check_run` clears it. The clearance (with your acceptance) is kept in the run's sealed record, so do not
publish a record you would not publish your name and acceptance in.

`tests/test_licence_gate.py` checks this walkthrough (`test_a_local_copy_is_declared_under_a_file_root_with_the_licence_of_its_source`),
and that no registry can label bucket or data-server data as MIT.

## Run configuration

```json
{
  "inputs": [{"asset": "my-scroll-copy", "uri": "/data/scroll/fibres"}],
  "terms_accepted_by": {
    "person": "<one individual's name>",
    "date": "YYYY-MM-DD",
    "accepted": [
      {"terms": "VESUVIUS-DATA-TERMS",
       "text_sha256": "sha256:9f7ce6e4621d7add76c11f3428ee61fda94c6ecce833a324441dbe1711b940b2"}
    ]
  }
}
```

- `inputs`: non-empty list; each entry has exactly the string fields `asset` (a registry id — not an alias)
  and `uri`. An extra field (e.g. a `licence` of its own) is refused, so a misleading label cannot be
  silently ignored.
- `terms_accepted_by`: optional; `null` means absent. Exactly `person`, `date`, `accepted`:
  - `person` — one individual. The text is checked after NFKC normalisation, which folds fullwidth forms
    (`＆ ， ／ ；`) into ASCII; it is recorded as written, without surrounding whitespace. Refused: not a
    string (a list of persons), empty or blank, a control character or line break, a list separator
    (`, ; & / + |`, the ideographic comma `、`, the Arabic comma and semicolon `، ؛`, a bullet `•`, a fraction
    or division slash), the word `and`, `und`, `et` or `et al` (any case), a group or organisation word
    (`team`, `everyone`, `everybody`, `all`, `members`, `staff`, `group`, `on behalf of`, `Ltd`, `Limited`,
    `Inc`, `LLC`, `GmbH`, `plc`, `Corp`, `company`; whole words, any case), or no letter at all. "Smith, John"
    must be written "John Smith". **What the heuristic misses:** "and" in other languages (`y`, `e`, `en`,
    `og`, ...), other separators (the middle dot `·` is left alone because Catalan names use it; the katakana
    middle dot `・` because it separates given and family names), names run together without a separator
    ("Jane Doe John Roe"), and any organisation not named by one of the words above. It is a guard against a
    careless entry, not proof of one individual.
  - `date` — a real calendar day written `YYYY-MM-DD`.
  - `accepted` — non-empty; each entry exactly `{terms, text_sha256}`; `terms` must be a terms id the
    registry declares, at most once; `text_sha256` is `null` or a `sha256:<64 hex>` digest, and must equal the
    registry's digest when the registry pins one (`LICENCE_TERMS_DIGEST_MISMATCH`), as it does for both
    shipped terms.
- Other top-level keys are ignored: other modules extend the run configuration.

The file is read as **strict JSON** (`apok_eval.jsonstrict`, [docs/jsonstrict.md](jsonstrict.md)): a duplicated key at any
depth, `NaN`/`Infinity`, a number that overflows a double, an integer past `int()`'s digit limit, nesting
deeper than 256 levels, a `\ud800`-style escape that decodes to a lone surrogate (not Unicode text: it could
not be encoded or digested), invalid UTF-8 or a BOM is refused (`json.loads` would silently keep the last
duplicate, so `"terms_accepted_by": {...}, "terms_accepted_by": null` could otherwise hide an acceptance).
The registry is read the same way. The refusal's `problem` is the parser's problem when the bytes are not
strict JSON (e.g. `duplicate key 'terms_accepted_by'`), and `<exception type>: <message>` when the file could
not be read (e.g. `NotRegularFile: ...`); nesting deep enough to exhaust Python's parser is such a problem
too, never an escaping `RecursionError`.

### Why an acceptance must be recorded here

`vesuvius.accept_terms --yes` prints the terms and writes the three bytes `yes` to a package-local
`agreement.txt` (`vesuvius/src/vesuvius/install/accept_terms.py:41-48`); nothing reads it and it records no
person, date or text. **apok-eval never treats it as evidence.** The terms are per person: "if I am working
in a team, every team member will sign this form separately" (`accept_terms.py:59`); the data-server licence
says the same. So the acceptance names exactly one individual, who accepts personally.

## Asset registry (`apok-asset-registry-v1`)

One strict-JSON object with exactly `schema`, `licences`, `terms`, `hosts`, `assets` and an optional
`note`. Every entry lists its fields exactly: a missing or unknown field is refused, never defaulted (the
only optional fields are an asset's `note` and `canonical_volume`). Ids (licences, terms, assets) match
`[A-Za-z0-9][A-Za-z0-9._+-]*` in full. Text fields are non-blank strings of Unicode text: a lone surrogate
is refused, in a file and in an in-memory document alike.

| table | key | fields |
|---|---|---|
| `licences` | licence id | `name` text, `url` text, `requires_terms` [distinct terms ids], `publication` `"allowed"`/`"forbidden"`/`"ask"`, `commercial_use` bool, `shippable_in_pipeline` bool, `obligations` [text], `evidence` text |
| `terms` | terms id | `title` text, `url` text, `per_person` bool, `text_sha256` digest or `null` (not yet pinned), `evidence` text |
| `hosts` | a host or prefix, any spelling (canonicalised) | `allowed_licences` [distinct licence ids], `requires_terms` [distinct terms ids], `note` text |
| `assets` | asset id | `kind` (`volume`, `dataset`, `segment`, `model-weights`, `code`, `synthetic`), `locations` [URIs, ≥1], `licence` id or `null`, `additional_terms` [distinct terms ids], `aliases` [distinct names], `canonical_volume` `{sample, volume_id}` (exactly these keys; `sample` non-blank text, `volume_id` 14 digits; required for `volume`, optional for `dataset` and `segment`, refused on other kinds), `evidence` text, optional `note` text |

Semantics:

- **Licence scope is per volume**, not per scroll: the catalog keeps it at `volumes[vid].properties.license`
  (`scrollprize.org/src/components/atlas/buildIndex.js:83-99`), and one scroll can have scans under different
  licences (`scrollprize.org/src/data/atlasOverlay.json:115-118`).
- **A `volume` asset is keyed by its 14-digit volume id**, not the scan id (they differ for ESRF scans,
  `scrollprize.org/static/data_browser/index.json:180-189`). Enforced: the asset id equals its
  `canonical_volume.volume_id`, and every location's last segment names that volume — `<id>-...` (bucket
  folders, `index.json:186-188`), `<id>.<ext>` (legacy `volumes_zarr/<id>.zarr`,
  `vesuvius/src/vesuvius/install/configs/scrolls.yaml:32,38`) or `<id>` (a legacy TIFF stack,
  `scrollprize.org/docs/06_tutorial3.md:256`). So a volume entry cannot cover a whole scroll, a bucket root or
  another volume.
- A **copy or derivative of a volume** whose folder does not carry the volume id (a legacy
  `volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr`, `scrolls.yaml:11`, or a segment traced on it) is a
  `dataset` or `segment` that declares the volume as its `canonical_volume`. Every asset declaring one volume
  id must give it one sample and carry one licence and one set of additional terms
  (`LICENCE_REGISTRY_VOLUME_SAMPLE`, `LICENCE_REGISTRY_VOLUME_CONFLICT`). **Limit:** the gate sees only the
  copies that declare their volume; an entry that omits `canonical_volume` is compared with others only
  through overlapping locations (`LICENCE_REGISTRY_LOCATION_OVERLAP`, same authority only).
- `licence: null` means *declared but not licensed*: refused when used, with the asset's note.
- `additional_terms` are terms ids that bind the asset beyond its licence (a publication notice that modifies a
  licence is one). They count as required terms and block release.
- A URI belongs to the **longest** declared host prefix it is under, so a narrow entry (one checked volume)
  overrides its bucket's default.
- A host's `requires_terms` are terms every input on or below that host needs, whatever its licence: a
  server's own access terms. The shipped data server requires `DATA-SERVER-LICENSE` (its LICENSE.txt binds
  every user individually), so even if a narrower entry allows another licence there, its inputs still need
  the data-server terms. They accumulate over **every** declared prefix the URI is under: a narrower entry
  chooses the allowed licences but cannot drop the terms of an entry that encloses it. An input's required
  terms are its licence's, those host terms, and its asset's additional terms.
- Hosts are needed for **every** input URI, local ones included: a local root (`file:///workspace/data`)
  must be declared with the licences allowed under it. This is stricter than "remote URIs on undeclared
  hosts are refused": a local copy of scroll data cannot be labelled MIT unless someone declares a root
  that allows MIT there.
- `canonical_volume.sample` is compared after alias normalisation, by the scroll-name key and alias table
  every module uses (`apok_eval.scrolls`): case and every character but a letter or digit are ignored, and
  the other names map to catalog ids — Scroll1/s1→PHercParis4, Scroll2→PHercParis3, Scroll3/PHerc332→PHerc0332,
  Scroll4/s4→PHerc1667, Scroll5/PHerc172→PHerc0172 (`vesuvius/src/vesuvius/install/configs/scrolls.yaml:11,20,26,32,38`;
  `scrollprize.org/static/data_browser/index.json:61-62,2230,3065-3068,7648-7651,8707,8746`; s1 and s4 from the
  fit-folder and volpkg names cited in `apok_eval/scrolls.py`). A name the table does not list is compared by
  its own key.

The registry's digest is `digest_json` of the document as read; every clearance records it.

### Hard-wired gating of the Vesuvius Challenge hosts

In code, not in the registry (`is_gated`): a host entry whose authority is one of `GATED_BUCKETS`
(`vesuvius-challenge-open-data`, `vesuvius-challenge`) or is, or is under, one of `GATED_DOMAINS`
(`ash2txt.org`: the data server `dl.`, the gateway `data.aws.`, the zarr proxy `zarr.aws.`, the thumbnail
services `thumbs(-vc).aws.` — `scrollprize.org/src/data/atlasDataAccess.json:12-44` — WebKnossos `wk.aws.`,
`atlasOverlay.json:215`, `volumes.aws.`, `lasagna/docs/tifxyz_training_integration.md:50`, and any other) may
allow only licences that **require terms, forbid commercial use, are not shippable in the pipeline and are
not freely publishable**. Domains match whole labels (`notash2txt.org` is not gated). Every spelling of a
bucket object that the gate accepts canonicalises to `s3://<bucket>/...`, and the spellings it cannot map are
refused (see URI canonicalisation), so no registry — not even one passed with `--registry` — can label data
on these buckets or hosts as MIT or synthetic. Loosening this is a code change.

What this does **not** cover: a mirror or CDN the gate does not know about (any other https host is judged
only by the registry entries that name it), and local copies (`file://`, a loopback server), which are
trusted as their declared root says.

### The shipped registry (`src/apok_eval/licence/registry.json`)

- Licences: `CC-BY-NC-4.0` (requires `VESUVIUS-DATA-TERMS`, publication `ask`), `EDUCELAB-SCROLLS`
  (requires `DATA-SERVER-LICENSE`, publication `forbidden`), `MIT`.
- Terms, both pinned; an acceptance must cite exactly these digests:

  | terms | text | `text_sha256` |
  |---|---|---|
  | `VESUVIUS-DATA-TERMS` | the bytes of the whole file `vesuvius/src/vesuvius/install/accept_terms.py` at villa `e0bbb8b` | `sha256:9f7ce6e4621d7add76c11f3428ee61fda94c6ecce833a324441dbe1711b940b2` |
  | `DATA-SERVER-LICENSE` | the bytes of https://dl.ash2txt.org/LICENSE.txt as served with Last-Modified 2025-05-12 | `sha256:9dc08a92194617d4ae68fb96d4fd0da99d1c7207f821f2a3750b0034eac8d154` |

  Recompute a pin before you accept: if the text has changed, the pin no longer matches, an acceptance of
  the new text is refused, and the registry's pin must be updated by whoever maintains your registry.
- Hosts: the open-data bucket (CC BY-NC or EduceLab); `s3://vesuvius-challenge` = the
  `data.aws.ash2txt.org/samples/` gateway (EduceLab only, the strictest, because it serves volumes under both
  licences: declare a narrower entry for a volume you have checked); `https://dl.ash2txt.org` (EduceLab only;
  requires `DATA-SERVER-LICENSE` as host terms); `synthetic://apok-fixtures` (MIT). The bucket hosts require
  no host terms: their licences carry them.
- Assets: `apok-synthetic-fixtures` only — `synthetic://apok-fixtures`, MIT, clears without terms.

**Policy choice:** `CC-BY-NC-4.0` scroll data also requires a recorded acceptance (`VESUVIUS-DATA-TERMS`),
because the bucket terms are accepted through `vesuvius.accept_terms`, which leaves no evidence a tool can
read. EduceLab publication is `forbidden` and CC BY-NC publication `ask`: the gate cannot decide whether
something derived from scroll data may be published, so release mode refuses both.

### Synthetic data

Two forms: `synthetic://<name>/...` names data apok-eval generates (the shipped `apok-synthetic-fixtures`), and
generated data written to disk (the demo's, the tests') is declared by a run-local registry as `file://` paths
under a declared local root (or `http://127.0.0.1:<port>/...` for a reader's loopback test server). A
`synthetic` asset may only have local locations (`file`, `synthetic`, loopback `http`). `apok-eval demo` writes
such a registry for its directory and runs every scenario through the gate.

## URI canonicalisation (`apok_eval.licence.uri`)

Every location — registry hosts and asset locations, inputs, and the URI a reader asks `covers()` about — is
canonicalised first. Forms (no trailing slash):

| canonical | spellings |
|---|---|
| `s3://<bucket>/<key>` | `s3://b/k`; `https://b.s3.amazonaws.com/k`; `https://b.s3.<region>.amazonaws.com/k`; `https://s3[.<region>].amazonaws.com/b/k`; `https://data.aws.ash2txt.org/samples/k` ≡ `s3://vesuvius-challenge/k` (`scrollprize.org/src/data/atlasDataAccess.json:8-15`, `volume-cartographer/apps/VC3D/OpenDataManifest.cpp:774-789`) |
| `https://<host>/<path>` | any other https URL, host lower-cased (e.g. `dl.ash2txt.org`), except the hosts refused below |
| `http://<loopback ip>[:port]/<path>` | http only for 127.0.0.0/8 or `[::1]`; `:80` dropped |
| `file:///<path>` | an absolute path or `file:///` URI, after `os.path.realpath`; the filesystem root is `file:///` |
| `synthetic://<name>/<path>` | name lower-cased |

**Other names for known objects are refused, not kept as plain https hosts** (`LICENCE_URI_AMBIGUOUS`,
`host` names the host), because a plain host is judged only by the registry entries that name it, and a
second canonical form would let one object get two answers:

- any other host that is or ends in `amazonaws.com` — dualstack, FIPS, `s3-accelerate`, `s3-external-1`,
  the legacy `s3-<region>` and website endpoints, `s3.amazonaws.com` without a bucket. Write `s3://`;
- the zarr proxy `zarr.aws.ash2txt.org`, which re-serves both buckets under `/data` and `/open-data`
  (`atlasDataAccess.json:33-44`; villa no longer routes through it, `scrollprize.org/src/components/atlas/dataAccess.js:12-16`).
  Write the bucket URI;
- an IP address (the last label is all digits), which may be any host, S3 included.

`.` segments are dropped, `..` removes the previous segment (on the URL path, before the gateway mapping, as
an HTTP client would). Local paths are resolved with `realpath`: a symlink inside a declared root that points
outside it is judged by its target, so it is refused as outside the asset.

Prefix tests compare **whole segments**: `.../PHerc0332` covers `.../PHerc0332/x`, never `.../PHerc03320`.

## Refusal codes

Registry (`apok_eval.licence.registry`):

| code | fires when (cause in `details`) |
|---|---|
| `LICENCE_REGISTRY_UNREADABLE` | the file is missing or no regular file (a FIFO or a device is refused without being read), cannot be opened (an `OSError`, or a `ValueError` for a path holding a NUL or an unencodable surrogate) or is not strict JSON (`apok_eval.jsonstrict`: a duplicate key, a lone surrogate, nesting past 256 levels included) (`source`, `problem`) |
| `LICENCE_REGISTRY_SCHEMA` | `schema` is not `apok-asset-registry-v1` (`schema`) |
| `LICENCE_REGISTRY_MALFORMED` | not an object; wrong top-level keys or table types; bad id (`field: "id"`); entry not an object; missing/unknown field (`missing`, `unknown`); ill-typed field, a lone surrogate in text included (`field`, `value`); `canonical_volume` missing on a volume or present on a kind that cannot name one (`field: "canonical_volume"`); a volume asset not keyed by its volume id (`field: "id"`, `volume_id`); a volume location that does not name its volume (`field: "locations"`, `locations`); one location twice in an asset; two host keys for one prefix (`prefix`, `other`); one name (id or alias, case-insensitive) for two assets (`name`, `other`) — always with `table` (`"$"` or `"note"` for the top level), and `entry` when an entry is at fault |
| `LICENCE_REGISTRY_UNKNOWN_REF` | a referenced licence or terms id is not declared, host `requires_terms` included (`table`, `entry`, `field`, `ref`) |
| `LICENCE_REGISTRY_GATED_HOST` | a gated host (a gated bucket, or a host under a gated domain) allows a licence that requires no terms, allows publication, allows commercial use or is shippable (`host`, `licence`, `weakness`) |
| `LICENCE_REGISTRY_SYNTHETIC_REMOTE` | a `synthetic` asset has a non-local location (`asset`, `locations`) |
| `LICENCE_REGISTRY_VOLUME_SAMPLE` | two assets (the volume, or a dataset or segment declaring it) give one volume id different samples after alias normalisation (`volume_id`, `assets`, `samples`) |
| `LICENCE_REGISTRY_VOLUME_CONFLICT` | two assets declaring one volume id differ in licence (null included) or additional terms (`volume_id`, `assets`, `licences`, `additional_terms`) |
| `LICENCE_REGISTRY_LOCATION_OVERLAP` | two assets with different licence or additional terms claim locations where one is a whole-segment prefix of the other (`assets`, `locations`) |

URIs (`apok_eval.licence.uri`; `what` names the field, `uri` the value):

| code | fires when |
|---|---|
| `LICENCE_URI_MALFORMED` | not a non-empty string; whitespace, control, invisible or backslash characters (`char`); invalid bucket (3-63 characters; `bucket`) or host (`host`); bad port (`port`); `file:` not `file:///<absolute path>`; a local path whose real location (after symlinks) contains a refused character or `%`, `?`, `#` (`real`, `char`) |
| `LICENCE_URI_AMBIGUOUS` | `%` (percent-encoding: `s3://` keys are literal, https paths are decoded), `?` or `#` (`char`); userinfo; explicit port on a non-http scheme; empty path segment (`a//b`; S3 keys may contain `//`); `..` above the root; an unrecognised `amazonaws.com` endpoint, the zarr proxy or an IP address (`host`) |
| `LICENCE_URI_RELATIVE` | no scheme and not absolute; scheme-relative `//host/...`; `scheme:` without `//`; no host or bucket |
| `LICENCE_URI_SCHEME` | a scheme other than s3, https, http, file, synthetic (`scheme`; e.g. `zarr2://`, `s3+region://`); http to anything but a loopback literal, `localhost` included (`host`) |

Gate (`apok_eval.licence.gate`), checked in this order:

| code | fires when (cause in `details`) |
|---|---|
| `LICENCE_RUN_UNREADABLE` | the run configuration file is missing or no regular file (a FIFO or a device is refused without being read), cannot be opened (an `OSError`, or a `ValueError` for a path holding a NUL or an unencodable surrogate) or is not strict JSON (`apok_eval.jsonstrict`: a duplicate key, a lone surrogate, nesting past 256 levels — under a key the gate ignores too — included) (`path`, `problem`) |
| `LICENCE_RUN_MALFORMED` | not an object; `inputs` missing, empty or not a list; an input not exactly string `asset` and `uri` (`field`) |
| `LICENCE_ACCEPTANCE_MALFORMED` | `terms_accepted_by` malformed as described above (`field`, `value`) |
| `LICENCE_TERMS_DIGEST_MISMATCH` | an accepted entry cites a digest (or null) other than the one the registry pins (`terms`, `cited`, `pinned`) |
| `LICENCE_ASSET_UNKNOWN` | the input's asset id is not in the registry; an alias or mis-cased id is named as such (`input`, `asset`, `name_of`) |
| `LICENCE_ASSET_UNLICENSED` | the asset has `licence: null` (`input`, `asset`, `note`) |
| `LICENCE_URI_*` | the input URI cannot be canonicalised (`what: "inputs[i].uri"`) |
| `LICENCE_URI_OUTSIDE_ASSET` | the URI is not under any of the asset's locations (`input`, `asset`, `uri`, `locations`) |
| `LICENCE_HOST_UNDECLARED` | the URI is under no declared host or local root (`input`, `asset`, `uri`) |
| `LICENCE_HOST_DISALLOWS` | that host does not allow the asset's licence (`input`, `asset`, `licence`, `host`, `allowed`) |
| `LICENCE_TERMS_ABSENT` | inputs require terms (licence, host or additional terms) and there is no acceptance (`required`, `required_by`) |
| `LICENCE_TERMS_UNCOVERED` | the acceptance misses required terms (`missing`, `required_by`, `person`) |
| `LICENCE_RELEASE_PUBLICATION` | `--for-release`: the licence's publication is not `allowed` (`input`, `asset`, `licence`, `publication`) |
| `LICENCE_RELEASE_UNSHIPPABLE` | `--for-release`: the licence is not shippable in the pipeline (`input`, `asset`, `licence`) |
| `LICENCE_RELEASE_ADDITIONAL_TERMS` | `--for-release`: the asset carries additional terms, which the gate cannot evaluate for release (`input`, `asset`, `additional_terms`) |
| `LICENCE_CLEARANCE_UNISSUED` | a `LicenceClearance` is constructed without the gate's private token (`token`) |

Registry errors come first (the registry is validated on every call, and `licence_check` / the CLI validate
it before reading the run configuration file), then the run configuration, the acceptance, each input in
order, terms, and release. The first failure refuses. The acceptance's `accepted` entries are checked one at
a time, so a digest mismatch in an earlier entry is reported before a malformed later one. A digest mismatch
is refused even for terms the run does not require: the record would be false.

## The clearance

`check_run(run, registry=None, *, for_release=False)` (registry: a path, a parsed document, or `None` for the
shipped one; anything else, a `Registry` object included, is a `TypeError`, because the gate validates the
registry itself on every call) returns a `LicenceClearance`:

- `inputs` — per input: asset, kind, given and canonical URI, host, licence id/name/url, obligations,
  additional terms, required terms, publication, commercial use, shippable;
- `acceptance` — person, date, and per accepted terms: id, title, cited digest, `pinned` (the registry pins
  the text and the citation matches), `per_person` (binds only this person);
- `required_terms`, `for_release`, `registry_digest`; `as_dict()` gives a canonical-JSON record
  (`apok-licence-clearance-v1`). The record never uses the key `terms_accepted_by`.
- `covers(uri)` canonicalises `uri` now and is true only if it is under (whole segments) a cleared input's
  canonical URI — not the asset's whole location, and never a parent. An uncanonicalisable URI is not
  covered (`False`, no exception).

**Issued only by the gate.** The constructor takes a module-private token as its first argument and refuses
without it (`LICENCE_CLEARANCE_UNISSUED`). **Immutable:** attributes cannot be set or deleted, collections
are tuples, and copy/pickle are refused (they rebuild by setting attributes); record it with `as_dict()`.
Python cannot stop deliberate forgery (`object.__new__` plus `object.__setattr__`, or importing the token):
the token stops accidental or careless construction, not an adversary inside the process.

## CLI

```
apok-eval licence-check RUN_CONFIG [--registry PATH] [--for-release] [--json]
```

When every input clears: prints, per input, the asset, canonical URI, host, licence,
publication/commercial/shippable flags, additional terms, required terms, obligations and the decision, then
the acceptance used; `--json` prints the clearance record. Exit 0. Default registry: the shipped one.

On a refusal, stdout carries only the refusal as JSON (exit 3). Before it, **stderr** carries each input
judged on its own (`input_decisions`): asset, given URI, licence, additional terms, obligations, and `cleared`
or `refused CODE: reason` — as text, or with `--json` as a JSON list — so the inputs that would clear are
reported too. An input "would clear" if it cleared as the run's only input (its own asset, URI, host,
required terms and, in release mode, release checks). Nothing per input is printed when the refusal concerns
the whole run (registry, run shape, acceptance). The report is a second, independent reading of both files: it
informs; the refusal decides. `input_decisions` never clears anything.

## No acceptance is committed — the hygiene guard

`tests/test_licence_hygiene.py` scans every git-tracked file (`git ls-files`) and fails if a file sets
`terms_accepted_by`. JSON, TOML and Python are **parsed, never pre-filtered on their raw text**, so a key
written with escapes (`"terms_accepted_by"`) or, in Python, by implicit concatenation
(`"terms_" "accepted_by"`) is seen as the parser — and the gate — sees it:

- JSON (`.json`, `.ipynb`, and every other tracked file whose whole text parses as JSON, because the gate
  reads a run configuration of any name as strict JSON) and TOML: a non-null value anywhere in the document.
  The whole-text JSON check runs on **every** tracked file on top of its suffix's own check, so a `.py`
  outside `src/`, a `.toml` or `.yaml` whose text is really JSON, or a `src/` `.py` nested too deeply to parse
  as Python is still read as the gate would read it;
- YAML, matched by text (no YAML parser is installed): a plain, single-quoted or double-quoted key (escapes
  decoded) whose value is anything but a plain `null`/`Null`/`NULL`/`~` — a quoted `'null'` is the string
  "null" and counts as set — and any explicit `? terms_accepted_by` key;
- Python under `src/`: a write (dict literal key, subscript store/delete, keyword argument, `setdefault`,
  `__setitem__`, `setattr`, attribute store); reading it is fine. This is why `apok-eval e7v3` takes your
  acceptance as a whole terms file and passes it on as written: no code in `src/` composes one.

It also scans `src/` on disk, so it runs in a scratch copy of the tree without `.git`, where the tracked-file
scan is skipped. Its scanner has its own tests, each escape form included.

Honest limits: this is a hygiene guard against an agent or a commit, **not identity proof**. Anyone can type a
name into a run configuration; untracked files (the terms files that carry an acceptance) are deliberately
not scanned; a key built at run time (`"terms_" + "accepted_by"`, a variable, JSON inside a Python string)
escapes an AST scan; YAML anchors, aliases and multi-line keys are not interpreted; Python outside `src/`
(test code builds synthetic acceptances in memory at test time), and `.cfg`, `.ini`, `.env` or prose, are
scanned only as whole-text JSON.

## What is enforced, what is policy, what is yours

Enforced in code: everything in the refusal tables; the gated hosts; per-volume licence consistency;
the strict JSON reading; the clearance's issuance and immutability.

Policy encoded in the shipped registry (change it in your own copy, and say so): CC BY-NC requires a
recorded acceptance; EduceLab publication `forbidden` and CC BY-NC `ask`; `commercial_use: false` for scroll
data; the `s3://vesuvius-challenge` gateway defaults to EduceLab.

Not checked: who the named person is; that the acceptance date is after the terms were published or not in
the future (no wall clock is read); the contents of terms texts (the gate compares the cited digest with the
pinned one, never the text itself); `per_person` beyond recording it; `commercial_use` in release mode (it is
recorded; `shippable_in_pipeline` is the release flag); where local bytes came from (a local root is trusted
as declared); the URI's object existing; time-of-check/time-of-use for local symlinks.

Yours, never the tool's: accepting terms; the licence of each volume you declare (read it at the source);
whether anything derived from scroll data may be published, which the data's terms decide and release mode
only refuses.

## Open questions

- Should the bucket hosts also require `VESUVIUS-DATA-TERMS` as host terms (the `accept_terms` text covers all
  Vesuvius Challenge datasets, `accept_terms.py:52-70`)? Today the licence carries it for CC BY-NC data only.
- The text of https://dl.ash2txt.org/LICENSE.txt is not in the repository, so the `EDUCELAB-SCROLLS`
  obligations are transcribed from `accept_terms.py:55-69` and the served file, not shipped with it.
- Does `s3://vesuvius-challenge` (the gateway) carry separate access terms? Source gives only the URL mapping.
- Should release mode also refuse `commercial_use: false`? Today it checks publication and shippability only.
- `http://localhost` is refused (it resolves by name); readers must use a loopback literal.
