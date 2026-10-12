# `apok_eval.io` — stores, the read log and OME-Zarr v2

The CT positive control of `apok-eval e7v3` reads its chunks through this module's guards (§3.5); `apok-eval
inspect` opens a volume with `open_volume`. Every test uses synthetic fixtures generated in a temporary
directory, and remote reads go only to a loopback HTTP server.

Modules: `src/apok_eval/io/store.py`, `src/apok_eval/io/readlog.py`, `src/apok_eval/io/omezarr.py`. The
coordinate frame (`VolumeFrame`, `check_points_within`) is the shared foundation in `src/apok_eval/io/frame.py`.

Citations of the form `path:line` are to ScrollPrize/villa at commit `fb8c2c4`. villa's root is MIT; its
`volume-cartographer/` (VC3D) sources are GPL-3.0. Both were read only to learn the formats, and no code from
them is copied ([THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)).

## 1. Stores (`apok_eval.io.store`)

A store maps a key to the bytes stored under it, or to `None` when nothing is there:
`get(key) -> bytes | None`. It also has `describe() -> dict`. Both `LocalStore` and `HttpStore` satisfy the
runtime-checkable protocol `Store`.

**Keys.** A key is relative and `/`-separated. Every segment matches `[A-Za-z0-9._-]+` and is neither `.` nor
`..`. Empty segments, leading or trailing `/`, `#`, `%`, `?`, `:`, spaces, backslashes and non-ASCII characters are
all refused (`IO_STORE_KEY_INVALID`) before any filesystem or network access. So a key cannot climb out of its
store, never needs URL encoding, and is always a valid `apok_eval.merkle` manifest key (those may not contain `#`).
Zarr keys such as `.zgroup`, `0/.zarray`, `0/1.2.3` and `0/1/2/3` all pass.

**`LocalStore(root)`** reads files under `root`:

- The root must be a non-empty path to an existing directory (`IO_STORE_ROOT_INVALID`). `""` is refused: it
  would silently mean the working directory (an explicit `"."` is accepted). The root is resolved once, so a
  symlinked root is followed at construction and recorded as resolved.
- Each key is resolved through symlinks with `os.path.realpath`, which handles loops the same way on Python
  3.11–3.13. If the result is outside the root, the read is refused (`IO_STORE_ESCAPE`, naming the key and
  the resolved path). This applies whether or not the target exists, so nothing outside the root is reported
  as absent either. Symlinks that stay inside the root are followed.
- A directory, a dangling or looping link, or any other non-regular file is refused (`IO_STORE_NOT_A_FILE`)
  rather than reported absent.
- A key **below** a dangling or looping link is refused too (`IO_STORE_NOT_A_FILE`, naming the key and the
  broken `component`). Otherwise a nested (`/`) chunk-row directory replaced by a broken link would read as a
  slab of absent chunks, that is, as fill.
- A key is absent (`None`) only when it has no directory entry and its deepest existing ancestor is a
  directory, a regular file, or a working link to one. A key below a regular file is absent, as an object
  store answers 404 for `a/b` when `a` is an object, so a local copy and a remote store give the same read log.
  "No directory entry" means `lstat` failing with `ENOENT` or `ENOTDIR` (or `ELOOP`, a looping link above the
  key, which is then refused as a broken link). Any other error, from `lstat` on the way (an unsearchable
  chunk-row or level directory, `ENAMETOOLONG`) or from reading the file (`EACCES`, `EIO`), is refused
  (`IO_STORE_UNREADABLE`, naming the key, the path and the errno). It is never read as an absent chunk, which
  would fill it and record it absent in the read log, nor as a missing `.zarray`.
- The tree is never written to. Reads are not hardened against the tree changing while they run, but the read
  log refuses an object that comes back different.

**`HttpStore(base_uri, clearance, *, retries=0, retry_wait_s=1.0, timeout_s=60.0)`** reads objects anonymously
using `urllib` only:

| base URI | fetched from |
|---|---|
| `s3://<bucket>[/<prefix>]` | `https://<bucket>.s3.us-east-1.amazonaws.com[/<prefix>]` (the one canonical rule, `S3_HTTPS_RULE`) |
| `https://<host>[:port]/<path>` | as given |
| `http://<loopback IP literal>[:port]/<path>` | as given (local mirrors and tests only): `127.0.0.0/8` or `[::1]`; not `localhost`, a name the licence gate never clears |

- The S3 rule is virtual-hosted, with the region fixed at us-east-1. VC3D does the same
  (`volume-cartographer/core/src/RemoteUrl.cpp:80-87`). Another VC helper drops the region
  (`volume-cartographer/utils/src/http_fetch.cpp:66-68`); apok-eval does not. apok-eval does not special-case
  `s3://vesuvius-challenge/`, which VC3D maps to `https://data.aws.ash2txt.org/samples/` (see Open questions).
- A bucket must be a DNS label of 3–63 characters from `[a-z0-9-]`, with no dots. A dotted bucket fails TLS host
  verification under virtual-hosted addressing.
- **Normal form.** A base URI must be printable ASCII without spaces, and `urlunsplit(urlsplit(uri))` must give
  it back unchanged. `urlsplit` lowercases the scheme and silently drops leading whitespace and control
  characters, tab/CR/LF anywhere, and an empty `?` or `#`. Accepting such a URI would map and fetch a
  different URI (another host or bucket, even) from the one the Clearance was asked about, and the
  transform would go unrecorded. A non-ASCII host would be IDNA-encoded by the resolver, which is also
  unrecorded. All of these are refused (`IO_READ_URI_INVALID`, naming the `uri` and what splitting it gives,
  `normalised`), as is a URI `urlsplit` itself rejects (an unclosed IPv6 bracket; `normalised: null`).
  `s3_to_https` applies the same rule.
- Path segments follow the key rule. User info, a query, a fragment, a malformed port, a missing host, any
  other scheme, and plain `http` to anything but a loopback IP literal (`localhost` included: it is resolved
  by name, and the licence gate refuses it) are all refused (`IO_READ_URI_INVALID`). The licence gate is not
  consulted about a URI apok-eval would never read.
- `describe()["s3_mapping"]` is `S3_HTTPS_RULE` exactly when the base URI's scheme is `s3`.
- **Clearance: only one the licence gate issued.** Construction requires an object whose type is exactly
  `apok_eval.licence.LicenceClearance` (`type(c) is LicenceClearance`), the nominal type that only
  `apok_eval.licence.check_run` constructs (its constructor refuses without the gate's private token,
  `LICENCE_CLEARANCE_UNISSUED`). Any other object is refused (`IO_READ_NOT_CLEARED`, naming the type received
  in `clearance_type`) before its `covers` is even called. Two weaker checks would let a stand-in answering
  `True` open the gate by accident:
  - `isinstance(c, apok_eval.clearance.Clearance)`: that Protocol is runtime-checkable, so it accepts any object
    with a `covers` method. The protocol stays for type annotations only.
  - `isinstance(c, LicenceClearance)`: `isinstance` consults `__class__`, so it accepts
    `unittest.mock.Mock(spec=LicenceClearance)` and `create_autospec(LicenceClearance, instance=True)` (the
    usual accidental stand-ins in a test), any object whose `__class__` names the class, and a subclass that
    skips the token and overrides `covers`. A test builds each of these, answering `True`, and checks that the
    store is refused and the server sees no request.
- The clearance's `covers(base_uri)` must then return `True`, the bool itself and not merely a truthy value
  (`IO_READ_NOT_CLEARED`, naming the `uri`). Before each request, `covers(object_uri)` must also return
  `True` (`IO_READ_NOT_CLEARED`, naming the `uri` and `key`). Both questions use the caller's form of the URI:
  `s3://…` if the base was given as `s3://…`. No request is made for an uncleared URI.
- **The per-object question is a guard, not a formality.** The gate compares whole segments of canonical URIs,
  so covering the base usually means covering every object under it, but not where the gate remaps a path
  prefix. It reads the gateway path `https://data.aws.ash2txt.org/samples/<k>` as
  `s3://vesuvius-challenge/<k>`, so a clearance of `https://data.aws.ash2txt.org` clears an `HttpStore` on
  that root but none of its `samples/…` objects, and there the per-object question is the only guard. A real
  clearance for that host needs a person's own acceptance of the Vesuvius Challenge terms (the registry refuses
  a terms-free licence on it, `LICENCE_REGISTRY_GATED_HOST`). The tests therefore pin the remap with
  `canonical_uri` and make the question fire by patching the gate's answer on a loopback store.
- **What the exact-type check does not stop.** Python cannot stop deliberate forgery: `object.__new__` with
  `object.__setattr__` builds an object of the exact type with any inputs, and assigning
  `LicenceClearance.covers` changes every clearance's answer (the tests do that on purpose, through
  `monkeypatch`, to simulate a gate answering otherwise). The check stops a clearance being faked by accident
  or by a careless caller, as the licence gate's token does ([docs/licence.md](licence.md), "The clearance").
- In the tests, every store that is built gets a real clearance: `tests/io_zarr_fixtures.clearance_for`
  writes a registry to the test's `tmp_path` (each URI's host declared with `requires_terms: []`, allowing
  MIT; each URI an MIT asset, `synthetic` on the loopback test server and `dataset` on fictional buckets and
  hosts no test contacts) and calls `check_run`; MIT needs no terms, so no run configuration carries
  `terms_accepted_by`. The IP-literal forms of the instance-metadata service cannot be declared in a
  registry at all (`LICENCE_URI_AMBIGUOUS`), so the address check below is tested on them directly.
- **Requests.** Each request is a plain `GET` with `User-Agent: apok-eval/<version>`; `http.client` adds
  `Accept-Encoding: identity` (a test checks the header). There are no credentials, no `Authorization` or
  cookie header, and no request signing. Environment proxy variables are ignored. Redirects are not followed.
  TLS is verified with the default context.
- **Addresses.** Every address the host resolves to is checked immediately before connecting. Unless all of
  them are loopback or global (Python's `ipaddress` `is_loopback` / `is_global`), nothing is connected
  (`IO_READ_ADDRESS_FORBIDDEN`, naming the `host`, `port` and the forbidden `address`). The connection then
  goes to a checked numeric address, so the name is not resolved a second time and cannot be rebound in
  between. The EC2 instance-metadata service (link-local `169.254.169.254`, private `fd00:ec2::254`) is
  therefore never contacted, however the host is written: an IP literal, a legacy numeric form such as
  `https://2852039166/` (which the resolver reads as 169.254.169.254), or a DNS name that resolves there.
  Link-local, private, shared, unspecified and other special-purpose addresses are refused for the same
  reason; a mirror on a private network is not a supported source.
- **Statuses.**
  - 200: the body exactly as served. Requests carry `Accept-Encoding: identity`, so a compliant server sends
    the stored bytes.
  - 404: absent.
  - 401: refused (`IO_READ_AUTH_REQUIRED`) and never retried. Reads are anonymous and a 401 refuses:
    credentials are never sent, so a server that wants them is not read.
  - 403: refused (`IO_READ_FORBIDDEN`). Anonymously, S3 also answers 403 for a missing key in a non-listable
    bucket, so a 403 is never treated as absent or as fill.
  - 5xx: retried (below); after the last retry, refused (`IO_READ_HTTP_STATUS`) with the status and the number of
    attempts.
  - Any other status, redirects included: refused at once (`IO_READ_HTTP_STATUS`) with the status and the number
    of attempts.
  - A connection, TLS or timeout error, a body shorter than its `Content-Length`, or a response `http.client`
    cannot parse (it raises `ValueError` for a negative chunk size): retried (below); after the last retry,
    refused (`IO_READ_NETWORK`), naming the underlying error's class (`error`) and the number of `attempts`.
- **Retries.** A 5xx answer or a failed transfer is retried up to `retries` times (0–5), with a linear wait of
  `retry_wait_s × attempt` before retry `attempt` (`retry_wait_s` in 0..3600 s, `timeout_s` in (0, 3600] s: larger
  values overflow `time.sleep` and socket timeouts). One counter covers both, so a read makes at most
  `retries + 1` requests whatever mix of failures it meets. Each retry is appended to `retry_events`, as
  `{"key", "status", "retry"}` after a 5xx or `{"key", "error", "retry"}` after a failed transfer, and appears in
  `describe()`. A refusal raised on the way to a request (`IO_READ_NOT_CLEARED`, `IO_READ_ADDRESS_FORBIDDEN`) is
  never retried, and neither is a local error writing `download`'s file (below). A failed transfer is retried as a
  5xx is, so a long read is not ended by one TLS handshake timeout. Tests make a loopback server reset the
  connection, stall past the timeout, and leave a TLS handshake unanswered.
- **Other reads.** Each is cleared like `get` (the gate is asked about its own URI first) and has the same
  statuses, retries and refusals:
  - `get_base()`: the object at the base URI itself, for a base that names one file.
  - `get_listing(key="")`: the listing of the folder `key` (the base when empty), whose URI ending in `/` is what
    the clearance must cover. Over http(s) it is the index page the folder's URL, ending in `/`, serves; under an
    `s3://` base it is the bucket's anonymous ListObjectsV2 answer for the folder's prefix, one level deep, at most
    1000 keys (`s3_list_url`).
  - `download(key, path)`: streams the object to `path` without holding it in memory and returns
    `(size, "sha256:<hex>")`, or `None` when absent. It writes `path.partial`, moves it into place only when
    complete, and removes it after every failed attempt, so a refused download leaves nothing at `path`. A body
    shorter than its `Content-Length` is a failed transfer. A `path.partial` that cannot be opened, written or
    closed (a missing directory, a full disk) is a local error, not a failed transfer: it is refused at once
    (`IO_READ_SINK_UNWRITABLE`, naming the `path` and the `error`) and removed, and the object is not requested
    again.
  - `get_range(key, start, length)`: `length` bytes from byte `start` (fewer at the object's end), asked for with
    a `Range` header: the server's 206 answer, or the slice of a 200 answer from a server that ignores ranges;
    416 (the range starts past the end) reads as `b""`, 404 as `None`. A `start` below 0, a `length` outside
    1..64 MiB or a non-int is `IO_READ_OPTION_INVALID` (`field: "range"`).

## 2. The read log (`apok_eval.io.readlog`)

`LoggedStore(store, log)` records every `get` in a `ReadLog` before the bytes reach the caller. The OME-Zarr
reader reads everything through one: `.zgroup`, `.zattrs`, the sidecars `meta.json` and `metadata.json`, every
`<k>/.zarray` and `<k>/.zattrs`, and every chunk. Entries follow the `apok_eval.merkle` chunk-manifest convention
exactly:

```
present: {"key": "<store key>", "digest": "sha256:<hex of the bytes as stored>", "size": <bytes>}
absent:  {"key": "<store key>", "absent": true}
```

- Digests are of the raw stored bytes, before any decompression.
- `entries()` is sorted by the UTF-8 bytes of the key, and `root()` is `apok_eval.merkle.manifest_root(entries())`
  (scheme `apok-merkle-v1`).
- Keys are relative to the volume root, so the same objects read locally and over HTTP give the same root. A
  test checks this.
- A key read a second time must come back identical. Different bytes, or present one time and absent the other,
  is refused (`IO_OBJECT_CHANGED`, naming the key and both records), and the first record is kept. Overlapping
  reads therefore re-fetch shared chunks and check them against the log.
- Absent chunks enter the manifest explicitly as absent. A manifest therefore cannot be satisfied by deleting
  chunks, even though an absent chunk reads as its fill value (see §5).

## 3. OME-Zarr v2: format support (`apok_eval.io.omezarr`)

apok-eval reads Zarr v2 itself. zarr-python is used only as a test oracle. `open_volume` performs every check below
before any chunk is read and refuses at the first violation.

### 3.1 Group and multiscales

- `.zgroup` must exist with `zarr_format` equal to the integer 2 (`IO_ZARR_GROUP`). `true` and `2.0` are refused.
- `.zattrs` must exist and hold `multiscales` as a non-empty list of objects. Only `multiscales[0]` is read, as
  VC does (`volume-cartographer/utils/src/zarr.cpp:419-462`). Its `datasets` must be a
  non-empty list of objects with string `path` values (`IO_ZARR_MULTISCALES`).
- Every metadata object must be strict UTF-8 JSON (`apok_eval.jsonstrict.loads`, [docs/jsonstrict.md](jsonstrict.md)) whose top
  level is an object (`IO_OBJECT_NOT_JSON`, naming `key`, and for a parse failure the parser's problem in `problem`; a top
  level that is not an object is named in the reason only, `.zgroup holds a JSON list, not an object`). Duplicate
  keys are refused, because parsers disagree on which value wins. So are `NaN`/`Infinity`, any number that
  overflows a double (such as `1e400`), an integer past `int()`'s digit limit, a byte-order mark, invalid UTF-8,
  nesting deeper than 256 levels (an ignored member included; deeper still, nesting that would exhaust the parser), and a lone-surrogate
  escape such as `"\ud800"` in any key or string. Such an escape is pure ASCII and parses, but the result is not
  Unicode: copied into `Volume.describe()`, it could not be written as UTF-8. A valid surrogate pair
  (`"\ud83d\ude00"`) is one character and is accepted.
- The sidecars `meta.json` and `metadata.json` are fetched and recorded, present or absent, so that a change to
  them changes the read-log root. They are not parsed or interpreted.
- **Axes.**
  - When `multiscales[0].axes` is present, it must name exactly `z, y, x` in that order, as strings or as
    objects with a `name` (`IO_ZARR_AXES`). This is VC's `requireZyxAxes`
    (`volume-cartographer/core/src/render/ZarrChunkFetcher.cpp:607-627`), which VC applies only when `axes` is
    present (`:677`).
  - When `axes` is absent, the caller must declare the order, `axes_declared=("z", "y", "x")` (a tuple or list
    of strings), together with who declares it, `axes_declared_by` (a non-blank source such as
    `"registry:<asset id>"`). Otherwise the open is refused (`IO_ZARR_AXES_UNDECLARED`).
  - The two come together or not at all: one without the other is `IO_ZARR_ARGUMENT_INVALID`, naming the
    missing one. A declared order other than exactly `z, y, x` is refused (`IO_ZARR_AXES`, `field:
    "axes_declared"`, naming the order and the declarer) before anything is read, whether or not the store has
    axes of its own.
  - `Volume.axes` records `order`, `source` (`.zattrs` when the store has axes, else `declared`), and the
    caller's `declared` order and `declared_by`, or `null` for both when there was no declaration.
  - A level's own `<k>/.zattrs`, when present, must also name `z, y, x` in any of `_ARRAY_DIMENSIONS`,
    `dimension_names` and `axes` it carries. This is VC's `validateArrayAxes` (`ZarrChunkFetcher.cpp:629-652`).
- **Dataset paths** must be exactly `"0", "1", …, "N"` in that order (`IO_ZARR_DATASET_PATHS`, naming the index,
  the path found and the path expected). That one rule covers non-canonical numerals (`"01"`), gaps, duplicates
  and reordering. VC requires canonical numerals with no gaps or duplicates (`ZarrChunkFetcher.cpp:686-702`).
  apok-eval also requires NGFF's highest-to-lowest resolution order.
- **At most 32 datasets** (paths `"0"`..`"31"`), VC's range (`ZarrChunkFetcher.cpp:695-696`). More is refused
  before any transform is read (`IO_ZARR_DATASET_PATHS`, `field: "multiscales[0].datasets"`, naming the `count`
  and the `limit`). This keeps every `2^k` small: without the bound, 1025 dyadic datasets made the dyadic
  check fail with a bare `OverflowError` (2^1024 does not fit a float) instead of a refusal.
- **Transforms** (VC's strict rules, `ZarrChunkFetcher.cpp:704-772`):
  - Multiscales-level `coordinateTransformations` are refused.
  - Per dataset, `coordinateTransformations` is either absent or a non-empty list. The list holds exactly one
    `scale` and at most one `translation`, each with three finite numbers (booleans are not numbers)
    (`IO_ZARR_TRANSFORM`).
  - Either every level declares transforms or none does. VC would default a missing level-0 scale to 1, and
    apok-eval refuses the mixed case instead.
  - Scales must be positive (`IO_ZARR_SCALE`, `rule: "positive"`), then isotropic (`IO_ZARR_SCALE`,
    `rule: "isotropic"`) and dyadic: `scale_k = scale_0 · 2^k` (`IO_ZARR_SCALE_NOT_DYADIC`).
  - Translations must be 0 (`IO_ZARR_TRANSLATION`): `|t| ≤ 1e-9 · scale` on each axis, that is, below 1e-9 of
    the level's own voxel.
  - Scale comparisons are relative at every magnitude: both values finite and
    `|a − b| ≤ 1e-9 · max(|a|, |b|)`. An expected scale that overflows to infinity therefore never matches.
  - **Deviation from VC.** VC's `finiteMetadataEqual` bounds by `1e-9 · max(1, |a|, |b|)`
    (`ZarrChunkFetcher.cpp:601-605`), and VC compares translations with it against 0. Below magnitude 1 that is
    an absolute 1e-9, so the verdict depends on the unit a writer chose: scales of `1e-10, 3e-10, 9e-10` passed
    as dyadic, metre-valued scales could be off by a relative 1e-4, and a translation of 1e-12 at scale 1e-10
    (a hundredth of a voxel) passed as zero, while 3e-7 at scale 3e5 (1e-12 of a voxel) was refused. apok-eval's
    tolerances are unit-free. They agree with VC's for scales of magnitude 1 or more, and are stricter below.
  - Scale values are never read as physical units. Writers label relative factors `"micrometer"`
    (`vesuvius/src/vesuvius/scripts/tifs_to_ome_zarr.py:441-476`), and VC writes physical values. Both pass, and
    both are recorded per level without interpretation. Datasets with no transforms are accepted, as VC's
    strict validator accepts them, and are recorded with `scale: null`.

### 3.2 Arrays (`<k>/.zarray`)

| field | rule | code |
|---|---|---|
| presence | `<k>/.zarray` must exist, with all eight v2 keys: `zarr_format, shape, chunks, dtype, compressor, fill_value, order, filters` | `IO_ZARR_ARRAY_META` (field = the first missing key) |
| `zarr_format` | the integer 2 | `IO_ZARR_ARRAY_META` |
| `shape`, `chunks` | three positive integers (not booleans or floats) | `IO_ZARR_ARRAY_META` |
| `dtype` | `"\|u1"` or `"<u2"` exactly, the same on every level. Big-endian, signed, float and other spellings (`"<u1"`) are refused. VC's fetcher supports uint8 and uint16 only, one dtype for all levels (`ZarrChunkFetcher.cpp:519-525`) | `IO_ZARR_DTYPE` |
| `order` | `"C"`. VC ignores the field; one villa reader honours `F`; apok-eval refuses rather than guess | `IO_ZARR_ORDER` |
| `filters` | `null` or `[]`. VC does not parse filters (`volume-cartographer/utils/src/zarr.cpp:103`), and `spiral-fitting/pack_resident_pools.py:58-59` refuses them | `IO_ZARR_FILTERS` |
| `dimension_separator` | absent means `"."` (recorded as `separator_declared: false`, as at `vesuvius/src/vesuvius/data/zarr_chunk_index.py:407`); otherwise `"."` or `"/"`. `null` and anything else are refused | `IO_ZARR_SEPARATOR` |
| `fill_value` | `null`, which reads as 0 (recorded as `fill_value_declared: null`, as VC maps null at `volume-cartographer/utils/src/zarr.cpp:96`), or a number with an integral value within the dtype. An integral float such as `0.0` is read as the integer (VC reads every number as a double, `zarr.cpp:93-94`) and recorded as declared (`fill_value_declared: 0.0`, `fill_value: 0`). Fractions (`1.5`), out-of-range values, strings and booleans are refused | `IO_ZARR_FILL` |
| `compressor` | see §3.3 | `IO_CODEC_*` |

### 3.3 Compressors

- **Allowed:** `null`; `blosc` with `cname` in `blosclz, lz4, lz4hc, zlib, zstd`; `zstd`; `zlib`; `gzip`; `lz4`.
- **Decoding:** through numcodecs, constructed at open time. If numcodecs cannot be imported, or lacks the codec
  class, the open is refused (`IO_CODEC_UNAVAILABLE`, naming the codec and the `.zarray`). Raw (`null`) volumes
  need no codec library.
- **Refused** (`IO_CODEC_UNSUPPORTED`):
  - villa's repo-private codecs `vc-delta3d` (`volume-cartographer/scripts/recompress_zarr.py:13-14`), `vcz1`
    (`spiral-fitting/pack_resident_pools.py:105`) and `c3d` (`zarr.hpp:301-304`);
  - every other id;
  - a compressor that is not an object with a string `id`;
  - a blosc compressor with a missing or other `cname`.
- **Decode errors** of any kind from the third-party decoder are refused (`IO_CODEC_DECODE`, naming the chunk key
  and codec).
- **Blosc frame header.** numcodecs 0.15 decodes a Blosc frame flagged MEMCPYED (stored uncompressed, which is
  what Blosc does with small chunks) by copying the header's `nbytes` without comparing them with the bytes
  present. A test probe showed a truncated 64-byte chunk "decoding" to adjacent heap memory at exactly the
  expected size, which the size check cannot catch. apok-eval therefore checks the 16-byte header before decoding.
  The header layout is c-blosc's: version, versionlz, flags, typesize, then u32 LE nbytes, blocksize, cbytes.
  - `cbytes` must equal the stored object's length, and must equal `nbytes + 16` when MEMCPYED is set, else
    `IO_CODEC_DECODE` with `stage: "header"`.
  - `nbytes` must be one of the two allowed decoded sizes, else `IO_ZARR_CHUNK_SIZE` with `stage: "header"`.
    This also bounds the decoder's output allocation.
- **Decompression bombs.** No chunk decodes to more than the full chunk's byte count plus one, whatever its codec,
  so a small hostile or corrupt chunk is refused, not allocated (a 1 MB zlib stream of 1 GiB of zeros once made
  the reader allocate 2 GB before the size check ran).
  - zstd: the first frame's header must declare a `Frame_Content_Size` (RFC 8878 §3.1.1.1; numcodecs 0.15 cannot
    decode a frame without one either) equal to one of the two allowed sizes. No zstd header, or no declared
    size, is `IO_CODEC_DECODE` with `stage: "header"`; another size is `IO_ZARR_CHUNK_SIZE` with
    `stage: "header"`. The decoder writes into a buffer of exactly that size, so a frame holding more fails.
  - lz4: numcodecs' 4-byte little-endian size prefix is checked the same way (an object shorter than 4 bytes is
    `IO_CODEC_DECODE`, `stage: "header"`).
  - zlib and gzip: the stream is inflated by the standard library with an output bound of the full chunk + 1
    bytes, as numcodecs would decode it otherwise (gzip members concatenated; bytes after a zlib stream
    ignored). When more remains, `IO_ZARR_CHUNK_SIZE` with `stage: "decoded"` and `size_at_least` (the bound
    reached, not the stream's full size, which is never computed). A stream that ends before its end marker is
    `IO_CODEC_DECODE`.

### 3.4 Chunks

- **Keys.** The chunk at grid index `(a, b, c)` of level `k` is `"<k>/" + sep.join((a, b, c))`, where `sep` is the
  level's separator. The grid is `ceil(shape / chunks)` per axis.
- **Absent chunks.** An absent chunk (no file, or HTTP 404) reads as the level's fill value
  (`spiral-fitting/fiber_direction_samples.py:56-58` does the same). Each read's record names the absent chunks
  (§4), so a read made wholly of fill is visible to its caller. That is what happens when the stored layout and
  the separator disagree: for example, nested `/` chunks whose `.zarray` omits `dimension_separator`, as zarr's
  `NestedDirectoryStore` wrote before zarr 2.8, are read under the `.` default, and every key is absent.
- **Decoded size.** A stored chunk must decode, in C order, to either:
  - the full chunk shape, whose padding beyond the array edge is never read (zarr-python always writes edge
    chunks this way), or
  - at an array edge, exactly the edge-cropped shape. VC-style writers may store edge chunks cropped, and
    `fiber_direction_samples.py:65-77` accepts both forms.

  Any other byte count is refused (`IO_ZARR_CHUNK_SIZE`, `stage: "decoded"`, naming the key, the size and the
  allowed sizes). Interior chunks therefore have exactly one allowed size.

### 3.5 Decoding without a volume: the CT control's reads

`chunk_codec(compressor, key)` is the allow-list of §3.3, and `decode_chunk(key, data, *, compressor, codec,
dtype, chunks, edge)` the decoding and size checks of §3.3–§3.4, without a `Volume`; `Volume.read` decodes every
chunk through `decode_chunk`. `edge` is the chunk's shape cropped at the array edge, or `chunks` itself for a
caller that takes only full-size chunks.

The CT positive control (`apok_eval.eval.local.open_ct`, used by `apok-eval e7v3 --volume`) reads its level
through them:

- The level's compressor must be allow-listed when its `.zarray` is read, before any chunk is fetched
  (`IO_CODEC_UNSUPPORTED`, naming the key and the codec).
- Every chunk is decoded by `decode_chunk`. A Blosc header that does not fit the object, a zstd or lz4 object
  declaring another size, and a zlib or gzip stream inflating past the full chunk are refused as in §3.3; a chunk
  of any other size is `IO_ZARR_CHUNK_SIZE`.
- Only full-size chunks are taken (`edge = chunks`): an edge-cropped chunk is `IO_ZARR_CHUNK_SIZE`.
  `ChunkCache.sample` (`src/apok_eval/eval/ctprofile.py`) bounds a voxel by its chunk's array, not by the level's
  shape, and a ray can reach past the edge, so padding a cropped chunk would make up values that are then read.
- Every object read (`.zattrs`, the level's `.zarray`, every chunk) goes through a `LoggedStore`, and the run's
  record lists them as its `volume` chunk manifest, so `apok-eval verify` reads them again.

The control does not open the volume with `open_volume`, so no `VolumeFrame` is declared and the group, axes,
transform and pyramid checks of §3.1, §3.2 and §4 are not run on its reads; it checks the level's shape against
the declared frame (`EVAL_RUN_FRAME`), and its dtype, order, filters and chunk rank, itself (`EVAL_RUN_VOLUME`).
`open_volume` takes only a level-0 frame (`IO_ZARR_SHAPE_MISMATCH` unless level 0 has the declared shape), while
a spiral fit's frame is often a downsampled level of the volume, which `open_volume` cannot be told as it stands.

## 4. Pyramid and coordinate rules

**The frame is checked before any read.** `open_volume(store, declared: VolumeFrame, ...)` refuses unless level 0's
`shape` equals `declared.shape_zyx` (`IO_ZARR_SHAPE_MISMATCH`, naming both shapes). The volume id and voxel size
are the declaration's, from the registry or the run configuration. They are never read from the store: neither
`.zattrs` units, nor sidecars, nor folder names.

**Pyramid shapes.** By default (`pyramid_tolerance="exact"`), level `k` must have shape `ceil(shape_0 / 2^k)` on
every axis. This equals `shape_k = ceil(shape_{k-1} / 2)`, by the identity
`ceil(ceil(n / 2^(k-1)) / 2) = ceil(n / 2^k)`. The legacy Scroll 1 pyramid follows it
(`vesuvius/src/vesuvius/examples/notebooks/example1_data_access.ipynb:231-242`).

With `pyramid_tolerance="chunk-padding"`, a level may exceed that shape by less than one of its own storage chunks
per axis. This is VC's tolerance, measured against level 0 (`ZarrChunkFetcher.cpp:590-593`, `936-948`). The
padding is recorded per level. A level smaller than the expected shape is refused in both modes
(`IO_ZARR_PYRAMID_SHAPE`, naming the level, the axis, the shape, the expected shape and the tolerance).

**Reads.** `Volume.read(zyx_start, zyx_stop, level=0, *, clip=False)` takes a half-open box `[start, stop)` in
**level-0** voxels.

- Starts and stops are ordered `(z, y, x)` triples: a tuple, a list or a 1-D NumPy array of Python or NumPy
  integers, not booleans or floats, with `start < stop` on every axis (`IO_ZARR_BOX_INVALID`). Sets, dicts,
  generators, ranges and other iterables are refused: they have no `(z, y, x)` order of their own (a set
  `{4, 0, 2}` would iterate as `(0, 2, 4)`).
- The level must exist (`IO_ZARR_LEVEL_INVALID`).
- A box reaching outside the level-0 shape is refused, with no silent clipping (`IO_ZARR_BOX_OUTSIDE`). With
  `clip=True` it is clipped, and the record says so. A box with nothing left after clipping is still refused.

**Mapping.** Level-0 index `i` maps to level-k index `floor(i / 2^k)`. Level-k index `j` covers level-0
`[j·2^k, (j+1)·2^k)`. The level-k box read is therefore `[floor(start / 2^k), ceil(stop / 2^k))`, the smallest
level-k box covering the request. It may cover more of level 0 than was asked for.

**Records.** Every read returns `(array, TransformRecord)`. The record holds:

- `volume_id`, `level` and `factor` (2^k);
- the requested box, whether clipping was requested, whether it happened, and the level-0 box after clipping;
- the level-k box actually read;
- the level-0 extent the returned voxels cover, `[j_start·2^k, min(j_stop·2^k, shape_0))`;
- `chunks`, the number of chunk keys the level-k box covers (each fetched), and `absent_chunks`, the keys of those
  found absent, in grid order, with `fill_value`, the value they were read as;
- the mapping text;
- the voxel convention (below).

## 5. Declared assumptions (recorded in `Volume.describe()["assumptions"]`)

1. **Voxel convention (`apok-voxel-v1`).** Index `i` on any level names the voxel occupying `[i, i+1)` of that
   level's grid, and a level-k voxel stands for the level-0 block it covers. Whether its value is the level-0
   sample at `j·2^k` (strided, which is what NGFF scale-only transforms imply) or an aggregate of the block
   (mean, max) is **UNKNOWN**: the downsampling method is neither read nor checked. Villa writers use strided,
   mean, max and binary-or.
2. **Units.** `.zattrs` scales are checked only for positivity, isotropy and dyadic ratios. No unit or voxel size
   is read from the store; the voxel size is the declared frame's.
3. **Pyramid.** Level k is taken to be a 2^k-per-axis downsample of level 0, as its shape (and its scale, when
   declared) is checked to be.
4. **Absent chunks.** An absent chunk reads as its level's fill value. By value, absent, masked and true-fill
   voxels cannot be told apart (`spiral-fitting/pack_resident_pools.py:134-140`). The read log therefore records
   every absent chunk as absent.

Also recorded:

- `fill_value: null` read as 0, and an integral float `fill_value` read as the integer;
- a missing `dimension_separator` read as `.`;
- the axes' source (`.zattrs`, or the caller's declaration), and any declared order and who declared it;
- per read, the chunks read as fill;
- the pyramid tolerance and per-level padding;
- every per-level scale and translation;
- the store's description, including the S3 mapping rule and any retries;
- the read-log root.

## 6. Refusal codes

Every code is raised by a literal `raise Refusal(...)` and is triggered by at least one test that asserts the code
and the named cause.

| code | triggered by | cause named in details |
|---|---|---|
| `IO_STORE_KEY_INVALID` | a key that is not safe `/`-separated `[A-Za-z0-9._-]` segments | `key` |
| `IO_STORE_ROOT_INVALID` | a local root that is missing, a file, empty (`""`), or not a path (holding NUL or an unencodable surrogate) | `root` |
| `IO_STORE_ESCAPE` | a key that resolves through a symlink outside the root, whether or not the target exists | `key`, `resolved`, `root` |
| `IO_STORE_NOT_A_FILE` | a key naming a directory or a dangling or looping link; a key below a dangling or looping link | `key`, `resolved`; `component` (the broken link) when below one |
| `IO_STORE_UNREADABLE` | a key whose entry or file cannot be looked up or read for a reason other than "no such entry" (`EACCES`, `EIO`, `ENAMETOOLONG`) | `key`, `path`, `errno` |
| `IO_READ_NOT_CLEARED` | a clearance whose type is not exactly `apok_eval.licence.LicenceClearance` (a stand-in with `covers()`, a `Mock(spec=LicenceClearance)`, an object claiming the class, a subclass, `None`, ...); the gate's `covers()` not `True` for the base URI or for an object URI (an object can be uncovered under a covered base where the gate remaps a path: `https://data.aws.ash2txt.org/samples/…`) | `uri` and `clearance_type` (the type received); `uri` and `key` for an object |
| `IO_READ_URI_INVALID` | a URI not in normal form (uppercase scheme, whitespace or control characters, empty `?`/`#`, non-ASCII, unparseable); an unsupported scheme; a bad bucket; user info, query, fragment, bad port or no host; http to anything but a loopback IP literal (`localhost` included); bad path segments | `uri` and `normalised`, `scheme`, `bucket`, `host` or `path` |
| `IO_READ_OPTION_INVALID` | `retries` outside 0..5 or not an int; a wait outside 0..3600 s or non-finite; a timeout outside (0, 3600] s or non-finite; a `get_range` start below 0 or length outside 1..64 MiB | `field`, `value` |
| `IO_READ_AUTH_REQUIRED` | HTTP 401: the server wants credentials, which apok-eval never sends; never retried | `key`, `url`, `status` |
| `IO_READ_FORBIDDEN` | HTTP 403 | `key`, `url`, `status` |
| `IO_READ_HTTP_STATUS` | any status other than 200/401/403/404, including redirects (`get_range` also takes 206 and 416); 5xx after the retries | `key`, `url`, `status`, `attempts` |
| `IO_READ_NETWORK` | a refused or reset connection, TLS failure, timeout, truncated body or unparseable response (e.g. a negative chunk size), after the retries | `key`, `url`, `error`, `attempts` |
| `IO_READ_SINK_UNWRITABLE` | `download`'s local `path.partial` cannot be opened, written or closed (a missing directory, a full disk); refused at once, never retried | `key`, `path`, `error` |
| `IO_READ_ADDRESS_FORBIDDEN` | a host resolving (in any of its answers) to an address that is neither loopback nor global, e.g. the instance-metadata service; nothing is connected | `host`, `port`, `address` |
| `IO_OBJECT_CHANGED` | the same key read twice with different bytes or presence | `key`, `before`, `now` |
| `IO_OBJECT_NOT_JSON` | a metadata object that is not strict UTF-8 JSON (`apok_eval.jsonstrict`: a duplicate key, a lone-surrogate escape, nesting past 256 levels included), or whose top level is not an object | `key`; `problem` (the parser's problem) for a parse failure only |
| `IO_ZARR_ARGUMENT_INVALID` | `store` without get/describe; `declared` not a VolumeFrame; an unknown `pyramid_tolerance`; a blank or non-string `axes_declared_by`; `axes_declared` without `axes_declared_by` or the reverse; a non-bool `clip` | `field` |
| `IO_ZARR_GROUP` | `.zgroup` absent, or `zarr_format` not the integer 2 | `key`, `field`, `value` |
| `IO_ZARR_MULTISCALES` | `.zattrs` absent; `multiscales` not a non-empty list of objects; `datasets` missing or empty; a dataset without a string `path` | `key`, `field` |
| `IO_ZARR_AXES` | axes present but not exactly z, y, x (in multiscales or in a level's `.zattrs`); a caller-declared order that is not exactly `("z", "y", "x")` | `key`, `field`, `axes` (and `declared_by` for a declaration) |
| `IO_ZARR_AXES_UNDECLARED` | no axes in multiscales and no `axes_declared_by` | `key`, `field` |
| `IO_ZARR_DATASET_PATHS` | dataset paths that are not `"0".."N"` in order; more than 32 datasets | `index`, `path`, `expected`; or `count`, `limit` |
| `IO_ZARR_TRANSFORM` | multiscales-level transforms; a transforms list that is not a list or is empty; an unsupported or repeated type; values that are not three finite numbers; no scale; transforms on some levels only | `field`, `type` or `level` |
| `IO_ZARR_TRANSLATION` | a translation that is not zero | `level`, `translation` |
| `IO_ZARR_SCALE` | a scale that is not positive, or not isotropic | `level`, `rule`, `scale` |
| `IO_ZARR_SCALE_NOT_DYADIC` | `scale_k ≠ scale_0 · 2^k` | `level`, `scale`, `expected` |
| `IO_ZARR_ARRAY_META` | `.zarray` absent; a missing required key; `zarr_format ≠ 2`; malformed `shape` or `chunks` | `key`, `field` |
| `IO_ZARR_DTYPE` | a dtype other than `\|u1`/`<u2`, or one that differs from level 0's | `key`, `field`, `value` |
| `IO_ZARR_ORDER` | `order ≠ "C"` | `key`, `value` |
| `IO_ZARR_FILTERS` | non-empty filters | `key`, `value` |
| `IO_ZARR_SEPARATOR` | a `dimension_separator` other than `.`/`/`, including null | `key`, `value` |
| `IO_ZARR_FILL` | a `fill_value` that is neither null nor an in-range integral number | `key`, `value` |
| `IO_CODEC_UNSUPPORTED` | a compressor id that is not allow-listed (repo-private ones named as such); a malformed compressor; a bad blosc `cname` | `key`, `codec`, `cname` |
| `IO_CODEC_UNAVAILABLE` | numcodecs missing, or lacking the codec | `key`, `codec` |
| `IO_CODEC_DECODE` | the decoder raised; a Blosc header inconsistent with the object, a zstd object without a header declaring its content size, an lz4 object without its size prefix (`stage: "header"`) | `key`, `codec`, `stage`, `size`, `cbytes` |
| `IO_ZARR_CHUNK_SIZE` | a decoded size, or a Blosc `nbytes`, zstd content size or lz4 size prefix, that is neither the full nor the edge-cropped size; a zlib/gzip stream inflating beyond the full size (`size_at_least`); in the CT control's reads, any chunk that is not full-size (§3.5) | `key`, `stage`, `size`, `size_at_least`, `expected` |
| `IO_ZARR_SHAPE_MISMATCH` | a level-0 shape that differs from the declared frame | `stored`, `declared` |
| `IO_ZARR_PYRAMID_SHAPE` | a level shape that is not `ceil(shape_0 / 2^k)`, or is outside the chunk-padding tolerance | `level`, `axis`, `shape`, `expected`, `tolerance` |
| `IO_ZARR_LEVEL_INVALID` | a read level that is not an int in `0..N` | `level`, `levels` |
| `IO_ZARR_BOX_INVALID` | a box that is not two ordered integer triples (tuple, list or 1-D array) with start < stop | `start`, `stop` |
| `IO_ZARR_BOX_OUTSIDE` | a box reaching outside the level-0 shape without clip; a box wholly outside even with clip | `start`, `stop`, `shape` |

The named negative fixtures, each refused with its own code: a non-dyadic scale, a non-zero translation, wrong axis
order, axes missing without a declaration, a vcz1 compressor, a filters entry, F order, a big-endian dtype, a bad
chunk size, a shape that mismatches the declared frame, and a level shape that does not ceil-halve. They are
tested together in `tests/test_io_zarr_refusals.py::test_named_negative_fixture_is_refused_with_its_code_and_cause`.

## 7. What is not checked

- **Downsampling.** Neither the downsampling method nor the values on coarser levels are checked against level 0.
  A level could hold anything of the right shape. See assumption 1.
- **Fill values across levels.** Fill values may differ between levels. Each level reads with its own, recorded.
  VC's strict open requires a single fill value; apok-eval does not.
- **Other metadata objects.** Zarr v3 `zarr.json` and consolidated `.zmetadata` are not fetched. A store that also
  carries them is read through its v2 files only, and a reader that prefers them (VC3D picks v3 when `zarr.json`
  exists) could see different metadata.
- **Sidecar contents.** `meta.json` and `metadata.json` are digested, not parsed. Voxel size, energy and z-crop
  are not cross-checked against them.
- **Volume identity.** The store is not checked to belong to the declared `volume_id`, for example by folder name,
  and its `meta.json`/`metadata.json` sidecars are digested, not parsed: their schema in the open-data stores is
  not established by the sources this reader follows, and comparing an unknown schema would refuse real volumes
  on a guess. The declaration is trusted for identity; shape is the only cross-check. (Surfaces and umbilicus
  metadata that name a volume id are compared: `apok_eval.io.frame.stated_identity`, docs/io-surfaces.md §2.5.)
- **Codec headers against `.zarray`.** Beyond the declared decoded size, a codec header is not compared with
  the `.zarray` compressor configuration (level, acceleration, checksum flag).
- **Blosc header against `.zarray`.** The Blosc header's compressor and typesize are not compared with the
  declared `cname`/`shuffle`. The decoded bytes are what count.
- **Blosc content.** Blosc frames carry no checksum, so a bit-flipped MEMCPYED chunk with a consistent header
  decodes to wrong values. The read log's digests are how such a change is detected against a reference manifest.
- **HTTP bodies.** HTTP `Content-Encoding` is not undone. Requests carry `Accept-Encoding: identity`, so a
  compliant server sends stored bytes.

## 8. Public API (what `apok-eval io inspect` composes)

- `apok_eval.io.store.LocalStore(root)`, `apok_eval.io.store.HttpStore(base_uri, clearance, *, retries, retry_wait_s,
  timeout_s)`: object stores. `describe()` gives where they read from. `HttpStore` also has `get_base()`,
  `get_listing(key="")`, `download(key, path)` and `get_range(key, start, length)` (§1), and `retry_events`.
- `apok_eval.io.store.s3_list_url(folder)`: the anonymous ListObjectsV2 request for an `s3://` folder.
- `apok_eval.io.store.s3_to_https(uri)`: the canonical S3 → HTTPS mapping. `S3_HTTPS_RULE` is the rule as text.
- `apok_eval.io.store.check_key(key)`: the key rule.
- `apok_eval.io.readlog.ReadLog`: `record`, `entries`, `root`, `as_dict`. `apok_eval.io.readlog.LoggedStore(store, log)`.
- `apok_eval.io.omezarr.open_volume(store, declared, *, axes_declared=None, axes_declared_by=None,
  pyramid_tolerance="exact") -> Volume`.
- `Volume.read(zyx_start, zyx_stop, level=0, *, clip=False) -> (np.ndarray, TransformRecord)`.
  `TransformRecord.as_dict()` gives the record as JSON-ready data.
- `Volume.describe() -> dict`: canonical-JSON data with the store, frame, axes, multiscales, tolerance, levels,
  assumptions and read-log root. `Volume.log` is the `ReadLog`, and `Volume.levels` the validated `Level`s.
- `apok_eval.io.omezarr.chunk_codec(compressor, key)` and `apok_eval.io.omezarr.decode_chunk(key, data, *, compressor,
  codec, dtype, chunks, edge) -> np.ndarray`: the allow-list and the chunk checks without a `Volume` (§3.5).

`apok-eval io inspect` (§9) composes these; opening reads only metadata objects.

## 9. Inspect (`apok-eval io inspect`, `src/apok_eval/io/cli.py`)

```
apok-eval io inspect PATH [--frame FRAME.json | --no-frame] [--policy NAME] [--json]
```

One command for every format `apok_eval.io` reads; this section covers what it shares and the OME-Zarr reading.
The surfaces' readings are in [docs/io-surfaces.md](io-surfaces.md), "Inspect". It prints a short text summary, or with
`--json` the whole record; a refusal is printed as JSON by `apok_eval.cli` (exit 3). `--frame` and `--no-frame`
together, or an unknown `--policy`, are usage errors (exit 2); `inspect_path`, the function behind the command,
refuses both itself (`IO_INSPECT_OPTION_INVALID`).

**Local only.** A `PATH` that starts like a URI (`<scheme>://`, any case: `s3://`, `https://`, `file://`, ...)
is refused before anything is read (`IO_INSPECT_PATH_INVALID`, `problem: "remote"`): a remote inspect is out of
scope, so the licence gate is never involved. A path naming nothing is `IO_INSPECT_PATH_INVALID`
(`problem: "missing"`).

**Detection** is by explicit markers, never by guessing, and only picks the reader (which then judges the
input and names what is wrong):

| kind | markers (`detected_by` in the record) |
|---|---|
| `ome-zarr-v2` | a directory with an entry `.zgroup` or `.zattrs` (a dangling link counts) |
| `tifxyz` | a directory with an entry `x.tif`, `y.tif` or `z.tif` |
| `mesh` | a file with extension `.obj` or `.ply`, any case |
| `point-collections` | a `.json` file (any case) whose top level is an object with `vc_pointcollections_json_version` or `collections` |
| `umbilicus` | a `.json` file whose top level is an array, or an object with `control_points` or `points` |

Markers of two kinds (e.g. a directory with `.zgroup` and `z.tif`) are `IO_INSPECT_AMBIGUOUS`, naming the
`kinds` and their `markers`. No marker (an empty directory, a Zarr v3 `zarr.json` alone, a JSON object with
none of the keys), a `.json` file that cannot be read (`problem: "unreadable"`, naming the `error`: no
permission, an I/O error) or is not strict JSON (`problem: "not strict JSON"`, `error` = the parser's problem:
undecodable, malformed, a duplicate key, NaN, an overflowing number, a lone surrogate, nesting past 256 levels
or past the parser's recursion limit), any other extension, or something that is neither a regular file nor a
directory (a dangling link, a FIFO) is `IO_INSPECT_UNKNOWN`, naming the `problem` (and what was `looked_for`).
The `.json` sniff parses with `apok_eval.jsonstrict.loads` ([docs/jsonstrict.md](jsonstrict.md)), the parser the point-collections
and umbilicus readers use, so it never guesses a kind from a document they would refuse; a file that parses is
then read, and judged, by the module's reader. (Before the sniff was strict, a duplicate key or NaN reached the
reader picked by the keys and was refused there, as `IO_PCL_JSON_INVALID` or `IO_UMBILICUS_JSON_INVALID`.)

**OME-Zarr.** `open_volume(LocalStore(PATH), frame)`: every check of §3–§4, metadata only, no chunk read,
`pyramid_tolerance="exact"`, axes as the store declares them (inspect declares none, so a store without axes
is `IO_ZARR_AXES_UNDECLARED`). The summary is `Volume.describe()`.

- A frame is required: `--frame FRAME.json`, or an explicit `--no-frame`; neither is
  `IO_INSPECT_FRAME_REQUIRED` (the coordinate contract: the frame is declared, never assumed). With `--frame`
  a level-0 shape other than the declared one is `IO_ZARR_SHAPE_MISMATCH`.
- `--no-frame` runs every other check. The volume is opened against a stand-in frame whose shape is level 0's
  own (read from `0/.zarray` with `apok_eval.jsonstrict.loads`, the parser `open_volume` uses), so the shape
  comparison is vacuous. Whenever `open_volume` accepts `0/.zarray` it parses that same shape; whenever the
  stand-in's reading fails (absent, not strict JSON, a shape that is not three positive integers, a broken
  link), `open_volume` refuses level 0 itself, in its own order, before it compares shapes. The stand-in's
  volume id and voxel size never leave the module: the record's `frame` is `{"checked": false, "status": "NOT
  CHECKED", "why": "--no-frame given: ..."}` and the summary's `frame` is the string `"NOT CHECKED"`.

**The frame file** is strict JSON (`apok_eval.jsonstrict.loads`: UTF-8 without BOM, no duplicate key, no
`NaN`/`Infinity`, no number that overflows a double, no lone-surrogate escape, no nesting past 256 levels) holding
exactly `volume_id`, `shape_zyx`, `voxel_size_um` and `source`, the fields of
`VolumeFrame.as_dict()`, and optionally `axes`, which must then be `["z", "y", "x"]` (so `as_dict()` output is
accepted as it is). A file that cannot be read as strict JSON is `IO_INSPECT_FRAME_INVALID` (`path`,
`problem`): `problem` is the parser's problem for bytes that are not strict JSON, e.g. `duplicate key 'volume_id'`,
and otherwise the exception's type and message (`apok_eval.jsonstrict.problem_of`), e.g. `NotRegularFile: ...: no such
file` (or `not a regular file (a FIFO, ...)`), or `ValueError: ...` for a path `open()` refuses, such as one with a
NUL character, whose message depends on the Python version (`inspect_path` can be given such a path, though a command
line cannot carry it). The document is then read by `VolumeFrame.from_dict` (`src/apok_eval/io/frame.py`), the one
frame-file schema, so the Python API and the command line refuse a declaration alike (`IO_FRAME_INVALID`): not an
object (`field: frame`), a missing or extra field (`field: keys`, `missing`, `unknown`), other axes (`field: axes`),
all with `declaration` = `--frame '<path>'`; then `VolumeFrame`'s own checks (`field`, `value`), including an integer
in `shape_zyx` or `voxel_size_um` beyond 2**53 in magnitude: frames are compared in float64, which holds every
integer up to 2**53 exactly, rounds those beyond, and overflows from 2**1024 (an `OverflowError`, before
`VolumeFrame` checked it). It is read only after detection and the option checks pass.

**The record** (`apok-io-inspect-v1`): `path` as given (bytes that are not UTF-8 written `\xNN`, as in every
summary, so the record is canonical JSON; so is the frame `file`), `kind`, `detected_by`, `read_as` (every reading choice
inspect made, e.g. `{"pyramid_tolerance": "exact", "axes_declared": null}`), `frame` (`checked: true` with the
`file`, the `declared` frame, the `rule` compared and the `count` of points checked, `null` for a volume; or
`NOT CHECKED` and why) and `summary` (the module's own summary). The text form prints the header lines, then
each summary field whose value is a scalar or a short flat list, and counts the others.

| code | fires when | cause in details |
|---|---|---|
| `IO_INSPECT_PATH_INVALID` | `PATH` is a URI, or nothing exists there | `path`, `problem` (`remote` / `missing`) |
| `IO_INSPECT_UNKNOWN` | no marker; a `.json` that cannot be read or is not strict JSON; another extension; neither a file nor a directory; a reader that failed with an `OSError` (`problem: "unreadable"`, `errno`) | `path`, `problem` (`looked_for` when no marker; `error` when unreadable: the exception type; when not strict JSON: the parser's problem) |
| `IO_INSPECT_AMBIGUOUS` | markers of two kinds | `path`, `kinds`, `markers` |
| `IO_INSPECT_FRAME_REQUIRED` | an OME-Zarr volume with neither `--frame` nor `--no-frame` | `path`, `kind` |
| `IO_INSPECT_FRAME_INVALID` | the `--frame` file cannot be read as strict JSON (its schema and values are `VolumeFrame.from_dict`'s: `IO_FRAME_INVALID`) | `path`, `problem` |
| `IO_INSPECT_OPTION_INVALID` | `--policy` for a kind other than tifxyz; a tifxyz `--frame` without `--policy`; given to `inspect_path`, a policy not in `POLICIES` or both a frame and `no_frame` (on the command line each is a usage error) | `path`, `option`, and `kind` or (unknown policy) `value` |

Not checked by inspect: a remote store; content-based detection of meshes (the extension decides, as
`read_mesh` dispatches); umbilicus text files (`vc3d-text`, `thaumato-text`: they have no marker and are
`IO_INSPECT_UNKNOWN`); the chunks of a volume (inspect opens it, reads no box).

## 10. Open questions

1. **Separator, compressor and chunk shape of the published CT volumes.** These are not pinned in source, and
   reading bucket metadata needs each person's acceptance of the terms. The reader accepts every combination
   villa writes, but the default `pyramid_tolerance="exact"` will refuse a published pyramid that is padded.
   Should the registry record the tolerance per volume?
2. **Downsampling method of published pyramids.** This decides whether level-k voxel `j` is centred at level-0
   `j·2^k` (strided) or at `j·2^k + (2^k − 1)/2` (block aggregate). apok-eval records the question as UNKNOWN rather
   than choosing.
3. **`s3://vesuvius-challenge/`.** VC3D maps it to `https://data.aws.ash2txt.org/samples/`, while apok-eval applies
   the generic rule. Should the registry carry HTTPS locators for that bucket, or should apok-eval add a second,
   recorded rule?
4. **Proxies.** HttpStore ignores environment proxies, which keeps reads direct and reproducible. An owner
   machine that needs a proxy will need an explicit, recorded option.
5. **403 from a listable bucket.** On the public open-data bucket (listable, so missing keys give 404), a 403
   means real denial. On a non-listable bucket every absent chunk would be refused. Is that acceptable, or
   should a registry flag declare listability?
6. **Fill values across levels.** Should apok-eval require one fill value across levels, as VC's strict open does
   (`ZarrChunkFetcher.cpp:950-955`)?
7. **Inspect declares nothing a reader leaves to its caller.** `apok-eval inspect` passes no axes declaration,
   reads point collections role-less and picks an umbilicus variant from its keys ([docs/io-surfaces.md](io-surfaces.md),
   "Inspect"). Should it take `--axes-declared-by`, `--role` and `--variant`, or read those from the registry
   entry of the asset once the frame comes from there too?
