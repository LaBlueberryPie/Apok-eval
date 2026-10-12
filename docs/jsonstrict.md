# `apok_eval.jsonstrict` — the one strict JSON parser, and its deterministic writer

Module `src/apok_eval/jsonstrict.py`, tests `tests/test_jsonstrict.py`. It raises no `Refusal`: it raises one
exception type, and every caller wraps it in its own refusal code, naming the problem.

## Why one parser

Python's `json` keeps the last of two equal keys without a word, so `{"licence": "EDUCELAB-SCROLLS", "licence":
"MIT"}` reads as MIT and `"terms_accepted_by": {...}, "terms_accepted_by": null` hides an acceptance. It accepts
`NaN`, `Infinity` and `-Infinity`, which are not JSON (RFC 8259 §6); turns `1e400` into infinity; decodes a
`"\ud800"` escape to a lone surrogate, which is not Unicode text and can be neither encoded, digested nor written
back; and raises `RecursionError` (not `ValueError`) on deep nesting. Every reader in apok-eval calls this module
instead: the licence registry and run configurations, the ledger's records, sidecars and index, the OME-Zarr
metadata, tifxyz `meta.json`, point collections, umbilicus files, VC3D fibres, `apok-eval inspect`'s `.json`
sniff and `--frame` files, and `apok-eval e7v3`'s settings, windings and terms files.

## `loads(data, *, constants="reject", max_depth=MAX_DEPTH)`

`data` must be `bytes` or `bytearray` (anything else is a `TypeError`): every caller hands over exactly the bytes it
read, and digested where it digests. It returns the parsed value with Python's `json` types (an integer stays an
exact `int`, `1.0` stays a `float`, object member order is kept), or raises `StrictJsonError`, a `ValueError` whose
`str()` is the precise problem. A bad `constants` or `max_depth` argument is a plain `ValueError` (a programming
error, never a `StrictJsonError`).

| refused | problem string (examples) |
|---|---|
| bytes that are not UTF-8 (a UTF-8-encoded surrogate included) | `not UTF-8 (invalid start byte at byte 0)` |
| a leading byte-order mark (RFC 8259 §8.1) | `starts with a byte-order mark (U+FEFF), which JSON text must not carry` |
| a syntax error, an empty document | `invalid JSON: Expecting value at line 1 column 1` |
| a key twice in one object, at any depth, equal values included | `duplicate key 'zarr_format'` |
| `NaN`, `Infinity`, `-Infinity` (with `constants="reject"`, the default) | `non-standard constant NaN (NaN, Infinity and -Infinity are not JSON)` |
| a number that overflows a double | `number 1e400 overflows a double` |
| an integer with more digits than `int()` converts | `integer 11111111111111111111... (5000 characters) has 5000 digits, more than int() converts (4300)` |
| arrays/objects nested deeper than `max_depth` levels | `arrays and objects nested deeper than 256 levels` |
| nesting deep enough to exhaust the parser | `arrays and objects nested deeper than the parser can recurse (the limit is 256 levels)` |
| a string or key holding a lone surrogate | `lone surrogate '\ud800' in the string at $.truth_source.id: not Unicode text (it cannot be encoded as UTF-8)`; for a key, `... in a key of the object at $.a ...` |

Details that matter:

- **Order.** The parse runs first (encoding, BOM, syntax, keys, constants, numbers, recursion), then the depth
  check, then the lone-surrogate check; the first problem found is the one raised. The first lone surrogate in
  document order is named, with its path: `$` is the document, `.member` a member whose key is a plain ASCII
  identifier (`[A-Za-z_][A-Za-z0-9_]*`), `["member"]` any other member, the key written as `json.dumps` writes it
  (raw UTF-8, `\"` and control characters escaped), and `[index]` an array element. So `{"a.b": {"c": "\ud800"}}`
  is `$["a.b"].c`, never the `$.a.b.c` of `{"a": {"b": {"c": ...}}}`, and the member `"0"` is `$["0"]`, not the
  element `$[0]`.
- **Depth.** The outermost array or object is level 1; a scalar document has depth 0. The default `MAX_DEPTH` is
  **256**, chosen so that everything `loads` returns can be handed to `apok_eval.canonical.canonical_dumps` and to
  `dumps`, which recurse once per level (CPython's default recursion limit is 1000); no format apok-eval reads nests
  beyond a handful of levels. The depth is measured iteratively after the parse; nesting beyond what the parser
  can recurse (9 998 levels of arrays on the CPython 3.13 this was tested with; the bound depends on the
  interpreter) is caught as `RecursionError` and refused, so no `RecursionError` escapes. Callers use the default; `max_depth` exists for a caller that must declare a smaller
  limit.
- **Integers.** The limit is the interpreter's (`sys.get_int_max_str_digits()`, 4300 unless configured otherwise;
  it guards `int()` against quadratic input). Integers shorter than that stay exact, however large: the caller's
  field check judges them (a 400-digit coordinate is refused by the field that holds it).
- **Surrogates.** Text decoded from valid UTF-8 holds no surrogate, so only a `\uD800`–`\uDFFF` escape (either
  case) can produce one; a valid pair such as `"😀"` is one character and is accepted, as is an escaped
  backslash followed by the letters `ud800`.
- **Constants.** With `constants="collect"` the three constants parse to `nan`, `inf` and `-inf` and `loads`
  returns `(value, tokens)`, the tokens in document order. Collecting relaxes nothing else (`1e400` is still an
  overflow, not `Infinity`). Only the tifxyz reader collects: spiral can write a NaN `bbox` into `meta.json`, so the
  reader keeps the file and records the finding `meta-nonstandard-json` ([docs/io-surfaces.md](io-surfaces.md) §1).
- **Not checked here:** the top-level type (each caller refuses a non-object with its own code), schemas, member
  order, duplicate *values*, `-0` (an integer zero), NUL and other escaped control characters in strings (valid
  JSON).

## `dumps(value) -> bytes`

Sorted keys (by code point), 4-space indent, raw UTF-8 (no `\u` escapes), `": "` and `","` separators, a trailing
`"\n"`: deterministic bytes for files that C++ tools also read (tifxyz `meta.json`, PointCollections, umbilicus).
It accepts only plain JSON data by the `apok_eval.canonical` rules (dict with `str` keys, list, `str`, `int`, finite
`float`, `bool`, `None`: no tuple, no `int` key, no NaN, no numpy scalar) and then reads its own bytes back with
`loads`, so it refuses everything the reader would refuse — nesting past 256 levels included — and never writes a
file its own reader rejects. Problems: `not plain finite JSON: TypeError: tuple is not canonical JSON (at .a)`,
`not plain finite JSON: RecursionError: ...`, `would not read back as strict JSON: arrays and objects nested deeper
than 256 levels`. It was `tifxyz.dump_json_strict`, used by three modules (tifxyz, point collections, umbilicus).

## Callers, and the refusal each wraps the problem in

The convention: a refusal for bytes that are not strict JSON carries the parser's problem string, `str(exc)`, in
`problem` (`problem_of(exc)` gives that, or `<type>: <message>` when the read itself failed). Two codes keep a
shape of their own: `IO_INSPECT_UNKNOWN`, whose `problem` names one of its causes (`not strict JSON`,
`unreadable`, `no marker`, ...) and puts the parser's problem in `error`, and `LEDGER_INDEX_MALFORMED`, whose
`problems` lists every problem of a line. The file is named under each module's own key (`path`, `key`,
`record`, `declaration`, `registry`, `store`, `scroll_spec`).

| caller | reads | refusal code | cause in the refusal |
|---|---|---|---|
| `licence.registry.read_strict_json` | the asset registry, run configurations | `LICENCE_REGISTRY_UNREADABLE`, `LICENCE_RUN_UNREADABLE` | `problem` = the problem, or `<type>: <message>` for the `OSError` (`NotRegularFile` for a missing file, a FIFO or a device: `apok_eval.fileio.read_regular`), or the `ValueError` `open()` raises for a path holding a NUL or an unencodable surrogate |
| `ledger` record, sidecar, figures | records, `manifests/*.jsonl` lines, figures outputs | `LEDGER_RECORD_UNREADABLE`, `LEDGER_MANIFEST_INVALID`, `LEDGER_FIGURES_INVALID` | `record` and `problem` for a record; as before for a sidecar or figures (`manifest`, `source`/`pointer`, …), the reason carrying the problem |
| `ledger.writer` config copy | the `config` argument of `open_run`, read back from its canonical bytes | `LEDGER_ARGUMENT_INVALID` | `field: "config"`; the reason names what is refused (tuples, NaN, non-string keys, lone surrogates, nesting past 256 levels), not the parser's problem: the config is a Python value, and `canonical_dumps` refuses most of these before the read-back |
| `ledger.index` | index lines | `LEDGER_INDEX_MALFORMED` | `line`; `problems` = [`the line is not strict JSON: <problem>`], also in the reason |
| `io.omezarr._json_object` | `.zgroup`, `.zattrs`, `.zarray`, level `.zattrs` | `IO_OBJECT_NOT_JSON` | `key`, `problem` |
| `io.tifxyz.read_tifxyz` (`constants="collect"`) | `meta.json` | `IO_TIFXYZ_META_INVALID` | `file`, `path`, `problem` |
| `io.pointcollection`, `io.umbilicus` | PointCollections, umbilicus JSON | `IO_PCL_JSON_INVALID`, `IO_UMBILICUS_JSON_INVALID` | `path`, `problem` |
| `io.tifxyz` / `io.pointcollection` / `io.umbilicus` writers (`dumps`, each on the whole document it writes) | — | `IO_TIFXYZ_WRITE_INVALID`, `IO_PCL_WRITE_INVALID`, `IO_UMBILICUS_WRITE_INVALID` | `problem` |
| `io.cli` `.json` sniff (`apok-eval io inspect`) | a point-collections or umbilicus candidate, to see its top level | `IO_INSPECT_UNKNOWN` | `path`, `problem: "not strict JSON"`, `error` = the problem |
| `io.cli.load_frame` | `--frame` files | `IO_INSPECT_FRAME_INVALID` | `path`, `problem` = the problem, or `<type>: <message>` for the `OSError` (`NotRegularFile` for a missing file, a FIFO or a device), or the `ValueError` `open()` raises for a path holding a NUL |
| `io.cli` `--no-frame` stand-in | level 0's `.zarray`, for its shape | — (a failure leaves the refusal to `open_volume`, which parses the same bytes with this module) | — |
| `eval.fibres.read_fibre` | VC3D fibre JSON | `EVAL_FIBRE_JSON_INVALID` | `fibre`, `problem` |
| `eval.local.load_settings`, `eval.local.open_ct` | `--settings` files; the CT's `.zattrs` and the level's `.zarray` | `EVAL_RUN_SETTINGS`, `EVAL_RUN_VOLUME` | `path` or `key`, `problem` |
| `eval.cli` | `--windings` and `--terms` files | `EVAL_CLI_FILE_INVALID` | `file`, `path`, `problem` |

**Stricter than villa.** villa parses its JSON (fibres, point collections, `meta.json`, the umbilicus) with
Python's `json`. A document with a duplicated key, NaN/Infinity, an overflowing number, a lone surrogate or more
than 256 levels of nesting is accepted there and refused here. For every strict document `loads` builds the same
Python values as `json.loads` (same `float()`, `int()` and string decoding, same member order), so a value
apok-eval computes from a strict document is the one villa would compute from it.

## Tests

`tests/test_jsonstrict.py` asserts every refusal's exact problem string, the depth boundary (256 accepted, 257
refused, through objects and the deepest branch), the recursion catch, collect mode, the argument checks, the
writer's exact bytes and every writer refusal. Each module's own tests assert that its readers refuse a duplicate
key (and the other cases, nesting past 256 levels included) with its own code and the problem as the named cause.

## The remaining uses of Python's `json`

There is one strict JSON parser in `src/apok_eval`: `jsonstrict.loads`. `object_pairs_hook` and
`parse_constant` appear nowhere else, and `json.loads` / `json.load` appear only in this module, which is the
parser (`grep -rn "json\.load" src/apok_eval`). Writers use `json.dumps` on values apok-eval built, never to read:
`apok_eval.canonical.canonical_dumps` (seals and digests), `jsonstrict.dumps`, the command lines printing records
and refusals (`apok_eval/cli.py`, `io/cli.py`), and the demo's and the synthetic world's writers of fibre and CT
metadata files. Tests use `json.loads` and `json.dumps` as oracles and to write fixtures: what Python's `json`
reads is what villa reads, so a test can show that a document villa accepts is one apok-eval refuses.
