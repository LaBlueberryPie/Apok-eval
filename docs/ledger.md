# `apok_eval.ledger` — sealed run records and `verify`

One sealed JSON record per run: input digests, including a Merkle root over every chunk read; code commit;
configuration; seeds; model digests; environment. `verify` recomputes all of it, and no figure is quoted
without a record. The defining property: changing any recorded byte, digest or figure makes `verify` fail.
It is measured on synthetic fixtures (below). `apok-eval e7v3` and `apok-eval demo` seal every run with it
("How apok-eval uses it" near the end).

## Why the ledger exists

A spiral fit's run folder does not, by itself, let a third party re-check exactly which bytes were read and by
which code. villa records some provenance of its own; the ledger does not replace it, it adds to it. Three gaps
it fills:

- A headless fit checkpoint stores `input_manifest` as an empty dict, because only the interactive driver
  fills it (`spiral-fitting/fit_spiral.py:3579`, villa `fb8c2c4`).
- The run-folder date is the *local* `datetime.date.today()` (`spiral-fitting/fit_spiral.py:5268`) and
  `checkpoint_fitted.ckpt` is overwritten every 1000 iterations (`spiral-fitting/fit_spiral.py:185`,
  `_HEADLESS_AUTOSAVE_INTERVAL`); neither is a reliable provenance timestamp.
- vesuvius sets boto3 request and response checksums to `when_required`, relying on "TLS + S3 ETag"
  (`vesuvius/src/vesuvius/data/utils.py:254-259`); an ETag is not a content hash for multipart objects.
  Chunk digests must be of the raw bytes as fetched, and metadata objects must be leaves too.

The ledger records every input, the code commit and the environment itself, with digests it computes.

## Run directory

```
<run_dir>/record.json             the sealed record (canonical JSON + "\n")
<run_dir>/manifests/<name>.jsonl  one sidecar per chunk manifest
<run_dir>/code.diff               the diff of a dirty tree (see `code` below), only when the tree was dirty
<run_dir>/...                     the run's outputs (any other path)
```

Run-relative paths in a record are *safe relative paths*: POSIX, non-empty, no empty / `.` / `..` component,
no backslash, no NUL, valid UTF-8. Reading them, `verify` refuses any path whose resolved location passes
through a symlink (`LEDGER_PATH_UNSAFE`); the writer never writes through one either (a sidecar in a resumed
run directory whose `manifests/` is a symlink is refused). The writer reserves `record.json`, `code.diff`,
`manifests/…` and names starting `.apok-tmp-` (its temporary files) — an output there is refused.

Every file the ledger reads (inputs, models, outputs, sidecars, the diff, figures, records, the index) is
opened without blocking and must be a *regular file* once open: a FIFO would block a reader forever and a
device such as `/dev/zero` never ends, so both are refused (as a missing file), never read. Everything the
ledger parses back is read by `apok_eval.jsonstrict.loads` ([docs/jsonstrict.md](jsonstrict.md)) with its default limits: UTF-8
without BOM, no duplicate key, no NaN/±Infinity, no number that overflows a double, no integer past `int()`'s
digit limit, no lone surrogate, nested at most 256 levels deep (deeper documents could not be re-serialised
canonically and are unreadable).

## Record schema `apok-record-v1`

A record is one JSON object, stored as its canonical JSON (`apok_eval.canonical`: sorted keys, no whitespace,
UTF-8, shortest float repr) followed by one `\n`. All members are required; `null` is written where the
table says so. Lists of named entries are sorted by name, names unique within the list. Names (and
`run_id`) match `[a-z0-9][a-z0-9._-]{0,127}` in full.

| member | content |
|---|---|
| `schema` | `"apok-record-v1"` |
| `run_id` | caller-chosen name of the run |
| `created_utc` | when the run was opened, `YYYY-MM-DDTHH:MM:SSZ` (UTC, wall clock: recorded, not derived) |
| `purpose` | free text, non-empty |
| `code` | see below |
| `config` | the full run configuration (a JSON object), copied at open |
| `config_digest` | `digest_json(config)` |
| `seeds` | `{name: int}` — every seed the run used; bools refused |
| `seed_pointers` | `{name: JSON pointer}` — same names as `seeds`; RFC 6901 pointer into `config` where that seed is set |
| `inputs` | `[{name, uri, path, digest, size}]` — `path` absolute (or `null` for a remote-only input, which then needs `uri`); `uri` optional otherwise |
| `chunk_manifests` | `[{name, source_uri, scheme: "apok-merkle-v1", count_present, count_absent, root, manifest_file, manifest_file_digest}]` |
| `models` | `[{name, uri, path, digest, size, source}]` — `path` absolute; `source` free text (where the model came from) |
| `environment` | `{python, implementation, platform, machine, packages: {name: version or null}}` |
| `outputs` | `[{name, path, digest, size}]` — at least one; `path` relative to the run directory |
| `figures` | `{source, pointer, values}` or `null` — `source` names an output; `values` is the non-empty JSON object at `pointer` in that output's parsed content |
| `predictions` | `{registry_input, registry_digest, ids}` or `null` — `registry_input` names an input (the pre-registered prediction registry); `registry_digest` is that input's digest; `ids` sorted, distinct, non-empty strings |
| `seal` | `digest_json` of the record without `seal` |

All digests are `"sha256:<64 hex>"` of exact bytes. Sizes are byte counts.

`code`:

| member | content |
|---|---|
| `package_version` | `apok_eval.__version__` of the code that ran |
| `commit` | `HEAD` (40 lowercase hex) when the package directory is a *checkout of the package*: inside a git work tree whose `HEAD` or index holds at least one file under it. `null` otherwise — no repository and no `.git` entry in the package directory or above it, or a work tree that tracks nothing under it (a package installed into an ignored virtualenv inside an unrelated project: that project's `HEAD` says nothing about this code) |
| `dirty` | `true` when `git status --porcelain --no-renames --untracked-files=all` lists anything for the whole work tree (ignored files do not count); `null` when `commit` is `null` |
| `diff_digest`, `diff_file` | when dirty: the digest of the diff and `"code.diff"`; else both `null`. The diff is `git diff --no-ext-diff --no-textconv --no-color --binary --no-renames --src-prefix=a/ --dst-prefix=b/ HEAD` of the whole work tree (tracked files), followed by one new-file patch (`git diff --no-index … /dev/null <path>`, same options) for every untracked file under the package directory outside `__pycache__`. So the commit plus the diff reproduce every file of the package; untracked files elsewhere make the tree dirty but are not in the diff |
| `source_root` | the package directory relative to the work-tree top (`"src/apok_eval"`), `null` without a commit |
| `source_manifest` | `[{path, digest}]` for every file under the package directory, paths relative to it, sorted by UTF-8 bytes; directories named `__pycache__` skipped; untracked and ignored files included |
| `source_manifest_root` | `apok_eval.merkle.merkle_root` over `(path, digest)` |

A symlink, a non-regular or unreadable file, or a file name that is not a safe relative path (a backslash,
an undecodable byte) in the package directory is refused (`LEDGER_SOURCE_UNSUPPORTED`): it could not be
compared with git or recorded. A `.git` entry git cannot open (a moved or pruned worktree) and a tracked
package on a branch with no commit yet are refused (`LEDGER_GIT_FAILED`) — never recorded as "no commit".

The *expected manifest* is the source manifest the git state implies, from git objects alone: for a clean
checkout, the commit's files under `source_root` (`git ls-tree -r`, the sha256 of each blob); for a dirty one,
those files written to a temporary directory and patched with the diff (`git apply` outside any repository,
limited to `source_root`, without the user's or system git configuration), then digested. A submodule
appears as `commit:<oid>` and a symlink as `symlink:<sha256 of its target>`, which never equal a file digest.

Git runs with `LC_ALL=C`, literal pathspecs and `GIT_NO_REPLACE_OBJECTS` (a commit's own objects, not
`git replace` substitutes), and without `GIT_DIR`, `GIT_WORK_TREE`, `GIT_INDEX_FILE`, `GIT_COMMON_DIR`,
`GIT_OBJECT_DIRECTORY`, `GIT_ALTERNATE_OBJECT_DIRECTORIES`, `GIT_NAMESPACE` and `GIT_CEILING_DIRECTORIES`, so
the repository is the one the package directory is in and an inherited variable cannot hide it. The
repository is an input and is never written to: `status` runs with `--no-optional-locks`, and `git diff` (which
refreshes stale stat data in the index it reads, `--no-optional-locks` notwithstanding) reads a temporary copy
of the index that keeps the original's mtime — a copy stamped "now" would switch off git's racy-clean
re-check and miss a same-size edit made within the second of the last index write.

`environment` never contains the host name, the user name or environment variables. `platform` is
`platform.platform()` (OS, release, machine, libc). **Other members do carry local paths:** `inputs[].path` and
`models[].path` are absolute (that is how `verify` finds the bytes again), and `chunk_manifests[].source_uri`
may be a `file://` URI, so a record names the directories its inputs lived in, often including a home
directory and a user name (`/home/<user>/...`). A sealed record is therefore not fit for publication as it
stands: redact or relativise those paths (and re-seal, or publish them beside the original sealed record)
before a record leaves the machine. Recording paths relative to a declared data root would be a format
change (see "Open questions"). `packages` always lists `apok-eval` (the running `__version__`) and the
declared dependencies `numpy`, `tifffile`, `numcodecs` and `scipy` (`null` when no installed distribution is
found), and `imagecodecs` (the `lzw` extra: whether an LZW tifxyz decodes or is refused) and `zarr` only when
installed; versions come from distribution metadata.

### Chunk manifests

Entries follow `apok_eval.merkle` (present `{key, digest, size}`, absent `{key, absent: true}`; digest of the
stored bytes before decompression; an absent chunk is never recorded as its fill value). Callers should
include metadata objects (`.zgroup`, `.zattrs`, each level's `.zarray`, sidecar `meta.json`) as leaves so
metadata changes alter the root. The entry rules are `apok_eval.merkle`'s: an absent entry is *exactly*
`{key, absent: true}`, `size` is a non-negative `int` and not a `bool`, keys are unique. So that each set of
entries has exactly one byte form, the sidecar holds one canonical-JSON entry per line, each followed by `\n`, sorted
strictly by the UTF-8 bytes of `key`, no blank lines; an empty manifest is the empty file. `root` is
`apok_eval.merkle.manifest_root` of the entries.

## The seal, and what it does and does not prove

`seal = digest_json(record without "seal")`. A seal is self-contained, so **anyone who edits a record can
recompute it**. The seal therefore makes a record self-consistent, not tamper-proof. Tamper *evidence*
comes from anchoring the seal somewhere the editor cannot reach — a git commit of the index, a public post,
a message to someone else — and checking against it: `verify(expect_seal=SEAL)` /
`apok-eval verify --expect-seal SEAL`.

## Writing a run (`apok_eval.ledger.open_run` → `RunWriter`)

`open_run(run_dir, *, run_id, purpose, config, expected_bytes, index, resume=False, allow_dirty=False,
source_dir=None)` checks before compute is spent, read-only checks first:

1. arguments (`run_id`, non-empty `purpose`, `config` a canonical-JSON object, `expected_bytes` a
   non-negative int, `source_dir` a directory; default: the running `apok_eval` package);
2. the output side: `record.json` already in `run_dir` → refused even with `resume`; a non-empty `run_dir`
   needs `resume=True`; free disk at `run_dir` (or its nearest existing parent) must be ≥ `expected_bytes`
   (the caller's stated estimate; no margin is added);
3. the code: its identity must be recordable in `apok-record-v1` (a SHA-256 git repository, whose commit ids
   are 64 hex, is refused here, `LEDGER_SOURCE_UNSUPPORTED` naming `code.commit` — not after the compute at
   `seal()`); a dirty tree needs `allow_dirty=True`; for a checkout the source manifest must already equal
   the expected manifest (else `verify` would fail later — e.g. an ignored file inside the package, clean or
   dirty);
4. the index, if given: it must verify (chain only) and must not list `run_id`;
5. only then writes: creates `run_dir` and probes it with a real write+fsync, writes `code.diff` when dirty,
   creates and probes the index directory.

Then: `add_input(name, path, uri=None)`, `add_remote_input(name, data, uri=...)` (bytes as fetched; path
recorded `null`), `add_model(name, path, source=..., uri=None)`, `add_chunk_manifest(name, entries,
source_uri=...)` (validates, writes the sidecar), `set_seed(name, value, config_pointer=...)`,
`set_predictions(registry_input, ids)`, `add_output(name, path)`, `set_figures(output_name, pointer="")`.
No method takes a digest: the ledger computes every file digest itself (passing `digest=` is a
`TypeError`). Chunk-manifest entry digests are the exception by nature (the reader computed them from the
bytes it fetched); the sidecar's own digest is computed by the ledger. Input and model paths are recorded
absolute, as given (made absolute, symlinks *followed* and the target's bytes digested — the recorded path
is the link). Output paths must stay inside `run_dir` without symlinks.

`seal()`:

1. refuses a run with no outputs;
2. re-captures the code state and refuses if the commit or the source manifest changed since `open_run`
   (code edited or committed while the run was open). `dirty`/`diff` are recorded as of `open_run` — the
   code that ran;
3. assembles the record (lists sorted by name, environment captured now) and its seal;
4. runs every `verify` check except the git comparison on the in-memory record against `run_dir`; any
   failure refuses the seal with `LEDGER_VERIFY_FAILED` (e.g. an input or output changed after it was
   added) — a sealed record verifies at the moment it is sealed;
5. publishes `record.json` atomically: a temporary file in `run_dir` is written and fsynced, then
   hard-linked to `record.json` (a link never replaces an existing file; `os.rename` would), the temporary
   removed and the directory fsynced. An existing `record.json` is never overwritten;
6. appends to the index (if given) and returns the seal. If that append is refused (for example the index
   was broken after `open_run`), the record is already published and the writer is sealed; the refusal
   names the index, and `append_index` can be called again once the index is repaired.

A refused `seal()` before step 5 leaves no record and the writer open: fix the cause and call `seal()`
again.

Not checked by the ledger: whether the machine will live long enough to finish the run (the caller's
concern before opening a run). Concurrent writers on one run directory are not supported.

## `verify(record_path, expect_seal=None, check_code=True, allow_missing_inputs=False, repo=None)`

Runs every check and returns a report `{record, seal, ok, counts, checks: [...]}`; each check is `PASS`,
`FAIL` (with `code`, `reason`, `details`) or `NOT CHECKED` (with `reason`). If anything failed it raises one
`LEDGER_VERIFY_FAILED` whose details list the failed checks (`failed`) and the unchecked ones
(`not_checked`). `run_verification` returns the report without raising. The run directory is the record
file's directory.

| check | recomputes / compares | failure codes |
|---|---|---|
| `record` | the file parses as a strict JSON object (`apok_eval.jsonstrict`: no duplicate keys, no NaN/±Infinity, no float overflow, no lone surrogate, UTF-8, at most 256 levels; canonically serialisable). If this fails nothing else runs. | `LEDGER_RECORD_UNREADABLE` |
| `canonical` | the file's bytes equal `canonical_dumps(record) + "\n"`; no byte of the file is free | `LEDGER_RECORD_NOT_CANONICAL` |
| `seal` | `seal` equals the digest of the record without `seal` | `LEDGER_SEAL_MISMATCH` |
| `expected_seal` | with `expect_seal`: the *recomputed* seal equals it (NOT CHECKED without one) | `LEDGER_SEAL_UNEXPECTED` |
| `schema` | `schema` is `apok-record-v1`; full structure as specified above (member sets, types, formats, sort order, cross-references). If it fails, the content checks below are one NOT CHECKED entry `content`. | `LEDGER_SCHEMA_UNSUPPORTED`, `LEDGER_RECORD_MALFORMED` (details: `field`, `problems`) |
| `config_digest` | `digest_json(config)` | `LEDGER_CONFIG_DIGEST_MISMATCH` |
| `seeds` | every seed an int (not bool); `config` holds the same int at its pointer | `LEDGER_SEED_INVALID`, `LEDGER_SEED_CONFIG_MISMATCH` |
| `input:<name>`, `model:<name>` | a regular file exists and can be read, size, sha256. Missing (or not a regular file) → NOT CHECKED with `allow_missing_inputs`; unreadable (an `OSError` such as `EACCES`, `EIO`, `ENAMETOOLONG`, named with its errno) is never excused. Remote-only input → NOT CHECKED (verify never fetches). | `LEDGER_FILE_MISSING`, `LEDGER_FILE_UNREADABLE`, `LEDGER_FILE_SIZE_MISMATCH`, `LEDGER_FILE_DIGEST_MISMATCH` |
| `output:<name>` | file inside the run directory without symlinks, size, sha256 (never excused by `allow_missing_inputs`) | the four above, `LEDGER_PATH_UNSAFE` |
| `chunk_manifest:<name>` | sidecar exists and has `manifest_file_digest`; lines canonical, valid, strictly sorted, unique, newline-terminated; Merkle root; present/absent counts. This is the sidecar's own consistency and says nothing about the chunks themselves. | `LEDGER_FILE_MISSING`, `LEDGER_FILE_DIGEST_MISMATCH`, `LEDGER_MANIFEST_INVALID`, `LEDGER_MANIFEST_ROOT_MISMATCH`, `LEDGER_MANIFEST_COUNT_MISMATCH`, `LEDGER_PATH_UNSAFE` |
| `chunk_data:<name>` | when `source_uri` is a local `file://` URI (no host, or `localhost`): every key of the sidecar is read again through `apok_eval.io.store.LocalStore` at that directory; a present entry must have its digest and size, an absent one must still be absent; a key the store now refuses to read differs (details `count`, `store`, the first 20 `differing` keys with what was recorded and found). A store that is gone fails (NOT CHECKED with `allow_missing_inputs`), one that cannot be looked at fails. Any other `source_uri` → NOT CHECKED (verify never fetches). | `LEDGER_CHUNK_MISMATCH`, `LEDGER_FILE_MISSING`, `LEDGER_FILE_UNREADABLE` |
| `figures` | `values` equal the value at `pointer` in the source output's strict-JSON content, compared as canonical JSON (so `1`, `1.0` and `true` differ); details name the differing members. NOT CHECKED when `figures` is `null`. | `LEDGER_FIGURES_INVALID`, `LEDGER_FIGURES_MISMATCH` |
| `predictions` | `registry_digest` equals the recorded digest of `registry_input` (whose file the `input:` check verifies). NOT CHECKED when `null`. | `LEDGER_PREDICTIONS_MISMATCH` |
| `code.source_manifest_root` | Merkle root of `source_manifest` | `LEDGER_SOURCE_ROOT_MISMATCH` |
| `code.diff` | dirty records: `code.diff` exists (inside the run directory, no symlink) with `diff_digest`. NOT CHECKED otherwise. | file codes, `LEDGER_PATH_UNSAFE` |
| `code.commit` | with `check_code`: the source manifest equals the expected manifest of the recorded commit — for a dirty record, the commit's files with `code.diff` applied (see `code` above). Details: `missing_from_commit`, `missing_from_record`, `content_differs`, or `diff_does_not_apply` (git's message) when the recorded diff does not apply to the commit. A commit missing from a repository passed explicitly as `repo` / `--repo` fails (`LEDGER_COMMIT_MISSING`, details `commit`, `repo`), and so does an explicit repository git cannot open — a path that does not exist, a file, a directory that is not a git work tree (`LEDGER_REPO_INVALID`, details `repo`, `commit`, `error`): a malformed explicit argument never downgrades the check. NOT CHECKED, with the reason, when disabled, when there is no commit, when `code.diff` cannot be read or lacks `diff_digest` (the `code.diff` check fails then), when no repository is available, when the commit is not in the *default* repository (the work tree holding the running `apok_eval` package), or when git fails on the default repository. | `LEDGER_CODE_MISMATCH`, `LEDGER_COMMIT_MISSING`, `LEDGER_REPO_INVALID` |
| `environment` | always NOT CHECKED: **the environment cannot be recomputed**; it is covered by the seal only | — |

### What `verify` cannot establish

- **Re-sealed edits to leaves no recomputation covers.** These are protected only by the seal, i.e. only an
  anchored seal (`expect_seal`) detects them: `run_id`, `created_utc`, `purpose`, `code.package_version`,
  every `environment` value, every `name`, `uri` and `source_uri`, `models[].source`,
  `predictions.ids`, and the `digest`/`size` of a remote-only input (verify never fetches).
- **The chunks of a remote store.** A chunk manifest whose `source_uri` is not a local `file://` URI is
  checked for its own consistency only (`chunk_manifest:<name>`); its chunks' bytes are NOT CHECKED
  (`chunk_data:<name>`), so they are protected by the sidecar's digests and the seal, not recomputed. A
  local store's chunks are read again. The same list
  holds for a record sealed from a dirty tree: its `code.diff_digest`, `code.diff_file`, `code.source_root`
  and source manifest are recomputed (the diff file, and the commit with the diff applied).
- **`code.commit`, `code.source_root` and the source manifest without the history.** They are compared
  with git objects only where the commit is: with `--repo` naming a repository that lacks it, verify fails
  (`LEDGER_COMMIT_MISSING`); with the default repository lacking it — or none, or `--no-code-check` — they
  are NOT CHECKED, so a re-sealed `code.commit` changed to any well-formed id passes plain `verify` and only
  `expect_seal` catches it (tested). Run `verify` with `--repo` pointing at the run's history to close this.
- **A fully consistent rewrite** — e.g. drop a chunk from the sidecar and recompute its digest, the root,
  the counts and the seal — passes plain `verify`. Only `expect_seal` catches it (tested).
- **Truth of the inputs to the record.** `verify` proves the files are the ones recorded, not that the run
  actually read them, nor that a dirty tree's diff is complete for untracked files outside the package
  (untracked files inside it are in the source manifest).
- **The code of a run without a commit** (no checkout of the package: NOT CHECKED, with the reason), and
  untracked files of a dirty work tree outside the package directory (they make the tree dirty; they are in
  neither the diff nor the source manifest).

### Measured: the defining property

`tests/test_ledger_verify.py` builds a realistic synthetic run in a temporary git repository (three local
inputs, one remote-only input, a model, a chunk manifest with six present entries — metadata objects and
chunks — and two absent ones, two seeds, two outputs, figures, predictions), seals it twice — once from a
clean tree and once from a dirty one (a tracked file changed inside the package and outside it, an untracked
file and an untracked empty file inside the package; `allow_dirty`) — then:

- flips every byte of `record.json` in turn: every flip fails `verify`;
- walks every leaf of each record: changed with the seal left stale, every leaf fails with
  `LEDGER_SEAL_MISMATCH` (details: both seals); changed **and re-sealed**, each leaf either fails a named
  recomputation check with the expected code *and the expected cause in its details* (the field, file path,
  entry name, recorded and recomputed values) while the seal check passes, or is seal-only, in which case
  plain `verify` passes and `--expect-seal` with the original seal fails with exactly
  `LEDGER_SEAL_UNEXPECTED` (details: both seals). `code.commit` is changed to a well-formed id that no
  repository has, with the run's repository passed explicitly (`LEDGER_COMMIT_MISSING`); a change to another
  existing commit is tested separately (`LEDGER_CODE_MISMATCH`). Every leaf must be classified, or the test
  fails. The dirty record has a few more leaves than the clean one (two more manifest files, and the
  diff's digest and file name, which the `code.diff` check recomputes), and the environment's leaf count
  varies with the installed packages. Every digest, size, count, root and figure leaf is caught by a
  recomputation, except the remote-only input's digest and size;
- edits the source manifest consistently (an entry's digest and the recomputed root) and re-seals: the
  `code.commit` check names the file, for the clean and the dirty record; edits the recorded diff and its
  digest consistently: the patched result differs (`content_differs`) or does not apply
  (`diff_does_not_apply`);
- flips one byte in each input, the model, each output and the sidecar (`LEDGER_FILE_DIGEST_MISMATCH` on
  that file's check), and changes one figure in the figures output (`LEDGER_FILE_DIGEST_MISMATCH` and
  `LEDGER_FIGURES_MISMATCH` naming the member);
- deletes, reorders, adds, repeats sidecar lines, writes a non-canonical or blank line or an absent entry
  with a digest: raw, `LEDGER_FILE_DIGEST_MISMATCH`; with the sidecar digest updated and the record
  re-sealed, `LEDGER_MANIFEST_ROOT_MISMATCH` or `LEDGER_MANIFEST_INVALID`.

## The append-only index (`apok_eval.ledger.index`, conventionally `ledger/index.jsonl`)

One line per sealed record, the canonical JSON of
`{seq, run_id, seal, record_path, prev_entry_digest}` plus `\n`. `seq` is the 1-based line number;
`record_path` is the record's path relative to the index file's directory in its one normal spelling: POSIX,
`..` only as leading components (`../runs/fold-a/record.json`; `../ledger/../runs/…`, a trailing `..` or an
absolute path are malformed), no empty or `.` component, no backslash, NUL or invalid UTF-8. The *entry
digest* of a line is the sha256 of its bytes without the newline; `prev_entry_digest` is the previous
line's entry digest (`null` on line 1).

`verify_index(index, anchor=None, check_records=True)` / `apok-eval ledger verify-index INDEX [--anchor D]
[--no-record-check]` refuses:

- a torn last line, a line that is not a canonical JSON object, or a malformed entry → `LEDGER_INDEX_MALFORMED`
  (`line`, and `problems`: for a line `apok_eval.jsonstrict` refuses, `the line is not strict JSON: <its problem>`,
  e.g. `duplicate key 'seq'` or `arrays and objects nested deeper than 256 levels`; for a strict line whose bytes
  are not its canonical form, `the line is not canonical JSON`; otherwise the entry's structural problems);
- a `seq` that is not the line number (gaps, deleted or reordered lines) → `LEDGER_INDEX_SEQUENCE_BROKEN`;
- a `prev_entry_digest` that is not the previous line's digest (edited lines) → `LEDGER_INDEX_CHAIN_BROKEN`;
- a repeated `run_id` → `LEDGER_INDEX_DUPLICATE_RUN`;
- with `check_records`: a listed record that is missing, not a regular file (a FIFO or device is never
  read), or whose `run_id`, recorded seal or recomputed seal differs from the line →
  `LEDGER_INDEX_RECORD_MISMATCH`. Nothing links the *last* line; this check is what catches its edited
  `run_id` or `seal`, a `record_path` that no longer leads to that record, or a forged appended line. It
  cannot catch the last line's `record_path` changed to *another file holding the same record bytes* (a
  copy): `run_id` and seal still agree. Only an anchor on the head digest detects that (tested);
- with `anchor`: no line has that entry digest → `LEDGER_INDEX_ANCHOR_MISSING`.

The chain alone cannot detect a consistent rewrite of the whole file or the removal of trailing lines.
Anchor an entry digest outside the file (commit the index to git; note the head digest elsewhere); passing it
as `anchor` pins that line and, through the chain, everything before it. Lines appended after an anchor are
legitimate appends as far as the anchor is concerned; the chain and the record check cover them.

Appends (`append_index`, called by `seal()`) take an exclusive `flock` on the index (which must be a regular
file: an index symlinked to `/dev/null` would swallow the append), re-verify the existing chain (without the
record check), refuse a run_id already present and a malformed entry, and fsync. `record_path` is computed
between the resolved (symlink-free) locations of the record and the index directory, which is how the
operating system follows it when `verify_index` opens it — so an index in a symlinked directory verifies. An
index kept in the repository makes the work tree dirty when appended to, so it would be committed (which also
anchors it) before the next run is opened, or the next `open_run` would need `allow_dirty=True`.
`apok-eval e7v3` opens its runs with `index=None`.

## CLI

```
apok-eval verify RECORD [--expect-seal SEAL] [--allow-missing-inputs] [--no-code-check] [--repo PATH] [--json]
apok-eval ledger verify RECORD ...   (the same command)
apok-eval ledger verify-index INDEX [--anchor ENTRY_DIGEST] [--no-record-check]
apok-eval ledger show RECORD
```

`verify` prints one line per check (or, with `--json`, the JSON report) and exits 0; on any
failure the per-check lines (with `--json`, the report) go to stderr and stdout holds only the refusal JSON
(`LEDGER_VERIFY_FAILED`, failed checks in `details.failed`), as for every `apok-eval` command, and it exits 3 —
also for a record that cannot be read (a FIFO, a device, JSON nested too deeply), never a traceback.
`--repo` names the repository holding the run's history; a commit missing from it fails.
`verify-index` prints a JSON summary (`entries`, `head`, `anchor`, `records_checked`). `show` prints the
record as indented JSON and notes on stderr that it verifies nothing.

## Refusal codes

Every refusal names its cause in `reason` and `details` (the field, file, entry, line or run_id).

| code | raised when |
|---|---|
| `LEDGER_RUN_ID_INVALID` | `open_run` with a run_id not matching the name pattern (details: `run_id`) |
| `LEDGER_NAME_INVALID` | an input/model/output/chunk-manifest/seed name not matching the pattern (`section`, `name`) |
| `LEDGER_NAME_DUPLICATE` | a name already used in that section (`section`, `name`) |
| `LEDGER_ARGUMENT_INVALID` | empty `purpose`; non-canonical, non-object or too deeply nested (over 256 levels) `config`; bad `expected_bytes`; `source_dir` not a directory; empty `uri`/`source`/`source_uri`; remote data not bytes; an input/model path that is not UTF-8 or holds NUL; a `run_dir`, `index` or `source_dir` that is not a str/path-like naming UTF-8 text without NUL, refused before anything is created (`field`) |
| `LEDGER_RUN_DIR_NOT_EMPTY` | `run_dir` exists, is not empty and `resume` was not given (`run_dir`) |
| `LEDGER_ALREADY_SEALED` | `record.json` exists at open (even with `resume`); any method on a sealed writer; `record.json` appeared before the seal was published |
| `LEDGER_UNWRITABLE` | creating or probe-writing the run directory or the index directory failed (`path`) |
| `LEDGER_DISK_INSUFFICIENT` | free bytes < `expected_bytes` (`path`, `free`, `expected_bytes`) |
| `LEDGER_GIT_FAILED` | git cannot be run, or a git command fails; a `.git` entry git cannot open (moved or pruned worktree); a tracked package with no commit yet; the dirty tree's diff does not apply to its commit (`source_dir`) |
| `LEDGER_SOURCE_UNSUPPORTED` | a symlink, non-regular or unreadable file, or a name that is not a safe relative path in the package directory (`path`); a code identity `apok-record-v1` cannot hold, e.g. a 64-hex SHA-256 commit (`field`, `commit`, `problems`) — all at `open_run`, before compute |
| `LEDGER_DIRTY_TREE` | the work tree is dirty and `allow_dirty` was not given (`changed`: status lines) |
| `LEDGER_CODE_CHANGED` | the commit or source manifest at `seal()` differs from `open_run` (`opened`, `now`) |
| `LEDGER_WRITE_FAILED` | writing a sidecar, `record.json` or an index line failed, or the index is not a regular file (`path`) |
| `LEDGER_NO_OUTPUTS` | `seal()` with no outputs |
| `LEDGER_PREDICTIONS_INVALID` | predictions set twice, registry not a recorded input, ids empty/duplicated/not non-empty strings (`field`) |
| `LEDGER_FILE_MISSING` | no regular file at an input/model/output/sidecar/diff path — nothing there, a path through a file, a directory, FIFO or device — in the writer or in `verify` (`what`, `path`) |
| `LEDGER_FILE_UNREADABLE` | a recorded path that cannot be opened or read (`EACCES`, `EIO`, `ENAMETOOLONG`, …), in the writer or in `verify`; never excused by `allow_missing_inputs` (`what`, `path`, `errno`) |
| `LEDGER_PATH_UNSAFE` | an output path outside the run directory, through a symlink, or reserved; a sidecar or figures source the writer would reach through a symlink (a symlinked `manifests/` in a resumed run directory); in `verify`, a run-relative path — output, sidecar, figures source, `code.diff` — through a symlink (`path`) |
| `LEDGER_SEED_INVALID` | a seed that is not an int, or is a bool (`seed`) |
| `LEDGER_SEED_CONFIG_MISMATCH` | the seed's pointer does not resolve in `config`, or `config` holds something other than the same int there (`seed`, `pointer`) |
| `LEDGER_MANIFEST_INVALID` | malformed/duplicated chunk-manifest entries (writer); a sidecar line that is not canonical, valid, sorted, unique or newline-terminated (`manifest`, `problem` naming the entry or line) |
| `LEDGER_FIGURES_INVALID` | figures set twice; source not a recorded output; output not a regular file or not strict JSON (including a lone surrogate, which canonical JSON cannot hold); pointer unresolvable; value not a non-empty object (`source`, `pointer`) |
| `LEDGER_CODE_MISMATCH` | the source manifest differs from the expected manifest (the commit's files, with the diff applied for a dirty tree): at open, or in `verify` (`commit` and the differing paths); in `verify` also a recorded diff that does not apply to the commit (`diff_does_not_apply`) |
| `LEDGER_RECORD_UNREADABLE` | the record file is missing, not a regular file, or not a strict JSON object nested at most 256 levels (`record`, `problem`: the parser's problem, `the top level is a list, not an object`, or the read failure's type and message) |
| `LEDGER_RECORD_NOT_CANONICAL` | the record's bytes are not its canonical JSON plus `\n` |
| `LEDGER_SCHEMA_UNSUPPORTED` | `schema` is not `apok-record-v1` (`schema`) |
| `LEDGER_RECORD_MALFORMED` | any structural violation (`field`: the first; `problems`: all, each prefixed by its member path) |
| `LEDGER_SEAL_MISMATCH` | `seal` is not the digest of the content (`recorded`, `recomputed`) |
| `LEDGER_SEAL_UNEXPECTED` | the recomputed seal differs from `expect_seal` (`expected`, `recomputed`) |
| `LEDGER_CONFIG_DIGEST_MISMATCH` | `config_digest` differs from the digest of `config` (`field`, `recorded`, `recomputed`) |
| `LEDGER_FILE_SIZE_MISMATCH` | a file's byte count differs from its recorded size (`what`, `path`, `recorded`, `actual`) |
| `LEDGER_FILE_DIGEST_MISMATCH` | a file's sha256 differs from its recorded digest (`what`, `path`, `recorded`, `actual`) |
| `LEDGER_MANIFEST_ROOT_MISMATCH` | the sidecar's Merkle root differs from `root` (`manifest`) |
| `LEDGER_MANIFEST_COUNT_MISMATCH` | the sidecar's present/absent counts differ (`manifest`, `recorded`, `recomputed`) |
| `LEDGER_CHUNK_MISMATCH` | a chunk of a local `file://` store read again differs from its recorded digest or size, appeared where the run found none, is gone, or is refused by the store (`manifest`, `store`, `count`, `differing`) |
| `LEDGER_FIGURES_MISMATCH` | `figures.values` differ from the output's content (`source`, `differing` members) |
| `LEDGER_PREDICTIONS_MISMATCH` | `registry_digest` differs from the registry input's digest (`registry_input`) |
| `LEDGER_SOURCE_ROOT_MISMATCH` | `source_manifest_root` is not the root of `source_manifest` (`field`, `recorded`, `recomputed`) |
| `LEDGER_COMMIT_MISSING` | the recorded commit is not in the repository passed explicitly as `repo` / `--repo` (`commit`, `repo`) |
| `LEDGER_REPO_INVALID` | git cannot open the repository passed explicitly as `repo` / `--repo` (`repo`, `commit`, `error`) |
| `LEDGER_VERIFY_FAILED` | `verify` found failures, or `seal()`'s pre-flight did (`failed`, `not_checked`) |
| `LEDGER_INDEX_UNREADABLE` | the index file cannot be read, is not a regular file, or its path holds NUL or an unencodable surrogate (`index`) |
| `LEDGER_INDEX_MALFORMED` | a torn, non-strict-JSON, non-canonical or malformed index line, or an attempt to append a malformed entry (`line`, `problems`, the first also in the reason; a strict-JSON problem is passed through) |
| `LEDGER_INDEX_SEQUENCE_BROKEN` | a line's `seq` is not its line number (`line`, `seq`) |
| `LEDGER_INDEX_CHAIN_BROKEN` | a line's `prev_entry_digest` is not the previous line's digest (`line`) |
| `LEDGER_INDEX_DUPLICATE_RUN` | a run_id appears twice in the index, or a new run's id is already listed (`run_id`) |
| `LEDGER_INDEX_RECORD_MISMATCH` | an indexed record is missing or disagrees with its line (`line`, `problem`) |
| `LEDGER_INDEX_ANCHOR_MISSING` | no line has the anchored entry digest (`anchor`) |

## Assumptions declared

- Git is the version-control system and uses SHA-1 object names (`commit` is 40 hex). A SHA-256 git
  repository is refused at `open_run` (`LEDGER_SOURCE_UNSUPPORTED`, field `code.commit`), before any compute.
- A package directory on a different file system than its work-tree top is refused (`LEDGER_GIT_FAILED`:
  git stops discovery at the boundary unless `GIT_DISCOVERY_ACROSS_FILESYSTEM` is set, and a `.git` entry
  above that git cannot reach is never read as "no checkout").
- `dirty` covers the whole work tree, not only the package; the diff covers tracked files only.
- File digests follow symlinks for inputs and models (recorded as such above); outputs, sidecars and the
  diff may not be reached through symlinks.
- `created_utc` is wall-clock time at `open_run` and is the only non-derived time in a record.

## How apok-eval uses it

`apok-eval e7v3` (and each scenario of `apok-eval demo`) opens one run with `index=None` and records:

- `config`: every setting in force (the rule's defaults included), the absolute paths of the inputs, and the
  licence gate's clearance, or `{"checked": false}` without a registry;
- `seeds`: the nulls' (`/settings/registration/seed`), the split's (`/settings/face/split_seed`), the
  ceiling's (`/settings/face/ceiling_seed`) and the CT sampling's (`/settings/ct/seed`);
- `inputs`: the umbilicus file, read again when the record is written; its digest and size must be those of
  the bytes the run parsed, which `e7v3.json` keeps under `/umbilicus`, else the run is refused
  (`EVAL_RUN_INPUT_CHANGED`) and not sealed;
- `chunk_manifests`: `segments` (every file the tifxyz reader read, `<segment>/<file>`), `fibres` (every
  fibre file read), and with a CT volume `volume` (its `.zattrs`, the level's `.zarray` and every chunk read),
  with the fit's input patches `patches`. Their `source_uri` is the local `file://` directory, so `verify`
  reads every one of those files again;
- `outputs`: `e7v3.json`, the whole result; `figures`: its `/figures` member (the verdicts, Q, the nulls'
  95th percentiles, the family offsets, the ceiling and the CT locking).

Anchor a seal you intend to quote: commit the record, or note the seal somewhere else, and check it later
with `apok-eval verify RECORD --expect-seal SEAL`.

## Open questions

- Should a remote-only input be refused unless covered by a chunk manifest, so that every recorded digest
  can be recomputed from local bytes? Today it is allowed (the schema has `path` null for remote-only
  inputs) and reported NOT CHECKED; its `digest` and `size` are therefore seal-only, an exception to the
  defining property (recomputing them would mean fetching, behind a Clearance, which `verify` never does).
- Should input paths be recorded relative to a declared data root, so that a record can be published without
  naming the directories it was made in?
- Should `verify` fail, rather than report NOT CHECKED, when the *default* repository lacks the recorded
  commit? Today only an explicitly named repository turns a missing commit into a failure.
- Should the ledger require an anchor (e.g. refuse to open a run while the index has uncommitted lines)?
  Today anchoring is the caller's practice.
- Should `verify` re-check the predictions registry's *content* (ids present in the registry)? The registry
  format is not defined yet; only its digest is bound.
