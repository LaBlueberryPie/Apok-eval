# apok-eval

[![CI](https://github.com/LaBlueberryPie/apok-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/LaBlueberryPie/apok-eval/actions/workflows/ci.yml)
[![Licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/python/required-version-toml?tomlFilePath=https://raw.githubusercontent.com/LaBlueberryPie/apok-eval/main/pyproject.toml&label=python)](pyproject.toml)

Verification-first evaluation of spiral fits for the [Vesuvius Challenge](https://scrollprize.org).

apok-eval scores a spiral fit (a set of winding surfaces in tifxyz format, each a grid of x, y and z coordinates
stored as TIFF images) against human fibre traces the fit did not use, after checking against the CT that those
fibres really sit on papyrus. Every run is sealed in a record that anyone can re-check byte for byte, and with the
licence gate on, nothing is read until every input is declared with a licence and the terms it requires are
covered by your own acceptance. The package ships the evaluator, the readers it needs, the run ledger and the
licence gate, and nothing else.

**This repository contains no scroll data and no results on scroll data.** The demo and every test run on
synthetic scrolls generated on the fly.

## Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Usage](#usage)
- [How it works](#how-it-works)
- [Verifiable records](#verifiable-records)
- [Licence gate](#licence-gate)
- [Limits](#limits)
- [Independence and data terms](#independence-and-data-terms)
- [Related work](#related-work)
- [Documentation and layout](#documentation-and-layout)
- [Contributing, security and conduct](#contributing-security-and-conduct)
- [Citing](#citing)
- [Licence](#licence)

## Installation

Python 3.11 or later.

From GitHub:

```sh
pip install "apok-eval @ git+https://github.com/LaBlueberryPie/apok-eval.git"
```

To pin a release, append its tag to the URL (for example `apok-eval.git@v0.1.0`).

From a clone, to work on the code:

```sh
git clone https://github.com/LaBlueberryPie/apok-eval.git && cd apok-eval
python -m venv .venv && . .venv/bin/activate
pip install -e ".[test]"
```

Either way you get the `apok-eval` command and the `apok_eval` package, with numpy, scipy, tifffile and
numcodecs. Two optional extras add more:

| extra | installs | for |
|---|---|---|
| `lzw` | imagecodecs | LZW-compressed tifxyz, such as the `mask.tif` written by VC3D ([villa](https://github.com/ScrollPrize/villa)'s Volume Cartographer); without it such files are refused (`IO_TIFXYZ_CODEC_MISSING`) |
| `test` | pytest, zarr < 3 | the test suite; zarr is the reference reader the OME-Zarr tests compare against |

Name them in square brackets: `pip install "apok-eval[lzw] @ git+https://github.com/LaBlueberryPie/apok-eval.git"`,
or `pip install -e ".[lzw,test]"` in a clone.

Once installed, apok-eval works offline: every command reads local files, and the readers' remote access is
open only through the licence gate. The dependencies' licences are listed under [Licence](#licence).

## Quickstart

```sh
apok-eval demo
```

The demo writes a synthetic scroll in a temporary directory: a breathing spiral (its sheets move in and out
with z), a layered CT as an OME-Zarr volume, a fit's tifxyz segments, an umbilicus, and 60 fibre traces as
VC3D JSON, H fibres on the sheets' front faces and V fibres 4 voxels behind them. It declares the directory in
a run-local licence registry and runs `e7v3` three times, sealing and verifying each run:

| scenario | what is wrong | control | E7 v3 |
|---|---|---|---|
| `front-face-fit` | nothing: the fit follows the sheets | CONFIRMED | PASS (all, H, V) |
| `fit-without-breathing` | the fit ignores how the sheets move with z | CONFIRMED | FAIL (all, H, V) |
| `fibres-off-papyrus` | each fibre sits at its own place between the CT's layers | REFUTED | NOT MEASURED |

It takes under a minute, uses the evaluator's default settings throughout, and exits 0 when every verdict is
the expected one and every record verifies, ending with:

```
demo passed: every verdict as expected and every record verifies
```

`apok-eval demo --keep DIR` keeps the synthetic scroll and the records, so you can run the commands below on
them; `--json` prints the whole report.

## Usage

### Scoring a fit

```sh
apok-eval e7v3 --segments path/to/fit/segments --fibres path/to/fibres \
  --umbilicus path/to/umbilicus.json --z0 3000 --z1 4000 --frame-shape Z Y X
```

| option | meaning |
|---|---|
| `--segments DIR` | the fit: one tifxyz directory per segment, named `...-wLLL-HHH` |
| `--fibres DIR` | VC3D fibre traces, one `.json` file each |
| `--umbilicus FILE` | the spiral fitter's umbilicus (`--umbilicus-variant` for the other formats) |
| `--z0`, `--z1` | the slab, in frame voxels (`--z0` inclusive, `--z1` exclusive) |
| `--frame-shape Z Y X` | the spiral frame's shape (z, y, x), when no `--volume` is given |
| `--volume ZARR`, `--level N` | the CT, for the positive control |
| `--patches DIR` | the fit's input patches, for the leakage audit |
| `--settings FILE` | overrides of the rule's defaults |
| `--registry FILE`, `--terms FILE` | the licence gate (see [Licence gate](#licence-gate)) |
| `--out DIR` | a new directory for the record (default `apok-eval-runs/<run id>`) |
| `--no-record` | print the verdicts only and seal nothing |
| `--json` | print the figures, the seal and the record's path as JSON |

`apok-eval e7v3 --help` lists every option. The inputs in more detail:

- **Segments** are tifxyz directories (`x.tif`, `y.tif`, `z.tif`, `meta.json`, optionally `mask.tif`), read
  under the `spiral` validity policy by default (`--policy` for `vc3d`, `vesuvius`, `lasagna` or
  `apok-strict`). Each name must end in its winding range, `-wLLL-HHH` (first and last winding, three digits
  each), as the spiral fitter writes them; for segments named otherwise, give `--windings map.json` with
  `{"segment-directory": [first, last], ...}`. Every segment must agree on the spiral's sense.
- **Fibres** are VC3D `vc3d_fiber` documents (versions 1, 3 and 4), one per `.json` file. Their coordinates are
  scaled into the frame by `coordinate_base_shape_zyx` (or by 0.25 when a fibre does not declare it), as villa
  does, and must then lie in the frame. A file that cannot be read is counted by its reason and left out. A
  fibre's family is its `hv_classification` tag when it has one, else its shape.
- **The frame** is the grid every coordinate is in: usually one level of the CT. Declare its shape with
  `--frame-shape`, or give the CT with `--volume ct.zarr --level N` and the frame is that level's shape.
- **The CT positive control** runs when `--volume` is given (a local OME-Zarr v2 volume; `|u1` or `<u2`;
  uncompressed or Blosc, zstd, lz4, zlib, gzip). Without it the control is NOT RUN, and the output says so.
- **Settings**: the rule's defaults are in [docs/method.md](docs/method.md#defaults); a `--settings` file
  overrides any of them, and the record keeps every value in force.

The output names the control's verdict, then each group's E7 v3 verdict with its Q and the null's 95th
percentile, the families' offsets in voxels, the ceiling, and the record's seal. On the demo's synthetic
scroll (your numbers will differ):

```
apok-eval e7v3: slab z [96, 800), frame [896, 448, 448]
  CT positive control: CONFIRMED
    fibres R 0.710, fit R 0.998, null R 0.015, layer period 16 voxels
  fibres: 60 read of 60 files
  E7 v3 all  PASS  Q 1.000 vs z-offset null p95 0.360
  ...
```

### Other commands

```sh
apok-eval verify RECORD.json       # re-check a sealed record: digests, re-read inputs, figures, seal
apok-eval inspect PATH             # what is this file or directory? tifxyz, OME-Zarr, umbilicus, mesh, PCL
apok-eval licence-check RUN.json   # the licence gate on its own
apok-eval --help
```

Every guard refuses with a stable code (`EVAL_*`, `IO_*`, `LEDGER_*`, `LICENCE_*`) and names its cause; a
refusal exits 3 with the refusal printed as JSON.

## How it works

Given a fit's segments, an umbilicus, a folder of VC3D fibre traces, a slab of z and, optionally, the CT, a run
goes through five steps:

1. **CT positive control** (with the CT): are the fibres on the scan's papyrus layers? Unless the control
   confirms it, the fit is not scored against them (NOT MEASURED).
2. **E7 v3**: held-out fibre agreement, overall and per fibre family. Verdicts: PASS, FAIL, INCONCLUSIVE or
   NOT MEASURED.
3. **Ceiling**: the score a perfect fit could reach with these fibres, so a modest score is read in context.
4. **Leakage audit** (optional): the score again on the fibre points that are not on the fit's own input
   patches.
5. **A sealed record** of everything read and every setting, which `apok-eval verify` re-checks.

[docs/method.md](docs/method.md) states every rule and default; in short:

- **Where a fibre point sits.** From the umbilicus, draw the ray through the point. The fit's sheets cross it at
  regular steps; the point's *gap position* is where it falls between the two nearest sheets: 0 on a sheet, ±½
  half way to the next.
- **Families.** Villa's documentation says horizontal fibres lie on a sheet's front face and vertical fibres on
  its back face, a few voxels apart, and that a fit can only follow one face. So a good fit does not put every
  fibre on its sheet: it puts each *family* at one steady place in its gaps. apok-eval tags each fibre H, V or
  untagged by villa's own rule.
- **Held-out offsets.** The fibres are split in two by a hash of their file names (the data cannot influence
  the split). On the calibration half, each family's *offset* is the place in the gap where most of its points
  sit. On the evaluation half, each fibre scores the fraction of its points within an eighth of a gap of its
  family's offset; Q is the mean over fibres. Points spread evenly through the gaps score about 0.25.
- **Nulls.** Move every fibre up or down by a few hundred voxels, or turn it about the umbilicus, and run the
  whole procedure again, offsets included, a hundred times each. A fit that really follows the sheets scores
  its fibres better than every moved copy.
- **Verdict.** PASS when Q is at least 0.40 and above every draw of the null; FAIL when Q is under 0.30 or not
  above the null's 95th percentile; INCONCLUSIVE in between; NOT MEASURED with fewer than 10 scored fibres.
  Overall, and separately for H and for V.
- **CT positive control.** Fibres that land nowhere in particular could mean a bad fit or fibres in the wrong
  frame. The CT tells them apart: things on papyrus share a phase in the scan's layer pattern. The control is
  CONFIRMED when the fibres are locked to the layers and random positions are not; when the control runs, E7 v3
  is scored only if it is CONFIRMED.
- **Ceiling.** The Q a perfect front-face fit would reach with these fibres if they carry 1, 2 or 3 voxels of
  tracing noise, simulated on the run's own points. It is reported so that a modest Q can be read against what
  is reachable; it is never a threshold.
- **Leakage audit.** With `--patches DIR`, every fibre point near one of the fit's input patches is set aside
  and E7 v3 is scored again on the rest, since a split by file name alone cannot tell that a point lies on a
  surface the fit has already seen.

## Verifiable records

Every `e7v3` run, and each demo scenario, writes a run directory with `record.json` and `e7v3.json`. The record
holds the digest and size of every byte read (every segment file, every fibre file, every CT chunk, the
umbilicus), the full settings and seeds, the code's identity (commit, or a digest of every source file, and
the diff of a dirty tree), the environment, and the output's digest; the figures it quotes are bound to the
output; a seal covers all of it. A run whose umbilicus file changes between the run's read and the record is
refused, not sealed.

`apok-eval verify RECORD.json` recomputes everything that can be recomputed: it re-reads every local input and
chunk and checks its digest, checks the figures against the output, checks the source against the recorded
commit, and recomputes the seal. A seal alone proves consistency, not that nobody re-sealed an edited record,
so anchor a seal you quote (commit it, or post it) and check against it with `--expect-seal`.
[docs/ledger.md](docs/ledger.md) gives the format and every check, and what `verify` cannot establish.

A record names the local paths it read and, with the licence gate on, your acceptance of the data terms: read
it before you share it.

## Licence gate

Scroll data comes under licences and terms that bind each person who uses it. With `--registry`, `e7v3`
refuses to read anything until every input is declared as an asset with a licence, on a host or local root
that allows that licence, and the terms it requires are covered by your own acceptance (`--terms`). The
shipped registry declares the Vesuvius Challenge licences, terms and hosts, but **no scroll data**: you
declare your own data, and you accept the terms yourself, as
[docs/licence.md, "Declaring your data"](docs/licence.md#declaring-your-data) walks through. Without
`--registry` nothing is checked, and the record says so. The gate never decides whether something derived
from scroll data may be published: that is for the data's terms and for you.

## Limits

- **One slab at a time**, and the verdict holds for that slab only.
- **The fibres must be held out from the fit.** E7 v3 cannot tell that a fit was built from the very fibres it
  is scored on; the split protects the offsets, not the fit. Use fibres the fit did not read, and the leakage
  audit when you have the fit's input patches.
- **Winding numbers are not checked.** A fit can PASS with every family at a steady offset and still label a
  sheet with the wrong winding.
- **The z-offset null assumes the sheets move with z** over a few hundred voxels. Where they do not, it is
  uninformative and the more lenient rotation null decides; the output says which null was used.
- **The CT control is optional**, and without it a FAIL cannot be told from fibres in the wrong frame.
- **Local inputs only** for `e7v3`. Segments must be tifxyz with winding-range names (or a windings map).
- **Run cost** grows with the number of fibres and of null draws: on the demo's 60 fibres, one run with the
  default 100 draws per null and the CT control takes about 15 seconds on one machine. The CT control holds the
  chunks its rays need in memory, within a budget (`ct.max_chunks`, `ct.memory_bytes`).

## Independence and data terms

apok-eval is an independent project by Static Signal Ltd. It is not affiliated with or endorsed by the Vesuvius
Challenge or Scroll Prize, Inc. The licence gate records how its maintainers read the data terms; it is not legal
advice, and you remain responsible for the terms of any data you read. PHerc. Paris 4 data in particular carries
a publication reservation (no scholarly claims, transcriptions or formal analyses before 25 June 2027 without
the organisers' permission): check the current terms on scrollprize.org before publishing any result.

## Related work

[spiralcheck](https://github.com/Nicodol/spiralcheck) (MIT) is a separate, independent evaluation tool for
spiral fits. What follows is limited to what this repository documents:

- **What apok-eval takes from it**: the idea of a *geometric leakage audit*, checking whether held-out points lie
  within touching distance of a fit's own inputs, since a split by name alone can leave such points.
  apok-eval's audit (`apok_eval.eval.leakage`) is its own implementation of that idea, for fibre points against
  tifxyz patches, and its reports credit spiralcheck.
- **What apok-eval adds around it**: per-family offsets learnt on a held-out half, z-offset and rotation nulls
  run through the same calibration, a CT positive control that gates scoring, a ceiling, a sealed and
  verifiable record of every byte read, and a licence gate.
- **What this repository does not claim**: anything else about spiralcheck's metrics, results, coverage or
  correctness, for which its own documentation is the reference. The two tools can be run on the same fit;
  their numbers answer different questions and are not interchangeable.

## Documentation and layout

The reference documentation is in [docs/](docs/README.md): the method and its defaults, the record format and
`verify`, the licence gate, the readers, and the strict JSON parser.

```
src/apok_eval/eval/     the evaluator: fit, fibres, registration, faces (E7 v3), ctprofile, leakage,
                        local runs (local.py), the demo and its synthetic world, the command line
src/apok_eval/io/       readers: tifxyz, OME-Zarr v2, umbilicus, meshes, point collections, local and http stores
src/apok_eval/ledger/   sealed run records and verify
src/apok_eval/licence/  the asset registry and the licence gate (registry.json is the shipped registry)
docs/                   the reference documentation (index: docs/README.md)
release/                the release guard's normaliser and how the guard works (release/README.md)
tests/                  synthetic tests; no test reads scroll data or the network
```

Run the tests with `python -m pytest` in a clone installed with the `test` extra. Every test is synthetic, and
a network guard blocks anything but loopback.

## Contributing, security and conduct

- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md) covers the development set-up, the tests, the release
  guards and the coding style. Bugs and feature requests go to the
  [issue tracker](https://github.com/LaBlueberryPie/apok-eval/issues).
- **Security**: please report vulnerabilities privately, as [SECURITY.md](SECURITY.md) describes, not in a
  public issue.
- **Conduct**: everyone taking part is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
- **Changes**: each release is listed in [CHANGELOG.md](CHANGELOG.md).

## Citing

If you use apok-eval in your work, please cite it. [CITATION.cff](CITATION.cff) holds the citation metadata;
GitHub's "Cite this repository" button turns it into APA or BibTeX.

## Licence

apok-eval is released under the MIT licence ([LICENSE](LICENSE)). The data it evaluates is not: see
[docs/licence.md](docs/licence.md).

**Credits.** [villa](https://github.com/ScrollPrize/villa) (ScrollPrize, MIT), whose formats and conventions the
readers follow and from which the fibre parsing and H/V rule are ported;
[spiralcheck](https://github.com/Nicodol/spiralcheck) (MIT), for the leakage audit's idea and the form of its
point-to-triangle routine. Their notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

**Dependencies.** None is vendored: each is installed by pip from its own project, under its own licence.

| package | needed for | licence |
|---|---|---|
| numpy | everything | BSD-3-Clause, with parts under 0BSD, MIT, Zlib and CC0-1.0 |
| scipy | the evaluator's k-d trees (exact distances, the leakage audit) | BSD-3-Clause |
| tifffile | tifxyz | BSD-3-Clause |
| numcodecs | compressed OME-Zarr chunks | MIT |
| imagecodecs (optional, `[lzw]`) | LZW-compressed tifxyz | BSD-3-Clause |
| pytest (tests only, `[test]`) | the test suite | MIT |
| zarr < 3 (tests only, `[test]`) | the reference reader the OME-Zarr tests compare against | MIT |

numpy's and scipy's binary wheels also bundle compiled libraries (such as OpenBLAS, LAPACK and the GCC runtime)
under their own licences, which each wheel's metadata lists.
