# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-12

First public release.

### Added

- `apok-eval e7v3`: held-out fibre agreement (E7 v3) between a spiral fit's tifxyz segments and VC3D fibre
  traces in one slab of z. Per-family (H, V) offsets are calibrated on one half of the fibres, split by a hash
  of their file names, and the other half is scored against z-offset and rotation nulls run through the same
  calibration. Verdicts are PASS, FAIL, INCONCLUSIVE or NOT MEASURED, overall and per family.
- A CT positive control on a local OME-Zarr v2 volume (`--volume`): when it runs, the fit is scored against the
  fibres only if the control is CONFIRMED.
- The ceiling: the score a perfect front-face fit would reach with the same fibres, reported beside each run.
- An optional geometric leakage audit (`--patches`) that scores E7 v3 again on the fibre points away from the
  fit's input patches.
- `apok-eval demo`: three scenarios on a synthetic scroll, each sealed and verified, with no data needed.
- Sealed run records (`apok_eval.ledger`): the digest of every byte read, with a Merkle root over every chunk,
  and the settings, seeds, code identity and environment. `apok-eval verify` recomputes them, `--expect-seal`
  checks an anchored seal, and `apok-eval ledger verify-index` checks a hash-chained, append-only index.
- The licence gate (`apok_eval.licence`, `apok-eval licence-check`, `e7v3 --registry` and `--terms`): a shipped
  registry of the Vesuvius Challenge licences, terms and hosts (and no scroll data), per-person acceptance of
  terms, URI canonicalisation, hard-wired gating of the Vesuvius Challenge buckets and hosts, and a release
  mode.
- Readers (`apok_eval.io`) for tifxyz under five validity policies, OME-Zarr v2, umbilicus files in four
  formats, OBJ and PLY meshes and VC3D point collections, with local stores and HTTP stores that read only
  under a licence clearance. `apok-eval inspect` identifies and summarises a local tifxyz surface, OME-Zarr
  volume, mesh, point-collection file or JSON umbilicus.
- `apok_eval.jsonstrict`: one strict JSON parser for every document apok-eval reads, and a deterministic
  writer.
- Stable refusal codes (`EVAL_*`, `IO_*`, `LEDGER_*`, `LICENCE_*`): a refusal names its cause, exits 3 and is
  printed as JSON.
- Reference documentation for the method, the record format, the licence gate and the readers (`docs/`).
- A fully synthetic test suite behind a network guard that blocks anything but loopback, and two release
  guards: one checks every tracked file, staged file and commit for numbers on a private list of withheld
  PHerc. Paris 4 figures (`release/README.md` says what it cannot see), the other keeps links to private
  workspaces out of tracked files and commit messages.
- Continuous integration across the supported Python versions, a contributing guide, a security policy, a code
  of conduct, issue and pull request templates, and citation metadata (`CITATION.cff`).

[Unreleased]: https://github.com/LaBlueberryPie/apok-eval/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/LaBlueberryPie/apok-eval/releases/tag/v0.1.0
