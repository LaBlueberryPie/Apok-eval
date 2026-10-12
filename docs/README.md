# apok-eval documentation

The reference documentation for apok-eval. For installation, the demo and an overview, start with the
[project README](../README.md). The pages below state the rules and formats in full, with every `IO_`, `LEDGER_`
and `LICENCE_` refusal code; the `EVAL_` codes are registered, with their reasons, in the evaluator's modules.

## The evaluator and its records

- [method.md](method.md): the E7 v3 method, the CT positive control, the ceiling, the leakage audit and the defaults.
- [ledger.md](ledger.md): sealed run records, what `verify` checks and cannot establish, and the append-only index.
- [licence.md](licence.md): the asset registry and the licence gate, including how to declare your own data.

## Readers and formats

- [io-surfaces.md](io-surfaces.md): the tifxyz, triangle-mesh, point-collection and umbilicus readers.
- [io-zarr.md](io-zarr.md): stores, the read log and the OME-Zarr v2 reader behind the CT positive control.
- [jsonstrict.md](jsonstrict.md): the one strict parser for every JSON document apok-eval reads, and its deterministic writer.

## Elsewhere in the repository

- [release/README.md](../release/README.md): the release guard and the reading pass that goes with it.
- [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md): the notices of the projects apok-eval ports from or follows.
