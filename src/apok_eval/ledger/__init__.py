"""apok_eval.ledger - sealed run records and ``verify``.

One sealed JSON record per run (schema ``apok-record-v1``): input digests,
including a Merkle root over every chunk read; the code commit and a digest
manifest of the package source; the full configuration and its seeds; model
digests; the environment; output digests; the figures the run quotes, backed
by an output; the pre-registered predictions in force; and a seal over all
of it. ``verify`` recomputes everything that can be recomputed and says
which parts cannot. An append-only, hash-chained index lists sealed records.

Modules: :mod:`~apok_eval.ledger.writer` (``open_run``), :mod:`~apok_eval.ledger.verify`,
:mod:`~apok_eval.ledger.index`, :mod:`~apok_eval.ledger.record` (schema and seal),
:mod:`~apok_eval.ledger.manifest` (chunk-manifest sidecars),
:mod:`~apok_eval.ledger.gitcode` (code identity), :mod:`~apok_eval.ledger.environment`.
``docs/ledger.md`` is the specification.
"""

from apok_eval.ledger import codes as _codes  # noqa: F401  (registers the LEDGER_ refusal codes first)
from apok_eval.ledger.index import append_index, verify_index
from apok_eval.ledger.record import RECORD_FILE, SCHEMA, compute_seal, load_record
from apok_eval.ledger.verify import run_verification, verify
from apok_eval.ledger.writer import RunWriter, open_run

__all__ = [
    "RECORD_FILE",
    "SCHEMA",
    "RunWriter",
    "append_index",
    "compute_seal",
    "load_record",
    "open_run",
    "run_verification",
    "verify",
    "verify_index",
]
