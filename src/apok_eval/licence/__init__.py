"""apok_eval.licence - the asset registry and the licence gate.

``apok-eval licence-check`` refuses any input that is not a declared asset with
a licence, any input on a host that does not allow that licence, and any input
whose licence or additional terms require terms that the run configuration's
``terms_accepted_by`` - the acceptance of the person running it, in their own
words - does not cover. A run that passes gets a :class:`LicenceClearance`,
which only the gate constructs.

A reader that requires a clearance issued by this gate checks the exact type,
``type(c) is LicenceClearance``, as :class:`apok_eval.io.store.HttpStore` does:
:class:`apok_eval.clearance.Clearance` is a runtime-checkable Protocol, so
``isinstance`` against it accepts any object with a ``covers`` method, and
``isinstance(c, LicenceClearance)`` consults ``__class__``, which a mock can
claim (see :mod:`apok_eval.clearance`).

* :mod:`apok_eval.licence.uri` - one canonical form for every location;
* :mod:`apok_eval.licence.registry` - the registry format and its validation;
* :mod:`apok_eval.licence.gate` - run configurations and the gate itself.

Importing the package registers every ``LICENCE_*`` refusal code. The full
specification is ``docs/licence.md``.
"""

from apok_eval.licence.gate import (
    Acceptance,
    AcceptedTerms,
    ClearedInput,
    InputDecision,
    LicenceClearance,
    check_run,
    input_decisions,
    licence_check,
    load_run_config,
)
from apok_eval.licence.registry import (
    GATED_BUCKETS,
    GATED_DOMAINS,
    SCHEMA,
    SHIPPED_REGISTRY,
    Registry,
    canonical_sample,
    is_gated,
    load_registry,
    parse_registry,
)
from apok_eval.licence.uri import CanonicalURI, canonical_uri

__all__ = [
    "Acceptance", "AcceptedTerms", "CanonicalURI", "ClearedInput", "GATED_BUCKETS", "GATED_DOMAINS",
    "InputDecision", "LicenceClearance", "Registry", "SCHEMA", "SHIPPED_REGISTRY", "canonical_sample",
    "canonical_uri", "check_run", "input_decisions", "is_gated", "licence_check", "load_registry",
    "load_run_config", "parse_registry",
]
