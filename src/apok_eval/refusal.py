"""The one way an apok-eval guard says no.

Every guard in apok-eval refuses by raising :class:`Refusal` with a registered,
stable ``code`` and a ``reason`` that names the cause. Guards never return a
convenient default in place of a refusal, and never refuse with a code that
was not declared: an undeclared code raises ``ValueError`` at the raise site,
so a typo cannot pass for a refusal.

Guards must be written as a literal ``raise Refusal(...)`` statement, so that
a mutation run can find every such statement, delete each in turn and require
the test suite to go red.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

_CODE_RE = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$")

#: The status a report gives a check it did not or could not make (``apok-eval verify``, the frame of
#: ``apok-eval inspect``). It is never a pass, and never silently dropped from a report.
NOT_CHECKED = "NOT CHECKED"

#: code -> one-line description. Filled by :func:`register_codes`.
REGISTRY: dict[str, str] = {}


def register_codes(owner: str, codes: Mapping[str, str]) -> None:
    """Declare refusal codes for ``owner`` (a module name).

    Codes are UPPER_SNAKE with at least two parts. Re-registering a code
    with a different description, or from a different owner, is an error.
    """
    for code, description in codes.items():
        if not _CODE_RE.match(code):
            raise ValueError(f"refusal code {code!r} is not UPPER_SNAKE with two or more parts")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"refusal code {code!r} has no description")
        entry = f"{owner}: {description.strip()}"
        existing = REGISTRY.get(code)
        if existing is not None and existing != entry:
            raise ValueError(f"refusal code {code!r} already registered as {existing!r}")
        REGISTRY[code] = entry


class Refusal(Exception):
    """A guard refused. ``code`` is stable; ``reason`` names the cause."""

    def __init__(self, code: str, reason: str, **details: Any) -> None:
        if code not in REGISTRY:
            raise ValueError(f"unregistered refusal code {code!r}")
        if not reason or not reason.strip():
            raise ValueError(f"refusal {code} raised without a reason")
        super().__init__(f"{code}: {reason}")
        self.code = code
        self.reason = reason
        self.details = details

    def as_dict(self) -> dict[str, Any]:
        return {"refused": True, "code": self.code, "reason": self.reason, "details": self.details}
