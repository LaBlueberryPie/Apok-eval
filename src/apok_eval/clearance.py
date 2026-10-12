"""The interface between the licence gate and anything that reads data.

``apok_eval.licence`` issues clearances; readers in ``apok_eval.io`` require one before
any remote read. The protocol lives here so neither package needs the other
for typing.

:class:`Clearance` is for type annotations only. Because it is
``runtime_checkable``, ``isinstance(x, Clearance)`` is true for ANY object
with a ``covers`` method, so it cannot tell a gate-issued clearance from a
stand-in. Readers therefore check the nominal type exactly,
``type(c) is apok_eval.licence.LicenceClearance`` (which only
``apok_eval.licence.check_run`` constructs), and then ``covers(uri) is True``;
``apok_eval.io.store.HttpStore`` does exactly that. ``isinstance`` against the
class is not enough either: it consults ``__class__``, so a
``unittest.mock.Mock(spec=LicenceClearance)`` passes it, as does a subclass
overriding ``covers``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Clearance(Protocol):
    def covers(self, uri: str) -> bool:
        """True only if the licence gate cleared ``uri`` for this run."""
        ...
