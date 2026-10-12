"""Installs the test network guard (``tests/netguard.py``) at the start of every Python subprocess a test
starts: ``tests/conftest.py`` puts this directory first on ``PYTHONPATH``. The ``sitecustomize`` it shadows,
if any, still runs."""

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("apok_test_netguard", os.path.join(os.path.dirname(_HERE),
                                                                                   "netguard.py"))
_guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_guard)
_guard.install()

_path, sys.path[:] = list(sys.path), [p for p in sys.path if os.path.abspath(p or os.curdir) != _HERE]
del sys.modules["sitecustomize"]
try:
    import sitecustomize  # noqa: F401  (the one this file shadows)
except ImportError:
    pass
finally:
    sys.path[:] = _path
