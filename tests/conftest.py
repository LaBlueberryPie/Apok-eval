"""Test-wide guards.

No test may reach the network beyond loopback, and neither may any Python process a test starts
(``tests/netguard.py``): the guard is installed here at import time, before any test module runs, and
``tests/netguard_site`` goes first on ``PYTHONPATH`` so that its ``sitecustomize`` installs the guard in
every subprocess that inherits the environment. A test that sets ``PYTHONPATH`` for a subprocess keeps
``os.environ["PYTHONPATH"]`` on it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import netguard  # noqa: E402

netguard.install()
NETGUARD_SITE = str(Path(__file__).resolve().parent / "netguard_site")
os.environ["PYTHONPATH"] = os.pathsep.join([NETGUARD_SITE, *filter(None, [os.environ.get("PYTHONPATH")])])
