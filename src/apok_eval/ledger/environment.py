"""The ``environment`` block of a record.

Recorded: the Python version and implementation, ``platform.platform()``
(operating system, release, machine and libc; it contains no host name) and
``platform.machine()``, and package versions: ``apok-eval`` (the ``__version__``
of the code that ran), the declared dependencies ``numpy``, ``tifffile``,
``numcodecs`` and ``scipy`` (``null`` when no installed distribution is found),
and ``imagecodecs`` (the ``lzw`` extra, which decides whether an LZW tifxyz
decodes) and ``zarr`` only when installed. Versions are read from distribution metadata, so nothing is
imported to learn them. Never recorded here: the host name, the user name, or
environment variables (privacy). Other members of a record are not so
clean: input and model paths are absolute and usually contain the user's
home directory (docs/ledger.md, "Record"); redact them before publishing a
record.

The environment cannot be recomputed from anything else in a record;
``verify`` reports it as NOT CHECKED. Only the seal protects it.
"""

from __future__ import annotations

import importlib.metadata
import platform

from apok_eval import __version__

# REQUIRED_PACKAGES are pyproject's [project] dependencies (a test keeps the two equal).
REQUIRED_PACKAGES = ("numpy", "tifffile", "numcodecs", "scipy")
OPTIONAL_PACKAGES = ("imagecodecs", "zarr")


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def capture_environment() -> dict:
    packages: dict[str, str | None] = {"apok-eval": __version__}
    for name in REQUIRED_PACKAGES:
        packages[name] = _version(name)
    for name in OPTIONAL_PACKAGES:
        version = _version(name)
        if version is not None:
            packages[name] = version
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
    }
