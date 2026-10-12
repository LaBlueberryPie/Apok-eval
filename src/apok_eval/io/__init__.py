"""apok_eval.io - readers and writers for the formats apok-eval measures, and the ``apok-eval io`` command.

* :mod:`apok_eval.io.frame` - the declared coordinate frame every reader checks against;
* :mod:`apok_eval.io.store`, :mod:`apok_eval.io.readlog`, :mod:`apok_eval.io.omezarr` - object
  stores (remote reads need a clearance issued by :mod:`apok_eval.licence`), the read
  log and the OME-Zarr v2 reader (``docs/io-zarr.md``);
* :mod:`apok_eval.io.tifxyz`, :mod:`apok_eval.io.mesh`, :mod:`apok_eval.io.pointcollection`,
  :mod:`apok_eval.io.umbilicus` - surfaces, meshes, point collections and the
  umbilicus (``docs/io-surfaces.md``);
* :mod:`apok_eval.io.cli` - ``apok-eval io inspect``.

Importing the package imports every module, so every ``IO_*`` refusal code is
registered whichever module a caller reaches first. Imports go one way:
``apok_eval.io.store`` imports :mod:`apok_eval.licence` (for the nominal clearance type);
nothing in :mod:`apok_eval.licence` imports ``apok_eval.io``.
"""

from apok_eval.io import frame, store, readlog, omezarr, tifxyz, mesh, pointcollection, umbilicus  # noqa: I001
from apok_eval.io import cli  # last: the command composes the modules above

__all__ = ["cli", "frame", "mesh", "omezarr", "pointcollection", "readlog", "store", "tifxyz", "umbilicus"]
