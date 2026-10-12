"""Synthetic spiral fits, fibres and shells for the evaluator tests: the package's own synthetic world
(:mod:`apok_eval.eval.synthetic`, which the demo also uses), re-exported under the names the tests use.
Nothing here is scroll data."""

from __future__ import annotations

from apok_eval.eval.synthetic import (CENTRE, FRAME, STEP, fibre_bytes, fibre_document, fibre_points, layered_block,
                                      meta_for, omega_columns, spiral_point, write_ct, write_fit, write_segment,
                                      write_shell, write_umbilicus_file)

_meta = meta_for

__all__ = ["CENTRE", "FRAME", "STEP", "fibre_bytes", "fibre_document", "fibre_points", "layered_block", "meta_for",
           "omega_columns", "spiral_point", "write_ct", "write_fit", "write_segment", "write_shell",
           "write_umbilicus_file", "_meta"]
