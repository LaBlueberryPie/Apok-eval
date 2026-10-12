"""Refusal codes of ``apok_eval.ledger``, registered at import of the package.

Each code names one kind of refusal; the refusal's ``reason`` and ``details``
name the offending field, file, entry or line. ``docs/ledger.md`` lists what
triggers each code.
"""

from __future__ import annotations

from apok_eval.refusal import register_codes

CODES = {
    # --- writing a run ----------------------------------------------------------
    "LEDGER_RUN_ID_INVALID": "a run_id does not match [a-z0-9][a-z0-9._-]{0,127}",
    "LEDGER_NAME_INVALID": "an input, model, output, chunk-manifest or seed name does not match the name pattern",
    "LEDGER_NAME_DUPLICATE": "a name is used twice within one section of a run",
    "LEDGER_ARGUMENT_INVALID": "a writer argument is missing, of the wrong type or empty",
    "LEDGER_RUN_DIR_NOT_EMPTY": "the run directory exists and is not empty, and resuming was not requested",
    "LEDGER_ALREADY_SEALED": "the run is already sealed; a sealed record is never overwritten",
    "LEDGER_UNWRITABLE": "a directory the run must write to cannot be created or written",
    "LEDGER_DISK_INSUFFICIENT": "free disk space is below the run's stated estimate",
    "LEDGER_GIT_FAILED": "git could not establish the commit or working-tree state of the source",
    "LEDGER_SOURCE_UNSUPPORTED": "the package source or its git state cannot be recorded in apok-record-v1 (a symlink, "
                                 "a non-regular or unreadable file, an unsafe file name, a commit id that is not "
                                 "40 hex)",
    "LEDGER_DIRTY_TREE": "the git working tree is dirty and allow_dirty was not given",
    "LEDGER_CODE_CHANGED": "the commit or package source changed between opening and sealing the run",
    "LEDGER_WRITE_FAILED": "writing a sidecar or the record failed",
    "LEDGER_NO_OUTPUTS": "a run cannot be sealed without at least one output",
    "LEDGER_PREDICTIONS_INVALID": "the predictions block names no recorded input, or its ids are malformed",
    # --- writing and verifying ---------------------------------------------------
    "LEDGER_FILE_MISSING": "no regular file exists at a recorded path",
    "LEDGER_FILE_UNREADABLE": "a recorded path cannot be opened or read (the error is named)",
    "LEDGER_PATH_UNSAFE": "a run-relative path leaves the run directory, passes a symlink or is reserved",
    "LEDGER_SEED_INVALID": "a seed is not an int (bools are refused)",
    "LEDGER_SEED_CONFIG_MISMATCH": "a seed is not found, as the same int, at its pointer in the configuration",
    "LEDGER_MANIFEST_INVALID": "a chunk manifest (entries or sidecar lines) is malformed, unsorted or duplicated",
    "LEDGER_FIGURES_INVALID": "the figures source is not a recorded JSON output, or its pointer does not resolve",
    "LEDGER_CODE_MISMATCH": "the source manifest differs from the files of the recorded commit (with the recorded "
                            "diff applied, for a dirty tree), or that diff does not apply",
    # --- verifying a record -----------------------------------------------------
    "LEDGER_RECORD_UNREADABLE": "the record file cannot be read as a strict JSON object",
    "LEDGER_RECORD_NOT_CANONICAL": "the record file's bytes are not its canonical JSON serialisation",
    "LEDGER_SCHEMA_UNSUPPORTED": "the record does not declare schema apok-record-v1",
    "LEDGER_RECORD_MALFORMED": "the record's structure violates apok-record-v1",
    "LEDGER_SEAL_MISMATCH": "the recorded seal is not the digest of the record's content",
    "LEDGER_SEAL_UNEXPECTED": "the record's content does not have the anchored (expected) seal",
    "LEDGER_CONFIG_DIGEST_MISMATCH": "config_digest is not the digest of the recorded configuration",
    "LEDGER_FILE_SIZE_MISMATCH": "a file's size differs from its recorded size",
    "LEDGER_FILE_DIGEST_MISMATCH": "a file's sha256 differs from its recorded digest",
    "LEDGER_MANIFEST_ROOT_MISMATCH": "a chunk manifest's Merkle root differs from the recorded root",
    "LEDGER_MANIFEST_COUNT_MISMATCH": "a chunk manifest's present/absent counts differ from the recorded counts",
    "LEDGER_CHUNK_MISMATCH": "a chunk read again from a local store differs from its recorded digest or presence",
    "LEDGER_FIGURES_MISMATCH": "the recorded figures differ from the content of their output file",
    "LEDGER_PREDICTIONS_MISMATCH": "the predictions registry digest differs from the recorded registry input",
    "LEDGER_SOURCE_ROOT_MISMATCH": "source_manifest_root is not the Merkle root of the source manifest",
    "LEDGER_COMMIT_MISSING": "the recorded commit is not in the repository verify was explicitly given",
    "LEDGER_REPO_INVALID": "the repository verify was explicitly given cannot be opened by git",
    "LEDGER_VERIFY_FAILED": "one or more verification checks failed (details list each)",
    # --- the append-only index --------------------------------------------------
    "LEDGER_INDEX_UNREADABLE": "the index file cannot be read",
    "LEDGER_INDEX_MALFORMED": "an index line is not a canonical, well-formed entry",
    "LEDGER_INDEX_SEQUENCE_BROKEN": "an index line's seq is not its line number (gap, deletion or reordering)",
    "LEDGER_INDEX_CHAIN_BROKEN": "an index line's prev_entry_digest is not the digest of the line before it",
    "LEDGER_INDEX_DUPLICATE_RUN": "a run_id appears in the index more than once",
    "LEDGER_INDEX_RECORD_MISMATCH": "an indexed record is missing or its run_id or seal differs from the index",
    "LEDGER_INDEX_ANCHOR_MISSING": "the anchored entry digest is not the digest of any index line",
}

register_codes("apok_eval.ledger", CODES)
