"""The append-only index of sealed records (conventionally ``ledger/index.jsonl``).

One line per sealed record, each the canonical JSON of::

    {"seq": n, "run_id": ..., "seal": "sha256:...", "record_path": ...,
     "prev_entry_digest": "sha256:..." | null}

followed by ``\\n``. ``seq`` is the 1-based line number. ``record_path`` is
the record's path relative to the index file's directory, in its one
normal spelling: POSIX, ``..`` only as leading components (to climb out of
the index directory), no empty, ``.`` or trailing ``..`` component, no
backslash, NUL or invalid UTF-8; so an absolute path is refused too. The
*entry digest* of a line is the sha256 of its bytes without the newline;
``prev_entry_digest`` is the entry digest of the line before (``null`` on
line 1), so each line commits to every line before it. A run_id appears at
most once.

:func:`verify_index` recomputes the chain and refuses a line that is not
canonical or well formed (``LEDGER_INDEX_MALFORMED``), a ``seq`` that is not
the line number (gaps, deletions, reorderings: ``LEDGER_INDEX_SEQUENCE_BROKEN``),
a broken link (edits: ``LEDGER_INDEX_CHAIN_BROKEN``) and a repeated run_id
(``LEDGER_INDEX_DUPLICATE_RUN``). With ``check_records`` (the default) each
``record_path`` must lead to a regular file holding a record whose run_id,
recorded seal and recomputed seal equal the entry's
(``LEDGER_INDEX_RECORD_MISMATCH``). Nothing links the *last* line, so this
is what catches its edited run_id or seal, or a forged appended line; an
edit of its ``record_path`` to another file holding the same record bytes (a
copy) is invisible to it, and only an anchor on the head (below) detects it.

What the chain cannot do on its own: anyone may rewrite the whole file
consistently, or drop lines from its end. Tamper *evidence* therefore comes
from anchoring an entry digest outside the file (a git commit of the index,
a note, a post) and passing it as ``anchor``: the anchored digest must be the
entry digest of some line (``LEDGER_INDEX_ANCHOR_MISSING``), which pins that
line and, through the chain, every line before it. Lines appended after the
anchor are legitimate appends as far as the anchor is concerned; they are
checked by the chain and the record check.

Appends take an exclusive ``flock`` on the index file (which must be a
regular file), re-verify the existing chain (without the record check) and
refuse a run_id already present. ``record_path`` is computed between the
resolved (symlink-free) locations of the record and of the index directory,
which is how :func:`verify_index` follows it.
"""

from __future__ import annotations

import fcntl
import itertools
import os
import stat
from pathlib import Path

from apok_eval import jsonstrict
from apok_eval.canonical import canonical_dumps, is_digest, sha256_bytes
from apok_eval.ledger.record import compute_seal, load_record
from apok_eval.ledger.util import NotRegularFile, is_name, is_utf8, read_regular
from apok_eval.refusal import Refusal

ENTRY_KEYS = frozenset({"seq", "run_id", "seal", "record_path", "prev_entry_digest"})


def _is_index_relpath(value: object) -> bool:
    if not isinstance(value, str) or "\\" in value or "\x00" in value or not is_utf8(value):
        return False
    parts = value.split("/")
    if any(part in ("", ".") for part in parts):  # also rules out "", an absolute path and a trailing slash
        return False
    rest = list(itertools.dropwhile(lambda part: part == "..", parts))
    return bool(rest) and ".." not in rest  # '..' only leading: one spelling per location


def entry_problems(entry: object) -> list[str]:
    """Structural problems of one index entry (empty when well formed)."""
    if not isinstance(entry, dict):
        return ["the line is not a JSON object"]
    if set(entry) != ENTRY_KEYS:
        return [f"members must be exactly {sorted(ENTRY_KEYS)}, got {sorted(entry)}"]
    checks = [
        (isinstance(entry["seq"], int) and not isinstance(entry["seq"], bool) and entry["seq"] >= 1,
         "seq must be a positive int"),
        (is_name(entry["run_id"]), "run_id must match [a-z0-9][a-z0-9._-]{0,127}"),
        (is_digest(entry["seal"]), "seal must be a sha256 digest"),
        (_is_index_relpath(entry["record_path"]), "record_path must be a relative POSIX path"),
        (entry["prev_entry_digest"] is None or is_digest(entry["prev_entry_digest"]),
         "prev_entry_digest must be null or a sha256 digest"),
    ]
    return [message for ok, message in checks if not ok]


def _parse(data: bytes, index_path: Path) -> list[tuple[dict, str]]:
    """``(entry, entry_digest)`` for every line, refusing any break in the chain."""
    if data and not data.endswith(b"\n"):
        raise Refusal("LEDGER_INDEX_MALFORMED", f"index {index_path}: the last line is not newline-terminated "
                      "(a torn append)", index=str(index_path), line=data.count(b"\n") + 1)
    entries: list[tuple[dict, str]] = []
    seen: set[str] = set()
    for number, line in enumerate(data.split(b"\n")[:-1], start=1):
        try:
            entry = jsonstrict.loads(line)
        except jsonstrict.StrictJsonError as exc:
            problems = [f"the line is not strict JSON: {exc}"]
        else:  # what jsonstrict returns serialises canonically: no NaN, lone surrogate or nesting past 256 levels
            problems = entry_problems(entry) if canonical_dumps(entry) == line else ["the line is not canonical JSON"]
        if problems:
            raise Refusal("LEDGER_INDEX_MALFORMED", f"index {index_path} line {number}: {problems[0]}",
                          index=str(index_path), line=number, problems=problems)
        if entry["seq"] != number:
            raise Refusal("LEDGER_INDEX_SEQUENCE_BROKEN", f"index {index_path} line {number} has seq "
                          f"{entry['seq']}", index=str(index_path), line=number, seq=entry["seq"])
        expected_prev = entries[-1][1] if entries else None
        if entry["prev_entry_digest"] != expected_prev:
            raise Refusal("LEDGER_INDEX_CHAIN_BROKEN", f"index {index_path} line {number}: prev_entry_digest is "
                          f"{entry['prev_entry_digest']}, the line before has digest {expected_prev}",
                          index=str(index_path), line=number, recorded=entry["prev_entry_digest"],
                          expected=expected_prev)
        if entry["run_id"] in seen:
            raise Refusal("LEDGER_INDEX_DUPLICATE_RUN", f"index {index_path} line {number}: run_id "
                          f"{entry['run_id']!r} appears earlier", index=str(index_path), line=number,
                          run_id=entry["run_id"])
        seen.add(entry["run_id"])
        entries.append((entry, sha256_bytes(line)))
    return entries


def read_index(index_path: str | os.PathLike[str]) -> list[tuple[dict, str]]:
    """Entries of a verified chain, with their entry digests."""
    path = Path(index_path)
    try:
        data = read_regular(path)
    except (OSError, ValueError) as exc:  # ValueError: a path holding NUL or an unencodable surrogate
        raise Refusal("LEDGER_INDEX_UNREADABLE", f"index {path} cannot be read: {exc}", index=str(path))
    return _parse(data, path)


def _check_indexed_record(index_dir: Path, entry: dict, number: int) -> None:
    path = index_dir.joinpath(*entry["record_path"].split("/"))
    try:
        record = load_record(path)
    except Refusal as refusal:
        problem = f"the record cannot be read ({refusal.reason})"
    else:
        found = {"run_id": record.get("run_id"), "seal": record.get("seal"), "recomputed seal": compute_seal(record)}
        wanted = {"run_id": entry["run_id"], "seal": entry["seal"], "recomputed seal": entry["seal"]}
        problem = "; ".join(f"{key} is {found[key]!r}, the index says {wanted[key]!r}"
                            for key in found if found[key] != wanted[key])
    if problem:
        raise Refusal("LEDGER_INDEX_RECORD_MISMATCH", f"index line {number} ({entry['run_id']}): {problem}",
                      line=number, run_id=entry["run_id"], record_path=entry["record_path"], problem=problem)


def verify_index(index_path: str | os.PathLike[str], *, anchor: str | None = None,
                 check_records: bool = True) -> dict:
    """Verify the chain (see the module docstring); return a summary or refuse."""
    path = Path(index_path)
    entries = read_index(path)
    if anchor is not None and anchor not in {digest for _, digest in entries}:
        raise Refusal("LEDGER_INDEX_ANCHOR_MISSING", f"no line of index {path} has the anchored entry digest "
                      f"{anchor}", index=str(path), anchor=anchor)
    if check_records:
        for number, (entry, _) in enumerate(entries, start=1):
            _check_indexed_record(path.parent, entry, number)
    return {"index": str(path), "entries": len(entries), "head": entries[-1][1] if entries else None,
            "anchor": anchor, "records_checked": len(entries) if check_records else 0}


def require_new_run(entries: list[tuple[dict, str]], run_id: str, index_path: Path) -> None:
    if any(entry["run_id"] == run_id for entry, _ in entries):
        raise Refusal("LEDGER_INDEX_DUPLICATE_RUN", f"run_id {run_id!r} is already in index {index_path}",
                      index=str(index_path), run_id=run_id)


def check_index_accepts(index_path: str | os.PathLike[str], run_id: str) -> None:
    """Pre-flight for a new run: an existing index must verify and must not list ``run_id``."""
    path = Path(index_path)
    if path.exists():
        require_new_run(read_index(path), run_id, path)


def append_index(index_path: str | os.PathLike[str], *, run_id: str, seal: str,
                 record_path: str | os.PathLike[str]) -> dict:
    """Append one entry under an exclusive lock; return it with its ``entry_digest``."""
    path = Path(os.path.abspath(index_path))
    # Between resolved locations: the OS resolves a symlinked index directory before applying '..'.
    rel = Path(os.path.relpath(os.path.realpath(record_path), os.path.realpath(path.parent))).as_posix()
    try:
        with open(path, "a+b") as fh:
            if not stat.S_ISREG(os.fstat(fh.fileno()).st_mode):
                raise NotRegularFile(path, "not a regular file")
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            fh.seek(0)
            entries = _parse(fh.read(), path)
            require_new_run(entries, run_id, path)
            entry = {"seq": len(entries) + 1, "run_id": run_id, "seal": seal, "record_path": rel,
                     "prev_entry_digest": entries[-1][1] if entries else None}
            problems = entry_problems(entry)
            if problems:
                raise Refusal("LEDGER_INDEX_MALFORMED", f"refusing to append a malformed entry: {problems[0]}",
                              index=str(path), problems=problems)
            line = canonical_dumps(entry)
            fh.write(line + b"\n")
            fh.flush()
            os.fsync(fh.fileno())
    except OSError as exc:
        raise Refusal("LEDGER_WRITE_FAILED", f"cannot append to index {path}: {exc}", path=str(path))
    return {**entry, "entry_digest": sha256_bytes(line)}
