"""The release guard: no number in this repository is a withheld figure.

Results computed on PHercParis4 data are not published here. Before a release, every number written in every
tracked text file, in every file staged in the index, and in every file and message of every commit on any ref
is reduced to one canonical form (``release/figures.py``) and its sha256 looked up in a list of the hashes of
the withheld figures (``release/figure_hashes.json``, or the file named by ``APOK_EVAL_FIGURE_HASHES``). The
list is **never committed**: its hashes are of short numeric strings and can be inverted by enumeration, so it
is as private as the figures.

A match is a candidate, not a finding: a list of every figure of many runs covers much of the space of short
numbers, so code constants, synthetic test values and line numbers in citations match by coincidence. Each
reviewed coincidence is named in a private allow-list (``release/figure_allowlist.txt``, or the file named by
``APOK_EVAL_FIGURE_ALLOWLIST``), which is never committed either: every entry is a member of the hash list, so
publishing one would confirm it. An entry allows one number on one reviewed line of one file, bound by the
line's hash; the same number on any other line, even in a file where it is already allowed, fails and goes to
a maintainer, and so does an entry that nothing uses any more (the list stays exact).

The object store must hold no unreachable object: a copy, archive or plain local clone of the directory carries
every object in it, including old versions of files that no commit holds and that the scans never read.

**What the list does not cover** (``release/README.md``, "What the guard does not see"): integers under 100,
numbers with one significant digit after canonicalisation, figures written in words or as rounded prose ranges,
and dates. Those need a reading pass at each release; this test cannot stand in for it.

Without the private files the scanning tests skip, saying so; the tests of the normaliser, of the allow-list's
format (on a synthetic example) and that neither private file is tracked always run.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RELEASE = ROOT / "release"
HASHES = Path(os.environ.get("APOK_EVAL_FIGURE_HASHES") or RELEASE / "figure_hashes.json")
ALLOWLIST = Path(os.environ.get("APOK_EVAL_FIGURE_ALLOWLIST") or RELEASE / "figure_allowlist.txt")
PRIVATE_PATHS = ("release/figure_hashes.json", "release/figure_allowlist.txt")
LINE_DIGEST = re.compile(r"[0-9a-f]{16}")


def _load_figures():
    spec = importlib.util.spec_from_file_location("apok_eval_release_figures", RELEASE / "figures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


figures = _load_figures()


def _git(*args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return subprocess.run(["git", *args], cwd=ROOT, env=env, capture_output=True, timeout=120)


def _is_checkout() -> bool:
    try:
        top = _git("rev-parse", "--show-toplevel")
    except FileNotFoundError:
        return False
    return top.returncode == 0 and Path(top.stdout.decode().strip()).resolve() == ROOT.resolve()


def line_digest(line: str) -> str:
    """The first 16 hex digits of the sha256 of ``line`` stripped of surrounding whitespace: what an entry binds."""
    return hashlib.sha256(line.strip().encode("utf-8")).hexdigest()[:16]


def parse_allowlist(text: str) -> dict[tuple[str, str, str], str]:
    """``{(form, path, line digest): reason}`` from the allow-list's ``FORM | PATH | LINE | REASON`` lines."""
    entries: dict[tuple[str, str, str], str] = {}
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = [p.strip() for p in line.split(" | ", 3)]
        assert len(parts) == 4, f"allow-list line {n}: expected FORM | PATH | LINE | REASON, got {line!r}"
        form, path, digest, reason = parts
        assert figures.canonical(form) == form, f"allow-list line {n}: {form!r} is not a canonical form"
        assert path and not path.startswith("/") and ".." not in path.split("/") and not set("*?[,") & set(path), \
            f"allow-list line {n}: {path!r} is not one repository path"
        assert LINE_DIGEST.fullmatch(digest), f"allow-list line {n}: {digest!r} is not a line digest (16 hex)"
        assert len(reason) >= 8, f"allow-list line {n}: an entry needs a reason"
        assert (form, path, digest) not in entries, f"allow-list line {n}: listed twice"
        entries[(form, path, digest)] = reason
    return entries


def scan_text(path: str, text: str, hashes, entries, seen: set) -> list[str]:
    """``path:line: token -> form`` for every match no entry allows on that line; the entries used go to ``seen``."""
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        digest = None
        for tok, form in figures.tokens(line):
            if figures.figure_hash(form) not in hashes:
                continue
            digest = digest or line_digest(line)
            if (form, path, digest) in entries:
                seen.add((form, path, digest))
            else:
                out.append(f"{path}:{n}: {tok!r} -> {form} ({figures.sig_digits(form)} significant digits; "
                           f"line {digest})")
    return out


def _text(data: bytes) -> str | None:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None  # binary: no written numbers


def _need_private_files() -> None:
    if not HASHES.is_file():
        pytest.skip(f"the private figure list is not here ({HASHES}); this guard runs before a release, on the "
                    "maintainers' machine (release/README.md)")
    if not _is_checkout():
        pytest.skip("not a git checkout: the guard scans tracked files and history")
    assert ALLOWLIST.is_file(), (f"the figure list is here but its allow-list is not ({ALLOWLIST}): both private "
                                 "files are needed (release/README.md)")


@lru_cache(maxsize=1)
def _hashes() -> frozenset[str]:
    return frozenset(json.loads(HASHES.read_text(encoding="utf-8"))["hashes"])


@lru_cache(maxsize=1)
def _entries() -> dict:
    return parse_allowlist(ALLOWLIST.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _scan_tree() -> tuple[tuple[str, ...], frozenset]:
    seen: set = set()
    found: list[str] = []
    listing = _git("ls-files", "-z")
    assert listing.returncode == 0
    paths = [p for p in listing.stdout.decode("utf-8").split("\0") if p]
    assert "release/figures.py" in paths and not set(PRIVATE_PATHS) & set(paths)
    for path in paths:
        if not (ROOT / path).is_file():
            continue
        text = _text((ROOT / path).read_bytes())
        if text is not None:
            found += scan_text(path, text, _hashes(), _entries(), seen)
    return tuple(found), frozenset(seen)


@lru_cache(maxsize=1)
def _scan_history() -> tuple[tuple[str, ...], frozenset]:
    """Every file and message of every commit on any ref, and every file staged in the index."""
    seen: set = set()
    found: list[str] = []
    blobs: dict[tuple[str, str], str] = {}
    listed = _git("rev-list", "--all")
    assert listed.returncode == 0
    for commit in listed.stdout.decode().split():
        message = _git("log", "-1", "--format=%B", commit).stdout.decode("utf-8", "replace")
        found += scan_text(f"<message of {commit[:12]}>", message, _hashes(), {}, seen)
        tree = _git("ls-tree", "-r", "-z", commit).stdout.decode("utf-8").split("\0")
        for item in filter(None, tree):
            meta, path = item.split("\t", 1)
            kind, oid = meta.split()[1:3]
            if kind == "blob":
                blobs.setdefault((oid, path), commit[:12])
    staged = _git("ls-files", "-s", "-z").stdout.decode("utf-8").split("\0")
    for item in filter(None, staged):
        meta, path = item.split("\t", 1)
        blobs.setdefault((meta.split()[1], path), "<index>")
    assert not set(PRIVATE_PATHS) & {path for _, path in blobs}, "a private file was committed or staged"
    for (oid, path), where in sorted(blobs.items(), key=lambda kv: kv[0][1]):
        text = _text(_git("cat-file", "blob", oid).stdout)
        if text is not None:
            found += [f"{where} {hit}" for hit in scan_text(path, text, _hashes(), _entries(), seen)]
    return tuple(found), frozenset(seen)


# --- the guard ---------------------------------------------------------------------------------------------


def test_the_private_files_are_never_tracked():
    if not _is_checkout():
        pytest.skip("not a git checkout")
    for path in PRIVATE_PATHS:
        assert _git("ls-files", "--error-unmatch", path).returncode != 0, f"{path} is tracked"
        history = _git("log", "--all", "--format=%H", "--", path)
        assert history.returncode == 0 and history.stdout.strip() == b"", f"{path} is in the history"
        assert _git("check-ignore", "-q", path).returncode == 0, f".gitignore does not ignore {path}"


def test_no_tracked_file_holds_a_withheld_figure():
    _need_private_files()
    found, _ = _scan_tree()
    assert found == (), "numbers that match a withheld figure:\n" + "\n".join(found)


def test_no_commit_holds_a_withheld_figure():
    _need_private_files()
    found, _ = _scan_history()
    assert found == (), "numbers in the history that match a withheld figure:\n" + "\n".join(found)


def test_every_allow_list_entry_is_still_used():
    _need_private_files()
    stale = sorted(set(_entries()) - _scan_tree()[1] - _scan_history()[1])
    assert stale == [], "allow-list entries that no file or commit uses any more (remove them):\n" + "\n".join(
        " | ".join(key) for key in stale)


def test_the_object_store_holds_no_unreachable_object():
    """A copy, archive or plain local clone of the directory carries every object, reachable or not."""
    _need_private_files()
    fsck = _git("fsck", "--unreachable", "--no-reflogs", "--no-progress")
    assert fsck.returncode == 0, fsck.stderr.decode("utf-8", "replace")
    unreachable = fsck.stdout.decode("utf-8", "replace").strip()
    assert unreachable == "", ("objects that no ref reaches (expire the reflog and prune them before a release: "
                               "release/README.md):\n" + unreachable)


# --- the normaliser and the allow-list --------------------------------------------------------------------


@pytest.mark.parametrize("text, expected", [
    ("12.5%", [("12.5%", "0.125")]),
    ("1,024", [("1,024", "1024"), ("1", "1"), ("024", "24")]),
    ("1e-5 and 1.2E+3", [("1e-5", "0.00001"), ("1.2E+3", "1200")]),
    ("10_000", [("10_000", "10000")]),
    ("-0.25 and −0.25", [("-0.25", "0.25"), ("−0.25", "0.25")]),
    ("2048.0", [("2048.0", "2048")]),
    ("q 0.25-0.75", [("0.25", "0.25"), ("0.75", "0.75")]),
    ("w003-007", [("007", "7")]),
    ("sha256:9f7c 0x1f a1b2 v2", []),
    ("x.tif:12", [("12", "12")]),
    (".5 0.50", [(".5", "0.5"), ("0.50", "0.5")]),
])
def test_the_normaliser_reduces_every_written_number_to_one_form(text, expected):
    assert list(figures.tokens(text)) == expected


def test_significant_digits_and_hash():
    assert [figures.sig_digits(c) for c in ("0.125", "1024", "2048", "100", "0.00001")] == [3, 4, 4, 1, 1]
    assert figures.figure_hash("0.5") == hashlib.sha256(b"0.5").hexdigest()


# A synthetic allow-list: the format, not the private list's content.
EXAMPLE_LINE = "CHUNK = 4096"
EXAMPLE = f"""# FORM | PATH | LINE | REASON
4096 | src/a.py | {line_digest(EXAMPLE_LINE)} | a chunk size | a reason may hold the separator
"""


def test_the_allow_list_format():
    assert parse_allowlist(EXAMPLE) == {("4096", "src/a.py", line_digest("  CHUNK = 4096 ")):
                                        "a chunk size | a reason may hold the separator"}
    good = f"4096 | src/a.py | {line_digest(EXAMPLE_LINE)} | a chunk size"
    for bad in (good.replace("4096 |", "4,096 |"), good.replace("src/a.py", "src/*.py"),
                good.replace("src/a.py", "/src/a.py"), good.replace("src/a.py", "src/a.py, src/b.py"),
                good.replace(line_digest(EXAMPLE_LINE), "line 3"), good.replace("a chunk size", "chunk"),
                "4096 | src/a.py | a chunk size", f"{good}\n{good}"):
        with pytest.raises(AssertionError):
            parse_allowlist(bad)


def test_the_private_allow_list_is_well_formed():
    if not ALLOWLIST.is_file():
        pytest.skip(f"the private allow-list is not here ({ALLOWLIST})")
    assert _entries(), "the allow-list is empty"


def test_the_scanner_allows_a_number_only_on_its_reviewed_line():
    # a list made up here: 0.12345678 stands for a withheld figure, 4096 for a coincidence
    hashes = frozenset({figures.figure_hash("0.12345678"), figures.figure_hash("4096")})
    entries = parse_allowlist(EXAMPLE)
    seen: set = set()
    doc = "Q was 0.12345678 and 4,096 points"
    hits = scan_text("docs/x.md", doc, hashes, entries, seen)
    assert hits == [f"docs/x.md:1: '0.12345678' -> 0.12345678 (8 significant digits; line {line_digest(doc)})",
                    f"docs/x.md:1: '4,096' -> 4096 (4 significant digits; line {line_digest(doc)})"]
    assert seen == set()
    source = "  CHUNK = 4096\nBATCH = 4096  # a new use, in a file where the number is allowed\n"
    hits = scan_text("src/a.py", source, hashes, entries, seen)
    assert len(hits) == 1 and hits[0].startswith("src/a.py:2: '4096' -> 4096")
    assert seen == set(entries)
