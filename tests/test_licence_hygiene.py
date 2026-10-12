"""Acceptance hygiene: ``terms_accepted_by`` is the accepting person's own words, so no committed file may set it.

The scan covers every git-tracked file (``git ls-files``). JSON, TOML and
Python are parsed, never pre-filtered on their raw text, so a key written with
escapes (``"terms_accepted\\u005fby"``) or, in Python, by implicit
concatenation (``"terms_" "accepted_by"``) is seen as the parser - and the
gate - sees it:

* JSON (``.json``, ``.ipynb``, and **every** other tracked file whose whole
  text parses as JSON - ``.py``, ``.toml`` and ``.yaml`` included, on top of
  their own check - since the gate reads a run configuration of any name as
  strict JSON) and TOML must not give the key a non-null value anywhere in
  the document;
* YAML is matched by text (no YAML parser is installed): a plain,
  single-quoted or double-quoted key (escapes decoded) whose value is anything
  but a plain ``null``/``Null``/``NULL``/``~`` counts as set - a quoted
  ``'null'`` is the string "null" - and so does an explicit ``? key``;
* Python under ``src/`` may only read it (a dict literal key, a subscript
  store or delete, a keyword argument, ``setdefault``, ``__setitem__``,
  ``setattr`` or an attribute store of that name is a write).

Test code may build acceptances at test time; outside ``src/`` its Python is
not scanned as Python (only as whole-text JSON, like every file).

This is a hygiene guard against an agent or a commit, not proof of identity:
anyone can type a name into a run configuration; a key built at run time
(``"terms_" + "accepted_by"``) escapes an AST scan; YAML anchors, aliases
and multi-line keys are not interpreted; Python outside ``src/``
and files that are none of the above (``.cfg``, ``.ini``, ``.env``,
Markdown) are scanned only as whole-text JSON. Without a git checkout (a scratch copy of the
tree) the tracked-file scan is skipped; the scanner's own tests below
still run.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import tomllib
from pathlib import Path

import pytest

KEY = "terms_accepted_by"
ROOT = Path(__file__).resolve().parent.parent
# A YAML key - double-quoted (any text: escapes are decoded below), single-quoted or plain - at the start of the text
# or after whitespace, '{', ',', '?' or a list dash, then ':' and the value up to a comment, ',' or '}'.
_YAML_KEY = re.compile(
    rf"""(?<![^\s{{,\-?])(?P<key>"(?:[^"\\\n]|\\.)*"|'{KEY}'|{KEY})[ \t]*:(?P<value>[^\n#,}}]*)""")
# An explicit key ("? key" on its own line): its value, on the next line, is not read, so it counts as set.
_YAML_EXPLICIT_KEY = re.compile(rf"""(?m)^[ \t]*\?[ \t]+(?P<key>"(?:[^"\\\n]|\\.)*"|'{KEY}'|{KEY})[ \t]*(?:#.*)?$""")
_YAML_ESCAPE = re.compile(r"\\(x[0-9A-Fa-f]{2}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8}|.)")
_YAML_SIMPLE_ESCAPES = {"0": "\0", "a": "\a", "b": "\b", "t": "\t", "\t": "\t", "n": "\n", "v": "\v", "f": "\f",
                        "r": "\r", "e": "\x1b", " ": " ", '"': '"', "/": "/", "\\": "\\", "N": "\x85", "_": "\xa0",
                        "L": "\u2028", "P": "\u2029"}
_YAML_NULL = {"null", "Null", "NULL", "~"}
_WRITING_METHODS = {"setdefault", "__setitem__"}


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _first_line(text: str) -> int:
    return _line_of(text, max(text.find(KEY), 0))


def _json_values(text: str) -> list[object] | None:
    """The values given to the key anywhere in the JSON ``text`` (escapes decoded), or None if it is not JSON."""
    found: list[object] = []

    def pairs(items):
        found.extend(value for key, value in items if key == KEY)
        return dict(items)

    try:
        json.loads(text, object_pairs_hook=pairs)
    except (ValueError, RecursionError):  # nesting too deep for json is far beyond what the gate's parser accepts
        return None
    return found


def _json_problems(text: str) -> list[tuple[int, str]]:
    values = _json_values(text)
    if values is None:
        return [(_first_line(text), "unparseable JSON mentions the key")] if KEY in text else []
    return [(_first_line(text), f"sets it to {value!r}") for value in values if value is not None]


def _json_like_problems(text: str) -> list[tuple[int, str]]:
    # A tracked file of another type is checked only if its whole text is JSON: the gate would read it.
    values = _json_values(text)
    return [] if values is None else [(_first_line(text), f"JSON sets it to {value!r}") for value in values
                                      if value is not None]


def _yaml_unescape(body: str) -> str:
    def one(match: re.Match[str]) -> str:
        code = match.group(1)
        if len(code) > 1:
            point = int(code[1:], 16)
            return chr(point) if point <= 0x10FFFF else "\ufffd"
        return _YAML_SIMPLE_ESCAPES.get(code, "\ufffd")

    return _YAML_ESCAPE.sub(one, body)


def _is_yaml_key(key: str) -> bool:
    return not key.startswith('"') or _yaml_unescape(key[1:-1]) == KEY


def _yaml_problems(text: str) -> list[tuple[int, str]]:
    problems = [(_line_of(text, match.start()), "an explicit '?' key sets it (its value is not read)")
                for match in _YAML_EXPLICIT_KEY.finditer(text) if _is_yaml_key(match.group("key"))]
    for match in _YAML_KEY.finditer(text):
        if not _is_yaml_key(match.group("key")):
            continue
        value = match.group("value").strip()
        if value not in _YAML_NULL:  # a quoted 'null' is the string "null"
            shown = value or "<a block>"
            problems.append((_line_of(text, match.start()), f"sets it to {shown!r} (only null is allowed)"))
    return problems


def _toml_problems(text: str) -> list[tuple[int, str]]:
    try:
        doc = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return [(_first_line(text), "unparseable TOML mentions the key")] if KEY in text else []
    stack, hits = [doc], 0
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            hits += KEY in node
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return [(_first_line(text), "sets it (TOML has no null)")] * hits


def _is_key(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and node.value == KEY


def _python_problems(text: str) -> list[tuple[int, str]]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return [(_first_line(text), "unparseable Python mentions the key")] if KEY in text else []
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            problems += [(k.lineno, "a dict literal sets it") for k in node.keys if _is_key(k)]
        elif isinstance(node, ast.Subscript) and isinstance(node.ctx, (ast.Store, ast.Del)) and _is_key(node.slice):
            problems.append((node.lineno, "a subscript assigns or deletes it"))
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store) and node.attr == KEY:
            problems.append((node.lineno, "an attribute assignment sets it"))
        elif isinstance(node, ast.Call):
            problems += [(node.lineno, "a keyword argument sets it") for kw in node.keywords if kw.arg == KEY]
            func = node.func
            first = node.args[0] if node.args else None
            if isinstance(func, ast.Attribute) and func.attr in _WRITING_METHODS and _is_key(first):
                problems.append((node.lineno, f"{func.attr}() sets it"))
            if isinstance(func, ast.Name) and func.id == "setattr" and len(node.args) > 1 and _is_key(node.args[1]):
                problems.append((node.lineno, "setattr() sets it"))
    return problems


def _checks_for(rel: str) -> list:
    """The checks for one tracked file: its suffix's own check, and the whole-text JSON check for every file.

    The gate reads a run configuration as strict JSON whatever the file is called, so a file of any
    suffix - Python outside ``src/``, TOML or YAML that is really JSON, Python too deeply nested to
    parse - is read as the gate would read it, not only as its suffix suggests.
    """
    suffix = Path(rel).suffix.lower()
    if suffix in (".json", ".ipynb"):
        return [_json_problems]  # already the whole-text JSON check, and stricter (unparseable is reported)
    if suffix in (".yaml", ".yml"):
        return [_yaml_problems, _json_like_problems]
    if suffix == ".toml":
        return [_toml_problems, _json_like_problems]
    if suffix == ".py" and rel.startswith("src/"):
        return [_python_problems, _json_like_problems]
    return [_json_like_problems]


def scan(root: Path, paths: list[str]) -> list[str]:
    """``path:line: why`` for every place in ``paths`` (relative to ``root``) that sets the key."""
    out = []
    for rel in sorted(paths):
        path = root / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        found: list[str] = []
        for check in _checks_for(rel):
            found += [f"{rel}:{line}: {why}" for line, why in check(text) if f"{rel}:{line}: {why}" not in found]
        out += found
    return out


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}  # never aim at another repository
    return subprocess.run(["git", *args], cwd=root, env=env, capture_output=True, timeout=60)


def tracked_files(root: Path) -> list[str] | None:
    """``git ls-files`` of ``root``, or None when ``root`` is not the top of a git checkout."""
    try:
        top = _git(root, "rev-parse", "--show-toplevel")
    except FileNotFoundError:
        return None
    if top.returncode != 0 or Path(top.stdout.decode().strip()).resolve() != root.resolve():
        return None
    listing = _git(root, "ls-files", "-z")
    assert listing.returncode == 0, listing.stderr
    return [p for p in listing.stdout.decode("utf-8").split("\0") if p]


# --- the guard ------------------------------------------------------------------------


def test_no_tracked_file_sets_terms_accepted_by():
    files = tracked_files(ROOT)
    if files is None:
        pytest.skip("not a git checkout (e.g. a scratch copy of the tree); nothing is tracked here")
    assert "src/apok_eval/refusal.py" in files and "pyproject.toml" in files  # the listing is real
    assert scan(ROOT, files) == []


def test_source_tree_on_disk_never_writes_terms_accepted_by():
    files = [p.relative_to(ROOT).as_posix() for p in (ROOT / "src").rglob("*") if p.is_file()]
    assert "src/apok_eval/licence/gate.py" in files and "src/apok_eval/licence/registry.json" in files
    assert KEY in (ROOT / "src/apok_eval/licence/gate.py").read_text(encoding="utf-8")  # it is read there ...
    assert scan(ROOT, files) == []  # ... and never written


# --- the scanner can fail ----------------------------------------------------------------


def _scan_one(tmp_path: Path, rel: str, text: str) -> list[str]:
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return scan(tmp_path, [rel])


@pytest.mark.parametrize(
    "rel, text",
    [
        ("run.json", '{"inputs": [], "terms_accepted_by": {"person": "Jane Example", "date": "2025-06-01", "accepted": []}}'),
        ("deep.json", '{"runs": [{"cfg": {"terms_accepted_by": "Jane Example"}}]}'),
        ("dup.json", '{"terms_accepted_by": {"person": "Jane Example"}, "terms_accepted_by": null}'),
        ("broken.json", '{"terms_accepted_by": {"person": "Jane Example"'),
        ("nb.ipynb", '{"cells": [], "metadata": {"terms_accepted_by": "Jane Example"}}'),
        ("run.yaml", "inputs: []\nterms_accepted_by: Jane Example\n"),
        ("block.yml", "terms_accepted_by:\n  person: Jane Example\n"),
        ("flow.yaml", "run: {terms_accepted_by: {person: Jane Example}}\n"),
        ("quoted.yaml", '"terms_accepted_by": "Jane Example"\n'),
        ("list.yaml", "- terms_accepted_by: Jane Example\n"),
        ("run.toml", 'terms_accepted_by = "Jane Example"\n'),
        ("table.toml", '[terms_accepted_by]\nperson = "Jane Example"\n'),
        ("dotted.toml", '[run]\nterms_accepted_by.person = "Jane Example"\n'),
        ("broken.toml", "terms_accepted_by = \n"),
        ("src/pkg/a.py", 'cfg = {"terms_accepted_by": who}\n'),
        ("src/pkg/b.py", 'cfg["terms_accepted_by"] = who\n'),
        ("src/pkg/c.py", 'cfg["terms_accepted_by"] += extra\n'),
        ("src/pkg/d.py", 'del cfg["terms_accepted_by"]\n'),
        ("src/pkg/e.py", "cfg.update(terms_accepted_by=who)\n"),
        ("src/pkg/f.py", 'cfg.setdefault("terms_accepted_by", who)\n'),
        ("src/pkg/g.py", 'cfg.__setitem__("terms_accepted_by", who)\n'),
        ("src/pkg/h.py", 'setattr(cfg, "terms_accepted_by", who)\n'),
        ("src/pkg/i.py", "cfg.terms_accepted_by = who\n"),
        ("src/pkg/j.py", "make(**{'terms_accepted_by': who})\n"),
        ("src/pkg/k.py", 'x = "terms_accepted_by" (\n'),
        # Escaped or concatenated keys: the raw text never contains the key, the parsed document does.
        ("escaped.json", '{"inputs": [], "terms_accepted\\u005fby": {"person": "An Agent"}}'),
        ("escaped.ipynb", '{"metadata": {"\\u0074erms_accepted_by": "Jane Example"}}'),
        ("escaped.toml", '"terms_accepted\\u005fby" = "Jane Example"\n'),
        ("escaped.yaml", '"terms\\x5faccepted_by": Jane Example\n'),
        ("escaped_u.yml", '{"terms_accepted\\u005Fby": Jane Example}\n'),
        ("src/pkg/l.py", 'cfg = {"terms\\x5faccepted_by": who}\n'),
        ("src/pkg/n.py", 'cfg = {"terms_" "accepted_by": who}\n'),
        ("src/pkg/o.py", 'cfg["terms_" "accepted_by"] = who\n'),
        # Quoted null is the string "null" in YAML.
        ("quoted-null.yaml", "terms_accepted_by: 'null'\n"),
        ("dquoted-null.yaml", 'terms_accepted_by: "null"\n'),
        ("explicit.yaml", "? terms_accepted_by\n: Jane Example\n"),
        # The gate reads a run configuration of any name.
        ("run.cfg", '{"inputs": [], "terms_accepted_by": {"person": "Jane Example"}}'),
        ("run-config", '{"terms_accepted\\u005fby": {"person": "Jane Example"}}'),
        # A run configuration's whole text is JSON whatever its suffix: the JSON reading is checked as well.
        ("tests/run_config.py", '{"inputs": [], "terms_accepted_by": {"person": "An Agent"}}'),
        ("tools/run.toml", '{"inputs": [], "terms_accepted\\u005fby": {"person": "An Agent"}}'),
        ("run-newline.yaml", '{"inputs": [], "terms_accepted_by"\n  : {"person": "An Agent"}}'),
        ("src/pkg/run_config.py",
         '{"inputs": [], "x": ' + "[" * 205 + "]" * 205 + ', "terms_accepted\\u005fby": {"person": "An Agent"}}'),
    ],
)
def test_scanner_flags_every_way_of_setting_the_key(tmp_path, rel, text):
    problems = _scan_one(tmp_path, rel, text)
    assert problems and all(p.startswith(f"{rel}:") for p in problems), problems


@pytest.mark.parametrize(
    "rel, text",
    [
        ("template.json", '{"inputs": [], "terms_accepted_by": null}'),
        ("escaped-null.json", '{"terms_accepted\\u005fby": null}'),
        ("broken-elsewhere.json", '{"inputs": ['),
        ("other.yaml", '"terms_accepted\\u005fbx": Jane Example\nnote: "terms_accepted_by is the person\'s own"\n'),
        ("README.md", 'Set "terms_accepted_by": {"person": "Jane Example"} yourself.\n'),
        ("data.cfg", "[run]\nterms_accepted_by = Jane Example\n"),
        ("list.cfg", '["terms_accepted_by"]'),
        ("tools/x.py", 'cfg = {"terms_accepted_by": who}\n'),
        ("other.json", '{"note": "terms_accepted_by is set by the person only", "my_terms_accepted_by": "x"}'),
        ("null.yaml", "terms_accepted_by: null\nother: ~\n"),
        ("tilde.yaml", "terms_accepted_by: ~  # the person fills this in\n"),
        ("prefixed.yaml", "my_terms_accepted_by: Jane Example\n"),
        ("explicit-other.yaml", "? my_terms_accepted_by\n: Jane Example\n? other\n: terms_accepted_by\n"),
        ("clean.toml", '[run]\nnote = "the person sets terms_accepted_by"\n'),
        ("src/pkg/read.py", 'acc = cfg.get("terms_accepted_by")\nx = cfg["terms_accepted_by"]\n'
                            'ok = "terms_accepted_by" in cfg\nmsg = f"no terms_accepted_by in {cfg}"\n'),
        ("tests/test_x.py", 'cfg = {"terms_accepted_by": {"person": "Jane Example"}}\n'),
        ("docs/licence.md", '"terms_accepted_by": {"person": "Jane Example"}\n'),
        ("notes.txt", "terms_accepted_by: Jane Example\n"),
    ],
)
def test_scanner_accepts_reads_nulls_tests_and_prose(tmp_path, rel, text):
    assert _scan_one(tmp_path, rel, text) == []


def test_scanner_reports_the_line(tmp_path):
    problems = _scan_one(tmp_path, "src/pkg/m.py", 'a = 1\nb = 2\ncfg["terms_accepted_by"] = who\n')
    assert problems == ["src/pkg/m.py:3: a subscript assigns or deletes it"]
    problems = _scan_one(tmp_path, "cfg.yaml", "a: 1\nterms_accepted_by: Jane\n")
    assert problems == ["cfg.yaml:2: sets it to 'Jane' (only null is allowed)"]


def test_tracked_files_come_from_git_and_untracked_files_are_ignored(tmp_path):
    if _git_missing():
        pytest.skip("git is not installed")
    assert _git(tmp_path, "init", "-q").returncode == 0
    (tmp_path / "committed.json").write_text('{"terms_accepted_by": {"person": "Jane Example"}}', encoding="utf-8")
    (tmp_path / "local-run.json").write_text('{"terms_accepted_by": {"person": "Jane Example"}}', encoding="utf-8")
    assert _git(tmp_path, "add", "committed.json").returncode == 0
    files = tracked_files(tmp_path)
    assert files == ["committed.json"]
    assert scan(tmp_path, files) == ["committed.json:1: sets it to {'person': 'Jane Example'}"]
    sub = tmp_path / "sub"
    sub.mkdir()
    assert tracked_files(sub) is None  # inside a checkout, but not its top: refuse to guess


def _git_missing() -> bool:
    try:
        subprocess.run(["git", "--version"], capture_output=True, timeout=30)
    except FileNotFoundError:
        return True
    return False
