"""No tracked file and no commit message links to a private working session or workspace.

Links to private chat sessions and note-taking workspaces (the hosts in ``PRIVATE_LINK``) open only for their owners,
but they name private records, so they never belong in this repository: not in a file, not in a commit message.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PRIVATE_LINK = re.compile(r"claude\.ai/code/session|claude\.ai/chat/|notion\.so/|notion\.site/|app\.notion\.com/",
                          re.IGNORECASE)


def _git(*args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return subprocess.run(["git", *args], cwd=ROOT, env=env, capture_output=True, timeout=120)


def _is_checkout() -> bool:
    top = _git("rev-parse", "--show-toplevel")
    return top.returncode == 0 and Path(top.stdout.decode().strip()).resolve() == ROOT.resolve()


def private_links(text: str) -> list[int]:
    """1-based numbers of the lines of ``text`` that hold a private link."""
    return [n for n, line in enumerate(text.splitlines(), 1) if PRIVATE_LINK.search(line)]


def test_the_pattern():
    assert private_links("see https://claude.ai/code/session_ABC\nok\nhttps://www.notion.so/x") == [1, 3]
    assert private_links("https://scrollprize.org and https://claude.ai/") == []


def test_no_tracked_file_holds_a_private_link():
    if not _is_checkout():
        pytest.skip("not a git checkout")
    pattern = PRIVATE_LINK.pattern.replace("\\", "")
    for path in filter(None, _git("ls-files", "-z").stdout.decode().split("\0")):
        if path == "tests/test_no_private_links.py":
            continue  # this file names the patterns it looks for
        try:
            text = (ROOT / path).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        assert not private_links(text), f"{path} links to a private session or workspace ({pattern})"


def test_no_commit_message_holds_a_private_link():
    if not _is_checkout():
        pytest.skip("not a git checkout")
    for commit in _git("rev-list", "--all").stdout.decode().split():
        message = _git("log", "-1", "--format=%B", commit).stdout.decode("utf-8", "replace")
        assert not private_links(message), f"commit {commit[:12]}'s message links to a private session or workspace"
