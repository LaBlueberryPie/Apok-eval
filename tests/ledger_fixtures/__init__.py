"""Synthetic fixtures for the ledger tests: a git repository holding a small
package source, and a realistic sealed run (inputs, a remote-only input, a
model, a chunk manifest with present and absent entries including metadata
objects, seeds, outputs, figures and pre-registered predictions).

Everything is generated under a pytest ``tmp_path``; nothing touches the
network or the repository the tests run from.
"""

from __future__ import annotations

import contextlib
import json
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from apok_eval.canonical import canonical_dumps, sha256_bytes
from apok_eval.ledger import open_run
from apok_eval.ledger.record import compute_seal
from apok_eval.refusal import Refusal

CONFIG = {
    "fit": {"optimizer_random_seed": 7, "z_begin": 100, "z_end": 300, "input_use_winding_inference": False,
            "dense_spacing_mode": "grad_mag", "loss_weight_dense_spacing": 12.0},
    "eval": {"seed": 11, "alpha": 0.1, "folds": ["a", "b"]},
}
FIGURES = {"alpha": 0.1, "achieved_rate": 0.93, "n": 120, "bound": 0.31, "certified": True, "label": "fold-a"}
METRICS = {"summary": FIGURES, "per_fold": [{"fold": "a", "n": 60}, {"fold": "b", "n": 60}]}
REGISTRY = {"predictions": [{"id": "p-001", "claim": "rate >= 0.9"}, {"id": "p-002", "claim": "n >= 100"}]}
REMOTE_URI = "https://example.invalid/synthetic/volume.zarr/.zattrs"
STORE_URI = "https://example.invalid/synthetic/volume.zarr"


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=apok-test", "-c", "user.email=test@example.invalid",
                           "-c", "commit.gpgsign=false", *args], check=True, capture_output=True,
                          text=True).stdout.strip()


@dataclass
class SourceRepo:
    repo: Path
    source_dir: Path
    commits: list[str]


def make_source_repo(base: Path) -> SourceRepo:
    """A git repository with a package at ``src/apok`` and two commits that differ in ``core.py``."""
    repo = base / "repo"
    src = repo / "src" / "apok"
    (src / "sub").mkdir(parents=True)
    (src / "data").mkdir()
    (src / "__init__.py").write_text('__version__ = "9.9.9"\n')
    (src / "core.py").write_text("ANSWER = 41\n")
    (src / "sub" / "__init__.py").write_text("")
    (src / "sub" / "mod.py").write_text("def f():\n    return 'x'\n")
    (src / "data" / "table.json").write_text('{"k": 1}\n')
    (repo / ".gitignore").write_text("__pycache__/\n")
    (repo / "README").write_text("synthetic\n")
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "first")
    first = git(repo, "rev-parse", "HEAD")
    (src / "core.py").write_text("ANSWER = 42\n")
    git(repo, "commit", "-q", "-am", "second")
    second = git(repo, "rev-parse", "HEAD")
    (src / "__pycache__").mkdir()
    (src / "__pycache__" / "core.cpython-313.pyc").write_bytes(b"\x00cache")
    return SourceRepo(repo, src, [first, second])


def make_dirty_source_repo(base: Path) -> SourceRepo:
    """:func:`make_source_repo`, then dirtied: a tracked file changed inside the package and outside it, and an
    untracked file and an untracked empty file added inside the package."""
    source = make_source_repo(base)
    (source.repo / "README").write_text("synthetic, edited outside the package\n")
    (source.source_dir / "core.py").write_text("ANSWER = 44  # uncommitted\n")
    (source.source_dir / "sub" / "untracked.py").write_text("NEW = 1\n")
    (source.source_dir / "sub" / "empty.py").write_text("")
    return source


def no_git_above(path: Path) -> bool:
    """Whether no ``.git`` entry exists in ``path`` or any parent (a precondition of the no-checkout tests)."""
    return not any((directory / ".git").exists() or (directory / ".git").is_symlink()
                   for directory in (path, *path.parents))


def chunk_entries() -> list[dict]:
    """Manifest entries of a two-level synthetic store: metadata objects, chunks, one absent chunk."""
    def present(key: str, data: bytes) -> dict:
        return {"key": key, "digest": sha256_bytes(data), "size": len(data)}

    return [
        present("volume.zarr/1/0.0.0", b"level-1 chunk"),
        present("volume.zarr/.zgroup", b'{"zarr_format":2}'),
        present("volume.zarr/.zattrs", b'{"multiscales":[]}'),
        present("volume.zarr/0/.zarray", b'{"chunks":[2,2,2]}'),
        present("volume.zarr/0/0.0.0", bytes(range(64))),
        {"key": "volume.zarr/0/0.0.1", "absent": True},
        present("volume.zarr/1/.zarray", b'{"chunks":[1,1,1]}'),
        {"key": "volume.zarr/0/1.0.0", "absent": True},
    ]


@dataclass
class SyntheticRun:
    run_dir: Path
    record_path: Path
    data_dir: Path
    seal: str
    source: SourceRepo | None
    index: Path | None
    files: dict[str, Path] = field(default_factory=dict)

    def record(self) -> dict:
        return json.loads(self.record_path.read_bytes())


def write_inputs(data_dir: Path) -> dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "volume": data_dir / "volume.bin",
        "labels": data_dir / "labels.json",
        "predictions": data_dir / "predictions.json",
        "model": data_dir / "model.pt",
    }
    files["volume"].write_bytes(bytes(range(256)) * 4)
    files["labels"].write_text(json.dumps({"patches": ["p1", "p2"]}))
    files["predictions"].write_text(json.dumps(REGISTRY, indent=1))
    files["model"].write_bytes(b"PK\x03\x04synthetic-model" + bytes(100))
    return files


def build_run(base: Path, source: SourceRepo | Path, *, run_id: str = "fold-a", index: Path | None = None,
              allow_dirty: bool = False, data_dir: Path | None = None) -> SyntheticRun:
    """Open, fill and seal a realistic run under ``base/runs/<run_id>``."""
    source_dir = source.source_dir if isinstance(source, SourceRepo) else source
    data = data_dir or base / "data"
    files = write_inputs(data) if not (data / "volume.bin").exists() else {
        "volume": data / "volume.bin", "labels": data / "labels.json", "predictions": data / "predictions.json",
        "model": data / "model.pt"}
    run = open_run(base / "runs" / run_id, run_id=run_id, purpose=f"synthetic held-out run {run_id}",
                   config=CONFIG, expected_bytes=1 << 20, index=index, allow_dirty=allow_dirty,
                   source_dir=source_dir)
    run.add_input("volume", files["volume"], uri="file:///synthetic/volume.bin")
    run.add_input("labels", files["labels"])
    run.add_input("predictions", files["predictions"])
    run.add_remote_input("remote-meta", b'{"multiscales":[]}', uri=REMOTE_URI)
    run.add_chunk_manifest("volume", chunk_entries(), source_uri=STORE_URI)
    run.add_model("winding", files["model"], source="synthetic; trained by no one", uri="file:///models/w.pt")
    run.set_seed("optimizer", 7, config_pointer="/fit/optimizer_random_seed")
    run.set_seed("eval", 11, config_pointer="/eval/seed")
    run.set_predictions("predictions", ["p-002", "p-001"])
    (run.run_dir / "out").mkdir(exist_ok=True)
    (run.run_dir / "metrics.json").write_text(json.dumps(METRICS, indent=2))
    (run.run_dir / "out" / "mesh.bin").write_bytes(b"mesh" * 50)
    run.add_output("metrics", "metrics.json")
    run.add_output("mesh", run.run_dir / "out" / "mesh.bin")
    run.set_figures("metrics", pointer="/summary")
    seal = run.seal()
    outputs = {"metrics": run.run_dir / "metrics.json", "mesh": run.run_dir / "out" / "mesh.bin",
               "sidecar": run.run_dir / "manifests" / "volume.jsonl"}
    return SyntheticRun(run.run_dir, run.record_path, data, seal,
                        source if isinstance(source, SourceRepo) else None, index, {**files, **outputs})


def write_record(path: Path, record: dict, *, reseal: bool) -> dict:
    """Write ``record`` canonically, recomputing its seal first when ``reseal``."""
    record = dict(record)
    if reseal:
        record["seal"] = compute_seal(record)
    path.write_bytes(canonical_dumps(record) + b"\n")
    return record


@contextlib.contextmanager
def deadline(seconds: int = 10):
    """Turn a call that would block forever (a FIFO opened for reading) into a ``TimeoutError``."""
    def expired(signum, frame):
        raise TimeoutError(f"blocked for more than {seconds} s")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def failures(refusal: Refusal) -> dict[str, dict]:
    """check name -> failed check, from a LEDGER_VERIFY_FAILED refusal."""
    assert refusal.code == "LEDGER_VERIFY_FAILED", refusal.code
    return {f["check"]: f for f in refusal.details["failed"]}


def failed_codes(refusal: Refusal) -> set[tuple[str, str]]:
    return {(name, f["code"]) for name, f in failures(refusal).items()}


def status_of(report: dict, check: str) -> dict:
    return next(c for c in report["checks"] if c["check"] == check)
