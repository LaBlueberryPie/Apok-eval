"""Every reader of a user-named file reads it through apok_eval.fileio: a FIFO is refused without blocking, a device
is never read until memory runs out, and the refusal names what was found (missing, its kind, not a path).

Each read runs in a thread with a deadline, so a reader that opens a FIFO fails the test instead of hanging it.
"""

from __future__ import annotations

import errno
import json
import os
import threading

import numpy as np
import pytest

from apok_eval.cli import main
from apok_eval.io import tifxyz as T
from apok_eval.io.mesh import read_mesh
from apok_eval.io.pointcollection import read_point_collections
from apok_eval.io.umbilicus import read_umbilicus
from apok_eval.licence.gate import load_run_config
from apok_eval.licence.registry import load_registry
from apok_eval.refusal import Refusal

DEVICE = "/dev/zero"  # a character device that never ends
pytestmark = pytest.mark.skipif(not (hasattr(os, "mkfifo") and os.path.exists(DEVICE)),
                                reason="needs FIFOs and /dev/zero")


def refusal_within(fn, *args, fifo=None, **kwargs) -> Refusal:
    """The Refusal ``fn`` raises, within 10 s; fails (after releasing a blocked FIFO reader) otherwise."""
    outcome: list = []

    def call():
        try:
            fn(*args, **kwargs)
            outcome.append(None)
        except BaseException as exc:  # noqa: BLE001 - inspected below
            outcome.append(exc)

    worker = threading.Thread(target=call, daemon=True)
    worker.start()
    worker.join(10)
    if worker.is_alive():
        if fifo is not None:
            os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
        pytest.fail(f"{getattr(fn, '__name__', fn)} blocked or kept reading")
    assert isinstance(outcome[0], Refusal), outcome[0]
    return outcome[0]


@pytest.fixture
def nodes(tmp_path):
    """A FIFO and a link to a character device under each name a reader may be given."""
    def make(name: str) -> tuple[str, str]:
        fifo, device = tmp_path / f"fifo-{name}", tmp_path / f"device-{name}"
        os.mkfifo(fifo)
        os.symlink(DEVICE, device)
        return str(fifo), str(device)
    return make


@pytest.mark.parametrize("suffix", [".ply", ".obj"])
def test_a_mesh_that_is_no_regular_file_is_refused_naming_its_kind(nodes, tmp_path, suffix):
    fifo, device = nodes("m" + suffix)
    exc = refusal_within(read_mesh, fifo, fifo=fifo)
    assert exc.code == "IO_MESH_FILE_MISSING" and exc.details == {"path": fifo, "problem": "not a regular file (a FIFO)"}
    exc = refusal_within(read_mesh, device)
    assert exc.details == {"path": device, "problem": "not a regular file (a character device)"}
    missing = str(tmp_path / ("missing" + suffix))
    exc = refusal_within(read_mesh, missing)
    assert exc.code == "IO_MESH_FILE_MISSING" and exc.details == {"path": missing, "problem": "missing"}
    (tmp_path / ("dir" + suffix)).mkdir()
    exc = refusal_within(read_mesh, str(tmp_path / ("dir" + suffix)))
    assert exc.details["problem"] == "not a regular file (a directory)"


def test_a_mesh_path_holding_nul_is_not_a_path_not_an_unknown_extension(tmp_path):
    name = str(tmp_path / "m.ply\x00x")
    exc = refusal_within(read_mesh, name)
    assert exc.code == "IO_MESH_FILE_MISSING" and exc.details == {"path": name, "problem": "not a valid path"}


def test_point_collections_and_umbilicus_name_the_kind_found(nodes, tmp_path):
    fifo, device = nodes("p.json")
    for read, code, kwargs in ((read_point_collections, "IO_PCL_FILE_MISSING", {"role": None}),
                               (read_umbilicus, "IO_UMBILICUS_FILE_MISSING", {})):
        exc = refusal_within(read, fifo, fifo=fifo, **kwargs)
        assert exc.code == code and exc.details == {"path": fifo, "problem": "not a regular file (a FIFO)"}
        assert "a FIFO, not a regular file" in exc.reason and "does not exist" not in exc.reason
        exc = refusal_within(read, device, **kwargs)
        assert exc.details["problem"] == "not a regular file (a character device)"
        name = str(tmp_path / "u\x00.json")
        exc = refusal_within(read, name, **kwargs)
        assert exc.code == code and exc.details == {"path": name, "problem": "not a valid path"}


def _surface(tmp_path) -> str:
    plane = np.ones((2, 3), "<f4")
    T.write_tifxyz(tmp_path / "s", x=plane, y=plane, z=plane,
                   meta={"format": "tifxyz", "type": "seg", "uuid": "s", "scale": [1.0, 1.0]})
    return str(tmp_path / "s")


def test_a_tifxyz_core_file_that_is_a_fifo_is_refused_naming_it(tmp_path):
    surface = _surface(tmp_path)
    os.remove(os.path.join(surface, "z.tif"))
    os.mkfifo(os.path.join(surface, "z.tif"))
    exc = refusal_within(T.read_tifxyz, surface, fifo=os.path.join(surface, "z.tif"))
    assert exc.code == "IO_TIFXYZ_FILE_KIND"
    assert exc.details == {"file": "z.tif", "path": surface, "kind": "a FIFO"}


def test_an_unreadable_entry_the_surface_does_not_need_is_recorded_not_fatal(tmp_path, monkeypatch):
    surface = _surface(tmp_path)
    with open(os.path.join(surface, "notes.txt"), "w") as fh:
        fh.write("x")
    real = T.digest_and_size

    def denied(path):
        if path.endswith("notes.txt"):
            raise PermissionError(errno.EACCES, "Permission denied", path)
        return real(path)

    monkeypatch.setattr(T, "digest_and_size", denied)
    read = T.read_tifxyz(surface)
    notes = [f for f in read.files if f["name"] == "notes.txt"]
    assert notes == [{"name": "notes.txt", "kind": "file", "unreadable": "PermissionError: Permission denied"}]
    assert {"finding": "entry-unreadable", "name": "notes.txt", "error": "PermissionError: Permission denied"} \
        in read.findings


def _inspect(capsys, *argv) -> tuple[int, dict]:
    rc = main(["io", "inspect", *map(str, argv)])
    out = capsys.readouterr().out
    return rc, json.loads(out) if rc == 3 else out


@pytest.mark.skipif(not os.path.isfile("/proc/self/mem"), reason="needs Linux /proc/self/mem")
def test_io_inspect_refuses_a_mesh_whose_read_fails(tmp_path, capsys):
    mesh = tmp_path / "procmem.ply"
    os.symlink("/proc/self/mem", mesh)  # a regular file whose read at offset 0 fails with EIO
    rc, doc = _inspect(capsys, mesh)
    assert rc == 3 and doc["code"] == "IO_INSPECT_UNKNOWN"
    assert doc["details"] == {"path": str(mesh), "problem": "unreadable", "error": "OSError", "errno": errno.EIO}


def test_io_inspect_refuses_a_frame_that_is_a_fifo_or_a_device(nodes, tmp_path, capsys):
    from apok_eval.io.mesh import TriangleMesh, write_obj

    mesh = tmp_path / "tri.obj"
    write_obj(mesh, TriangleMesh(vertices_xyz=np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]], "<f8"),
                                 faces=np.array([[0, 1, 2]], "<i4")))
    fifo, device = nodes("frame.json")
    for frame in (fifo, device):
        outcome: list = []
        worker = threading.Thread(target=lambda f=frame: outcome.append(_inspect(capsys, mesh, "--frame", f)),
                                  daemon=True)
        worker.start()
        worker.join(10)
        if worker.is_alive():
            os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
            pytest.fail(f"--frame {frame} blocked or kept reading")
        rc, doc = outcome[0]
        assert rc == 3 and doc["code"] == "IO_INSPECT_FRAME_INVALID" and doc["details"]["path"] == frame
        assert "not a regular file" in doc["details"]["problem"]


def test_the_licence_gate_refuses_a_run_configuration_or_registry_that_is_no_regular_file(nodes):
    fifo, device = nodes("run.json")
    for path in (fifo, device):
        exc = refusal_within(load_run_config, path, fifo=fifo)
        assert exc.code == "LICENCE_RUN_UNREADABLE" and exc.details["path"] == path
        assert "not a regular file" in exc.details["problem"]
        exc = refusal_within(load_registry, path, fifo=fifo)
        assert exc.code == "LICENCE_REGISTRY_UNREADABLE" and exc.details["source"] == path
        assert "not a regular file" in exc.details["problem"]


@pytest.mark.parametrize("reader", ["read_obj", "read_ply"])
def test_the_format_readers_refuse_a_path_holding_nul_themselves(tmp_path, reader):
    from apok_eval.io import mesh

    name = str(tmp_path / "m\x00.obj")
    exc = refusal_within(getattr(mesh, reader), name)
    assert exc.code == "IO_MESH_FILE_MISSING" and exc.details == {"path": name, "problem": "not a valid path"}
