"""E7 v3 on local files: a fit's tifxyz segments, VC3D fibres, an umbilicus, and optionally the CT.

This is what ``apok-eval e7v3`` and ``apok-eval demo`` run. Every input is read once, through the readers of
:mod:`apok_eval.io`, and every byte read is listed in the run's sealed record (:mod:`apok_eval.ledger`), so
``apok-eval verify`` can re-read the inputs and recompute every digest. The umbilicus is the one input the
record lists by path: :func:`seal_run` reads it again and refuses (``EVAL_RUN_INPUT_CHANGED``) unless its digest
and size are those of the bytes the run parsed, which the result keeps.

- **Frame.** The spiral frame's shape (z, y, x) is declared: ``frame_shape``, or the shape of the CT level
  given, or both, which must then agree (``EVAL_RUN_FRAME``). Fibre coordinates are scaled to it
  (:func:`apok_eval.eval.fibres.read_fibre`) and every point is checked against it.
- **Segments.** Each subdirectory of the segments directory holding an ``x.tif``, ``y.tif`` or ``z.tif`` is one
  tifxyz segment, valid under the declared validity ``policy``. Its name must end in ``-wLLL-HHH`` (its winding
  range); a ``windings`` map ``{directory name: [first, last]}`` names segments that do not
  (``EVAL_RUN_WINDINGS``). A segment with a valid vertex more than one voxel outside the frame is refused
  (``EVAL_RUN_SEGMENT_OUTSIDE_FRAME``).
- **Fibres.** Every ``.json`` file directly in the fibres directory, read through a logged local store. A file
  that is refused is counted by its refusal code and left out; a run with no fibre read is refused
  (``EVAL_RUN_FIBRES``). Families are villa's H / V / untagged (:func:`apok_eval.eval.fibres.hv_family`).
- **CT positive control** (with a CT volume): :func:`apok_eval.eval.ctprofile.diagnose` on the same slab, before
  E7 v3. CONFIRMED when the fibres are LOCKED and the null NOT LOCKED; REFUTED when the fibres are NOT LOCKED
  (with a NOT LOCKED null); INCONCLUSIVE otherwise. When it is not CONFIRMED, E7 v3 is NOT MEASURED. Without a
  volume the control is NOT RUN and E7 v3 is scored, and the result says so.
- **E7 v3:** :func:`apok_eval.eval.faces.diagnose`.
- **Leakage audit** (with the fit's input patches): :func:`apok_eval.eval.leakage.audit`, then E7 v3 again on
  the fibre points farther than ``leakage_radius`` voxels from every patch.
- **Licence gate** (with a registry): every input must lie under a location of an asset the registry declares
  (``EVAL_RUN_UNDECLARED``) and the run must clear :func:`apok_eval.licence.check_run`, with the terms
  fragment given (the person's own ``terms_accepted_by``), before anything is read. Without a registry nothing
  is checked, and the record says so.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np

from apok_eval import jsonstrict
from apok_eval.eval import ctprofile, faces, leakage
from apok_eval.eval.ctprofile import CTConfig
from apok_eval.eval.faces import FaceConfig
from apok_eval.eval.fibres import hv_family, read_fibre
from apok_eval.eval.fit import SegmentGrid, WindingFit, winding_range
from apok_eval.eval.registration import RegConfig
from apok_eval.fileio import read_regular
from apok_eval.io.omezarr import chunk_codec, decode_chunk
from apok_eval.io.readlog import LoggedStore, ReadLog
from apok_eval.io.store import LocalStore
from apok_eval.io.tifxyz import COORDINATE_FILES, MASK_FILE, META_FILE, POLICIES, read_tifxyz, validity
from apok_eval.io.umbilicus import read_umbilicus
from apok_eval.refusal import Refusal, register_codes

register_codes(
    "apok_eval.eval.local",
    {
        "EVAL_RUN_INPUT": "a local input of an E7 v3 run is missing or is not what the run needs",
        "EVAL_RUN_FRAME": "the frame shape is not declared, or the declared frame and the CT level disagree",
        "EVAL_RUN_WINDINGS": "a windings map is malformed or names a segment directory that is not there",
        "EVAL_RUN_SEGMENT_OUTSIDE_FRAME": "a segment's valid vertices lie outside the declared frame by more than "
                                          "one voxel",
        "EVAL_RUN_FIBRES": "no fibre file of the run could be read",
        "EVAL_RUN_SETTINGS": "a settings file does not hold the E7 v3 run's settings",
        "EVAL_RUN_VOLUME": "the CT level is not a C-ordered unsigned 8- or 16-bit array without filters",
        "EVAL_RUN_UNDECLARED": "a local input lies under no location of any asset the registry declares",
        "EVAL_RUN_INPUT_CHANGED": "an input's bytes changed between the run's read and its record",
    },
)

SCHEMA = "apok-eval-e7v3-v1"
CONTROL_RULE = ("CONFIRMED when the fibres are LOCKED and the null NOT LOCKED; REFUTED when the fibres are NOT "
                "LOCKED with a NOT LOCKED null; INCONCLUSIVE otherwise; E7 v3 is NOT MEASURED unless CONFIRMED")
DTYPES = {"|u1": np.uint8, "<u1": np.uint8, "<u2": np.uint16, "|u2": np.uint16}
SEGMENT_FILES = (META_FILE, *COORDINATE_FILES, MASK_FILE)


@dataclass(frozen=True)
class Settings:
    """Every setting of a run. ``z_range`` is the slab; the rest default to the evaluator's fixed rule."""

    z_range: tuple[float, float]
    registration: dict = field(default_factory=dict)  # RegConfig fields other than z_range
    face: dict = field(default_factory=dict)  # FaceConfig fields
    ct: dict = field(default_factory=dict)  # CTConfig fields other than z_range
    margin: float = 128.0
    leakage_radius: float = 2.0

    def reg_config(self) -> RegConfig:
        return _build(RegConfig, {**self.registration, "z_range": self.z_range}, "registration")

    def face_config(self) -> FaceConfig:
        return _build(FaceConfig, self.face, "face")

    def ct_config(self) -> CTConfig:
        return _build(CTConfig, {**self.ct, "z_range": self.z_range}, "ct")

    def check(self) -> None:
        self.reg_config().check()
        self.face_config().check()
        self.ct_config().check()
        if not (self.margin >= 0 and self.leakage_radius > 0):
            raise Refusal("EVAL_RUN_SETTINGS", "margin must be >= 0 and leakage_radius > 0",
                          margin=self.margin, leakage_radius=self.leakage_radius)

    def as_dict(self) -> dict:
        """Every value in force, the defaults included, as canonical JSON (what the record keeps)."""
        reg, face, ct = (_plain(asdict(c)) for c in (self.reg_config(), self.face_config(), self.ct_config()))
        return {"z_range": [float(v) for v in self.z_range], "registration": reg, "face": face, "ct": ct,
                "margin": float(self.margin), "leakage_radius": float(self.leakage_radius)}


_TUPLES = {"z_range", "z_offset", "rotation", "sigmas", "period_range"}


def _build(cls, values: dict, where: str):
    names = {f.name for f in fields(cls)}
    unknown = sorted(set(values) - names)
    if unknown:
        raise Refusal("EVAL_RUN_SETTINGS", f"{where} settings name {unknown}, which are not {cls.__name__}'s "
                      f"({sorted(names)})", section=where, unknown=unknown)
    out = {}
    for key, value in values.items():
        if key == "bands":
            value = tuple(tuple(float(v) for v in band) for band in value)
        elif key in _TUPLES and isinstance(value, (list, tuple)):
            value = tuple(value)
        out[key] = value
    return cls(**out)


def _plain(value):
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def load_settings(path: str | os.PathLike[str] | None, z_range: tuple[float, float]) -> Settings:
    """Settings from a strict-JSON file with any of ``registration``, ``face``, ``ct``, ``margin`` and
    ``leakage_radius`` (``z_range`` comes from the command line); the defaults without a file."""
    if path is None:
        settings = Settings(z_range=z_range)
    else:
        try:
            doc = jsonstrict.loads(read_regular(path))
        except (OSError, jsonstrict.StrictJsonError) as exc:
            raise Refusal("EVAL_RUN_SETTINGS", f"settings file {os.fspath(path)!r} is not readable strict JSON: "
                          f"{exc}", path=os.fspath(path), problem=str(exc)) from None
        allowed = {"registration", "face", "ct", "margin", "leakage_radius"}
        if not isinstance(doc, dict) or set(doc) - allowed:
            raise Refusal("EVAL_RUN_SETTINGS", f"settings file {os.fspath(path)!r} must be an object with only "
                          f"{sorted(allowed)}", path=os.fspath(path),
                          unknown=sorted(set(doc) - allowed) if isinstance(doc, dict) else None)
        for section in ("registration", "face", "ct"):
            if not isinstance(doc.get(section, {}), dict):
                raise Refusal("EVAL_RUN_SETTINGS", f"settings '{section}' must be an object", section=section)
        if "z_range" in doc.get("registration", {}) or "z_range" in doc.get("ct", {}):
            raise Refusal("EVAL_RUN_SETTINGS", "the slab is given on the command line (--z0, --z1), not in the "
                          "settings file", path=os.fspath(path))
        settings = Settings(z_range=z_range, registration=doc.get("registration", {}), face=doc.get("face", {}),
                            ct=doc.get("ct", {}), margin=float(doc.get("margin", 128.0)),
                            leakage_radius=float(doc.get("leakage_radius", 2.0)))
    try:
        settings.check()
    except TypeError as exc:  # a value of the wrong type reached a dataclass check
        raise Refusal("EVAL_RUN_SETTINGS", f"a setting has the wrong type: {exc}", problem=str(exc)) from None
    return settings


# --- inputs ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Inputs:
    """The local inputs of one run (paths as given)."""

    segments: str
    fibres: str
    umbilicus: str
    umbilicus_variant: str = "fitter"
    policy: str = "spiral"
    frame_shape: tuple[int, int, int] | None = None
    windings: dict | None = None
    volume: str | None = None
    level: int = 0
    patches: str | None = None

    def paths(self) -> dict[str, str]:
        out = {"segments": self.segments, "fibres": self.fibres, "umbilicus": self.umbilicus}
        if self.volume is not None:
            out["volume"] = self.volume
        if self.patches is not None:
            out["patches"] = self.patches
        return {k: os.path.abspath(v) for k, v in out.items()}


def _tifxyz_dirs(directory: str, what: str) -> list[Path]:
    root = Path(directory)
    if not root.is_dir():
        raise Refusal("EVAL_RUN_INPUT", f"{what} {directory!r} is not a directory", input=what, path=directory)
    found = sorted(p for p in root.iterdir() if p.is_dir() and any((p / m).exists() for m in COORDINATE_FILES))
    if not found:
        raise Refusal("EVAL_RUN_INPUT", f"{what} {directory!r} holds no tifxyz directory (a subdirectory with "
                      "x.tif, y.tif or z.tif)", input=what, path=directory)
    return found


def _read_entries(surface, prefix: str) -> list[dict]:
    """Manifest entries for the files the tifxyz reader read (their digests are of the bytes it decoded)."""
    return [{"key": f"{prefix}/{f['name']}", "digest": f["digest"], "size": f["size"]}
            for f in surface.files if f["name"] in SEGMENT_FILES and "digest" in f]


def check_windings_map(windings: Any) -> dict[str, tuple[int, int]]:
    """``{directory name: [first, last]}`` with 0 <= first <= last <= 999, else ``EVAL_RUN_WINDINGS``."""
    if not isinstance(windings, dict):
        raise Refusal("EVAL_RUN_WINDINGS", "a windings map must be a JSON object {directory: [first, last]}",
                      value=type(windings).__name__)
    out = {}
    for name, rng in windings.items():
        ok = (isinstance(rng, list) and len(rng) == 2
              and all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 999 for v in rng)
              and rng[0] <= rng[1])
        if not ok:
            raise Refusal("EVAL_RUN_WINDINGS", f"windings[{name!r}] must be [first, last], integers with "
                          f"0 <= first <= last <= 999, not {rng!r}", segment=name, value=repr(rng))
        out[name] = (rng[0], rng[1])
    return out


def read_segments(directory: str, *, policy: str, frame_shape, windings: dict | None = None):
    """The fit's segment grids, the manifest of the files read, and a summary per segment."""
    if policy not in POLICIES:
        raise Refusal("EVAL_RUN_INPUT", f"validity policy {policy!r} is not one of {list(POLICIES)}",
                      input="policy", value=policy)
    dirs = _tifxyz_dirs(directory, "segments")
    names = {d.name for d in dirs}
    windings = check_windings_map(windings) if windings is not None else {}
    missing = sorted(set(windings) - names)
    if missing:
        raise Refusal("EVAL_RUN_WINDINGS", f"the windings map names {missing}, which are not segment directories "
                      f"under {directory!r}", missing=missing)
    shape = np.asarray(frame_shape, dtype=np.float64)
    grids, entries, infos = [], [], []
    for d in dirs:
        surface = read_tifxyz(d)
        name = surface.dir_name
        if name in windings:
            lo, hi = windings[name]
            name = f"{name}-w{lo:03d}-{hi:03d}"
        winding_range(name)  # EVAL_SEGMENT_NAME unless it ends in -wLLL-HHH
        valid = validity(surface, policy).valid
        zyx = surface.zyx.astype(np.float64)
        pts = zyx[valid]
        if len(pts) and ((pts.min(axis=0) < -1).any() or (pts.max(axis=0) > shape + 1).any()):
            raise Refusal("EVAL_RUN_SEGMENT_OUTSIDE_FRAME", f"segment {surface.dir_name}: valid vertices span z,y,x "
                          f"{pts.min(axis=0).tolist()} to {pts.max(axis=0).tolist()}, outside the frame "
                          f"{[int(v) for v in shape]} by more than one voxel", segment=surface.dir_name,
                          frame_shape_zyx=[int(v) for v in shape])
        grids.append(SegmentGrid(name=name, zyx=zyx, valid=valid))
        entries += _read_entries(surface, d.name)
        infos.append({"directory": d.name, "name": name, "shape_hw": list(surface.shape), "valid": int(valid.sum())})
    return grids, entries, infos


def read_fibres(directory: str, *, frame_shape):
    """Every ``.json`` fibre directly in ``directory``; refused files are counted by code and left out."""
    if not Path(directory).is_dir():
        raise Refusal("EVAL_RUN_INPUT", f"fibres {directory!r} is not a directory", input="fibres", path=directory)
    store = LoggedStore(LocalStore(directory), ReadLog())
    names = sorted(e.name for e in os.scandir(directory) if e.name.lower().endswith(".json"))
    fibres, refused, refused_files = [], {}, []
    for name in names:
        try:
            data = store.get(name)
            if data is None:
                raise Refusal("EVAL_RUN_INPUT", f"fibre {name!r} disappeared while being read", input="fibres",
                              path=name)
            fibres.append(read_fibre(data, name=name, base_shape_zyx=list(frame_shape)))
        except Refusal as r:
            refused[r.code] = refused.get(r.code, 0) + 1
            refused_files.append({"name": name, "code": r.code})
    if not fibres:
        raise Refusal("EVAL_RUN_FIBRES", f"none of the {len(names)} .json files in {directory!r} is a fibre the "
                      f"evaluator reads (refusals: {refused})", path=directory, files=len(names), refused=refused)
    info = {"files": len(names), "read": len(fibres), "refused": refused, "refused_files": refused_files}
    return fibres, info, store.log


# --- the CT ------------------------------------------------------------------------------------------------


def open_ct(volume: str, level: int):
    """The CT level's chunk loader, chunk shape, level shape and description, read through a logged store.

    The level is ``multiscales[0].datasets[level].path`` of the volume's ``.zattrs``; its ``.zarray`` must
    describe a C-ordered ``|u1`` or ``<u2`` array without filters, under an allow-listed compressor. Each chunk
    is decoded by :func:`apok_eval.io.omezarr.decode_chunk` and must be full-size (an edge-cropped chunk is
    refused, ``IO_ZARR_CHUNK_SIZE``): a ray can reach past the array's edge, and padding would make up values."""
    store = LoggedStore(LocalStore(volume), ReadLog())

    def json_object(key: str) -> dict:
        data = store.get(key)
        if data is None:
            raise Refusal("EVAL_RUN_VOLUME", f"the volume has no {key}", key=key)
        try:
            doc = jsonstrict.loads(data)
        except jsonstrict.StrictJsonError as exc:
            raise Refusal("EVAL_RUN_VOLUME", f"{key} is not strict JSON: {exc}", key=key, problem=str(exc)) from None
        if not isinstance(doc, dict):
            raise Refusal("EVAL_RUN_VOLUME", f"{key} is not a JSON object", key=key)
        return doc

    zattrs = json_object(".zattrs")
    try:
        path = zattrs["multiscales"][0]["datasets"][int(level)]["path"]
        if not isinstance(path, str):
            raise TypeError(path)
    except (KeyError, IndexError, TypeError, ValueError):
        raise Refusal("EVAL_RUN_VOLUME", f"the volume's .zattrs names no dataset at level {level}",
                      level=level) from None
    za = json_object(f"{path}/.zarray")
    shape, chunks = za.get("shape"), za.get("chunks")
    ok = (isinstance(shape, list) and len(shape) == 3 and isinstance(chunks, list) and len(chunks) == 3
          and all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in shape + chunks)
          and za.get("dtype") in DTYPES and za.get("order", "C") == "C" and not za.get("filters"))
    if not ok:
        raise Refusal("EVAL_RUN_VOLUME", f"level {path} is not a C-ordered unsigned 8- or 16-bit array without "
                      "filters", key=f"{path}/.zarray", shape=shape, dtype=za.get("dtype"))
    compressor = za.get("compressor")
    codec = chunk_codec(compressor, f"{path}/.zarray")  # the allow-list, before any chunk is read
    sep = za.get("dimension_separator", ".")
    if sep not in (".", "/"):
        raise Refusal("EVAL_RUN_VOLUME", f"level {path} declares dimension_separator {sep!r}", key=f"{path}/.zarray")
    dtype = np.dtype(DTYPES[za["dtype"]])
    chunk_shape = tuple(int(c) for c in chunks)

    def load(index):
        key = f"{path}/{sep.join(str(i) for i in index)}"
        data = store.get(key)
        if data is None:
            return None
        return decode_chunk(key, data, compressor=compressor, codec=codec, dtype=dtype, chunks=chunk_shape,
                            edge=chunk_shape)

    described = {"level": int(level), "path": path, "shape_zyx": list(shape), "chunks": list(chunk_shape),
                 "dtype": za["dtype"], "compressor": compressor}
    return load, chunk_shape, tuple(int(v) for v in shape), described, store.log


def control_verdict(ct_verdicts: dict) -> str:
    """The CT positive control's verdict from the CT-profile verdicts (``CONTROL_RULE``)."""
    if ct_verdicts.get("null") != "NOT LOCKED":
        return "INCONCLUSIVE"
    return {"LOCKED": "CONFIRMED", "NOT LOCKED": "REFUTED"}.get(ct_verdicts.get("fibres"), "INCONCLUSIVE")


# --- licence -----------------------------------------------------------------------------------------------


def run_configuration(paths: dict[str, str], registry, terms: dict | None) -> dict:
    """The licence gate's run configuration for local inputs: each path's asset is the declared asset with the
    longest location above it (``EVAL_RUN_UNDECLARED`` when none). ``terms`` is the person's own terms fragment,
    an object holding only their ``terms_accepted_by``; it is passed on as written, never composed here."""
    from apok_eval.licence.registry import load_registry, parse_registry
    from apok_eval.licence.uri import canonical_uri

    reg = parse_registry(registry) if isinstance(registry, dict) else load_registry(registry)
    inputs = []
    for what, path in paths.items():
        uri = canonical_uri(path, what=f"{what} path")
        best = None
        for asset in reg.assets.values():
            for location in asset.locations:
                if uri.is_under(location) and (best is None or len(str(location)) > len(str(best[1]))):
                    best = (asset.id, location)
        if best is None:
            raise Refusal("EVAL_RUN_UNDECLARED", f"{what} {str(uri)!r} lies under no location of any asset the "
                          f"registry ({reg.source}) declares; declare it first (docs/licence.md, \"Declaring your "
                          "data\")", input=what, uri=str(uri), registry=reg.source)
        inputs.append({"asset": best[0], "uri": str(uri)})
    run: dict = dict(terms or {})
    run["inputs"] = inputs
    return run


# --- the run -----------------------------------------------------------------------------------------------


def _family_counts(families) -> dict:
    return {g: sum(f == g for f, _ in families) for g in faces.FAMILIES}


def figures(result: dict) -> dict:
    """The figures a run quotes, all read from ``result`` (the record backs them with its output)."""
    rep = result.get("e7v3") or {}
    verdicts = rep.get("verdicts") or {}
    groups = rep.get("groups") or {}
    ct = (result.get("control") or {}).get("ct") or {}
    lock = ct.get("locking") or {}
    out = {
        "control": (result.get("control") or {}).get("verdict"),
        "e7v3": {k: v.get("e7v3") for k, v in verdicts.items()},
        "Q": {k: ((groups.get(k) or {}).get("observed") or {}).get("Q") for k in ("all", "H", "V", "U")},
        "null": {k: v.get("null") for k, v in verdicts.items()},
        "null_Q_p95": {k: (v.get("null_Q") or {}).get("p95") for k, v in verdicts.items()},
        "offsets_voxels": {g: ((rep.get("families") or {}).get(g) or {}).get("D_voxels") for g in faces.FAMILIES},
        "ceiling": rep.get("ceiling"),
        "fibres": {k: result["fibres"][k] for k in ("files", "read", "refused")},
        "ct": {"period": ct.get("period"), "fibres_R": (lock.get("fibres") or {}).get("R"),
               "fit_R": (lock.get("fit") or {}).get("R"), "null_R": (lock.get("null") or {}).get("R"),
               "verdicts": ct.get("verdicts")} if ct else None,
    }
    far = (result.get("leakage") or {}).get("e7v3_far")
    if far:
        out["leakage"] = {"share_near": result["leakage"]["audit"]["share_near"],
                          "e7v3_far": {k: v.get("e7v3") for k, v in (far.get("verdicts") or {}).items()}}
    return out


def evaluate(inputs: Inputs, settings: Settings, *, registry=None, terms: dict | None = None,
             threads: int = 8, progress=None) -> tuple[dict, dict]:
    """Run the control and E7 v3 on ``inputs``; the JSON-ready result and what the record needs to list.

    ``registry`` (a path, a parsed document or None) turns the licence gate on; ``terms`` is the terms
    fragment of the person running it (:func:`run_configuration`). ``progress(text)`` is told each stage."""
    say = progress or (lambda text: None)
    settings.check()
    paths = inputs.paths()
    licence: dict = {"checked": False, "reason": "no registry given: the inputs were not licence-checked"}
    if registry is not None:
        from apok_eval.licence import check_run

        clearance = check_run(run_configuration(paths, registry, terms), registry)
        licence = {"checked": True, "clearance": clearance.as_dict()}
        say(f"licence gate: {len(clearance.inputs)} inputs cleared")

    ct = None
    if inputs.volume is not None:
        load, chunks, level_shape, described, ct_log = open_ct(inputs.volume, inputs.level)
        ct = (load, chunks, level_shape, described, ct_log)
    frame_shape = inputs.frame_shape
    if frame_shape is None and ct is None:
        raise Refusal("EVAL_RUN_FRAME", "declare the frame: give its shape (--frame-shape Z Y X) or a CT volume and "
                      "level whose shape it is (--volume, --level)")
    if frame_shape is not None:
        if (len(frame_shape) != 3 or not all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) and v > 0
                                             for v in frame_shape)):
            raise Refusal("EVAL_RUN_FRAME", f"the frame shape must be three positive integers, not {frame_shape!r}",
                          frame_shape=repr(frame_shape))
        frame_shape = tuple(int(v) for v in frame_shape)
        if ct is not None and tuple(ct[2]) != frame_shape:
            raise Refusal("EVAL_RUN_FRAME", f"the CT level's shape {list(ct[2])} is not the declared frame "
                          f"{list(frame_shape)}", level_shape=list(ct[2]), frame_shape=list(frame_shape))
    else:
        frame_shape = tuple(ct[2])
    frame = {"shape_zyx": list(frame_shape),
             "source": "declared" if inputs.frame_shape is not None else f"CT level {inputs.level}"}

    say("reading the umbilicus, segments and fibres")
    umb = read_umbilicus(inputs.umbilicus, variant=inputs.umbilicus_variant)
    grids, seg_entries, seg_infos = read_segments(inputs.segments, policy=inputs.policy, frame_shape=frame_shape,
                                                  windings=inputs.windings)
    fibres, fibre_info, fibre_log = read_fibres(inputs.fibres, frame_shape=frame_shape)
    families = [hv_family(f) for f in fibres]
    z_range = tuple(float(v) for v in settings.z_range)
    say(f"building the fit from {len(grids)} segments")
    fit = WindingFit(grids, umb, z_range=z_range, margin=settings.margin)

    result: dict = {
        "schema": SCHEMA, "frame": frame, "slab": list(z_range), "policy": inputs.policy,
        "umbilicus": {"variant": inputs.umbilicus_variant, "digest": umb.digest, "size": umb.size,
                      "z_span": [float(v) for v in umb.z_span]},
        "segments": seg_infos, "fit": fit.report, "fibres": fibre_info,
        "families": _family_counts(families),
        "family_sources": {s: sum(src == s for _, src in families) for s in sorted({s for _, s in families})},
        "licence": licence,
    }
    manifests = {"segments": (seg_entries, Path(paths["segments"]).as_uri()),
                 "fibres": (fibre_log.entries(), Path(paths["fibres"]).as_uri())}

    reg, face = settings.reg_config(), settings.face_config()
    if ct is None:
        result["control"] = {"verdict": "NOT RUN", "rule": CONTROL_RULE,
                             "reason": "no CT volume given: E7 v3 is scored without its positive control"}
        gate = True
    else:
        load, chunks, level_shape, described, ct_log = ct
        say(f"CT positive control on level {described['path']} (chunks {list(chunks)})")
        cache = ctprofile.ChunkCache(chunks, frame_shape, load)
        fam = {f.name: families[i][0] for i, f in enumerate(fibres)}
        with ThreadPoolExecutor(max_workers=max(1, int(threads))) as pool:
            ct_report = ctprofile.diagnose(fit, fibres, cache, config=settings.ct_config(), mapper=pool.map,
                                           families=fam)
        cache.arrays.clear()
        verdict = control_verdict(ct_report.get("verdicts", {}))
        result["control"] = {"verdict": verdict, "rule": CONTROL_RULE, "volume": described, "ct": ct_report}
        manifests["volume"] = (ct_log.entries(), Path(paths["volume"]).as_uri())
        gate = verdict == "CONFIRMED"
        say(f"control {verdict}: fibres {ct_report.get('verdicts', {}).get('fibres')}, "
            f"null {ct_report.get('verdicts', {}).get('null')}, fit {ct_report.get('verdicts', {}).get('fit')}")

    if gate:
        say(f"E7 v3 on {len(fibres)} fibres ({reg.null_draws} z-offset and {reg.rotation_draws} rotation draws)")
        result["e7v3"] = faces.diagnose(fit, fibres, config=reg, face=face, families=families)
    else:
        result["e7v3"] = {"verdicts": {"all": {"verdict": "NOT MEASURED", "e7v3": "NOT MEASURED",
                                               "reason": f"CT positive control {result['control']['verdict']}"}}}

    if inputs.patches is not None:
        say("leakage audit against the fit's input patches")
        patch_dirs = _tifxyz_dirs(inputs.patches, "patches")
        patch_entries: list[dict] = []

        def surfaces():
            for d in patch_dirs:
                s = read_tifxyz(d)
                patch_entries.extend(_read_entries(s, d.name))
                yield s.zyx.astype(np.float64), validity(s, inputs.policy).valid

        audit, masks = leakage.audit(fibres, surfaces(), z_range=z_range, radius=settings.leakage_radius)
        far = leakage.far_fibres(fibres, masks)
        kept = {f.name for f in far}
        far_families = [fam for f, fam in zip(fibres, families) if f.name in kept]
        result["leakage"] = {"audit": audit, "fibres_far": len(far)}
        if gate and far:
            result["leakage"]["e7v3_far"] = faces.diagnose(fit, far, config=reg, face=face, families=far_families)
        manifests["patches"] = (patch_entries, Path(paths["patches"]).as_uri())

    result["figures"] = figures(result)
    return result, {"manifests": manifests, "paths": paths, "settings": settings.as_dict(), "licence": licence}


def seal_run(result: dict, listing: dict, out_dir: str | os.PathLike[str], *, run_id: str, purpose: str,
             allow_dirty: bool = False) -> str:
    """Write ``result`` as ``e7v3.json`` in a new run directory and seal its record; return the seal.

    The umbilicus is recorded from its path, so its bytes are read again here; they must be the bytes the run
    parsed (``result["umbilicus"]``'s digest and size), else ``EVAL_RUN_INPUT_CHANGED`` and nothing is sealed:
    a record never binds bytes the run did not use. Every other input is recorded from the reads themselves."""
    from apok_eval.ledger import open_run

    config = {"settings": listing["settings"], "inputs": listing["paths"], "licence": listing["licence"]}
    writer = open_run(out_dir, run_id=run_id, purpose=purpose, config=config, expected_bytes=0, index=None,
                      allow_dirty=allow_dirty)
    writer.set_seed("nulls", int(config["settings"]["registration"]["seed"]),
                    config_pointer="/settings/registration/seed")
    writer.set_seed("split", int(config["settings"]["face"]["split_seed"]), config_pointer="/settings/face/split_seed")
    writer.set_seed("ceiling", int(config["settings"]["face"]["ceiling_seed"]),
                    config_pointer="/settings/face/ceiling_seed")
    writer.set_seed("ct_sampling", int(config["settings"]["ct"]["seed"]), config_pointer="/settings/ct/seed")
    recorded = writer.add_input("umbilicus", listing["paths"]["umbilicus"])
    parsed = result.get("umbilicus") or {}
    if (recorded["digest"], recorded["size"]) != (parsed.get("digest"), parsed.get("size")):
        raise Refusal("EVAL_RUN_INPUT_CHANGED", f"the umbilicus {recorded['path']} changed after the run read it "
                      f"(read {parsed.get('digest')}, {parsed.get('size')} bytes; now {recorded['digest']}, "
                      f"{recorded['size']} bytes): the run is not sealed", input="umbilicus", path=recorded["path"],
                      read_digest=parsed.get("digest"), read_size=parsed.get("size"),
                      now_digest=recorded["digest"], now_size=recorded["size"])
    for name, (entries, uri) in listing["manifests"].items():
        writer.add_chunk_manifest(name, entries, source_uri=uri)
    Path(out_dir, "e7v3.json").write_bytes(jsonstrict.dumps(result))
    writer.add_output("e7v3", "e7v3.json")
    writer.set_figures("e7v3", pointer="/figures")
    return writer.seal()


__all__ = ["CONTROL_RULE", "Inputs", "SCHEMA", "Settings", "control_verdict", "evaluate", "figures",
           "load_settings", "open_ct", "read_fibres", "read_segments", "run_configuration", "seal_run"]
