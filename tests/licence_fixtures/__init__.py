"""Builders for licence-gate tests.

The registry here is a minimal, self-contained ``apok-asset-registry-v1``
document. Its volume ids (``2099...``) and object keys are fictional, so no
test asserts a licence for a real volume. Acceptances are built at test time
and never written to a tracked file (see test_licence_hygiene.py).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

OPEN = "s3://vesuvius-challenge-open-data"
CC_VOLUME_ID = "20990101000001"
CC_VOLUME_URI = f"{OPEN}/PHercParis4/volumes/{CC_VOLUME_ID}-test.zarr"
SPIRAL_URI = "https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4"
SYNTHETIC_URI = "synthetic://apok-fixtures"
EDUCELAB_GATEWAY_VOLUME = "PHercParis4/volumes/20990101000002-7.910um-54keV-masked.zarr"

_REGISTRY: dict[str, Any] = {
    "schema": "apok-asset-registry-v1",
    "licences": {
        "CC-BY-NC-4.0": {
            "name": "Creative Commons Attribution-NonCommercial 4.0 International",
            "url": "https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt",
            "requires_terms": ["BUCKET-TERMS"],
            "publication": "ask",
            "commercial_use": False,
            "shippable_in_pipeline": False,
            "obligations": ["cite the dataset", "non-commercial only"],
            "evidence": "test fixture",
        },
        "EDUCELAB-SCROLLS": {
            "name": "EduceLab-Scrolls data-server licence",
            "url": "https://dl.ash2txt.org/LICENSE.txt",
            "requires_terms": ["SERVER-TERMS"],
            "publication": "forbidden",
            "commercial_use": False,
            "shippable_in_pipeline": False,
            "obligations": ["no redistribution without written approval"],
            "evidence": "test fixture",
        },
        "MIT": {
            "name": "MIT License",
            "url": "https://opensource.org/license/mit",
            "requires_terms": [],
            "publication": "allowed",
            "commercial_use": True,
            "shippable_in_pipeline": True,
            "obligations": ["keep the notice"],
            "evidence": "test fixture",
        },
    },
    "terms": {
        "BUCKET-TERMS": {"title": "bucket terms", "url": "https://example.org/bucket", "per_person": True,
                         "text_sha256": None, "evidence": "test fixture"},
        "SERVER-TERMS": {"title": "server terms", "url": "https://example.org/server", "per_person": True,
                         "text_sha256": None, "evidence": "test fixture"},
        "RESERVATION": {"title": "publication reservation", "url": "https://example.org/reservation",
                        "per_person": True, "text_sha256": None, "evidence": "test fixture"},
    },
    "hosts": {
        OPEN: {"allowed_licences": ["CC-BY-NC-4.0", "EDUCELAB-SCROLLS"], "requires_terms": [],
               "note": "open-data bucket"},
        "s3://vesuvius-challenge": {"allowed_licences": ["EDUCELAB-SCROLLS"], "requires_terms": [],
                                    "note": "gateway bucket"},
        "https://dl.ash2txt.org": {"allowed_licences": ["EDUCELAB-SCROLLS"], "requires_terms": ["SERVER-TERMS"],
                                   "note": "data server"},
        SYNTHETIC_URI: {"allowed_licences": ["MIT"], "requires_terms": [], "note": "generated"},
    },
    "assets": {
        CC_VOLUME_ID: {
            "kind": "volume",
            "locations": [CC_VOLUME_URI],
            "licence": "CC-BY-NC-4.0",
            "additional_terms": [],
            "aliases": [],
            "canonical_volume": {"sample": "PHercParis4", "volume_id": CC_VOLUME_ID},
            "evidence": "test fixture",
        },
        "spiral-input": {
            "kind": "dataset",
            "locations": [SPIRAL_URI],
            "licence": "EDUCELAB-SCROLLS",
            "additional_terms": ["RESERVATION"],
            "aliases": ["spiral input"],
            "evidence": "test fixture",
        },
        "synthetic": {
            "kind": "synthetic",
            "locations": [SYNTHETIC_URI],
            "licence": "MIT",
            "additional_terms": [],
            "aliases": [],
            "evidence": "test fixture",
        },
    },
}


def registry_doc() -> dict[str, Any]:
    """A fresh deep copy of the fixture registry document."""
    return copy.deepcopy(_REGISTRY)


def host(*allowed: str, requires_terms: list[str] | None = None, note: str = "test host") -> dict[str, Any]:
    """A hosts entry allowing ``allowed`` licences."""
    return {"allowed_licences": list(allowed), "requires_terms": requires_terms or [], "note": note}


def asset(kind: str, locations: list[str], licence: str | None, *, additional_terms: list[str] | None = None,
          aliases: list[str] | None = None, volume: tuple[str, str] | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {"kind": kind, "locations": locations, "licence": licence,
                             "additional_terms": additional_terms or [], "aliases": aliases or [],
                             "evidence": "test fixture"}
    if volume is not None:
        entry["canonical_volume"] = {"sample": volume[0], "volume_id": volume[1]}
    return entry


def acceptance(*terms: str, person: str = "Jane Example", date: str = "2025-06-01",
               digests: dict[str, str | None] | None = None) -> dict[str, Any]:
    """A terms acceptance object, built in memory at test time."""
    digests = digests or {}
    return {"person": person, "date": date,
            "accepted": [{"terms": t, "text_sha256": digests.get(t)} for t in terms]}


def run(*inputs: tuple[str, str], accepted: dict[str, Any] | None = None) -> dict[str, Any]:
    config: dict[str, Any] = {"inputs": [{"asset": a, "uri": u} for a, u in inputs]}
    if accepted is not None:
        config["terms_accepted_by"] = accepted
    return config


def write_json(path: Path, doc: Any) -> Path:
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path
