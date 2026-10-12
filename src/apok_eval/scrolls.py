"""Scroll names: one comparison key and one alias table for every module.

A scroll (a "sample" in the licence registry) has a catalog id, such as
``PHercParis4``, and other names in use: legacy names (``Scroll1``), unpadded
catalog numbers (``PHerc332``), spaced or dotted spellings (``PHerc. Paris 4``)
and the short names of fit folders (``s1``). Two names are compared by
:func:`scroll_key`: case folded and every character that is not a letter or a
digit dropped, so spacing, dots, dashes and underscores never make two names
differ.

:data:`SCROLLS` maps each catalog id to the other names documented for it (the
evidence is beside the table). :func:`canonical_scroll` resolves a name through it and returns
``None`` for a name it does not list: an unknown name is not an identity.

``apok_eval.licence`` compares a ``canonical_volume.sample`` by
``canonical_sample``: the key of the catalog id when the name is listed here,
else the name's own key.
"""

from __future__ import annotations

import re
from types import MappingProxyType
from typing import Any

_NOT_ALNUM = re.compile(r"[^0-9a-z]")

#: catalog id -> the other names documented for it. Evidence (villa fb8c2c4):
#: * legacy names and unpadded catalog numbers: vesuvius/src/vesuvius/install/configs/scrolls.yaml:11,20,26,32,38
#:   (ScrollN/<id>.volpkg); scrollprize.org/static/data_browser/index.json:61-62, 2230, 3065-3068, 7648-7651, 8707
#:   (catalog ids, with "legacy" links to the ScrollN volpkgs) and 8746 (PHercParis3 is "Scroll 2");
#: * s1: vesuvius/src/vesuvius/neural_tracing/winding_models/NATIVE_PHASE_CACHE.md:19-22 (PHercParis4 cache made from
#:   fit folder 2026-07-27_s1_..., reference PHercParis4/volumes/s1_2um_ds2.zarr);
#:   lasagna/docs/tifxyz_training_integration.md:65-69 (PHercParis4.volpkg/volumes/s1_uint8.zarr);
#: * s4: lasagna/docs/tifxyz_training_integration.md:59-63 (Scroll4.volpkg/volumes/s4.zarr holds the PHerc1667
#:   segments);
#: * PHerc0139: vesuvius/src/vesuvius/neural_tracing/winding_models/train_config.json:60-67.
SCROLLS = MappingProxyType({
    "PHercParis4": ("Scroll1", "s1"),
    "PHercParis3": ("Scroll2",),
    "PHerc0332": ("Scroll3", "PHerc332"),
    "PHerc1667": ("Scroll4", "s4"),
    "PHerc0172": ("Scroll5", "PHerc172"),
    "PHerc0139": (),
})


def scroll_key(name: str) -> str:
    """The comparison key of a scroll name: case folded, every character but a letter or digit dropped."""
    return _NOT_ALNUM.sub("", name.casefold())


def _alias_keys() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for canonical, aliases in SCROLLS.items():
        for name in (canonical, *aliases):
            key = scroll_key(name)
            if not key or mapping.get(key, canonical) != canonical:
                raise ValueError(f"scroll name {name!r} is empty or names two scrolls")  # a bad edit of SCROLLS
            mapping[key] = canonical
    return mapping


_BY_KEY = MappingProxyType(_alias_keys())


def canonical_scroll(name: Any) -> str | None:
    """The catalog id :data:`SCROLLS` lists ``name`` under, or ``None`` (an unknown name, or not a string)."""
    return _BY_KEY.get(scroll_key(name)) if isinstance(name, str) else None
