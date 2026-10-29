"""Discovers the objects of every catalog and fetches their series into the store.

For each quantity with a catalog: the catalog lists objects and their data sources, the series
are fetched by ``mst.data.update`` (so a revision adds a version, as for macro variables), and
the drivers whose series holds a value are written to the catalog file with the objects they
hang from. A catalog that fails keeps the quantity's previous entry; a series that fails is
reported and its driver stays only if the store already holds its history.
"""

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import httpx

from mst.data.store import SeriesStore
from mst.data.update import update as update_series
from mst.store import write_json

from .objects import series_id
from .registry import Registry


@dataclass
class Outcome:
    drivers: dict[str, int] = field(default_factory=dict)
    """Drivers kept per quantity."""
    added: dict[str, int] = field(default_factory=dict)
    """Rows added per series."""
    failed: dict[str, str] = field(default_factory=dict)
    """Why a catalog or a series was not fetched."""
    skipped: dict[str, str] = field(default_factory=dict)
    """What a catalog left out and why."""


def update(
    registry: Registry,
    store: SeriesStore,
    http: httpx.Client,
    catalog: Path,
    *,
    fetched_on: date,
    start: date,
    only: list[str] | None = None,
) -> Outcome:
    outcome = Outcome()
    old = json.loads(catalog.read_text(encoding="utf-8")) if catalog.exists() else {}
    quantities = dict(old.get("quantities", {}))
    for quantity in registry.quantities:
        if quantity.catalog is None or (only and quantity.id not in only):
            continue
        try:
            found = quantity.catalog.discover(http)
        except (httpx.HTTPError, RuntimeError, ValueError, KeyError) as error:
            outcome.failed[quantity.id] = f"{type(error).__name__}: {error}"
            continue
        outcome.skipped |= {f"{quantity.id}:{k}": v for k, v in found.skipped.items()}
        sources = {series_id(quantity.id, o): s for o, s in found.drivers.items()}
        fetched = update_series(store, sources, http, fetched_on=fetched_on, start=start)
        outcome.added |= fetched.added
        outcome.failed |= fetched.failed
        kept = [o for o in found.drivers if store.read(series_id(quantity.id, o))]
        parents = {o.id: o.parent for o in found.objects}
        needed = set()
        for object_id in kept:
            while object_id is not None and object_id not in needed:
                needed.add(object_id)
                object_id = parents.get(object_id)
        quantities[quantity.id] = {
            "objects": [
                o.model_dump(mode="json", by_alias=True, exclude_defaults=True)
                for o in found.objects
                if o.id in needed
            ],
            "drivers": kept,
            "sources": {o: found.drivers[o].model_dump(mode="json") for o in kept},
        }
        outcome.drivers[quantity.id] = len(kept)
    # Every estimate reads this file, so a crash must leave the old one or the new one.
    write_json(catalog, {"updated": fetched_on.isoformat(), "quantities": quantities})
    return outcome
