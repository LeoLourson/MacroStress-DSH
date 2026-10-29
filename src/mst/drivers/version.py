"""A satellite version: the estimated drivers of one macro version, named by their hash.

``build`` estimates every quantity on the panel of a macro model version, checks each driver out
of sample and estimates the residual correlation of the drivers with history. ``<root>/satellites/
<sha256>/`` holds ``meta.json`` (registry, spec, calculation date, macro version, nodes, checks)
and ``arrays.npz`` (posterior draws, the driver histories and ``x``), written and read by
``mst.versions``. A run takes the satellites of the macro version it simulates.
"""

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np

from mst import versions
from mst.data.store import SeriesStore
from mst.macro.panel import Panel
from mst.macro.version import load as load_macro

from .check import Check, check
from .estimate import Fit, History, correlation, fit, history, residuals
from .registry import Registry
from .spec import Spec


@dataclass(frozen=True)
class Satellites:
    macro: str
    """The macro model version whose panel the satellites were estimated on."""
    as_of: date
    registry: Registry
    spec: Spec
    fits: dict[str, Fit]
    checks: dict[str, dict[int, Check]]
    """Per quantity, the check of every node with a driver."""
    correlated: list[str]
    """Drivers with history, in the order of ``correlation``."""
    correlation: np.ndarray

    def node(self, driver: str) -> tuple[str, int]:
        quantity = driver.split(":", 1)[0]
        hist = self.fits[quantity].history if quantity in self.fits else None
        if hist is None or driver not in hist.driver:
            raise KeyError(driver)
        return quantity, hist.driver.index(driver)

    def served(self) -> list[str]:
        return [
            self.fits[q].history.driver[i]
            for q, checks in self.checks.items()
            for i, item in checks.items()
            if item.served
        ]


def macro_panel(root: Path, name: str) -> tuple[Panel, date]:
    """The panel a macro version was estimated on, and its calculation date."""
    _, meta = load_macro(root, name)
    with np.load(root / "macro" / name / "arrays.npz") as stored:
        y, approximate = stored["panel_y"], stored["panel_approximate"]
    panel = Panel(
        ids=meta["ids"],
        n_global=meta["n_global"],
        months=[date.fromisoformat(m) for m in meta["months"]],
        y=y,
        approximate=approximate,
    )
    return panel, date.fromisoformat(meta["as_of"])


def build(
    registry: Registry, spec: Spec, panel: Panel, store: SeriesStore, as_of: date, macro: str
) -> Satellites:
    fits, checks, correlated, series = {}, {}, [], []
    for quantity in registry.quantities:
        if not any(d.quantity == quantity.id for d in registry.drivers):
            continue
        hist = history(registry, quantity.id, panel, store, as_of)
        fits[quantity.id] = fit(hist, spec)
        checks[quantity.id] = check(hist, spec)
        for node, driver in enumerate(hist.driver):
            if driver is not None and fits[quantity.id].rows[node] > 0:
                correlated.append(driver)
                series.append(residuals(fits[quantity.id], node))
    return Satellites(
        macro=macro,
        as_of=as_of,
        registry=registry,
        spec=spec,
        fits=fits,
        checks=checks,
        correlated=correlated,
        correlation=correlation(series),
    )


def _parts(sat: Satellites) -> tuple[dict, dict[str, np.ndarray]]:
    meta = {
        "macro": sat.macro,
        "as_of": sat.as_of.isoformat(),
        "registry": sat.registry.model_dump(mode="json", by_alias=True),
        "spec": sat.spec.model_dump(mode="json"),
        "correlated": sat.correlated,
        "quantities": {},
    }
    arrays = {"correlation": sat.correlation}
    for q, item in sat.fits.items():
        hist = item.history
        meta["quantities"][q] = {
            "months": [m.isoformat() for m in hist.months],
            "nodes": hist.nodes,
            "parent": hist.parent.tolist(),
            "classes": hist.classes,
            "driver": hist.driver,
            "rows": item.rows.tolist(),
            "checks": {str(node): asdict(c) for node, c in sat.checks[q].items()},
        }
        arrays |= {
            f"{q}.gamma": item.gamma,
            f"{q}.sigma": item.sigma,
            f"{q}.d": hist.d,
            f"{q}.approximate": hist.approximate,
            f"{q}.x": hist.x,
        }
    return meta, arrays


def save(sat: Satellites, root: Path) -> str:
    """Writes the version unless it exists; returns its hash."""
    meta, arrays = _parts(sat)
    return versions.save(root / "satellites", meta, arrays)


def load(root: Path, name: str) -> Satellites:
    """The satellites of a version; refuses a changed version."""
    meta, arrays = versions.read(root / "satellites", name, "satellite version")
    registry = Registry.model_validate(meta["registry"])
    fits, checks = {}, {}
    for q, part in meta["quantities"].items():
        hist = History(
            quantity=registry.quantity(q),
            months=[date.fromisoformat(m) for m in part["months"]],
            nodes=part["nodes"],
            parent=np.array(part["parent"], dtype=int),
            classes=part["classes"],
            driver=part["driver"],
            d=arrays[f"{q}.d"],
            approximate=arrays[f"{q}.approximate"],
            x=arrays[f"{q}.x"],
        )
        fits[q] = Fit(
            history=hist,
            gamma=arrays[f"{q}.gamma"],
            sigma=arrays[f"{q}.sigma"],
            rows=np.array(part["rows"], dtype=int),
        )
        checks[q] = {int(node): Check(**c) for node, c in part["checks"].items()}
    return Satellites(
        macro=meta["macro"],
        as_of=date.fromisoformat(meta["as_of"]),
        registry=registry,
        spec=Spec.model_validate(meta["spec"]),
        fits=fits,
        checks=checks,
        correlated=meta["correlated"],
        correlation=arrays["correlation"],
    )


def catalogue(root: Path) -> list[dict]:
    """Every saved satellite version, newest first, as its meta records it."""
    return [_summary(path) for path in versions.folders(Path(root) / "satellites")]


def describe(root: Path, name: str) -> dict:
    """One version: the drivers its check served, and those it rejected with the reason."""
    sat = load(root, name)
    served, rejected = [], []
    for quantity, checks in sat.checks.items():
        for node, item in checks.items():
            driver = sat.fits[quantity].history.driver[node]
            if item.served:
                served.append(driver)
            else:
                rejected.append(
                    {
                        "driver": driver,
                        "reason": item.reason,
                        "satellite": item.satellite,
                        "baseline": item.baseline,
                        "forecasts": item.forecasts,
                    }
                )
    return {
        "version": name,
        "macro": sat.macro,
        "as_of": sat.as_of.isoformat(),
        "served": served,
        "rejected": rejected,
    }


def _summary(folder: Path) -> dict:
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    checks = [item for part in meta["quantities"].values() for item in part["checks"].values()]
    return {
        "version": folder.name,
        "macro": meta["macro"],
        "as_of": meta["as_of"],
        "quantities": sorted(meta["quantities"]),
        "served": sum(1 for item in checks if item["served"]),
        "checked": len(checks),
    }
