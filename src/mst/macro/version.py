"""Macro model versions: ``<root>/macro/<sha256>/``, written and read by ``mst.versions``.

A version is the metadata and arrays a macro model plugin returned from its ``estimate``; the
platform checks the keys the model SDK names (``model``, ``as_of``, ``ids``, ``n_global``,
``months``, ``shocks``, the panel arrays) and stores the rest untouched. Only the plugin that
estimated a version reads it back: ``load`` dispatches on the ``model`` the meta records, so a
version never depends on the code that reads it. A version saved before the meta recorded
``model`` (a pre-plugin version) is read by the only installed macro model, if there is exactly
one; otherwise ``load`` refuses it with its name. Versions live in the workspace, out of git.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from mst import versions
from mst.macro.panel import Panel
from mst.model_sdk.contracts import (
    MACRO_ARRAYS_REQUIRED,
    MACRO_META_REQUIRED,
    MacroModel,
    ScenarioPlan,
)

# The keys of the meta a macro version must record, beyond the model SDK's own: the block
# structure of the panel, which ``panel_of`` needs to rebuild it.
META_REQUIRED = (*MACRO_META_REQUIRED, "n_global")

BASE_DRAWS = 400
"""Enough for the quantiles of the base path a scenario is written against."""


@dataclass(frozen=True)
class MacroVersion:
    """A macro version loaded back: its plugin, the model it loaded and the recorded meta."""

    name: str
    """The hash that names the version's folder."""
    plugin: MacroModel
    model: object
    meta: dict
    panel: Panel
    """The panel the version was estimated on, in the model scale: the tolerance of a
    point-fixed constraint reads the typical monthly movement of its variables from here."""

    @property
    def as_of(self) -> date:
        return date.fromisoformat(self.meta["as_of"])

    @property
    def ids(self) -> list[str]:
        return list(self.meta["ids"])

    @property
    def months(self) -> list[date]:
        return [date.fromisoformat(m) for m in self.meta["months"]]

    @property
    def shocks(self) -> list[str]:
        return list(self.meta["shocks"])

    @property
    def latents(self) -> tuple[str, ...]:
        """The hidden coordinates the model declares beyond the parameters and the shocks."""
        return tuple(getattr(self.plugin, "latents", ()))


def save(model_id: str, meta: dict, arrays: dict, root: Path) -> str:
    """Write a version of ``model_id`` unless it exists; returns its hash.

    ``meta`` is stored with ``model`` prepended, so the version names the plugin that
    estimated it and ``load`` can dispatch on it.
    """
    full = {"model": model_id, **meta}
    missing = [key for key in META_REQUIRED if key not in full]
    if missing:
        raise ValueError(
            f"a macro version must record {', '.join(META_REQUIRED)}: no {', '.join(missing)}"
        )
    absent = [key for key in MACRO_ARRAYS_REQUIRED if key not in arrays]
    if absent:
        raise ValueError(
            f"a macro version must carry {', '.join(MACRO_ARRAYS_REQUIRED)}: "
            f"no {', '.join(absent)}"
        )
    return versions.save(Path(root) / "macro", full, arrays)


def read(root: Path, name: str) -> tuple[dict, dict]:
    """The meta and arrays of a version; refuses a changed version."""
    return versions.read(Path(root) / "macro", name, "macro version")


def load(root: Path, name: str, models: Mapping[str, MacroModel]) -> MacroVersion:
    """The version ``name`` loaded by the plugin that estimated it.

    ``models`` maps model ids to the installed macro models (``mst.plugins.registry().macro``).
    A version is estimated by a model that is not installed is refused with its name; a version
    whose meta does not name its model at all is refused the same way: only the plugin that
    estimated a version reads it back.
    """
    meta, arrays = read(root, name)
    model_id = meta.get("model")
    if model_id is None:
        raise KeyError(
            f"macro version {name} does not name the model that estimated it, so no installed "
            "model can read it; re-estimate the version"
        )
    if model_id not in models:
        raise KeyError(
            f"macro version {name} was estimated by {model_id!r}, which is not "
            f"installed; installed: {', '.join(sorted(models)) or 'none'}"
        )
    return MacroVersion(
        name=name,
        plugin=models[model_id],
        model=models[model_id].load(meta, arrays),
        meta=meta,
        panel=Panel(
            ids=meta["ids"],
            n_global=meta["n_global"],
            months=[date.fromisoformat(m) for m in meta["months"]],
            y=arrays["panel_y"],
            approximate=arrays["panel_approximate"],
        ),
    )


def panel_of(root: Path, name: str) -> tuple[Panel, date]:
    """The panel a macro version was estimated on, and its calculation date."""
    meta, arrays = read(root, name)
    panel = Panel(
        ids=meta["ids"],
        n_global=meta["n_global"],
        months=[date.fromisoformat(m) for m in meta["months"]],
        y=arrays["panel_y"],
        approximate=arrays["panel_approximate"],
    )
    return panel, date.fromisoformat(meta["as_of"])


def catalogue(root: Path) -> list[dict]:
    """Every saved version, newest first, as its meta records it."""
    return [_summary(path) for path in versions.folders(Path(root) / "macro")]


def describe(root: Path, name: str) -> dict:
    """One version: the model that estimated it, its dates, variables, shocks and start.

    ``origin`` is what a scenario is written against: the month ``t = 0`` and the last known
    value of each variable in the model scale, seasonally adjusted where the panel is.
    """
    meta, arrays = read(root, name)
    diagnostics = meta.get("diagnostics", {})
    identified, estimated = diagnostics.get("identified"), diagnostics.get("estimated")
    return {
        "version": name,
        "model": meta["model"],
        "as_of": meta.get("as_of"),
        "ids": meta.get("ids"),
        "months": meta.get("months"),
        "shocks": meta.get("shocks"),
        "latents": _latents(meta["model"]),
        "variables": _variables(meta.get("ids") or []),
        "origin": _origin(meta, arrays),
        "diagnostics": diagnostics,
        "identification_share": identified / estimated if identified and estimated else None,
    }


def base(
    root: Path,
    name: str,
    models: Mapping[str, MacroModel],
    *,
    horizon: int = 12,
    only: list[str] | None = None,
) -> dict:
    """The law ``P`` of the paths by month: what a deviation from the base is measured against.

    The quantiles are in the model scale, the scale of the conditions, for the months
    ``1 .. horizon``. The draws are few and the seed is fixed: this is for writing a scenario,
    and a calculation draws its own paths.
    """
    version = load(root, name, models)
    unknown = sorted(set(only or ()) - set(version.ids))
    if unknown:
        raise ValueError(f"the macro version has no variable {', '.join(unknown)}")
    simulation = version.plugin.simulate(
        version.model, ScenarioPlan(horizon=horizon), draws=BASE_DRAWS, seed=0
    )
    paths = {}
    for k, id_ in enumerate(simulation.ids):
        if only and id_ not in only:
            continue
        low, middle, high = np.nanquantile(simulation.paths[:, 1:, k], [0.05, 0.5, 0.95], axis=0)
        paths[id_] = {
            "p5": [round(float(value), 4) for value in low],
            "p50": [round(float(value), 4) for value in middle],
            "p95": [round(float(value), 4) for value in high],
        }
    return {
        "version": name,
        "draws": BASE_DRAWS,
        "months": [month.isoformat() for month in simulation.months[1:]],
        "paths": paths,
    }


def _latents(model_id: str) -> list[str]:
    """The hidden coordinates the model declares; none when it is not installed."""
    from mst import plugins

    return list(getattr(plugins.registry().macro.get(model_id), "latents", ()))


def _variables(ids: list[str]) -> list[dict]:
    """The scale of each variable a scenario names: its transform, unit and seasonality."""
    from mst.macro.registry import load_registry

    known = {variable.id: variable for variable in load_registry().variables}
    return [
        {
            "id": id_,
            "transform": known[id_].transform,
            "unit": known[id_].unit,
            "seasonal": known[id_].seasonal,
        }
        for id_ in ids
        if id_ in known
    ]


def _origin(meta: dict, arrays: dict) -> dict:
    """The month ``t = 0`` and each variable's last value in the panel with its month."""
    months, y = meta["months"], arrays["panel_y"]
    last = {}
    for k, id_ in enumerate(meta["ids"]):
        known = np.flatnonzero(np.isfinite(y[:, k]))
        if known.size:
            last[id_] = {"month": months[known[-1]], "y": float(y[known[-1], k])}
    return {"month": months[-1], "last": last}


def _summary(folder: Path) -> dict:
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    diagnostics = meta.get("diagnostics", {})
    return {
        "version": folder.name,
        "model": meta["model"],
        "as_of": meta.get("as_of"),
        "months": len(meta.get("months", [])),
        "shocks": meta.get("shocks"),
        "identified": diagnostics.get("identified"),
        "estimated": diagnostics.get("estimated"),
    }
