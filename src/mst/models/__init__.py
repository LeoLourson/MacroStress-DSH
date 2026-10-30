"""The version lifecycle of the models: what the ``model`` tool runs as a job.

The models themselves live in plugins (``mst.plugins``); this package estimates them on the
history of a workspace, publishes their versions, lists and describes them. A macro model is
chosen by its id, or the only one installed. A satellite bundle is assembled from the
quantities each satellite model serves: by default the model that declares a quantity
estimates it — a quantity claimed by two models is a start-up error, so the default is
unambiguous — and an explicit ``quantity=model_id`` assignment overrides the default for any
installed satellite model.

Nothing here estimates anything itself: ``estimate_macro`` and ``estimate_satellites`` build
the context each plugin estimates on and store what it returns, untouched beyond the keys the
model SDK names.
"""

from collections.abc import Mapping
from datetime import date
from pathlib import Path

from mst.data.store import SeriesStore
from mst.drivers import bundle as satellite_bundle
from mst.drivers.registry import load_registry as load_drivers_registry
from mst.macro import version as macro_version
from mst.macro.registry import load_registry as load_macro_registry
from mst.model_sdk.contracts import (
    MacroModel,
    SatelliteEstimate,
    SatelliteEstimateContext,
    SatelliteModel,
)


def registry() -> Mapping[str, MacroModel]:
    """The installed macro models, by model id."""
    from mst import plugins

    return plugins.registry().macro


def satellites() -> Mapping[str, SatelliteModel]:
    """The installed satellite models, by model id."""
    from mst import plugins

    return plugins.registry().satellites


def selected(model_id: str | None = None) -> MacroModel:
    """The macro model to work with: the named one, or the only one installed."""
    models = dict(registry())
    if model_id is not None:
        if model_id not in models:
            raise ValueError(
                f"no installed macro model {model_id!r}; installed: "
                f"{', '.join(sorted(models)) or 'none'}"
            )
        return models[model_id]
    if len(models) == 1:
        return next(iter(models.values()))
    raise ValueError(
        "several macro models are installed, name one: " + ", ".join(sorted(models))
    )


def estimate_macro(
    model_id: str | None, *, store: Path, models: Path, as_of: date
) -> str:
    """Estimate the macro model on the history known on ``as_of``; returns the version hash."""
    from mst.model_sdk.contracts import MacroEstimateContext

    model = selected(model_id)
    estimate = model.estimate(
        MacroEstimateContext(
            store=SeriesStore(store),
            registry=load_macro_registry(),
            as_of=as_of,
        )
    )
    return macro_version.save(model.model_id, estimate.meta, estimate.arrays, models)


def assignment(
    quantities: list[str], overrides: Mapping[str, str] | None = None
) -> tuple[dict[str, str], list[str]]:
    """The satellite model of every quantity, and the quantities no model serves.

    ``overrides`` maps a quantity to a satellite model id explicitly; it names an installed
    model or the assignment is refused. A quantity with no model and no override is unserved:
    recorded, not estimated.
    """
    from mst import plugins

    installed = plugins.registry()
    models = dict(installed.satellites)
    unknown = sorted(
        {model_id for model_id in (overrides or {}).values() if model_id not in models}
    )
    if unknown:
        raise ValueError(
            f"no installed satellite model {', '.join(unknown)}; installed: "
            f"{', '.join(sorted(models)) or 'none'}"
        )
    chosen: dict[str, str] = {}
    unserved: list[str] = []
    for quantity in quantities:
        if overrides and quantity in overrides:
            chosen[quantity] = overrides[quantity]
            continue
        model_id = installed.quantity_model.get(quantity)
        if model_id is None:
            unserved.append(quantity)
        else:
            chosen[quantity] = model_id
    return chosen, unserved


def estimate_satellites(
    macro: str,
    *,
    store: Path,
    models: Path,
    drivers: Path,
    overrides: Mapping[str, str] | None = None,
) -> tuple[str, dict]:
    """Estimate the satellite bundle of the macro version ``macro``; the hash and its summary.

    Every quantity with drivers is estimated by the satellite model assigned to it, one
    context per model; the parts are assembled into one version with the residual correlation
    of every driver with history.
    """
    panel, as_of = macro_version.panel_of(models, macro)
    registry = load_drivers_registry(catalog=drivers)
    quantities = [
        quantity.id
        for quantity in registry.quantities
        if any(driver.quantity == quantity.id for driver in registry.drivers)
    ]
    chosen, unserved = assignment(quantities, overrides)
    by_model: dict[str, list[str]] = {}
    for quantity, model_id in chosen.items():
        by_model.setdefault(model_id, []).append(quantity)
    estimates: dict[str, SatelliteEstimate] = {}
    for model_id in sorted(by_model):
        estimates[model_id] = satellites()[model_id].estimate(
            SatelliteEstimateContext(
                store=SeriesStore(store),
                registry=registry,
                panel=panel,
                as_of=as_of,
                macro=macro,
                quantities=tuple(by_model[model_id]),
            )
        )
    meta, arrays = satellite_bundle.assemble(
        registry, macro, as_of, estimates, unserved=unserved
    )
    name = satellite_bundle.save(meta, arrays, models)
    return name, _checks_table(meta)


def _checks_table(meta: dict) -> dict:
    """The check of every driver of a bundle, as the estimate command prints it."""
    rows = []
    for quantity in sorted(meta["quantities"]):
        part = meta["quantities"][quantity]
        for node, row in sorted(part["checks"].items(), key=lambda item: int(item[0])):
            rows.append(
                {
                    "driver": part["driver"][int(node)],
                    "months": part["rows"][int(node)],
                    "satellite": row["satellite"],
                    "baseline": row["baseline"],
                    "forecasts": row["forecasts"],
                    "served": row["served"],
                    "reason": row["reason"],
                    "model": part["model"],
                }
            )
    return {"rows": rows, "unserved": meta["unserved"]}


def catalog() -> dict:
    """The installed model plugins: their macro and satellite models with what they serve."""
    from mst import plugins

    registry = plugins.registry()
    rows = []
    for plugin_id in sorted(registry.plugins):
        plugin = registry.plugins[plugin_id]
        rows.append(
            {
                "plugin": plugin_id,
                "label": plugin.label,
                "version": plugin.version,
                "macro": [model.describe() for model in plugin.macro],
                "satellites": [model.describe() for model in plugin.satellites],
                "position_models": sorted(model.model_id for model in plugin.models),
            }
        )
    return {
        "plugins": rows,
        "macro": sorted(registry.macro),
        "satellites": sorted(registry.satellites),
        "quantities": dict(sorted(registry.quantity_model.items())),
    }
