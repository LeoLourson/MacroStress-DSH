"""The version lifecycle of the models: what the ``model`` tool runs as a job.

The models themselves live in plugins (``mst.plugins``); this package estimates them on the
history of a workspace, publishes their versions, lists and describes them. A macro model is
chosen by its id, or the only one installed; a satellite bundle is assembled from the
quantities each satellite model serves, one model per quantity, and the platform refuses a
quantity claimed by two models at discovery, so the assignment is unambiguous.

Nothing here estimates anything itself: ``estimate_macro`` builds the context the plugin
estimates on and stores what it returns, untouched beyond the keys the model SDK names.
"""

from collections.abc import Mapping
from datetime import date
from pathlib import Path

from mst.data.store import SeriesStore
from mst.macro import version as macro_version
from mst.macro.registry import load_registry as load_macro_registry
from mst.model_sdk.contracts import MacroModel


def registry() -> Mapping[str, MacroModel]:
    """The installed macro models, by model id."""
    from mst import plugins

    return plugins.registry().macro


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
                "position_models": sorted(
                    model.model_id for model in plugin.models
                ),
            }
        )
    return {
        "plugins": rows,
        "macro": sorted(registry.macro),
        "satellites": sorted(registry.satellites),
        "quantities": dict(sorted(registry.quantity_model.items())),
    }
