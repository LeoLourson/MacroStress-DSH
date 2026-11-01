# mst.engine

Position calculation without branching by asset type: models come from the registry `mst.plugins`, and the engine asks the catalog, prepares inputs and checks what a model returned. The contracts of the model and the paths are in [platform.en.md](../../../docs/reference/platform.en.md).

| File | What it does |
|---|---|
| `model_catalog.py` | `Catalog`: models by id and the model serving an asset type; the registry is read lazily, on first access |
| `dispatch.py` | `prepare`: `terms` by the model's schema, its `requirements`, a check of macro variables against the paths, driver resolution in the catalog of the satellite version, `validate`; `calculate`: the calculation and a check of the result against the declaration; `not_run`: a position with the status `invalid` and a reason |
| `run.py` | `run`: all positions on one set of paths; `Batch` with results in position order and the worst status |
| `status.py` | The order of statuses `ok` < `needs_review` < `invalid` and the worst of a set |

A position stays in the run with a refusal reason if its asset type is absent from the catalog or two models serve it, if `terms` fail the schema, if a declared macro variable is absent from the paths, if a required quantity has no served driver, or if the model broke its declaration. An exception inside a model gives `invalid` with the code `model_error:<Type>` and does not stop the other positions. `invalid` does not enter the portfolio arithmetic; `needs_review` does, and the report discloses it.

Tests are in [../../../tests/engine](../../../tests/engine/README.en.md).
