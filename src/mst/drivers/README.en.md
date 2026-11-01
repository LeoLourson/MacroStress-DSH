# mst.drivers

Quantities of objects and their drivers: a quantity registry with object catalogs, history preparation on the panel of a macro version and the assembly of a satellite version from the estimates of several models. The satellite models themselves are connected plugins: the hierarchical Gibbs lives in [mst.plugins.hier](../plugins/hier/README.en.md), and another model is a separate package serving its own quantities; several models may claim a quantity, and then an explicit `--model QUANTITY=MODEL` names the one that estimates it.

```powershell
python -m mst.drivers update [QUANTITY ...] --store DIR --drivers FILE
python -m mst.models estimate-satellites --macro SHA256 --store DIR --models DIR --drivers FILE [--model QUANTITY=MODEL_ID]
```

`update` runs the catalog of each quantity, downloads the series of the objects found and writes drivers with history to `--drivers` (`workspace/drivers/catalog.json`). `estimate-satellites` assembles the version bundle: each quantity is estimated by the model assigned to it, and the parts and the residual correlation form one version. Start it only through `python -m`: `__main__.py` sets `OPENBLAS_NUM_THREADS=1`.

| File | What it does |
|---|---|
| `registry.toml`, `registry.py` | Quantities: unit, transformation (`identity`, `log100`, `logit`), regressors and features $\max(0, y-c)$, lags, `span`, catalog, `level`. Objects and drivers from the catalog are added on load; `match` and `resolve` find a position's driver by code |
| `objects.py` | The contracts of an object (class, parent, codes) and a driver `<quantity>:<object>`; a driver series is `drv.<quantity>.<object>` |
| `catalogs/` | Object catalogs by source, see [catalogs/README.en.md](catalogs/README.en.md) |
| `update.py` | Catalog → series download through `mst.data` → `catalog.json`. A failed catalog keeps the quantity's previous record |
| `history.py` | A quantity's history on the panel: months, the object tree, observations in the model scale, `x` from regressors and features; `x` from simulation paths; residual correlation |
| `bundle.py` | A satellite version from parts: assembly (`assemble`), saving and loading with dispatch by model, `served` by the checks, simulation of drivers by their models level by level, a flag for leaving the training range |

A new object of an existing source appears by itself at the next `update`. A new quantity is a table `[[quantities]]` in `registry.toml`; if its source has no catalog, a new class in `catalogs/` is needed. A new satellite model of the same quantity is a plugin with an explicit assignment.

Tests are in [../../../tests/drivers](../../../tests/drivers/README.en.md) and [../../../tests/hier](../../../tests/hier/).
