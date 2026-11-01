# mst.macro

The contract of level 0 of the [formalization](../../../docs/concept/formalization.en.md): the macro variable registry, the assembly of the panel on the calculation date and the store of macro model versions. The macro model itself is a connected plugin: the BVAR with stochastic volatility lives in [mst.plugins.bvar](../plugins/bvar/README.en.md), the MS-VAR in [mst.plugins.msvar](../plugins/msvar/README.en.md), and another model is a separate package with its own `model_id`, connected through the registry `mst.plugins` (entry points, `MST_PLUGINS_PATH` or a package in the namespace), see [docs/howto/plugins.en.md](../../../docs/howto/plugins.en.md).

```powershell
python -m mst.models estimate-macro bvar.sv --as-of 2026-10-29 --store DIR --models DIR
python -m mst.models list --models DIR
```

| File | What it does |
|---|---|
| `registry.toml`, `registry.py` | The registry: 23 variables with a transformation $g$, the blocks `gl.`/`ru.`; named shocks belong to the macro model identified by them |
| `panel.py` | The panel: series on the date in the model scale on a grid of months, the global block first, seasonal adjustment, missing values as NaN |
| `version.py` | Macro model versions on top of `mst.versions`: a check of mandatory keys, saving, loading by the plugin that estimated it (`meta["model"]`), the panel of a version, list and description |

An edit of `registry.toml` affects only new versions: a version stores both the registry and the specification it was estimated with. Estimation is deterministic by the seed written in the specification.

Tests are in [../../../tests/macro](../../../tests/macro/README.en.md); BVAR tests are in [../../../tests/bvar](../../../tests/bvar/README.en.md).
