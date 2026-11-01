# mst.plugins

Models as extensions, of three kinds. Each subpackage is a plugin: `plugin.py` with the declaration `plugin = Plugin(...)`, a model, tests and a `README.md`. At start-up the platform imports all packages `mst.plugins.<id>` (plus the entry points of the group `mst.plugins` and the directories of `MST_PLUGINS_PATH`), checks the declarations together and builds a registry by kind: position, macro and satellite. All declarations are checked at once, but several satellite models may claim a quantity, and this is not a start-up error; an ambiguous assignment is the error, and an explicit `--model QUANTITY=MODEL` resolves it. The core (`mst.engine`, `mst.models`, `mst.runs`, `mst.metrics`) gets models only from the registry and names no concrete one.

| Plugin | Kind | Model | What it serves |
|---|---|---|---|
| [`equity`](equity/plugin.py) | position | `equity.paths.v1` | `common_stock`, driver `share_price` |
| [`bond`](bond/plugin.py) | position | `bond.paths.v1` | `bond`, driver `bond_spread` |
| [`credit`](credit/plugin.py) | position | `credit.paths.v1` | `loan`, `deposit`, driver `overdue_share` |
| [`housing`](housing/plugin.py) | position | `housing.paths.v1` | `real_estate`, drivers `housing_price_*` |
| [`bvar`](bvar/plugin.py) | macro | `bvar.sv` | level 0: 23 variables, 5 named shocks |
| [`msvar`](msvar/plugin.py) | macro | `msvar.markov` | level 0: the same 23 variables, 5 named shocks and 2 regime elements |
| [`hier`](hier/plugin.py) | satellite | `hier.gibbs` | all quantities of the driver registry by default |
| [`reflevel2`](reflevel2/plugin.py) | satellite | `reflevel2.regression` | the reference of the second level; no quantity of the main registry, assigned explicitly |

```powershell
python -m mst.plugin new <id> [--kind position|satellite|macro] [--quantity Q] [--label TEXT] [--dest DIR]
python -m mst.plugin check [<plugin_id> | --path DIR]
python -m mst.plugin demo-run <plugin_id> [--path DIR]
```

A new model of any kind is a package here or beside it (`--dest`, entry point, `MST_PLUGINS_PATH`), with no core edits. The structure of a plugin, the model contracts and the checks are in [../../../docs/howto/plugins.en.md](../../../docs/howto/plugins.en.md); the calculation contracts are in [../../../docs/reference/platform.en.md](../../../docs/reference/platform.en.md).

Registry and conformance tests are in [../../../tests/plugins](../../../tests/plugins/README.en.md); BVAR tests in [../../../tests/bvar](../../../tests/bvar/README.en.md), MS-VAR in [../../../tests/msvar](../../../tests/msvar/), the hierarchical satellite in [../../../tests/hier](../../../tests/hier/README.en.md).
