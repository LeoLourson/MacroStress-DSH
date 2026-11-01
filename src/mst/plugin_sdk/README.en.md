# mst.plugin_sdk

The surface a position model is written against: what it declares, what it reads from a run and what it returns. A model imports only this package and `mst.contracts`; the workspace, the network and the macro and satellite models themselves are closed to it. The plugin workflow is in [plugins.en.md](../../../docs/howto/plugins.en.md), the platform contracts in [platform.en.md](../../../docs/reference/platform.en.md).

| File | What it does |
|---|---|
| `model.py` | `CalculationModel` (requirements → check → calculation), `Position`, `RunOptions`, `PreparedInputs`, `DataRequirements`, `DriverRequirement`, `Diagnostic`, `CalculationContext` |
| `paths.py` | The protocol `Paths`: `R` draws over the months `t = 0 … H`, original units; `FixedPaths` for tests; `relative` and `change` are the ratio and the difference to `t = 0` |
| `results.py` | `CalculationResult`, `MetricValue`, `Quantiles` and the helpers `money`, `scalar`, `distribution`, `quantiles`, `envelope`, `expected_shortfall`; P5/P50/P95 go together, and losses also have ES95 |
| `fx.py` | The exchange rate of a position's currency to the base currency from the macro paths: `exposure`, `spot`, `ratio`, `variable` |
| `plugin.py` | `Plugin`, the declaration of a package; `DemoCase`, a self-contained case for `check` and `demo-run` |
| `testing.py` | `check_plugin`, the conformance check through the platform dispatcher; `calculate_demo`, a demo calculation for plugin tests |
| `version.py` | `SDK_VERSION` and `compatible`: the version range a plugin declares |

A model is pure: it does not read files or the network and does not draw random numbers, but takes ready realizations from `context.paths`. Driver flags (a scenario impact, parameters from the object one level up, an approximate publication date) reach the result through `envelope`, not through the model's code. A capability a model lacks is added to the SDK by a separate change that raises the minor version; the plugin declares a new `sdk` range.

Tests are in [../../../tests/plugins](../../../tests/plugins/README.en.md) (conformance and registry) and [../../../tests/engine](../../../tests/engine/README.en.md) (dispatcher and models).
