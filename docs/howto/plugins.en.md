# Plugins: position, macro and satellite models

The platform is extended by plugins, and everything connects this way, not just positions. A plugin is a package with the declaration `plugin = Plugin(...)`, its own models, tests and a README. What it declares sets its kind: `models` are position models, `macro` are macro models of level 0, `satellites` are satellite models of quantities. The platform's reference models, BVAR (`src/mst/plugins/bvar`) and the hierarchical satellite (`src/mst/plugins/hier`), are ordinary plugins: they are developed and connected like any other, and the core does not know them.

The core does not change for this. If a new model forced an edit of `mst.engine`, `mst.scenario`, `mst.runs` or `mst.metrics`, the SDK lacks an extension point, and that is a separate SDK change, not part of the plugin.

The contracts a plugin relies on are described in [../reference/platform.en.md](../reference/platform.en.md).

## 1. Quick start

```powershell
python -m mst.plugin new gold --label "Gold"                       # position model
python -m mst.plugin new walk --kind macro --label "Random walk"   # macro model
python -m mst.plugin new link --kind satellite --quantity share_price
python -m mst.plugin check gold                     # the contract; for positions also a demo calculation
python -m mst.plugin demo-run gold                  # P5/P50/P95 on the demo paths
python -m mst.plugin check                          # all installed plugins
```

`new` creates a package that **passes** `check` **at once**:

```
src/mst/plugins/<id>/
  __init__.py        exports MODEL and plugin
  plugin.py          plugin = Plugin(...)
  model.py           the model: its own steps (see the kind below)
  tests/             package tests
  README.md          what the plugin gives and what to change in it
```

The platform connects a package in `src/mst/plugins/` on the next start. A package outside the repository is created with `--dest DIR` (in `DIR/<id>`) and checked there: `python -m mst.plugin check --path DIR/<id>`; `demo-run --path DIR/<id>` calculates its demo case without installing it. A model developed as a separate distribution connects through an entry point of the group `mst.plugins`, and directories with packages through the environment variable `MST_PLUGINS_PATH` (separated by `os.pathsep`): three ways, one registry.

A package connected by path (`--path`, `MST_PLUGINS_PATH`) is imported under the synthetic name `mst_plugin_<folder>`, so relative imports are needed inside it (`from .model import MODEL`). An absolute `import <folder>.model` works only for a package installed as a distribution (entry point) or located in the namespace `mst.plugins.<id>`; by path it fails with `ModuleNotFoundError`. A path in `MST_PLUGINS_PATH` that is not a directory is a start-up error naming it, not a silent "plugin not installed".

## 1a. Model plugins: macro and satellites

A macro model and a satellite speak the contracts of `mst.model_sdk` and only those:

- **Macro model** (`macro=...`), level 0: `estimate(context) → metadata and arrays of the version`, `load(meta, arrays) → the model of the version`, `simulate(model, plan, draws, seed) → paths` and optionally `constraint_law`, the law of the conditions given $R$ that the price is measured on. The plan (`ScenarioPlan`) holds the horizon, the conditions, the causes, the regime event and `inner`, the number of draws of the causes per value of $R$. Shocks are the model's own set; hidden coordinates such as a regime path are declared in the class attribute `latents`; a scenario is checked against the version. Estimation: `python -m mst.models estimate-macro <model_id> --as-of ...`.
- **Satellite** (`satellites=...`), the levels above: the same three steps, but `estimate` returns a part for each assigned quantity (the platform prepares the months, the object tree and the observations through `mst.drivers.history`), and `simulate` receives `UpperPaths` (the macro simulation and the paths of the levels above), its own drivers, the impacts $O$, its own random stream and ready `shocks` (see below). Residual correlation is a parameter, and the bundle passes its slice. Estimation: `python -m mst.models estimate-satellites --macro <hash>`. A model's declaration assigns a quantity; if several models claim a quantity, an explicit `--model QUANTITY=MODEL` decides (ambiguity without a name is an error, not a choice); an explicit assignment can also give a model a quantity it did not claim.

The platform draws the joint shocks of a level itself: for repetition $r$ of level $l$ the stream is `default_rng([seed, 0, l, r])`, `z ~ N(0, I)` over all correlated drivers of the request, and the shocks are `z @ chol(Σ).T` by the version's full correlation matrix (`assemble` makes it positive definite). Levels draw from separate streams, so shocks are independent between levels, as the product measure of the canon requires. Each model receives its slice `(R, steps, k)`: standardized residuals only of its drivers with history, in the order of its `drivers` argument; drivers marked `parameters_from_above` stay on the model's independent noise. The platform provides `horizon + max(len(part.months)) - 1` rows, as many as the "longest" model needs if its last observed month is zero; a model reads the first rows it needs. A model must use these columns, otherwise cross-model correlation does not reach the joint paths.

A version carries `meta["model"]`, the id of the model that estimated it; only this plugin reads the version back, so the format of meta and arrays (other than the mandatory contract keys) is the model's business. The mandatory keys are: for a macro version `as_of`, `ids`, `months`, `shocks` and the panel (`panel_y`, `panel_approximate`); for a part of a quantity `months`, `nodes`, `driver`, `rows` (the number of months each node was estimated on) and `checks`. The platform writes the key `model` of a part itself: a part naming another model is rejected, so the provenance of a version cannot be forged.

Model SDK 2.0 supports $L>1$: a quantity declares its `level`, the bundle draws the levels in ascending order, and a satellite of level $l$ reads the levels below its own through `UpperPaths`. The reference is the plugin `reflevel2`; the steps are in the section "How to add a level" of [../../src/mst/model_sdk/README.en.md](../../src/mst/model_sdk/README.en.md).

## 2. Declaration

```python
from mst.plugin_sdk import DemoCase, FixedPaths, Plugin, Position

plugin = Plugin(
    plugin_id="gold",
    version="0.1.0",
    sdk=">=1.0,<2.0",
    label="Gold",
    description="A gold position on the paths of its price.",
    models=(MODEL,),
    demo=(DemoCase(...),),
)
```

| Field | Meaning |
|---|---|
| `plugin_id` | The package name and the identifier namespace: `gold.paths.v1` |
| `version` | The plugin version `x.y.z`; `check` verifies it, it enters the registry, and a run records the model version of each position |
| `sdk` | The SDK version range; an incompatible plugin blocks the start and is named |
| `label`, `description` | What the plugin is called and what it gives |
| `models` | `CalculationModel` models; their ids start with `<plugin_id>.` |
| `demo` | Demo cases on which `check` and `demo-run` calculate the position |

A model id and a metric id have the form `<plugin_id>.<name>`. An asset type is global and cannot be served by two models. The registry checks these rules at start-up.

## 3. Model

A model inherits `mst.plugin_sdk.CalculationModel` and has three steps:

```python
class GoldModel(CalculationModel):
    model_id = "gold.paths.v1"
    version = "0.1.0"
    label = "Gold on the price paths"
    instrument_types = ("gold_position",)
    terms_schema = Terms
    metrics = ("gold.value_current", "gold.pnl", "gold.loss")
    required_metrics = metrics

    def requirements(self, position, options):
        """What is needed from the paths: drivers and macro variables. Paths are not read yet."""

    def validate(self, prepared):
        """A judgement on the prepared inputs; an error stops the position."""

    def calculate(self, prepared, context):
        """CalculationResult on the common paths of the run; pure computation only."""
```

- **`requirements`** returns `DataRequirements`: `DriverRequirement` (a quantity and, if needed, the object or codes of the position) and names of macro variables. The platform finds the driver in the catalog of the satellite version; a missing driver leaves the position uncalculated and names the quantity.
- **`validate`** returns `Diagnostic`s. An `error` turns the position into `invalid` with a reason; it stays in the run and in the report's coverage.
- **`calculate`** reads only `prepared` (resolved drivers, the value at `t = 0`, `terms`) and `context` (paths, date, horizon, seed, base currency). A model does not read files, the network or the workspace and does not draw random numbers. The result is assembled by `envelope`; required metrics must be in the result, and the length of `value_change` must equal the number of draws.

The `terms` schema is a pydantic model with `extra="forbid"`. Its fields are analyst inputs (duration, LGD, housing market) and have no silent defaults: a missing required field is a refusal naming the field.

## 4. Demo case

```python
DEMO = FixedPaths(
    drivers={"gold_price:demo": np.array([[100.0, 102.0, 104.0], [100.0, 98.0, 96.0]])},
    start=date(2026, 9, 1),
)

DemoCase(
    name="demo",
    position=Position(id="demo", type="gold_position", notional=100.0),
    paths=DEMO,
    drivers={"gold_price": "gold_price:demo"},
)
```

`drivers` maps a required quantity to a driver id in the paths. `FixedPaths` is an ordinary `Paths`: it has `draws`, `horizon`, `months`, `variables`, `macro`, `driver`, `flags`. A demo case is self-contained: it needs no workspace, estimated models or network.

`check` calculates each case through the platform dispatcher twice: the repeat must give the same numbers, a position with `completes=True` must calculate, and required metrics must be present. `demo-run` prints P5/P50/P95, ES95 for losses and the result flags.

## 5. Check and acceptance

`python -m mst.plugin check` verifies:

- the plugin identifier, the versions of the plugin and models, the SDK range (`sdk` for positions, `model_sdk` for macro and satellite models), a non-empty `label` and at least one model in `models`, `macro` or `satellites`;
- model names and their membership in the plugin's namespace, the uniqueness of asset types;
- `metrics`: ids of the form `<plugin_id>.<name>`, no repeats, `required_metrics` within `metrics`;
- the `terms` schema: it has a JSON Schema;
- the demo case: a model serves its type, all required drivers map to a driver present in the paths, and the macro variables are in the paths;
- the demo case calculation through `engine.run`: status, required metrics, determinism.

A plugin with position models and no demo case fails the check, because `demo-run` would have nothing to verify. A model plugin (only `macro` or `satellites`) needs no demo case: the contract and service tests check it. The function `mst.model_sdk.testing.check_shocks` checks the satellite shock contract: it calls `simulate` with different `shocks` on one seed and returns a finding in three cases: the paths coincide (the model ignored the shocks), the model does not accept the argument, or the model changes the shape of the paths. The plugin's own tests live in `src/mst/plugins/<id>/tests/` and run after the package is installed:

```powershell
pytest src/mst/plugins/gold/tests
```

The common check of installed plugins is `tests/plugins/test_conformance.py`. Separately, `tests/plugins/test_boundary.py` fails if the core starts importing a concrete asset type or naming its identifier.

## 6. Boundaries

A plugin imports the contracts and helpers of the core, not its orchestration:

- allowed: `mst.plugin_sdk`, `mst.model_sdk`, `mst.contracts`, `mst.macro.panel`, `mst.macro.registry`, `mst.drivers.history`, `mst.drivers.registry` and its own package; BVAR also reads the scenario contract (`mst.scenario.contracts`) in the adapter `to_scenario`, because the scenario tool passes this contract to the model;
- forbidden: `mst.engine`, `mst.scenario.build`, `mst.runs`, `mst.portfolio`, `mst.metrics`, `mst.models`, `mst.plugins` (the registry and other plugin packages) and models of other plugins.

The core names no concrete asset type: `mst.engine` asks the registry, and the registry is assembled from the packages `mst.plugins.<id>`.

If a model lacks an SDK capability (a path function, a result field, a flag), it is added to the SDK by a separate core change, and the SDK minor version goes up. A plugin that needs the new capability declares a new `sdk` range.
