# Platform: calculation contracts

The general scheme of the system is in [../concept/formalization.en.md](../concept/formalization.en.md); this page gives the contracts that implement it.

MacroStress is a modular monolith with no database, web layer or orchestrator. State is the files of one directory `workspace/`, and long operations run as jobs. The core knows neither asset types nor concrete models: it provides data, model versions, the scenario, the calculation dispatcher, the aggregate and the report. Everything domain-specific (stock, bond, credit, housing, the macro model and the satellite of a quantity) connects through plugin packages ([../howto/plugins.en.md](../howto/plugins.en.md)). The data flow and the workspace layout are described in [../concept/architecture.en.md](../concept/architecture.en.md).

## 1. Layers

| Layer | Package | What it gives |
|---|---|---|
| Data | `mst.data` | Monthly series with the publication date of each value; `as_of` does not see future revisions |
| Level 0 | `mst.macro` + plugin | The variable registry, the panel on a date, versions; the model itself (BVAR `bvar.sv`, MS-VAR `msvar.markov` and any other) is a `MacroModel` plugin |
| Levels above | `mst.drivers` + plugins | Objects and quantities, history preparation, the version bundle with its own satellite model per quantity (`SatelliteModel`), the out-of-sample check in the part of the version, the loop over levels |
| Scenario | `mst.scenario` | $s=(N,m,O,D,\mathcal C)$: the narrative, the versions, the impacts at any level, the causes, the conditions; `build` gives the paths, `overlay` the tilt, `reverse` the reverse problem |
| Model SDK | `mst.model_sdk` | The scenario plan, impacts, estimation contexts, the three model steps, the conditional machinery, the price of a law |
| Position SDK | `mst.plugin_sdk` | `Paths`, requirements, prepared inputs, the result, the SDK version |
| Versions | `mst.models` | `estimate-macro`, `estimate-satellites` (assignment of quantities to models), list and description |
| Calculation core | `mst.engine` | The model catalog, input preparation, position calculation, result check |
| Plugins | `mst.plugins.<id>` | Position, macro and satellite models; the registry is assembled from packages, entry points and `MST_PLUGINS_PATH` |
| Quality | `mst.metrics` | Cells $(e,t,q)$, CRPS, band coverage, admission, benchmarks, the episode backtest, the rolling check |
| Portfolio | `mst.portfolio` | The portfolio, the check against versions, value and report |
| Run | `mst.runs` | Paths, the conditional forecast of indicators and results in `runs/<id>/`; the same seed gives the same numbers |
| Server | `mst.mcp`, `mst.jobs`, `mst.store` | MCP tools, jobs, the workspace directory |

## 2. Workspace and versions

A model version is an immutable directory `<sha256>/` with `meta.json` and `arrays.npz`. The hash covers the metadata and the arrays, and a changed file is rejected on load. A version names the model that estimated it (`meta["model"]`) and only this model reads it back; the platform checks the mandatory keys (dates, variables, shocks, panel), and the rest is the model's own format. A satellite version is a bundle: a part for each quantity, its own model per part, the residual correlation of all drivers with history. Satellites are estimated on the panel of a specific macro version and store its hash. A scenario names both versions; a run fixes them in `meta.json`, so a change of version does not mix the numbers of one run.

Series store each value with its publication date. `as_of(day)` returns what was known on that day; on the first download the revision date is estimated and marked `approximate = 1`. Model estimation, the scenario and the run read only data no later than their date.

## 3. Position and paths

`Position` says what it is, what it is worth and by which object it is matched:

| Field | Meaning |
|---|---|
| `id`, `name` | The identifier and label of the position |
| `type` | The asset type; the registry selects the model by it |
| `currency` | The position's currency; the portfolio's base currency is the ruble |
| `notional` | The current value in millions of the currency |
| `codes` | `secid`, `isin`, `inn`, `okved`, `okato`: object codes for matching with a driver |
| `terms` | The model's own inputs by its schema (`duration`, `lgd`, `market`) |

`Paths` is one object per run: `draws` (R), `horizon` (H), `months` (t = 0 … H), `variables`, `seed`; `macro(name)` and `driver(id_)` return arrays `(R, H + 1)` in original units, `flags(id_)` the driver's flags (`parameters_from_above`, `approximate_data`, `impact`). Month `t = 0` is the calculation month, the last month of history.

## 4. Model

A model is `mst.plugin_sdk.CalculationModel`:

```
requirements(position, options) -> DataRequirements
validate(prepared)              -> list[Diagnostic]
calculate(prepared, context)    -> CalculationResult
```

- `DataRequirements` lists drivers (`DriverRequirement`: a quantity, the object or codes of the position) and macro variables. The platform resolves a driver in the catalog of the satellite version; an unresolved required driver leaves the position uncalculated and names it.
- `PreparedInputs` holds the position, the model, the date, the horizon, the base, the value at `t = 0`, the resolved drivers, the unresolved ones with a reason, and the original requirements.
- `CalculationContext` holds the date, the horizon, the seed, the base and `paths`.
- `CalculationResult` holds the status (`ok`/`needs_review`/`invalid`), the value change by draw, metrics (a value and/or P5/P50/P95, ES95 for losses), flags and diagnostics.

A model is pure: it does not read the workspace or the network and does not draw random numbers. Numbers are calculated on the common realizations of the scenario, so two positions of one object see the same paths, and the order of positions does not change the result.

## 5. Dispatcher and statuses

`engine.prepare` checks `terms` against the model's schema, asks for `requirements`, checks the macro variables against the paths, resolves the drivers and passes the inputs to `validate`. An error at any step gives `invalid` with a reason; the position stays in the run and in the report's coverage.

`engine.calculate` calls `calculate` and checks the output against the declaration: the identity of the position, model and version, the status, the length of `value_change`, the declared and required metrics. A violation gives `invalid`; a model exception gives `invalid` with the code `model_error:<Type>`. One broken model does not stop the other positions.

`engine.run` calculates a list of positions on one set of paths and returns a `Batch`: the results in position order and the worst status. `invalid` does not enter the portfolio arithmetic; `needs_review` does, and the report discloses it.

## 6. Plugin registry

`mst.plugins.discover` imports each package `mst.plugins.<id>` and reads its `plugin = Plugin(...)`. `mst.plugins.build` checks the declarations together and builds a `Registry`: plugins, models by id, the owner of each model. The checks:

- `plugin_id` is lower_snake_case and does not repeat;
- plugin and model versions are `x.y.z`, the SDK range is compatible with `SDK_VERSION`;
- a model or metric id starts with `<plugin_id>.` and does not repeat; exactly one model serves an asset type;
- `required_metrics` lie within `metrics`; the `terms` schema gives a JSON Schema;
- a plugin with position models has a demo case, a model serves its type, required drivers map to drivers present in the paths; a plugin with only macro or satellite models needs no demo case.

A problem in any plugin blocks the start with a list of all problems and plugin names. The catalog `mst.engine.model_catalog` is built from the registry; the core has no other route to models.

## 7. Scenario, run, report

A scenario is the file `scenarios/<id>.json`: the narrative, the causes, the conditions on paths, on means and on probabilities, the impacts on macro variables or drivers, references to corpus fragments, the horizon, both versions and the person who approved it. An impact on a macro variable (level 0) is an additive shift after $G^0$ and does not spread within level 0: the other macro variables do not get the reaction a shock of $D$ would give. The conditions refer to the final path $F_O(\omega)$, so the macro model solves them against the shifted path (both bounds fall by $w\Delta$), and the impact is added to the paths after the draw. `mst.scenario.build` builds the conditional forecast of the macro model under the conditions, tilts the causes for conditions on means, returns the variables to original units, calculates the driver paths on the same draws and the price $\mathrm{dist}(s)$. A condition its causes cannot reach, an unknown variable, an impact on an unserved driver and a foreign satellite version are refusals naming the reason.

`mst.runs` writes the request, metadata, paths, results and report to `runs/<id>/`. A run is a function of the scenario, the versions, the number of draws and the seed: the same request gives the same numbers bit for bit. The portfolio report adds losses **within a draw** and only then takes P5, P50, P95 and ES95; the sum of position P95 values is not passed off as the portfolio's P95. An excluded position is named with a reason, and the coverage is the share of value the numbers cover; `dist` is the price of the scenario with its components, absent without conditions.

## 8. MCP and DSH

`mst.mcp` is the only server: a tool parses arguments and calls one package function or starts a job. Long operations (estimation, download, run) run as a child process `python -m mst.<package>` on their resource; a second job on the same resource is refused. The reference is [tools.en.md](tools.en.md).

`dsh/` is the DSH plugin: it starts this server, registers the skills and adds a system prompt section from `workspace.json`. The skills explain to the agent how to assemble a scenario from the corpus, how to read a report and what "satellite rejected" means.

## 9. What the platform does not have

- a database, an HTTP API and a web layer: state is the workspace files;
- LangGraph and calculation agents: an external DSH agent assembles the scenario over MCP;
- data snapshots and portfolio slices: a position is set by its notional, and a model reads the paths;
- admission rules for `needs_review` and snapshot prices: the report shows the status but does not decide for the risk manager.
