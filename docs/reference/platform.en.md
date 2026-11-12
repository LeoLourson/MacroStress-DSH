# Platform: calculation contracts

The general scheme of the system is in [../concept/formalization.en.md](../concept/formalization.en.md); this page gives the contracts that implement it.

MacroStress is a modular monolith with no database, web layer or orchestrator. State is the files of one directory `workspace/`, and long operations run as jobs. The core knows neither asset types nor concrete models: it provides data, model versions, the scenario, the calculation dispatcher, the aggregate and the report. Everything domain-specific (stock, bond, credit, housing, the macro model and the satellite of a quantity) connects through plugin packages ([../howto/plugins.en.md](../howto/plugins.en.md)). The data flow and the workspace layout are described in [../concept/architecture.en.md](../concept/architecture.en.md).

## 1. Layers

| Layer | Package | What it gives |
|---|---|---|
| Data | `mst.data` | Monthly series with the publication date of each value; `as_of` does not see future revisions |
| Level 0 | `mst.macro` + plugin | The variable registry, the panel on a date, versions; the model itself (BVAR `bvar.sv`, MS-VAR `msvar.markov`, joint Student walk `rw.joint` and any other) is a `MacroModel` plugin |
| Levels above | `mst.drivers` + plugins | Objects and quantities, history preparation, the version bundle with its own satellite model per quantity (`SatelliteModel`), the out-of-sample check in the part of the version, the loop over levels |
| Scenario | `mst.scenario` | $s=(N,I,\mathcal A)$: the narrative, the interventions and the observations; `translate` turns units and dates into a plan, `build` gives the paths, `overlay` the tilt under observations on means, `reverse` the reverse problem |
| Model SDK | `mst.model_sdk` | The scenario plan with interventions, the default structural form and declared forms, the inner groups of paths, estimation contexts, the three model steps, the conditional machinery, the price of a law |
| Position SDK | `mst.plugin_sdk` | `Paths`, requirements, prepared inputs, the result, the SDK version |
| Versions | `mst.models` | `estimate-macro`, `estimate-satellites` (assignment of quantities to models), `diagnose-satellites` (what a saved bundle implies: rho, elasticities, the pull to the long-run level), list and description |
| Calculation core | `mst.engine` | The model catalog, input preparation, position calculation, result check |
| Plugins | `mst.plugins.<id>` | Position, macro and satellite models; the registry is assembled from packages, entry points and `MST_PLUGINS_PATH` |
| Quality | `mst.metrics` | Cells $(e,t,q)$, CRPS, band coverage, admission, benchmarks, the episode backtest, the rolling check |
| Portfolio | `mst.portfolio` | The portfolio, the check against versions, value and report |
| Run | `mst.runs` | Paths, the conditional forecast of indicators and results in `runs/<id>/`; the same seed gives the same numbers |
| Server | `mst.mcp`, `mst.jobs`, `mst.store` | MCP tools, jobs, the workspace directory |

## 2. Workspace and versions

A model version is an immutable directory `<sha256>/` with `meta.json` and `arrays.npz`. The hash covers the metadata and the arrays, and a changed file is rejected on load. A version names the model that estimated it (`meta["model"]`) and only this model reads it back; the platform checks the mandatory keys (dates, variables, shocks, panel), and the rest is the model's own format. A satellite version is a bundle: a part for each quantity, its own model per part, the residual correlation of all drivers with history. Satellites are estimated on the panel of a specific macro version and store its hash. A scenario names no versions: the calculation request chooses them, and a run fixes the pair in `meta.json`, so a change of version does not mix the numbers of one run.

Series store each value with its publication date. `as_of(day)` returns what was known on that day; on the first download the revision date is estimated and marked `approximate = 1`. Model estimation, the scenario and the run read only data no later than their date.

The answers of a model's elicitation live in `workspace/elicitation/<date>/<key>.json`: a model whose estimate reads answers rather than series alone declares their keys through the `input_keys` method of the model SDK, and `elicit(next)` gives one small question while `elicit(answer)` validates and saves a part and returns the next question. Partial answers stay in `elicitation/.steps/` and cannot enter an estimate. Assembly supplies no numbers and preserves the existing full-answer validation; full question/write actions remain under `model(kind="inputs")`; no human approves the answers — their form and the model itself check them. Such a model's estimate receives the answers folder in its context (`inputs`), a checkpoint folder of its own (`checkpoints`, removed once the version is saved) and a callable telling whether a fragment of the model corpus is visible (`fragment`). The models have a corpus of their own — `workspace/corpus-model` with a separate manifest and index (the `market` source kind — market expectations); the `corpus` tool picks it with `corpus="model"`, and the agent sees only material published before the calculation date.

The price of a scenario reads the law of the observations from the model: `constraint_law` gives rows mixing gaussian laws, and when no mixture carries the law (the model's inputs are nonlinear), the model may return its own box probabilities through `log_box_probability` — the price and its error are then the mean and the spread of `-log P(S|R)` over the values of `R`, and the one-sided price takes its medians from an unconstrained simulation with the same interventions.

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

A scenario is the file `scenarios/<id>.json`: the narrative with references to corpus fragments, the interventions, the observations and the name of the person who approved it. It carries no model versions, no horizon and no path count — those belong to the calculation request — so one file runs unchanged on any model system. Numbers are written in the quantity's own units (`level`, `change`, `percent`) and in calendar months `YYYY-MM`; `mst.scenario.translate` maps them to the registry scale and to horizon months counted from the last month of the model panel, not from the calculation date.

An intervention sets (`set`) or shifts (`shift`) the value of a quantity in the named months, and a pair (quantity, month) belongs to at most one intervention. In the default form the installed value is put on the path after the draw (`mst.model_sdk.interventions.apply_default`) and does not spread within level 0 in the same month: the other variables get only what the model gives through the lags, and the lower levels through their own models. A model with a structural form executes the intervention inside its own recursion and returns the paths with it already applied. An observation is a requirement on the measure: a band on a quantity (an exact value is expanded to $value\pm\tau$, where $\tau$ is a tenth of the typical monthly move), bounds on a mean, or a probability of at least a stated value. The observation refers to the final path, so the macro model meets it after the installed values are substituted into the constraints; `overlay` performs the tilt for observations on means, and observations on drivers are met by reweighting inside the groups when `inner` is at least 2.

`mst.scenario.build` builds the conditional forecast of the macro model, returns the variables to their own units, computes the driver paths on the same draws and the price $\mathrm{dist}(s)$ with its one-sided companion $\mathrm{dist}^+(s)$. The price measures the observations alone: a scenario of interventions only has a price of zero, because the event is imposed and not observed. An unknown quantity, a month before the last panel month or beyond the horizon, a mixed observation, a repeated (quantity, month) pair, `percent` on a quantity outside the `log100` scale, a foreign satellite version and a missing approval are refusals naming the reason.

`mst.runs` writes the request, the metadata, the paths, the results and the report to `runs/<id>/`. The request names the scenario, the number of parameter draws, the seed, `horizon` (twelve by default and never shorter than the last month a statement names), `inner` (how many paths share one draw of parameters and interventions) and the `macro`/`satellites` versions; without them the newest finished pair of the workspace is used. A run is a function of the scenario, the versions, these numbers and the seed: the same request gives the same numbers bit for bit. The portfolio report adds losses **within a draw** and only then takes P5, P50, P95 and ES95 — at the end of the horizon, by every month and over the whole path (`losses_by_month`, the monthly array in `path_losses.npz`); the sum of position P95 values is not passed off as the portfolio's P95. The report compares the same losses with the empty scenario $s_0$ on the shared random numbers (`comparison`: the shift of the centre apart from the change of width, each with an interval), and admits the calculation only while fewer than five percent of the cells of month and quantity break each admission condition — otherwise the status is `not_admitted`, the broken conditions and quantities are named, and the numbers stay for diagnostics. The `interventions` block names the quantity, the month, the kind, the share of groups in which the intervention was active, and the remoteness `alpha` — the quantile of the installed value in the base law, with a distribution across groups for a random intervention. The share of paths on which the observations need a driving shock above six standard deviations (`DRIVING_LIMIT`) raises the flag `driving_shock_above_limit` once it reaches the same limit: the responses to such a shock are extrapolated. It refuses nothing, because the price measures how unlikely the event is. The report also names the structural form of every model that executed the interventions. An excluded position is named with a reason, and the coverage is the share of value the numbers cover; `dist` is the price of the scenario with its components and its `estimated` mark: absent for the empty scenario, and exactly zero, written as zero, for a scenario of interventions only.

## 8. MCP and DSH

`mst.mcp` is the only server: a tool parses arguments and calls one package function or starts a job. Long operations (estimation, download, run) run as a child process `python -m mst.<package>` on their resource; a second job on the same resource is refused. The reference is [tools.en.md](tools.en.md).

`dsh/` is the DSH plugin: it starts this server, registers the skills and adds a system prompt section from `workspace.json`. The skills explain to the agent how to assemble a scenario from the corpus, how to read a report and what "satellite rejected" means.

## 9. What the platform does not have

- a database, an HTTP API and a web layer: state is the workspace files;
- LangGraph and calculation agents: an external DSH agent assembles the scenario over MCP;
- data snapshots and portfolio slices: a position is set by its notional, and a model reads the paths;
- admission rules for `needs_review` and snapshot prices: the report shows the status but does not decide for the risk manager.
