# First session

A first session follows the scheme: models are prepared from history, then a scenario is written, then the calculation runs. Once the server is connected, the client has seven tools: `jobs`, `series`, `corpus`, `scenario`, `run`, `portfolio`, `model`; the full reference is [../reference/tools.en.md](../reference/tools.en.md).

Long operations start as jobs: the tool returns a `job` at once, and progress is visible through `jobs(action="status", id=...)` and `jobs(action="log", id=...)`.

## 1. History: series

1. `jobs(action="state")`: what the workspace holds: model versions, the corpus index, the number of jobs.
2. `series(action="update")`: download all macro series. FRED needs a key in the environment variable `FRED_API_KEY`.
3. `jobs(action="status", id=...)` and `jobs(action="log", id=...)`: wait for the job to end.

## 2. Macro model, level 0

4. `model(kind="macro", action="estimate", as_of="2026-10-29")`: estimate a macro model on the data known by that date. Two are installed, `bvar.sv` and `msvar.markov`; with more than one installed, `model_id` names the one to estimate. The result is a version with a hash in its name.

## 3. Satellites, the levels above

5. `series(action="update", quantity="share_price")`: find the objects of the quantity and download their series; likewise for `bond_spread`, `overdue_share`, `housing_price_primary`, `housing_price_secondary`.
6. `model(kind="satellites", action="estimate", version=<macro version hash>)`: assemble the satellite bundle of this macro version: each quantity is estimated by the model assigned to it (`hier.gibbs` by default; `assignment` names another).
7. `model(kind="satellites", action="describe", version=<hash>)`: which quantities passed the out-of-sample check and why the others were rejected.

## 4. Scenario

8. `corpus(action="search", query=..., as_of=...)`: corpus fragments known on the date; `corpus(action="index")` rebuilds the index as a job.
9. `scenario(action="save", id=..., spec={...})`: save a scenario: the narrative, the causes, the conditions, the impacts and the person who approved it; `scenario(action="validate", id=...)` names everything that keeps it from being calculated.

## 5. Calculation and result

10. `portfolio(action="save", id=..., spec={...})`: save a portfolio; `portfolio(action="check", id=..., scenario=...)`: check it against the scenario.
11. `run(action="start", scenario=..., portfolio=..., draws=..., seed=..., targets=...)`: start a run; `run(action="result", id=...)`: read the conditional forecast of the target indicators ($P_5$/$P_{50}$/$P_{95}$ by month) and the portfolio report with ES95, the scenario price $\mathrm{dist}(s)$ and the coverage.

## 6. Quality metrics

12. `model(kind="macro", action="backtest", as_of=...)`: a check on historical episodes: CRPS, the coverage of the forecast bands and the admission of models (`mst.metrics`).

## Further reading

The method behind the calculation steps is in [../concept/scenario-formalization.en.md](../concept/scenario-formalization.en.md), the number contracts in [../reference/platform.en.md](../reference/platform.en.md), the general scheme in [../concept/formalization.en.md](../concept/formalization.en.md).
