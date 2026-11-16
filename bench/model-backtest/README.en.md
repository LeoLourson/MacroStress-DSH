# Checking the models on history

The set checks the models of every level on history: the macro model, the satellites and the position models of a portfolio. One plan gives a grid of dates and past events. On the grid the model gives its ordinary forecast on dates nobody chose. On the events it is told what happened then to the central bank's decisions, the rouble or oil, and the check reads how well it forecasts the rest. Both checks follow one procedure of the package `mst.metrics`. The report starts with what fell out, goes on with the sanity of every quantity, and gives the scores last. The windows, cells, scores, admission and the decision on a version are described in the [package guide](<../../src/mst/metrics/README.en.md>).

The numbers of the set describe a version of a plugin on these dates and events. They are no forecast for a talk while the sanity check names unreasonable quantities.

## What the plan holds

The set is defined by one file, the [episode plan](<episodes.toml>): the common parameters, the grid of dates and the episodes.

The common parameters are 1000 paths, the seed and a window of 12 months after the start month. The `[rolling]` block sets the grid: a step of 3 months and the validation half from January 2021. The `[[exclusions]]` block concerns the AA, A and BBB spreads: in January 2021 the Moscow Exchange changed the scale of their indexes. If the history of a window ends before the splice, the facts of the spreads from January 2021 are on another scale and stay out of the scores.

An episode is a past event recorded as a scenario. The plan holds 20 episodes, R13–R32, from March 2014 to June 2025. Episodes R1–R12 are left out: the training history before them is shorter than two years.

| Field | What it sets |
|---|---|
| `start` | The start month of the event; the calculation date is the last day of the month before |
| `observations` | Quantities the economy produces itself (the rouble, oil, inflation, a spread, an index): in the scenario they follow their realized monthly path to the end of the window |
| `interventions` | Decisions of a subject outside the models, now only the key rate of the central bank: in the scenario its realized level is set in those months |
| `half` | `tuning`: the model may be tuned on the episode; `validation`: the episode is for the final check only; 12 and 8 episodes |
| `source` | Who wrote the scenario: `manual` a person, `llm` a language model; all episodes are `manual` now |
| `card`, `reason`, `sources` | The corpus card of the same event, the reason of the event in words, the sources |

An observation differs from an intervention by who chose the value, not by the quantity. An OPEC quota moves oil but does not set its price, so oil stays an observation; a change of the key rate is a decision of the central bank, and the plan records it as an intervention.

## Running

Run the command from the root of the repository in the project environment. The check of the macro model `rw.joint` on the data of the workspace:

```powershell
python -m mst.metrics backtest rw.joint --as-of 2026-10-29 --store workspace/series `
  --models workspace/models --episodes bench/model-backtest/episodes.toml --save
```

`--drivers workspace/drivers/catalog.json` adds the level of the drivers. `--portfolio ID` adds the levels of the positions and the portfolio with its tail of losses and needs `--drivers` too. Estimating the satellites on every date takes long, so for a portfolio a catalog cut down to its drivers is convenient. `--details FILE` writes the full report with its cells, `--against FILE` compares the version with the `--details` file of the incumbent. `python -m mst.metrics backtest --help` prints the help.

The model versions estimated on the dates of the episodes and the grid are cached in `backtest/` inside the models folder. The cache key holds the data on the date, the code and the settings of the model, so a rerun of the same version is faster and a changed model never gets an old estimate.

## Results

`--save` writes the report to `results/` beside the plan, over the previous report of the same model; earlier ones stay in the Git history. Saving needs the whole plan: a report on part of the episodes (`--only`) cannot be compared with others.

```text
bench/model-backtest/results/
  <model>.json   the report without cells: what fell out, sanity, scores, halves, tail, versions, data and code
  <model>.md     the same tables the command prints
```

The cells stay out of the result so that the file can be kept in the repository; a decision on a version reads the `--details` file. Exit code 1 means a refused window or a quantity that failed admission; the sections of what fell out and of sanity name the cause.

## Limits

There are 20 episodes and their windows partly overlap, so the interval over episodes is narrow, and the selection of crises shifts the calibration shares. Publication dates are approximate and the facts come in today's edition of history: cutting by date does not restore earlier vintages of statistics. The price of events on the grid is checked on the family `[[rolling.price_events]]`; the current plan sets none, so the report has no check of the price.
