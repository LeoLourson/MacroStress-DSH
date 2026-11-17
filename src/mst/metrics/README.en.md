# Checking a model on history

The criteria of trust in a model, with the formulas, are collected in [success-criteria.en.md](<../../../docs/concept/success-criteria.en.md>).

The package `mst.metrics` checks a model of any level on history: the macro model, the satellites and the position models. One procedure serves every plugin. The model is set back to a past date, estimated only on the data published by then, and its forecast is compared with what happened. The report first names what could not be checked, then shows where the forecast looks unreasonable, and only after that gives the scores. The scores describe a version of a plugin on a given set of dates and events; they are no proof of the quality of the product.

The normative rules are in the [general formalization](<../../../docs/concept/formalization.en.md>), section "Quality metrics". This page describes how they work in code and how to read the report.

## The window

The unit of a check is a *window*: a calculation date, the horizon months after the last month of the macro panel, and a scenario. There are two kinds.

- A date of the grid. The command `rolling` opens a window on every date of the grid in the `[rolling]` block of the [episode plan](<../../../bench/model-backtest/episodes.toml>), with the empty scenario. Nobody chose the dates, so quiet months and crises both enter the sample, and the calibration shares keep their ordinary meaning.
- An episode. The command `backtest` opens a window on the day before a past event starts. The scenario is what happened: the quantities in `observations` follow their realized path, the decisions in `interventions` are set at their realized level. The episodes are chosen with hindsight, so their calibration shares are for reference only.

A window ends `after_months` months after its start month; a series published with a lag adds its missing months to the horizon.

On every window the platform estimates the models of every level again. It caches the macro version in `<models>/backtest/` by the state of the data on the date, the model's configuration and the expert's answers, and the satellite bundle by its own inputs. The dates do not depend on one another, so before the windows open the missing versions of every date are estimated in parallel by `--workers` processes; the windows are then simulated by the same number of processes and the report is assembled in the original order; the paths depend on the seed of the plan alone, so the number of processes does not change the report. The default is half the logical processors, and one on a single-core machine. A refusal of the model on a date is cached too, with the hash of the code of the `mst` package: the same code on the same data refuses again, and fixed code estimates the date anew. With `--portfolio` the window also gets the history of the portfolio: the same positions valued by their own models over the past windows known on the calculation date.

## Levels and cells

A *cell* is a quantity at a month of the window's horizon whose fact is known. It holds the fact, the model's forecast and the benchmarks' forecasts on the same cell.

| Level | Quantity | Plain benchmark | Other comparisons |
|---|---|---|---|
| `macro` | macro variable | random walk with drift | the walk that knows the path of the fixed quantities; the base run |
| `driver/<quantity>` | satellite driver | random walk with drift | the value a year earlier with trend; the base run |
| `position` | return of a position, % of current value | historical simulation | the base run |
| `portfolio` | return of the portfolio | historical simulation | the base run |

The base run is the forecast of the same models on the same date under the empty scenario: the models are not told of the event, nor is it ruled out. The comparison with it shows what the scenario adds. A date of the grid has the empty scenario, so it has no base run and no conditional walk. The historical simulation is the return of the same positions over the past windows of the same length that ended before the calculation date.

A cell whose value the scenario fixes (an exact observation or an intervention) is shown in the table of quantities and kept out of the scores: the model was told its value.

## What fell out

The first section of the report lists everything that did not enter the scores, with the reason.

- A window refused (`refused`): there was enough data, but the model gave no version or the scenario could not be built. The scores do not cover it, and the grid with a refused date fails: a score over the surviving dates would hide the failures.
- A window skipped (`skipped`, the grid only): too little history for any model.
- A series left out (`left_out`): the model said through `ShortHistory` of `mst.model_sdk` that the history of this variable or driver is too short. The platform estimates the model again without it, and only its cells fall out. An episode is lost when its scenario fixes the series left out.
- A position excluded (`excluded`): the engine could not value it on the paths of the window.
- Cells without a finite path (`non_finite`).

## Sanity

The second section is a table per quantity, read before the scores.

- The share of cells whose band from the 5th to the 95th percentile leaves the admissible domain, and the share whose band is wider than the whole range of the history known on the date. Both width rules read that history from the whole series store, not from the months the model is estimated on: a model that starts later must not meet a narrower yardstick. The second rule reads only the cells whose history spans at least ten years: the range of a short history is no yardstick for a band, and such a cell stays out of the share.
- The share of cells whose band is wider than the range of the quantity's changes over as many months as separate the cell's month from the last known value. The band forecasts such a change, so a year-ahead band is compared with the moves of a year, not with the range of the levels. A mean-reverting series moves less over a year than a walk, and this rule catches its band. The rule reads the moves of one length when they cover at least five years: five years hold many moves of a year even when the levels have covered a small part of their range. The span is counted per length because a series can change its frequency: the housing price, annual until 2020 and quarterly since 2021, has only a few three-month moves.
- The median ratios of the band width to both ranges: 2 means a band twice as wide as the whole history or as any move of the same length, numbers one reads without the unit of the series.
- The share of cells where the scenario moved the median from the base run towards the fact. A move below a tenth of the base band is noise. Below one half the scenario pushes the forecast away from the fact.
- The plain benchmark of the level, the quantity's score against it, and the score of the level without the quantity. When the latter differs much, the aggregate rests on that quantity.

*Admission* is passed by every quantity on its own: fewer than 5 % of its cells may break each of the three band rules. One quantity over the threshold keeps the version out whatever the share over all cells, which the report prints for reference. On the episodes the probability of every interval observation under the base run must also be above 1 %. For an exact value that probability would measure the tolerance $\tau$, so point observations stay out of this threshold.

## The relative score

The error of a cell is CRPS, the continuous ranked probability score:

$$
\operatorname{CRPS}(F,y)=\int_{-\infty}^{\infty}\bigl(F(z)-\mathbf 1\lbrace z\ge y\rbrace\bigr)^2\,dz,
$$

where $F$ is the distribution function of the forecast and $y$ the fact. It is zero for a distribution concentrated at the fact and penalises both a shifted centre and needless width. CRPS keeps the unit of the quantity, so quantities are compared only through a ratio to a benchmark on the same cells. For quantity $q$ at month $t$ take the ratio $\rho_{q,t}$ of the mean scores of the first forecast and the second over the windows, and average the logarithms with equal weights, first over the quantities of a level, then over the months:

$$
S=\exp\Bigl(\frac1T\sum_{t}\frac1{Q_t}\sum_{q}\log\rho_{q,t}\Bigr),
$$

where $Q_t$ is the number of quantities of the level with both scores at month $t$ and $T$ the number of such months. With $S<1$ the first forecast wins. A quantity that barely moves weighs as much as any other, because its small score sits in both parts of its ratio. For every level the report gives every pair the rows carry: `model/conditional_walk`, `model/walk`, `base/walk`, `model/historical`, `base/historical`, `model/base`, `model/naive`.

The 95 % interval is a bootstrap with 1000 repetitions over whole windows, since the cells of one window are dependent: episodes are drawn one by one, dates of the grid in blocks as long as a window, because neighbouring windows overlap. The interval of the episodes is still narrow, as their windows overlap in part too. `--mc-bootstrap` measures the error of the finite number of paths apart.

## Calibration, halves and the use of the scenario

A right forecast puts the facts below its 5th percentile, between the 5th and the median, between the median and the 95th and above the 95th with shares 0.05, 0.45, 0.45 and 0.05. The report gives these shares per level, at months 1, 3, 6 and 12 and per quantity, with the coverage of the band and its median width. On the episodes the shares are for reference, since crises were selected.

The plan splits the windows in advance into tuning and validation: the episodes by their field `half`, the dates of the grid by `validation_from`. The report prints the halves beside the full table, so a reader sees whether a conclusion was fitted to the tuning windows.

The use of the scenario is the ratio `model/base` at the target level: over the level, per episode and as the share of episodes where it is below one. Scenarios of a person and of a language model are reported apart: a language model may remember the outcome of a past event.

## Deciding on a new version

With `--against` the check compares the version with the `--details` file of the incumbent built on the same plan. The comparison reads the validation half of the target level by $\log S$, against the conditional walk where the level has one and the plain benchmark of the level otherwise; the standard error is the spread of $\log S$ over resampled windows. The cells of the two versions must match, otherwise the check stops with the list of differences.

1. A version without admission is rejected whatever its score.
2. A version is accepted if it beats the benchmark and the incumbent by more than two standard errors and the admission shares of its worst quantity did not grow.
3. A version is rejected if it is worse by more than two standard errors.
4. Otherwise the incumbent stays: an undetected difference does not prove equality.

With one window the error of the score is unknown, and no difference is detected.

## The tail of the portfolio

With `--portfolio` the report gives the Acerbi–Székely statistic over the portfolio's cells at the last month of every window. Let $\Delta_e$ be the realized loss of the window, minus the return of the portfolio, $\mathrm{VaR}_e$ the forecast 95th percentile of the loss and $\mathrm{ES}_e$ the mean loss in the worst 5 % of outcomes. Then

$$
Z=\frac1{0.05\,E}\sum_{e=1}^{E}\frac{\Delta_e\mathbf 1\lbrace\Delta_e>\mathrm{VaR}_e\rbrace}{\mathrm{ES}_e}-1.
$$

With a right tail the expectation of $Z$ is zero, and a positive value means the extreme losses are understated. Its interval comes from the same bootstrap of windows, and the report prints $Z$ without the heaviest 1 % of paths beside it. Admission does not read $Z$.

## Running

The commands run from the root of the repository; replace the capitals with your own values.

```powershell
python -m mst.metrics rolling MODEL_ID --as-of DATE --store STORE --models MODELS --episodes EPISODES `
  [--drivers CATALOG] [--portfolio ID] [--details FILE] [--against FILE]
python -m mst.metrics backtest MODEL_ID --as-of DATE --store STORE --models MODELS --episodes EPISODES `
  [--drivers CATALOG] [--portfolio ID] [--details FILE] [--against FILE] [--save]
```

`--drivers` adds the level of the drivers, `--portfolio` the levels of the positions and the portfolio and needs `--drivers` too. `backtest` also runs the grid unless told `--no-rolling`. `--details` writes the full report with its cells, `--save` keeps the report without cells and its tables in `results/` beside the plan. A decision with `--against` needs the cells, so it reads a `--details` file. Exit code 1 means a refused window or a failed admission; the section of what fell out and the table of sanity name the cause.

## Where things are in the code

| Module | What it does |
|---|---|
| [windows.py](<windows.py>) | the window, the cache of versions, `ShortHistory`, the satellites and the portfolio of a window, the facts |
| [prepare.py](<prepare.py>) | the versions of every date estimated in parallel before the windows open |
| [levels.py](<levels.py>) | the cells of every level, the rows, what fell out |
| [valuation.py](<valuation.py>) | the paths in own units and the valuation of positions |
| [sanity.py](<sanity.py>) | sanity and admission per quantity |
| [quality.py](<quality.py>), [block.py](<block.py>) | the relative score and its interval |
| [scores.py](<scores.py>) | the shared judgement: levels, halves, quantities, tail |
| [decision.py](<decision.py>) | the two-standard-error rule |
| [es.py](<es.py>) | the ES of a run and the statistic $Z$ |
| [rolling.py](<rolling.py>), [backtest.py](<backtest.py>) | the grid with the price of events; the episodes with the gate of their conditions and the use of the scenario |
| [tables.py](<tables.py>), [cli.py](<cli.py>) | printing and the command line |

## Limits

Publication dates are approximate, and the facts come in today's edition of history: cutting by date does not restore earlier vintages of statistics. The historical simulation of positions reads today's panel on the calculation date and the same satellite bundle as the forecast. An event of probability 1 % is expected 2–3 times in 20 years of monthly history, so the check of the price confirms the tail weakly. Checking positions on every date of the grid estimates the satellite bundle on the whole catalog, which takes long; for a quick check pass a catalog cut down to the drivers of the portfolio.
