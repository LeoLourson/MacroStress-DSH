# Model check on history

The set answers how accurately the macro model and the satellites forecast the past, and whether a new version of a model should replace the current one. These are different questions, and one score cannot answer both. The episode check puts the model on the day before a known crisis, tells it what then happened to the central bank's decisions, the rouble or oil, and measures how well it forecasts the other quantities. The rolling check puts the model on dates nobody selected and compares its ordinary forecast with a random walk. The portfolio numbers and the band coverage for the report are checked by another set, [forecast](<../forecast/README.en.md>): it has its own out-of-sample date grid, unrelated to the grid of this set.

The set is defined by one file, the [episode plan](<episodes.toml>), and computed by the package `mst.metrics`. The full description of the metrics, intervals and decision rules is in the [package guide](<../../src/mst/metrics/README.en.md>).

## What the plan holds

An episode is a past event written down as a scenario. The plan has 20 episodes, R13–R32, from March 2014 to June 2025. Episodes R1–R12 are not in the plan: the training history before them is shorter than two years.

| Field | What it sets |
|---|---|
| `start` | The month the event starts; the calculation date is the last day of the month before |
| `observations` | Quantities the economy produces itself (the rouble, oil, inflation, a spread, an index): in the scenario they follow their realized monthly path to the end of the window |
| `interventions` | Decisions of a subject outside the models, at present only the key rate of the central bank: in the scenario its realized level is set in those months |
| `half` | `tuning`: the model may be tuned on the episode; `validation`: the episode is kept for the final check; 12 and 8 episodes |
| `source` | Who wrote the scenario: `manual` for a human, `llm` for a language model; all episodes are `manual` at present |
| `card`, `reason`, `sources` | The corpus card of the same event, the cause of the event in words, and the sources |

What separates an observation from an intervention is who chose the value, not the quantity. An OPEC quota moves oil but does not set its price, so oil stays an observation; a change of the key rate is a decision of the central bank, and the plan records it as an intervention.

Besides the episodes the plan sets the common parameters: 1000 paths, the generator seed and a window of 12 months after the start. The `[rolling]` block sets the rolling grid: a step of 3 months and the validation half from January 2021. The `[[exclusions]]` block concerns the AA, A and BBB spreads, whose index scale the Moscow Exchange changed in January 2021. When the history of an episode ends before the splice, the spread facts from January 2021 on are on another scale and do not enter the score.

## Episode check

On the calculation date of episode $e$ the set estimates the models again from what was published by that date, and they forecast under the episode's scenario $s_e$. A cell $(e,t,q)$ is quantity $q$ in month $t$ of the window; it holds the fact $y$ and the forecast distribution $F$. Cells of the quantities in `observations` and `interventions` do not enter the score: the model was told their values, so there is nothing to guess. The score comes from the other quantities, for example the stock index and the spreads in an episode where the rouble is given.

The error of a cell is CRPS, the continuous ranked probability score:

$$
\operatorname{CRPS}(F,y)=\int_{-\infty}^{\infty}\bigl(F(z)-\mathbf 1\lbrace z\ge y\rbrace\bigr)^2\,dz .
$$

It is zero for a distribution concentrated on the fact and penalizes both a shifted centre and excess width. To add errors of a rate in percentage points to errors of the rouble in roubles, the set divides CRPS by $\sigma_q(t)$, the median absolute historical change of the quantity over $t$ months. The normalized errors are averaged first over the episodes within a quantity, then with equal weights over the quantities, then over the horizons, so a series with more cells does not outweigh the rest.

The same score is computed for three references on the same cells:

- the conditional walk `conditional_walk` knows the same path of the given quantities as the model: it takes the monthly change of every other quantity from a regression on the changes of the given quantities over the history known on the calculation date, plus random noise; the model is useful if it beats this simple transfer;
- the random walk with drift and, for target indicators, last year's value with a trend do not know the event; beating them may come from knowing the future condition alone;
- the base calculation is the unconditional forecast of the same model on the same date: the model is not told about the event, nor is the event ruled out, and crisis paths enter the forecast in the share the model finds likely; the difference from it is the value of the scenario itself. If the model's joint law is right, knowing the realized path of the given quantities does not worsen the forecast of the rest on average, since CRPS is a strictly proper score. A larger error with the scenario therefore means the model passes the move of the given quantities to the rest wrongly.

The value of a scenario is the share of episodes where the forecast under the scenario beats the base, and the mean difference of CRPS, reported apart for scenarios of a human and of a language model. A language model may remember the outcome of a past event, so its scenarios can only be checked fairly on events after the date they were written.

The report shows coverage as the shares of facts in four regions of the forecast: below the 5th percentile, between the 5th and the 50th, between the 50th and the 95th, above the 95th. For a correct forecast the shares are 0.05, 0.45, 0.45 and 0.05. On episodes they are for reference: the plan selects crises, and even a correct model need not show the usual frequencies on crises alone.

## Admission and the decision on a version

Before accuracy is compared, a version must pass admission. Fewer than 5 % of the cells may have a band leaving the admissible domain of the quantity in the registry, and fewer than 5 % a band wider than the historical range of the quantity; the set counts the shares for each rule apart. The probability of every interval observation of the scenario in the base calculation must exceed 1 %.

With `--against` the set compares the new version with the report of the current one on the same cells of the validation half:

1. A version without admission is rejected whatever its CRPS.
2. A version is accepted if it beats both the reference and the current version by more than two standard errors and the admission shares did not get worse.
3. A version is rejected if it is worse than the current one by more than two standard errors.
4. Otherwise the current version stays: a difference not detected does not prove equality.

The standard error and the 95 % interval of the aggregate come from a bootstrap of whole episodes, 1000 resamples. Cells of one episode are dependent, so the bootstrap resamples episodes, not cells.

## Rolling check

The `backtest` command runs the rolling check together with the episode check unless `--no-rolling` is given; the `rolling` command runs it alone. On every date of the grid the set estimates the model again, and it gives an unconditional forecast without a scenario. The score is the mean CRPS of the model over the mean CRPS of a random walk on the same cells, averaged in logarithms with equal weights over the quantities and the horizons. A value below one means the model is more accurate than the walk. There is no division by the median change here: for the key rate it is close to zero, and one quantity would weigh as much as dozens of others. The windows of neighbouring dates overlap, so the interval uses moving blocks as long as a window. The model's parameters are chosen on the dates before `validation_from`, and the decision is read on the dates after it.

## Running

Run the command from the repository root in the project environment. An example for the macro model `rw.joint` on the workspace data:

```powershell
python -m mst.metrics backtest rw.joint --as-of 2026-10-29 --store workspace/series `
  --models workspace/models --episodes bench/model-backtest/episodes.toml --save
```

`--as-of` is the data date; `--store` and `--models` are the stores of series and model versions. With `--satellites --drivers FILE` the set checks the drivers too, and with `--portfolio ID` also the tail of the portfolio's losses by the Acerbi–Székely statistic. `--against REPORT.json` turns on the decision on a version. `python -m mst.metrics backtest --help` prints the help.

The set caches the model versions estimated on the dates of the episodes and the grid in `backtest/` inside the models folder. The cache key includes the data on the date, the code and the settings of the model, so a repeated run of the same version is faster and a changed model never gets an old estimate.

## Results

`--save` writes the report to `results/` beside the plan, over the previous report of the same model; earlier ones stay in the Git history. Saving needs the whole plan: a report on part of the episodes (`--only`) cannot be compared with others.

```text
bench/model-backtest/results/
  <model>.json   the full report: cells, aggregates, admission, value, halves, versions, data and code
  <model>.md     the same tables the command prints
```

Exit code 1 means that a required episode failed or admission was violated; the list of failures in the report names the cause.

## Limitations

There are 20 episodes and their windows partly overlap, so the interval of the aggregate is wide, and the selection of crises shifts the coverage frequencies. Publication dates are approximate and the facts are read in today's vintage of history: a cut by date does not restore earlier vintages of the statistics. The rolling grid prices a scenario on the family `[[rolling.price_events]]`, and the present plan does not define that family, so the report has no price check.
