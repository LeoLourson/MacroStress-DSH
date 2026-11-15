# Forecast check by level

The set answers whether the numbers of the models can be trusted: whether the forecast is as wide as the past, whether the shares in it move together as they did, whether the portfolio's gains and losses have the scale of history, and whether the forecasts of past dates covered what then happened. It reads a macro version, a satellite version and the [plan](<plan.toml>), forecasts the empty scenario of the plan's portfolio, and compares every level with history by the same rules. The thresholds of the plan are fixed before a result is read; changing one is a commit of the plan.

## What is checked

The *width* is the 95th minus the 5th percentile of the $h$-month change. Quantiles are used rather than the standard deviation because, with heavy-tailed monthly steps, the sample deviation of a few thousand paths rests on single extreme paths. History is every $h$-month window whose changes start on or after `history_start` (January 2015, after the rouble began to float) and touch no statistical splice listed in `splices`. The windows overlap, so their number exceeds the number of independent observations.

| Level | Check | Compared with | Gate |
|---|---|---|---|
| Macro | Width of the 1-, 3- and 12-month change of every series in the model scale | The same width of historical changes | Ratio 0.75–1.40 for 80 % of the series of each horizon |
| Portfolio shares | Width of every share's return, $100\ln(P_h/P_0)$ | Historical returns | Ratio 0.75–1.40 for 80 % of the shares |
| Portfolio shares | Mean pair correlation of 12-month returns | The same correlation in history, each pair on its shared windows | Difference at most 0.15 |
| Portfolio shares | The satellite's error on the held-out history | The error of an AR(1) without macro variables | For reference: the satellite decides what it serves |
| Portfolio | Width of the return and the expected loss of the worst 5 % (ES95) over 1, 3 and 12 months | The same portfolio valued by the same position models on the observed paths of every historical window | Ratio 0.75–1.40 |
| Portfolio | Percentiles and mean | The same | For reference |
| Out of sample | Share of facts inside the 90 % band at every level and horizon | The nominal 90 % | 80–97 % |
| Out of sample | Mean of $F(y)$, CRPS against the historical simulation, the Acerbi–Székely statistic and the coverage of the historical simulation itself, for the portfolio | 0.5; the simulation's CRPS; 0; 90 % | For reference |
| Events | The admission of the run of every scenario of `[events]` | The admission conditions of a run | The run is admitted |
| Events | The shift of the median return by month, of ES95 at the end of the horizon and over the path, the peak shift, the price of the observations | The base scenario, and the empty scenario on the same random numbers | For reference |

In history the portfolio is held as in the forecast: in every window the position models receive the observed paths of the macro variables and the drivers and value the change. A position enters a window when everything it reads was observed in every month of the window, and the return of the window is the sum of those positions' changes over the sum of their current values. A share listed later therefore does not turn a window into a loss.

The events are ordinary runs of the workspace on the checked versions, with the plan's portfolio, draws and seed and the contribution of every element of the scenario. A run compares itself with the empty scenario on the same random numbers, and the set compares the median by month with the plan's base scenario on the same seed. The run ids, the remoteness of the interventions, the contributions and the comparison blocks are kept in `events` of the result.

Out of sample, on every date of the plan, quarterly from 2017 to 2025, the models are estimated again on what was published by the 15th of the next month, and the forecast is compared with what happened. The facts come from the history of the working version. *CRPS*, the continuous ranked probability score, compares the whole forecast distribution with the fact; lower is better. The historical simulation is the distribution of the returns of the same positions over the windows that ended by the forecast date. The Acerbi–Székely statistic is zero when large losses are on average as the tail predicted; a positive value means the tail was underestimated.

## Running

Run from the repository root in the project environment. `MACRO` and `SATELLITES` are version hashes of the workspace or their prefixes; the satellites must be estimated on that macro version.

```powershell
python -m mst.bench forecast --macro MACRO --satellites SATELLITES
python -m mst.bench forecast --macro MACRO --satellites SATELLITES --no-rolling --no-events
```

Without the out-of-sample dates and the events the set takes about 10 seconds. The dates add about two minutes: the macro model and the satellites are estimated on every date. Every event adds about half a minute. Exit code 1 means that a gate failed.

## Results

```text
bench/forecast/results/
  <data date>-<macro>-<satellites>.json   rows, gates, versions, commit and the out-of-sample dates
  <data date>-<macro>-<satellites>.md     the same tables
  latest.md                               the summary of the last run
```

Every number is a row with its level, check, item, horizon, model value, reference value, sample size and verdict: `pass`, `fail`, `info` (for reference) or `short` (too few windows or dates to decide). Rows of different versions are compared on the key level, check, item, horizon. The numbers of a report on the model are read from the JSON result, not assembled by hand.

## Limitations

The satellites of the out-of-sample dates are estimated on a reduced catalog: the drivers of the portfolio, the objects above them and the drivers of those objects. A full estimate takes about half an hour a date. The reduced one pools fewer siblings, so its coefficients are close to those of the working version but not equal. When on an early date the satellite does not serve a share because its equation is not more accurate than an AR(1), the portfolio of that date is valued without it, and the composition is recorded in `rolling_dates`.

The one-month width of a step series, such as the key rate, compares poorly: in most months the rate does not move, so the historical quantile width is small, while the model draws the change from a continuous law. The yearly width of such series compares correctly. The historical tail of the portfolio rests on a few overlapping windows of 2022, so the ES95 ratio is read with its number of windows.

The agent checks — corpus search, scenario building and the interview context — are listed in the [index of the sets](<../README.en.md>).
