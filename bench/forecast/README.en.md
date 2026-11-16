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
| Events | The admission of the run of every scenario of `[events]` | The admission conditions of a run | The run is admitted |
| Events | The shift of the median return by month, of ES95 at the end of the horizon and over the path, the peak shift, the price of the observations | The base scenario, and the empty scenario on the same random numbers | For reference |

The events are ordinary runs of the workspace on the checked versions, with the plan's portfolio, draws and seed and the contribution of every element of the scenario. A run compares itself with the empty scenario on the same random numbers, and the set compares the median by month with the plan's base scenario on the same seed. The run ids, the remoteness of the interventions, the contributions and the comparison blocks are kept in `events` of the result.

## How the portfolio is compared with history

The model gives a distribution of the portfolio's return at the data date, while history gives one fact for each past window. One forecast cannot be judged against one fact: a hit or a miss of the band says nothing about the model, so the forecast is compared with history statistically. The scale check puts the forecast at the data date next to the cloud of all past windows of the same portfolio and asks whether the spread and the tail are the same. Forecasts at past dates, each from what was published by that day, are made by another set, the [check of the models on history](<../model-backtest/README.en.md>): with `--drivers` and `--portfolio` it judges the macro series, the shares, the positions and the portfolio with its tail of losses on a grid of dates and on events.

### Return on drawn and observed paths

The portfolio holds positions $p=1,\dots,P$ with current values $V_p$ and total value $V=\sum_p V_p$. The model of position $p$ takes a path of the macro variables and the drivers $x_{0:h}=(x_0,\dots,x_h)$ over $h$ months and returns the change of value $\Delta_p(x_{0:h})$. The driver of a share is its own price; the bond is revalued along the path of the curve. The portfolio's $h$-month return in percent is

$$
R_h(x)=100\,\frac{\sum_{p}\Delta_p(x_{0:h})}{V}.
$$

The forecast puts into $R_h$ the paths drawn by the macro model and the satellites: $N$ paths $x^{(1)},\dots,x^{(N)}$ (the plan's `draws`) give a sample $R_h^{(1)},\dots,R_h^{(N)}$, whose empirical distribution we denote $\hat F_h$.

History puts the observed path into the same function. The window starting at month $t$ is the observed values $x^{\text{fact}}_{t},\dots,x^{\text{fact}}_{t+h}$. A position enters the window when everything its model reads was observed in every month of the window; we write this as the indicator $o_{p,t,h}\in\lbrace 0,1\rbrace$. The return of the window is

$$
R^{\text{hist}}_{t,h}=100\,\frac{\sum_{p} o_{p,t,h}\,\Delta_p\bigl(x^{\text{fact}}_{t:t+h}\bigr)}{\sum_{p} o_{p,t,h}\,V_p}.
$$

The denominator takes only the observed positions, so a share listed later does not turn the early windows into a loss. Both sides use the same position models, and the forecast differs from history only through the paths: drawn against observed.

The first month of change is `history_start`. All horizons share one set of window starts, because each window must hold the plan's longest horizon. Yearly windows of adjacent months share 11 of 12 months, so there are twelve times as many yearly windows as independent years.

### Scale check

For a sample $S$ of size $n$ let $q_\alpha(S)$ be its $\alpha$-quantile and $S_{(1)}\le\dots\le S_{(n)}$ its ordered values, and put

$$
W(S)=q_{0.95}(S)-q_{0.05}(S),\qquad
\mathrm{ES}_{95}(S)=-\frac1k\sum_{j=1}^{k}S_{(j)},\quad k=\lceil 0.05\,n\rceil .
$$

$W$ is the width of the central 90 % band and $\mathrm{ES}_{95}$ is the mean loss of the worst 5 % of outcomes. With 2000 forecast paths $k=100$; with 130 historical windows $k=7$. The plan's gate for every horizon $h$ is

$$
0.75\le\frac{W(\hat F_h)}{W(R^{\text{hist}}_{\cdot,h})}\le1.40,\qquad
0.75\le\frac{\mathrm{ES}_{95}(\hat F_h)}{\mathrm{ES}_{95}(R^{\text{hist}}_{\cdot,h})}\le1.40 .
$$

The median, the quantiles and the mean are shown for reference and are not gated. The centre of the forecast depends on the state of the economy at the data date, while history averages over every state since `history_start`, so the centres are not expected to agree. The width and the tail measure the scale of risk, and a mismatch there is a model error. The check claims that the scale matches, not that the distributions do.

## Running

Run from the repository root in the project environment. `MACRO` and `SATELLITES` are version hashes of the workspace or their prefixes; the satellites must be estimated on that macro version.

```powershell
python -m mst.bench forecast --macro MACRO --satellites SATELLITES
python -m mst.bench forecast --macro MACRO --satellites SATELLITES --no-events
```

Without the events the set takes about 10 seconds; every event adds about half a minute. Exit code 1 means that a gate failed.

## Results

```text
bench/forecast/results/
  <data date>-<macro>-<satellites>.json   rows, gates, versions and commit
  <data date>-<macro>-<satellites>.md     the same tables
  latest.md                               the summary of the last run
```

Every number is a row with its level, check, item, horizon, model value, reference value, sample size and verdict: `pass`, `fail`, `info` (for reference) or `short` (too few windows or dates to decide). Rows of different versions are compared on the key level, check, item, horizon. The numbers of a report on the model are read from the JSON result, not assembled by hand.

## Limitations

The one-month width of a step series, such as the key rate, compares poorly: in most months the rate does not move, so the historical quantile width is small, while the model draws the change from a continuous law. The yearly width of such series compares correctly. The historical tail of the portfolio rests on a few overlapping windows of 2022, so the ES95 ratio is read with its number of windows.

The forecast and the history of the portfolio use the same position models. An error of a position model itself, such as a wrong sensitivity of the bond to the rate, is not seen by the scale check, because it sits on both sides of the comparison. The same holds for the level of the positions in the check on history, whose benchmark is the same historical simulation.

The agent checks — corpus search, scenario building and the interview context — are listed in the [index of the sets](<../README.en.md>).
