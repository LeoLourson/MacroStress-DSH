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

The events are ordinary runs of the workspace on the checked versions, with the plan's portfolio, draws and seed and the contribution of every element of the scenario. A run compares itself with the empty scenario on the same random numbers, and the set compares the median by month with the plan's base scenario on the same seed. The run ids, the remoteness of the interventions, the contributions and the comparison blocks are kept in `events` of the result.

## How the portfolio is compared with history

The model gives a distribution of the portfolio's return at the data date, while history gives one fact for each past window. One forecast cannot be judged against one fact: a hit or a miss of the band says nothing about the model, so the forecast is compared with history statistically. The scale check puts the forecast at the data date next to the cloud of all past windows of the same portfolio and asks whether the spread and the tail are the same. The out-of-sample check makes forecasts at past dates, each from what was published by that day, and asks how often the fact fell inside the band and whether it was biased against the forecast.

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

### Out-of-sample check

The forecast dates $m$ run every `step_months` from `first` to `last` of the plan's `[rolling]` section; the last date leaves 12 observed months. On every date the macro model and the satellites are estimated again on what was published by the 15th of the next month and draw the `[rolling]` number of paths, which gives the distribution $\hat F_{m,h}$. The fact $y_{m,h}=R^{\text{hist}}_{m,h}$ comes from the same history calculation as in the scale check, for the positions the models of that date can value. The facts are read in the vintage of history on which the working version is estimated.

At every date and horizon the set records the hit of the band and the rank of the fact:

$$
H_{m,h}=\mathbf 1\bigl\lbrace q_{0.05}(\hat F_{m,h})\le y_{m,h}\le q_{0.95}(\hat F_{m,h})\bigr\rbrace,\qquad
u_{m,h}=\hat F_{m,h}(y_{m,h})=\frac1N\,\#\bigl\lbrace i:\ R^{(i)}_{m,h}\le y_{m,h}\bigr\rbrace .
$$

For a correct model the coverage $C_h$, the mean of $H_{m,h}$ over the dates, is close to 0.90, and the plan's gate is 0.80 to 0.97. The rank $u$ of a correct model is uniform on $[0,1]$ with mean close to 0.5. A mean above 0.5 means that the facts were systematically better than the forecast, below 0.5 worse. The mean rank catches a bias but not a band that is too narrow, since misses on both sides also give 0.5, so coverage and mean rank are read together. The coverage of the macro series and the shares uses the same formula, with $y$ the value of the series $h$ months after the date and $\hat F$ the forecast distribution of that value.

The reference is the historical simulation. Its distribution $\hat G_{m,h}$ at date $m$ is the returns $R^{\text{hist}}_{s,h}$ of the same positions over every window that ended by that date, $s+h\le m$. The simulation gets the same coverage and the same continuous ranked probability score, CRPS:

$$
\mathrm{CRPS}(F,y)=\mathbb E\lvert X-y\rvert-\tfrac12\,\mathbb E\lvert X-X'\rvert,
$$

where $X$ and $X'$ are independent with distribution $F$. CRPS compares the whole distribution with the fact rather than one band; a lower mean over the dates is better.

The yearly tail is tested with the Acerbi–Székely statistic on losses $L_m=-y_{m,12}$. Let $\mathrm{VaR}_m$ be the 95th percentile of the forecast losses at date $m$, $\mathrm{ES}_m$ the mean of the forecast losses at or above $\mathrm{VaR}_m$, and $M$ the number of dates. Then

$$
Z=\frac{1}{0.05\,M}\sum_{m}\frac{L_m\,\mathbf 1\lbrace L_m>\mathrm{VaR}_m\rbrace}{\mathrm{ES}_m}-1 .
$$

$Z$ is near zero when the forecast tail is right; a positive value means large losses came more often or deeper than the tail promised.

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

The forecast and the history of the portfolio use the same position models. An error of a position model itself, such as a wrong sensitivity of the bond to the rate, is seen by neither the scale check nor the out-of-sample check, because it sits on both sides of the comparison. The out-of-sample dates overlap too: the yearly facts of adjacent quarters share 9 of 12 months, so 35 yearly facts carry about nine independent years. The coverage gate of 80–97 % allows for this spread, about 5 points.

The agent checks — corpus search, scenario building and the interview context — are listed in the [index of the sets](<../README.en.md>).
