# mst.metrics

A check of models on historical episodes. Models are rolled back to a date before the start of a past event and, on the data known only by that date, predict what happened. The forecast is compared with the fact for each quantity and each month of the horizon, so you see where exactly a model errs, beyond "fine on average".

What is compared and by what rules is described in the [general scheme of the system](../../../docs/concept/formalization.en.md), section "Quality metrics". This page covers how it works in code and how to read the report.

```powershell
python -m mst.metrics backtest [MODEL_ID] --as-of DATE --store DIR --models DIR --corpus FILE \
  [--drivers FILE] [--satellites] [--only ID ...] [--details FILE] \
  [--against REPORT.json] [--target-level LEVEL] [--portfolio ID] [--mc-bootstrap]
```

## Words the report needs

**Episode**: a past event with a recorded scenario: which variables are set by the fact, which shocks explained the event, by which sources. The calculation date is the day before the episode starts; models are estimated anew on the data of this date and cannot peek at the future. The list of episodes is [episodes.toml](episodes.toml). An episode is data, not part of a model, and the set grows as new events are analyzed.

**Cell**: an episode, a month of the horizon and a quantity. A cell has a fact and a forecast distribution. Cells of one episode are dependent, since they share one model, one data set and one scenario, so shares are counted over cells and intervals over episodes. In a cell whose value the scenario sets exactly, the forecast repeats the fact and checks nothing: it is printed but does not enter the scores.

**Halves.** Episodes are split in advance into tuning (`tuning`) and validation (`validation`). The split by halves shows whether a conclusion was fitted to the episodes the models were tuned on; the final table is built over all episodes.

**Target level**: the level the verdict is passed on and the value of the scenario is calculated on: `macro` by default, and with `--satellites` you can name a driver level, for example `driver/share_price`.

## What is calculated

**CRPS**: the distance between the forecast distribution and the fact. It penalizes both bias and excess width; lower is better, and zero is an exact hit.

**Normalization.** To compare quantities in different units, CRPS is divided by the quantity's typical move over the same horizon, the median absolute change over $t$ months in history. If history is too short for the horizon, $\sigma_q(t)$ comes from the nearest smaller horizon, then a larger one, then the median over the quantities of the level; the source and the number of pairs are written in the cell row (`sigma_source`, `sigma_pairs`), so you can see what the normalization rests on. For quantities with quarterly averaging the "month" step stays calendar.

**Benchmarks.** Two are calculated on the same cells: a random walk with drift (for all quantities) and a naive forecast, the value a year earlier with a trend (for target indicators). A model must beat the benchmark, otherwise it adds nothing to a simple "tomorrow like yesterday".

**Base calculation**: the same calculation without conditions, shocks and impacts, that is, the model's unconditional forecast on the episode. A scenario is useful if the forecast under it is more accurate than the base calculation.

**Admission.** A version is admitted to comparison if fewer than 5 % of the evaluated cells break each of two conditions: the band $[p_5,p_{95}]$ lies within the admissible values of the quantity, and the band's width is no larger than the range of its history. The shares of each rule are counted separately; the union of the rules and the split by quantities are given for reference.

**Probability of a condition.** A scenario must not rest on a rare tail: for each condition the share of base calculation paths where it holds is calculated, and this share must exceed 1 %. In an episode rebuilt from the fact the condition is a point, whose probability measures the tolerance $\tau$ and not the scenario: there the share is marked as not applicable and excluded from the check.

**Coverage and sharpness.** With a correct forecast the fact lands in the bands below $p_5$, from $p_5$ to $p_{50}$, from $p_{50}$ to $p_{95}$ and above $p_{95}$ with the shares 0.05 / 0.45 / 0.45 / 0.05; a deviation says the model is biased. Sharpness is the median width of the band; it has meaning only when coverage passes, so it is reported beside it.

**Aggregate.** A total score in which each quantity of a level weighs the same, however many cells it has: first the mean over episodes within a quantity at month $t$, then the mean over the quantities of the level, then over horizons. Otherwise a quantity with a long history would outweigh the rest. The report holds a "level × horizon" table for the model, the random walk, the base calculation and the naive forecast, with two differences (from the random walk and from the base calculation) and a split by halves.

**Value of a scenario**: the share of episodes where the forecast under the episode's scenario is more accurate than the base calculation, and the mean difference. There is a split by markup (`manual` / `llm`): all episodes in [episodes.toml](episodes.toml) are marked by hand, so the LLM column is empty. A scenario from an LLM is reliable only for events after the date of its writing, because the model remembers the outcome of past episodes.

**Intervals**: three kinds, labeled in the report:

- by episodes: a 95 % bootstrap of the same statistic as in the report, 1000 repetitions, `seed` from `episodes.toml`; the verdict reads the standard error of this bootstrap;
- wild: the same episode contributions reweighted with Webb weights, with a divergence flag beside it; with fewer than eight episodes it reads as the main one, because an ordinary bootstrap understates the interval there;
- by draws (`--mc-bootstrap`): the Monte Carlo error separately from the sampling error over episodes.

**Verdict** (`--against REPORT.json`): a comparison of a candidate with the current version by two standard errors:

1. failed admission: rejected whatever its CRPS;
2. better than the benchmark and the current version by more than 2 SE with admission shares not worse: accepted;
3. worse than the current version by more than 2 SE: rejected;
4. otherwise no difference is detected, and the current version stays.

Both reports are matched by cells: an incomplete pair is an error with a list, not a comparison on the intersection; cells without a normalized score are counted and named. The verdict is calculated on the validation half and names it.

**Portfolio tail** (`--portfolio ID`, requires `--satellites`): a check of whether a model understates extreme losses. The Acerbi–Székely statistic $Z$: with a correct tail its mean is zero, and $Z>0$ means the tail is understated. The predictive $\mathrm{VaR}$ and $\mathrm{ES}$ are the P95 and ES95 of portfolio losses on the conditional paths of an episode; the realized loss is the change in portfolio value on the facts at the start and end of the window. The report holds an interval by episodes, a wild interval, $Z$ with the 1 % most extreme draws trimmed, a horizon mark and the dropped episodes by name. Admission does not read $Z$.

**Leaving the training sample.** The share of a satellite's paths outside the range of a node's training sample must be below 1 %. The bounds are written into the version at estimation (`train_low` / `train_high`); a part without them is not read. A quantity with an unverified tail is marked in the report.

**Sensitivity to the causes and the rolling check.** Uniformity of $u$ is counted on the calculation without named causes, and the sensitivity to the choice of causes is the block `cause_sensitivity`. The rolling block (uniformity at $s_0$ over all dates and the check of the price on the event family `[[price_events]]`) appears when the plan carries `[rolling]`.

## What the report prints

| Section | What it holds |
|---|---|
| Episodes | Episodes: calculation date, horizon, set variables, shocks, the worst probability of a condition |
| Admission | Two admission rules: how many cells were broken, what share this is, which episodes are to blame |
| Aggregate | A "level × horizon" table: the model, the random walk, the base calculation, the naive forecast and two differences, from the random walk and from the base calculation |
| Scenario usefulness | The value of the scenario by markup source |
| Halves | The same aggregates by the tuning and validation halves |
| Quantities | Quantities: normalized CRPS, benchmarks, coverage, sharpness, admission shares |
| Decision | The verdict by two standard errors, with `--against` |
| Tail | $Z$, intervals, trimming, dropped episodes, with `--portfolio` |
| Reproducibility | Versions of models and data, date, seed, commit |

The report is printed as Markdown tables to stdout; the full report with all quantities and cells goes to the file `--details`. Exit code 1 means some episode was not checked (absent from the manifest, short history, an inadmissible scenario) or an admission condition was broken.

## Cache

Episode versions are written to `<models>/backtest/`. The key of a macro version includes the data on the calculation date, the configuration of the model package (`spec.toml`, `signs.toml` and the code), the macro variable registry and the model SDK version, so an edited configuration does not reuse an old version. The satellite key is `satellites-<digest>`: the macro version name, the bytes of the driver catalog, their series on the calculation date, the configuration and versions of satellite plugins, the plugin SDK version and the assignments of quantities.

## What is not here

- Scenarios from an LLM: the split by `source` exists, but there are no episodes with `source = "llm"`.
- A check of the tail of the price on rare events: on 20 years of monthly history an event of probability 1 % is expected 2–3 times, the group table by $\pi$ says so itself, and the report builds no verdict on it.

Coverage and sharpness are reference information, not a goal.

## Package layout

| File | What it does |
|---|---|
| `cells.py` | A cell $(e,t,q)$: fact, draws, random walk, base calculation and naive forecast; CRPS by the exact rank estimate, the mid-PIT $\hat F(y)$, the band, $\sigma_q(t)$ with a replacement chain and source, the historical range, cell scores, the bootstrap over episodes |
| `aggregate.py` | The aggregate: the mean within a quantity → equal weight of quantities → the mean over horizons, by levels; the bootstrap of the aggregate over episodes (SE and a percentile interval) and episode contributions for the wild check; the value of a scenario; the Monte Carlo interval by a bootstrap over draws |
| `wild.py` | The cluster wild bootstrap with Webb weights: a check of each interval with few episodes |
| `decision.py` | The two standard errors rule: four steps on the aggregates of the target level |
| `es.py` | Acerbi–Székely $Z$: the conditional paths of an episode in own units, the factual paths of the window, valuation of portfolio positions through the engine, episode contributions, intervals, the sensitivity to trimming the 1 % most extreme paths |
| `baselines.py` | Benchmarks: a random walk with drift (pairs at the exact distance, so gappy series do not break the drift) and a naive annual forecast; admission rules; the probability of a condition in the base calculation |
| `episodes.py`, `episodes.toml` | Episodes: variables, shocks, window, exclusions; the half `half` and the scenario source `source` of each; the calculation date is the day before the start; the rolling grid and the event family |
| `backtest.py` | The calculation: model versions, cells, checks of admission and conditions, intervals (by episodes and wild), the aggregate with splits, value, halves, the verdict `--against`, the tail `--portfolio`, reproducibility metadata |
| `rolling.py`, `block.py` | The rolling check over the dates of history: the uniformity at $s_0$, the logarithmic score of $P(S)$ against the historical frequency, the block bootstrap |
| `cli.py`, `__main__.py` | The command `python -m mst.metrics` |

Tests are in [../../../tests/metrics](../../../tests/metrics/README.en.md).
