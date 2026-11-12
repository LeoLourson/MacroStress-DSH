---
name: run-report
description: Start a portfolio calculation and read P5/P50/P95, ES95, the scenario price and the report flags
whenToUse: when a scenario and a portfolio exist and losses, coverage and the scenario price are needed
---

# Calculation and report

## 1. Start

`run(action="start", scenario="...", portfolio="...", draws=2048, seed=20261029)` creates `runs/<id>/` and a job on the `runs` resource; it returns `run` and `job`. The request also carries the calculation settings the scenario file does not name:

- `horizon` — months, twelve by default and never shorter than the last month a statement names;
- `inner` — how many paths share one draw of the parameters and one realization of the interventions, one by default; together with `draws` it gives `draws * inner` paths;
- `macro` and `satellites` — the version hashes; the newest finished pair of the workspace is taken when they are omitted.

`draws` is the number of parameter draws and `seed` an integer in 0..2^32-1. The same scenario, versions, `draws`, `inner`, `horizon` and `seed` give the same numbers.

`run(action="status", id=<run>)` gives the meta of the calculation and the job; `jobs(action="wait", id=<job>, timeout_sec=30)` holds the call until the job ends, so call it again while it returns `running: true`, and `jobs(action="log", id=<job>)` gives the log. `run(action="start")` also takes `positions` instead of a portfolio (`base_currency` RUB only).

`portfolio(action="check", id=..., scenario=...)` names every problem that keeps the portfolio from running under the scenario. Its `warnings` carry what an intervention on a macro variable reaches: one per position whose driver reads that variable, with the driver's long-run loading (below 0.1 in absolute value the driver "barely reacts") and the regressors it loads on most. An intervention aimed at a position can still miss it through a small loading; read the warnings before you call the run a measurement of that event.

## 2. Result

`run(action="result", id=<run>)` returns `run`, `positions`, `forecast` and `summary`. For a portfolio `summary` is the report:

| Field | How to read it |
|---|---|
| `loss` | Portfolio losses at the end of the horizon; `null` when no position was computed, which is not zero |
| `loss.p05`, `loss.p50`, `loss.p95` | Portfolio losses P5/P50/P95, always together, in millions of the base currency |
| `loss.es95` | ES95, the mean tail beyond P95; it does not replace P95 and is not shown without it |
| `losses_by_month` | The loss by month and over the whole path: `by_month` holds one row per month (`p05`, `p50`, `p95`, `es95`) and `path` the row of the maximum of the loss over the horizon; the last row of `by_month` equals `loss`. An intervention that ended before the horizon shows in its months alone |
| `comparison` | The same losses against the base $s_0$ on the same random numbers: `centre` (the difference of medians) apart from `width` (the difference of $p_{95}-p_5$), beside `p95` and `es95`, each with its 95 % interval. `null` when the scenario has no observations and no interventions |
| `dist` | The scenario price; `null` when the scenario has neither observations nor interventions. A scenario of interventions alone carries a price of zero rather than a missing one |
| `dist.distance` | $\mathrm{dist}(s)$: how far the observations of the scenario depart from the base models, in nats; infinity means the scenario contradicts the models, with the reason in `dist.message`; exactly zero when the scenario has interventions and no observations, because the event is imposed and not observed |
| `dist.estimated` | `false` when the standard error of the price exceeds a tenth of the price: the number stays visible, but it is not an estimate, and the reverse problem does not use it |
| `dist.one_sided` | The one-sided price $\mathrm{dist}^+(s)$: every interval observation opened to a ray, so the width of the interval does not enter it; `dropped` numbers the observations whose interval held the median of the functional. Prices of events with unequal boxes are compared by `dist.one_sided.distance`, never by `dist.distance`; `null` without support observations or with observations on means and probabilities |
| `dist.log_probability` | $\log P(S)$: the log probability of the observations under the base models |
| `dist.standard_error` | The error of the price estimate over the groups of $R$ |
| `dist.by_level` | The price by level when there are observations on drivers: the key is the level number (0 is macro variables), the sum equals `dist.distance`; it shows on which level the scenario is furthest from the base models |
| `dist.tolerances` | The tolerances of exact values; prices are comparable only at equal tolerances |
| `interventions` | One row for every intervention and month: `quantity`, `month`, `kind`, `active_share`, `alpha` (`alpha_distribution` when the imposed value differs across groups); `[]` without interventions |
| `admission` | The verdict on the paths of the calculation: `shares` of cells of month and quantity that break each condition, `broken` naming every quantity with a positive share, worst first |
| `status` | `ok`, `needs_review`, `incomplete` or `not_admitted` |
| `coverage.share` | The share of measurable value the numbers cover; `null` means a position is unmeasurable |
| `coverage.excluded` | Positions outside the numbers: id, status and reason |
| `coverage.unknown` | Ids of positions with unmeasurable value (no reason) |
| `flags` | The report flags |

Losses add up **within a draw**, and only then are quantiles taken. The sum of the P95 of positions is not the P95 of the portfolio: the difference is diversification, so do not present one as the other.

The `run` field of the result is the meta of the calculation: the versions, the horizon, `inner`, the format of the scenario (`scenario_format`) and `structural_forms` — what the macro model and every satellite model declared about the links inside their levels. Its `basis` lists what every observation and intervention rests on. A calculation written before the intervention format carries no `scenario_format`: its meta holds the copy of the scenario as it was, and that absence is the mark of the old format.

### The interventions block

Each row says which quantity, in which calendar month, was set or shifted, in how many paths the intervention was active, and how far the imposed value stands from the base.

- `kind`: `set` replaces the value the model drew; `shift` displaces it by the stated amount.
- `active_share`: the share of groups where the intervention was active. A deterministic intervention is active in every group and shows 1; a `with_probability` intervention shows the share of groups that had it, which is the law drawn from the seed.
- `alpha`: the remoteness of the imposed value — the share of the empty scenario's paths whose value is at or below it, that is, the quantile of the imposed value in the base law. A `set` at the median shows 0.5; at 0.05 or 0.95 the value stood in a tail of the model's own forecast; near 0 or 1 the model almost never goes there by itself. For a `shift`, the displaced value of each base path is compared with the base paths, so a shift upwards gives an `alpha` near 1: the displaced path lies above the whole base distribution.
- `alpha_distribution`: `p05`, `p50` and `p95` of the remoteness, instead of a single `alpha`, when the imposed value differs across groups.

`alpha` is a diagnostic. It enters neither the price nor the admission of the calculation. It answers "how strong is the event for this model", which is the question the price cannot answer for an imposed value.

### The price of an intervention-only scenario

A scenario with interventions and no observations chooses no measure, so there is nothing to price: `dist` is present with `distance` 0, `log_probability` 0, `standard_error` 0 and `estimated` true. Say "the price is zero" and give the reason: the event is imposed, not observed. Do not treat the zero as a failure, and do not compare it with the price of an event that is observed.

A scenario with neither interventions nor observations is the base $s_0$: `dist` is `null`, and that is not zero either.

How strong the imposed event is, the report says with the `interventions` block and `comparison`, not with the price.

### Structural forms

The meta of the result carries `structural_forms.macro` and `structural_forms.satellites.<model>`, which say what a model declared about the links inside its level:

- `null`: the default form. A quantity's value does not read the other quantities of its own level in the same month, so a set value replaces the drawn value and the other variables of that month do not react; later months read it through the model's lags.
- `{"order_within_month": [...], "time_recursion": true}`: a declared form. The set value is executed inside the month in that order, and later months read it through the recursion.

The form is a model's assumption about which quantities act on which, not a fact of the data. One model under two forms has the same law of paths without a scenario and different responses to the same intervention. Name the form with the numbers whenever it is not the default, and do not compare two runs as if only the model differed when their forms differ too.

### Two models on one scenario

The same scenario file runs unchanged on a second model: start two runs that differ only in `macro`, and in `satellites` estimated on that macro. Use the same `draws`, `inner`, `horizon` and `seed`, so the models are the only difference. Compare:

- `loss` and `losses_by_month` (P5/P50/P95 and ES95) and the forecast P5/P50/P95 of every target by month;
- `dist` — the models price the same observations differently, and a price that is estimated in one model may not be in the other;
- `interventions` — `alpha` says where the imposed values stood in each model's own base law, which explains why one model answers the same set with a wider fan than the other.

Each model draws its own paths, so a draw of one run is not the same draw of the other: compare quantiles and intervals, never draw-by-draw differences.

A conclusion that holds under both models depends on the event; a conclusion that appears under one model only is a warning about that model.

## 3. Statuses and flags

`positions` of the result are per position: `status`, `value_current`, `metrics`, `flags`, `diagnostics`. An `invalid` position has no metrics and no value change: name its `diagnostics` and do not count it as a zero loss.

`not_admitted` means five percent or more of the cells of month and quantity on the paths break an admission condition — the band leaves the admissible domain or grows wider than the history: **the numbers are then diagnostics, not an estimate of losses**. Name the broken conditions and quantities from `admission.broken`; do not soften the verdict by rerunning with more draws.

Flags:

- `not_admitted`: the report status above, repeated as a flag;
- `tail_unverified`: some position's path leaves the training range of its satellite, so the tail of the whole report is unverified — losses add up within one path;
- `price_not_estimated`: the price, or its one-sided companion, carries an error above a tenth of itself;
- `constraint_rare:<number>`: the numbered observation has a base probability below the threshold, so the scenario rests on a rare tail;
- `driving_shock_above_limit`: five percent or more of the paths need a driving shock above six standard deviations (`admission.driving_above_limit`): the event lies far beyond the model's history and the responses to it are extrapolated; read the price for how unlikely it is;
- `approximate_data:<driver>`: the publication date of the driver's history is estimated, not taken from the source; the number may be less accurate on the calculation date;
- `parameters_from_above:<driver>`: the driver has no history of its own, its parameters come from the nearest object above;
- `impact:<driver>`: the driver's path is displaced by an intervention of the scenario;
- `portfolio_exclusions`: positions are left out of the report; `portfolio_incomplete`: the uncovered share exceeds `coverage.limit` (5 %) or some value is unmeasurable;
- `exposure_unknown:<position>`: the value of the position cannot be measured; `coverage.share = null`.

These are disclosures, not errors. `needs_review` means the numbers can be read with a caveat; `incomplete` means coverage is partial and the report says so.

## 4. What to say

Write in the language of the user.

- Give all three quantiles together with the unit; do not round beyond need.
- Call the price the scenario price, not a probability; the probability of the observations is $e^{\text{log\_probability}}$.
- Compare the prices of events with unequal boxes by `dist.one_sided.distance`, and say which observations were dropped.
- Give the `interventions` block with the numbers: which quantity, which month, set or shift, and `alpha`. A value in a tail of the base law (an `alpha` near 0 or 1) is worth a sentence: the model almost never goes there by itself.
- For a scenario of interventions alone, say that the price is zero because the event is imposed, and give `alpha` instead of the price.
- A value set by an intervention is not a forecast of that quantity: the forecast of it equals the set value. Read the forecast of the other quantities, which the models derive.
- When `structural_forms` is not `null`, name the form that produced the numbers, and do not compare it with a run under another form as if the model were the only difference.
- Comparing two models, compare quantiles and intervals, never single draws: the draws are not paired across models.
- Lower ES95 under an observation does not by itself mean lower risk: an observation fixes part of the path and can narrow the law instead of moving it. Read `comparison.centre` and `comparison.width` with their intervals before saying the event lowers risk.
- A loss that vanishes before the end of the horizon is still a loss: read `losses_by_month`, and the maximum over the path in `losses_by_month.path`.
- Name an excluded position with its reason and the coverage share.
- `run(action="list")` finds a calculation; `run(action="result")` gives its numbers.
