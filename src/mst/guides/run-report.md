---
name: run-report
description: Start a portfolio calculation and read P5/P50/P95, ES95, the scenario price and the report flags
whenToUse: when a scenario and a portfolio exist and losses, coverage and the scenario price are needed
---

# Calculation and report

## 1. Start

`run(action="start", scenario="...", portfolio="...", draws=2048, seed=20261029)` creates `runs/<id>/` and a job on the `runs` resource; it returns `run` and `job`. `draws` is the number of path draws, `seed` an integer in 0..2^32-1. The same scenario, versions, `draws` and `seed` give the same numbers.

`run(action="status", id=<run>)` gives the meta of the calculation and the job; `jobs(action="wait", id=<job>, timeout_sec=30)` holds the call until the job ends — call it again while it returns `running: true` — and `jobs(action="log", id=<job>)` gives the log. `run(action="start")` also takes `positions` instead of a portfolio (`base_currency` RUB only).

## 2. Result

`run(action="result", id=<run>)` returns `run`, `positions` and `summary`. For a portfolio `summary` is the report:

| Field | How to read it |
|---|---|
| `loss` | Portfolio losses at the end of the horizon; `null` when no position was computed, which is not zero |
| `loss.p05`, `loss.p50`, `loss.p95` | Portfolio losses P5/P50/P95, always together, in millions of the base currency |
| `loss.es95` | ES95, the mean tail beyond P95; it does not replace P95 and is not shown without it |
| `losses_by_month` | The loss by month and over the whole path: `by_month` holds one row per month (`p05`, `p50`, `p95`, `es95`) and `path` the row of the maximum of the loss over the horizon; the last row of `by_month` equals `loss`. An impact that ended before the horizon shows in its months alone |
| `comparison` | The same losses against the base $s_0$ on the same random numbers: `centre` (the difference of medians) apart from `width` (the difference of $p_{95}-p_5$), beside `p95` and `es95`, each with its 95 % interval. `null` when the scenario has no conditions and no impacts |
| `dist` | The scenario price; `null` as a whole when there are no conditions, which is not zero |
| `dist.distance` | $\mathrm{dist}(s)$: how far the scenario departs from the base models, in nats; infinity means the scenario contradicts the models, with the reason in `dist.message` |
| `dist.estimated` | `false` when the standard error of the price exceeds a tenth of the price: the number stays visible, but it is not an estimate, and the reverse problem does not use it |
| `dist.one_sided` | The one-sided price $\mathrm{dist}^+(s)$: every interval condition opened to a ray, so the width of the interval does not enter it; `dropped` numbers the conditions whose interval held the median of the functional. Prices of events with unequal boxes are compared by `dist.one_sided.distance`, never by `dist.distance`; `null` without support conditions or with conditions on means and probabilities |
| `dist.log_probability` | $\log P(S)$: the log probability of the event under the base models |
| `dist.standard_error` | The error of the price estimate over the groups of $R$ |
| `dist.by_level` | The price by level when there are conditions on drivers: the key is the level number (0 is macro variables), the sum equals `dist.distance`; it shows on which level the scenario is furthest from the base models |
| `dist.causes`, `dist.tolerances` | The named causes and the tolerances of exact values; prices are comparable only at equal tolerances |
| `admission` | The verdict on the paths of the calculation: `shares` of cells of month and quantity that break each condition, `broken` naming every quantity with a positive share, worst first |
| `status` | `ok`, `needs_review`, `incomplete` or `not_admitted` |
| `coverage.share` | The share of measurable value the numbers cover; `null` means a position is unmeasurable |
| `coverage.excluded` | Positions outside the numbers: id, status and reason |
| `coverage.unknown` | Ids of positions with unmeasurable value (no reason) |
| `flags` | The report flags |

Losses add up **within a draw**, and only then are quantiles taken. The sum of the P95 of positions is not the P95 of the portfolio: the difference is diversification, so do not present one as the other.

## 3. Statuses and flags

`positions` of the result are per position: `status`, `value_current`, `metrics`, `flags`, `diagnostics`. An `invalid` position has no metrics and no value change: name its `diagnostics` and do not count it as a zero loss.

`not_admitted` means five percent or more of the cells of month and quantity on the paths break an admission condition — the band leaves the admissible domain or grows wider than the history: **the numbers are then diagnostics, not an estimate of losses**. Name the broken conditions and quantities from `admission.broken`; do not soften the verdict by rerunning with more draws.

Flags:

- `not_admitted`: the report status above, repeated as a flag;
- `tail_unverified`: some position's path leaves the training range of its satellite, so the tail of the whole report is unverified — losses add up within one path;
- `price_not_estimated`: the price, or its one-sided companion, carries an error above a tenth of itself;
- `constraint_rare:<number>`: the numbered condition has a base probability below the threshold, so the scenario rests on a rare tail;
- `approximate_data:<driver>`: the publication date of the driver's history is estimated, not taken from the source; the number may be less accurate on the calculation date;
- `parameters_from_above:<driver>`: the driver has no history of its own, its parameters come from the nearest object above;
- `impact:<driver>`: the path is shifted by an impact of the scenario;
- `portfolio_exclusions`: positions are left out of the report; `portfolio_incomplete`: the uncovered share exceeds `coverage.limit` (5 %) or some value is unmeasurable;
- `exposure_unknown:<position>`: the value of the position cannot be measured; `coverage.share = null`.

These are disclosures, not errors. `needs_review` means the numbers can be read with a caveat; `incomplete` means coverage is partial and the report says so.

## 4. What to say

Write in the language of the user.

- Give all three quantiles together with the unit; do not round beyond need.
- Call the price the scenario price, not a probability; the probability of the event is $e^{\text{log\_probability}}$.
- Compare the prices of events with unequal boxes by `dist.one_sided.distance`, and say which conditions were dropped.
- Lower ES95 under a condition does not by itself mean lower risk: a condition fixes part of the path and can narrow the law instead of moving it. Read `comparison.centre` and `comparison.width` with their intervals before saying the event lowers risk.
- A loss that vanishes before the end of the horizon is still a loss: read `losses_by_month`, and the maximum over the path in `losses_by_month.path`.
- Name an excluded position with its reason and the coverage share.
- `run(action="list")` finds a calculation; `run(action="result")` gives its numbers.
