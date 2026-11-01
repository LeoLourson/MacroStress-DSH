---
name: run-report
description: Start a portfolio calculation and read P5/P50/P95, ES95, the scenario price and the report flags
whenToUse: when a scenario and a portfolio exist and losses, coverage and the scenario price are needed
---

# Calculation and report

## 1. Start

`run(action="start", scenario="...", portfolio="...", draws=2048, seed=20261029)` creates `runs/<id>/` and a job on the `runs` resource; it returns `run` and `job`. `draws` is the number of path draws, `seed` an integer in 0..2^32-1. The same scenario, versions, `draws` and `seed` give the same numbers.

`run(action="status", id=<run>)` gives the meta of the calculation and the job; `jobs(action="status", id=<job>)` and `jobs(action="log", id=<job>)` give progress and the log. `run(action="start")` also takes `positions` instead of a portfolio (`base_currency` RUB only).

## 2. Result

`run(action="result", id=<run>)` returns `run`, `positions` and `summary`. For a portfolio `summary` is the report:

| Field | How to read it |
|---|---|
| `loss` | Portfolio losses; `null` when no position was computed, which is not zero |
| `loss.p05`, `loss.p50`, `loss.p95` | Portfolio losses P5/P50/P95, always together, in millions of the base currency |
| `loss.es95` | ES95, the mean tail beyond P95; it does not replace P95 and is not shown without it |
| `dist` | The scenario price; `null` as a whole when there are no conditions, which is not zero |
| `dist.distance` | $\mathrm{dist}(s)$: how far the scenario departs from the base models, in nats; infinity means the scenario contradicts the models, with the reason in `dist.message` |
| `dist.log_probability` | $\log P(S)$: the log probability of the event under the base models |
| `dist.standard_error` | The error of the price estimate over the groups of $R$ |
| `dist.by_level` | The price by level when there are conditions on drivers: the key is the level number (0 is macro variables), the sum equals `dist.distance`; it shows on which level the scenario is furthest from the base models |
| `dist.causes`, `dist.tolerances` | The named causes and the tolerances of exact values; prices are comparable only at equal tolerances |
| `status` | `ok`, `needs_review` or `incomplete` |
| `coverage.share` | The share of measurable value the numbers cover; `null` means a position is unmeasurable |
| `coverage.excluded` | Positions outside the numbers: id, status and reason |
| `coverage.unknown` | Ids of positions with unmeasurable value (no reason) |
| `flags` | The report flags |

Losses add up **within a draw**, and only then are quantiles taken. The sum of the P95 of positions is not the P95 of the portfolio: the difference is diversification, so do not present one as the other.

## 3. Positions and flags

`positions` of the result are per position: `status`, `value_current`, `metrics`, `flags`, `diagnostics`. An `invalid` position has no metrics and no value change: name its `diagnostics` and do not count it as a zero loss.

Flags:

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
- Name an excluded position with its reason and the coverage share.
- `run(action="list")` finds a calculation; `run(action="result")` gives its numbers.
