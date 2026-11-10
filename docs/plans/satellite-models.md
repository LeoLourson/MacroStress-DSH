# Plan: repair the satellite model `hier.gibbs`

Audience: the engineer (a coding agent) who implements this. If something here does not match the code or the data, stop and report instead of guessing. Versions are chosen by measured gates, so a step can end in "this did not help": report that, and do not tune until a number passes.

## What is wrong

Measured on the bundle `a08a…` (on macro version `5c57…`) and on 2048 draws of the base scenario.

**1. Median returns come from a spurious long-run level.** Share prices follow, in 100 ln scale, $d_t=c+\rho d_{t-1}+\beta_0^\top x_t+\beta_1^\top x_{t-1}+e_t$, with $x$ in levels: `ru.imoex`, `ru.usdrub`, `ru.key_rate`, `gl.brent`. All these are trending or nominal, so the level the driver is pulled to is $d^\ast=(c+(\beta_0+\beta_1)^\top x)/(1-\rho)$, which sits far from today's price. The 12-month pull is $(d^\ast-d_0)(1-\rho^{12})$, and it tracks the median return of the base:

| Share | $\rho$ | Long-run elasticity to USD/RUB | to key rate | $d^\ast-d_0$ | 12-month pull, log points | Base median return |
|---|---|---|---|---|---|---|
| LKOH | 0.941 | 0.92 | −1.38 | 7 | +4 | +24 % |
| ROSN | 0.923 | 0.81 | −2.02 | 49 | +30 | +31 % |
| SIBN | 0.942 | 1.97 | −4.14 | 57 | +29 | +36 % |
| SNGS | 0.952 | −0.74 | 0.74 | 25 | +11 | +12 % |
| GAZP | 0.958 | −1.02 | −0.41 | 3 | +1 | +2 % |
| VTBR | 0.970 | −2.10 | 0.48 | −65 | −20 | −10 % |
| BSPB | 0.971 | 4.04 | 0.96 | 113 | +33 | +58 % |
| PLZL | 0.950 | 0.75 | 8.88 | 16 | +7 | +14 % |
| OZON | 0.938 | 0.06 | 4.40 | 38 | +20 | +18 % |

The pull is computed at today's $x$, so the rouble's path adds to it. The portfolio's median gain of 22 % in the base, against about 7 % for IMOEX in the same paths, comes mainly from here. A long-run elasticity of 4 to the dollar for a bank, or of +8.9 to the key rate for a gold miner, is an artefact of levels on levels.

**2. Coverage.** 138 of 263 share nodes are not served: 101 are "not more accurate than without macro", SBER among them (RMSE 15.9 against 10.5), 34 have too few observations and 3 no holdout forecast. Among served shares the median ratio of RMSE to the autoregression is 0.72.

**3. No credit channel for shares.** `share_price` does not read spreads, so a bank's share cannot react to a spread shock at all.

**4. Leaving the training range is routine.** In the base, the median share of paths that leave a node's training range at some month is 29 % for shares, 56 % for bonds, above 90 % for housing. All quantities exceed the canon's 1 % on every driver measured. A persistent price that starts near the edge of its past range leaves that range often without anything being wrong, so this is partly the rule and partly the model. Do not change the rule (step 6).

**Not defects:**

- The bond `RU000A0JXPG2` barely reacts to a BBB shock because it belongs to `group_aa`. Its long-run loadings are 0.52 to `ru.spread_aa`, 0.33 to `ru.spread_a` and 0.04 to `ru.spread_bbb`, and 0.04 × 1.25 is the observed 0.05. This is the scenario's choice of target. Step 5 makes it visible.
- PLZL has no gold regressor because the macro model has no gold variable. Out of scope.

## Scope

In scope: `plugins/hier`, `src/mst/drivers/registry.toml` (`share_price` regressors only), the `portfolio check` warning, tests, READMEs, new bundles in `workspace/models/satellites`.

Out of scope: the canon, the macro models, the position plugins, new variables, a new bank model, the training-range and admission rules.

Prerequisite: `plans/canon-alignment.md`, including its step on the scale error in `_checks`. Do not trust a `band_wider_than_history` of a driver in any run made before that fix: those numbers are wrong (see the table in step 1). Steps 1–5 can run on the present macro version. Step 7 needs `plans/macro-models.md` to have finished.

Branch `satellite-models` from `main`.

## Working rules

- One commit per step, in order. Push the branch; merge nothing. Commit messages in Russian, as in `git log`. In Russian text never use the word «прогон» or its forms.
- Write the test first, on the simulated data of `tests/hier` and `tests/drivers/simulated.py`, with known answers.
- After each step: `uv run ruff check src tests`, `uv run pytest -q -m "not slow"`, `uv run pytest -q tests/hier tests/drivers`.
- A bundle's digest changes with its inputs, so old bundles stay. Never overwrite one. Never report a number you did not produce.
- Do not weaken a gate. Thresholds are fixed in advance.

**Stop and report** when: a step's stop condition fires; a bundle takes over 2 hours to estimate; a change would need the canon, the macro plugins or the rules above.

## Gates

| Gate | Check | Threshold |
|---|---|---|
| S1 | Base: median 12-month log return of served shares, minus the median of `ru.imoex` over the same months | within ±10 log points (now +21.8) |
| S2 | Base: each of the 9 portfolio shares, median 12-month log return minus that of `ru.imoex` | within ±35 log points (now BSPB +38.9) |
| S3 | Base, driver cells, after the fix of `_checks`: band wider than history | below 5 % |
| S4 | Served share nodes against the reference bundle | not fewer than 115 of the present 125 |
| S5 | Backtest with `--satellites`, driver level, checking half of the episodes, on the canon's rule | CRPS not worse than the reference by more than two standard errors; coverage consistent with 0.05, 0.45, 0.45, 0.05 |
| S6 | Portfolio shares, largest absolute long-run elasticity to USD/RUB | below 2.5 (now 4.04, BSPB) |

S5 is the guard: a satellite that only stopped moving fails it.

## Step 1. Measure

Add `python -m mst.models diagnose-satellites <bundle> --models DIR` that prints as JSON, per quantity and for named objects: $\rho$; long-run elasticities $(\beta_0+\beta_1)/(1-\rho)$ with their posterior 5–95 %; $d^\ast-d_0$ and the 12-month pull; counts of the check's reasons; the RMSE ratio's quantiles.

Add `scripts/satellite_gates.py`, which reads run ids and prints S1–S3, S6, the share of paths leaving the training range, and a decomposition of the 12-month band of each named driver: residuals only, with $x$ fixed at its median path; and the full band.

Reproduce the table above on `a08a…`, and the corrected driver shares below, recomputed from the paths:

| Run, base | Recorded driver width share | True driver width share |
|---|---|---|
| `bvar.sv` | 21.9 % | 7.5 % |
| `msvar.markov` | 24.4 % | 13.3 % |

If they differ, stop and report. Save the output to `plans/satellite-models-results.md`, under the heading "Before".

## Step 2. Shares in differences

Fit `share_price` in monthly changes: $\Delta d_t=c+\rho\,\Delta d_{t-1}+\beta_0^\top\Delta x_t+\beta_1^\top\Delta x_{t-1}+e_t$. The long-run level disappears, the coefficients are elasticities, and $c$ is a monthly drift.

The existing machinery fits this with no new sampler: feed it the differenced series, and accumulate on simulation.

- `plugins/hier/spec.toml`: add `differences = ["share_price"]`. Housing has a quarterly `span`, and bonds and overdue shares are stationary: they stay in levels.
- `estimate.py`: for a listed quantity, difference $d$ and $x$ before the sampler. A month missing in $d$ gives missing $\Delta d$ for two months, which `Path` already draws; if `Path` or the stationary start cannot take this, **stop and report**. Record `train_low`, `train_high` of the levels as now, so that the flag keeps its meaning.
- `simulate.py`: simulate changes, then accumulate from the last known level. The start logic for a driver with an older last value and for a driver without history must be kept: test both. Impacts of the scenario apply to the level, after accumulation.
- `check.py`: compare in levels, by the same reach, against the same autoregression in levels. The verdict must stay comparable.
- The drift $c$: the `constant_spread` prior is wide. Candidates `constant_spread` in `{10, 1, 0.1}`. Keep whichever wins by driver-level CRPS on the **tuning half** of the episodes. Do not add others.

Tests:

- simulated data with known elasticities are recovered;
- a simulated level path equals the cumulative sum of its changes plus the start;
- an impact lands on the level in exactly its months and does not accumulate;
- without the option, results are unchanged.

Estimate on `5c57…`. Run `diagnose-satellites`, then S1–S6, with `fifth-base` on the new bundle.

**Decision.** Keep the change if S5 holds. If S5 fails, stop and report: the owner decides whether levels with an error-correction term are worth a further step.

## Step 3. Served shares

Report how the check's verdicts changed against the reference: the count per reason, gained and lost by name. Report SBER by its RMSE ratio. Do not force any share to be served.

## Step 4. Credit channel for shares

Add `ru.spread_bbb` to the `share_price` regressors in `registry.toml`. It is one regressor in the hierarchy, so a sector can load on it and other sectors stay near zero.

Estimate as a candidate. Keep it only if all of these hold:

- S5 is not worse than step 2's bundle;
- the `financials` node's elasticity is negative, with its 5–95 % interval excluding zero;
- `vtbr` and `bspb` inherit a negative elasticity.

Otherwise revert, and record the numbers: the data would then not support the channel. Do not build a bank-specific model. Spreads of different ratings correlate at about 0.9, so do not add the other two.

Test: a scenario with a shift of `ru.spread_bbb` moves `share_price:vtbr` when the regressor is on, and leaves it unchanged when off.

## Step 5. Show what a macro shift reaches

In `portfolio check`, given a scenario with an impact on a macro variable, add a warning per position whose driver reads that variable: the long-run loading, and the regressors it loads on most. A loading below 0.1 in absolute value is named as "barely reacts". This is the check that would have exposed the BBB shock on an AA bond.

The output must also appear in `run-report` and `tools` documents, in both languages where both exist.

Tests: a driver with a known small loading is named, and a driver with a large one is not.

Do not edit `bank-admin-run-top5`. Report to the owner that the shock should also move the higher ratings, for them to decide.

## Step 6. Training range: measure, do not change

Report, on the final bundle: the share of paths leaving the range per quantity, and the same share if the range is widened by a historical 12-month move on each side. **The canon's rule stays.** The owner decides whether a persistent driver near the edge of its range should count as outside. Until then the mark on the report remains, and the results say so.

## Step 7. Final bundle on the repaired macro models

After `plans/macro-models.md`: estimate the bundle for each final macro version with the options kept in steps 2 and 4. Update scenario copies and run the base and 11 scenarios on each, `draws=2048`, `seed=20261029`. Leave `approved_by` of new copies empty until the owner confirms.

Write in `plans/satellite-models-results.md`: S1–S6 per bundle and the macro version under it; the macro's share and the satellite's share of each named driver's band; how many of the 12 calculations are admitted.

Update `plugins/hier/README.md` and its English twin, `docs/reference/platform` if the equation is stated there, and `CHANGELOG` in both languages. `tests/test_docs_translation.py` must pass.

## Acceptance

The reviewer reruns these from the saved artefacts.

1. S1–S6 hold for at least one bundle, or the results state which failed, on which driver, with the measured value. A failed gate is an acceptable result if it is stated; a loosened gate is not.
2. S5 holds for any bundle recommended for use.
3. The results reproduce "Before" from saved runs.
4. `a08a…`, `2aec…` and their runs are untouched.
5. `uv run ruff check src tests` and `uv run pytest -q` pass, slow tests included, and CI of the branch is green.
