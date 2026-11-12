# Plan: repair the macro models `bvar.sv` and `msvar.markov`

Audience: the engineer (a coding agent) who implements this. If something here does not match the code or the data, stop and report instead of guessing. This plan chooses between model versions by measured gates, so a step can legitimately end in "this did not help": report that, and do not tune until a number passes.

## What is wrong

Measured on the versions `5c57…` (`bvar.sv`) and `ecaa…` (`msvar.markov`), as of 2026-10-29, with 2048 draws under each of 12 scenarios on the portfolio `fifth-test`.

$\rho$ below is the spectral radius of the companion matrix of a parameter draw.

| Measure | `bvar.sv` | `msvar.markov` |
|---|---|---|
| Median $\rho$ | 1.013 | calm 1.024, stress 1.328 |
| Share of draws with $\rho>1$ | 83 % | calm 93 %, stress 100 % |
| P95 of $\rho$, and its 12-month growth | 1.053, ×1.85 | calm 1.097, ×3.0; stress 1.550, ×192 |
| Parameter draws surviving identification | 967 of 1000 | 367 of 1000 |
| Base: macro band months wider than history | 2.5 % | 19.6 % |
| Base: macro band months outside the domain | 3.3 % | 6.9 % |
| Base: paths with USD/RUB above 300 at some month | 0.3 % | 7.2 % |
| Base: paths with a negative key rate | 0.1 % | 4.2 % |
| Rate cut 14→9: paths with USD/RUB above 300 | 27 % | 28 % |
| Rate cut 14→9: largest portfolio gain, mln RUB on 10 000 | $1.4\cdot10^{47}$ | $2.8\cdot10^{54}$ |
| Fed at zero: months with the Fed rate below zero, share of paths | 99 % | 98 % |
| Fed at zero: Brent in month 4, USD | 100 | 26 |

Five causes follow, ranked by evidence. Causes 1 and 2 are measured directly; 3, 4 and 5 are hypotheses that the steps below test.

1. **Explosive draws.** No code restricts $\rho$ in either macro model, although the hierarchical satellite truncates its own persistence (`_stationary` in `plugins/hier/estimate.py`). A path simulated from a draw with $\rho>1$ grows without bound, and conditions on a path multiply the effect.
2. **The stress regime is underdetermined.** Every lag coefficient switches by regime. That is about 70 coefficients per equation, on a regime that holds a minority of 174 months, so its $\rho$ is above 1.18 in every draw. The last month of history is in stress in every draw, with stress staying at 0.70 a month, so every forecast starts from that regime. The regime is also poorly labelled: its share is 0.83 in 2026, 0.42 in 2025 and 0.33 in 2023, against 0.43 in 2022 and 0.48 in 2014.
3. **The prior carries little weight.** An audit on 2026-10-30 found the Minnesota prior weighs far less than its nominal value because regressors are in levels. Re-measure this on the current versions.
4. **Tails.** The degrees-of-freedom grid in `plugins/bvar/sv.py` starts at 3, and the median draw sits at that floor for `gl.fed_rate`, `ru.key_rate`, `ru.spread_a`, `ru.spread_bbb`, `ru.retail`.
5. **The monetary shock does not act like one.** After a cut, inflation falls below base for months in both models. `signs.toml` restricts rates, USD/RUB and IMOEX for `ru.monetary`, and leaves prices free.

The levels are also unbounded below: rates and spreads are linear in the shocks.

Two things look like model defects and are not:

- Under conditions, an adverse event can lower ES95. Conditioning narrows the law; the canon handles this.
- `ru.retail` in month 0 differs between the models (460.7 and 464.8). The panel has no value for that month, and each model imputes it. Do not change this.

## Scope

In scope: `plugins/bvar`, `plugins/msvar`, a stability helper in `model_sdk`, the variable registry's bounds, `signs.toml`, `spec.toml` of both models, tests, READMEs, new versions in `workspace/models`.

Out of scope: the canon, position plugins, the satellite model code, scenario contracts, adding variables (gold, dollar index). Satellites have their own plan, `plans/satellite-models.md`. Driver cells count in admission: with the scale error of `_checks` fixed (`plans/canon-alignment.md`, step 0), driver bands in the base are wider than history in 7.5 % of cells on `bvar.sv` and 13.3 % on `msvar.markov`, and these shares should fall with the macro fix. Report what remains.

Order: start after `plans/canon-alignment.md` is merged, because that plan changes the scenario file format and writes admission into reports. Branch `macro-models` from `main`.

## Working rules

- One commit per step, in order. Push the branch; merge nothing. Commit messages in Russian, as in `git log`. In Russian text never use the word «прогон» or its forms.
- Write the test first and see it fail for the expected reason. Tests use simulated data with known answers, as `tests/bvar/simulated.py` and `tests/msvar/simulated.py` do.
- After each step: `uv run ruff check src tests`, `uv run pytest -q -m "not slow"`, and `uv run pytest -q tests/bvar` or `tests/msvar` for the model touched, slow tests included. Run the long commands in the background.
- Do not overwrite a version in `workspace/models`. The hash in its name changes with the specification, so the old ones stay for comparison. Do not touch the backtest cache by hand.
- Change one thing per candidate version. Never report a number you did not produce. Failures are reported with their output.
- Do not weaken a gate. All thresholds below are fixed in advance.

**Stop and report** when: a step's stop condition fires; fewer than 200 parameter draws survive all filters; an estimation takes over 2 hours; a full backtest takes over 4 hours; a change would need the canon or the scenario contract.

## Gates

`G` is run by a script added in step 1. A candidate version passes a gate only on the numbers it actually produced.

| Gate | Check | Threshold |
|---|---|---|
| G1 | Share of kept draws with $\rho>\rho_{\max}$, every regime | 0 |
| G2 | Base, macro cells: band outside the domain | below 5 % |
| G3 | Base, macro cells: band wider than history | below 5 % |
| G4 | Each of the 12 scenarios, macro cells: both shares | below 5 % |
| G5 | Each of the 12 scenarios: largest portfolio gain | below 10 times the notional |
| G6 | The rolling check on the checking half of the date grid, unconditional forecast | Relative CRPS not worse than the current version by more than two standard errors, and the coverage shares consistent with 0.05, 0.45, 0.45, 0.05 |

G6 is the guard: a model that is tame only because it no longer predicts anything fails it. Sharpness is compared only when coverage agrees, as the canon says.

## Step 1. Measure

Add `src/mst/model_sdk/stability.py`:

```python
def companion_radius(lags: np.ndarray) -> np.ndarray:
    """The spectral radius of the companion matrix; ``lags`` is (..., p, n, n)."""
```

Verify it on matrices with known roots, in `tests/model_sdk`.

Add the command `python -m mst.models diagnose <version> --models DIR`, which prints as JSON: $\rho$ quantiles and the shares above 1, 1.005, 1.01, 1.02 per regime; surviving draws; the regime's share per year and the stress share of the last month; degrees of freedom by variable where they exist; prior weight per equation, computed as in the audit, $P_0/(X^\top X+P_0)$ on the diagonal of the lag block.

Add `scripts/macro_gates.py`, which reads a list of run ids and prints G2–G5 from the macro cells of `meta.json["checks"]` and `changes.npz`, with the absurd-value shares: USD/RUB above 300, key rate below 0, Fed rate below 0, BBB spread below 0.

Run both on `5c57…` and `ecaa…` and on the saved runs. The output must reproduce the table above; if it does not, stop and report the difference before going on. Save it as `plans/macro-models-results.md` under the heading "Before". This file gets a section per step.

## Step 2. Stability filter in both models

Add `rho_max` to `spec.toml` of both, value `1.005`. A unit root is admissible and expected, because most of these variables are nonstationary; growth without bound is not.

After sampling, keep only the draws where `companion_radius` is at or below `rho_max`, in every regime. This is the posterior restricted to the stable region, and it is the same mechanism as the identification filter. Apply it before rotation search: `plugins/bvar/identify.py` and `plugins/msvar/identify.py` already filter, so add it beside them, and record `stable` in `diagnostics` beside `estimated` and `identified`.

Expected survival in `bvar.sv`: about 30 % of draws (17 % at or below 1, 43 % at or below 1.01), so raise `sampler.draws` to 4000 and keep `thin`. If too few survive, report; do not raise `rho_max`.

Tests:

- a simulated panel with an explosive draw inserted loses that draw and only that draw;
- a kept draw never has a radius above the limit;
- zero surviving draws raises an error naming the model and the limit.

Estimate a new `bvar.sv` version. `msvar.markov` will have no surviving draws until step 3. This is expected, so do not estimate it here beyond confirming the error. Run G1–G5 on the new `bvar.sv` with satellites estimated for it (`python -m mst.models estimate-satellites --macro <hash>`).

**Decision.** Keep the filter unconditionally. Record the numbers, and do not stop if G2–G5 still fail.

## Step 3. Regimes switch the mean and covariance, not the lags

The canon's equation allows $\mu(\zeta_t)$ and $B(\zeta_t)$ to switch; lag matrices equal across regimes are a special case of it. With 23 variables and 3 lags, the data cannot support a second set of lag matrices.

In `plugins/msvar`: add `switching = "mean_covariance"` to `spec.toml`, with the alternative value `"all"` for the present behaviour.

- `estimate.py`: in the `for j in range(m)` loop of the coefficient step, sample one lag block from all months, with each month's errors weighted by the covariance of its regime, and sample the intercept per regime given it. Keep the FFBS, the transition and the imputation steps as they are.
- `forecast.py`, `identify.py`: they read `lags` as `(S, m, p, n, n)`; keep this shape with equal copies, so they do not change.
- Regime labels: order by the trace of the covariance, not by the initial sort, if the code does not already. Confirm by test.

Tests in `tests/msvar`: simulated data with common lags and two covariances recovers the lags and separates the regimes; the shape contract holds; with `"all"` existing results are unchanged.

Estimate a new version. Run `diagnose`. Then G1–G5.

**Stop and report** if the stress share of the last month stays above 0.8 with the stress months still spread evenly over calm years. The regime would then not be a stress regime, and adding a third regime or a persistence prior is a decision for the owner.

## Step 4. The prior in both models

Add to `spec.toml` of both, and to `minnesota` and the coefficient steps of `estimate.py`, the sum-of-coefficients and initial-observation dummies (Sims and Zha, 1998; Giannone, Lenza and Primiceri, 2015). The parameters are `mu` and `delta`. They act on levels, which is where the audit found the Minnesota prior loses its weight, and they pull the system toward unit roots and away from explosive roots.

Test simulated data with a known posterior: the dummy-augmented regression equals the closed form.

Candidates: `tightness` in `{0.05, 0.1, 0.2}`, and `mu = delta` in `{1, 5}`, with the present prior as the reference. That gives 6 candidates per model. Select on the **tuning half of the date grid** only, by the relative CRPS of the unconditional forecast against the random walk. Then evaluate the winner on the checking half for G6.

Use the rolling command (`python -m mst.metrics rolling <model> --as-of 2026-10-29 ... --against <report>`), and report the conditional forecast of the episodes beside it (`python -m mst.metrics backtest`). Do not add a tuning method beyond this grid.

Record in the results: for each candidate the CRPS, coverage, survival share, median $\rho$, G2, G3.

## Step 5. Tails in `bvar.sv`

Start `DOF_GRID` in `plugins/bvar/sv.py` at the value in the spec, `dof_min`, which the spec sets to `5`, and keep `3` as a candidate for comparison. Variance is infinite at 2 and kurtosis at 4, and for rates that sit still and then step, the floor absorbs what the volatility process should carry.

Tests: `draw_dof` never returns below the floor, and the sampler recovers known degrees of freedom on simulated data.

Keep the change only if G6 holds against step 4's winner. Otherwise revert and record it.

## Step 6. Bounds on rates and spreads

Rates and spreads cannot be negative in the registry's domain, and the linear model does not know it. Treat the model's variable as a shadow value and the output as the observed one: $x_t=\max(s_t,\text{bound})$. Lags feed on $s_t$, the shadow, so the dynamics are unchanged.

- `src/mst/macro/registry.py`: add an optional `lower` to a variable. Set `0` for `gl.fed_rate`, `gl.ust_10y`, `ru.key_rate`, `ru.ruonia`, `ru.ofz_1y`, `ru.ofz_5y`, `ru.ofz_10y`, `ru.mortgage_rate`, `ru.unemployment`, and for `ru.spread_aa`, `ru.spread_a`, `ru.spread_bbb` after checking that the history never goes below. If a spread's history is below zero, report it and leave it unbounded.
- A shared helper in `model_sdk`, called by both `simulate` functions on the final paths, in the model scale.
- Conditions and the price stay on the shadow, which is the gaussian law the machinery knows. An exact condition at the bound, such as "Fed at 0", means shadow within $\tau$ of 0. Write this in the plugin READMEs. If this makes any test of the price fail, stop and report.

Estimation still treats history as observed; this is an approximation, stated in the READMEs. A proper shadow-rate estimation is out of scope.

Tests: no output below the bound; the shadow continues below the bound, and a later rise starts from it; the condition holds on the shadow; without `lower`, paths are identical to before.

G2 and G4 should improve on the domain share. Record the share of paths touching each bound: a large share in the base means the model is pressing against the bound, and it is a finding.

## Step 7. Monetary shock

In `signs.toml` of both models add `ru.cpi` to `falls` of `ru.monetary`: a tightening lowers prices within the restriction window. The window is `sign_lag = 2` months; if prices do not respond that early in the data, identification will show it.

Measure on the candidate: the survival share of identification; the path of `ru.cpi` under `cbr-cut-5pp-unscheduled-2026-11` in months 2–7, which should now rise relative to base; the size of the driving shock the condition needs, in standard deviations, per month.

Add that size to `diagnostics` of the simulation. A cut of 5 points through one shock needs a large shock, and this is the reason the scenario moves so much: the number must be visible.

**Stop and report** if identification survival falls below 20 %. Do not add further restrictions in that case: which restrictions are correct is the owner's decision.

Recorded for the owner: the response of the exchange rate and IMOEX to the rate cut, against the episodes 2014–2015 and 2022.

## Step 8. Final versions and all scenarios

Estimate the final `bvar.sv` and `msvar.markov`, then satellites on each. Do not repoint existing scenario files. Create copies with the suffixes `-v2` and `-msvar-v2`, with the new version hashes. Interval conditions written as a distance from the base move by the difference of the base medians between the old and the new version of that model, as was done for the MS-VAR copies; absolute numbers from the event stay. Leave `approved_by` empty and ask the owner to confirm before calculating.

Run the base and 11 scenarios on each model, `draws=2048`, `seed=20261029`, 24 calculations. Run the gates. Complete the backtest for both final versions against `5c57…` and `ecaa…`.

Write into `plans/macro-models-results.md`: the table of the first section with a column per version; G1–G6 per version; the driver band share from the corrected check, reported separately; which single model the data favour, by G6 and then by the gates.

Update `plugins/bvar/README.md`, `plugins/msvar/README.md` and their English twins, `docs/howto/first-session` if the recommended command changes, and `CHANGELOG` in both languages. `tests/test_docs_translation.py` must pass.

## Acceptance

The reviewer reruns these from the saved artefacts.

1. G1–G5 hold for at least one final version on all 12 calculations, or the results file states which gate failed, for which variable, with the measured value. A failed gate is an acceptable result if it is stated; a loosened gate is not.
2. G6 holds for any version recommended for use.
3. The results file reproduces "Before" from the saved runs, so the comparison is on one method.
4. `5c57…`, `ecaa…` and their runs are untouched.
5. `uv run ruff check src tests` and `uv run pytest -q` pass, slow tests included, and CI of the branch is green.

## Later

Found on 2026-11-11, not started; each needs the rolling check on the control half before it replaces a version.

- **Asymmetric volatility in `bvar.sv`.** The volatility rises equally after falls and after rises. On the control dates IMOEX fell below p5 in 17 % of the months instead of 5 %, and rates and spreads rose above p95 in 25–36 %, so ES95 is probably understated. Let the log variance respond to the sign of the last shock (`sv.py` and the sampler).
- **Rate tails by policy regime.** The impulse dummies of February and March 2022 take the largest moves of the key rate out of the history: the largest monthly key-rate surprise the volatility ever reached is 0.57 pp (median draw of `6d63e6de`), 0.27 pp now. A 5 pp unscheduled cut is then a shock of 9 standard deviations or more whatever the identification, and `cbr-cut-5pp-unscheduled-2026-11` is refused by the admission check. Compare `msvar.markov` with `bvar.sv` on the same check, by the tails of rates; show both models if `msvar.markov` is better there. Screened on 2026-11-12 (1000 draws, the full sample): on the control half the key rate and RUONIA fall below p5 in 9.3 % of the cells and rise above p95 in 24.3 %, the Fed rate in 8.2 % and 15.7 %; over two months the model gives a key-rate move of 3 pp or more 0.13 % (down) and 0.27 % (up) of the time against 1.7 % and 2.9 % of the months of 2012-2026. Dropping the impulse dummies of 2014 and 2022 from the rate equations raises the largest surprise of the history from 0.58 to 0.94 pp but leaves the current one at 0.26 pp. A loose vol-of-vol prior (df 5) for the key rate and RUONIA gives s = 1.39 a month, a current surprise of 0.06 pp and a 12-month sd of 34.8 pp for the key rate (history 4.25) and 70 for the rouble (19.3); with the Fed rate too the paths reach 1e8. A random-walk log volatility cannot carry a rate that sits still and then steps: the fix is a jump or regime component for the policy rates, not a looser prior.
- **Satellite noise and the tail rule.** `constant_spreads` was chosen on 2026-11-11 by the mean holdout RMSE, not by the driver CRPS on the episodes the satellite plan asks for. Every run carries the flag of an unchecked tail because no driver meets the rule of fewer than 1 % of paths outside the training range: revise the rule or widen the range.
- **Bond spreads around put dates.** Before a put or a coupon reset the exchange yield to the put jumps (RU000A0JXPG2 before 2026-04-17: 16–60 % with a duration of 1–10 days), and the monthly median of March 2026 is 7.71 pp against 3.6 after it. Leave out of `MoexBondSpread` the days with less than a month of duration, fetch the series again and re-estimate the satellites.
- **Income of shares and loans.** `bond.paths.v1` counts the yield earned over the horizon since 2026-11-11; `equity.paths.v1` counts the price only, without dividends, and `credit.paths.v1` loses `LGD` times the rise of the overdue share without the interest earned. Add a dividend yield and an interest rate as analyst terms, or state the convention in every report.
- **Hidden volatility under an exact condition.** With `ru.fed_rate = 0` in month 4 (`fed-zero-dxy-gold-2026-11-v7`) the effective number of volatility candidates is 1 of 64 on every path: the density of the exact value gives one candidate all the weight. Draw more candidates, or temper the weights, and report the effective number in the admission.
- **The served verdict on a near tie.** PLZL is refused by 32.9 against 32.4 of the AR(1), on a holdout that holds the fall of 2026-07-08 (−26 % in a day). A verdict on the difference of two RMSE without its interval swings on one event; consider a paired test of the forecast errors before refusing a driver that served before.
