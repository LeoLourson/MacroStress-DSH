# Criteria of trust in a forecasting model

A model is trusted when the backtest has confirmed every property of its forecast with a statistical criterion of known characteristics. A correct model fails the backtest in at most 10 % of cases, and for every named defect the probability of finding it is known. The criteria, the thresholds and the order of reading the data are the same for every plugin. A release of a version comes with a trust report that gives the power of every criterion beside its result, so the reader sees both what is proven and the defects the backtest cannot notice on the data at hand.

## How the backtest works

The backtest places the model at a past date $e$ and gives it only the data published by that day. From them the model draws $N$ random paths of every quantity $q$ over the months $t=1,\dots,H$, with $H$ at most 14, and the backtest compares them with the fact $y_{e,q,t}$. The dates come from a regular grid with a step of 3 months and from crisis episodes. In an episode the model knows the scenario, that is the realized values of a few quantities over the horizon, and forecasts the rest. Every level of the forecast is judged: the macro variables, market quantities such as share prices and bond spreads, the single positions of the portfolio and the portfolio as a whole.

Let $x^{(1)},\dots,x^{(N)}$ be the paths of quantity $q$ in month $t$ of window $e$, $\hat F$ their empirical distribution function and $Q_p$ its $p$-quantile. The triple $(e,q,t)$ is a cell, and the 90 % interval $B=[Q_{0.05},Q_{0.95}]$ of width $w=Q_{0.95}-Q_{0.05}$ is the band of the cell. The number $u=\hat F(y)$ is the probability integral transform of the fact: the fact lies below the band when $u<0.05$ and above it when $u>0.95$.

The error of a cell is the continuous ranked probability score (CRPS)

$$
\operatorname{CRPS}(\hat F,y)=\int_{-\infty}^{\infty}\bigl(\hat F(z)-\mathbf 1\lbrace z\ge y\rbrace\bigr)^2dz=\frac1N\sum_i\lvert x^{(i)}-y\rvert-\frac1{2N^2}\sum_{i,j}\lvert x^{(i)}-x^{(j)}\rvert .
$$

It is zero for a forecast concentrated at the fact and grows with a shifted centre and with excess width of the band. The model $M$ is compared with a benchmark $R$ on the common cells:

$$
\rho_{q,t}=\frac{\sum_e\operatorname{CRPS}^M_{e,q,t}}{\sum_e\operatorname{CRPS}^R_{e,q,t}},\qquad S_{M/R}=\exp\Bigl(\frac1J\sum_t\frac1{K_t}\sum_q\log\rho_{q,t}\Bigr),
$$

where the sums run over the windows in which both forecasts are scored, $K_t$ is the number of quantities scored in month $t$ and $J$ the number of such months. Every quantity weighs the same, so the aggregate does not depend on units, and $S_{M/R}<1$ means the model is more accurate than the benchmark. The score $\rho_q$ of one quantity is the same formula with $K_t=1$.

The benchmark is the random walk with drift on the known history, and, when the scenario fixes the path of some quantities, the conditional walk that knows that path. For the positions and the portfolio the benchmark is the historical simulation: past price changes over the same horizon, revalued by the same position models.

Neighbouring windows of the grid overlap and the cells of one window move together, so no cell is an independent observation. The criteria compute their statistics on batches: the dates of the grid are cut into batches of consecutive windows as long as a window, and an episode is a batch of its own. The months of the horizon no longer than the step of the grid are the exception: there the windows of neighbouring dates do not overlap, and every date is a batch. Significance comes from a t-test over the batch values with one degree of freedom fewer than batches, and with one batch the property is not tested. The intervals of the scores in the report tables come from a block bootstrap of 1000 repetitions with the same blocks.

## Properties of a forecast

A MacroStress forecast is the distribution of the quantities given an event, and the user reads bands, tails and portfolio losses from it. The backtest checks these properties of the forecast.

| Property | Content | Where it is checked |
|---|---|---|
| Completeness | there is a forecast on every date with enough data | grid and episodes |
| Domain | the band stays within the admissible values of the quantity | grid and episodes |
| Plausible width | the band is not wider than the range of past changes of the same length | grid and episodes |
| Plausible scenario | the model without the scenario does not find its conditions almost impossible | episodes |
| Calibrated tails | at every level the fact lies below and above the band in at most 5 % of cases | grid |
| No failing quantity | no quantity loses to its benchmark by more than allowed | grid |
| Accuracy | the aggregate $S$ loses to the benchmark by no more than allowed | grid |
| Harmless scenario | the forecast with the scenario is not worse than the same model's without it | episodes |
| Correct portfolio tail | the frequency and the size of losses beyond VaR95 agree with the forecast | grid |
| Useful scenario | the forecast is more accurate than the conditional walk | episodes |

Every property except the useful scenario is required, and a model that breaks any of them is not trusted. The usefulness of the scenario is additional. When it is not proven the model stays usable, and the trust report says that carrying the scenario to the other quantities added no accuracy.

Calibration and accuracy are judged on the grid because nobody selected its dates. The episodes were selected as crises, and the share of facts outside the band in them reflects the selection. The episodes therefore only compare two forecasts on the same windows, and the selection acts on both at once.

## How a criterion is built

Every criterion is defined by a statistic, the data and the null hypothesis "the model has the property", and it fails only when the data are significantly against it. The threshold of failure follows from the error budget rather than being set by hand. For every criterion the trial on synthetic data measures the share of false rejections of a correct model and the power against named defects (section "Trial of the criteria").

There are two kinds of thresholds. A statistical threshold separates a deviation from chance and depends on the error budget and the number of windows. An allowance states what loss of accuracy does not matter to the user, and a human approves it. All allowances are in one table:

| Symbol | Value | Meaning |
|---|---|---|
| $\delta_S$ | 0.02 | how much the aggregate $S$ on the grid may lose to the benchmark |
| $\delta_q$ | 0.10 | how much the score $\rho_q$ of one quantity may lose to its benchmark |
| $p_{\min}$ | 0.01 | the smallest probability of an interval condition of a scenario in the forecast without it |

Properties a correct model never breaks have no allowance. With enough data a correct model does not refuse to forecast and does not put its band outside the domain, so completeness and the domain allow no break at all.

## Error budget

A correct model fails the backtest with probability at most $\alpha=0.10$. Half of the budget goes to the admission conditions and half to the decision on a version, and within each half the share is split evenly between the statistical criteria. The admission has four (the moves rule, the tails, the failing quantity and the portfolio tail), each with $\alpha_c=\alpha/8=0.0125$. The decision has two (the accuracy and the harm of the scenario), each with $\alpha_c=\alpha/4=0.025$. A criterion that tests $m$ quantities with the share $\alpha_c$ fails by Bonferroni's correction: when the smallest of its $m$ p-values is below $\alpha_c/m$. Holm's step-down procedure at the same level lists the quantities that broke the property. The correction holds under any dependence between the quantities. Under strong dependence it is conservative: the true share of false rejections is below $\alpha_c$, but so is the power, and the trial of the criteria measures both.

Without the correction a correct model would almost surely fail the backtest. The 23 macro variables have two tails each, 46 hypotheses in all, and if each were tested independently at 5 %, at least one would be rejected with probability $1-0.95^{46}\approx0.91$. The correction takes power in its turn: with 277 quantities the threshold of the p-value falls to $0.0125/277\approx4.5\cdot10^{-5}$, and a t-test over a dozen batches finds almost nothing. The tails are therefore tested for a level as a whole, and a single quantity answers for itself through the failing-quantity criterion.

## Trial of the criteria

The characteristics of the criteria are measured on synthetic data where the correct model is known. A synthetic world repeats the shape of the grid: 23 quantities, 240 months of history before the first date, 55 dates 3 months apart, the last 23 of them the validation half, a horizon of 14 months. The histories come from a multivariate Student walk with four degrees of freedom, a common scale multiplier, unit variance of a step, pairwise correlation 0.3 and a drift of 0.05 a month. The portfolio of the world is the mean change of the quantities. The correct model forecasts by the same law, a defective one by the law with one defect added, and the benchmark is the same random walk with drift as in the real backtest. The synthetic cells go through the same code as the real ones.

Every criterion is tried against the defects it must find:

| Defect | How it is added | Criterion that must find it |
|---|---|---|
| Narrow band | width times 0.8 | calibrated tails |
| Wide band | width times 1.3 | accuracy, plausible width |
| Shifted centre | drift moved by half the standard deviation of a year's change | calibrated tails, accuracy |
| Thin tails | normal step with the same 95th percentile, ES below the correct one | correct portfolio tail |
| Failing quantity | width of one quantity times 2 | no failing quantity |

Over 200 synthetic worlds per model the trial counts the share of worlds in which every criterion and the backtest as a whole reject. For the correct model that is the share of false rejections, for a defective one the power against its defect. The trial of 17 November 2026 gave these shares of worlds with a rejection:

| Model | Moves rule | Tails | Failing quantity | Accuracy | Portfolio tail | Whole backtest |
|---|---:|---:|---:|---:|---:|---:|
| correct | 0.035 | 0 | 0 | 0 | 0 | 0.035 |
| narrow band | 0 | 0.555 | 0 | 0 | 0.015 | 0.555 |
| wide band | 0.895 | 0 | 0 | 0.025 | 0 | 0.895 |
| shifted centre | 0.035 | 1 | 0.54 | 0.685 | 1 | 1 |
| thin tails | 0.005 | 0.005 | 0 | 0 | 0 | 0.01 |
| failing quantity | 0.93 | 0 | 0.025 | 0 | 0 | 0.93 |

The correct model fails the backtest in 3.5 % of worlds, well within the 10 % budget. The wide band, the shifted centre and the failing quantity are found almost always, though the moves rule catches the failing quantity rather than its own criterion. A band narrowed by 20 % is found in 56 % of worlds. A tail lightened beyond the 95th percentile with the right frequency of exceedances is not seen: over 55 windows the statistic $Z$ tells only a gross understatement of ES. These numbers draw the boundary of trust: a model that passed the backtest may, with probability about 0.45, have a band 20 % narrower than correct, and may understate ES at a correct VaR95.

The trial is repeated whenever the set of windows or the code of the criteria changes. It is run by `python -m mst.metrics trial`, and a test in `tests/metrics` on 20 small worlds checks that the correct model fails the grid criteria in no more than a quarter of the worlds. The harm and the usefulness of a scenario need episodes with a scenario, and the trial does not cover them yet.

A rule that does not pass the trial is not a criterion. That is why the levels rule was dropped, which broke a cell when its band was wider than the range of all past levels of the quantity. On a Student walk with four degrees of freedom and an exact 90 % band it breaks 19 % of the cells at a horizon of 12 months and 26 % at 14 months with 120 months of history, 6 % and 10 % with 170 months. In effect the rule demands a reversion to the mean that a walk does not have. If a quantity is bounded, its domain sets the bound.

## Data and the order of reading

Only data the author did not see while tuning reveal fitting to the data. The windows are therefore split into sets with different orders of reading:

| Set | Windows | When it is read |
|---|---|---|
| Tuning | grid dates before 2021 and episodes marked `tuning` | without limits |
| Validation | grid dates from 2021 and episodes marked `validation` | when comparing versions, every reading logged in the journal of edits |
| Closed | windows that were not in the data on the freeze date | once per release of a version |

Every date before the freeze has already been read many times, so the closed set can consist only of windows that appeared later: the journal of future forecasts and the episodes added after the freeze. Every month the forecast of the version on the main branch is saved at every level, and 14 months later the window has its full fact. An episode added after the freeze joins the closed set although the model saw its dates on the grid, because its scenario and its outcome took no part in the tuning.

While the closed set is smaller than the power found by the trial requires, the trust report rests on the validation set and states how many times it has been read.

## Admission conditions

The admission conditions sort out versions whose forecast cannot be taken seriously before accuracy is compared. The admission reads every window of a check, not only the validation half: every date of the grid is forecast from its own past, and there are twice as many batches.

Completeness. The grid must have no date with enough data where the model gave no forecast, otherwise the score over the remaining dates would hide a weakness of the model. When the model lacks the history of one quantity, only its cells drop out, and an episode drops out whole when such a quantity is part of its scenario. The report lists every dropped quantity, episode and position with the reason.

Domain. No band leaves the admissible values of its quantity; a price band, for example, does not go below zero.

Plausible width. A cell of month $t$ breaks the moves rule when

$$
w>\max_s\Delta_s-\min_s\Delta_s,\qquad \Delta_s=y_{s+k}-y_s,\qquad k=t+g,
$$

where $g$ is the number of months from the last known value of the quantity to the end of the history. The band of month $t$ forecasts a change over $k$ months, so it is compared with past changes of the same length. The rule applies when at least 200 such changes are known; then on a Student walk with four degrees of freedom it breaks under 0.5 % of the cells of a correct band. Let $\pi_q$ be the share of breaks among the cells of quantity $q$ the rule applies to. The criterion fails when $\pi_q$ of some quantity is significantly above 0.005, the share of a correct band; significance comes from a t-test over batches with Bonferroni's correction for the number of quantities.

Plausible scenario. Every interval condition $c$ of a scenario must have the probability $P_0(c)>p_{\min}$ in the forecast without the scenario, otherwise the scenario rests on an event the model finds almost impossible. Point conditions are outside this threshold, since the tolerance of the condition sets their probability.

Calibrated tails. For every level the shares of facts below and above the band on the grid, $\hat a^-$ and $\hat a^+$, are counted over the cells of its quantities in the months of the horizon no longer than the step of the grid, where every date is an independent batch; at the long months the windows overlap and leave about eleven batches in the whole history. The null hypothesis of each share is that it is at most 0.05; the p-value comes from a t-test over the batches, and the failure from Bonferroni's correction for twice the number of levels. The tails are counted apart because a shifted forecast can cover the fact at the right overall frequency while the fact leaves it more often on one side and less often on the other. A fact leaving the band too rarely means excess width, which the score penalises, so the tails criterion does not reject it.

No failing quantity. For every quantity on the grid the hypothesis $\rho_q\le1+\delta_q$ is tested, and the criterion fails when, with Bonferroni's correction, some quantity is significantly worse than its benchmark beyond the allowance.

Correct portfolio tail. The portfolio tail is the calibration of the main number of the product, a property of the law and not a comparison of versions, so it belongs to the admission and reads every window of the grid at the month of the horizon equal to the step of the grid. It is checked by the Acerbi–Szekely statistic

$$
Z=\frac1{0.05E}\sum_{e=1}^{E}\frac{\Delta_e\mathbf 1\lbrace\Delta_e>\mathrm{VaR}_e\rbrace}{\mathrm{ES}_e}-1
$$

and by the share of exceedances of VaR95. Here $\Delta_e$ is the realized loss of the portfolio in window $e$, $\mathrm{VaR}_e$ the forecast 95th percentile of the loss, $\mathrm{ES}_e$ the mean loss over the worst 5 % of paths and $E$ the number of windows. Under a correct tail $\mathbb E Z=0$, and a positive $Z$ means the extreme losses are understated. The criterion fails when $Z$ is significantly above zero or the share of exceedances significantly above 5 %; each of the two tests gets half the share of the criterion, and significance comes from a t-test with every date a batch. Without a portfolio the property is not checked, and there is no trust.

## Decision on a version

The decision compares and reads the validation set of the target level, and at a release the closed set. A version without admission is rejected. A version with admission is trusted when both comparisons hold:

- accuracy: $S_{M/R}$ on the grid is not significantly above $1+\delta_S$;
- harmless scenario: $S_{M/M_0}$ in the episodes is not significantly above one, where $M_0$ is the same model without the scenario.

The usefulness of the scenario is proven when $S_{M/R}$ in the episodes, with $R$ the conditional walk, is significantly below one.

A new version replaces the incumbent when it is trusted and either the incumbent cannot be trusted, or the new version is significantly better on at least one property and significantly worse on none. A difference below the statistical threshold is not proven, and then the incumbent stays.

## Trust report and journal of edits

The metrics package builds the trust report at a release of a version. For every property it gives the statistic, its interval, the threshold, the verdict and the power from the trial of the criteria, for example "a band narrowed 0.8 times is found by the criterion with probability 0.56". The report lists the proven additional properties, the number of readings of the validation set and the size of the closed set.

The author of an edit logs it in the journal before reading the validation set: the hypothesis, that is which property the edit should improve and why, the changed code and the result on the tuning set. The result on the validation set is added afterwards. An edit that improved the validation set but worsened the tuning set is accepted only with a logged explanation of why its effect depends on the period, for example on the regime that began in 2022; without one, the improvement counts as luck. The journal shows why the version is built as it is and how many attempts stand behind its result.

## What the backtest does not see

The validation half of the grid holds 23 dates from 2021 to 2026, 3 months apart, and the horizon is 14 months, so it has about five non-overlapping windows, and the whole grid about eleven. A defect of one quantity is therefore found only when it is gross, while the aggregate of a level notices smaller defects; how much smaller is what the trial of the criteria shows on a synthetic world with the same shape of grid.

The backtest does not see a change of regime that did not happen in 2012–2026, since all windows come from that period. The uncertainty of the model's parameters is not part of the forecast and shows only through the calibration. The backtest checks the joint distribution only through the portfolio, and the decision does not read the correlations of quantities outside it. Scenarios written by a language model may carry knowledge of the outcome, so the report scores them apart from scenarios written by a human.

## Where it is in the code

[baselines.py](<../../src/mst/metrics/baselines.py>) checks the domain and moves rules and the condition on a scenario, [cells.py](<../../src/mst/metrics/cells.py>) computes the ranges of the moves and the CRPS. [sanity.py](<../../src/mst/metrics/sanity.py>) collects the shares of breaks per quantity, [quality.py](<../../src/mst/metrics/quality.py>) computes the score $S$ and the bootstrap, [es.py](<../../src/mst/metrics/es.py>) the statistic $Z$. [episodes.toml](<../../bench/model-backtest/episodes.toml>) sets the episodes, their halves and the grid of dates, and the [package README](<../../src/mst/metrics/README.en.md>) describes how to run it.

[trust.py](<../../src/mst/metrics/trust.py>) implements the admission and decision criteria, the error budget and the allowances, [trial.py](<../../src/mst/metrics/trial.py>) the trial of the criteria, [decision.py](<../../src/mst/metrics/decision.py>) the rule replacing the incumbent. The journal of edits is kept in [journal.md](<../../bench/model-backtest/journal.md>). The journal of future forecasts, the closed set and the lines of power in the trust report are not implemented yet, and the trial does not cover the harm and the usefulness of a scenario.
