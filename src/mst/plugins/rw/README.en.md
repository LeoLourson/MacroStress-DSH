# Joint random walk with heavy tails

The `rw` plugin gives one level-0 macro model, `rw.joint`. It is chosen explicitly, on a par with `bvar.sv` and `msvar.markov`. This text was written before the code and states what the code must do, so where they differ the code is fixed.

## Why this model

The system needs a level-0 macro model for two things: the base law of the macro variables over a horizon of up to 15 months, and carrying the conditions of a scenario over to the other variables. `bvar.sv` failed at both.

Its base law is no more accurate than the baseline: on rolling dates of 2021–2026 its CRPS is 1.02 of the random walk's. Worse, it carries conditions through the responses to named shocks, and the data do not determine their size. Signs fix the direction of a response, not its strength: per 1 pp of the key rate the rouble moves between −17 % and −1 % in the month of the shock, output between −11 % and +8 % (the 5th and 95th percentiles over the draws of version 51ead920), and by month 12 the lags multiply these numbers. A scenario that cut the key rate by 5 pp thus gave a 95th percentile of 284 roubles per dollar after two months and a median rise of IMOEX by 39 %.

Two conclusions follow, and the model rests on them.

First, the causal effect of a Bank of Russia decision cannot be estimated from the Russian data of 2012–2026. The monthly changes of the key rate and of the log of USD/RUB correlate at +0.6, and the regression gives 2.97 % of rouble weakening per 1 pp of hike: this is how the data describe a central bank that answers a rouble collapse. Without the 9 months of the largest rouble moves the correlation falls to 0.19. The scenario therefore sets the central bank's decision and the market's reaction to it, and the model carries what was set over to the other variables by their joint law, claiming nothing about causes.

Second, a stress test needs the tails more than the dynamics. The kurtosis of the monthly changes is 47 for the key rate, 56 for the CPI, 18 for unemployment, 12 for USD/RUB and 9 for IMOEX, against 3 for the normal law. In a crisis month the variables move together and far. The model carries this by one common scale multiplier per month.

The model is simple on purpose: no lags, no rotations and no sign table. It differs from the baseline in heavy tails, a joint law of the variables and zero drift for rates, so a CRPS comparison with the baseline tests exactly these three choices.

## Model

Denote by $y_t\in\mathbb R^n$ the vector of the $n=23$ macro variables of the registry `src/mst/macro/registry.toml` in month $t$ on the model scale: rates, spreads and unemployment in their own units, prices, exchange rates, indices and volumes as $100\ln X$. The variables follow the panel order: first the $n_g=4$ global ones (`gl.`), then the $n_r=19$ domestic ones (`ru.`). The model:

$$
y_t=y_{t-1}+\mu+\sqrt{\lambda_t}\,Bz_t,\qquad z_t\sim N(0,I_n),\qquad \frac{1}{\lambda_t}\sim\frac{\chi^2_\nu}{\nu},
$$

where all $z_t$ and $\lambda_t$ are independent over $t$ and of each other. Here $\mu$ is the drift, $\Sigma=BB^\top$ the scale matrix of the changes, $\lambda_t>0$ the common scale multiplier of month $t$, and $\nu>2$ the degrees of freedom. The change of a month has a multivariate Student law with covariance $\frac{\nu}{\nu-2}\Sigma$. The multiplier is common to all variables, so in a month with a large $\lambda_t$ every variable moves widely at once. That is the crisis month of the model.

The shocks $z_{t,j}$ carry the names of variables: the shock `ru.key_rate` is the key rate's own innovation. The matrix $B$ is the Cholesky factorisation of $\Sigma$ in the order the scenario chooses (section "Causes"). Without a scenario the order does not matter, because the law of the paths depends on $\Sigma$ alone.

In any order the global variables come before the domestic ones, so $B$ is block lower-triangular and domestic shocks do not move global variables. This is the small-open-economy assumption of `bvar.sv`.

The parameters $\theta=(\mu,\Sigma,\nu)$ are estimated once and stored in the version. The level-0 outcome in the sense of the general scheme is

$$
\omega^0=(\theta,u,\boldsymbol\lambda,\mathbf z),\qquad \boldsymbol\lambda=(\lambda_1,\dots,\lambda_H),\qquad \mathbf z=(z_1,\dots,z_H),
$$

where $u$ are the standard normal coordinates of the values of month 0 not yet published on the calculation date (section "Starting value"). The model declares the path of multipliers $\boldsymbol\lambda$ as the hidden coordinate `scale_path` (the `latents` field of the `MacroModel` contract).

## Estimation

### Data

The model reads the panel of `bvar.sv`: months from 2012-01 to the calculation date, only the values published by that date, seasonality removed by the platform, a month without a value left as a gap. Changes are taken between neighbouring months where both values are known. The change of a month named in the `breaks` of the specification is left out for the variables listed there: these are breaks of the series, not market moves. There is one break now — the splice of the MOEX spread indices in 2021-01 for `ru.spread_aa`, `ru.spread_a`, `ru.spread_bbb`.

### Drift

The drift is non-zero only for the variables in the `drifts` list of the specification: nominal prices, exchange rates and volumes that grow with the price level (`ru.cpi`, `ru.usdrub`, `ru.cnyrub`, `ru.output`, `ru.real_wage`, `ru.retail`, `ru.credit`, `ru.housing_price`, `ru.imoex`). For such a variable $\mu_i$ is the mean of its monthly changes over the whole panel. For the others $\mu_i=0$: the sample trend of 2012–2026 in rates, yields, spreads, unemployment, Brent and the VIX (+0.60 pp a year for the key rate, for example) is no forecast.

The drift is a plain mean taken before the scale is estimated and stays fixed in that estimation. This is a simplification: the maximum likelihood location of a Student law is a weighted mean. The plain mean equals the baseline's drift.

### Scale and degrees of freedom

$\Sigma$ and $\nu$ are the maximum likelihood estimates of a multivariate Student law on changes with gaps, found by the ECME algorithm (Liu, Rubin, 1995). Each iteration:

1. For month $t$ take the known changes $d_t^{o}$ (the subvector of variables $o_t$). The distance from the centre is $\delta_t=(d_t^{o}-\mu_{o})^\top\Sigma_{oo}^{-1}(d_t^{o}-\mu_{o})$, the weight of the month is $w_t=(\nu+p_t)/(\nu+\delta_t)$, where $p_t$ is the number of known changes.
2. The unknown changes of the month are replaced by their conditional mean given the known ones, giving a full vector $\hat d_t$. Denote by $V_t$ the conditional covariance of the unknown changes given the known ones, padded with zeros in the rows and columns of the known ones.
3. The new scale matrix is $\Sigma=\frac{1}{T}\sum_t\big[w_t(\hat d_t-\mu)(\hat d_t-\mu)^\top+V_t\big]$, the sum running over the $T$ months with at least one known change.
4. The new $\nu$ maximises the observed likelihood on the interval `nu_bounds` at the new $\Sigma$. This is a one-dimensional problem solved by Brent's method.

The iterations stop when the log likelihood changes by less than `em_tol`, or after `em_iterations` iterations; in the latter case the estimation stops with an error that names the change reached. The starting values are the sample covariance over pairwise known months, corrected to positive definiteness, and $\nu=10$.

A crisis month gets a small weight $w_t$, so the correlations come mostly from ordinary months, and crises enter the tail of the law through $\nu$. The same mechanism weakens the bias described above: the correlation of the rate with the rouble is no longer set by 9 crisis months.

### Housing price

`ru.housing_price` is known in 32 months out of 177 and has no monthly changes. It does not enter ECME. Its scale and correlations are estimated from the changes between neighbouring observations as those of a Gaussian random walk with gaps. Let $s_0<\dots<s_m$ be the months of observation, $k_j=s_j-s_{j-1}$, $D_j$ the change over the interval. Then

$$
\sigma_h^2=\frac{1}{m}\sum_{j=1}^{m}\frac{(D_j-k_j\mu_h)^2}{k_j},\qquad
\hat\rho_{hi}=\frac{1}{q\,\sigma_h\sigma_i}\sum_{l=1}^{q}\frac{(D^h_l-k_l\mu_h)(D^i_l-k_l\mu_i)}{k_l},
$$

where in the second formula the intervals are those shared by the housing price and variable $i$, $D^i_l$ is the change of variable $i$ over the same interval, and $\sigma_i^2=\frac{\nu}{\nu-2}\Sigma_{ii}$ is the variance of its monthly change. The row and column of the housing price in $\Sigma$ are $\frac{\nu-2}{\nu}\,\sigma_h\sigma_i\hat\rho_{hi}$, the covariance carried to the scale of the Student law. If the resulting matrix is not positive definite, its correlation part is replaced by the nearest correlation matrix in the Frobenius norm with eigenvalues at least `eig_floor` (Higham, 2002); the diagonal stays. The housing estimates are an approximation: they rest on 31 intervals and on a Gaussian law.

### Parameter uncertainty

The uncertainty of $\theta$ is not counted: the version stores one estimate. With 177 months it would widen the 12-month spread of USD/RUB by about 3 %: a standard error of the drift of 0.4 % a month gives 5 % over a year against 19 % of noise, 19.7 % in quadrature. Hence the $R=(\theta,\dots)$ of the general scheme holds here only the coordinates the scenario does not name.

## Starting value

Series are published with different lags, so in month 0 — the last month in which at least one variable is known — some variables are unknown. The model carries an unknown component to month 0 by its own independent steps from the last known value: $k$ steps give the law $N\!\left(k\mu_i,\,k\,\frac{\nu}{\nu-2}\Sigma_{ii}\right)$, written as $m_0+L_0u$ with a diagonal $L_0$. The baseline does the same. Conditioning on the known moves of other variables (a Kalman filter) is deferred: on the panel to 2026-09 the real wage lags by two months, unemployment, retail, credit and the mortgage rate by one, the rest are known in month 0.

## Causes

A scenario names its causes — the coordinates of the outcome whose law it changes. For `rw.joint` these are the names of variables (the $J$ set of a version is all 23 registry ids) and the hidden coordinate `scale_path`.

A named variable comes first within its block in the Cholesky order. The order is: global variables among the causes in registry order, the other global ones, domestic variables among the causes in registry order, the other domestic ones. Denote by $P$ the matrix of this permutation and by $C$ the lower-triangular Cholesky factor of $P\Sigma P^\top$; then $B=P^\top CP$. In substance: the named variables move by their own innovations, and the others respond in the same month by the regression of their changes on the changes of the named ones. The strength of the response comes from the joint law and says nothing about causality.

Different scenarios give different $B$, but always $BB^\top=\Sigma$, so the base law $P$ does not depend on the order. The order is a choice of the representation of the outcome that makes the named causes coordinates.

If `scale_path` is among the causes, the conditions change the law of the monthly multipliers too: the event can choose a crisis scale. If it is not, the multipliers keep their base law, and a large event is met by large innovations of the named variables at an ordinary scale. If no causes are named, by the general scheme every coordinate except $\theta$ is a cause, `scale_path` included.

### Central bank decisions and the market's reaction

The model does not know how the market reacts to a central bank decision. The regression of the rouble on the rate describes a central bank that answers a rouble collapse and gives the wrong sign for an unscheduled cut. A scenario that sets a central bank decision therefore also sets the market's reaction: the path of the exchange rate and, where something is known about it, of OFZ yields. The numbers are marked by the general rules — by the event, by a corpus source or by the word "assumption" — and all such variables are named as causes. The model carries the other variables by the joint law.

## Forecast without a scenario

$$
y_t=y_0+t\mu+B\sum_{s=1}^{t}\sqrt{\lambda_s}\,z_s,\qquad t=1,\dots,H.
$$

With $y_0$ known, the mean of $y_t$ is $y_0+t\mu$ and its covariance $t\frac{\nu}{\nu-2}\Sigma$; the tails are the heavier the shorter the horizon, since a sum of $t$ independent Student changes approaches the normal law as $t$ grows.

Rates, unemployment and the BBB spread have a lower bound in the registry. As with `bvar.sv`, the model draws an unbounded shadow value, the conditions and their price are computed on the shadow, and the platform reads the observed value $\max(s_t,\text{lower})$ (`mst.model_sdk.bounds.observed`).

## Scenario

A scenario reaches the model as a plan: the horizon $H$, constraints $\mathbf a\le W\mathbf y\le\mathbf b$ on the path $\mathbf y=(y_1,\dots,y_H)$ and the causes $D$. The platform widens an exact value beforehand to an interval of half-width $\tau$. It subtracts the impacts $O$ from the bounds and adds them to the paths after the draw, so the model does not see them.

Given the path of multipliers $\boldsymbol\lambda$ the path is linear in $\varepsilon=(u,\mathbf z)$:

$$
W\mathbf y=c+Z(\boldsymbol\lambda)\,\varepsilon,
$$

where $c$ is the value of $W\mathbf y$ at $\varepsilon=0$. A term $w\,y_{i,t}$ of a constraint contributes $w(L_0)_{i\cdot}$ to the columns of $u$ and $w\sqrt{\lambda_s}B_{ij}$ to the column of $z_{s,j}$ for every $s\le t$. Given $\boldsymbol\lambda$ the problem is therefore that of a Gaussian model, and the common machinery of `mst.model_sdk.conditional` solves it (Antolín-Díaz, Petrella, Rubio-Ramírez, 2021): exact rows are fixed, interval rows are drawn by a Gibbs sampler (Geweke, 1991), linearly dependent rows are refused.

The path of multipliers is drawn as the volatility path of `bvar.sv` is. If `scale_path` is among the causes (or no causes are named), the model draws `LATENT_CANDIDATES` candidate $\boldsymbol\lambda$ from the base law, weighs each by the probability of the box under it (`log_condition_weight`) and picks one by the weights (`pick_latent`). The innovations in $D$ are then drawn under the conditions at the chosen $\boldsymbol\lambda$. If `scale_path` is not among the causes, the path of multipliers belongs to $R$ and is drawn from the base law without weights.

A scenario contradicts the model if some linear combination of the constraints does not depend on the named causes but is random given $R$. Most often the constraints fix the variables of a set $I$ in every month of the horizon. The scenario is then feasible only if every fixed variable is named as a cause or is a linear function of the named ones in the same month. Since domestic shocks do not move global variables, a fixed global variable needs a named global cause. The model refuses such a scenario and names the constraints (`InfeasibleScenario`).

`constraint_law` returns the law of $W\mathbf y$ given $R$ in rows of groups, one row per candidate of multipliers with its weight, as `bvar.sv` does. Without named causes there are no groups, and the rows describe the unconditional law of $W\mathbf y$ under the candidates.

The model reports the size of the innovations of the named causes in standard deviations (`diagnostics["driving_sd"]`). As with the other models, this is a flag of the report, not a refusal.

## Difference from the baseline

The `random_walk` baseline continues every variable on its own by Gaussian steps with the sample drift and spread. `rw.joint` differs from it in three places: the drift of rates, spreads, unemployment, Brent and the VIX is zero; the tails are heavy with a common crisis scale; the law is joint. The difference in CRPS between the model and the baseline on the episodes therefore measures what these choices add. If only the fixed variables are named as causes and the scale is not, the conditional law of the change of the other variables in month $t$ given $\lambda_t$ is

$$
N\big(\mu_y+\Sigma_{yc}\Sigma_{cc}^{-1}(\Delta c_t-\mu_c),\ \lambda_t(\Sigma_{yy}-\Sigma_{yc}\Sigma_{cc}^{-1}\Sigma_{cy})\big),
$$

the same linear projection as the conditional baseline `conditional_walk`, with coefficients in which crisis months weigh less and with a heavy-tailed residual.

## Assumptions

There is no mean reversion: the spread grows as $\sqrt t$, rates and spreads included. Over a horizon of up to 15 months this is acceptable; over a longer one it overstates the spread of rates. The links between variables are only contemporaneous: the model does not carry the delayed pass-through of the exchange rate into prices, which comes from the scenario or a satellite. The crisis scale is common to all variables: a crisis in the rouble alone and a crisis in everything differ only by the innovations. The drift of prices and exchange rates comes from the history of 2012–2026. Domestic variables do not affect global ones. Parameter uncertainty is not counted.

## Parameters

| Parameter | Value | Meaning |
|---|---|---|
| `start` | 2012-01-01 | the first month of the panel, as for `bvar.sv` |
| `drifts` | the nine variables above | the variables with a non-zero drift |
| `breaks` | 2021-01: three spreads | changes left out at breaks of the series |
| `nu_bounds` | $[2.5,\ 60]$ | the search interval of $\nu$ |
| `em_tol` | $10^{-8}$ | the threshold on the change of the log likelihood |
| `em_iterations` | 500 | the largest number of ECME iterations |
| `eig_floor` | $10^{-8}$ | the lower bound of the eigenvalues of the correlation matrix |
| `min_changes` | 3 | the least number of changes or intervals of a variable |
| `seed` | 20261029 | the seed of the model's draws |

## Plugin and version

`plugin = Plugin(plugin_id="rw", ..., model_sdk=">=3.0,<4.0", macro=(MODEL,))`, the variables are the whole registry, `latents = ("scale_path",)`. The plugin imports only `mst.model_sdk`, `mst.macro.panel`, `mst.macro.registry`, `mst.plugin_sdk` and its own package; the plugin rules forbid importing from `mst.plugins.bvar`.

`estimate` writes to the meta `as_of`, `ids`, `n_global`, `months`, `shocks` (the variable ids in panel order), `nu`, `diagnostics` and `spec`, and to the arrays `panel_y`, `panel_approximate`, `drift` of length $n$, `scale` of size $n\times n$ ($\Sigma$), `weights` (the weights $w_t$ of the months at the last iteration), `start_mean` of length $n$ and `start_sd` of length $n$. `load` restores the model without re-estimating. `simulate` returns a `MacroSimulation`: shadow paths of shape $(R,H+1,n)$, where row $t=0$ is the path's $y_0$, `history` is the same $y_0$, and the group numbers.

In `diagnostics` the version writes $\nu$, the number of iterations and the last change of the likelihood, the ten months with the smallest weight $w_t$ (the model's crisis months), the number of changes of every variable, and the distance of the Higham correction if one was needed.

## Work plan

Files and functions:

- `src/mst/plugins/rw/spec.toml`, `spec.py` — the parameters of the table and their checks: the variables of `drifts` and `breaks` are in the registry, the interval `nu_bounds` lies above 2.
- `src/mst/plugins/rw/estimate.py` — `changes(panel, spec)` (changes without the breaks), `drift(changes, spec)`, `ecme(changes, mu, spec) -> (scale, nu, weights, trace)`, `housing(panel, mu, scale, nu)`, `nearest_correlation(matrix, floor)`.
- `src/mst/plugins/rw/forecast.py` — `order(ids, n_global, causes) -> permutation`, `impact(scale, order)`, `scale_candidates(nu, horizon, count, rng)`, `simulate(model, plan, draws, seed)`, `constraint_law(model, plan, draws, seed)`.
- `src/mst/plugins/rw/model.py`, `plugin.py`, `__init__.py` — the model class under the `MacroModel` contract and the plugin declaration.
- `tests/rw/` — the checks below.
- `docs/reference/platform.md` and `.en.md` — a line on the model in the list of macro models; `docs/plans/bench-backtest.md` — `rw.structural` replaced by `rw.joint`.

Checks for the code:

1. ECME on simulated Student changes with $\nu=5$ and 20 % gaps recovers $\Sigma$ and $\nu$ within two standard errors; on Gaussian data the estimate of $\nu$ hits the upper bound.
2. The log likelihood does not decrease over the ECME iterations.
3. The drift of the variables in `drifts` equals the `drift` of `random_walk` to $10^{-12}$; the others have zero drift.
4. The changes of the `breaks` months do not enter the estimate: moving the splice to another month changes the estimate only through that change.
5. `scale` is symmetric and positive definite; on a matrix with a negative eigenvalue the correction gives a smallest eigenvalue of at least `eig_floor`.
6. Without a scenario the sample mean and covariance of $y_t-y_0$ agree with $t\mu$ and $t\frac{\nu}{\nu-2}\Sigma$; the kurtosis of a monthly change agrees with $3(\nu-2)/(\nu-4)$ for $\nu>4$.
7. The order of the causes does not change the law of the paths without a scenario (Kolmogorov–Smirnov by months for two orders).
8. With the cause `ru.key_rate` and an exact condition on the rate, the conditional mean change of another variable equals the projection formula of the section "Difference from the baseline".
9. A domestic cause does not move global variables; a scenario that fixes a global variable through domestic causes alone is refused with the constraint numbers.
10. With `scale_path` among the causes, a condition on a large move picks candidates with a large $\lambda$: the mean multiplier of the picked paths is above the base one.
11. `constraint_law` agrees with `simulate`: the share of unconditional paths in the box equals the probability of the row within the Monte Carlo error.
12. Interval conditions hold on every path, exact ones within $\tau$.
13. One seed gives bit-identical paths; `load` after `estimate` gives the same paths.
14. No observed path of a variable with `lower` lies below the bound.

Acceptance on the data — the same calculations as for `bvar.sv`:

1. The base forecast on 12 rolling dates of 2021–2026: mean CRPS at most 1.00 of the baseline; the shares of facts below the 5th and above the 95th percentile each between 2 and 10 % over all variables and over the rates separately.
2. The base portfolio calculation is admitted: fewer than 5 % of cells with a band wider than the history.
3. The corpus scenarios are admitted; in the rate-cut scenario with the exchange rate set, the bands of IMOEX and USD/RUB in the second month lie within the historical range.

Stop conditions: if the mean CRPS of the base forecast is above 1.02 of the baseline, the model does not become the default, and the drift rule — the only systematic difference of the centre from the baseline — is checked first. If ECME does not converge in 500 iterations on the working panel, the estimation stops with an error, and the switch to the Gaussian variant ($\nu=\infty$) is decided separately.

## References

Antolín-Díaz J., Petrella I., Rubio-Ramírez J. (2021). Structural scenario analysis with SVARs. Journal of Monetary Economics 117.

Geweke J. (1991). Efficient simulation from the multivariate normal and Student-t distributions subject to linear constraints and the evaluation of constraint probabilities.

Higham N. (2002). Computing the nearest correlation matrix — a problem from finance. IMA Journal of Numerical Analysis 22(3).

Liu C., Rubin D. (1995). ML estimation of the t distribution using EM and its extensions, ECM and ECME. Statistica Sinica 5(1).
