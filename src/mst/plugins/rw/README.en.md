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
y_t=A y_{t-1}+c+\sqrt{\lambda_t}\,Bz_t,\qquad z_t\sim N(0,I_n),\qquad \frac{1}{\lambda_t}\sim\frac{\chi^2_\nu}{\nu},
$$

where $A=\operatorname{diag}(a_1,\dots,a_n)$ is a diagonal mean reversion, $c$ a constant intercept, $\Sigma=BB^\top$ the scale matrix of the changes, $\lambda_t>0$ the common scale multiplier of month $t$, and $\nu>2$ the degrees of freedom. Without a scenario the $z_t$ are independent, and with $\lambda_t$ not varying over $t$ the change of a month has a multivariate Student law with covariance $\frac{\nu}{\nu-2}\Sigma$.

By default $A=I$ and $c=\mu$: the former joint random walk with drift. A variable named in `reversion` gets $a_i<1$ and $c_i=(1-a_i)m_i$, where $m_i$ is the mean level of its panel: the model returns to the mean level. Variables in `drifts` (nominal prices, exchange rates, volumes, IMOEX) keep $A=I$: they trend, and a return to a fixed level would fit the trend rather than a return. The `reversion` list is empty by default: the AR(1) coefficients on the panel are 0.97–0.99, the return moves the centre weakly and narrows the band, and the coverage falls (see "Implementation check").

The multiplier is common to all variables, so in a month with a large $\lambda_t$ every variable moves widely at once. That is the crisis month of the model. The path $\lambda_t$ is either independent over months (`stochastic_scale = false`) or follows a log AR(1):

$$
s_t=\phi s_{t-1}+\sigma\eta_t,\qquad \lambda_t=\exp\!\left(\log\frac{\nu}{\nu-2}-\frac{\sigma^2}{2(1-\phi^2)}+s_t\right),
$$

so $E[\lambda_t]=\frac{\nu}{\nu-2}$: persistence changes the shape of the path, not the unconditional monthly covariance. `stochastic_scale = false` by default: at 12 months the validation CRPS improves by 0.0014 against a threshold of 0.005, so the persistent scale is off.

The shocks $z_{t,j}$ carry the names of variables: the shock `ru.key_rate` is the key rate's own innovation. The matrix $B$ is the Cholesky factorisation of $\Sigma$ in panel order; the base law of the paths does not depend on the order.

The panel places global variables first as a storage convention. The joint law imposes no global/domestic independence: an intervention in a domestic variable can change the conditional distribution of a global one through the estimated correlations. This statistical response is not a causal effect.

The parameters $\theta=(A,c,\Sigma,\nu,\phi,\sigma)$ are estimated once and stored in the version. The level-0 outcome in the sense of the general scheme is

$$
\omega^0=(\theta,u,\boldsymbol\lambda,\mathbf z),\qquad \boldsymbol\lambda=(\lambda_1,\dots,\lambda_H),\qquad \mathbf z=(z_1,\dots,z_H),
$$

where $u$ are the standard normal coordinates of the values of month 0 not yet published on the calculation date (section "Starting value"). The model declares the path of multipliers $\boldsymbol\lambda$ as the hidden coordinate `scale_path` (the `latents` field of the `MacroModel` contract).

## Estimation

### Data

The model reads the panel of `bvar.sv`: months from 2012-01 to the calculation date, only the values published by that date, seasonality removed by the platform, a month without a value left as a gap. Changes are taken between neighbouring months where both values are known. The change of a month named in the `breaks` of the specification is left out for the variables listed there: these are breaks of the series, not market moves. There is one break now — the splice of the MOEX spread indices in 2021-01 for `ru.spread_aa`, `ru.spread_a`, `ru.spread_bbb`.

### Drift

The drift is non-zero only for the variables in the `drifts` list of the specification: nominal prices, exchange rates and volumes that grow with the price level (`ru.cpi`, `ru.usdrub`, `ru.cnyrub`, `ru.output`, `ru.real_wage`, `ru.retail`, `ru.credit`, `ru.housing_price`, `ru.imoex`). For such a variable $\mu_i$ is the mean of its monthly changes over the whole panel. Housing has no adjacent monthly observations, so its drift is the total change over valid observation intervals divided by their total duration in months. For the others $\mu_i=0$: the sample trend of 2012–2026 in rates, yields, spreads, unemployment, Brent and the VIX (+0.60 pp a year for the key rate, for example) is no forecast.

A variable named in `reversion` gets a constant intercept $c_i=(1-a_i)m_i$ instead of a drift: its mean level $m_i$ is the panel mean of the level, $a_i$ the AR(1) coefficient. The drift $\mu_i$ of such a variable does not enter the forecast mean but still centres the changes before the scale is estimated. If a variable is named in both `drifts` and `reversion`, the return wins: the fixed mean level replaces the drift.

The drift is a plain mean taken before the scale is estimated and stays fixed in that estimation. This is a simplification: the maximum likelihood location of a Student law is a weighted mean. The plain mean equals the baseline's drift.

### Mean reversion

The coefficient $a_i$ is the slope of a regression of the level on its previous month with a constant. The mean level $m_i$ is the panel mean of the level, not the estimated constant over $1-a_i$: for a trending series that ratio gives a level far beyond the data, and a coefficient above one has no constant at all. The estimate is two-step: ECME gives the scale and $\nu$ on the changes first, the fit is independent of them.

On the 2012–2026 panel the coefficients are 0.97–0.99 for rates, 0.89–0.91 for spreads, 0.78 for the VIX. At those values the centre moves by 20–30 % of the gap to the mean over 12 months, while the band narrows by 10–25 %. On the rolling grid that lowers coverage: rates and the real block move in cycles, and a walk with no return describes their width better. The `reversion` list is therefore empty by default; the code and the specification keep it switchable.

### Scale and degrees of freedom

$\Sigma$ and $\nu$ are the maximum likelihood estimates of a multivariate Student law on changes with gaps, found by the ECME algorithm (Liu, Rubin, 1995). Each iteration:

1. For month $t$ take the known changes $d_t^{o}$ (the subvector of variables $o_t$). The distance from the centre is $\delta_t=(d_t^{o}-\mu_{o})^\top\Sigma_{oo}^{-1}(d_t^{o}-\mu_{o})$, the weight of the month is $w_t=(\nu+p_t)/(\nu+\delta_t)$, where $p_t$ is the number of known changes.
2. The unknown changes of the month are replaced by their conditional mean given the known ones, giving a full vector $\hat d_t$. Denote by $V_t$ the Gaussian Schur complement $\Sigma_{mm}-\Sigma_{mo}\Sigma_{oo}^{-1}\Sigma_{om}$ for the missing coordinates $m$, padded with zeros in observed rows and columns. In the scale-mixture EM update this is $\mathbb E[(d_m-\hat d_m)(d_m-\hat d_m)^\top/\lambda_t\mid d_o]$, not the conditional covariance of the Student law.
3. The new scale matrix is $\Sigma=\frac{1}{T}\sum_t\big[w_t(\hat d_t-\mu)(\hat d_t-\mu)^\top+V_t\big]$, the sum running over the $T$ months with at least one known change.
4. The new $\nu$ maximises the observed likelihood on the interval `nu_bounds` at the new $\Sigma$. This is a one-dimensional problem solved by Brent's method.

The iterations stop when the log likelihood changes by less than `em_tol`, or after `em_iterations` iterations; in the latter case the estimation stops with an error that names the change reached. The starting values are the sample covariance over pairwise known months, corrected to positive definiteness, and $\nu=10$.

A crisis month gets a small weight $w_t$, so the correlations come mostly from ordinary months, and crises enter the tail of the law through $\nu$. The same mechanism weakens the bias described above: the correlation of the rate with the rouble is no longer set by 9 crisis months.

### Level of the scale

The same mechanism has a price. A small weight of a crisis month means that the estimated $\Sigma$ describes mostly calm months, and the unconditional covariance of the changes of a variable with clustered large moves is below the sample one. On the 2012–2026 panel the difference is large: for the key rate the Student law with $\nu=3.35$ gives a monthly covariance 2.5 times below the sample, for the CPI 1.5 times below; for output and the VIX it is above. The second step (`sample_scale = true`) therefore takes the level and the correlations of the scale from the sample covariance of the changes:

$$
C=\frac{1}{T-1}\sum_{t\in T}(d_t-\bar d)(d_t-\bar d)^\top,\qquad \Sigma=\frac{\nu-2}{\nu}\,C,
$$

so that the monthly covariance of the law, $\frac{\nu}{\nu-2}\Sigma$, equals $C$. The matrix $C$ is computed over pairwise known months with the breaks excluded, and its correlation part is projected onto a positive-definite one when needed (as for the housing price). The window $T$ starts at `scale_start = 2015-01-01`: the rouble floated in November 2014, and the rate-to-market transfer is measured from 2015. Housing has no monthly changes and keeps the interval estimate.

In substance this restores the sample correlations of the joint law. A regression of the monthly IMOEX change on the key-rate change gives −1.34 % per 1 pp instead of −0.46 % for the ECME estimate; BBB +0.61 pp, USD/RUB +2.81 % (see "Implementation checks"). The signs and orders are those of the data; this is a statistical transfer, not a causal effect.

### Persistence of the multiplier

With `stochastic_scale = true` the path of the multiplier is a log AR(1) with estimated $\phi$ and $\sigma$. The series for the estimate is the inverse ECME weight: $w_t$ is proportional to $E[1/\lambda_t\mid\text{data}]$, so $\log(1/w_t)$ estimates the log scale. A regression of that series on its previous month gives $\phi$ and $\sigma$; on the panel of 2026-10-29 they are $\phi=0.63$, $\sigma=0.69$. The estimate is two-step and approximate: $1/w_t$ is a posterior mean shrunk toward one, so $\sigma$ is likely understated. The level of the path is normalised so that $E[\lambda_t]=\frac{\nu}{\nu-2}$: persistence changes the shape of the path, not the unconditional monthly covariance. The switch is off by default: the validation CRPS improves by 0.0014 against a threshold of 0.005.

### Housing price

`ru.housing_price` is known in 32 months out of 177 and has no monthly changes. It does not enter ECME. Its scale and correlations are estimated from the changes between neighbouring observations as those of a Gaussian random walk with gaps. Let $s_0<\dots<s_m$ be the months of observation, $k_j=s_j-s_{j-1}$, $D_j$ the change over the interval. Then

$$
\sigma_h^2=\frac{1}{m}\sum_{j=1}^{m}\frac{(D_j-k_j\mu_h)^2}{k_j},\qquad
\hat\rho_{hi}=\frac{1}{q\,\sigma_h\sigma_i}\sum_{l=1}^{q}\frac{(D^h_l-k_l\mu_h)(D^i_l-k_l\mu_i)}{k_l},
$$

where in the second formula the intervals are those shared by the housing price and variable $i$, $D^i_l$ is the change of variable $i$ over the same interval, and $\sigma_i^2=\frac{\nu}{\nu-2}\Sigma_{ii}$ is the variance of its monthly change. The row and column of the housing price in $\Sigma$ are $\frac{\nu-2}{\nu}\,\sigma_h\sigma_i\hat\rho_{hi}$, the covariance carried to the scale of the Student law. If the resulting matrix is not positive definite, its correlation part is replaced by the nearest correlation matrix in the Frobenius norm with eigenvalues at least `eig_floor` (Higham, 2002); the diagonal stays. The housing estimates are an approximation: they rest on 31 intervals and on a Gaussian law.

### Parameter uncertainty

The uncertainty of $\theta$ is not counted: the version stores one estimate. With 177 months it would widen the 12-month spread of USD/RUB by about 3 %: a standard error of the drift of 0.4 % a month gives 5 % over a year against 19 % of noise, 19.7 % in quadrature. Hence the $R=(\theta,\psi)$ of the general scheme holds here the parameters and the realized interventions of the scenario.

## Starting value

Series are published with different lags, so in month 0 — the last month in which at least one variable is known — some variables are unknown. The model carries an unknown component to month 0 by its own independent steps from the last known value: $k$ steps give the law $N\!\left(m_i+a_i^k(y_i-m_i),\ \frac{1-a_i^{2k}}{1-a_i^2}\,\frac{\nu}{\nu-2}\Sigma_{ii}\right)$, written as $m_0+L_0u$ with a diagonal $L_0$; at $a_i=1$ this is the former $N\!\left(k\mu_i,\,k\,\frac{\nu}{\nu-2}\Sigma_{ii}\right)$, and a known value has no spread. The baseline does the same. Conditioning on the known moves of other variables (a Kalman filter) is deferred: on the panel to 2026-09 the real wage lags by two months, unemployment, retail, credit and the mortgage rate by one, the rest are known in month 0.

## Interventions and observations

A scenario under SDK 4.0 is $s=(N,I,\mathcal A)$: decisions (interventions) and requirements on the
outcome (observations). The file holds no model versions, horizon or number of paths — the
calculation request names them.

An intervention fixes a quantity $k$ in months $T$: `set` puts the value $a_t$, `shift` adds $o_t$
to the model output. A scenario does not name the hidden coordinate `scale_path`: observations
reweight its candidates.

The model executes the interventions itself, so it declares a structural form (`StructuralForm`).
The default form would replace only the cell of a set quantity and leave the other variables of
that month as drawn; the design of `rw.joint` carries the decision to them by the joint law. The
Cholesky order is therefore built over the set cells: all set variables come first in registry
order, followed by all other variables, without a global/domestic restriction. Let $P$ be the matrix of this permutation and $C$ the lower-triangular Cholesky factor of
$P\Sigma P^\top$; then $B=P^\top CP$. In substance: the set variables move by the given values,
and the others respond in the same month by the conditional law of changes given them,
$\mu_o+\Sigma_{oc}\Sigma_{cc}^{-1}(\Delta c_t-\mu_c)$, with residual covariance
$\lambda_t(\Sigma_{oo}-\Sigma_{oc}\Sigma_{cc}^{-1}\Sigma_{co})$. The strength of the response
comes from the joint law and says nothing about causality.

Different scenarios give different $B$, but always $BB^\top=\Sigma$, so the base law $P$ does not
depend on the order.

An observation is a condition on a path, a mean or a probability. After the interventions are
substituted the path is linear in the free innovations, $W\mathbf y=c^\psi+Z^\psi\varepsilon_{\text{free}}$,
and the free innovations are drawn from $\mathcal N(0,I)$ truncated to the observation bounds (a
Gibbs sampler, 50 sweeps), at the chosen path of multipliers. A row wholly determined by the set
values leaves the system; its value outside the bounds means the scenario contradicts the model. A
row linearly dependent on the rows already fixed (threshold $10^{-9}$) is refused with its number
(`InfeasibleScenario`).

The path of multipliers is drawn as the volatility path of `bvar.sv` is. Under observations the
model draws `LATENT_CANDIDATES` candidates of $\boldsymbol\lambda$ from the base law, weighs each
by the probability of the box under it (`log_condition_weight`) and picks one by the weights
(`pick_latent`). Without observations the path of multipliers is drawn from the base law without
weights.

`constraint_law` returns the law of $W\mathbf y$ given $R=(\theta,\psi)$ **after the
interventions**, in rows of groups, one row per candidate of multipliers with its weight, as
`bvar.sv` does; the platform turns it into $P(S\mid R)$ and the price. The model reports the size
of the free innovations the observations need, in standard deviations (`diagnostics`); as with the
other models, this is a flag of the report, not a refusal.

### Central bank decisions and the market's reaction

The model does not know how the market reacts to a central bank decision. The regression of the
rouble on the rate describes a central bank that answers a rouble collapse and gives the wrong sign
for an unscheduled cut. A scenario that sets a central bank decision therefore also sets the
market's reaction: the path of the exchange rate and, where something is known about it, of OFZ
yields. The numbers are marked by the general rules — by the event, by a corpus source or by the
word "assumption" — and all such variables are set as interventions. The model carries the other
variables by the joint law.

## Forecast without a scenario

$$
y_t=A^t y_0+\left(I-A^t\right)(I-A)^{-1}c+B\sum_{s=1}^{t}A^{t-s}\sqrt{\lambda_s}\,z_s,\qquad t=1,\dots,H.
$$

With $y_0$ known, the mean of $y_t$ is $A^t y_0+\left(I-A^t\right)(I-A)^{-1}c$ (for a `reversion` variable, $m_i+a_i^t(y_{0,i}-m_i)$) and its covariance $\frac{\nu}{\nu-2}\sum_{s=0}^{t-1}A^s\Sigma A^s$ (for $A=I$ this is $t\frac{\nu}{\nu-2}\Sigma$; for a diagonal $A$, $(1-a_i^{2t})/(1-a_i^2)$ per variable). The tails are the heavier the shorter the horizon, since a sum of $t$ independent Student changes approaches the normal law as $t$ grows; a persistent multiplier, on the contrary, makes the distribution of the sum heavier at the same spread.

Rates, unemployment and the BBB spread have a lower bound in the registry. As with `bvar.sv`, the model draws an unbounded shadow value, the conditions and their price are computed on the shadow, and the platform reads the observed value $\max(s_t,\text{lower})$ (`mst.model_sdk.bounds.observed`).

## Scenario

A scenario reaches the model as a plan: the horizon $H$, observations $\mathbf a\le W\mathbf y\le\mathbf b$ on the path $\mathbf y=(y_1,\dots,y_H)$, the realized interventions (one row per group of paths) and `inner`. The platform widens an exact observation beforehand to an interval of half-width $\tau$.

Given the path of multipliers $\boldsymbol\lambda$ and the substituted interventions the path is linear in the free innovations $\varepsilon=(u,\mathbf z)$:

$$
W\mathbf y=c^\psi+Z^\psi(\boldsymbol\lambda)\,\varepsilon,
$$

where $c^\psi$ is the value of $W\mathbf y$ at $\varepsilon=0$ with the interventions executed. A term $w\,y_{i,t}$ contributes $w(L_0)_{i\cdot}$ to the columns of $u$ and $w\sqrt{\lambda_s}B_{ij}$ to the column of $z_{s,j}$ for every $s\le t$; the set cells go into $c^\psi$, and their shocks are not drawn. Given $\boldsymbol\lambda$ the problem is therefore that of a Gaussian model, and the common machinery of `mst.model_sdk.conditional` solves it (Antolín-Díaz, Petrella, Rubio-Ramírez, 2021): exact rows are fixed, interval rows are drawn by a Gibbs sampler (Geweke, 1991), linearly dependent rows are refused.

## Difference from the baseline

The `random_walk` baseline continues every variable on its own by Gaussian steps with the sample drift and spread. `rw.joint` differs from it in three places: the drift of rates, spreads, unemployment, Brent and the VIX is zero; the tails are heavy with a common crisis scale; the law is joint. The difference in CRPS between the model and the baseline on the episodes therefore measures what these choices add. If the scenario sets only part of the variables, the conditional law of the change of the other variables in month $t$ given $\lambda_t$ is

$$
N\big(\mu_y+\Sigma_{yc}\Sigma_{cc}^{-1}(\Delta c_t-\mu_c),\ \lambda_t(\Sigma_{yy}-\Sigma_{yc}\Sigma_{cc}^{-1}\Sigma_{cy})\big),
$$

the same linear projection as the conditional baseline `conditional_walk`, with coefficients in which crisis months weigh less and with a heavy-tailed residual.

## Assumptions

By default there is no mean reversion: the spread grows as $\sqrt t$, rates and spreads included. The code supports a diagonal AR(1) (`reversion`), but on this panel it does not help (section "Mean reversion"). The links between variables are only contemporaneous: the model does not carry the delayed pass-through of the exchange rate into prices, which comes from the scenario or a satellite. The crisis scale is common to all variables: a crisis in the rouble alone and a crisis in everything differ only by the innovations. The covariance of the changes is the sample covariance from 2015, so crises also set the correlations: this is a statistical transfer, not a causal effect. The drift of prices and exchange rates comes from the history of 2012–2026. Conditioning uses the full joint law, including statistical links from domestic to global variables. Parameter uncertainty is not counted.

## Parameters

| Parameter | Value | Meaning |
|---|---|---|
| `start` | 2012-01-01 | the first month of the panel, as for `bvar.sv` |
| `drifts` | the nine variables above | the variables with a non-zero drift |
| `reversion` | `[]` | the variables with mean reversion; empty by default |
| `stochastic_scale` | `false` | the path of the common multiplier is a log AR(1) instead of independent months |
| `sample_scale` | `true` | the level and correlations of the scale are the sample covariance of the changes |
| `scale_start` | 2015-01-01 | the first month of the sample covariance: the floating rouble |
| `breaks` | 2021-01: three spreads | changes left out at breaks of the series |
| `nu_bounds` | $[2.5,\ 60]$ | the search interval of $\nu$ |
| `em_tol` | $10^{-8}$ | the threshold on the change of the log likelihood |
| `em_iterations` | 500 | the largest number of ECME iterations |
| `eig_floor` | $10^{-8}$ | the lower bound of the eigenvalues of the correlation matrix |
| `min_changes` | 3 | the least number of changes or intervals of a variable |
| `seed` | 20261029 | the seed of the model's draws |

The estimated parameters are written to the version meta. `diagnostics.mean_reversion` is a row per panel variable: the coefficient $a_i$, the mean level $m_i$ and the source (`ar1`, `walk`, `short`). `diagnostics.scale_path` holds $\phi$, $\sigma$, the number of months and the regression constant. `diagnostics.scale_correction` holds the mode (`sample` or `ecme`) and the sample and ECME standard deviations of the changes of every variable. The version arrays carry `persistence` ($a_i$) and `intercept` ($c_i$).

## Plugin and version

`plugin = Plugin(plugin_id="rw", ..., model_sdk=">=4.0,<5.0", macro=(MODEL,))`, the variables are the whole registry, `latents = ("scale_path",)`. The model executes the interventions itself, so `structural_form` returns a `StructuralForm` with registry order as its base order (all set variables move first in each intervened month) and `time_recursion = True`; the platform does not apply the interventions again. The plugin imports only `mst.model_sdk`, `mst.macro.panel`, `mst.macro.registry`, `mst.plugin_sdk` and its own package; the plugin rules forbid importing from `mst.plugins.bvar`.

`estimate` writes to the meta `as_of`, `ids`, `n_global`, `months`, `shocks` (the variable ids in panel order), `nu`, `diagnostics`, `spec` and `structural_form`, and to the arrays `panel_y`, `panel_approximate`, `drift` of length $n$, `scale` of size $n\times n$ ($\Sigma$), `weights` (the weights $w_t$ of the months at the last iteration), `start_mean` of length $n$, `start_sd` of length $n$, `persistence` ($a_i$) and `intercept` ($c_i$). `load` restores the model without re-estimating; a version without `persistence` and `intercept` (estimated before 0.2.0) reads as the former walk: $a_i=1$, $c_i=\mu_i$, $\phi=\sigma=0$. `simulate` returns a `MacroSimulation`: shadow paths of shape $(R,H+1,n)$ with the interventions executed, where row $t=0$ is the path's $y_0$, `history` is the same $y_0$, and the group numbers.

In `diagnostics` the version writes $\nu$, the number of iterations and the last change of the likelihood, the ten months with the smallest weight $w_t$ (the model's crisis months), the number of changes of every variable, the distance of the Higham correction if one was needed, and the three blocks above (`mean_reversion`, `scale_path`, `scale_correction`).

## Work plan

Files and functions:

- `src/mst/plugins/rw/spec.toml`, `spec.py` — the parameters of the table and their checks: the variables of `drifts` and `reversion` are in the registry, the interval `nu_bounds` lies above 2, `scale_start` is the first of a month.
- `src/mst/plugins/rw/estimate.py` — `changes(panel, spec)` (changes without the breaks), `drift(changes, ids, spec)`, `ecme(changes, mu, spec) -> (scale, nu, weights, trace)`, `mean_reversion(panel, spec) -> (a, m, report)`, `scale_path_parameters(weights) -> {persistence, volatility}`, `sample_covariance(panel, data, spec)`, `housing(panel, mu, scale, nu, spec)`, `nearest_correlation(matrix, floor)`.
- `src/mst/plugins/rw/forecast.py` — `transfer(scale, interventions) -> (mean, loadings)` (the conditional projection of a month under the set cells), `scale_candidates(nu, horizon, count, rng, phi, sigma)`, `simulate(model, plan, draws, seed)`, `constraint_law(model, plan, draws, seed)`.
- `src/mst/plugins/rw/model.py`, `plugin.py`, `__init__.py` — the model class under the `MacroModel` contract, `structural_form` and the plugin declaration; version 0.2.0.
- `tests/rw/` — the checks below.
- `docs/reference/platform.md` and `.en.md` — a line on the model in the list of macro models; `docs/plans/bench-backtest.md` — `rw.structural` replaced by `rw.joint`.

Checks for the code:

1. ECME on simulated Student changes with $\nu=5$ and 20 % gaps recovers $\Sigma$ and $\nu$ within two standard errors; on Gaussian data the estimate of $\nu$ hits the upper bound.
2. The log likelihood does not decrease over the ECME iterations.
3. The drift of the variables in `drifts` equals the `drift` of `random_walk` to $10^{-12}$; the others have zero drift.
4. The changes of the `breaks` months do not enter the estimate: moving the splice to another month changes the estimate only through that change.
5. `scale` is symmetric and positive definite; on a matrix with a negative eigenvalue the correction gives a smallest eigenvalue of at least `eig_floor`.
6. Without a scenario the sample mean and covariance of $y_t-y_0$ agree with $t\mu$ and $t\frac{\nu}{\nu-2}\Sigma$; the kurtosis of a monthly change agrees with $3(\nu-2)/(\nu-4)$ for $\nu>4$.
7. The panel order does not change the law of the paths without a scenario (Kolmogorov–Smirnov by months for two orders); a set changes only the response of the other variables of the same month.
8. With `ru.key_rate` set in month $t$, the conditional mean change of another variable in the same month equals the projection formula of the section "Difference from the baseline".
9. A set value stands on every path of its group and only in its month; the other variables of the same month respond by the projection instead of staying untouched.
10. Under an observation on a large move, candidates of the multipliers with a large $\lambda$ get more weight: the mean multiplier of the picked paths is above the base one.
11. `constraint_law` agrees with `simulate`: the share of paths in the box equals the probability of the row within the Monte Carlo error.
12. Interval conditions hold on every path, exact ones within $\tau$.
13. One seed gives bit-identical paths; `load` after `estimate` gives the same paths.
14. No observed path of a variable with `lower` lies below the bound.
15. `mean_reversion` on a simulated AR(1) recovers the slope and the mean level, and a variable outside `reversion` stays a walk.
16. `sample_covariance` uses the `scale_start` window and gives the sample covariance of the changes in the Student scale; `estimate` with `sample_scale = true` reproduces it up to the projection.
17. `scale_path_parameters` recovers $\phi$ and $\sigma$ of the log AR(1); `scale_candidates` with $\sigma>0$ keeps $E[\lambda]=\nu/(\nu-2)$ and correlates neighbouring months; at $a<1$ the base law converges to the mean, and the same-month response to a set is computed from $a y+c$, not from the level.

Acceptance on the data — the same calculations as for `bvar.sv`:

1. The base forecast on 12 rolling dates of 2021–2026: mean CRPS at most 1.00 of the baseline; the shares of facts below the 5th and above the 95th percentile each between 2 and 10 % over all variables and over the rates separately.
2. The base portfolio calculation is admitted: fewer than 5 % of cells with a band wider than the history.
3. The corpus scenarios are admitted; in the rate-cut scenario, where both the rate and the exchange rate are set, the bands of IMOEX and USD/RUB in the second month lie within the historical range.

Stop conditions: if the mean CRPS of the base forecast is above 1.02 of the baseline, the model does not become the default, and the drift rule — the only systematic difference of the centre from the baseline — is checked first. If ECME does not converge in 500 iterations on the working panel, the estimation stops with an error, and the switch to the Gaussian variant ($\nu=\infty$) is decided separately.

## Implementation checks

On the panel known on 2026-10-29 (177 months, 23 variables), ECME converges in 81 iterations to $\nu=3.3525$; the last likelihood change is $9.17\cdot10^{-9}$. Housing uses 31 intervals. The estimated version of the final specification (`sample_scale = true`, `stochastic_scale = false`, `reversion = []`) is

**`2e459db83e0c5a8dba04299fa1ed9716f680d8e5df1580cb40f09b68c6c94596`**

### Rolling grid

23 dates 2021-01…2026-07, step 3 months, 1,000 paths, 12 forecast months; another 19 dates 2016-04…2020-10 are the tuning half. The plan splits the grid at `validation_from = 2021-01-01`; a variant is decided on the tuning half.

| half | dates | cells | CRPS / baseline | 95 % interval | coverage | below P5 | above P95 | shares 5/50/95 |
|---|---:|---:|---:|---|---:|---:|---:|---|
| tuning | 19 | 4993 | 0.9371 | 0.926–0.959 | 0.881 | 0.068 | 0.050 | 0.07 0.51 0.37 0.05 |
| validation | 23 | 5779 | 0.9916 | 0.967–1.013 | 0.809 | 0.059 | 0.129 | 0.06 0.41 0.40 0.13 |
| all dates | 42 | 10772 | 0.9774 | 0.956–0.998 | 0.842 | 0.063 | 0.092 | 0.06 0.46 0.39 0.09 |

| horizon | tuning | validation | all dates |
|---:|---:|---:|---:|
| 1 | 0.900 | 0.871 | 0.884 |
| 3 | 0.897 | 0.837 | 0.864 |
| 6 | 0.882 | 0.785 | 0.830 |
| 12 | 0.858 | 0.766 | 0.811 |

12-month coverage by variable (the share of facts inside P5…P95):

| variable | tuning | validation |
|---|---:|---:|
| `ru.ofz_1y` | 0.947 | 0.450 |
| `ru.credit` | 0.842 | 0.550 |
| `ru.imoex` | 0.947 | 0.550 |
| `ru.key_rate` | 1.000 | 0.600 |
| `ru.ruonia` | 0.947 | 0.600 |
| `ru.ofz_5y` | 1.000 | 0.600 |
| `gl.fed_rate` | 0.158 | 0.650 |
| `ru.ofz_10y` | 1.000 | 0.700 |
| `ru.cpi` | 0.368 | 0.700 |
| `ru.spread_a` | 1.000 | 0.737 |
| `gl.ust_10y` | 0.684 | 0.750 |
| `ru.usdrub` | 0.947 | 0.800 |
| `ru.cnyrub` | 0.947 | 0.800 |
| `ru.unemployment` | 0.737 | 0.800 |
| `ru.mortgage_rate` | 0.895 | 0.850 |
| `ru.spread_aa` | 1.000 | 0.895 |
| `ru.spread_bbb` | 1.000 | 0.895 |
| `ru.real_wage` | 0.895 | 0.947 |
| `gl.brent` | 0.895 | 1.000 |
| `gl.vix` | 1.000 | 1.000 |
| `ru.output` | 0.842 | 1.000 |
| `ru.retail` | 0.895 | 1.000 |

The target "every variable between 0.75 and 0.99" is not met: on the validation half 10 variables are below 0.75 and 4 at 1.00. The cause is regime reversals, not the shape: February 2022 (rate 9.5→20 %), 2023–2024 (7.5→21 %) and 2025–2026 (the cuts) fall inside 12-month windows of the grid. On 23 dates that include the 2022 crash, a walk with no trend inertia does not give it; the numbers above are the limit of these data, not a tuned result. Four dates of the tuning half (2015-04…2016-01) are refused: the housing price has fewer than three observation intervals; another five are skipped as too little history.

### Variants

Each variant re-estimates every date of the grid. "Rate", "IMOEX", "BBB" are the 12-month change standard deviations of the version of 2026-10-29.

| variant | tuning: coverage / CRPS | validation: coverage / CRPS | rate | IMOEX | BBB |
|---|---|---|---:|---:|---:|
| unchanged | 0.862 / 0.9204 | 0.754 / 1.0122 | 2.27 | 19.94 | 4.13 |
| A: rate and spread reversion | 0.823 / 0.9382 | 0.728 / 1.0409 | 1.94 | 19.94 | 2.76 |
| B: persistent multiplier | 0.864 / 0.9235 | 0.768 / 1.0107 | 2.27 | 20.04 | 4.14 |
| C: real-block reversion | 0.786 / 0.9867 | 0.693 / 1.1115 | 2.27 | 19.94 | 4.13 |
| S: sample covariance | 0.881 / 0.9371 | **0.809 / 0.9916** | **4.10** | **18.07** | **4.94** |
| S+B | 0.882 / 0.9412 | 0.824 / 0.9902 | 4.11 | 18.38 | 4.96 |
| S+A+B+C | 0.774 / 1.0170 | 0.733 / 1.1160 | 3.52 | 18.38 | 3.30 |

A and C are off: the AR(1) coefficients are 0.97–0.99, the return barely moves the centre and narrows the band. B is off: the validation CRPS beats S by 0.0014 against a threshold of 0.005. S is adopted.

### Width and rate transfer

12-month change standard deviation on 2026-10-29 against the realised one in the store:

| series | model | 2015+ | 2021+ | calm 2023+ |
|---|---:|---:|---:|---:|
| `ru.key_rate` | 4.10 pp | 4.43 | 6.01 | 5.70 |
| `ru.imoex` | 18.07 % | 23.31 | 29.24 | 17.84 |
| `ru.spread_bbb` | 4.94 pp | 3.39 | 4.55 | 5.15 |

For the index, the exchange rate and the spread the calm-window annual variance equals twelve monthly ones: the walk describes them correctly, and the 29.2 % of IMOEX comes from the windows with the 2022 crash, not from a defect of the shape. The rate moves in cycles of 8–13 pp, so its annual variance exceeds twelve monthly ones; the model gives 4.10 on the full window against the realised 4.43 (2015+) and does not reach 6.01 (2021+) without trend inertia. The key rate at a 14 % start over 12 months from 2026-09: before the change P5/P50/P95 = 10.35 / 14.01 / 17.63, after — 7.34 / 14.02 / 20.63.

The same-month transfer of a rate decision to the market (the full conditional form, the key rate set 5 pp below 14 %):

| series | slope per 1 pp | response to −5 pp |
|---|---:|---:|
| `ru.imoex` | −1.34 % | +6.94 % |
| `ru.spread_bbb` | +0.61 pp | −3.07 pp |
| `ru.usdrub` | +2.81 % | −13.1 % |

The data regression gives −1.56 % per 1 pp for IMOEX and +0.63 pp for BBB; the signs and orders are preserved. This is the statistical transfer of the joint law, not a causal effect: a scenario that sets the central bank's decision also sets the market's reaction.

### What level 0 gives

Version `2e459db8…` on 2026-10-29, 12 months from 2026-09, 8,000 paths, in own units (for `log100` variables the level is converted back; for the CPI the seasonally adjusted panel index is shown):

| variable | 2026-09 | P5 | P50 | P95 | sd | unit |
|---|---:|---:|---:|---:|---:|---|
| `ru.key_rate` | 14.00 | 7.34 | 14.01 | 20.63 | 4.10 | % |
| `ru.imoex` | 2271.54 | 1753.18 | 2338.97 | 3142.15 | 444.49 | points |
| `ru.usdrub` | 85.11 | 66.15 | 91.04 | 123.42 | 18.62 | RUB per USD |
| `gl.brent` | 114.16 | 57.94 | 113.94 | 224.25 | 61.65 | USD per barrel |
| `ru.spread_bbb` | 11.55 | 3.37 | 11.61 | 19.53 | 4.94 | pp |
| `ru.ofz_1y` | 13.73 | 8.91 | 13.71 | 18.46 | 2.95 | % |
| `ru.cpi` | 276.73 | 285.51 | 296.63 | 308.08 | 7.04 | index |

12-month change in percent for the `log100` variables: IMOEX P5/P50/P95 = −22.8 / +3.0 / +38.3; USD/RUB = −22.3 / +7.0 / +45.0; Brent = −49.3 / −0.2 / +96.4; CPI = +3.2 / +7.2 / +11.3.

`tests/rw`: 60 passed. The former form (all switches off) reproduces the previous grid on the same data: the validation half gives 0.7536 coverage and CRPS 1.0122, as before the change — S only changes the covariance estimate.

## References

Antolín-Díaz J., Petrella I., Rubio-Ramírez J. (2021). Structural scenario analysis with SVARs. Journal of Monetary Economics 117.

Geweke J. (1991). Efficient simulation from the multivariate normal and Student-t distributions subject to linear constraints and the evaluation of constraint probabilities.

Higham N. (2002). Computing the nearest correlation matrix — a problem from finance. IMA Journal of Numerical Analysis 22(3).

Liu C., Rubin D. (1995). ML estimation of the t distribution using EM and its extensions, ECM and ECME. Statistica Sinica 5(1).
