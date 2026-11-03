# Specification of the macro model `msvar.markov`

This description comes from the design: the general scheme of the system (`docs/concept/formalization.md`), the macro model repair plan (`plans/macro-models.md`), the plugin READMEs, the `mst.model_sdk` contracts and the macro variable registry. The plugin code was not read while writing it, so a mismatch between the code and this text is a finding of the check, not a reason to edit the text. Where the sources are silent or disagree, the text says "not set by the design" and points to "Open questions".

## 1. Purpose

The system answers how the target measures of an object are distributed if a given event $S$ has happened. Quantities are split into levels $l=0,\dots,L$; level 0 holds the macro variables of the economy. A level-0 macro model is a triple $(\Omega^0,P^0,G^0)$: a probability measure on a space of outcomes and a map from outcomes to the trajectory $x^0$. The scheme requires of a macro model only the law of $x^0$ given the available information.

`msvar.markov` is the second level-0 macro model beside `bvar.sv`: a vector autoregression with Markov regime switching (Hamilton, 1989; Sims, Waggoner, Zha, 2008). The regime lets a scenario state "the economy enters a crisis" as a statement about a latent coordinate rather than only through variable paths; `bvar.sv` has only stochastic volatility for this.

The model answers two questions:

- unconditional forecast: the law of the path of 23 macro variables over a horizon of $H$ months under the base measure $P$;
- conditional forecast: the law of the same path under the scenario overlay $P_s$ (the measure closest to $P$ in Kullback–Leibler divergence among measures under which the event happened), and the scenario price $\mathrm{dist}(s)=\mathrm{KL}(P_s\Vert P)$.

Consumers of the output: satellite models of levels $l\ge1$ read the macro paths (`UpperPaths.macro`) and know nothing of regimes; position models read macro variables in original units; `mst.scenario.build` turns the model's constraint law into a price; `mst.metrics` checks the forecast on the date grid and on episodes.

## 2. Data

The variables are the 23 series of the registry `src/mst/macro/registry.toml`, shared with `bvar.sv`. Let $X_{i,t}$ be the value of variable $i$ in month $t$ in original units, $y_{i,t}=g_i(X_{i,t})$ its value in model scale, and $y_t\in\mathbb R^n$, $n=23$, the vector of the month. The transform $g_i$ is $g_i(X)=X$ for rates and spreads (`identity`) and $g_i(X)=100\ln X$ for prices, exchange rates, indices and volumes (`log100`).

| Id | Variable | Unit | $g_i$ | Lower bound `lower` |
|---|---|---|---|---|
| `gl.brent` | Brent oil | USD per barrel | $100\ln X$ | none |
| `gl.fed_rate` | Fed rate | % | $X$ | 0 |
| `gl.ust_10y` | 10-year UST yield | % | $X$ | 0 |
| `gl.vix` | VIX | points | $100\ln X$ | none |
| `ru.key_rate` | key rate | % | $X$ | 0 |
| `ru.ruonia` | RUONIA | % | $X$ | 0 |
| `ru.ofz_1y`, `ru.ofz_5y`, `ru.ofz_10y` | OFZ yields 1, 5, 10 years | % | $X$ | 0 |
| `ru.spread_aa`, `ru.spread_a` | AA and A spreads to the G-curve | pp | $X$ | none, `negative = true` |
| `ru.spread_bbb` | BBB spread | pp | $X$ | 0 (see open questions) |
| `ru.usdrub`, `ru.cnyrub` | official USD and CNY rates | RUB | $100\ln X$ | none |
| `ru.cpi` | CPI | index | $100\ln X$, seasonally adjusted | none |
| `ru.output` | output of basic industries | index | $100\ln X$, seasonally adjusted | none |
| `ru.unemployment` | unemployment | % | $X$, seasonally adjusted | 0 |
| `ru.real_wage` | real wage | index | $100\ln X$, seasonally adjusted | none |
| `ru.retail` | retail turnover | index | $100\ln X$, seasonally adjusted | none |
| `ru.credit` | credit to firms and households | mln RUB | $100\ln X$ | none |
| `ru.mortgage_rate` | mortgage rate | % | $X$ | 0 |
| `ru.housing_price` | housing price | RUB per m² | $100\ln X$ | none |
| `ru.imoex` | MOEX index | points | $100\ln X$ | none |

The prefix is the block: `gl.` global, `ru.` Russian. The platform removes seasonality from five series when building the panel: it subtracts a constant calendar profile estimated from the mean growth of each calendar month on data known at the calculation date.

Frequency is monthly. Month $t=0$ is the last month for which at least one variable is known at the calculation date; the horizon is $t=1,\dots,H$, $H\le360$. Series keep each value with its publication date, and the panel at the calculation date sees only the versions published by then. On the first load of a revised series the publication date is estimated as the month end plus the source's `lag_days`, and the row is marked `approximate`. A month with no value (ragged publication edge, quarterly housing price, a series that starts late) stays missing, and the model draws it (section 4).

Sample start: the store's series start in 2012-01, and `bvar.sv` starts estimation at $T_0$ = January 2012; for `msvar.markov` the sample start is not set by the design.

## 3. Model

Let $S=2$ be the number of regimes and $\zeta_t\in\lbrace1,2\rbrace$ the regime of month $t$; regime 1 is calm (`regime.calm`), regime 2 is stress (`regime.stress`). Let $p$ be the number of lags. The model is

$$
y_t=\mu(\zeta_t)+\sum_{j=1}^{p}A_j(\zeta_t)\,y_{t-j}+B(\zeta_t)\,z_t,\qquad z_t\sim N(0,I_n)\ \text{independent over }t,
$$

where $\mu(k)\in\mathbb R^n$ is the intercept of regime $k$, $A_j(k)$ are $n\times n$ matrices, $B(k)$ is the $n\times n$ impact matrix, and $\Sigma(k)=B(k)B(k)^\top$ is the covariance of the errors $u_t=B(\zeta_t)z_t$ in regime $k$. The regime is a homogeneous Markov chain:

$$
\Pr(\zeta_t=l\mid\zeta_{t-1}=k)=\Pi_{kl},\qquad \sum_{l=1}^{S}\Pi_{kl}=1.
$$

The parameter `switching` sets what switches:

- `"mean_covariance"` (the design value): $A_j(k)=A_j$ for all $k$; $\mu(k)$ and $\Sigma(k)$ switch. Reason: with 23 variables and three lags there are about 70 coefficients per equation, and the stress regime, a minority of 174 months, cannot determine a second set of lag matrices (in version `ecaa…` its spectral radius exceeds 1.18 in every draw).
- `"all"`: $\mu$, all $A_j$ and $\Sigma$ switch. This is the previous behaviour, kept for comparison.

The companion matrix of regime $k$ is

$$
F(k)=\begin{pmatrix}A_1(k)&A_2(k)&\cdots&A_{p-1}(k)&A_p(k)\\ I_n&0&\cdots&0&0\\ \vdots&&\ddots&&\vdots\\ 0&0&\cdots&I_n&0\end{pmatrix},
$$

and $\rho(k)$ is its spectral radius.

The parameters are $\theta=(\mu(k),A_j(k),B(k),\Pi)_{k\le S,\,j\le p}$. The level-0 outcome in the general scheme is $\omega^0=(\theta,\zeta,\mathbf z)$, where $\zeta=(\zeta_1,\dots,\zeta_H)$ is the regime path on the horizon and $\mathbf z=(z_1^\top,\dots,z_H^\top)^\top$ all horizon shocks. The probability of a regime path given $\theta$ and the regime of the last history month $\zeta_0$ is

$$
\pi_\theta(\zeta)=\Pi_{\zeta_0\zeta_1}\prod_{t=2}^{H}\Pi_{\zeta_{t-1}\zeta_t}.
$$

The design does not name the law of $\zeta_0$ (open questions).

Block exogeneity (Russian variables do not affect global ones) is written into `bvar.sv` as zeros in $A_j$ and a block structure of $B$; for `msvar.markov` the design does not mention it (open questions).

Lower bounds. For a variable with `lower` the model simulates a shadow value $s_{i,t}$ by the equation above, and the observed value in model scale is

$$
x_{i,t}=\max(s_{i,t},\text{lower}_i).
$$

Lags, impacts, conditions and the price use the shadow, so the gaussian linear machinery is unchanged; after a fall below the bound, a later rise starts from the shadow.

## 4. Estimation

Estimation is Bayesian, by a Gibbs sampler. The posterior sample is $M$ parameter versions $\theta^{(r)}$ with the historical regime path; the number of draws, burn-in and thinning (`sampler.draws`, `thin`) are not set by the design for `msvar.markov`. Estimation is deterministic given the seed written in the specification.

Assumption about coefficients before data (Minnesota; Litterman, 1986; Doan, Litterman, Sims, 1984): the coefficient $A_j[i,i']$ is normal with mean $\delta_i$ for $j=1$, $i'=i$ and zero otherwise, and standard deviation

$$
\operatorname{sd}\big(A_j[i,i']\big)=\frac{\lambda}{j}\cdot\frac{\sigma_i}{\sigma_{i'}}.
$$

Here $\lambda$ is the overall tightness (`tightness`), $\sigma_i$ the residual scale of variable $i$, $\delta_i=1$ for a nonstationary variable and $\delta_i=0$ for a stationary one. The $1/j$ form, the factor for other variables' lags, how $\sigma_i$ is obtained and the list of stationary variables are not set by the design for `msvar.markov` (in `bvar.sv` `gl.vix` and the three spreads are stationary). The intercept gets a diffuse assumption (not set by the design).

Dummy observations (Sims, Zha, 1998; Giannone, Lenza, Primiceri, 2015) are added to the regression because the 2026-10-30 audit found that, with regressors in levels, the Minnesota weight is $10^{-5}$ to $10^{-2}$ times its nominal value, while these dummies act on levels and pull the system toward unit roots and away from explosive ones. Let $\bar y_0$ be the mean of the first $p$ observations of the sample, $\phi_{\mathrm{sc}}$ the sum-of-coefficients weight (`mu`) and $\phi_{\mathrm{io}}$ the initial-observation weight (`delta`). For each variable $i$ the sum-of-coefficients dummy adds the row

$$
y^{\ast}=\frac{\bar y_{0,i}}{\phi_{\mathrm{sc}}}e_i^\top,\qquad x^{\ast}=\Big(0,\ \underbrace{\tfrac{\bar y_{0,i}}{\phi_{\mathrm{sc}}}e_i^\top,\dots,\tfrac{\bar y_{0,i}}{\phi_{\mathrm{sc}}}e_i^\top}_{p}\Big),
$$

and the initial observation adds one row

$$
y^{\ast}=\frac{\bar y_0^\top}{\phi_{\mathrm{io}}},\qquad x^{\ast}=\frac{1}{\phi_{\mathrm{io}}}\big(1,\ \bar y_0^\top,\dots,\bar y_0^\top\big).
$$

In the Giannone, Lenza, Primiceri notation a smaller weight means a stronger pull; the design does not say which convention `spec.toml` follows (open questions). Candidates: $\lambda\in\lbrace0.05;\ 0.1;\ 0.2\rbrace$, $\phi_{\mathrm{sc}}=\phi_{\mathrm{io}}\in\lbrace1;\ 5\rbrace$, and the previous assumption as reference; the winner is chosen by the relative CRPS of the unconditional forecast against the random walk on the tuning half of the date grid.

`msvar.markov` has no crisis-month dummies (2014-11, 2022-02, 2022-03, the 2021-01 step): the stress regime absorbs crisis months. The 2021-01 splice of the spreads is therefore not modelled separately either (open questions).

Steps of one sampler sweep:

1. Missing panel months are drawn from the conditional normal law given the current $\theta$ and $\zeta$ (data augmentation).
2. The historical regime path is drawn as a whole by forward-filtering backward-sampling (Carter, Kohn, 1994; Kim, Nelson, 1999): the Hamilton filter forward, then sampling backward.
3. Coefficients by equation. Under `"mean_covariance"` one lag block $A_1,\dots,A_p$ is drawn from all months, each month's error weighted by the covariance $\Sigma(\zeta_t)$ of its regime, then the intercept $\mu(k)$ of each regime given the lag block. Under `"all"` all coefficients come from the months of their regime. The design does not say how equation-by-equation sampling handles a non-diagonal $\Sigma(k)$.
4. $\Sigma(k)$ from an inverse Wishart law on the residuals of regime-$k$ months; the parameters of its assumed law are not set by the design.
5. Row $\Pi_{k\cdot}$ from a Dirichlet law with the assumed parameters plus the number of transitions $k\to l$ in the regime path; the assumed parameters are not set by the design.
6. Relabelling: regimes are ordered by increasing $\operatorname{tr}\Sigma(k)$, and $\mu$, $A_j$, the rows and columns of $\Pi$ and the path $\zeta$ are permuted with $\Sigma$. Regime 1 thus stays calm from sweep to sweep, which removes label switching (Frühwirth-Schnatter, 2001). The order of `[[regimes]]` in `signs.toml`, calm to stress, is this order.

Stability filter. After sampling and before the rotation search, only versions with $\rho(k)\le\rho_{\max}$ in every regime are kept, $\rho_{\max}=1.005$. A unit root is admissible because most series are nonstationary; unbounded growth is not. This gives the posterior restricted to the stable region, by the same mechanism as the identification filter. Zero surviving versions is an error naming the model and the limit. The design forbids raising $\rho_{\max}$ when too few survive.

## 5. Identification

The data determine $\Sigma(k)$ but not $B(k)$. In each regime $k$ set $B(k)=C(k)Q(k)$, where $C(k)$ is a factor of $\Sigma(k)$ (the Cholesky factor in `bvar.sv`) and $Q(k)$ is an orthogonal matrix drawn from the uniform law (Rubio-Ramírez, Waggoner, Zha, 2010). A rotation is accepted if every sign in the table below holds in the impact month and the two following months (`sign_lag = 2`). The response of regime $k$ is computed with the regime held at $k$ (not explicitly set by the design; open questions). A version with no admissible rotation in some regime is dropped.

| Shock | Rises | Falls |
|---|---|---|
| `ru.monetary` | `ru.key_rate`, `ru.ruonia`, `ru.ofz_1y` | `ru.usdrub`, `ru.imoex`, `ru.cpi` |
| `ru.inflation` | `ru.cpi`, `ru.key_rate` | `ru.real_wage` |
| `ru.fx_risk` | `ru.usdrub`, `ru.ofz_10y`, `ru.spread_bbb` | `ru.imoex` |
| `gl.oil` | `gl.brent` | `ru.usdrub` |
| `gl.financial` | `gl.vix` | `gl.ust_10y`, `ru.imoex` |

`ru.cpi` was added to `falls` of `ru.monetary` because after a rate cut inflation stayed below base for months in both models. The five names match `bvar.sv`, so scenarios carry over between models. The remaining $n-5$ columns are unnamed (`gl.unnamed_k`, `ru.unnamed_k` in `bvar.sv`). The number of rotation attempts per version and a block structure of $Q(k)$ are not set by the design for `msvar.markov`.

Regime elements. The version's set $J$ (key `shocks`) holds the five economic shocks, the unnamed shocks and two regime elements, `regime.calm` and `regime.stress`. The latent coordinate is the regime path $\zeta$, declared by the class attribute `latents`; a scenario names it as the cause `regime_path`.

## 6. Forecast

The scenario plan (`ScenarioPlan`) holds the horizon $H$, constraints $\mathbf a\le W\mathbf y\le\mathbf b$ on the path $\mathbf y=(y_1^\top,\dots,y_H^\top)^\top$, the causes $D$, the regime event `regime_event`, and `inner`, the number of cause draws per value of $R$. An exact value widens to an interval of half-width $\tau$; the platform sets $\tau$, not this document. Impacts $O$ on macro variables are already taken out of the bounds: each bound of a term with weight $w$ is lowered by $w\,o_t$, and the impact is added to the paths after the draw.

Given $\theta$ and $\zeta$ the path is linear in the shocks:

$$
\mathbf y=c(\theta,\zeta)+\Phi(\theta,\zeta)\,\mathbf z,\qquad W\mathbf y=Wc+Z_D\mathbf z_D+Z_{\bar D}\mathbf z_{\bar D},
$$

where $c$ is the zero-shock forecast along the regime path $\zeta$, and $Z_D=W\Phi_D$, $Z_{\bar D}=W\Phi_{\bar D}$ are the responses of the constraint rows to the causes and to the other shocks.

Unconditional forecast (scenario $s_0$): for each draw $r$, $\theta^{(r)}$ comes from the sample, $\zeta$ from $\pi_\theta$, $\mathbf z\sim N(0,I)$. The regime mixture is realised path by path: each draw carries its own $\zeta$.

Conditional forecast. The causes are drawn from the normal law truncated to

$$
\mathbf a-Wc-Z_{\bar D}\mathbf z_{\bar D}\le Z_D\mathbf z_D\le\mathbf b-Wc-Z_{\bar D}\mathbf z_{\bar D},
$$

by the Gibbs sampler for the truncated normal (Geweke, 1991), 50 passes for interval constraints (Antolín-Díaz, Petrella, Rubio-Ramírez, 2021). A constraint is dependent if the causes move it beyond the rows already fixed by less than $10^{-9}$ of how much all shocks move it; the scenario is then rejected with the constraint numbers. Four cases for the regime path follow from the general scheme:

| Case | $R$ | Regime path |
|---|---|---|
| no causes named | $\theta$ | latent coordinate: 64 candidates (`LATENT_CANDIDATES`) from $\pi_\theta$, candidate weight $\propto P(S\mid\theta,\zeta)$, so $P_s(\zeta\mid\theta)\propto\pi_\theta(\zeta)P(S\mid\theta,\zeta)$ |
| causes are shocks $\mathbf z_D$ | $(\theta,\zeta,\mathbf z_{\bar D})$ | base law $\pi_\theta$, unchanged by the conditions |
| causes are shocks and `regime_path` | $(\theta,\mathbf z_{\bar D})$ | candidates from $\pi_\theta$ weighted $\propto P(S\mid R,\zeta)$ |
| `regime_event` $=k$ | — | $\zeta=(k,\dots,k)$ over the whole horizon |

If with named shocks some $n$ has $n^\top Z_D=0$ and $n^\top Z_{\bar D}\ne0$ on a set of positive probability, then $P(S\mid R)=0$ with positive probability and the scenario contradicts the models.

The regime event `regime_event` names `regime.stress` or `regime.calm` and fixes the regime path over the whole horizon, with or without conditions; the conditions are then met by the causes at the fixed regime. With no conditions on $y$ and no named causes the shocks keep their base law, and the price is

$$
\mathrm{dist}(s)=\mathbb E_P\big[-\log\pi_\theta(k,\dots,k)\big].
$$

How the price of a regime event combines with conditions on $y$ is not set by the design (open questions).

Constraint law (`constraint_law`): given $R$ the model returns the gaussian law of the rows of $W\mathbf y$ (means and covariances), grouped by the value of $R$, with latent candidates and log weights. The platform turns a group's rows into $P(S\mid R)$ with the GHK sampler and averages over groups:

$$
\mathrm{dist}(s)=\mathbb E_P\big[-\log P(S\mid R)\big].
$$

The platform imposes conditions on means and probabilities by exponential tilting within groups.

Diagnostics: the size of the needed driving shock in standard deviations per month, the effective size of the candidate weights, the counts `estimated`, `stable`, `identified`.

Bounds apply to the final paths after impacts and tilting: the output is $\max(s,\text{lower})$ in model scale, conditions are on the shadow, and an exact condition at the bound ("Fed at zero") means the shadow within $\tau$ of the bound.

## 7. Parameters

| Symbol | Meaning | Value | Source |
|---|---|---|---|
| $n$ | number of variables | 23 | registry, `src/mst/plugins/README.md` |
| $S$ | number of regimes | 2 (calm, stress) | plugin README, `src/mst/plugins/README.md` |
| `switching` | what switches | `"mean_covariance"`; alternative `"all"` | `plans/macro-models.md` |
| $p$ | number of lags | not set by the design (3 in `bvar.sv` and in the plan's reasoning) | `plans/macro-models.md`, `bvar-method-legacy.md` |
| $T_0$ | sample start | not set by the design (series from 2012-01) | `src/mst/metrics/README.md` |
| $\lambda$ | `tightness` | candidates 0.05, 0.1, 0.2; final not set | `plans/macro-models.md` |
| $\phi_{\mathrm{sc}}$, $\phi_{\mathrm{io}}$ | `mu`, `delta` | candidates $\phi_{\mathrm{sc}}=\phi_{\mathrm{io}}\in\lbrace1;5\rbrace$; final not set | `plans/macro-models.md` |
| $\delta_i$ | mean of the first own lag | 1 or 0; stationary list not set | `bvar-method-legacy.md` (for BVAR) |
| — | inverse Wishart parameters | not set by the design | — |
| — | Dirichlet parameters for $\Pi$ | not set by the design | — |
| $\rho_{\max}$ | spectral radius ceiling | 1.005 in every regime | `plans/macro-models.md` |
| `sign_lag` | sign window after impact | 2 months | `plans/macro-models.md`, `bvar-method-legacy.md` |
| `LATENT_CANDIDATES` | regime path candidates per group $R$ | 64 | `bvar-method-legacy.md`, `mst.model_sdk` |
| — | Gibbs passes for intervals | 50 | `bvar-method-legacy.md` |
| — | constraint dependence threshold | $10^{-9}$ | `bvar-method-legacy.md` |
| $\tau$ | half-width of an exact condition | set by the platform | `formalization.md` |
| `lower` | lower bounds | 0 for 9 rates, unemployment and `ru.spread_bbb` | `registry.toml` |
| `sampler.draws`, `thin` | sample size | not set by the design | — |

## 8. Inputs and outputs

Declaration: `plugin = Plugin(...)` with `macro=(MODEL,)`, `model_id` `msvar.markov`, a `model_sdk` range compatible with Model SDK 3.0 (impacts as paths, `AdditiveImpact`). The core gets the model only from the `mst.plugins` registry. The plugin imports only `mst.model_sdk`, `mst.contracts`, `mst.macro.panel`, `mst.macro.registry` and its own package.

- `estimate(context) → (meta, arrays)`: input is the series store, the registry, the date `as_of` and the specification. Required keys: `as_of`, `ids` (variable order, global block first), `months`, `shocks` (the set $J$: economic shocks and regime elements), `panel_y`, `panel_approximate`. The platform writes the key `model`. The arrays hold $\theta^{(r)}$; `lags` has shape `(M, S, p, n, n)` and under `"mean_covariance"` holds equal copies across regimes.
- `load(meta, arrays) → model of the version`; only this plugin reads it.
- `simulate(model, plan, draws, seed) → MacroSimulation`: paths `(R, H + 1, n)` in model scale, groups $R$, diagnostics; a pure function of the seed.
- `constraint_law(model, plan, …) → ConstraintLaw`: the law of $W\mathbf y$ by groups $R$ with candidates and weights.
- `latents`: class attribute declaring the regime path.

Estimation command: `python -m mst.models estimate-macro msvar.markov --as-of ... --store ... --models ...`.

## 9. Assumptions and limits

- Errors are gaussian within a regime; heavy tails come only from the mixture of two regimes, with no stochastic volatility or Student tails.
- Two regimes with common lags: stress changes the level and the spread, not the propagation of shocks.
- The Markov chain is homogeneous: transition probabilities do not depend on the variables.
- Estimation reads rate history as observed rather than shadow; this is an approximation.
- Publication dates of revisions before the first load are approximate.
- There is no feedback from the portfolio or satellites to the macro variables.
- The sign table is a hypothesis tested by backtest.
- Admission gates (`plans/macro-models.md`): G1, the share of kept versions with $\rho>\rho_{\max}$ in any regime is 0; G2, G3, below 5 % of base macro cells with a band outside the domain and wider than history; G4, the same in each of the 12 scenarios; G5, the largest portfolio gain below 10 times the notional; G6, relative CRPS not worse than the current version by more than two standard errors, with coverage shares consistent with 0.05, 0.45, 0.45, 0.05.

## 10. Checks the code must pass

1. The model is declared in the `msvar` plugin as its only macro model, with `model_id` `msvar.markov`.
2. The version's `ids` holds exactly the 23 registry variables, global block first.
3. Variables with `log100` enter the model as $100\ln X$, with `identity` as $X$; the inverse transform of the output to original units is exact.
4. Seasonality is removed only for `ru.cpi`, `ru.output`, `ru.unemployment`, `ru.real_wage`, `ru.retail`.
5. Estimation at `as_of` uses no value published after `as_of`.
6. A missing month is neither dropped nor filled with a constant: on simulated data with gaps, augmentation recovers the missing values within the posterior spread.
7. The number of regimes is 2, and the order of `[[regimes]]` is calm, then stress.
8. Under `switching = "mean_covariance"` the lag matrices of all regimes are equal element by element in every version, while intercepts and covariances differ.
9. Under `switching = "all"` results equal the previous behaviour.
10. On simulated data with common lags and two covariances the sampler recovers the lags, $\Pi$ and the regime shares.
11. The regime path is drawn by FFBS as a whole, not month by month.
12. The rows of $\Pi$ are nonnegative and sum to 1 in every version.
13. After each sweep $\operatorname{tr}\Sigma(1)\le\operatorname{tr}\Sigma(2)$ in every kept version, and the permutation is applied consistently to $\mu$, $A_j$, $\Sigma$, $\Pi$ and $\zeta$.
14. The sum-of-coefficients and initial-observation dummies are built from $\bar y_0$ and the weights `mu`, `delta`; the dummy-augmented regression equals the closed-form posterior on simulated data.
15. `msvar.markov` estimation has no crisis-month dummies.
16. No kept version has $\rho(k)>1.005$ in any regime.
17. A version set with an inserted explosive draw loses that draw and only that draw.
18. Zero surviving versions raise an error naming the model and the limit.
19. The stability filter runs before the rotation search; the diagnostics carry `estimated`, `stable`, `identified`.
20. In every kept version and every regime the signs of the table in section 5 hold in the impact month and the two following months, including the fall of `ru.cpi` after `ru.monetary`.
21. The five shock names equal those of `bvar.sv`.
22. The version's `shocks` key holds the economic shocks and the elements `regime.calm`, `regime.stress`.
23. The model declares the regime path in `latents`.
24. `lags` in the arrays has shape `(M, S, p, n, n)`.
25. `simulate` is a pure function of the seed: two calls with one seed give bitwise equal paths.
26. Without conditions each draw carries its own regime path, and regime shares on the horizon agree with $\pi_\theta$.
27. With no causes named, regime path candidates (64 per group $R$) are weighted in proportion to $P(S\mid\theta,\zeta)$.
28. With shocks as causes and no `regime_path`, the law of the regime path under conditions equals the base law.
29. Under `regime_event = regime.k` every horizon month of every draw is in regime $k$, and without conditions the price equals $\mathbb E_P[-\log\pi_\theta(k,\dots,k)]$ within its error (checked against the closed form).
30. Interval conditions hold on every path, up to the tolerance $\tau$.
31. A scenario with dependent constraints is rejected with the constraint numbers.
32. `constraint_law` returns a law of the rows of $W\mathbf y$ consistent with the paths of `simulate`: the share of paths in the box equals $P(S\mid R)$ within Monte Carlo error.
33. No output path of a variable with `lower` lies below its bound; the shadow continues below the bound, and a later rise starts from it.
34. Conditions hold on the shadow; without `lower`, paths equal the paths without bounds.
35. `ru.spread_aa` and `ru.spread_a` get no bound.
36. The size of the driving shock in standard deviations per month is written to the calculation diagnostics.
37. An impact on a macro variable shifts only that variable's path and only in the impact months; a condition on an impacted variable holds on the final path.

## 11. Open questions and contradictions

1. What switches. The plugin README describes switching of the intercept, lags and covariance; `plans/macro-models.md` makes `"mean_covariance"` with common lags the default. Which value the released version uses is unknown.
2. "Mean" or intercept. The plan calls the step "regimes switch the mean and covariance", while estimation switches the intercept $\mu(k)$; with common lags the unconditional mean of a regime is $(I-\sum_jA_j)^{-1}\mu(k)$, undefined at unit roots.
3. Bound of `ru.spread_bbb`. The registry and the README set `lower = 0` (history minimum +0.99), while `plans/macro-models-results.md` says all three spreads were left unbounded.
4. Who applies bounds. The plan: both `simulate` functions on the final paths; the README: the platform after impacts and tilting.
5. The law of $\zeta_0$ and the starting regime of the forecast, whether the last regime of the version's posterior path or the filtered probability, is not stated; $\pi_\theta$ and the regime event price depend on it.
6. Price of a regime event together with conditions on $y$: the README fixes the path and meets conditions by causes, while the canon prices only a regime event without conditions.
7. Block exogeneity of the global block is not mentioned for `msvar.markov`; `bvar.sv` has it.
8. The response used for signs in regime $k$, with the regime held fixed or with transitions, is not set.
9. Convention for `mu`, `delta`: whether a larger weight means a stronger pull (Sims, Zha, 1998) or a weaker one (Giannone, Lenza, Primiceri, 2015), and which regime's intercept the initial-observation dummy uses.
10. Not set: $p$, $T_0$, final $\lambda$, `mu`, `delta`, the Wishart and Dirichlet parameters, draws, burn-in, thinning, rotation attempts, the list of stationary variables.
11. Equation-by-equation coefficient sampling with a non-diagonal $\Sigma(k)$: the method (the triangular factorisation of Carriero, Clark, Marcellino, 2019, or another) is not set.
12. The 2021-01 spread splice: `bvar.sv` absorbs it with a step dummy; `msvar.markov` has no dummies, so the level jump falls on the regime.
13. Plan stop condition: if the stress share of the last month stays above 0.8 while stress months are spread evenly over calm years, the regime is not a stress regime, and a third regime or a persistence assumption is the owner's decision. The measurement is not in the repository.
14. `plans/macro-models-results.md` holds only the "before" state (version `ecaa…`: 367 of 1000 versions survived identification, 100 % of stress draws with $\rho>1$); there are no measurements of a new version, so passing gates G1–G6 is not confirmed.
15. Seasonal variable list: `bvar-method-legacy.md` points to `src/mst/macro/spec.toml`, while the registry marks them with the field `seasonal`.

## References

Antolín-Díaz J., Petrella I., Rubio-Ramírez J. (2021). Structural scenario analysis with SVARs. JME.
Carter C., Kohn R. (1994). On Gibbs sampling for state space models. Biometrika.
Doan T., Litterman R., Sims C. (1984). Forecasting and conditional projection using realistic prior distributions. Econometric Reviews.
Frühwirth-Schnatter S. (2001). Markov chain Monte Carlo estimation of classical and dynamic switching and mixture models. JASA.
Geweke J. (1991). Efficient simulation from the multivariate normal and Student-t distributions subject to linear constraints.
Giannone D., Lenza M., Primiceri G. (2015). Prior selection for vector autoregressions. REStat.
Hamilton J. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. Econometrica.
Kim C.-J., Nelson C. (1999). State-Space Models with Regime Switching. MIT Press.
Litterman R. (1986). Forecasting with Bayesian vector autoregressions. JBES.
Rubio-Ramírez J., Waggoner D., Zha T. (2010). Structural vector autoregressions. REStud.
Sims C., Zha T. (1998). Bayesian methods for dynamic multivariate models. IER.
Sims C., Waggoner D., Zha T. (2008). Methods for inference in large multiple-equation Markov-switching models. J. Econometrics.
