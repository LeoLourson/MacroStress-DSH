# Reference description of the satellite model `hier.gibbs`

This description is built from the design documents, not from the code, so that the code of the `hier` plugin can be checked against it. Sources: `docs/concept/formalization.md` (the canon below), `docs/concept/architecture.md`, `docs/howto/plugins.md`, `docs/reference/platform.md`, `plans/satellite-models.md` (the satellite plan), `plans/canon-alignment.md` (the canon-alignment plan), the READMEs of `mst.plugins`, `mst.plugins.hier`, `mst.model_sdk`, `mst.plugin_sdk`, `mst.metrics`, `mst.drivers`, `mst.drivers.catalogs`, and the data file `src/mst/drivers/registry.toml` (the driver registry). Where the sources are silent or disagree, the question goes to section 10 and is not settled here.

## 1. Purpose and place in the system

The canon splits quantities into levels $l=0,\dots,L$. Level 0 holds the macro variables, whose paths come from the macro model (`bvar.sv` or `msvar.markov`, 23 variables). A satellite model of level $l\ge 1$ is a triple $(\Omega^l,P^l,G^l)$ with

$$
x^l=G^l(x^{<l},\omega^l),
$$

so the output of a level depends on the levels below and on its own random outcomes, and does not affect them (no feedback).

`hier.gibbs` is a level-1 satellite model that turns macro paths into driver paths. A *quantity* is a measure taken on objects (a share price, a bond spread). An *object* is an issuer, an issue, an industry or a region; it has a class, a parent and codes. A *driver* is the quantity of one object, written `<quantity>:<object>`, and its series is `drv.<quantity>.<object>`. By default the model serves every quantity of the driver registry; another model gets a quantity only by the explicit assignment `--model QUANTITY=MODEL`.

The inputs are the macro variables the registry lists as regressors of the quantity:

| Quantity | Regressors |
|---|---|
| `share_price` | `ru.imoex`, `ru.usdrub`, `ru.key_rate`, `gl.brent` |
| `bond_spread` | `ru.spread_aa`, `ru.spread_a`, `ru.spread_bbb`, `ru.key_rate`, `ru.ofz_5y` |
| `overdue_share` | `ru.key_rate`, `ru.output`, `ru.usdrub`, `ru.imoex`; feature $\max(0,z-15)$ with $z$ = `ru.key_rate` |
| `housing_price_primary`, `housing_price_secondary` | `ru.mortgage_rate`, `ru.key_rate`, `ru.cpi`, `ru.real_wage` |

The output is read by the position plugins: `equity` (`share_price`), `bond` (`bond_spread`), `credit` (`overdue_share`), `housing` (`housing_price_*`). A position finds its driver in the catalogue of the satellite version by quantity and code; an unresolved required driver leaves the position without a result. A level-2 model (reference: `reflevel2`) may read `hier.gibbs` drivers through `UpperPaths.levels[1]`.

## 2. Data

**Quantities and units.** For each quantity the registry sets a unit and a transform $g$ into the model scale $y=g(X)$, where $X$ is the value in original units:

| Quantity | Unit of $X$ | Transform | Model scale $y$ | `lags` | `span` |
|---|---|---|---|---|---|
| `share_price` | roubles per share | `log100` | $100\ln X$ | 1 | 1 |
| `bond_spread` | percentage points | `identity` | $X$ | 1 | 1 |
| `overdue_share` | share | `logit` | $\ln\frac{X}{1-X}$ | 1 | 1 |
| `housing_price_primary` | roubles per m² | `log100` | $100\ln X$ | 1 | 3 |
| `housing_price_secondary` | roubles per m² | `log100` | $100\ln X$ | 1 | 3 |

`lags` is the deepest lag of $x$ in the equation. `span` is the number of months one stored value covers: 3 for housing prices (a quarter's average), and the months in between are drawn. A missing `span` means one value per month.

The transform makes the model scale unbounded: $100\ln$ keeps a price positive, logit keeps a share inside $(0,1)$, and a spread may be negative, so it stays untransformed.

**Levels and differences.** All quantities are now modelled in levels of the model scale. The satellite plan proposes monthly changes for `share_price` (section 3, variant B); housing (quarterly `span`), bonds and overdue shares stay in levels, the last two because they are stationary. Gate S5 decides whether the change is kept (section 10, question 1).

**Objects and the tree.** The source catalogues named in the registry find the objects and their series. Each object sits under a parent that carries the same quantity one level higher:

| Quantity | Root | Inner nodes | Leaves |
|---|---|---|---|
| `share_price` | "Moscow Exchange shares" | 10 MOEX sector indices (`oil_gas`, `metals`, `financials`, `telecom`, `consumer`, `electric`, `transport`, `chemicals`, `it`, `real_estate`) | shares of the trading boards |
| `bond_spread` | "Corporate bonds" | rating groups `group_aa`, `group_a`, `group_bbb`; an issuer under the group of its first issue | issues |
| `overdue_share` | "ВСЕГО" (total) | industries of Bank of Russia table 01_02_A, level by the indent of the label | — |
| `housing_price_*` | Russia (643) | federal districts | regions by OKATO |

An object without a series (a driver without `series`) has no history of its own and takes the parameters of the level above; its path carries the flag `parameters_from_above`.

**Sample.** Series keep each value with its publication date; estimation reads only data known at the calculation date (`as_of`). Satellites are estimated on the panel of a given macro version and store its hash: the history of $x_t$ comes from that version's panel. The platform (`mst.drivers.history`), not the plugin, prepares the quantity's history on the panel: months, object tree, observations in the model scale, $x$ from regressors and features.

**Training range.** For each node $i$, estimation records the bounds $[\underline y_i,\overline y_i]$ (`train_low`, `train_high`), the smallest and largest observed value. A part without them is not read. When estimating in differences, the bounds are recorded in levels, so that the out-of-range flag keeps its meaning.

## 3. Model

Notation. A node $i$ is an object of the quantity's tree; $\pi(i)$ is its parent; $t$ is a month; $y_{i,t}$ is the value of node $i$'s driver in the model scale; $x_t\in\mathbb R^{p}$ is the regressor vector of month $t$ (macro variables in the model scale and declared features $\max(0,z_t-a)$ with variable $z$ and threshold $a$ from the registry). The registry sets the number of regressors $p$.

**Variant A, levels (current, all quantities).** The satellite plan gives the driver equation with `lags = 1`:

$$
y_{i,t}=c_i+\rho_i\,y_{i,t-1}+\beta_{0,i}^\top x_t+\beta_{1,i}^\top x_{t-1}+\sigma_i\varepsilon_{i,t},
$$

where $c_i$ is the constant, $\rho_i$ the coefficient on the own lag, $\beta_{0,i},\beta_{1,i}\in\mathbb R^p$ the loadings on current and lagged regressors, $\sigma_i>0$ the residual standard deviation, $\varepsilon_{i,t}$ the standardized residual. The own lag is there because the check compares the model with an autoregression without macro, and the model must contain it as a special case.

The long-run elasticity to the regressors and the level the driver is pulled to:

$$
\eta_i=\frac{\beta_{0,i}+\beta_{1,i}}{1-\rho_i},\qquad d_i^\ast(x)=\frac{c_i+(\beta_{0,i}+\beta_{1,i})^\top x}{1-\rho_i}.
$$

The canon's special case "satellite with a regression" reads $x^l_t=\beta^\top g(x^{<l}_t)+\sigma e_t$. Variant A follows with $g(x^{<l}_t)=(1,y_{i,t-1},x_t,x_{t-1})$, but $y_{i,t-1}$ belongs to the model's own level, a case the canon does not treat (section 10, question 3).

**Stationarity.** $\lvert\rho_i\rvert\le\rho_{\max}<1$, where $\rho_{\max}$ is the specification parameter `rho_max`. The bound makes $\eta_i$ and $d^\ast_i$ exist and keeps paths from exploding over the horizon.

**Variant B, differences (candidate for `share_price`).** The satellite plan:

$$
\Delta y_{i,t}=c_i+\rho_i\,\Delta y_{i,t-1}+\beta_{0,i}^\top\Delta x_t+\beta_{1,i}^\top\Delta x_{t-1}+\sigma_i\varepsilon_{i,t},
$$

where $\Delta y_{i,t}=y_{i,t}-y_{i,t-1}$ and $\Delta x_t=x_t-x_{t-1}$. Here $c_i$ is a monthly drift, $\beta_{0,i}$ are elasticities, and there is no long-run level. Reason: in variant A the levels of trending regressors create a spurious level $d^\ast_i$ far from today's price, and this pull sets the median share return in the base scenario. Variant B is estimated by the same sampler on differenced data. The regressor `ru.spread_bbb` for `share_price` is a further candidate of the plan (a credit channel for bank shares).

**Joint residuals.** The standardized residuals of all drivers with history are jointly normal in each month and independent over time:

$$
\varepsilon_t=(\varepsilon_{1,t},\dots,\varepsilon_{k,t})^\top\sim\mathcal N(0,\Sigma),\qquad \Sigma_{jj}=1,
$$

where $k$ is the number of drivers with history in the whole bundle and $\Sigma$ is the residual correlation matrix, estimated from residuals at the posterior mean of the parameters (section 4) and made positive definite when the bundle is assembled. $\Sigma$ is a parameter of the bundle, not of one part; the model receives its slice.

**Hierarchy.** Let $\phi_i=(c_i,\rho_i,\beta_{0,i},\beta_{1,i})$ be the coefficient vector of a node. The plugin README says: an object's parameters are drawn around its parent's, a class's around the mean. The natural form of this (Lindley and Smith, 1972) for an object is

$$
\phi_i\mid\phi_{\pi(i)}\sim\mathcal N\big(\phi_{\pi(i)},\ \operatorname{diag}(\tau^2)\big),
$$

where $\tau$ is the vector of scales of deviation from the parent (specification parameters). Pooling is partial: a node with a short history is pulled to its parent, one with a long history is set by its own data. Reason: hundreds of objects with short series cannot be estimated one by one, while one regression for all would erase the differences between sectors. What "class" and "mean" are, the distribution of the root and of $\sigma_i$ are not set by the design (section 10, question 4).

**Quarterly data.** With `span = 3` a stored value covers three months; the equation stays monthly, and the months in between are unobserved and drawn. The design does not say how the stored value relates to the monthly $y_{i,t}$: an average in the model scale or in original units (section 10, question 6).

## 4. Estimation

**Assumptions about parameters before data.** The design sets them through the specification: deviations of a node from its parent (plugin README), `rho_max`, the spread of the constant `constant_spread`. The design documents give no numbers, except the candidates `constant_spread` $\in\lbrace 10,1,0.1\rbrace$ for variant B, chosen by driver-level CRPS on the tuning half of the episodes. Regressor columns are scaled before the sampler, so that one deviation scale means the same for regressors of different magnitude. The scaling method is not set by the design.

**Gibbs sampler** (Gelfand and Smith, 1990). One sweep draws in turn from the full conditionals:

1. $\phi_i$ of each node given its data, $\sigma_i$, its parent and children: the normal law of a linear regression with the assumption $\mathcal N(\phi_{\pi(i)},\operatorname{diag}(\tau^2))$, truncated to $\lvert\rho_i\rvert\le\rho_{\max}$;
2. $\sigma_i^2$ given the residuals;
3. parameters of upper nodes given their children;
4. unobserved values $y_{i,t}$ given the parameters (data augmentation, Tanner and Wong, 1987).

The specification (sampler section) sets the number of sweeps, burn-in and thinning; the design gives no numbers. The sample of posterior parameter draws is stored; residuals for $\Sigma$ are taken at the posterior mean.

**Missing data.** A month without an observation is an unobserved value, not a dropped row, and it is drawn in step 4. In variant B a missing level in month $t$ gives missing $\Delta y$ in months $t$ and $t+1$, drawn by the same mechanism. The first value of a series is drawn from the node's stationary law.

**Out-of-sample check.** For each node the last months of history are held out; the model is estimated without them and forecasts up to the horizon `reach`. Its RMSE is compared with that of an AR(1) of the same node without macro variables on the same months. A node is served (`served`) if the model is more accurate. When a node's own equation loses, the coefficients of the nearest node above it estimated on data of its own are checked on the same months against the same autoregression; when they are more accurate, they serve the node, the part records the source in `sources`, and the driver's path carries the flag `parameters_from_above`. Otherwise the part records the reason: "not more accurate than without macro", "too few observations", "no holdout forecast". In variant B the comparison is in levels against the same autoregression in levels, so that the verdict stays comparable. The specification sets the holdout length and `reach`.

**Reason for the check:** macro regressors help a driver only if they beat its own persistence.

## 5. Simulation

The input is $R$ paths of macro variables on months $t=0,\dots,H$ ($t=0$ is the calculation month, the last month of history), with level-0 interventions already applied. For path $r$:

1. A posterior parameter draw is chosen; the draw number sets the model's own random stream.
2. Regressors $x_t$ are built from the paths the same way as in estimation (`mst.drivers.history`).
3. **Start.** The driver path starts from the last known value. If that value is older than $t=0$, the months between it and $t=0$ are filled by the equation. A driver without history starts with the parameters of the level above.
4. **Noise.** The model uses the columns of `shocks` of shape $(R,\text{ steps},k_m)$: standardized joint residuals of its $k_m$ drivers with history, in the order of the `drivers` argument, reading the first rows it needs. The platform draws them as $z\mathrm{chol}(\Sigma)^\top$ with $z\sim\mathcal N(0,I)$ from the stream `default_rng([seed, 0, l, r])`. Drivers with `parameters_from_above` get independent noise from the model's stream. Without `shocks` the model draws joint noise itself from the version's correlation. Using `shocks` is required: otherwise correlation between models does not reach the joint paths.
5. **Recursion.** $y_{i,t}=c_i+\rho_i y_{i,t-1}+\beta_{0,i}^\top x_t+\beta_{1,i}^\top x_{t-1}+\sigma_i\varepsilon_{i,t}$; in variant B the model simulates $\Delta y_{i,t}$ and accumulates them from the last known level.
6. **Interventions.** An intervention on a driver is an `Intervention` with `target`, `months`, `kind` (`set` or `shift`) and `values` of shape (groups, months). A set puts the value in its month, a shift adds to it; the model applies them once after the recursion (`apply_default`), so a set value does not enter the own lag and does not accumulate, and other drivers do not react to it. In the default form the other months and variables keep the model output; the paths of the level go to the levels below with the interventions applied.
7. **Inverse transform.** The path is returned in original units: $X=e^{y/100}$ for `log100`, $X=1/(1+e^{-y})$ for `logit`, $X=y$ for `identity`.
8. **Leaving the training range.** A driver path that leaves $[\underline y_i,\overline y_i]$ in at least one month counts toward the out-of-range share. Canon: this share must be below 1 %; until that holds and the backtest passes, the tail of the target is marked unverified. A position on such a driver gets the flag `outside_training_range:<driver>`, the report gets `tail_unverified`; the status of the calculation does not change. The model does not clip paths to the range: the rule only flags.

The design sets no bounds other than the range of the transform.

## 6. Parameters

| Symbol | Meaning | Value | Source |
|---|---|---|---|
| $c_i$ | constant (monthly drift in variant B) | estimated | satellite plan |
| $\rho_i$ | own-lag coefficient | estimated, $\lvert\rho_i\rvert\le\rho_{\max}$ | satellite plan, plugin README |
| $\beta_{0,i},\beta_{1,i}$ | loadings on $x_t$ and $x_{t-1}$ | estimated | satellite plan |
| $\sigma_i$ | residual standard deviation | estimated | canon (regression special case) |
| $\Sigma$ | correlation of the bundle's standardized residuals | estimated at the posterior mean | `docs/howto/plugins.md` |
| $\rho_{\max}$ | stationarity bound | not set by the design | plugin README (`rho_max`) |
| $\tau$ | deviations of a node from its parent | not set by the design | plugin README |
| `constant_spread`, `constant_spreads` | spread of the assumption on $c_i$ | 0.1 for shares and housing, 10 for bond spreads and overdue shares: of the plan's candidates $\lbrace 10,1,0.1\rbrace$, the one with the smaller mean holdout RMSE | `spec.toml` |
| `lags` | deepest lag of $x$ | 1 for all quantities | driver registry |
| `span` | months per stored value | 3 for `housing_price_*`, else 1 | driver registry |
| $a$ | threshold of the feature `key_rate_above_15` | 15.0 | driver registry |
| holdout, `reach` | held-out length and check horizon | not set by the design | plugin README |
| sampler settings | sweeps, burn-in, thinning | not set by the design | plugin README |
| out-of-range limit | share of paths outside $[\underline y_i,\overline y_i]$ | 1 % | canon |
| calculation admission limit | share of cells breaking a condition | 5 % | canon |

## 7. Inputs and outputs per the SDK contracts

The model is declared in `plugin.py` as `Plugin(..., satellites=(MODEL,))` with `model_id = "hier.gibbs"` and implements `SatelliteModel` of `mst.model_sdk` (Model SDK 4.0).

**`estimate(context) → meta and arrays`.** The context carries the store, the registries, the date and the quantity history prepared by the platform. It returns one part per assigned quantity with the required keys `months`, `nodes`, `driver`, `rows` (the number of months each node was estimated on), `checks` (the out-of-sample result), and also `train_low`, `train_high`. The platform writes the key `model`; a part naming another model is rejected. The part carries the quantity's declaration, so loading needs no registry.

**`load(meta, arrays) → version model`.** Reads only parts estimated by `hier.gibbs`.

**`simulate(model, upper, drivers, interventions, seed, shocks) → DriverPaths`.** `upper` is `UpperPaths`; a level-1 model reads only `upper.macro`. `drivers` are the requested drivers; `interventions` the realized interventions of its drivers; `shocks` an array $(R,\text{ steps},k_m)$ or absent. The output is paths $(R,H+1)$ per driver in original units with the interventions applied, and flags (`parameters_from_above`, `outside_training_range`). The function is pure: the same inputs and seed give the same paths. `mst.model_sdk.testing.check_shocks` checks that the model accepts `shocks`, uses them and keeps the path shape.

## 8. Assumptions and limits

- No feedback: drivers do not affect macro variables (the canon's definition of a satellite model).
- Residuals are normal, independent over time, with constant correlation $\Sigma$; regimes and stochastic volatility enter only through macro paths.
- The relation is linear in the model scale; nonlinearity enters only through declared features $\max(0,z-a)$.
- Parameters are constant over the sample; under a scenario their law is kept (canon: the law of $\theta$ does not change).
- The model is reliable inside the training range. By the satellite plan, in the base scenario the median share of paths leaving it is 29 % for shares, 56 % for bonds and above 90 % for housing, so the 1 % rule fails on every measured driver.
- Nodes that fail the out-of-sample check are not served: 138 of 263 share nodes were not served in version `a08a…`.
- The macro model has no gold variable, so gold miners have no gold regressor.

## 9. Checks the code must pass

1. Each quantity uses the registry's transform: `log100` gives $y=100\ln X$, `logit` gives $y=\ln\frac{X}{1-X}$, `identity` gives $y=X$; paths return to original units by the inverse.
2. The regressors of a quantity are exactly those of the registry plus declared features; the feature `key_rate_above_15` equals $\max(0,z-15)$ with $z$ = `ru.key_rate`.
3. The equation holds $x_t$ and $x_{t-1}$ (`lags = 1`), the own lag $y_{t-1}$ and a constant.
4. On simulated data with known $c,\rho,\beta_0,\beta_1,\sigma$, posterior means fall inside an interval consistent with the posterior spread.
5. Every posterior draw satisfies $\lvert\rho_i\rvert\le\rho_{\max}$.
6. A node with a short series is pulled to its parent: as its number of observations shrinks, its posterior mean $\phi_i$ approaches $\phi_{\pi(i)}$.
7. A driver without a series takes the parameters of the node above and carries the flag `parameters_from_above`. A driver with a series served by the coefficients of a node above (`sources`) carries the same flag and keeps its own start, noise and residual spread.
8. Unobserved months inside the sample are drawn, not dropped; with `span = 3` the months in between are drawn.
9. Estimation reads only data published no later than `as_of` and the history of $x$ from the panel of its macro version.
10. A version part holds `months`, `nodes`, `driver`, `rows`, `checks`, `train_low`, `train_high`; a part without `train_low`/`train_high` is not read.
11. `checks` records, per node, the RMSE of the model and of AR(1) without macro on one holdout and one `reach`; a node is served only if the model's RMSE is lower; a refusal names one of the three reasons of section 4.
12. $\Sigma$ is estimated from residuals at the posterior mean, has a unit diagonal and is positive definite after assembly.
13. Given `shocks`, the model takes their columns in the order of `drivers` and the first rows it needs; paths with and without `shocks` differ (`check_shocks` returns no finding).
14. Drivers with `parameters_from_above` get noise from the model's stream, independent of `shocks`.
15. A path starts from the driver's last known value; if it is older than $t=0$, the months up to $t=0$ are filled by the equation, and the value in the month of the last observation equals the observation.
16. The same inputs and seed give the same paths.
17. A set puts the value in its month on every path of its group and in no other month; a shift adds the value to the level in its months.
18. An intervention on a driver does not enter its own lag and does not change other drivers.
19. An intervention on a macro variable changes the drivers that read it, through $x_t$.
20. Groups of paths with different intervention values do not mix: the value stands on the paths of its group.
21. Each driver path has shape $(R,H+1)$; `log100` values are positive, `logit` values lie in $(0,1)$.
22. The share of paths outside $[\underline y_i,\overline y_i]$ is computed per node; at a share of 1 % or more, a position on the driver gets the flag `outside_training_range:<driver>`; paths are not clipped.
23. A level-1 model does not read `upper.levels`.
24. If variant B is on: the level path equals the starting level plus the cumulative sum of simulated changes; an intervention applies after accumulation and does not accumulate; `train_low`, `train_high` are recorded in levels; without variant B results are unchanged.
25. If the regressor `ru.spread_bbb` is on for `share_price`: a shift of `ru.spread_bbb` in a scenario moves `share_price:vtbr`; without the regressor it does not.

## 10. Open questions and contradictions

1. **Levels or differences for `share_price`.** The plugin README and the registry describe a model in levels. The satellite plan calls it a defect (a spurious long-run level, an elasticity of 4.04 to the dollar for BSPB) and proposes differences, but adoption depends on gate S5, whose result is not recorded in the documents.
2. **Regressors of `share_price`.** Registry: four variables. Satellite plan: candidate `ru.spread_bbb`, kept only if the `financials` node's elasticity is negative. The outcome is not recorded.
3. **The canon and the own lag.** The canon's special case writes $x^l_t=\beta^\top g(x^{<l}_t)+\sigma e_t$ with no own lag or state; the plan's equation has $\rho_i y_{i,t-1}$. The canon does not say whether the starting value and the filled months belong to $\omega^l$.
4. **Hierarchy.** Plugin README: "an object around its parent, a class around the mean". It is not defined what a "class" is (the object's class from the object contract, a sector, the whole quantity) or the mean of what; the form of the distribution (normal or not), the assumption on the root, and whether $\sigma_i$ and $\tau$ are shared or per node are not set.
5. **Section references.** The plugin README refers to "section 6.2 of the formalization", the registry to `docs/concept/bvar-method-legacy.md`, section 6. The current canon has no section on the satellite hierarchy; the older document is not among the sources of this description.
6. **Quarterly observation.** It is unknown whether a stored value with `span = 3` is the mean of monthly $y$ in the model scale or the mean of prices in roubles.
7. **Scale of macro regressors.** Registry: regressors "in the model scale $y=g(X)$"; the scale of the macro variables (levels, logs, percent) is set by the variable registry, which is not among the sources.
8. **Mapping of draws.** Plugin README: the model's stream is keyed "by the number of the posterior draw". How path number $r$ maps to a draw number when $R$ exceeds the number of stored draws is not set.
9. **Training range.** Canon: a 1 % limit. Satellite plan: the limit fails on every driver, a persistent driver near the edge of its range leaves it without anything being wrong, the rule stays, and the owner decides. It is not defined whether the share counts a path leaving in any month or counts cells.
10. **Specification numbers.** $\rho_{\max}$, $\tau$, `constant_spread`, holdout, `reach`, the sampler settings and the column scaling are not written in the design documents.
11. **Conditions on drivers.** The canon allows conditions on any level and keeps the law of $\theta$; the sources do not say whether conditions on `hier.gibbs` drivers are allowed and how $\varepsilon_{i,t}$ would then be tilted.
12. **Level of a quantity.** The Model SDK introduces a `level` field; the driver registry does not state it. Whether `hier.gibbs` serves quantities of level 2 and above under explicit assignment is not set.

## References

- Lindley D. V., Smith A. F. M. (1972). Bayes estimates for the linear model. *JRSS B*, 34(1).
- Tanner M. A., Wong W. H. (1987). The calculation of posterior distributions by data augmentation. *JASA*, 82(398).
- Gelfand A. E., Smith A. F. M. (1990). Sampling-based approaches to calculating marginal densities. *JASA*, 85(410).
- Gelman A., Carlin J. B., Stern H. S., Dunson D. B., Vehtari A., Rubin D. B. (2013). *Bayesian Data Analysis*, 3rd ed. CRC Press.
