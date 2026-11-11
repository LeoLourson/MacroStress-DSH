# mst.plugins.bvar

The reference macro model of level 0: a vector autoregression with a Minnesota prior, block exogeneity (the global block does not respond to the Russian one), stochastic volatility and Student tails. Shocks are identified by signs (`signs.toml`), and the conditional forecast under conditions follows Antolín-Díaz, Petrella and Rubio-Ramírez (2021); the common machinery lives in `mst.model_sdk.conditional`.

The package connects like any model plugin: `plugin = Plugin(...)` with `macro=(MODEL,)`. The core knows it only through the registry `mst.plugins`; to estimate and publish a version: `python -m mst.models estimate-macro bvar.sv --as-of ... --store ... --models ...`.

Rates and unemployment cannot go negative, and a linear model does not know it: the registry gives them a lower bound `lower` (zero), the plugin simulates a shadow value, and after the impacts and the tilt the platform reads the observed one: $x_t=\max(s_t,\text{lower})$ in the model scale (`mst.model_sdk.bounds.observed`). The lags, the impacts, the conditions and the price stay on the shadow — the gaussian law the machinery knows; an exact condition at the bound ("the Fed at zero") means the shadow within its tolerance $\tau$ of the bound. Estimation still reads the history as observed — an approximation; a proper shadow-rate estimation is separate work. The BBB spread carries the bound too: its history never went below +0.99. The AA and A spreads carry none: their history has been below zero (`negative = true` in the registry).

The variables listed in `differences` of the spec are estimated on monthly changes $w_t=y_t-y_{t-1}$, the rest (the VIX and the spreads) in levels. Let $S$ be the diagonal matrix with ones at the variables in changes; the sampler estimates $p$ lags $G_k$ of the system $w_t=y_t-Sy_{t-1}$, and the version stores the equivalent VAR in levels with $p+1$ lags:

$$
A_1=S+G_1,\qquad A_k=G_k-G_{k-1}S\ (k=2,\dots,p),\qquad A_{p+1}=-G_pS.
$$

The forecast, the conditions, the identification and the core therefore read the version unchanged, the unit roots are exact, and the constant of a change equation is a monthly drift. The dummy rows `mu` and `delta` are written for levels and are refused together with `differences`. A regime dummy of the spec shifts a level, so a change equation reads its change: a step in month $m$ as an impulse in $m$, an impulse in $m$ as impulses in $m$ and $m+1$. Read as it is, a step would bend the drift for good. A dummy stays out of an equation whose variable is unobserved in every month where the dummy is non-zero, since no data fix its coefficient there. The choice of the list and of `tightness` is recorded in the comments of `spec.toml`.

Signs alone admit rotations in which a named shock barely moves the variable it is named after. The optional `shares` of a shock in `signs.toml` bound that: for each listed variable, the least share of its impact-month variance within the block that the shock carries, a bound on the size of the response in the manner of Kilian and Murphy (2012). The monetary shock carries at least a tenth of the key rate's surprise; without the bound a 5 pp cut needed a monetary shock of 20–30 standard deviations and was refused by the admission check. The numbers are in the comment of `signs.toml`.

| File | What it does |
|---|---|
| `plugin.py` | The plugin declaration: one macro model `bvar.sv` |
| `model.py` | The adapter to `mst.model_sdk`: estimate/load/simulate/constraint_law and `parts` (the meta and arrays of a version); the forecast reads the neutral plan directly |
| `estimate.py`, `sv.py`, `var.py` | Estimation: two samplers (conjugate and triangular CCD), augmentation of missing months, stochastic volatility; the prior's dummy observations (`mu`, `delta` — Sims-Zha / Giannone-Lenza-Primiceri) and the degrees-of-freedom floor `dof_min` |
| `identify.py` | Sign restrictions by uniform rotations; before them, the stability filter `rho_max` on the companion radius |
| `forecast.py` | The conditional forecast (candidates of the volatility path with unnamed causes or `volatility_path` among the causes, the size of the driving shocks in standard deviations in the diagnostics, which reach the run's `meta.json`) and the law of the conditions |
| `signs.py`, `signs.toml` | Named shocks and their signs, the set `J` of each version |
| `spec.py`, `spec.toml` | The estimation specification: lags, priors, seasonal adjustment, regime dummies, sampler, the stability ceiling, the weights of the dummy observations and the tails' floor |

The check on historical episodes lives in the core, [mst.metrics](../../metrics/README.en.md) (`python -m mst.metrics backtest bvar.sv ...`), along with the benchmarks and CRPS; the model package does not have it.

Tests are in `tests/bvar`. Another macro model is a separate package with another `model_id`, and the core does not change.
