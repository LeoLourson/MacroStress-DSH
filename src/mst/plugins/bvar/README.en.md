# mst.plugins.bvar

The reference macro model of level 0: a vector autoregression with a Minnesota prior, block exogeneity (the global block does not respond to the Russian one), stochastic volatility and Student tails. Shocks are identified by signs (`signs.toml`), and the conditional forecast under conditions follows Antolín-Díaz, Petrella and Rubio-Ramírez (2021); the common machinery lives in `mst.model_sdk.conditional`.

The package connects like any model plugin: `plugin = Plugin(...)` with `macro=(MODEL,)`. The core knows it only through the registry `mst.plugins`; to estimate and publish a version: `python -m mst.models estimate-macro bvar.sv --as-of ... --store ... --models ...`.

Rates and unemployment cannot go negative, and a linear model does not know it: the registry gives them a lower bound `lower` (zero), the model works with a shadow value, and the simulation reports the observed one — the shadow read through the bound, `x_t = max(s_t, lower)` in the model scale (`mst.model_sdk.bounds`). The lags, the conditions and the price stay on the shadow — the gaussian law the machinery knows; an exact condition at the bound ("the Fed at zero") means the shadow within its tolerance of the bound. Estimation still reads the history as observed — an approximation; a proper shadow-rate estimation is separate work. The spreads of the AA, A and BBB groups carry no bound: their history has been below zero (`negative = true` in the registry).

| File | What it does |
|---|---|
| `plugin.py` | The plugin declaration: one macro model `bvar.sv` |
| `model.py` | The adapter to `mst.model_sdk`: estimate/load/simulate/constraint_law and `parts` (the meta and arrays of a version); the forecast reads the neutral plan directly |
| `estimate.py`, `sv.py`, `var.py` | Estimation: two samplers (conjugate and triangular CCD), augmentation of missing months, stochastic volatility; the prior's dummy observations (`mu`, `delta` — Sims-Zha / Giannone-Lenza-Primiceri) and the degrees-of-freedom floor `dof_min` |
| `identify.py` | Sign restrictions by uniform rotations; before them, the stability filter `rho_max` on the companion radius |
| `forecast.py` | The conditional forecast (candidates of the volatility path with unnamed causes or `volatility_path` among the causes, the registry's floors on the final paths, the size of the driving shocks in standard deviations in the diagnostics) and the law of the conditions |
| `signs.py`, `signs.toml` | Named shocks and their signs, the set `J` of each version |
| `spec.py`, `spec.toml` | The estimation specification: lags, priors, seasonal adjustment, regime dummies, sampler, the stability ceiling, the weights of the dummy observations and the tails' floor |

The check on historical episodes lives in the core, [mst.metrics](../../metrics/README.en.md) (`python -m mst.metrics backtest bvar.sv ...`), along with the benchmarks and CRPS; the model package does not have it.

Tests are in `tests/bvar`. Another macro model is a separate package with another `model_id`, and the core does not change.
