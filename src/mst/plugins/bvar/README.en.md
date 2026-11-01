# mst.plugins.bvar

The reference macro model of level 0: a vector autoregression with a Minnesota prior, block exogeneity (the global block does not respond to the Russian one), stochastic volatility and Student tails. Shocks are identified by signs (`signs.toml`), and the conditional forecast under conditions follows Antolín-Díaz, Petrella and Rubio-Ramírez (2021); the common machinery lives in `mst.model_sdk.conditional`.

The package connects like any model plugin: `plugin = Plugin(...)` with `macro=(MODEL,)`. The core knows it only through the registry `mst.plugins`; to estimate and publish a version: `python -m mst.models estimate-macro bvar.sv --as-of ... --store ... --models ...`.

| File | What it does |
|---|---|
| `plugin.py` | The plugin declaration: one macro model `bvar.sv` |
| `model.py` | The adapter to `mst.model_sdk`: estimate/load/simulate/constraint_law and `parts` (the meta and arrays of a version); the forecast reads the neutral plan directly |
| `estimate.py`, `sv.py`, `var.py` | Estimation: two samplers (conjugate and triangular CCD), augmentation of missing months, stochastic volatility |
| `identify.py` | Sign restrictions by uniform rotations |
| `forecast.py` | The conditional forecast (candidates of the volatility path with unnamed causes or `volatility_path` among the causes) and the law of the conditions |
| `signs.py`, `signs.toml` | Named shocks and their signs, the set `J` of each version |
| `spec.py`, `spec.toml` | The estimation specification: lags, prior, seasonal adjustment, regime dummies, sampler |

The check on historical episodes lives in the core, [mst.metrics](../../metrics/README.en.md) (`python -m mst.metrics backtest bvar.sv ...`), along with the benchmarks and CRPS; the model package does not have it.

Tests are in `tests/bvar`. Another macro model is a separate package with another `model_id`, and the core does not change.
