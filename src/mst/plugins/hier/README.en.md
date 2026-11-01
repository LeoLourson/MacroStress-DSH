# mst.plugins.hier

The reference satellite model of levels above zero: a regression of each driver on its own past and the macro inputs `x_t`, a parameter hierarchy along the object tree (Gibbs: an object around its parent, a class around the mean), a draw of unobserved months, an out-of-sample check against an autoregression without macro variables.

The package connects like any model plugin: `plugin = Plugin(..., satellites=(MODEL,))` with the model `hier.gibbs`, which serves the quantities of the driver registry. Several satellite models may claim a quantity (a plugin in development beside an installed one); then the model that estimates it is named explicitly: `--model QUANTITY=MODEL` in `python -m mst.models estimate-satellites`; an assignment without a name is ambiguous and rejected.

| File | What it does |
|---|---|
| `plugin.py` | The plugin declaration: one satellite model `hier.gibbs` |
| `model.py` | The adapter to `mst.model_sdk`: estimate/load/simulate; a version part carries the declaration of the quantity, so loading needs no registry |
| `estimate.py` | Gibbs over the object tree, column scaling, stationarity of `rho`, residuals at the posterior mean |
| `check.py` | The out-of-sample check: holdout, the reach horizon, RMSE against AR(1) without macro |
| `simulate.py` | Driver paths on the macro paths: a start from the last known value, completing months up to `t=0`, joint noise by the version's correlation, impacts; the platform's `shocks` are used, while the model's stream stays for the number of the posterior draw and the independent noise |
| `spec.py`, `spec.toml` | The specification: prior deviations, `rho_max`, holdout/reach, sampler |

Tests are in `tests/hier`. Data preparation (a quantity's history on the panel, `x` from the paths, residual correlation) stays in the core, `mst.drivers.history`; the assembly of a version from parts of several models is `mst.drivers.bundle`. Another satellite model is a separate package serving its own quantities.
