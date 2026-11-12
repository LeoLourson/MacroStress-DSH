# mst.plugins.msvar

A macro model of level 0 with Markov switching: a vector autoregression whose constant, lag coefficients and error covariance depend on a discrete regime (`spec.toml` fixes the lags and priors, `signs.toml` the regimes and shocks). Estimation is Gibbs: the regime path is sampled by forward-filtering backward-sampling (Carter–Kohn), the coefficients by regime equation by equation with a Minnesota prior, the covariances by inverse Wishart, the transitions by Dirichlet; missing panel months are completed by augmentation. Shocks are identified by signs inside each regime (`signs.toml`, the same five names as the BVAR, so the same interventions carry over between models); the conditional forecast under observations uses the common machinery of `mst.model_sdk.conditional` given a regime path.

The regime path is a hidden coordinate of the model, not a field of the scenario. Under observations the model draws candidates of the regime path, weighs each by the probability its law gives to the box of the observations and picks one by the weights; without observations the path is drawn from the base law of transitions. The mixture over regimes is path by path: each draw carries its own path $\zeta$, and satellites do not know about regimes.

Regime labels are held by the overall volatility: after each sampler pass the regimes are renumbered in ascending order of the covariance trace, so the order of declaration in `signs.toml`, from calm to stress, does not change from pass to pass. The order of `[[regimes]]` in `signs.toml` is this order.

Rates and unemployment cannot go negative, and a linear model does not know it: the registry gives them a lower bound `lower` (zero), the plugin simulates a shadow value, and after the interventions and the tilt the platform reads the observed one (`mst.model_sdk.bounds.observed`). The lags, the interventions, the observations and the price stay on the shadow; an exact observation at the bound means the shadow within its tolerance $\tau$ of the bound. Estimation still reads the history as observed — an approximation. The BBB spread carries the bound too: its history never went below +0.99. The AA and A spreads carry none: their history has been below zero (`negative = true` in the registry).

The package connects like any model plugin: `plugin = Plugin(...)` with `macro=(MODEL,)`. The core knows it only through the registry `mst.plugins`; to estimate and publish a version: `python -m mst.models estimate-macro msvar.markov --as-of ... --store ... --models ...` (with two installed macro models, the BVAR and this one, the model name is mandatory).

| File | What it does |
|---|---|
| `plugin.py` | The plugin declaration: one macro model `msvar.markov` |
| `model.py` | The adapter to `mst.model_sdk`: estimate/load/simulate/constraint_law and `parts` (the meta and arrays of a version; the version's `shocks` are the economic shocks plus the regime elements) |
| `estimate.py` | The Gibbs sampler: FFBS of the regime path, coefficients (`switching`: "all" — by regime, "mean_covariance" — common lags with the mean and covariance switching), covariances, transitions, augmentation of missing months; the prior's dummy observations (`mu`, `delta`); renumbering of regimes by volatility |
| `identify.py` | Sign restrictions by uniform rotations inside each regime; before them, the stability filter `rho_max` in every regime |
| `forecast.py` | The conditional forecast by the regime path (path candidates weighted by the probability of the observations, the size of the driving shocks in the diagnostics, which reach the run's `meta.json`), the law of the observations of the mixture and the execution of interventions in the default form |
| `signs.py`, `signs.toml` | Named shocks with signs and regimes, the set `J` of each version |
| `spec.py`, `spec.toml` | The estimation specification: lags, priors, transitions, sampler; no dummies, since the stress regime takes the crisis months |
| `var.py` | The algebra of the switching VAR: regressors, the block mask, responses, iteration along a regime path |

The check on historical episodes lives in the core, [mst.metrics](../../metrics/README.en.md) (`python -m mst.metrics backtest msvar.markov ...`), along with the benchmarks and CRPS; the model package does not have it. Tests are in `tests/msvar` (simulated data from known parameters: recovery of transitions and regime shares, augmentation of missing values, signs in both regimes, the conditional forecast, the mixture over regime paths and the price of observations against the closed form).
