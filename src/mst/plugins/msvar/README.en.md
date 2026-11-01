# mst.plugins.msvar

A macro model of level 0 with Markov switching: a vector autoregression whose constant, lag coefficients and error covariance depend on a discrete regime (`spec.toml` fixes the lags and priors, `signs.toml` the regimes and shocks). Estimation is Gibbs: the regime path is sampled by forward-filtering backward-sampling (Carter–Kohn), the coefficients by regime equation by equation with a Minnesota prior, the covariances by inverse Wishart, the transitions by Dirichlet; missing panel months are completed by augmentation. Shocks are identified by signs inside each regime (`signs.toml`, the same five names as the BVAR, so scenarios carry over between models); the conditional forecast under conditions uses the common machinery of `mst.model_sdk.conditional` given a regime path.

A scenario speaks of regimes in two ways. The field `regime_event` names `regime.stress` or `regime.calm`, and the model fixes the regime path of the event over the whole horizon: the narrative "the economy enters a crisis" becomes a statement about the event, and its price is the mean of $-\log\pi_\theta(k,\dots,k)$. The path is fixed even with no conditions, and conditions are then met by the causes at the fixed regime. The cause `regime_path` explains an event through the regime without fixing it: the path is a hidden coordinate, and candidates are weighted by the probability they give the conditions. The mixture over regimes is path by path: each draw carries its own path $\zeta$, and satellites do not know about regimes.

Regime labels are held by the overall volatility: after each sampler pass the regimes are renumbered in ascending order of the covariance trace, so the order of declaration in `signs.toml`, from calm to stress, does not change from pass to pass. The order of `[[regimes]]` in `signs.toml` is this order.

The package connects like any model plugin: `plugin = Plugin(...)` with `macro=(MODEL,)`. The core knows it only through the registry `mst.plugins`; to estimate and publish a version: `python -m mst.models estimate-macro msvar.markov --as-of ... --store ... --models ...` (with two installed macro models, the BVAR and this one, the model name is mandatory).

| File | What it does |
|---|---|
| `plugin.py` | The plugin declaration: one macro model `msvar.markov` |
| `model.py` | The adapter to `mst.model_sdk`: estimate/load/simulate/constraint_law and `parts` (the meta and arrays of a version; the version's `shocks` are the economic shocks plus the regime elements) |
| `estimate.py` | The Gibbs sampler: FFBS of the regime path, coefficients by regime, covariances, transitions, augmentation of missing months; renumbering of regimes by volatility |
| `identify.py` | Sign restrictions by uniform rotations inside each regime |
| `forecast.py` | The conditional forecast by the regime path (path candidates with unnamed causes and the cause `regime_path`, the regime event `regime_event` separately) and the law of the conditions of the mixture |
| `signs.py`, `signs.toml` | Named shocks with signs and regimes, the set `J` of each version |
| `spec.py`, `spec.toml` | The estimation specification: lags, priors, transitions, sampler; no dummies, since the stress regime takes the crisis months |
| `var.py` | The algebra of the switching VAR: regressors, the block mask, responses, iteration along a regime path |

The check on historical episodes lives in the core, [mst.metrics](../../metrics/README.en.md) (`python -m mst.metrics backtest msvar.markov ...`), along with the benchmarks and CRPS; the model package does not have it. Tests are in `tests/msvar` (simulated data from known parameters: recovery of transitions and regime shares, augmentation of missing values, signs in both regimes, the conditional forecast, the regime element of an event, the price of a regime event against the closed form).
