# reflevel2: reference description

This description follows the design: concept documents, SDK documents and commit messages,
without reading the plugin's code. Where the design is silent, no decision is invented; the
gap goes to "Open questions and contradictions".

## 1. Purpose and place in the system

MacroStress is a recursion over levels $l=0,\dots,L$: level 0 comes from the macro model, level
$l\ge 1$ from a satellite model $x^l=G^l(x^{<l},\omega^l)$, where $x^{<l}$ are the paths of
lower-numbered levels and $\omega^l$ is the random outcome of level $l$, its parameters
included. Level $l$ has no feedback into lower levels.

`reflevel2` is a plugin with one satellite model, `reflevel2.regression`, of level 2. "Reference
of level 2" means a reference implementation: the simplest model that shows how to connect a
quantity above level 1. The model reads paths of level 0 (macro variables) and level 1 (drivers
of first-level satellites). The plugin arrived together with support for $L>1$, because before
it no model read level-1 paths.

The model declares no quantity of the main driver registry, since that registry holds no
level-2 quantity. A catalog file brings the quantity, and only an explicit assignment leads it
to the model: `--model QUANTITY=reflevel2.regression` in `python -m mst.models
estimate-satellites`, or the `assignment` field of the `model` tool. Its users are the developer
of a new level (as a template) and the tests of the level recursion.

## 2. Data, units, transformations

Let $q$ be the level-2 quantity assigned to the model, $j$ its driver (a pair "quantity,
object"), $t$ a month.

- The platform (`mst.drivers.history`) prepares observations $y_{j,t}$ in model scale: the
  registry unit with the quantity's declared transformation applied, `identity`, `log100`
  ($100\ln$) or `logit`.
- The platform builds the regressors $g(x^{<2})$ from the quantity's registry entry: regressors
  are macro variables and quantities of levels below 2, features $\max(0,y-c)$, lags. For
  estimation, series of the levels above arrive in `upper_history`; an object without its own
  series takes the nearest object above it in the object tree.
- For simulation the same regressors are built from paths: `upper.macro` (level 0) and
  `upper.levels[1]` (level 1, with that level's impacts already applied).

The model neither loads nor transforms series itself, so estimation and simulation share one
scale by construction.

## 3. Model

For driver $j$ and month $t$

$$
y_{j,t}=\beta_j^\top g_{j,t}+\sigma_j e_{j,t},\qquad e_{j,t}\sim\mathcal N(0,1),
$$

where $y_{j,t}$ is the driver's observation in model scale, $g_{j,t}\in\mathbb R^{k}$ the
regressor vector built from $x^{<2}$ for month $t$, $\beta_j\in\mathbb R^{k}$ the coefficients,
$\sigma_j>0$ the residual standard deviation, $e_{j,t}$ the standardized residual. Within level
2 the vector $(e_{j,t})_j$ is correlated with correlation matrix $\Sigma$, measured and stored
by the platform; across levels residuals are independent, as the product measure
$P=\bigotimes_l P^l$ requires.

The design's formula has no own lag $y_{j,t-1}$; whether it enters through declared lags is not
defined (see open questions).

A scenario impact on level 2 is a path $o_j$: the value $o_{j,t}$ is added to the model output
in month $t$,

$$
x^2_{j,t}=\beta_j^\top g_{j,t}+\sigma_j e_{j,t}+o_{j,t}.
$$

An impact has no dynamics of its own: where $o_{j,t}=0$ the path equals the model output.

## 4. Estimation

The parameters $(\beta_j,\sigma_j^2)$ are estimated on observations up to the as-of date with a
conjugate posterior: under a normal likelihood the conjugate family is normal-inverse-gamma,

$$
\sigma_j^2\sim\mathrm{IG}(a_n,b_n),\qquad \beta_j\mid\sigma_j^2\sim\mathcal N\big(m_n,\sigma_j^2V_n\big),
$$

$$
V_n=\big(V_0^{-1}+G_j^\top G_j\big)^{-1},\quad m_n=V_n\big(V_0^{-1}m_0+G_j^\top Y_j\big),
$$

$$
a_n=a_0+\tfrac{n_j}{2},\quad b_n=b_0+\tfrac12\big(Y_j^\top Y_j+m_0^\top V_0^{-1}m_0-m_n^\top V_n^{-1}m_n\big),
$$

where $Y_j$ is the vector of the driver's $n_j$ observations, $G_j$ the regressor matrix over
the same months, and $(m_0,V_0,a_0,b_0)$ the initial assumption about the parameters.
Conjugacy is chosen because the posterior then has a closed form and needs no sampler. The
design does not set $(m_0,V_0,a_0,b_0)$.

Together with the parameters the model records training bounds, the range of values it was
estimated on, so that the platform can flag paths that leave it.

The output of estimation is the quantity's part of the version, with required keys `months`,
`nodes`, `driver`, `rows` (months used to estimate each node) and `checks`. The platform writes
the `model` key.

## 5. Simulation and outputs

For replication $r$ and horizon month $h=1,\dots,H$:

1. regressors $g_{j,h}^{(r)}$ are built from the paths `upper.macro` and `upper.levels[1]` of
   replication $r$;
2. parameters $(\beta_j,\sigma_j)^{(r)}$ are taken from the posterior of section 4, since the
   canon keeps the law of $\theta$ under a scenario;
3. the residual $e_{j,h}^{(r)}$ is driver $j$'s column of the `shocks` array of shape
   `(R, steps, k)` passed by the platform (first rows needed); a driver without history marked
   `parameters_from_above` gets independent noise from the model's stream;
4. $x^{2,(r)}_{j,h}=\beta_j^{(r)\top}g_{j,h}^{(r)}+\sigma_j^{(r)}e_{j,h}^{(r)}+o_{j,h}$.

The output is driver paths (`DriverPaths`), "replications × horizon months" per driver, in
model scale. Simulation is a pure function of the seed.

## 6. Parameters

| Symbol | Meaning | Value | Source |
|---|---|---|---|
| $\beta_j$ | regression coefficients of driver $j$ | estimated | design (regression formula) |
| $\sigma_j$ | residual SD of driver $j$ | estimated | design |
| $\Sigma$ | residual correlation of level-2 drivers | measured by the platform | SDK |
| $m_0,V_0$ | initial assumption about $\beta_j$ | not set by the design | — |
| $a_0,b_0$ | initial assumption about $\sigma_j^2$ | not set by the design | — |
| $k$, contents of $g$ | number and contents of regressors | from the quantity's registry or catalog entry | driver registry |
| level | the quantity's level | 2 | plugin name, SDK |
| `model_id` | model identifier | `reflevel2.regression` | plugin registry |

## 7. Inputs and outputs per the SDK contracts

- `estimate(SatelliteEstimateContext)`: input is the store, registries, as-of date, the
  prepared history of the quantity and `upper_history`; output is the part's meta and arrays
  with the keys of section 4.
- `load(meta, arrays)`: reads only a version whose `meta["model"]` is `reflevel2.regression`.
- `simulate(model, UpperPaths, drivers, impacts, rng, shocks)`: output is driver paths. Impacts
  are `AdditiveImpact` with `target`, `start` and a path `values`: `values[k]` is added in month
  `start + k`.
- Plugin declaration: the model claims no quantity of the main registry.

## 8. Assumptions and limits

- The relation is linear in the regressors; the residual is normal and homoscedastic.
- Level 2 has no feedback into levels 0 and 1, as the canon requires.
- The model is not meant for $L>2$: it reads only levels 0 and 1.
- The model is a template for connecting a level, not a production model; the design does not
  assess its forecast quality.
- Linear extrapolation outside the training bounds is unchecked; the platform only flags it.

## 9. Checks the code must pass

1. A model with `model_id = "reflevel2.regression"` is declared as a satellite model of plugin
   `reflevel2`.
2. The model declaration claims no quantity; without explicit assignment a level-2 quantity
   does not reach the model.
3. With explicit assignment of a level-2 catalog quantity, `estimate` succeeds and returns a
   part with keys `months`, `nodes`, `driver`, `rows`, `checks`.
4. The platform rejects a part naming another `model`; `load` does not read another model's
   version.
5. On data generated from $y=\beta^\top g+\sigma e$ with known $\beta,\sigma$, the posterior
   mean of $\beta$ converges to the true value as $n$ grows.
6. The posterior $(m_n,V_n,a_n,b_n)$ matches the formulas of section 4 for a given initial
   assumption about the parameters (checked by hand on a small example).
7. `simulate` reads `upper.levels[1]`: replacing level-1 paths at the same seed and shocks
   changes level-2 paths when the matching coefficient is nonzero.
8. `simulate` uses the passed `shocks`: `mst.model_sdk.testing.check_shocks` returns no
   findings.
9. Same seed, inputs and version give bitwise identical paths.
10. An impact with path `values` from month `start` is added exactly in months `start + k` and
    leaves other months unchanged.
11. Two overlapping impacts on one driver add up.
12. An impact whose path extends past the horizon is rejected, naming the target.
13. An impact on level 1 reaches level 2 through the regressors; an impact on level 2 does not
    change level-0 or level-1 paths.
14. Level-2 shocks are independent of level-1 shocks: the levels use different streams.
15. In a three-level chain with known coefficients the model recovers them within statistical
    error.
16. The version holds training bounds, and the bundle flags a path that leaves them.

## 10. Open questions and contradictions

1. Initial assumption about the parameters $(m_0,V_0,a_0,b_0)$: the design says "conjugate
   posterior" but gives no values, nor whether it is flat.
2. Contents of $g$: the formula $g(x^{<2})$ leaves the function open. The registry provides
   regressors, features $\max(0,y-c)$ and lags; whether a constant and the own lag $y_{j,t-1}$
   enter is not stated.
3. Scale, levels or differences: the satellite plan moves `share_price` to monthly changes
   because regressing levels on levels gives a spurious long-run level; for `reflevel2` the
   design writes the regression in levels and says nothing about differences.
4. Parameters in simulation: the canon keeps the law of $\theta$, so parameters should be drawn
   from the posterior per replication; the SDK description does not require this of
   `reflevel2` explicitly. The alternative is plugging in the posterior mean.
5. Link between `shocks` and $e$: platform shocks are standardized residuals; whether the model
   scales them by a drawn or a mean $\sigma_j$ is not stated.
6. One $\beta$ per driver, or one per quantity pooled over the object tree: the formula has no
   driver index, while the part carries `nodes` and per-node `rows`.
7. Training bounds: the design says "with training bounds" but not what they are taken over
   ($y$, regressors, or both) or whether they only flag or also clip paths.
8. Contents of `checks`: the key is required, and other satellites put an out-of-sample check
   against an autoregression there; for `reflevel2` the check is not specified.
9. Meaning of "reference": in `mst.plugins` and the SDK it is a reference implementation of
   connecting a level; in the metrics, "reference" forecasts are a random walk with drift and
   the year-earlier value. Whether anything uses `reflevel2` as a forecast benchmark in the
   backtest, the design does not say.
10. Missing data: how the model handles a driver with missing months, or a driver without
    history beyond the general `parameters_from_above` rule, is not described.
11. Impact scale: whether $o_{j,t}$ is added in model scale (after transformation) or in the
    original unit is not stated for this plugin.
12. Size: the SDK calls the plugin "about two hundred lines"; a guide, not a requirement.
