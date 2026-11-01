# mst.model_sdk

The contracts of macro and satellite models: what a plugin speaks, and only that. A macro model is level 0 of the scheme (the shock distribution and the transmission $G^0$), satellite models are the levels above ($G^l$, $l \geq 1$). The core knows of a model only its declaration and three steps; a version stores the name of the plugin that estimated it, and only this plugin reads it back.

| File | What it does |
|---|---|
| `contracts.py` | `ScenarioPlan` (horizon, conditions, causes $D$, regime event, `inner`), `AdditiveImpact` (an impact $o^l$ on any level), `MacroSimulation` (with the groups of $R$ and candidate diagnostics), `ConstraintLaw` (the law of $Wy$ given $R$), `Dist` (the price of a scenario by level), `UpperPaths`/`UpperHistory` (the levels above), `DriverPaths`, `Check`, estimation contexts and the abstract `MacroModel` (with `latents` and `constraint_law`) and `SatelliteModel`; `LATENT_CANDIDATES`; the mandatory keys of versions |
| `conditional.py` | The common conditional machinery: the truncated normal, Geweke's box Gibbs, `draw_driving` after Antolín-Díaz et al. (2021), `pick_latent` (choosing a hidden coordinate by weight), the GHK sampler `log_box_probability` of the box probability in logarithms, accurate in both tails; for any model of the form $y = \mu + \Phi\varepsilon$ |
| `dist.py` | The price of a scenario by the canon: $\mathrm{dist}(s) = \mathbb E_P[-\log P(S \mid R)]$ over the groups of $R$ of the law of the conditions (`price_of_law`), with $\log P(S)$, a standard error and Lemma 2 (`ContradictsModels`) |
| `overlay.py` | The overlay for a model given by a sample of paths (the canon, section 8): `reweight` gives the weights $1_S$ and the price $-\log(\lvert S\rvert/n)$; `resample` |
| `testing.py` | A check of the shock contract: `check_shocks` returns a finding if a satellite ignores the `shocks` the platform passes (the paths coincide under different shocks), does not accept the argument or changes the shape of the paths |
| `version.py` | The SDK version of the model contracts and a check of a plugin's `model_sdk` range |

The three steps of a model:

1. **`estimate(context) → meta and arrays`**: a pure estimate by the store, the registries and the date; the platform checks the mandatory keys and saves the version.
2. **`load(meta, arrays) → the model of the version`**: reading back only by its own plugin.
3. **`simulate(model, paths of the levels above, ...) → paths`**: a pure function of the seed: a macro model returns the paths of variables under the conditions, a satellite the paths of its drivers on these paths, with impacts and its own random stream. A satellite receives `UpperPaths`: the macro simulation and the driver paths of all levels above its own (with the impacts of these levels already applied), so a model of the first level reads only `upper.macro`, and a model of the second also reads `upper.levels[1]`. The platform itself draws the joint residuals of each level by the measured correlation matrix (each level has its own stream, and levels are independent) and passes each model its slice `shocks`: `(R, steps, k)` standardized residuals of its drivers with history in the order of the `drivers` argument; a model must use these columns and read the first rows it needs.

## How to add a level

Levels are calculated in ascending order: a quantity of level `l` reads macro variables (level 0) and quantities of levels below `l`, and the registry rejects a quantity that reads its own level or higher. To raise the system above the first level:

1. declare a quantity of level 2+ in the driver registry with its inputs (`level = 2`, `regressors` are macro variables and quantities of lower levels); a catalog file brings the quantity's objects and drivers, as [tests/drivers/level2_catalog.json](../../../tests/drivers/level2_catalog.json) does, because a second-level quantity does not live in the main registry;
2. write a satellite model of this level: `estimate` receives `upper_history`, the observed series of the quantities of the levels above from which its regressors are built (an object with no series of its own takes the nearest object above), and `simulate` receives `upper.levels` with the paths of these quantities. The reference implementation is the plugin [reflevel2](../plugins/reflevel2): the regression `x^2 = beta' g(x^{<2}) + sigma e` with a conjugate posterior, about two hundred lines;
3. assign the model to the quantity explicitly: `--model QUANTITY=MODEL` in `python -m mst.models estimate-satellites` or the field `assignment` of the tool `model`: a second-level quantity has no natural declarer, and only an explicit assignment leads it to a model. A chain of three levels with known coefficients and checks of impacts between levels is in [tests/drivers/test_levels.py](../../../tests/drivers/test_levels.py).

Another model is a package with another `model_id` ([docs/howto/plugins.en.md](../../../docs/howto/plugins.en.md)); the reference implementations are [mst.plugins.bvar](../plugins/bvar/README.en.md) and [mst.plugins.hier](../plugins/hier/README.en.md). Tests are in [../../../tests/model_sdk](../../../tests/model_sdk/).
