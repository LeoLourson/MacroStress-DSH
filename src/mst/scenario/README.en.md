# mst.scenario

The stress-test scenario $(N, m, O, D, \mathcal C)$ from section 3 of the [canon](../../../docs/concept/formalization.en.md): the narrative $N$ with sources, the model versions $m$, the impacts $O$, the causes $D$ (coordinates of $\omega$ beyond the parameters: shocks and hidden coordinates) and the conditions $\mathcal C$ on the support, on means and on probabilities, of any level. A scenario is the file `workspace/scenarios/<id>.json`; a run builds from it paths $R \times (H+1)$.

| File | What it does |
|---|---|
| `contracts.py` | `Scenario` (with `causes`, `regime_event`, `expectations`, `probabilities`, `inner`, `approved_by`), `Constraint` (an interval `low < high` or a point `value`, a basis), `Term` (a macro variable or a driver `<quantity>:<object>`), `Expectation`, `ProbabilityCondition`, `Impact`; a month beyond the horizon, a mixed condition or a repeated variable is a refusal |
| `spec.py` | The file `scenarios/<id>.json`: the same `Scenario` plus the hashes of the macro version and satellites it is calculated on |
| `build.py` | `build`: the conditional forecast by the macro model plugin (a plan with expanded points), impacts on macro variables enter the path before the satellites, the tilt of the causes under conditions on means (`overlay`), the `mst.drivers` bundle simulates drivers by level, conditions on the lower levels carry weights inside the groups of $R$; the price $\mathrm{dist}(s) = \mathbb E_P[-\log P(S \mid R)]$ with a decomposition by level. `validate` names each infeasible combination, an unknown cause, a missing approval and impacts on unknown targets |
| `overlay.py` | `solve_lambda`: the convex dual problem of the tilt $e^{\lambda'f}$ (L-BFGS-B, sign bounds for one-sided conditions): weights inside the groups of $R$, the price, the effective size, reachability; `resample`: systematic resampling inside groups |
| `reverse.py` | The reverse problem of the canon: `Family` with fixed $m, O, D$, `sup_rho` under the price $\kappa$ and `inf_dist` over the level $L^\ast$; `python -m mst.scenario reverse` |
| `tool.py` | Operations of the tool `scenario`: `list`, `get`, `save`, `reverse`, `validate` |

Condition bounds are in the model scale $y=g(X)$: percent for a rate, $100\ln X$ for an exchange rate. The registry `mst.macro.registry` checks the variable names, and the macro version checks the shock names (the set is a property of the model).

Paths (`Paths` in `build.py`) are $R$ draws over $H$ months: macro variables and drivers in original units, the flags of driver paths (impact, parameters from above, approximate data, leaving the training range), the scenario price with a decomposition by level, the effective size of the tilt and the checks of what was imposed. Position models read them in one and the same form.

The checks of `validate`:

- an unknown variable or shock, by the registry `src/mst/macro/registry.toml`;
- the connection of conditions: a row the causes cannot move on their own is a refusal with the condition number;
- an impact on a variable absent from the macro version, and on a driver the bundle does not serve, with the refusal reason;
- satellites of another macro version, with both months;
- a corpus fragment named in the scenario and published after the calculation date, if the corpus index is built.

Tests are in [../../../tests/scenario](../../../tests/scenario/README.en.md).
