# mst.scenario

The stress-test scenario $s=(N,I,\mathcal A)$ of section 3 of the [canon](../../../docs/concept/formalization.en.md): the narrative $N$ with its sources, the interventions $I$ (a quantity, calendar months, a set or a shift, values in the quantity's own units) and the observations $\mathcal A$ (a band on a quantity, bounds on a mean, a probability). The file carries no model versions, no horizon and no path count: the calculation request names them, so one file runs on any model system. A scenario is the file `workspace/scenarios/<id>.json`, and a run builds from it paths $R\times(H+1)$.

| File | What it does |
|---|---|
| `contracts.py` | `Scenario` (`narrative`, `sources`, `interventions`, `observations`, `approved_by`); `Intervention` (`quantity`, `months` as `YYYY-MM`, `kind` — `set` or `shift`, `values` one per month, `unit` — `level`, `change` or `percent`, `reason`, `basis`); `Term` (quantity, month, weight); `Constraint` (an interval `low < high` or a point `value`), `Expectation` and `ProbabilityCondition` (`low`, `high`, `at_least`) make an `Observation`. A month outside the pattern, a repeated month, a random law of the value and a pair (quantity, month) in two interventions are refused |
| `spec.py` | `Spec = Scenario`: the file `scenarios/<id>.json` without versions or calculation settings |
| `translate.py` | `month_index` and `calendar_month` (the horizon month counted from the last panel month, not from the calculation date), `to_scale` and `shift_to_scale` (`level` is the quantity's level, `change` the move from the last known value, `percent` a share for quantities in `log100`), `observation_bounds`, `observations_of`, `observation_constraints`, `plan_of` → `ScenarioPlan`; it reads no model |
| `context.py` | `TranslationData` built from the macro panel and the satellite histories: the calendar origin, the quantities' scales and their last known values |
| `build.py` | `build` draws the paths: the macro model under the observations, the interventions of the default form applied after the draw (`apply_default`), while a model with a structural form executes them itself; the tilt for observations on means (`overlay`), the satellites by level, observations on drivers met by reweighting inside the groups when `inner` is at least 2, and the price $\mathrm{dist}(s)$. `dist_of` prices alone; `validate` returns `problems`, `labels` and `warnings` |
| `overlay.py` | `solve_lambda`: the convex dual problem of the tilt $e^{\lambda'f}$ (L-BFGS-B, sign bounds for one-sided observations) — the weights inside the groups of $R$, the price, the effective size, reachability; `resample` is the systematic resampling inside groups |
| `legacy.py` | `read_legacy` reads a file of the old format (impacts become shifts, conditions become observations) and writes the old causes and the regime event into the narrative as a line; it writes no file |
| `migrate.py` | `migrate` converts a directory of old files, writes each of them in the new format and, where causes were named, the list `<directory>-migration-causes.txt`; `python -m mst.scenario migrate DIR --workspace DIR` |
| `reverse.py` | The reverse problem of the canon (section 6): `Family` with fixed versions and interventions, `sup_rho` under the price $\kappa$ and `inf_dist` over the level $L^\ast$; `python -m mst.scenario reverse` |
| `tool.py` | Operations of the `scenario` tool: `list`, `get`, `save`, `validate`, `reverse` |

Numbers are written in the quantities' own units and in calendar months, never in the model scale and never as horizon month numbers. `level` is the value itself, `change` the move from the last known value in the same units, `percent` a relative move for quantities the registry keeps in `log100`. `translate` maps them to the registry scale and to horizon months counted from the last month of the model panel. Random intervention values (`uniform`, `with_probability`) are refused for now.

`Paths` in `build.py` are $R$ draws over $H$ months: the macro variables and the drivers in original units, the flags of the driver paths (an intervention, parameters from above, approximate data, leaving the training range), the price of the scenario split by level, the effective size of the tilt and the checks of what was imposed. A scenario of interventions alone has no observation, so its price is zero and is written as zero. Position models read the paths in one and the same form.

The checks of `validate`:

- an unknown quantity, by the registry of macro variables and the quantities of the satellite bundle;
- a month before the last panel month or beyond the horizon of the check, with the month named;
- an observation that does not translate into one scale: macro variables together with drivers, mixed scales, a mean on a `log100` quantity, weights of the wrong sum;
- `percent` on a quantity outside the `log100` scale, and a missing last known value to count the change from;
- a repeated (quantity, month) pair and a random law of the value, refused already by `save`;
- satellites of another macro version, with both hashes;
- a corpus fragment named in the scenario and published after the calculation date, if the corpus index is built;
- a missing approval; and, as warnings, no sources, every element on assumptions, or neither observations nor interventions, which is the base $s_0$.

Tests are in [../../../tests/scenario](../../../tests/scenario/README.en.md).
