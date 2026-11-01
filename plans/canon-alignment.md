# Plan: bring the code and the scenarios to the canon

Audience: the engineer (a coding agent) who implements this. Every change below is specified; if something here does not match the code, stop and report instead of guessing.

## Goal

The canon is [docs/concept/formalization.en.md](../docs/concept/formalization.en.md). It now requires things the code does not do:

| Canon requires | Code today |
|---|---|
| Risk functionals by month, $\rho_t$, and over the path, $\rho_{\max}$ | One number at the end of the horizon |
| Impacts are arbitrary paths $o^l$ | A constant `delta` on `start..end` |
| Comparison with $s_0$ separates the shift of the centre from the change of width, with error intervals | Not reported |
| A contribution is $\rho(s)-\rho(s')$ with an error interval | No interval |
| One-sided price $\mathrm{dist}^+(s)$ beside the price | Absent |
| A price whose standard error exceeds 10 % of it is "not estimated", and the reverse problem skips it | Absent |
| A calculation breaking admission gets the status "not admitted", and an unverified tail marks the whole report | Checks are written to the meta and ignored; the status stays `ok` |

## Scope

In scope: `src/mst` platform code, the guides, the reference documents, scenario files in `workspace/scenarios`.

Out of scope, do not change:

- `docs/concept/formalization*.md`. If it looks wrong or ambiguous, stop and report.
- The macro and satellite estimation code and specifications: `plugins/bvar`, `plugins/msvar`, `plugins/hier/estimate.py`, `plugins/hier/spec.toml`, `signs.toml`, the variable registry. The only allowed edit in a model plugin is the impact line in `plugins/hier/simulate.py` from step 2.
- Model versions in `workspace/models`. Do not estimate anything.
- Gold, the dollar index, bank spread channels, new plugins.

Do not soften admission to make calculations pass. **All calculations are expected to be "not admitted" at the end, and that is a correct result** (see Acceptance).

## Working rules

- Work on the branch `canon-alignment`, which holds this plan and starts from `main`. One commit per step, in order. Push the branch, merge nothing: the owner's reviewer reads each commit.
- Write the test first and see it fail for the expected reason. Tests assert behaviour, not implementation.
- After each step: `uv run ruff check src tests` and `uv run pytest -q -m "not slow"`. A step is finished only when both pass. Do not skip, weaken or delete an existing test to get there; if an existing test contradicts this plan, stop and report it. Updating a test for the impact format in step 2 is expected.
- Match the surrounding code: docstrings in English, no compatibility shims (the repository removes old formats outright), no new dependencies.
- Commit messages in Russian, as in `git log`. In Russian text never use the word «прогон» or its forms.
- `workspace/` is git-ignored and holds real data. Touch it only in steps 10 and 11.
- Never present a result you did not run. Report failures with their output.

**Stop and report** when: a step needs a file outside its list beyond a mechanical call-site update; a formula here disagrees with the canon; `-m "not slow"` fails on a test you did not touch; a slow test you must run takes over 40 minutes.

## Step 1. Fix `model(kind="macro", action="list")`

`src/mst/macro/version.py`, `_summary`: `meta["model"]` raises `KeyError` on a version whose meta has no `model` (version `a323…` in the workspace). Use `meta.get("model")`, so the entry reports `null`.

Test in `tests/test_versions.py`: a saved version without `model` is listed with `model` equal to `None`.

## Step 2. Impacts are paths

Replace `start`, `end`, `delta` by `start` and `values`.

`src/mst/scenario/contracts.py`, `Impact`, and `src/mst/model_sdk/contracts.py`, `AdditiveImpact`:

```python
target: ...
start: ...          # the first month the impact touches
values: list[...]   # values[k] is added in month start + k; at least one value
```

Add the property `end = start + len(values) - 1` and the method `value_at(month) -> float`, which returns `0.0` outside `start..end`. A gap inside the path is written as `0.0`. Reject an empty `values` and non-finite numbers. The old fields are refused by the contract.

Call sites:

- `src/mst/scenario/build.py`: `_impact_shift` uses `term.weight * impact.value_at(term.month)` with no month filter; the conversion to `AdditiveImpact` near line 316; the application near line 829 becomes `paths[:, impact.start : impact.end + 1, k] += np.asarray(impact.values)`.
- `src/mst/plugins/hier/simulate.py` near line 156: the same.
- `src/mst/plugins/reflevel2`, `src/mst/drivers/bundle.py`, `src/mst/scenario/overlay.py`, `reverse.py`, `tool.py`: run `grep -rn "\.delta\|\"delta\"" src tests dsh docs` and update every hit that is about impacts. Hits about something else stay.
- `_impact_problems` in `build.py` must refuse a path that ends after the scenario horizon, naming the target.

Bump `MODEL_SDK_VERSION` to `3.0.0` in `src/mst/model_sdk/version.py` and the `model_sdk` ranges of the installed plugins, because the contract changes. Check what `tests/model_sdk` and `tests/test_versions.py` expect.

Update `src/mst/guides/scenario-from-corpus.md` (the example and the bullets on `delta`, near lines 92–101) and its copy under `dsh/skills`: a phased measure is now one path, and a constant shift is a repeated value. The rule that overlapping impacts add up stays.

Tests in `tests/scenario/test_spec.py` and `test_build.py`:

- a decaying path on a driver and on a macro variable lands on exactly those months and nowhere else;
- two overlapping impacts add up;
- a condition on an impacted variable holds on the final path when the impact varies by month;
- the old format is refused;
- a path past the horizon is refused by name.

## Step 3. Risk functionals by month and over the path

Canon: $\Delta_t$ is the loss by month $t$, $\rho_t(s)=\rho(\Delta_t)$, $\rho_{\max}(s)=\rho(\max_{t\le H}\Delta_t)$.

Position models read `path[:, -1]` as the end of the horizon (`equity`, `bond`, `credit`, `housing`, `plugin_sdk/fx.py`). So the loss by month $t$ is the same calculation on paths cut at $t$. Plugins do not change.

- `FixedPaths`: add `up_to(t) -> FixedPaths`, which keeps months `0..t` of every macro variable and driver, the flags, `start` and `seed`.
- `src/mst/portfolio/report.py`: add `path_block(monthly: np.ndarray, months: list[str], seed: int) -> dict`. `monthly` is `(R, H)`, the portfolio loss by months `1..H`. The result is:

  ```json
  {"by_month": [{"month": "...", "p05": 0, "p50": 0, "p95": 0, "es95": 0}],
   "path": {"p05": 0, "p50": 0, "p95": 0, "es95": 0, "es95_interval": [0, 0]}}
  ```

  `path` is computed on `monthly.max(axis=1)` by the existing `_loss`. Each `by_month` row reuses `_loss` without the interval fields.
- `src/mst/runs.py`, `execute`: for a portfolio run compute `monthly` with `losses_of(valued(cut).results)` for each `t` in `1..H`, where `valued` receives the cut paths; pass it to the report as `summary["losses_by_month"]`; save it as `path_losses.npz` with the array `losses`. `summary["loss"]` keeps its meaning and its numbers.

If the monthly calculation adds more than 30 seconds to a run on `fifth-test`, stop and report.

Tests in `tests/test_runs.py` and `tests/portfolio/test_report.py`:

- the row of the last month equals `summary["loss"]` exactly;
- an impact on months 2–3 changes those rows and leaves later rows at the values without the impact, while `loss` does not change;
- `path.es95 >= max(row.es95)`;
- losses inside a month add up within a draw: the row is not a sum of position quantiles.

## Step 4. Comparison with $s_0$ and intervals for contributions

Add to `src/mst/metrics/es.py`:

```python
def paired_difference(a, b, *, seed: int, replications: int = 1000) -> dict:
    """Differences of a-b for the median, p95-p5, p95 and ES95, with 95 % bootstrap intervals."""
```

Resample one set of indices per replication and apply it to both arrays: draw `i` of `a` and draw `i` of `b` come from the same random numbers, and the pairing keeps whatever link they have. Never report a difference per draw. Use a generator seeded `[seed, <a stream constant not used elsewhere>]`.

Result shape: `{"centre": {"value": 0, "interval": [0, 0]}, "width": {...}, "p95": {...}, "es95": {...}}`, where `centre` is the difference of medians and `width` the difference of `p95 - p5`.

- `runs.execute`: value `empty` as well, and put `paired_difference(losses, empty_losses, seed=...)` into `summary["comparison"]`, for the horizon loss and for the path maximum. For a scenario with no conditions and no impacts `comparison` is `null`.
- Contributions in `runs.execute`: keep `loss` and `difference`, and add `interval` from the same function, for the same keys it has.

Tests:

- a scenario of impacts only gives a narrow `centre` interval that excludes zero, and the value matches the shift;
- two arrays with the same centre and a narrower `a` give `width` below zero with `centre` near zero;
- the same input and seed give the same output.

## Step 5. Admission of a calculation

Canon: the calculation is admitted if fewer than 5 % of the cells of month and quantity on its paths break each condition. Otherwise the status is "not admitted" with the broken conditions and quantities named, and the numbers stay.

- `src/mst/runs.py`: add `_admission(checks) -> dict`:

  ```json
  {"admitted": false, "limit": 0.05,
   "shares": {"band_outside_domain": 0.0, "band_wider_than_history": 0.2},
   "broken": [{"condition": "band_wider_than_history", "quantity": "driver/...", "share": 1.0}]}
  ```

  `shares` weighs all quantities of `checks["paths"]` equally, macro and drivers together, all 205 drivers included. Every quantity has the same months, so the mean of its shares is the share of cells. `broken` lists every quantity with a positive share, worst first. Name the limit `ADMISSION_LIMIT` and reuse an existing 5 % constant in `metrics` if there is one with this meaning.
- `src/mst/portfolio/report.py`: `PortfolioStatus` gains `"not_admitted"`; `build` receives `admission`. Precedence: `not_admitted`, then `incomplete`, `needs_review`, `ok`. The summary gets `admission` and the flag `not_admitted`. Coverage flags stay.
- Add the flag `tail_unverified` when any position carries `outside_training_range:<driver>`. It does not change the status.
- Add the flag `constraint_rare:<number>` for each entry of `checks["constraint_probabilities"]` below `constraint_threshold`. It does not change the status.
- `_write`: compute the checks before building the report. The meta `status` follows the report. For a run with `positions` and no portfolio there is no report: write `admission` into the meta, and the meta `status` is `not_admitted` when not admitted.

`ResultStatus` in `src/mst/contracts.py` is the status of a position and does not change.

Tests in `tests/test_runs.py`, `tests/portfolio/test_report.py`:

- share `0.0499` is admitted and `0.05` is not;
- a run with a band wider than history has the status `not_admitted`, names the quantity, and still has its `loss`;
- precedence holds when positions are also excluded;
- both flags appear and leave the status alone.

Update every reader of the status: `grep -rn "needs_review" src dsh`.

## Step 6. A price with a large error is not estimated

- `src/mst/model_sdk/dist.py`: add `PRICE_ERROR_LIMIT = 0.10` and `price_estimated(distance, standard_error) -> bool`: true when both are finite and `standard_error <= PRICE_ERROR_LIMIT * distance`.
- `src/mst/portfolio/report.py`, `dist`, and `src/mst/runs.py`, `_dist_meta`: add `estimated`. The report gets the flag `price_not_estimated`, and the number stays visible. The status does not change.
- `src/mst/scenario/reverse.py`: `Point` gains `estimated: bool`. `sup_rho` and `inf_dist`, in the initial filters and in bisection, treat a point with `estimated == False` like a point that does not qualify. `_rows` reports `estimated`. An answer never rests on such a point: when none qualifies, `answer` is `None`, as it is now for an empty set.

Tests:

- the boundary in `price_estimated`, including infinity;
- in `tests/scenario/test_reverse.py`, a family where the cheapest or worst point has a large error is not chosen.

## Step 7. One-sided price

Canon: $A^+$ replaces each interval condition on $y$ by a ray. An interval below the median of $y$ under $P$ and the same $O$ becomes the ray down from its upper bound; an interval above the median becomes the ray up from its lower bound; a condition whose interval contains the median is dropped. It is defined for conditions on the support only.

This does not need a second overlay: `price_of_law` takes the same `ConstraintLaw` with other bounds.

In `src/mst/scenario/build.py`, `dist_of`:

1. Keep the `law`. If the scenario has conditions on means or probabilities (`expectations`) or no constraints, `one_sided` is `None`.
2. `median_of_law(law, rng, samples=64)` in `src/mst/model_sdk/dist.py`: for each coordinate, the weighted median of draws from the marginal normal of every row, weights `exp(log_weight)`. The medians are of the unshifted functionals, and the plan's bounds already have level-0 impacts taken out, so they are on the same footing.
3. `one_sided_box(low, high, medians)` returns `(low, high, kept)`: `high < median` gives `(-inf, high)`, `low > median` gives `(low, +inf)`, else the coordinate is dropped.
4. Slice `law.means[:, kept]` and `law.covs[:, kept][:, :, kept]` and call `price_of_law`. If nothing is kept, the distance and log probability are `0.0`, the error is `0.0`.

Confirm that `log_box_probabilities` in `src/mst/model_sdk/conditional.py` accepts infinite bounds; if it does not, stop and report before changing the sampler.

Use a new stream constant beside `DIST_STREAM`, checked unused. `Dist` gains `one_sided: OneSided | None` with `distance`, `log_probability`, `standard_error`, `dropped: tuple[int, ...]`. The report and the meta expose them with `estimated`, the same rule as step 6.

Tests in `tests/model_sdk` and `tests/scenario/test_build.py`:

- one gaussian coordinate with a known mean: the one-sided price equals $-\log\Phi(\cdot)$ within the sampling error;
- narrowing an interval raises `distance` and leaves `one_sided.distance` unchanged within the error;
- an interval holding the median is dropped;
- `one_sided.distance <= distance` within two standard errors;
- with `expectations`, `one_sided` is `None`.

## Step 8. Warning on short impacts

`src/mst/scenario/build.py`, the `warnings` of validation: for an impact with `end < horizon`, warn that it is absent from the loss at the end of the horizon, and that `losses_by_month` shows it. Test in `tests/scenario/test_tool.py`.

## Step 9. Documents

Update in both languages where both exist, in the existing style, without references to section numbers or to this plan. `tests/test_docs_translation.py` must pass.

- `src/mst/guides/run-report.md` and its copy under `dsh/skills`: the status `not_admitted` and how to read it (the numbers are not an estimate of losses); `losses_by_month`; `comparison`, with the rule that lower ES95 under a condition does not by itself mean lower risk; `dist.estimated`, `dist.one_sided`, and that prices are compared by the one-sided price; the new flags. The guide is in English.
- `docs/reference/tools.md`, `tools.en.md`: `run result`, `scenario validate`, `scenario reverse`.
- `docs/reference/platform.md`, `platform.en.md`: the impact format, the run outputs.
- `docs/howto/plugins.md`, `plugins.en.md`: `AdditiveImpact`, the model SDK version.
- `CHANGELOG.md`, `CHANGELOG.en.md`, under the unreleased section. State that the scenario file format changed and old files are refused.

## Step 10. Scenario files

`workspace/` is not in git, so this step makes no commit. Back up `workspace/scenarios` to `workspace/scenarios.bak-canon-alignment` first.

**Migrate.** Rewrite every file in `workspace/scenarios`: `{start, end, delta}` becomes `{start, values: [delta] * (end - start + 1)}`. Change nothing else. Then `scenario(action="validate")` on all of them, comparing `problems` with those before: they must be the same.

**Aftereffects.** Extend these impacts to month 12. For months $m>\text{end}$ the value is $\text{delta}\cdot2^{-(m-\text{end})/3}$, rounded to 2 decimals; half-life three months, an assumption: keep each impact's `reason` and add to it that the decay is an assumption.

| Scenario | Impacts |
|---|---|
| `fuel-export-ban-2026-11` | All, after month 7 |
| `hormuz-closure-2m` | All, after month 3 |
| `sbp-cyber-outage-10d-2026-11` | The `ozon` impact, after month 2 |

Impacts already ending at month 12 stay as they are.

**Banks.** In `bank-admin-run-top5` add an impact on `share_price:vtbr`: `-10.54` in months 2–4, then the same decay, `basis` `"assumption"`. VTBR is the only top-5 bank the satellites serve, and `share_price:sber` is not served. Mention this in the narrative.

**Opposite sign.** Create `cn-bank-payment-stop-weak-rub-2026-11` from `cn-bank-payment-stop-2026-11`:

- each bound of `ru.usdrub` and `ru.cnyrub` is mirrored around the base median of that variable and month from `model(kind="macro", action="base", version=...)`: $(\ell,r)\mapsto(2m-r,\,2m-\ell)$;
- `causes` is `["ru.fx_risk"]`;
- the narrative states that this is the weaker-rouble outcome, and why both outcomes are kept.

If `validate` or the price reports that the causes cannot reach the event, set `causes` to `[]`, and record that outcome in the results.

**Approval. Stop here and ask the owner.** A person approves a scenario. List the four changed scenarios and the new one with their diffs. Their `approved_by` must be cleared until the owner confirms; do not fill it in yourself. Step 11 needs the confirmation.

## Step 11. Recalculate and write down the results

Portfolio `fifth-test`, `draws=2048`, `seed=20261029`, one after another: `fifth-base`, the 11 scenarios of the check, and the new one, 13 calculations. Runs take 2–6 minutes each; the old runs stay.

Write `plans/canon-alignment-results.md`, in English, one table row per calculation: scenario, run id, status, `admission.shares`, P5/P50/P95 and ES95 at the horizon and over the path, `dist`, its error, `estimated`, `one_sided`, the flags. Below it, the checks of Acceptance with the observed numbers. Do not interpret beyond these checks. Commit this file.

## Acceptance

The reviewer reruns the checks from the saved runs. All must hold:

1. `uv run ruff check src tests` and `uv run pytest -q -m "not slow"` pass, the CI of the branch is green, and the slow suite `uv run pytest -q -m slow` passes once at the end.
2. Each of the 13 calculations has a correct status, recomputed independently from `meta.json["checks"]`.
3. Expected, given that models do not change: **all 13 are `not_admitted`**, because at the last check the driver bands were wider than history in about 22 % of driver cells even in the base. If any is admitted, that is a finding to explain and not a success to report.
4. Every report carries `tail_unverified`, since all drivers carry `outside_training_range` in the current bundle.
5. `fuel-export-ban-2026-11`: `losses_by_month` differs from the base in months 2–7, and more mildly after.
6. `hormuz-closure-2m` and `cbr-cut-5pp-unscheduled-2026-11` have `estimated == false` if their errors stay above 10 %, as they were before: 71 % and 25 %.
7. `ru-migrant-ban-2026-11`: `one_sided.distance` is well below `distance`, with the dropped conditions listed.
8. In the old-format comparison, `summary["loss"]` of `dividend-freeze-payout-a` and `-b` equals the saved run's numbers: their impacts are unchanged. A difference is a defect of step 2 or 3.
9. Reverse problem: `scenario(action="reverse")` on one scenario returns an answer that rests only on estimated points, or `None`.
