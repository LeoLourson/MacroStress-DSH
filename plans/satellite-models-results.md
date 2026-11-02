# Results of `plans/satellite-models.md`

The branch `satellite-models` implements the code of every step. The measurements of the
plan — the table of step 1, the gates S1–S6, the verdicts of step 3, the decision of step 4,
the shares of step 6 and the final bundles of step 7 — need the workspace (`workspace/` is
git-ignored: the store, the macro version `5c57…`, the bundle `a08a…`, the scenarios, the
runs), which is not part of the repository this branch was written on. **No number below was
produced by this branch**; every command that produces one is listed where it belongs. The
owner runs them on the workspace and fills the numbers in.

## Before

Not produced. On the workspace:

```powershell
python -m mst.models diagnose-satellites a08a… --models <workspace>/models
python scripts/satellite_gates.py --workspace <workspace> --run <id of the base run of a08a…>
```

`diagnose-satellites` prints, per quantity and per named object, rho, the long-run
elasticities of every input with their posterior 5–95 %, `d*−d0` and the twelve-month pull
(`null` for a quantity in differences — the level has no long run then), the counts of the
check's reasons and the quantiles of the served nodes' RMSE ratio. `satellite_gates.py`
prints S1–S3 and S6 of the base run, the share of paths leaving the training range (the
canon's rule, and beside it the same share with the range widened by a historical
twelve-month move), and the band of each named driver at month 12 twice: the full band and
the band with every macro path fixed at its median, on the same random numbers. S3 is
recomputed from the paths in the model scale of every driver; the recorded share of runs
made before the canon alignment is printed beside it, not trusted — that is the check of the
"recorded vs true driver width share" table of step 1 (bvar.sv 21.9 % recorded vs 7.5 % true,
msvar.markov 24.4 % vs 13.3 %, as the plan states them).

## What the branch changes

1. **Step 1, measure.** `python -m mst.models diagnose-satellites <bundle> --models DIR`
   (`src/mst/models/diagnose.py`) and `scripts/satellite_gates.py`. Tested on the simulated
   data of `tests/drivers/simulated.py` with known answers: the closed form of a hand-made
   bundle (rho, elasticities, `d*−d0`, the pull, reason counts, RMSE-ratio quantiles) and the
   recovery of the true parameters of a simulated model.
2. **Step 2, shares in differences.** `differences = ["share_price"]` in
   `plugins/hier/spec.toml`; `estimate.py` differences `d` and `x` before the sampler and
   records the training bounds in levels; `simulate.py` accumulates the changes from the
   last known level (an older last value and a driver without history keep their start; a
   missing month before the last value starts from the stationary mean of the changes),
   impacts land on the level in exactly their months; `check.py` compares levels against the
   same levels AR(1). A quantity not named is fitted exactly as before — the test pins the
   pre-change draws bit for bit. **The `constant_spread` of the drift ({10, 1, 0.1}) was not
   tuned**: the episodes live in the workspace. It stays 10.0; the tuning half of the
   episodes decides.
3. **Step 3, served shares.** `diagnose-satellites` of the new bundle beside the reference
   gives the count per reason and the gained and lost by name (the reason counts are in
   `checks`, every driver's verdict in `objects`), SBER by its `rmse_ratio`. Not produced:
   no bundle was estimated.
4. **Step 4, the credit channel.** `ru.spread_bbb` is in the `share_price` regressors of
   `src/mst/drivers/registry.toml`. **The keep-or-revert decision was not made**: it needs
   the candidate estimation on the workspace. Keep it only if the `financials` elasticity is
   negative with a 5–95 % interval excluding zero (see the `long_run` of the sector in the
   diagnosis), `vtbr` and `bspb` inherit a negative elasticity, and S5 of the new bundle is
   not worse than step 2's; otherwise revert this one line.
5. **Step 5, what a macro shift reaches.** `portfolio check` warns, per position whose
   driver reads an impacted macro variable, the driver's long-run loading — below 0.1 in
   absolute value it is named as "barely reacts" — and the regressors it loads on most.
   Documented in `tools.md`/`tools.en.md`, the `run-report` guide and its `dsh/skills` copy.
   `bank-admin-run-top5` was not touched, as the plan says. **To the owner:** the BBB shock
   of the scenarios should also move the higher ratings for the AA bond to react; that is
   the scenario's choice, not the model's.
6. **Step 6, the training range.** The gates script reports the share of paths leaving the
   range per quantity and the same share with the range widened by a historical twelve-month
   move on each side. **The canon's rule is unchanged.** The numbers are not produced (no
   workspace); until the owner decides, the report keeps the mark.
7. **Step 7, the final bundles.** Blocked twice over: `plans/macro-models.md` does not exist
   in the repository, and the estimation needs the workspace. The documentation part is
   done: the `hier` README twins state the equation in differences, the platform and tools
   references name `diagnose-satellites`, the CHANGELOG twins carry the entry, and
   `tests/test_docs_translation.py` passes. When the macro models are repaired, estimate per
   final macro version with the options kept above, `draws=2048`, `seed=20261029`, run the
   base and the eleven scenarios, and fill S1–S6 per bundle into this file.

## The gates, when the workspace is there

| Gate | Check | Where |
|---|---|---|
| S1 | Base: median 12-month log return of served shares minus the median of `ru.imoex`, within ±10 log points | `satellite_gates.py` |
| S2 | The same, per named share (`--shares`), within ±35 | `satellite_gates.py` |
| S3 | Driver cells wider than history, recomputed from the paths, below 5 % | `satellite_gates.py` |
| S4 | Served share nodes not fewer than 115 of the present 125 | `estimate-satellites` output / `model(kind="satellites", action="describe")` |
| S5 | Backtest with `--satellites`, driver level, the canon's rule: CRPS not worse by more than two standard errors | `python -m mst.metrics backtest … --satellites --target-level driver/share_price --against <reference report>` |
| S6 | Largest absolute long-run elasticity to `ru.usdrub` among the portfolio shares, below 2.5 | `satellite_gates.py` |

A failed gate is an acceptable result if it is stated with its driver and value; a loosened
gate is not. `a08a…`, `2aec…` and their runs are untouched — nothing in the workspace was
written by this branch.
