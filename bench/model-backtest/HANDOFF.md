# Resume the model-improvement experiments

## Assignment and current conclusion

The user requested actual iterative model-improvement experiments on a new Git branch, governed by the unchanged [success criteria](../../docs/concept/success-criteria.md). The user's latest request is to leave instructions so an agent without this conversation can continue. Do not start another experiment merely to finish this handoff.

Work is on branch `experiments/model-trust-2026-11-17`, created from `main` at `01d253c`. Do not switch to `main`, overwrite inherited work, change criteria, or push without permission. All experimental jobs have completed. No candidate has been enabled in the shipped specification, merged, or published.

The first experiment series is complete. The `state` candidate (dynamic monthly scale initialized from the latest filtered state) **formally passes the current implemented criteria** on the evaluated data and research portfolio; `baseline` fails the upper macro tail. The production replacement rule returns `replaces=true` for `state` at every evaluated target level. However, the candidate's portfolio tails remain visibly underdispersed. Formal non-rejection is not proof of adequate risk forecasting: the portfolio has 8 VaR95 breaches in 40 available grid windows, instead of the nominal 5% frequency. Preserve this distinction in any report.

The next useful investigation is the satellite predictive residual law, not further arbitrary inflation of macro bands. No satellite model has been changed in this series.

## Environment

- Repository: `C:\Users\LeoLeon\Documents\MacroStress\macrostress-dsh`.
- Python: `.venv\Scripts\python.exe` from the repository root.
- Facts are read as known on `2026-10-29`; each historical model is estimated using only its own calculation-date data.
- The unchanged [episode plan](episodes.toml) supplies 1000 paths, seed 20261029, 12 months after episode start (actual simulation horizons can reach 14 months because of publication lags), quarterly grid, and validation origins from 2021.
- Current DSH policy at handoff: `danger-full-access`, approvals disabled. Do not request sandbox escalation.
- When confined to workspace-write in a future session, Python multiprocessing pipes and temporary directories created with restrictive permissions fail. These were environment failures, not model failures. Do not work around an explicit sandbox denial; follow the session escalation policy.
- Windows long paths broke satellite checkpoint publication under long model roots. Use short roots such as `experiments/mt/bl` and `experiments/mt/sl`.
- Set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` before numerical imports. The [runner](run_experiment.py) does this itself.

Read the repository instructions and load applicable skills (`coding`, `text-hygiene`, graph navigation) before modifying code. Use the MacroStress tools for ordinary workspace state/portfolio inspection. The experimental Python runners deliberately use existing production APIs to read data and write isolated experimental caches.

## Commits and changes

The following commits are already on the experiment branch:

| Commit | Purpose |
|---|---|
| `d6c8a55` | Preserve inherited, initially uncommitted state-scale and central-band changes; original 91 rw tests passed |
| `1ff8648` | Separate the central-band hypothesis behind `band_fit`; inject/copy explicit rw specs and hash them into version identity; add runner and experiment plan |
| `808bcfd` | Record tuning results; add short cache-root option, portfolio-specific report labels, summarizer and harness tests |
| `7e3fdef` | Freeze selected `state` candidate and known risks before reading validation |
| `15fcdb2` | Fix review findings: reject portfolio without drivers, guard report scope/catalog changes, summarize empty reports, preserve stationary starts for old stochastic-scale versions |

The original worktree had seven modified files and six unrelated untracked old result files. The seven modifications were intentionally preserved in `d6c8a55`. The old results in [results](results/) predate the new criteria and are not evidence for this experiment series; they were not staged or overwritten.

Relevant model changes:

- [rw adapter](../../src/mst/plugins/rw/model.py): `RwModel(spec=...)` deep-copies a supplied validated Spec and appends its canonical JSON hash to `version`. This makes the plugin picklable and avoids per-run global configuration edits. Default construction still reads the shipped configuration.
- [rw specification](../../src/mst/plugins/rw/spec.py): `band_fit=False` is an independent experimental switch. It stops inherited band fitting from silently changing every sample-covariance baseline.
- [rw estimator](../../src/mst/plugins/rw/estimate.py): band-based degrees of freedom are used only with `sample_scale and band_fit`.
- [rw forecast](../../src/mst/plugins/rw/forecast.py): inherited dynamic scale starts at the filtered latest state. A missing stored state now loads as NaN, retaining the old stationary-start distribution for old stochastic-scale versions.
- [shipped specification](../../src/mst/plugins/rw/spec.toml) remains `stochastic_scale=false`. The experiment winner has NOT been enabled there.

Before resuming, inspect `git status --short` and the latest commit. The handoff preparation also stages the collector, compact evidence, portfolio specification, journal updates and these instructions in a final checkpoint commit. Do not assume unrelated untracked files belong to this task.

## Experiment variants and selection

The [runner](run_experiment.py) defines these explicit variants:

| Variant | Changes relative to baseline | Tuning grid CRPS / walk | Tuning scenario CRPS / base |
|---|---|---:|---:|
| `baseline` | Student degrees of freedom from joint likelihood; independent monthly scales | 0.944423 | 1.060711 |
| `band` | Fit Student degrees of freedom to central 90% band | 0.940087 | 1.064693 |
| `state` | Filtered-state log-AR(1) scale path | 0.938497 | 1.061434 |
| `band-state` | Both switches | 0.938497 | not calculated |
| `state-no-momentum` | State scale, disable annual variance-ratio momentum | 0.952969 | 1.053961 |

Tuning grid: 28 origins, 8573 identical scored macro cells. Tuning scenarios: 18 episodes, 5091 macro cells. No model-refused windows in these checks. Short housing/output histories remove their own cells with reasons.

`band-state` is not an independent gain: its scores match `state` within roughly 2e-13. In the lognormal scale branch, the Student degrees of freedom cancel against covariance normalization. Do not waste another experiment on the same combination.

`state-no-momentum` was rejected: slight improvement against its own base does not compensate for worse grid and conditional-walk accuracy. It was NOT sent to validation.

`state` was frozen before validation as the best tuning-grid candidate, but its gain was not significant: paired tuning-grid CRPS ratio interval to baseline was [0.9872, 1.0065]. The [journal](journal.md) records the sequence and risks.

## Completed validation and formal verdict

Validation grid: 23 origins, same scored cells for baseline and state, no refused dates.

| Quantity | Baseline | State |
|---|---:|---:|
| Macro CRPS / walk | 0.990276 | 0.979766 |
| Macro nominal-90% coverage across all horizons | 76.02% | 82.44% |
| Portfolio CRPS / historical simulation | 1.041415 | 1.036383 |
| Portfolio nominal-90% coverage across all horizons | 66.33% | 70.67% |
| Three-month VaR95 breaches | 7 / 23 | 6 / 23 |
| Three-month portfolio Z | 5.7496 | 4.6471 |

The paired validation-grid macro ratio is 0.989387, with 95% bootstrap interval [0.980891, 1.007288]; do not call the grid improvement proven.

All 14 validation episodes were computed at macro, driver, position and portfolio levels. For `state`:

- Macro scenario/base CRPS 1.015040, harm-test p=0.292961: no formal refusal, not evidence of positive utility.
- Macro scenario/conditional-walk CRPS 0.981065, p=0.191542: utility is NOT proven on validation.
- Portfolio scenario/base CRPS 0.843267, harm-test p=0.826106.

The full grid has 51 macro origins but only 40 available portfolio origins. The `state` portfolio has Z=4.363831, 8/40 VaR95 breaches; baseline Z=4.821448, 9/40 breaches. The current portfolio criterion does not refuse either at its allocated error share. `state` passes the full implemented admission; baseline fails upper macro tails. Adding more levels changes the multiplicity adjustment, so macro-only admission and all-level admission need not match. This is not a reason to change the criteria or add irrelevant levels.

The [archived evidence](experiment-results/2026-11-17/evidence.json) contains per-target `verdicts`, `replacement`, configurations and compact full-grid reports. It is generated by [collect_evidence.py](collect_evidence.py), which calls production `trust.admission`, `trust.decision`, `trust.verdict` and `decision.compare`; it does not invent a new test. The collector joins disjoint tuning/validation episode rows and rejects duplicate cell keys. It uses the grid's actual block length.

Criteria SHA256: `9745465763fba0951d732f006eb7e337b8d9f7dc7320a8dc3a5035f660842d9f`. Both [success criteria](../../docs/concept/success-criteria.md) and production [trust tests](../../src/mst/metrics/trust.py) are unchanged from `01d253c`.

Validation history was read many times before this series; it is not a closed dataset. This series added one candidate evaluation/read for baseline and one for state, covering grid and episodes; re-aggregation of the same fixed outputs is not a new tuning attempt. No other candidate's validation was opened.

## Portfolios and coverage caveats

The stored AFK portfolio was checked first. On tuning, only MTSS could be priced, on eight grid origins. That is not a test of the complete AFK portfolio. Its local outputs retain all exclusions.

A separate research portfolio `model-trust-liquid` was fixed before evaluating its model scores: equal RUB notionals of LKOH, ROSN, GAZP and SIBN, 25000 each. Its exact [specification](experiment-portfolio.json) is versioned; it was saved through the MacroStress portfolio tool. The compact [liquid catalog](experiment-results/2026-11-17/liquid-catalog.json) includes the necessary ancestors and sector driver. This is an oil-sector research portfolio, not a diversified market sample.

Even the research portfolio has changing historical service coverage: only 17 tuning and 23 validation grid origins have portfolio rows, and some stock positions are excluded in some windows. Existing metrics value the intersection of priced positions at each window. Never call those numbers a test of all four stocks at every historical origin. The compact reports preserve excluded positions and left-out data.

## Where data and evidence live

Versioned compact artifacts: [experiment-results/2026-11-17](experiment-results/2026-11-17/). They include negative tuning candidates, full baseline/state evidence, both episode halves and compact catalogs. Compact artifacts omit raw rows; replacement calculations need the local detailed reports.

Ignored local detailed reports/caches:

- `experiments/model-trust-2026-11-17/<variant>/configuration.json`
- `experiments/model-trust-2026-11-17/<variant>/tuning-rolling.json` and `tuning-episodes.json`: macro-only ablations.
- `experiments/model-trust-2026-11-17/{baseline,state}/{tuning,validation}-rolling-model-trust-liquid.json`
- `experiments/model-trust-2026-11-17/{baseline,state}/{tuning,validation}-episodes-model-trust-liquid.json`
- `experiments/model-trust-2026-11-17/{baseline,state}/all-rolling-model-trust-liquid.json`
- Short cache roots: `experiments/mt/bl` baseline, `experiments/mt/sl` state, `experiments/mt/sn` no-momentum candidate.

Model cache identity includes rw source/spec and explicit Spec hash. Successful macro-cache entries do not hash the whole platform. If platform/satellite code changes, use new isolated roots, rather than assuming old cache entries are valid. Do not modify code while running workers are reading it.

The runner writes `.scope.json` sidecars to reject changed driver-catalog identity. The [summarizer](summarize_experiments.py) skips sidecars and reports null frequencies when there are no scored short cells. Reports created before the scope-guard patch have no sidecars; do not silently overwrite them under another scope.

## Reproduction commands

These are examples for the SAME frozen configurations, not permission to reopen validation while tuning a new candidate. Register a new hypothesis in the journal first. Run from the repository root:

```powershell
# Tuning-only macro ablation; --checks episodes never implicitly opens the full grid.
.\.venv\Scripts\python.exe bench/model-backtest/run_experiment.py state --half tuning --checks rolling episodes --workers 4

# Existing frozen candidate, full grid and research portfolio.
.\.venv\Scripts\python.exe bench/model-backtest/run_experiment.py state --half all --checks rolling --workers 4 --models experiments/mt/sl --drivers experiments/model-trust-2026-11-17/liquid-catalog.json --portfolio model-trust-liquid

# Rebuild compact evidence from finished detailed reports, without new simulations.
.\.venv\Scripts\python.exe bench/model-backtest/collect_evidence.py baseline state

# Read only tuning summaries.
.\.venv\Scripts\python.exe bench/model-backtest/summarize_experiments.py --half tuning --variants baseline band state state-no-momentum
```

The runner process returns zero when the calculation completed, even if the metrics verdict is one. It prints `metrics exit ...` explicitly and preserves the report. An exception/nonzero runner exit is a calculation failure that must be investigated. Do not confuse completion with passing criteria.

For portability to another checkout, restore the saved portfolio through `portfolio(action="save", id="model-trust-liquid", spec=<versioned JSON>)`; read the relevant guide before a portfolio calculation. The catalogs are archived but market series are not committed.

## Tests, review and known limitations

Successful checks:

- Initial inherited rw state: 91 tests.
- After specification isolation: 93 rw tests, 169 metrics tests, four harness tests.
- After review fixes: combined rw + metrics + harness, 270 passed in 117.88 seconds.
- One collector duplicate-cell regression was added afterward; the final harness suite has eight tests, all passed. Together the corresponding suites contain 271 tests (94 rw, 169 metrics, eight harness); the combined 271-test invocation has not been repeated.
- Ruff passes on changed adapter/spec/tests and experiment scripts; `git diff --check` passes.
- New regression tests were observed failing before fixes for constructor/spec isolation, separating band fitting, unknown old scale state, missing drivers and empty summaries.
- Independent read-only reviews found and led to the scope, empty-summary and old-state fixes. Collector joining received a separate read-only review with no concrete joining bug found.

The full repository test suite was NOT run. `coverage`, `pytest_cov`, `radon`, and `mutmut` are not installed, so formal coverage/CRAP/duplication/mutation checks were not run. Do not claim them.

There is an existing Ruff B023 warning in rw momentum's root-finding lambda; the lambda is consumed synchronously inside the loop, and it was not changed in this series. Do not misreport the targeted lint check as all-repository lint success.

The code graph was updated successfully with embeddings once after the model/test edits. A later collector/test addition may require the final graph refresh; check graph HEAD and working-tree coverage before trusting it.

The collector's first real run exposed NumPy bool serialization; it now uses production `backtest.jsonable`, and the second run completed successfully. No model estimation was needed for the retry.

## Next-agent procedure

1. Read this handoff, the unchanged success criteria, [journal](journal.md), and archived evidence. Verify the branch and worktree. Load skills before edits.
2. Do not repeat completed expensive scenarios just to discover state. Inspect existing local details. Finish any final report/commit checks left by the handoff checkpoint before new modeling.
3. Decide whether to enable `stochastic_scale=true` in the experimental shipped spec, with documentation/tests, or leave it an explicit candidate. The production rule favors replacement, but no enablement has been done because visible portfolio tail misses still warrant disclosure. Do not silently merge to main.
4. Start the next hypothesis from tuning only. A promising satellite issue is consistency of the predictive residual law when a driver borrows its parent's coefficients. First establish whether affected stocks actually use fallback on the relevant dates. Simulation uses parent gamma but own sigma; exported residuals use the unused own equation. Inspect [hier simulation](../../src/mst/plugins/hier/simulate.py), [hier adapter](../../src/mst/plugins/hier/model.py), and `residuals()` in [hier estimation](../../src/mst/plugins/hier/estimate.py). A correction must describe errors around the equation actually predicted, including any nonzero residual mean; do not bury center bias in noise variance.
5. Another coherent hypothesis is a fitted common monthly heavy-tailed equity-residual scale. [Bundle shocks](../../src/mst/drivers/bundle.py) are correlated Gaussian; hier already draws posterior sigma/coefficients once per path, but has no common monthly residual scale. If investigated, fit the tail law to historical residuals and preserve covariance; estimation and prediction must use the same law. Independent per-stock Student noise or an arbitrary variance multiplier is not a justified replacement.
6. Tuning diagnostics also show directional and calendar errors: Fed zero-drift misses during hiking phases, credit January bias, residual seasonal patterns in output. Output already receives seasonal adjustment; do not claim it was missing. Each quarter-origin breakdown has only seven tuning origins, so patterns are descriptive, not established mechanisms.
7. Log each hypothesis and tuning result before validation. Keep negative candidates and incomplete coverage. Do not change test thresholds, omit troublesome windows, treat excluded positions as zero, or declare product-quality risk forecasts from non-rejection alone.

## Task-state note

An automatic goal was created for the whole experiment objective. The user has now requested a handoff. It should be paused at the checkpoint rather than continuing unattended; a later explicit request to continue must resume/rearm it. The persistence identifier is `goal-5ff169ee-d137-4ca7-8985-6e1e89249d62`; read the live goal before updating its revision.
