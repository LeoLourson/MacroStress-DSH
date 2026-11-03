# Plan: the macro model `expert.elicited`

Audience: the engineer (a coding agent) who implements this. The reference is [src/mst/plugins/expert/README.en.md](../src/mst/plugins/expert/README.en.md); where this plan and the README disagree, stop and report. Every step names its files, functions and tests; if the code does not match what a step assumes, stop and report instead of guessing.

## Goal

A level-0 macro plugin `expert` with two models, `expert.elicited.rw` and `expert.elicited.market`, whose law comes from answers the MCP agent writes. Done means:

- the agent can list the questions of a date, read one, write an answer and see what is missing, all through MCP, and a malformed answer is refused with a message that names the field;
- `model(kind="macro", action="estimate", model_id="expert.elicited.rw", as_of=...)` builds a version from the stored answers and the series store, or refuses with the variables where the residual covariance is negative;
- the version simulates the base, imposes a scenario and prices it through the platform exactly as `bvar.sv` does;
- every step below is covered by tests on synthetic answers. No LLM is called, and no real answers are written.

## Decisions to confirm before code

The README leaves three things open that change the code. The owner decided on 2026-11-03: D1 as proposed, D2 option (a), D3 as proposed. Implement them as written below.

**D1. The mean of the shocks in the `rw` base path.** The README subtracts the shocks' share from the covariance of history ($\hat\Sigma_{\Delta x}-\Sigma_{\text{shocks}}$) but not from its mean: $b_t=x_0+t\hat\mu$, where $\hat\mu$ already holds the average movement the shocks made in history. If the expert's shocks have a nonzero mean (inflation shocks mostly upward), the `rw` variant counts that mean twice, as the covariance would be counted twice without the subtraction. Proposal: $b_t=x_0+t(\hat\mu-\mu_{\text{shocks}})$, with $\mu_{\text{shocks}}$ the mean monthly change the shock stream makes, computed by the same simulation as $\Sigma_{\text{shocks}}$. The equality with `random_walk` at $\lambda_j=0$ still holds.

**D2. Causes that name shocks but not the residual.** The canon fixes $R$ = every coordinate the causes do not name. When a scenario names `ru.monetary`, the residual $u$ and the other shocks are in $R$, and given $R$ the constraint functionals are a deterministic, nonlinear function of the named shock's occurrences, sizes and copula levels. That law is not a mixture of gaussian rows, so `ConstraintLaw` cannot carry it, and its box probability can be far below what a plain Monte Carlo sees. Two options:

- (a) subset simulation (Au, Beck, 2001, Probabilistic Engineering Mechanics 16): a Markov chain in the standard normal space of the named shock's inputs that estimates a small box probability level by level and leaves samples inside the box. It gives both the price and the imposition. The model SDK gains an optional method `MacroModel.log_box_probability(model, plan, low, high, *, draws, seed)` that the platform calls instead of the law when it returns a value, for the two-sided and the one-sided price; medians for the one-sided price come from the model's unconstrained paths. This is the README's "MCMC" made specific, and it touches `mst.scenario.build` and `mst.metrics.rolling`.
- (b) for such scenarios `constraint_law` returns `None`: the platform reports no price, and `simulate` imposes by subset simulation. Smaller, but those scenarios have no price, which the canon-alignment plan treats as "not admitted".

Decided: (a). Step 10 implements it.

**D3. A cause names a shock for every month.** The README says a scenario may name a shock together with its month, but `Scenario.causes` holds bare names and the plan carries no months. Proposal: a cause `ru.monetary` puts $\varepsilon_{\text{ru.monetary},\tau}$ of every month $\tau$ into $D$, and the README sentence is corrected. A cause per month would change the scenario contract and every macro model.

Besides these, the plan adds `residual` to the shock set of a version (the $J$ set the platform checks causes against): a scenario that names it puts $u$ into $D$, which is the case the platform's gaussian machinery prices exactly (step 9). With no causes at all, $D$ holds every shock and the residual, as the README says.

## Scope

In scope: `src/mst/plugins/expert/`, the additions to `src/mst/model_sdk/contracts.py` and `version.py`, `src/mst/models/__init__.py` and `cli.py`, `src/mst/mcp.py`, a new `src/mst/inputs.py`, `src/mst/store.py`, `src/mst/corpus/manifest.py` and `tool.py` for the model corpus, one new guide in `src/mst/guides/` and its copy in `dsh/skills/`, the reference documents named in step 11, tests.

Out of scope:

- `docs/concept/formalization*.md`; if a step seems to need a change there, stop and report;
- the other plugins (`bvar`, `msvar`, `hier`, `rw`) and the variable registry;
- filling the model corpus with documents (episodes, CBR decisions, market data): this plan builds the place and the search, not the content;
- any real answer in `workspace/inputs`, any estimate on the real workspace, any backtest of the new models: there are no answers yet.

## Working rules

- Work on `main` locally, one commit per step, in order. Do not push; the owner pushes.
- Write the test first and see it fail for the expected reason. Tests assert behaviour, not implementation.
- After each step: `uv run ruff check src tests` and `uv run pytest -q -m "not slow"`. A step is finished only when both pass. Do not weaken or delete an existing test; if one contradicts this plan, stop and report.
- Docstrings and the text the agent reads (questions, guide) in English; commit messages in Russian, as in `git log`. In Russian text never use the word «прогон» or its forms.
- No new dependencies: numpy and scipy are enough.
- A long computation reports its progress with `mst.model_sdk.checkpoint.report` and keeps finished units with `checkpoint.save`, so an interrupted estimate resumes.
- After the last step, update the code graph once (the command in `CLAUDE.md`).

**Stop and report** when: a step needs a file outside its list beyond a mechanical call-site update; a formula here disagrees with the README; `-m "not slow"` fails on a test you did not touch; an estimate on the synthetic fixture takes over 5 minutes.

## Step 1. Model inputs in the SDK

The estimate of this model reads answers that are neither series nor registry, and the SDK context has no place for them. Add one, generic, so the core never names the plugin.

`src/mst/model_sdk/contracts.py`:

- `MacroEstimateContext` gains `inputs: Path | None = None`, the folder of the plugin's inputs in the workspace; `checkpoints: Path | None = None`, with the same meaning as in `SatelliteEstimateContext`; and `fragment: Callable[[str], bool] | None = None`, whether a fragment of the model corpus with this id is visible on `as_of`. The platform fills `fragment` from `mst.corpus.tool.get` on the model corpus, so a plugin checks sources without importing the corpus.
- `MacroModel` gains three optional methods with defaults that declare no inputs:
  - `input_keys(self, as_of: date) -> tuple[str, ...]`: the inputs the model needs for a date; default `()`;
  - `question(self, context: MacroEstimateContext, key: str) -> dict`: what the agent reads before answering, `{"key", "text", "schema"}` with the JSON schema of the answer; default raises `KeyError`;
  - `check_input(self, context: MacroEstimateContext, key: str, payload: dict) -> dict`: the answer in its normal form, or `ValueError` naming the field; default raises `KeyError`.

`src/mst/model_sdk/version.py`: `MODEL_SDK_VERSION = "3.1.0"` (a minor release only adds).

`src/mst/models/__init__.py`, `estimate_macro`: new keyword `inputs: Path | None`; it passes `inputs / <plugin_id>` and `checkpoints = models / ".checkpoints" / "macro" / <model_id>` to the context and removes the checkpoint folder after `macro_version.save`, as the satellites do. `src/mst/models/cli.py`: `estimate-macro --inputs DIR`. `src/mst/mcp.py`, `_macro_argv`: passes `ws.inputs`.

`src/mst/store.py`, `Workspace`: property `inputs = root / "inputs"`.

Tests, `tests/model_sdk/test_contracts.py` and `tests/models/`:

- a macro model that declares nothing has `input_keys(...) == ()`, and its `question` and `check_input` raise `KeyError`;
- `estimate_macro` hands a fake macro model the folder `<inputs>/<plugin_id>` and a checkpoint folder, and the checkpoint folder is gone after the version is saved;
- `bvar` and `msvar` estimate as before (their existing tests stay green).

## Step 2. Inputs in the platform and the MCP tool

`src/mst/inputs.py` (new), pure file work, no model logic:

- `path(root, plugin_id, as_of, key) -> Path`: `<root>/<plugin_id>/<as_of>/<key>.json`;
- `write(root, plugin_id, as_of, key, normal: dict) -> str`: an atomic write (temporary file and `os.replace`) of the normal form with `written_at`; returns its SHA-256;
- `read(root, plugin_id, as_of) -> dict[str, dict]`;
- `status(root, model, as_of) -> dict`: for every key of `model.input_keys(as_of)`, written or missing, with the hash and time.

`src/mst/mcp.py`, tool `model`: a new kind `inputs` with actions `list` (the keys and their status), `question` (`model.question`), `write` (`model.check_input`, then `inputs.write`; the result is the normal form's hash and what is still missing) and `status`. Parameters `key` and `payload` are added to the tool; `model_id` and `as_of` are required for this kind. The tool's description says the inputs are written by the agent and that no human approves them.

`src/mst/guides/model-inputs.md` and the identical `dsh/skills/model-inputs.md`: the work order for any model with inputs: `list`, then for every key `question`, the corpus search with `corpus="model"` and the same `as_of`, `write`, and `status` until nothing is missing; then `estimate`. Model-specific instructions are in the question text, not in the guide. `INSTRUCTIONS` in `mcp.py` gains one sentence pointing to the guide.

Tests, `tests/test_inputs.py` and `tests/test_mcp.py`:

- a write followed by a read returns the normal form; a second write replaces it and changes the hash; an interrupted write (the temporary file left behind) does not count as written;
- `write` with a payload the fake model refuses returns the model's message and writes nothing;
- `status` lists exactly the model's keys, missing first;
- the guide copies are equal (the existing test of the guides covers the new file once it is listed).

## Step 3. The model corpus

The README gives the model a corpus of its own, so the expert does not read scenario sets while it sets the base law.

- `src/mst/store.py`, `Workspace`: `model_corpus = root / "corpus-model"`.
- `src/mst/corpus/manifest.py`: `Kind` gains `"market"` (market expectations: futures, forwards, the OFZ curve, the consensus survey).
- `src/mst/mcp.py`, tool `corpus`: parameter `corpus: "scenario" | "model"`, default `"scenario"`, which picks `ws.corpus` or `ws.model_corpus` for every action. `_corpus_argv` passes the chosen root.

Tests, `tests/test_corpus_tool.py`: a fragment indexed in the model corpus is found with `corpus="model"` and not with the default, and the reverse; `as_of` hides a fragment published after it in the model corpus too.

## Step 4. The plugin skeleton

`src/mst/plugins/expert/`:

- `shocks.toml`: the 18 shocks of the README, each with `id`, `meaning` (the README's text, in English) and `size_unit` (what one unit of size is: "percent change of the Brent price", "percentage points of the key rate beyond the rule", and so on). The unit is part of the question; without it sizes of different answers are not comparable.
- `spec.toml` and `spec.py` (`Spec`, a `Contract` like `bvar.spec`): `family: "pareto" | "skew_t"`, `alpha: float > 2` (default 4), `levels: int` (the level grid of the tables, default 513), `stream_draws: int` (simulated months for $\Sigma_{\text{shocks}}$ and $\mu_{\text{shocks}}$, default 200 000), `candidates: int` (default `LATENT_CANDIDATES`), `subset_samples: int` (default 2000, step 10), `history_start: date`, `seed: int`.
- `model.py`: `ExpertModel(MacroModel)` with `model_id`, `label`, `description`, `base: Literal["rw", "market"]`; `input_keys` returns the 18 shock ids, plus `"market"` for the `market` variant. `estimate`, `load`, `simulate` and `constraint_law` delegate to the modules of the later steps and raise `NotImplementedError` until then.
- `plugin.py`: `Plugin(plugin_id="expert", model_sdk=">=3.1,<4.0", macro=(ExpertModel("rw"), ExpertModel("market")))`.
- `__init__.py`.

Tests, `tests/expert/test_plugin.py`: `python -m mst.plugin check expert` passes; the registry lists both models; `models.selected("expert.elicited.rw")` picks the right one; `input_keys` differ by `"market"` only. `src/mst/plugins/README.md` and `.en.md`: a row for `expert` in the table.

## Step 5. The answer contract and the questions

`src/mst/plugins/expert/answers.py`:

- `Quantiles(Contract)`: `q5 <= q50 <= q95`, finite.
- `ShockAnswer(Contract)`: `shock`, `frequency: float >= 0` (per year), `size: Quantiles` (signed, in `size_unit`), `responses: dict[variable, dict[horizon, dict[size_node, Quantiles]]]` with horizons exactly `1, 3, 6, 12`, size nodes exactly `q5, q50, q95`, every variable of the registry present; `rho: float` in `[0, 1]`; `nonlinearity: str` (where and why the response departs from proportionality, non-empty); `sources: list[str]` (fragment ids of the model corpus, may be empty).
- `MarketAnswer(Contract)`: `values: dict[variable, MarketPath]`, `MarketPath` with `horizons: dict[1|3|6|12, float]` in the model scale and `fragment: str`; only registry variables.
- `linear(answer) -> bool`: the README's rule: for every variable, horizon and quantile, the ratio of the responses at two size nodes equals the ratio of the sizes within `1e-6` relative. A size node of zero size is skipped.
- `check(context, key, payload) -> dict`: parse, then refuse a linear answer with the message the question asks to correct; for `market` and for the `sources` of a shock check every fragment id with `context.fragment`; a context without `fragment` refuses answers that cite fragments, so the check never passes silently.
- `question(context, key) -> dict`: the text of the question in English for a shock (its meaning, size unit, the variables with their unit and model scale and the last value on `as_of` from `context.store`, the four horizons, the instruction that the month of the shock is the first, that sizes are the 5/50/95 % quantiles of its own size law, that proportionality is not required and must be explained where broken, and how `rho` is read) and the JSON schema of `ShockAnswer`; for `market`, the same for `MarketAnswer`.

`src/mst/plugins/expert/model.py`: `question` and `check_input` call these.

Tests, `tests/expert/test_answers.py`, on a fixture `tests/expert/fakes.py` that writes 18 consistent synthetic answers (a nonlinear response, one one-sided shock, one shock with zero frequency) and a market answer:

- the fixture passes; each of these is refused with the field named: a crossed quantile, a missing variable, a horizon 2, `rho = 1.5`, a negative frequency, a proportional answer, a market fragment that does not exist on `as_of`;
- the question of a shock names its size unit and the last value of every variable on `as_of`, and nothing published after `as_of`.

## Step 6. Laws from three quantiles

`src/mst/plugins/expert/laws.py`, pure functions on arrays:

- `pareto_quantile(q5, q50, q95, p, alpha)`: the README's formula, vectorised over cells and levels;
- `skew_t_fit(q5, q50, q95, nu) -> (loc, scale, shape, miss)`: Azzalini's skew $t$ with `nu` degrees of freedom, fitted by `scipy.optimize.least_squares` on the three quantiles; `miss` is the largest error as a share of `q95 - q5`;
- `skew_t_quantile(loc, scale, shape, nu, p)`: by inverting the CDF on the level grid;
- `table(q5, q50, q95, spec) -> (levels, values, misses)`: the quantile table of every cell on the level grid (uniform in the normal score between 0.0005 and 0.9995); a cell with `q5 == q95` is a constant.

Memory check: 18 shocks × 3 sizes × 4 horizons × 23 variables × 513 levels × 8 bytes ≈ 20 MB; months between horizons are interpolated on values, not tabulated.

Tests, `tests/expert/test_laws.py`:

- `pareto` passes through the three quantiles exactly, its density does not jump at 5 and 95 %, and its tail follows $p^{-1/\alpha}$;
- `skew_t` on a symmetric answer misses by less than 1e-6 of the width, on a skewed one by less than 1 %, and on an extreme skew reports a miss above 1 %;
- a constant cell gives the constant at every level; tables are monotone in the level.

## Step 7. The shock stream and the responses

`src/mst/plugins/expert/stream.py`:

- `Inputs`: the standard normal inputs of a shock's months, `(draws, H, 2 + n + 1)`: occurrence, size, the copula's gaussian vector and its chi-square scale. Every random quantity of a shock comes from these through quantile transforms, so step 10 can move them by a Markov chain.
- `contributions(tables, inputs, horizon) -> (draws, H + 1, n)`: occurrence when $\Phi(z_0) < 1-e^{-\lambda/12}$; size from the size table; levels $v_i$ from the $t$ copula with correlation $\rho_j$ between all pairs and $\nu=\alpha$; the response of variable $i$ at size node and horizon node at level $v_i$ or $1-v_i$ by the sign of the cell's median; the broken line in size through $(0,0)$ and the three nodes, constant beyond the outermost; linear in $k$ between $k=0,2,5,11$; summed over months of occurrence.
- `draw(tables, rng, horizon, draws) -> Inputs` for every shock.

Tests, `tests/expert/test_stream.py`:

- the share of months with an occurrence matches $1-e^{-\lambda/12}$ within three standard errors; $\lambda=0$ gives zero contributions;
- at $\rho_j=1$ all variables take one level; a variable with a negative median moves away from zero when the level rises;
- a size beyond the 95 % node gives the 95 % node's response; a size of the opposite sign of a one-sided shock gives zero;
- at $k=1$ the response is the mean of the $k=0$ and $k=2$ values;
- the same seed gives the same contributions bit for bit.

## Step 8. Estimate and load

`src/mst/plugins/expert/estimate.py`:

1. Read the answers from `context.inputs / as_of` and refuse with the missing keys if any is absent. Hash them (sorted keys, the normal forms).
2. Build the panel with `mst.macro.panel.build_panel` from `spec.history_start` to `as_of`; take $x_0$, $\hat\mu$ and $\hat\Sigma_{\Delta x}$ from the monthly changes, a change counting only when both ends are known, as in `random_walk`.
3. Build the tables of every shock (step 6), one shock per checkpoint unit, keyed by the answer hash of that shock and the spec; `report` after each shock. Collect the cells whose `skew_t` fit missed by more than 1 % into `meta["fit_misses"]`.
4. Simulate `spec.stream_draws` months of the stationary stream (responses up to $k=11$, constant beyond) for $\Sigma_{\text{shocks}}$ and $\mu_{\text{shocks}}$.
5. $\Sigma_u=\hat\Sigma_{\Delta x}-\Sigma_{\text{shocks}}$; if its smallest eigenvalue is below $-10^{-10}$ times its largest, raise `ResidualNotAdmitted` with the variables whose diagonal is negative or, if none, the variables of the eigenvector of the negative eigenvalue with weight above 0.1.
6. The base path: `rw` by D1; `market` by the README's formula from the market answer, linear between horizons, and the `rw` path for variables without a market value.
7. Return `MacroEstimate` with `meta`: `as_of`, `ids`, `months`, `shocks` (18 ids and `residual`), `base`, `family`, `alpha`, `answers` (the hash of every key), `fit_misses`, `spec`; `arrays`: `panel_y`, `panel_approximate`, `x0`, `base_path`, `chol_u`, the tables and the shock parameters.

`load` rebuilds them into a frozen dataclass `Expert`.

Tests, `tests/expert/test_estimate.py`, on a synthetic series store (`tests/macro` has the pattern):

- with every frequency zero, $\Sigma_u=\hat\Sigma_{\Delta x}$ and the base equals $x_0+t\hat\mu$: the model is the benchmark;
- answers that attribute more variance than history are refused, and the message names the variable;
- changing one answer changes the version hash; the same answers give the same version;
- an interrupted estimate (an exception after the third shock) resumes from the checkpoint and gives the same version;
- the `market` variant without the market answer is refused with the missing key.

## Step 9. Simulate and price when the residual is among the causes

Given everything the shocks make (occurrences, sizes, levels), the path is gaussian: $y=b+c+u$ with $u$ a gaussian random walk. This is the structure of `msvar.markov`, with the shock inputs in the place of the regime path, and the same platform machinery applies.

`src/mst/plugins/expert/forecast.py`:

- `_split(plan)`: $D$ from `plan.causes` (empty means every shock and `residual`); this step handles plans where `residual` is in $D$, and raises `NotImplementedError` for the rest until step 10.
- `constraint_law(model, plan, draws, seed)`: `draws` groups; in every group the inputs of the shocks outside $D$ are drawn once and held, and `spec.candidates` candidates of the shocks in $D$ are drawn; every candidate is one row with mean $W(b+c)$ and covariance $W\Sigma_{\text{path}}W^\top$, where $\Sigma_{\text{path}}$ is the covariance of the random walk's path; `log_weight` zero.
- `simulate(model, plan, draws, seed)`: for every group and every one of `plan.inner` repetitions, pick a candidate by its box probability (`mst.model_sdk.conditional.pick_latent` with `log_box_probabilities`), then draw $u$ given the box (`draw_driving` with the Cholesky factor of the path); raise `InfeasibleScenario` when a constraint is a combination of earlier ones. `group` marks the repetitions of one group. `diagnostics` carries the effective size of the candidates as `msvar` does.

Tests, `tests/expert/test_forecast.py`:

- with no constraints and every frequency zero, the paths' mean and standard deviation per variable and month match `random_walk` within Monte Carlo error;
- every simulated path meets every constraint; an expectation condition moves the mean (through the platform's overlay, as `tests/msvar` checks it);
- with every frequency zero and one interval constraint, the price equals $-\log$ of the gaussian interval probability within three standard errors;
- a shock with a large size makes a scenario of a large move cheaper than the same scenario with that shock's frequency set to zero;
- the same seed gives the same paths and price.

## Step 10. Causes that name shocks only (D2, option a)

`src/mst/plugins/expert/subset.py`: subset simulation (Au, Beck, 2001) on the standard normal inputs of the shocks in $D$, with the shocks outside $D$ and $u$ held at their values in $R$:

- the performance function $g$ is the largest violation of the box over the constraints, each divided by its interval width (or tolerance);
- level probability $p_0=0.1$, `spec.subset_samples` samples per level (default 2000), the modified Metropolis sampler of Au and Beck with unit proposal spread, at most 20 levels; when $p_0$ of the samples reach $g\le 0$ the estimate is $p_0^{m}$ times the final share;
- returns the log probability, its coefficient of variation (Au, Beck, formula for correlated chains) and the samples inside the box.

`src/mst/model_sdk/contracts.py`: optional `MacroModel.log_box_probability(model, plan, low, high, *, draws, seed) -> np.ndarray | None` (one value per group of $R$; default `None`). `src/mst/scenario/build.py` (`_dist_of` and `_one_sided_of`) and `src/mst/metrics/rolling.py`: when the method returns a value, the price and its error are the mean and the spread of $-\log P(S\mid R)$ over the groups, as `price_of_law` reads them; the medians of the one-sided price come from the model's unconstrained `simulate` with the same causes. The plugin's `constraint_law` returns `None` for these plans.

`simulate` for these plans: per group, the samples inside the box from subset simulation, resampled to `plan.inner` repetitions.

Tests, `tests/expert/test_subset.py` and `tests/scenario/`:

- on a one-dimensional gaussian $g$ with a known tail probability of $10^{-6}$, the estimate lies within three of its own standard errors and the samples are inside the box;
- for an expert model whose named shock occurs with probability one and moves one variable linearly, the price matches the closed form;
- a fake macro model that returns `log_box_probability` is priced by it, two-sided and one-sided, and a model that returns `None` is priced by its law as before (the existing tests of `build.py` stay green).

## Step 11. Documents

- `src/mst/plugins/expert/README.md` and `.en.md`: the inputs path `workspace/inputs/expert/<date>/<key>.json`; `residual` in the set of causes; a cause names a shock for every month (D3); the imposition section names the two cases (residual among the causes: candidates and the gaussian residual; otherwise subset simulation); D1 in the base path.
- `docs/reference/platform.md` and `.en.md`: model SDK 3.1 (`inputs`, `checkpoints`, the input methods, `log_box_probability`), the `model(kind="inputs")` tool, the model corpus.
- `docs/howto/plugins.md` and `.en.md`: how a model declares inputs.
- `tests/test_docs_translation.py` keeps the pairs aligned; run it.

## Step 12. Close

`uv run pytest -q` (the slow tests included), `uv run ruff check src tests`, then the code graph update from `CLAUDE.md`. Report the run times and anything skipped.
