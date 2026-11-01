# Tools and commands

The general scheme is in [../concept/formalization.en.md](../concept/formalization.en.md); this page is the reference of calls and commands.

## MCP tools

The server `macrostress` (`src/mst/mcp.py`) gives eight tools, the prompt `scenario_from_event(event)` and instructions that any client receives when it connects. They send the agent to the `guide` tool before it starts work. An argument error comes back as a call error whose text lists the allowed values. A tool that starts a job returns its id at once, without waiting for the end (`run start` also returns the run id).

### `guide`

| Arguments | Result |
|---|---|
| none | The list of guides: `name`, `description`, `whenToUse` |
| `name` | The text of a guide: `scenario-from-corpus`, `run-report` or `drivers` |

These are the same texts the DSH plugin registers as skills.

### `jobs`

| `action` | Arguments | Result |
|---|---|---|
| `state` | none | A snapshot of the workspace; it also marks jobs without a live process as finished and rewrites `workspace.json` |
| `status` | `id` | The job record (`argv`, `pid`, `started`, `finished`, `cancelled`), `running` and `exit_code` (`null` while running or if cancelled) |
| `log` | `id`, `tail` (50; 0 is the whole log) | The last lines of the log |
| `cancel` | `id` | Stops the job's process tree and releases the lock |

### `series`

| `action` | Arguments | Result |
|---|---|---|
| `search` | `text` | Sources of the built-in and the workspace `sources.toml` whose id or loader contains the text |
| `get` | `id`, `as_of` (today by default) | Series values known on the date: `month`, `value`, `published`, `approximate` |
| `add` | `entry` | Adds a source to `<workspace>/sources.toml`, which is read on top of the built-in `src/mst/data/sources.toml`; the id must not repeat a known one. `entry` is a table of the file plus the field `id` |
| `update` | `ids` (empty means all) | A job on the resource `series`: `python -m mst.data update` |
| `update` | `quantity` | A job on the resource `series`: the quantity's catalog finds the objects, then their series download (`python -m mst.drivers update`) |

`ids` and `quantity` are not accepted together.

### `corpus`

The corpus root is `<workspace>/corpus`. `search` and `get` read the index and need no network.

| `action` | Arguments | Result |
|---|---|---|
| `search` | `query`, `as_of`, `kind`, `region`, `factors`, `k` (8) | The best fragments visible on `as_of`: `chunk_id`, document, heading path, pages, text |
| `get` | `id` (a `chunk_id`), `as_of` | One fragment; one published after `as_of` is an error with the publication date |
| `list` | none | What the manifest holds: sources and their files (address, downloaded, converted), episodes, statuses |
| `add` | `key`, `file`, `url` | Registers a file downloaded by hand in `raw/` and `downloads.json`, as `fetch` does |
| `index` | none | A job on `corpus`: `python -m mst.corpus index` |
| `fetch` | `ids` (document keys) | A job on `corpus`: `python -m mst.corpus fetch` |

### `scenario`

A scenario is `scenarios/<id>.json`; `validate` reads both versions the file names and calculates nothing over the network.

| `action` | Arguments | Result |
|---|---|---|
| `list` | none | Scenarios with versions, horizon, the number of conditions and impacts; a file that fails the contract is named separately |
| `get` | `id` | The scenario file as it is |
| `save` | `id`, `spec` | Checks the contract and writes `scenarios/<id>.json`; the id is lowercase letters, digits, hyphens |
| `validate` | `id` | `ok`, `problems`, `labels` (how many elements of $O$ and $\mathcal C$ are marked with the event, a source and an assumption) and `warnings` (no sources, everything on assumptions, a shift of a macro variable with the quantities that read it, a sector shift with the companies it does not reach, no conditions and so a zero price, neither conditions nor impacts and so the base $s_0$; none of them stops a calculation). `problems` names each problem: an unknown variable or cause, a condition the causes cannot reach, a regime event without a hidden coordinate among the causes, a missing approver, an impact on an unserved driver, satellites of another macro version, a corpus fragment cited by the narrative, a condition or an impact and published after the calculation date (if the index is built) |
| `reverse` | `id`, `portfolio`, `draws`, `seed`, `kappa` or `level` | Searches the scenario's family: the worst loss under the price `kappa`, or the lowest price whose loss reaches `level`; returns the answer and every point of the search |

### `run`

A run is `runs/<id>/`: the request, the meta, the paths, the position contributions and, for a portfolio, the report. The same request with the same seed gives the same numbers.

| `action` | Arguments | Result |
|---|---|---|
| `start` | `scenario`, `draws`, `seed`, `portfolio` or `positions`, `targets` (empty means all drivers), `base_currency` (RUB, for `positions`), `contributions` | Writes the request and starts a job on `runs` (`python -m mst.runs`); returns `run` and `job` |
| `status` | `id` | The run meta (status, scenario, versions, horizon) and the job record |
| `result` | `id` | The meta, the result of each position (status, metrics, reason), the conditional forecast of the target indicators (`forecast.json`: P5/P50/P95 by month, in own units) and the portfolio report: P5/P50/P95, ES95, the price, the coverage |
| `list` | none | Runs, newest first: status, scenario, horizon, the number of draws and positions |

### `portfolio`

A portfolio is `portfolios/<id>.json`: the base currency (the ruble: models calculate against the ruble) and positions (asset type, currency, notional in millions, object codes, the model's `terms`).

| `action` | Arguments | Result |
|---|---|---|
| `list` | none | Portfolios with the base currency and the number of positions; a file that fails the contract is named separately |
| `get` | `id` | The portfolio file as it is |
| `save` | `id`, `spec` | Checks the contract and writes `portfolios/<id>.json` |
| `delete` | `id` | Deletes the file; a missing portfolio is an error |
| `check` | `id`, `scenario` | `{"ok": true, "problems": []}` or each reason by name: problems of the scenario itself (unreachable conditions, a foreign satellite version, corpus fragments after the date), an unknown asset type, `terms` the model does not read, a currency with no exchange rate path, a macro variable missing from the version, an object no driver covers, and a driver rejected by the satellite check, with its reason |
| `exposure` | `id`, `scenario` | The value of each position in the calculation month of the macro version: notional, exchange rate, value in the base currency |

### `model`

| `kind` | `action` | Arguments | Result |
|---|---|---|---|
| `macro` | `list` | none | Macro model versions, newest first |
| `macro` | `describe` | `version` | The version metadata: model, date, variables, shocks and the share of draws that passed the sign restrictions |
| `macro` | `base` | `version`, `horizon` (default 12), `only` | The base path under $s_0$ by month: `p5`, `p50`, `p95` in the model scale; conditions are written against it. It computes no drivers and starts no job |
| `macro` | `estimate` | `as_of`, `model_id` (may be omitted if only one is installed) | A job on `models`: `python -m mst.models estimate-macro` |
| `macro` | `backtest` | `as_of`, `model_id`, `only` | A job on `models`: `python -m mst.metrics backtest`: cells, CRPS, band coverage, admission, the aggregate by level and horizon, and the tuning and validation halves |
| `satellites` | `list` | none | Satellite versions (bundles) |
| `satellites` | `describe` | `version` | Served drivers and rejected ones with the reason |
| `satellites` | `estimate` | `version` (the macro version hash), `assignment` (quantity → model) | A job on `models`: `python -m mst.models estimate-satellites` assembles the bundle, with its own model for each quantity |
| `catalog` | `list` | none | Installed model plugins: macro, satellite, position, and which model serves each quantity |

## Commands

Jobs start the same commands you can run in a terminal from the repository root. Models are estimated only through `python -m`: the entry point `mst.drivers` sets `OPENBLAS_NUM_THREADS=1` before importing numpy, otherwise small dense systems are solved in 20 threads and satellite estimation takes hours instead of minutes.

```powershell
# series
python -m mst.data update [SERIES ...] --workspace DIR

# macro model
python -m mst.models estimate-macro bvar.sv --as-of 2026-10-29 --store DIR --models DIR
python -m mst.models estimate-satellites --macro SHA256 --store DIR --models DIR --drivers FILE [--model QUANTITY=MODEL_ID]
python -m mst.metrics backtest bvar.sv --as-of 2026-10-29 --store DIR --models DIR --corpus FILE [--satellites --drivers FILE]
    [--only ID ...] [--details FILE] [--against REPORT.json] [--target-level LEVEL]
    [--portfolio ID] [--mc-bootstrap]

# drivers
python -m mst.drivers update [QUANTITY ...] --store DIR --drivers FILE

# server
python -m mst.mcp --workspace DIR

# corpus
python -m mst.corpus index --corpus DIR
python -m mst.corpus search QUERY --as-of DAY --corpus DIR

# calculation
python -m mst.runs <id> --workspace DIR

# reverse problem
python -m mst.scenario reverse ID --workspace DIR --portfolio ID --draws N --seed S (--kappa K | --level L)
```

Here `--store` is `<workspace>/series`, `--models` is `<workspace>/models`, `--drivers` is `<workspace>/drivers/catalog.json`, `--corpus` is `<workspace>/corpus/manifest.json` (for `mst.corpus` it is the root `<workspace>/corpus`).

`backtest` prints the tables of episodes, admission, the "level × horizon" aggregate, the value of the scenario, the halves and the quantities: the normalized CRPS of the conditional forecast beside the random walk, the base calculation and the naive forecast, the coverage at $p_5/p_{50}/p_{95}$, sharpness and admission (the band within the admissible area and no wider than the historical range, the probability of a condition in the base calculation above 1 %). With `--against` it prints the verdict by two standard errors on the same cells and the validation half; with `--portfolio` (requires `--satellites`) the Acerbi–Székely tail section ($Z$, intervals, the 1 % trimming, dropped episodes); with `--mc-bootstrap` the Monte Carlo error interval separately from the sampling over episodes; `--target-level` selects the level versions are compared on (`macro` by default). The full report goes to `--details`. Episode versions are cached in `<models>/backtest/` by the state of the data, satellite bundles by a digest of all their inputs. Episode models are written to `<models>/backtest/`, not `<models>/macro/`. Exit code 1 means some episode was not checked (absent from the manifest, short history, an inadmissible scenario) or an admission condition was broken.
