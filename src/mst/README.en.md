# mst

The core of MacroStress: the MCP server, jobs, the workspace and the calculation packages. The only state is the workspace files; each MCP tool parses arguments and calls one package function or starts a job. The whole layout is in [../../docs/concept/architecture.en.md](../../docs/concept/architecture.en.md).

| Module or package | What it does |
|---|---|
| `mcp.py` | The FastMCP server with the tools `jobs`, `series`, `corpus`, `scenario`, `run`, `portfolio`, `model` ([../../docs/reference/tools.en.md](../../docs/reference/tools.en.md)); `python -m mst.mcp --workspace DIR` |
| `plugin.py` | `python -m mst.plugin new --kind position\|satellite\|macro`, `check`, `demo-run`: a model package of any kind, its contract and a demo calculation ([../../docs/howto/plugins.en.md](../../docs/howto/plugins.en.md)) |
| `jobs.py` | Jobs: a child process `python <argv>` under the wrapper `python -m mst.jobs`, which writes the exit code, with a log in `jobs/<id>/`, a resource lock (`series`, `models`, `corpus`, `code`, `runs`), cancellation of the process tree |
| `runs.py` | A run on the paths of a scenario: the request, the meta, the paths, the conditional forecast of indicators (`forecast.json`), position contributions and the report in `runs/<id>/`; `python -m mst.runs <id> --workspace DIR` |
| `store.py` | Workspace directories (`Workspace`), atomic JSON writes, the canonical hash `sha256_of`, the snapshot `workspace.json` |
| `versions.py` | Model versions on disk: a hash over metadata and arrays, writing through a draft, reading with a check of the hash and name, a list without drafts |
| `contracts.py` | The base class of pydantic contracts (`extra="forbid"`, no NaN or infinities) and common types: identifier, hash, horizon, seed, currency, object code, result status |
| `importer.py` | Copying series from a CSV directory (`<source>/data/series/*.csv`) to the workspace |
| [data/](data/README.en.md) | Download and storage of monthly series |
| [macro/](macro/README.en.md) | The macro variable registry, the panel on the calculation date and macro model versions; the model itself is a plugin |
| [drivers/](drivers/README.en.md) | Quantities and drivers: object catalogs, a quantity's history on the panel, the satellite bundle, driver paths |
| [corpus/](corpus/README.en.md) | The source corpus: manifest, download, Markdown, episodes, index and search |
| [scenario/](scenario/README.en.md) | The scenario: contract, check, assembly of paths, overlay and reverse problem |
| [plugin_sdk/](plugin_sdk/README.en.md) | What a position model declares, reads and returns: `Paths`, requirements, the result; `testing.py` is the conformance check |
| [model_sdk/](model_sdk/README.en.md) | The contracts of macro and satellite models: the scenario plan, the macro simulation, version parts, the common conditional machinery |
| [engine/](engine/README.en.md) | The model catalog, input preparation and position calculation without branching by asset type |
| [plugins/](plugins/README.en.md) | Model packages: position, macro and satellite; discovery (namespace, entry points, `MST_PLUGINS_PATH`) and a registry by kind; the reference models are BVAR (`bvar`), MS-VAR (`msvar`), the hierarchical satellite (`hier`) and the second-level regression (`reflevel2`) |
| [models/](models/README.en.md) | The life cycle of versions: estimation of the macro model and the satellite bundle, publication, list, description, plugin catalog |
| [metrics/](metrics/README.en.md) | Version quality on historical episodes: cells $(e, t, q)$, CRPS, benchmarks, admission, the aggregate by level and horizon, the decision by two standard errors, the portfolio tail |
| [portfolio/](portfolio/README.en.md) | The portfolio, its check, position value and the run report |

Dependencies between packages go in one direction: `data` → `macro` → `drivers`; `model_sdk` sits above the contracts; `scenario` needs `macro` and `drivers`, the metrics need `corpus`; `engine` needs `plugin_sdk` and the registry `plugins`; `models` and `metrics` need the registry `plugins`; `runs` needs `scenario`, `engine`, `portfolio` and both model versions. The core names no concrete models: it asks the registry `plugins`, assembled from packages, entry points and the directories of `MST_PLUGINS_PATH`.
