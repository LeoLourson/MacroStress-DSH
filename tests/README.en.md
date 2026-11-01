# tests

pytest tests; none goes to the network: source answers are substituted, and models are checked on data generated from known parameters.

```powershell
.venv\Scripts\python.exe -m pytest -q
# without the heavy model estimation
.venv\Scripts\python.exe -m pytest -q -m "not slow"
```

`test_estimate.py`, `test_forecast.py` and `test_identify.py` run the sampler and take minutes; they are marked `slow` and are skipped if the model has not changed.

| File or folder | What it checks |
|---|---|
| `conftest.py` | Common fixtures: a macro version with satellites per session, a workspace with versions and a scenario |
| `test_store.py` | The canonical hash, atomic JSON writes, the workspace snapshot |
| `test_versions.py` | The version hash has not changed, reading and refusing a changed version, a name that is not a hash, drafts |
| `test_jobs.py` | Jobs: start, log, exit code, refusal of a busy resource, release after the end, cancellation |
| `test_mcp.py` | The tool `jobs`: snapshot, update of `workspace.json`, status, log, cancel, refusal without `id`; the arguments of `scenario(action="reverse")` |
| `test_series_tool.py` | The tool `series`, import of series from a CSV directory, a download job |
| `test_corpus_tool.py` | The tool `corpus`: `search`, `get` and `list` read the index, `index` and `fetch` start a job |
| `test_runs.py` | `mst.runs`: request, meta, paths and results in `runs/<id>/`, the tool `run` |
| [data/](data/README.en.md) | Series loaders and the store |
| [macro/](macro/README.en.md) | Registry, panel, versions, the tool `model` |
| [bvar/](bvar/README.en.md), `msvar/` | Estimation, identification, forecast and the law of the conditions of the macro models |
| [hier/](hier/README.en.md) | The hierarchical satellite |
| [drivers/](drivers/README.en.md) | Catalogs, the bundle, levels, the tool `model(kind="satellites")` |
| [corpus/](corpus/README.en.md) | Manifest, download, conversion, episodes, index and search |
| [scenario/](scenario/README.en.md) | The scenario contract, the assembly of paths, overlay, reverse problem, the tool `scenario` |
| [metrics/](metrics/README.en.md) | Cells, backtest, rolling check |
| `model_sdk/` | Conditional machinery, price of a law, overlay of a sample of paths, contracts |
| [engine/](engine/README.en.md) | The dispatcher, position models, statuses and the result set |
| [plugins/](plugins/README.en.md) | Plugin discovery, registry, conformance and the core's boundaries |
| [portfolio/](portfolio/README.en.md) | The portfolio, its check, the report and the portfolio run |
