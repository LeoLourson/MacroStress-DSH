# tests/macro

Tests of the core `mst.macro`: the variable registry, the panel and the version store. [../bvar](../bvar/README.en.md) checks the estimation, identification and conditional forecast of the BVAR; here are the contract of level 0 and how the platform stores and reads versions.

| File | What it checks |
|---|---|
| `test_registry.py` | The registry matches the tables of the formalization, transformations, seasonal variables, the check of scenario names |
| `test_panel.py` | Seasonality, the order of blocks, data on the date, the panel's start |
| `test_version.py` | The mandatory keys of a version, dispatch to the estimating plugin (`meta["model"]`), the panel of a version, a bitwise repeat, `estimate_macro` publishes a version from the store |
| `test_model_tool.py` | The tool `model(kind="macro")`: list, description, `model_id`, jobs on the resource `models`, `kind=catalog` |
