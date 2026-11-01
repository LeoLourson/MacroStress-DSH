# tests/drivers

Tests of the core `mst.drivers`: the quantity registry, catalogs, history preparation and the satellite version bundle. [../hier](../hier/README.en.md) checks the hierarchical model itself; here are the data and the assembly of a version from parts of several models.

| File | What it checks |
|---|---|
| `simulated.py` | The generator of the panel, object tree and drivers |
| `test_catalogs.py` | Object names, shares under industry indices, issues under an issuer and a group, Bank of Russia industries by indent and OKVED codes |
| `test_registry.py` | The registry and matching by code, broken trees, transformations |
| `test_bundle.py` | The bundle: its own model per quantity, loading with dispatch, simulation splits drivers between models with their impacts, assignment and ambiguity |
| `test_levels.py` | A chain of three levels with known coefficients, `level2_catalog.json`, impacts between levels, independent shocks of levels, conditions on drivers and their price |
| `test_tool.py` | The tool `model(kind="satellites")` with `assignment` and `series(action="update", quantity=...)` |
| `test_update.py` | Catalog updates and the download of driver series |
