# tests/scenario

Tests of `mst.scenario`. The macro version and satellites for the checks are assembled from known models, so scenario conditions can be checked on the paths to machine precision.

| File | What it checks |
|---|---|
| `simulated.py` | A macro version and its satellites from known models, the common fixture `versions` |
| `test_spec.py` | The scenario file and contracts: a condition with weights and bounds, an impact, refusal of a month beyond the horizon and of `low >= high` |
| `test_build.py` | `build` keeps the given conditions, converts variables to original units, calculates drivers on the same draws, a repeat with the same seed gives the same numbers, the price of a scenario, conditions on means and probabilities, `validate` names each infeasible combination |
| `test_overlay.py` | The tilt against the closed form of a normal shift, a probability condition, inactive and broken conditions over groups, resampling keeps the law of $R$ |
| `test_reverse.py` | The search on a family with a closed answer, a boundary between grid points, a family whose impacts move, a point of the family, the command line |
| `test_tool.py` | The tool `scenario`: `save` writes a file, `list` and `get` read, `validate` answers by versions and corpus fragments |
