# tests/engine

Tests of `mst.engine` and the built-in position models. Inputs are set by paths and a driver registry assembled in the test, so the expected position value is calculated directly by the formula.

| File | What it checks |
|---|---|
| `fixed.py` | Paths where each driver goes from a given value to a given value: the expected position value is calculated in the head |
| `simulated.py` | A macro version and satellites from known autoregressions: the same quantities position models read |
| `test_dispatch.py` | A driver is found by the position codes (`secid`, then `inn`), a position without a driver is not calculated and names the code, an unknown asset type is rejected by name, `terms` are checked by the schema, a currency with no rate path and a macro variable not in the paths are named, a result of the wrong form and a repeated position id are rejected |
| `test_models.py` | A share equals the price ratio, a foreign share goes by the exchange rate; a bond loses duration × spread widening; a credit loses LGD × the overdue increase; a deposit reads the same model; housing follows the price; quantiles are ordered, the tail goes beyond P95; driver flags reach the result; the same inputs give the same numbers |
| `test_run.py` | The status of a set is the worst of the position statuses, a position with no model keeps its value, results go in position order |
