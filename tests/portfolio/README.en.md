# tests/portfolio

Tests of `mst.portfolio` and the portfolio run. Paths, model versions and the driver registry are assembled in the test, so expected losses are calculated by the same formulas as in the position models.

| File | What it checks |
|---|---|
| `conftest.py` | The positions of a saved portfolio for the checks |
| `test_spec.py` | A portfolio is a name, a base currency and positions; a repeated position id and a base other than rubles are rejected |
| `test_tool.py` | `save`, `get`, `list`, `delete` and `check`: each refusal reason by name, `exposure` by the panel of the macro version |
| `test_report.py` | The losses P5/P50/P95 and ES95 by the sum within a draw, an excluded position with a reason, coverage and unmeasurable value, the statuses `ok`/`needs_review`/`incomplete`, the scenario price and its absence without conditions |
| `test_run.py` | `mst.runs` over a portfolio: the report beside the positions, a repeat with the same seed gives the same numbers, what is said about a position with no calculation, the reverse search through the tool on real positions |
