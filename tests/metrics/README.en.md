# tests/metrics

Tests of the quality metrics `mst.metrics`.

| File | What it checks |
|---|---|
| `simulated.py` | Fake macro and satellite models, fast enough for the episode calculation to run whole |
| `test_cells.py` | CRPS against the analytics of the normal distribution, PIT, the typical move, cell scores, the bootstrap, admission rules, benchmarks |
| `test_backtest.py` | The calculation: episode cells, the version cache by the state of the data, an unknown shock, the start month from the plan, `--details`, drivers with a naive forecast, the episode config agrees with the BVAR shocks, a point always pins its cell |
| `test_rolling.py` | The rolling check through the real version cache: uniformity on a right model, drift on a biased one, the price of an event, the block bootstrap, an event of two terms reads their sum |
| `test_cli.py` | The command line: `backtest` and `rolling` do not start without `--episodes`; the hash of the plan does not depend on where the file lies |
| `test_quality.py` | The relative CRPS: equal weight of the quantities whatever their scale, the interval by blocks; the conditional walk, the missing months and the floor in both walks; publishing a version under a file lock |
