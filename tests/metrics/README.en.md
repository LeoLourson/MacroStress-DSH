# tests/metrics

Tests of the quality metrics `mst.metrics`.

| File | What it checks |
|---|---|
| `simulated.py` | Fake macro and satellite models, fast enough for the episode calculation to run whole |
| `test_cells.py` | CRPS against the analytics of the normal distribution, PIT, the typical move, cell scores, the bootstrap, admission rules, benchmarks |
| `test_backtest.py` | The calculation: episode cells, the version cache by the state of the data, an unknown shock and an episode outside the manifest, `--details`, drivers with a naive forecast, the episode config agrees with the BVAR shocks, a point always pins its cell |
| `test_rolling.py` | The rolling check through the real version cache: uniformity on a right model, drift on a biased one, the price of an event, the block bootstrap, an event of two terms reads their sum |
