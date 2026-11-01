# tests/bvar

Tests of the plugin `mst.plugins.bvar`, the reference macro model `bvar.sv`. The model is checked on a panel generated from known parameters (`simulated.py`): intervals must cover the true values with the declared frequency. `test_estimate.py`, `test_forecast.py` and `test_identify.py` run the sampler, are marked `slow` and take minutes.

| File | What it checks |
|---|---|
| `simulated.py` | The panel generator and common fixtures |
| `test_estimate.py` | Coverage of the true coefficients, block exogeneity, a bitwise repeat, the draw of missing values, stochastic volatility |
| `test_sv.py` | The Gibbs steps of volatility: the mixture, degrees of freedom, the log variance path |
| `test_identify.py` | Each kept draw satisfies the signs, rotations are uniform, unidentifiable draws are discarded |
| `test_forecast.py` | Exact and interval conditions, a bitwise repeat, inadmissible scenarios, the law of the conditions against the analytic value, candidates of the volatility path |
| `test_signs.py` | The shock table: consistency with the variable registry, signs, uniqueness |
