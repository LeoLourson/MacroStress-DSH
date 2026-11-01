# tests/hier

Tests of the plugin `mst.plugins.hier`, the reference satellite model `hier.gibbs`. The model is checked on an object tree generated from known parameters ([../drivers/simulated.py](../drivers/simulated.py)).

| File | What it checks |
|---|---|
| `test_satellite.py` | Coverage of the true coefficients, an object with no history takes the parameters from above, the out-of-sample check, a bundle version loads as saved, paths with impacts, a lagging driver is brought to $t=0$, quarterly and explosive drivers, a bitwise repeat |
