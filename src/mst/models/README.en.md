# mst.models

The life cycle of model versions: what the tool `model` starts as a job. The models themselves are plugins (`mst.plugins`); here they are estimated on the workspace history, published as versions, listed and described.

```powershell
python -m mst.models estimate-macro [MODEL_ID] --as-of DATE --store DIR --models DIR
python -m mst.models estimate-satellites --macro SHA256 --store DIR --models DIR --drivers FILE [--model QUANTITY=MODEL_ID]
python -m mst.models list --models DIR
python -m mst.models describe --version SHA256 --models DIR
python -m mst.models diagnose --version SHA256 --models DIR
python -m mst.models catalog
```

| Action | What it does |
|---|---|
| `estimate-macro` | Estimates a macro model (the named one or the only installed one) on the panel known on the date and publishes a version |
| `estimate-satellites` | Assembles the satellite bundle of a macro version: each quantity with drivers is estimated by the model assigned to it, by declaration, and in case of ambiguity by an explicit `--model QUANTITY=MODEL`; prints the check of each driver |
| `list`, `describe` | Versions of both kinds: model, date, served drivers |
| `diagnose` | The diagnostics of a saved macro version without estimating again: the spectral radius of every parameter draw per regime with the shares above 1, 1.005, 1.01 and 1.02, the surviving draws, the regime shares by year and of the last month, the degrees of freedom by variable, the prior weight `P_0/(X'X+P_0)` per equation |
| `catalog` | Installed model plugins: macro, satellite, position; which model serves each quantity |

Assignment of a quantity: the model that declared it estimates it by default; a quantity declared by several models (a plugin in development beside an installed one) needs an explicit name. A quantity without a model is written to the bundle as `unserved` and is not estimated.

Tests are in [../../../tests/drivers/test_bundle.py](../../../tests/drivers/test_bundle.py), [../../../tests/macro/test_version.py](../../../tests/macro/test_version.py) and [../../../tests/models/test_diagnose.py](../../../tests/models/test_diagnose.py).
