---
name: drivers
description: Update the driver catalog, estimate satellites and understand a rejection by the check
whenToUse: when a position is not computed because of a driver, or a new object or a new satellite version is needed
---

# Drivers and satellites

A driver is a quantity of an object, `<quantity>:<object>`. Take concrete ids from `model(kind="satellites", action="describe")` or `workspace/drivers/catalog.json`, for example `share_price:sber`, `bond_spread:ru000a0jxpg2`, `overdue_share:stroitelstvo`, `housing_price_primary:okato_77000000000`. A satellite is the model of a driver given the macro variables; a satellite version is estimated on the panel of one macro version.

## 1. Update the catalog

`series(action="update", quantity="share_price")` starts a job on `series`: the catalog finds the objects of the quantity, loads their series and writes `workspace/drivers/catalog.json`. The same works for `bond_spread`, `overdue_share`, `housing_price_primary`, `housing_price_secondary`. A new object of a source appears by itself on the next `update`; a failed catalog keeps the previous record of the quantity.

`jobs(action="status", id=<job>)` and `jobs(action="log", id=<job>)` show the progress of a job.

## 2. Estimate the satellites

`model(kind="satellites", action="estimate", version=<macro version>)` is a job on `models`: estimation on the panel of that macro version, an out-of-sample check and a version written to `workspace/models/satellites/<hash>/`. `model(kind="satellites", action="list")` lists the versions.

## 3. Read a rejection

`model(kind="satellites", action="describe", version=<satellites>)` returns `served` and `rejected`. A rejected driver has `reason`, `satellite`, `baseline`, `forecasts`. The reasons:

- `too few observations before the holdout`: the history is too short for the check;
- `no observation of the holdout forecast`: nothing to forecast in the check window;
- `not more accurate than without macro`: the satellite is no more accurate than an autoregression without macro variables;
- `no checked driver above it`: the driver has no history of its own and no checked object above it;
- `the driver above, <driver>, is not served`: its parameters come from an object that is itself rejected.

A rejection is neither a data error nor a zero. The platform does not simulate such a driver; a position that needs it stays `invalid` with `driver_missing:<quantity>`, and the report excludes it with the reason and shows coverage. Do not present an excluded position as a zero loss.

## 4. Commands

Estimation runs only through `python -m` from the repository root, with the interpreter of the server: the entry point sets `OPENBLAS_NUM_THREADS=1`, otherwise small systems are solved in 20 threads and estimation takes hours instead of minutes.

```powershell
python -m mst.drivers update share_price --store workspace/series --drivers workspace/drivers/catalog.json
python -m mst.models estimate-satellites --macro <hash> --store workspace/series --models workspace/models --drivers workspace/drivers/catalog.json
```

`estimate` writes a version, `check` does the same without writing. A new quantity means editing `src/mst/drivers/registry.toml` and, if the source has no catalog, adding a class in `src/mst/drivers/catalogs/`; that is a change to the platform code, not agent work in the workspace.
