# mst.drivers.catalogs

A catalog finds all objects of a quantity at a source and declares for each a series source from `mst.data`, so driver series are downloaded and stored like macro series. A catalog is declared in `../registry.toml` under its quantity by a table with `loader` and publication conditions (`lag_days`, `revised`). A new object of an existing source needs no code change.

| File | `loader` | What it finds |
|---|---|---|
| `base.py` | none | The base `Catalog`, the result `Discovery` (objects, series sources, what was skipped with a reason), the object names `slug` and `name` |
| `moex.py` | `moex_shares` | Ordinary and preferred shares of a trading mode, each under the MOEX industry index it belongs to. Modes before T+2 and splits come from ISS |
| `moex.py` | `moex_bonds` | Issues from corporate bond indices (one index per rating group) under an issuer, and the issuer under the group of its first issue |
| `cbr.py` | `cbr_activities` | Industries of the Bank of Russia table 01_02_A; the level follows the indent of the label, the root is "ВСЕГО", OKVED 2 codes follow the label |
| `rosstat.py` | `rosstat_regions` | Russia (643), federal districts and regions by OKATO codes from the housing price tables; the objects `okato_<code>` are common to the primary and secondary markets |

A new source is a subclass of `Catalog` with a field `loader: Literal[...]` and a method `discover(http) -> Discovery`, added to the union `CatalogSource` in `__init__.py`. Objects in `Discovery.objects` come after their parent.
