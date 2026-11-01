# mst.data

Monthly series with the publication date of each value, and the loaders that get them. How value versions are stored is described in [../../../docs/concept/architecture.en.md](../../../docs/concept/architecture.en.md#series-and-publication-dates).

```powershell
python -m mst.data update [SERIES ...] --workspace DIR
```

The command downloads from January 2012 all series of `sources.toml` (or only the named ones) and adds new versions to `<workspace>/series/<id>.csv`. Exit code 1 means some series failed to download; the others are saved anyway. Requests go only to the sources in `sources.toml`.

| File | What it does |
|---|---|
| `sources.toml` | Where each series comes from: a table per id with `loader`, `lag_days` (days from the end of the month to publication), `revised` (whether the source revises values) and loader fields |
| `sources.py` | Reading `sources.toml` together with `<workspace>/sources.toml`, searching and adding a source (the tool `series`) |
| `store.py` | `SeriesStore`: one CSV per series, `as_of(day)` gives the values known on the date; `new_versions` decides which rows to add after a download |
| `update.py` | The download loop: an error in one series does not stop the others |
| `cli.py`, `__main__.py` | The command `python -m mst.data`; an HTTP client with certificate checks |
| `release.py` | Common source fields: `lag_days`, `revised` |
| `daily.py` | Daily values to monthly: mean, median, a mean over calendar days for step rates |
| `fred.py` | FRED: Brent, the Fed rate, UST 10Y, VIX; the key from `FRED_API_KEY` |
| `cbr.py` | Bank of Russia: the key rate and RUONIA (SOAP), official exchange rates, the zero-coupon OFZ curve |
| `cbr_stats.py` | Bank of Russia banking statistics: credit, the mortgage rate, the overdue share by industry; new months from the open data service, old ones from archive Excel tables |
| `moex.py` | Moscow Exchange ISS: closes of indices and shares (with splits), spreads of bond indices and single issues over the OFZ curve |
| `rosstat.py` | Rosstat: monthly tables, "Short-term economic indicators", housing prices by territory; increments on the previous month are built up into a level |
| `mintsifry-roots.pem` | Root certificates of the Russian Ministry of Digital Development; applied only to `rosstat.gov.ru` |

A new series of an existing kind is a new table in `sources.toml`. The tool `series(action="add")` writes it not here but to `<workspace>/sources.toml`: this file is read on top of the built-in one, and ids in the two files must not repeat. A new kind of source is a class with a field `loader: Literal[...]` and a method `fetch(http, start) -> dict[date, float]`, added to the union `Source` in `sources.py`. Rosstat series are stored without seasonal adjustment; `mst.macro` removes it.

Tests are in [../../../tests/data](../../../tests/data/README.en.md), without network.
