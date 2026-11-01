# tests/data

Tests of `mst.data`. HTTP answers of the sources are substituted, so the tests check the parsing of MOEX, Bank of Russia, Rosstat and FRED formats, not the availability of sites.

| File | What it checks |
|---|---|
| `test_store.py` | The store: writing and reading, `as_of`, dating of the first download and revisions, the flag `approximate`, a series id does not leave the directory |
| `test_update.py` | The download loop: new rows are written once, errors go by series |
| `test_sources.py` | `sources.toml`: search, adding a table, refusal of a taken id; each source feeds a registry variable |
| `test_cli_client.py` | Certificates: each host is checked, the ministry roots apply only to `rosstat.gov.ru` |
| `test_fred.py` | FRED: monthly means, missing values, a clear error without a key |
| `test_cbr.py` | The key rate with steps before 2013, RUONIA, exchange rates per unit, the zero-coupon curve |
| `test_cbr_stats.py` | Credit, the mortgage rate, the overdue share: the service against the archive, refusal of a foreign or unreadable table |
| `test_moex.py` | Indices page by page, trading modes and share splits, a stop at an undeclared split |
| `test_spread.py` | Curve interpolation, the spread of an index over the curve with a splice, skipped days |
| `test_rosstat.py` | Finding a workbook by a pattern, chaining increments, sections of "Short-term indicators", housing prices by years and quarters |
