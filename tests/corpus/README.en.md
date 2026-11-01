# tests/corpus

Tests of `mst.corpus`. The network is substituted: source answers and pages are set in the test, so format parsing, conversion and the index are checked, not the availability of sites.

| File | What it checks |
|---|---|
| `pdfs.py` | A minimal text PDF assembled by hand: one font, line by line |
| `test_manifest.py` | The manifest reads and saves byte for byte, editions unfold into files and paths, an address is a link or a page with a pattern, a foreign edition is rejected |
| `test_fetch.py` | Page links are absolute and have text, `page` + `match` takes the first fitting one, `fetch` downloads once and writes `downloads.json` |
| `test_convert.py` | Gluing of lines and hyphenation, dropping headers, footers and page numbers, PDF to Markdown with a head and page marks, the end of the year for an unknown day, HTML with headings, paragraphs and tables |
| `test_series.py` | BIS rows from SDMX CSV, ruble denomination only on a marked jump, the year-on-year growth of an index |
| `test_episodes.py` | A window from three months before `start` to twelve after, the move and recovery from the last value before `start`, a card keeps its written sections, a factor without data stays `unmeasured`, `series.toml` names only known series |
| `test_index.py` | Numbering of sections with pages and heading paths, a long section is cut with whole tables, search hides fragments after `as_of` and ranks by stems, question markup and recall, vectors find a fragment with no common words, are reused and are checked on load, `state.json` names the day and the number of fragments |
