# mst.corpus

The source corpus of a scenario ([corpus/README.en.md](../../../corpus/README.en.md)): a manifest with the addresses of originals, download, conversion of PDFs and web pages to Markdown, episode cards with tables of moves, a fragment index and search over it.

```powershell
python -m mst.corpus check --corpus DIR
python -m mst.corpus fetch [ID ...] --corpus DIR [--force]
python -m mst.corpus add KEY FILE [--url URL] --corpus DIR
python -m mst.corpus convert [ID ...] --corpus DIR
python -m mst.corpus series [SERIES ...] --corpus DIR --store DIR
python -m mst.corpus episodes [ID ...] --corpus DIR --store DIR
python -m mst.corpus index --corpus DIR
python -m mst.corpus search QUERY --as-of DAY --corpus DIR
python -m mst.corpus eval --corpus DIR
```

`--corpus` is the corpus root in the workspace (`workspace/corpus`). Git holds only the manifest, `series.toml`, the check questions and the templates (`corpus/` in the repository root); downloaded originals, texts and the index are workspace data.

| File | What it does |
|---|---|
| `manifest.py` | Manifest contracts: sources and their editions, the addresses of originals (`url` or `page` + `match`), episode cards, series; `Document` is one file of a source |
| `fetch.py` | Download of originals to `raw/`, a record in `downloads.json` with the hash and date; a file with the same hash is not downloaded again |
| `convert.py` | A PDF or web page to Markdown: the pypdf text layer, headings from the table of contents, headers, footers and page numbers dropped, a head with the publication date |
| `series.py` | Daily series D1–D5 for the tables of moves: a loader from `mst.data` or BIS, a cache in `series/`, the monthly series from the store as a fallback |
| `episodes.py` | An episode card: window, base, extreme, change, days to the extreme and to recovery; sections written by hand are kept |
| `index.py` | Cutting Markdown into fragments, `chunks.jsonl`, e5 vectors, BM25 + cosine with an `as_of` filter, recall@5 by `eval/questions.md` |
| `embed.py` | multilingual-e5-small vectors on the CPU through ONNX Runtime |
| `tool.py` | Operations of the tool `corpus`: `search`, `get`, `list`, `add` |
| `cli.py`, `__main__.py` | The command `python -m mst.corpus` |

Tests are in [../../../tests/corpus](../../../tests/corpus/).
