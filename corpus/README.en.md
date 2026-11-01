# Scenario corpus

A stress-test scenario is built from these materials: regulators' methodologies, their published scenarios, works on the transmission of shocks and descriptions of past crises. The full list is in [`manifest.json`](manifest.json). Git holds the manifest, `series.toml` and the check questions; downloaded originals, Markdown texts and the index are data: they live in `workspace/corpus/` and stay out of git.

## Contents

| `kind` | Content | ID | Volume |
|---|---|---|---|
| `methodology` | the choice of scenario severity, horizon and factors, the consistency of paths, satellite models | M1–M21 | 21 documents, 30 editions |
| `scenario_set` | regulators' scenarios with numbers | S1–S7 | 7 series, about 90 editions |
| `transmission` | the transmission of shocks to rates, the exchange rate, credit and spreads | T1–T5 | about 40 documents, selected by episode |
| `episode` | the course of a past crisis | R1–R32, X1–X35, G1–G43 | 110 cards |

Series D1–D5 (Bank of Russia, MOEX, FRED, IMF IFS, BIS) are not in the index: a script calculates from them the tables of moves for episode cards. [`series.toml`](series.toml) sets which series measures a factor in a country. D4 (IMF IFS) is replaced by BIS series with the same content: exchange rates, central bank rates, inflation, housing prices.

## Folders

In git:

```
corpus/
  manifest.json                 the corpus list and the addresses of originals, edited by hand
  series.toml                   series for the tables of moves and their link to factors and countries
  eval/questions.md             check questions for search
  templates/source.md           document template
  templates/episode.md          episode template
```

In the workspace (`--corpus workspace/corpus`, outside git) the same layout plus everything the commands write:

```
workspace/corpus/
  manifest.json, series.toml, eval/, templates/   a copy of the above
  downloads.json                what was downloaded: address, sha256, size, date; written by fetch
  sources/methodology/M1.md     a document with one edition
  sources/scenario-sets/S1/2023.md   a series: a file per edition
  sources/transmission/T2/<slug>.md
  episodes/ru/R16.md            also em/ and dm/
  raw/<id>/<edition>.pdf        downloaded originals
  text/<id>/<edition>.md        text of books and journals with text_policy: summary
  series/<series>.csv           daily series D1–D5
  index/                        chunks.jsonl, vectors.npy, vectors.json, state.json
```

The field `path` in the manifest sets a file's path, so a file with no manifest record does not enter the index.

## Build

`mst.corpus` (`src/mst/corpus`) builds the corpus; `--corpus` is the corpus root in the workspace:

```
python -m mst.corpus check --corpus DIR              what is there and what is missing
python -m mst.corpus fetch [ID ...] --corpus DIR     originals to raw/, a record in downloads.json
python -m mst.corpus discover URL [--match RE]       links to documents from a page, to add to files
python -m mst.corpus add KEY FILE [--url URL] --corpus DIR   a file downloaded by hand, as if fetch had got it
python -m mst.corpus convert [ID ...] --corpus DIR   originals to Markdown
python -m mst.corpus series [SERIES ...] --corpus DIR --store DIR  daily series D1–D5 to series/
python -m mst.corpus episodes [ID ...] --corpus DIR --store DIR    episode cards
python -m mst.corpus index --corpus DIR              index/chunks.jsonl and vectors
python -m mst.corpus search QUERY --as-of DAY --corpus DIR  search
python -m mst.corpus eval --corpus DIR               recall@5 by eval/questions.md
python -m mst.corpus all --corpus DIR --store DIR    fetch, convert, series, episodes, index in a row
```

The same is available through the tool `corpus`: `search`, `get`, `list` and `add` answer at once, while `index` and `fetch` run as a job on the resource `corpus`. `--store` is the monthly series store (`workspace/series`): the commands look there when there is no daily series.

The address of an original lies in the manifest: for a document with one edition in the field `file`, for a series in `files` by edition. An address is either a direct link `url` or a page `page` and a regular expression `match`: then the first link on the page whose address or text matches the expression is taken. This way files with changing or unknown addresses get downloaded. `format: html` means the document is the web page itself. `published` sets the publication date if the publisher gives it. `fetch` skips an edition without an address and `check` lists it; its address is found through `discover`. A file already on disk with the hash from `downloads.json` is not downloaded again.

FRED requires a key in the variable `FRED_API_KEY`. The other sources need no key. imf.org answers 403 to script requests, so M8–M11 and S7 are downloaded by hand in a browser and registered through `add`: `python -m mst.corpus add M11 wp1368.pdf --corpus workspace/corpus`. economy.gov.ru and iss.moex.com do not answer requests from abroad: the S3 forecasts come from a copy on legalacts.ru, and MOEX series download when the exchange answers.

## Statuses

| | `planned` | `raw` / `draft` | `ready` |
|---|---|---|---|
| document | no text | text extracted by `convert`, not proofread; indexed | proofread, all editions in place |
| episode | no card | a card with a table of moves, chronology not written; indexed | chronology, mechanism and sources written |

`convert` moves a source from `planned` to `raw`, `episodes` moves an episode from `planned` to `draft`. The move to `ready` is made by hand after proofreading.

## Document

A file in `sources/` starts with a YAML head by [`templates/source.md`](templates/source.md), followed by the document's text. The field `published` holds the publication date to the day: the `as_of` filter uses it to hide documents published after the run date. `published_basis` says where the date comes from: `manifest` (the field `published` in the file address), `pdf_metadata` (the PDF creation date if its year matches the edition year) or `year_end` (December 31 of the edition year, the latest possible date, so that `as_of` does not show a document too early). With `text_policy: full` the whole text is stored, while books and subscription journals get `summary` and one's own summary with page references. `convert` puts the text of such sources in `<corpus>/text/`, not in the index: it is material for the summary.

`convert` takes the PDF text layer without recognition. A scanned page stays empty and is marked `<!-- p. N: нет текстового слоя -->`; if more than half of the pages are like that, `convert` reports that the file needs recognition. A page number is the label the PDF shows. Headings come from the PDF table of contents, and headers and footers (lines repeated on most pages) are removed. Tables come out as lines of text, not Markdown tables: moving them is part of the proofreading for `ready`.

The original's headings carry over as `#`–`####`, because the index is cut by them. Before the text of each page stands a comment `<!-- p. 12 -->`, so a quote can be given with its page. Scenario tables carry over as Markdown tables, and units go in the column heading ("Key rate, % per year"). `source_sha256` records the hash of the source file, `retrieved` the download date.

## Episode

The card `episodes/<region>/<ID>.md` by [`templates/episode.md`](templates/episode.md) describes only what happened, with references to sources of that time. Each statement of the chronology rests on a document published in the episode's window: a central bank press release, a report, a review.

The default window starts 3 months before `start` and ends 12 months after. `python -m mst.corpus episodes` fills the table "Moves of factors" and the fields `movements` and `unmeasured` from the series D1–D5, the same way for all episodes, and they are not edited by hand. The chronology, mechanism, analogy and sources are written by hand, and a repeated start keeps them. The field `known_from` equals the end of the window: before this date search does not show the episode, otherwise a scenario for a date inside a crisis would be built with a known outcome.

For each factor the first series from `series.toml` with data both before and after `start` is taken. From it the following are calculated:

- the base: the last observation before the first day of the month of `start`;
- the extreme: the observation from `start` to the end of the window furthest from the base;
- the change: the extreme minus the base in percentage points for rates and spreads, the ratio of the extreme to the base minus one in percent for exchange rates, indices and prices;
- days to the extreme: from the first day of the month of `start`;
- days to recovery: from the extreme to the first observation that returned to or passed the base level; empty if this did not happen in the window.

A factor with no market series (bank liquidity and capital, capital flows, sanctions, default) goes to `unmeasured` and is described in the chronology. Until the daily series are downloaded, monthly series from `workspace/series` (`--store`) are used if `monthly` in `series.toml` names them. These series start in 2012, and an extreme within a month is not visible in them; the table says so, and the column "Series" marks the resolution.

## Index fragment

Each line of `index/chunks.jsonl` describes one fragment:

| Field | Value |
|---|---|
| `chunk_id` | `S1/2023#3.2/1`: document, edition, section number, number of the piece in the section |
| `doc_id`, `edition`, `kind`, `region`, `jurisdiction` | from the manifest; a document's `region` is `ru` for Russian ones and empty for the rest |
| `visible_from` | the document's `published` or the episode's `known_from` |
| `title` | the `#` heading of the document or card |
| `heading_path` | `["Рисковые сценарии", "Глобальный финансовый кризис"]` |
| `pages` | numbers from the comments `<!-- p. N -->` |
| `factors` | tags of the head |
| `text` | the text of the fragment |
| `sha256` | the hash of the normalized text: a quote is checked against the source by it |

## Indexing

The indexer reads the manifest and skips records with the status `planned`, and counts a `ready` record with no file as an error; for a `raw` record the missing editions are only listed. A document is cut by headings down to the level `####`; a token is a word. A section longer than 800 tokens is split by paragraphs into pieces of 300–800 tokens with an overlap of one paragraph. A table always stays in one piece with the paragraph before it. An episode card is cut by its sections, and the table of moves becomes a separate fragment. Before hashing, the text is brought to NFC, spaces collapse and page comments are removed.

Search ranks fragments twice: by BM25 over `title`, `heading_path` and `text` with Russian and English stemming, and by the cosine of e5 vectors (`index/vectors.npy`, written by `python -m mst.corpus index`). The two lists of 100 places are added by reciprocal ranks, $1/(60+r)$. The filters `visible_from <= as_of`, `kind`, `region` and `factors` apply before ranking:

```
search(query, as_of, kind=None, region=None, factors=None, k=8) -> [chunk]
```

The same search is available through the tool `corpus(action="search")`, and a fragment by `chunk_id` through `corpus(action="get")`. Both filter by the same date `as_of` and need no network: `index/state.json` stores the index build date and the number of fragments.

## Search check

`eval/questions.md` holds 20–30 questions that arise when building a scenario, and for each the correct `chunk_id`s. The metric is recall@5 over all questions; any change of the cutting, the stemming or the search model comes with its recalculation.

## What the corpus holds now

- Sources: 141 of 150 files are downloaded, 140 are converted to Markdown and have the status `raw`. Not downloaded: M8–M11 and S7 (imf.org gives 403 to script requests: the files are taken in a browser and registered through `add`), S3 for 2018–2019 (no copy), M19 (no Russian text of IFRS 9 was found), T5 (GFSR chapters, imf.org). M16 has its text in `<corpus>/text/`, and the summary is not written.
- Episodes: 110 cards with tables of moves from daily series, at least one factor is measured in 105. Chronologies are written for R1–R32 and X1–X35 but not G1–G43; all episodes are in `draft`.
- Index: 7036 fragments with vectors, recall@5 = 0.271 on 24 marked questions. All questions on episodes are found (Q17–Q24); of 16 questions on documents, Q7, Q8 and Q15. Ten misses fall on English documents: a Russian query reaches the answer neither by BM25 nor by the e5-small vector.

## Next

1. Download the imf.org files in a browser and register them through `add`; select GFSR chapters for T5.
2. Write episode chronologies from sources of that time, starting with the Russian R1–R32.
3. Proofread texts: scenario tables in Markdown, word breaks; move sources to `ready`.
4. Choose English query wordings or a multilingual vector model for English documents and recalculate recall@5.
