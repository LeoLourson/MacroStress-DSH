# mst.corpus

Корпус источников сценария ([corpus/README.md](../../../corpus/README.md)): манифест с
адресами оригиналов, загрузка, перевод PDF и веб-страниц в Markdown, карточки эпизодов с
таблицами движений, индекс фрагментов и поиск по нему.

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

`--corpus` — корень корпуса в workspace (`workspace/corpus`). В git лежат только манифест,
`series.toml`, проверочные вопросы и шаблоны (`corpus/` в корне репозитория); скачанные
оригиналы, тексты и индекс — данные workspace.

| Файл | Что делает |
|---|---|
| `manifest.py` | Контракты манифеста: источники и их издания, адреса оригиналов (`url` или `page` + `match`), карточки эпизодов, ряды; `Document` — один файл источника |
| `fetch.py` | Загрузка оригиналов в `raw/`, запись в `downloads.json` с хешем и датой; файл с тем же хешем повторно не скачивается |
| `convert.py` | PDF и веб-страница в Markdown: текстовый слой pypdf, заголовки из оглавления, колонтитулы и номера страниц выброшены, шапка с датой публикации |
| `series.py` | Дневные ряды D1–D5 для таблиц движений: загрузчик из `mst.data` или BIS, кеш в `series/`, месячный ряд из хранилища как запасной вариант |
| `episodes.py` | Карточка эпизода: окно, база, экстремум, изменение, дни до экстремума и восстановления; написанные руками разделы сохраняются |
| `index.py` | Нарезка Markdown на фрагменты, `chunks.jsonl`, векторы e5, BM25 + косинус с фильтром `as_of`, recall@5 по `eval/questions.md` |
| `embed.py` | Векторы multilingual-e5-small на CPU через ONNX Runtime |
| `tool.py` | Операции инструмента `corpus`: `search`, `get`, `list`, `add` |
| `cli.py`, `__main__.py` | Команда `python -m mst.corpus` |

Тесты — [../../../tests/corpus](../../../tests/corpus/).
