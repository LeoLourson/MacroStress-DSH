# Инструменты и команды

## MCP-инструменты

Сервер `macrostress` (`src/mst/mcp.py`) даёт четыре инструмента. Ошибка в аргументах
возвращается как ошибка вызова с текстом, где перечислены допустимые значения.
Инструмент, который запускает задачу, возвращает `{"job": <id>, "argv": [...]}` сразу,
не дожидаясь её конца.

### `jobs`

| `action` | Аргументы | Результат |
|---|---|---|
| `state` | — | Снимок workspace; заодно отмечает завершёнными задачи без живого процесса и переписывает `workspace.json` |
| `status` | `id` | Запись задачи (`argv`, `pid`, `started`, `finished`, `cancelled`), `running` и `exit_code` (`null`, пока идёт или если отменена) |
| `log` | `id`, `tail` (50; 0 — весь лог) | Последние строки лога |
| `cancel` | `id` | Останавливает дерево процессов задачи и снимает блокировку |

### `series`

| `action` | Аргументы | Результат |
|---|---|---|
| `search` | `text` | Источники встроенного и workspace-файла `sources.toml`, чей id или загрузчик содержит текст |
| `get` | `id`, `as_of` (по умолчанию сегодня) | Значения ряда, известные на дату: `month`, `value`, `published`, `approximate` |
| `add` | `entry` | Добавляет источник в `<workspace>/sources.toml`, который читается поверх встроенного `src/mst/data/sources.toml`; id не может повторять известный. `entry` — таблица файла плюс поле `id` |
| `update` | `ids` (пусто — все) | Задача на ресурсе `series`: `python -m mst.data update` |
| `update` | `quantity` | Задача на ресурсе `series`: каталог величины находит объекты, затем загружаются их ряды (`python -m mst.drivers update`) |

`ids` и `quantity` вместе не принимаются.

### `corpus`

Корень корпуса — `<workspace>/corpus`. `search` и `get` читают индекс и сети не требуют.

| `action` | Аргументы | Результат |
|---|---|---|
| `search` | `query`, `as_of`, `kind`, `region`, `factors`, `k` (8) | Лучшие фрагменты, видимые на `as_of`: `chunk_id`, документ, путь заголовков, страницы, текст |
| `get` | `id` — `chunk_id`, `as_of` | Один фрагмент; вышедший позже `as_of` — ошибка с датой публикации |
| `list` | — | Что в манифесте: источники и их файлы (адрес, скачан, переведён), эпизоды, статусы |
| `add` | `key`, `file`, `url` | Регистрирует скачанный руками файл в `raw/` и `downloads.json`, как `fetch` |
| `index` | — | Задача на `corpus`: `python -m mst.corpus index` |
| `fetch` | `ids` (ключи документов) | Задача на `corpus`: `python -m mst.corpus fetch` |

### `model`

| `kind` | `action` | Аргументы | Результат |
|---|---|---|---|
| `macro` | `list` | — | Версии, новые первыми |
| `macro` | `describe` | `version` | Метаданные версии и доля розыгрышей, прошедших знаковые ограничения |
| `macro` | `estimate` | `as_of` | Задача на `models`: оценка по данным, известным на дату |
| `macro` | `backtest` | `as_of`, `only` (id эпизодов) | Задача на `models`: проверка на эпизодах `backtest.toml` |
| `satellites` | `list` | — | Версии сателлитов |
| `satellites` | `describe` | `version` | Обслуживаемые драйверы и отклонённые с причиной |
| `satellites` | `estimate` | `version` — хеш макроверсии | Задача на `models`: оценка и проверка сателлитов на её панели |

## Команды

Задачи запускают те же команды, что можно выполнить в терминале из корня репозитория.
Модели оцениваются только через `python -m`: точка входа `mst.drivers` ставит
`OPENBLAS_NUM_THREADS=1` до импорта numpy, иначе мелкие плотные системы решаются в 20
потоков и оценка сателлитов занимает часы вместо минут.

```powershell
# ряды
python -m mst.data update [SERIES ...] --workspace DIR

# макромодель
python -m mst.macro estimate --as-of 2026-10-29 --store DIR --models DIR
python -m mst.macro backtest --as-of 2026-10-29 --store DIR --models DIR --corpus FILE `
    [--signs FILE ...] [--only ID ...] [--details FILE]

# драйверы
python -m mst.drivers update [QUANTITY ...] --store DIR --drivers FILE
python -m mst.drivers estimate --macro SHA256 --store DIR --models DIR --drivers FILE
python -m mst.drivers check --macro SHA256 --store DIR --models DIR --drivers FILE

# сервер
python -m mst.mcp --workspace DIR

# корпус
python -m mst.corpus index --corpus DIR
python -m mst.corpus search QUERY --as-of DAY --corpus DIR
```

Здесь `--store` — `<workspace>/series`, `--models` — `<workspace>/models`, `--drivers` —
`<workspace>/drivers/catalog.json`, `--corpus` — `<workspace>/corpus/manifest.json` (у
`mst.corpus` — корень `<workspace>/corpus`).

`backtest` печатает покрытие интервала $[P_5, P_{95}]$ по эпизодам и по переменным;
каждый файл `--signs` — альтернативная таблица знаков в формате `[[shocks]]` реестра, и
она проверяется рядом с основной. Модели эпизодов пишутся в `<models>/backtest/`, а не в
`<models>/macro/`; код выхода 1 — какой-то эпизод не проверен (нет в манифесте, короткая
история, недопустимый сценарий). `check` делает то же, что `estimate`, но версию не
сохраняет.
