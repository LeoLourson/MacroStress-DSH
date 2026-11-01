# mst

Ядро MacroStress: MCP-сервер, задачи, workspace и пакеты расчёта. Единственное состояние —
файлы workspace; каждый инструмент MCP разбирает аргументы и вызывает одну функцию
пакета или запускает задачу. Устройство целиком — в
[../../docs/concept/architecture.md](../../docs/concept/architecture.md).

| Модуль или пакет | Что делает |
|---|---|
| `mcp.py` | Сервер FastMCP с инструментами `jobs`, `series`, `corpus`, `scenario`, `run`, `portfolio`, `model` ([../../docs/reference/tools.md](../../docs/reference/tools.md)); `python -m mst.mcp --workspace DIR` |
| `plugin.py` | `python -m mst.plugin new --kind position\|satellite\|macro`, `check`, `demo-run`: пакет модели любого вида, его контракт и демо-расчёт ([../../docs/howto/plugins.md](../../docs/howto/plugins.md)) |
| `jobs.py` | Задачи: дочерний процесс `python <argv>` под обёрткой `python -m mst.jobs`, которая пишет код выхода, с логом в `jobs/<id>/`, блокировка ресурса (`series`, `models`, `corpus`, `code`, `runs`), отмена дерева процессов |
| `runs.py` | Расчёт на путях сценария: запрос, мета, пути, условный прогноз показателей (`forecast.json`), вклады позиций и отчёт в `runs/<id>/`; `python -m mst.runs <id> --workspace DIR` |
| `store.py` | Каталоги workspace (`Workspace`), атомарная запись JSON, канонический хеш `sha256_of`, снимок `workspace.json` |
| `versions.py` | Версии моделей на диске: хеш по метаданным и массивам, запись через черновик, чтение с проверкой хеша и имени, список без черновиков |
| `contracts.py` | Базовый класс контрактов pydantic (`extra="forbid"`, без NaN и бесконечностей) и общие типы: идентификатор, хеш, горизонт, сид, валюта, код объекта, статус результата |
| `importer.py` | Копирование рядов из каталога CSV (`<source>/data/series/*.csv`) в workspace |
| [data/](data/README.md) | Загрузка и хранение месячных рядов |
| [macro/](macro/README.md) | Реестр макропеременных, панель на дату расчёта и версии макромодели; сама модель — плагин |
| [drivers/](drivers/README.md) | Величины и драйверы: каталоги объектов, история величины на панели, бандл сателлитов, пути драйверов |
| [corpus/](corpus/README.md) | Корпус источников: манифест, загрузка, Markdown, эпизоды, индекс и поиск |
| [scenario/](scenario/README.md) | Сценарий: контракт, проверка, сборка путей, наложение и обратная задача |
| [plugin_sdk/](plugin_sdk/README.md) | Что модель позиции объявляет, читает и возвращает: `Paths`, требования, результат; `testing.py` — конформанс-проверка |
| [model_sdk/](model_sdk/README.md) | Контракты макро- и сателлитных моделей: план сценария, макросимуляция, части версии, общая условная механика |
| [engine/](engine/README.md) | Каталог моделей, подготовка входов и расчёт позиций без ветвления по типу актива |
| [plugins/](plugins/README.md) | Пакеты моделей: позиции, макро- и сателлитные; обнаружение (пространство имён, entry points, `MST_PLUGINS_PATH`) и реестр по видам; ссылочные модели — BVAR (`bvar`), MS-VAR (`msvar`), иерархический сателлит (`hier`) и регрессия второго уровня (`reflevel2`) |
| [models/](models/README.md) | Жизненный цикл версий: оценка макромодели и бандла сателлитов, публикация, список, описание, каталог плагинов |
| [metrics/](metrics/README.md) | Качество версий на исторических эпизодах: ячейки $(e, t, q)$, CRPS, эталоны, допуск, агрегат по уровням и горизонтам, решение по двум стандартным ошибкам, хвост портфеля |
| [portfolio/](portfolio/README.md) | Портфель, его проверка, стоимость позиций и отчёт расчёта |

Зависимости между пакетами идут в одну сторону: `data` → `macro` → `drivers`;
`model_sdk` — над контрактами; `scenario` нужен `macro` и `drivers`, `corpus` — метрикам;
`engine` — `plugin_sdk` и реестр `plugins`; `models` и `metrics` — реестр `plugins`;
`runs` — `scenario`, `engine`, `portfolio` и обеим версиям моделей. Ядро не называет
конкретных моделей: оно спрашивает реестр `plugins`, собранный из пакетов, entry points
и каталогов `MST_PLUGINS_PATH`.
