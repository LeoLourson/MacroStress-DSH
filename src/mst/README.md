# mst

Ядро MacroStress: MCP-сервер, задачи, workspace и пакеты расчёта. Единственное состояние —
файлы workspace; каждый инструмент MCP разбирает аргументы и вызывает одну функцию
пакета или запускает задачу. Устройство целиком — в
[../../docs/architecture.md](../../docs/architecture.md).

| Модуль или пакет | Что делает |
|---|---|
| `mcp.py` | Сервер FastMCP с инструментами `jobs`, `series`, `corpus`, `scenario`, `run`, `portfolio`, `model` ([../../docs/tools.md](../../docs/tools.md)); `python -m mst.mcp --workspace DIR` |
| `jobs.py` | Задачи: дочерний процесс `python <argv>` под обёрткой `python -m mst.jobs`, которая пишет код выхода, с логом в `jobs/<id>/`, блокировка ресурса (`series`, `models`, `corpus`, `code`, `runs`), отмена дерева процессов |
| `runs.py` | Прогон на путях сценария: запрос, мета, пути, вклады позиций и отчёт в `runs/<id>/`; `python -m mst.runs <id> --workspace DIR` |
| `store.py` | Каталоги workspace (`Workspace`), атомарная запись JSON, канонический хеш `sha256_of`, снимок `workspace.json` |
| `versions.py` | Версии моделей на диске: хеш по метаданным и массивам, запись через черновик, чтение с проверкой хеша и имени, список без черновиков |
| `contracts.py` | Базовый класс контрактов pydantic (`extra="forbid"`, без NaN и бесконечностей) и общие типы: идентификатор, хеш, горизонт, сид, валюта, код объекта, статус результата |
| `importer.py` | Разовый перенос рядов старого проекта (`<old>/data/series/*.csv`) в workspace |
| [data/](data/README.md) | Загрузка и хранение месячных рядов |
| [macro/](macro/README.md) | Макромодель: оценка, идентификация, условный прогноз, тяжесть, бэктест |
| [drivers/](drivers/README.md) | Драйверы: каталоги, сателлиты, их проверка и пути |
| [corpus/](corpus/README.md) | Корпус источников: пока только манифест |
| [scenario/](scenario/README.md) | Контракт сценария |
| [plugin_sdk/](plugin_sdk/__init__.py) | Что модель позиции объявляет, читает и возвращает: `Paths`, требования, результат |
| [engine/](engine/__init__.py) | Каталог моделей, подготовка входов и расчёт позиций без ветвления по типу актива |
| [plugins/](plugins/__init__.py) | Модели позиций: акция, облигация, кредит или депозит, недвижимость |
| [portfolio/](portfolio/__init__.py) | Портфель, его проверка, стоимость позиций и отчёт прогона |

Зависимости между пакетами идут в одну сторону: `data` → `macro` → `drivers`; `scenario`
нужен `macro` и `drivers`, `corpus` — бэктесту `macro`; `engine` и `runs` — `plugin_sdk`,
`plugins`, `scenario` и обеим версиям моделей; `portfolio` — `engine` и `scenario`.
