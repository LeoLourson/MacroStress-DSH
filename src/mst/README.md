# mst

Ядро MacroStress: MCP-сервер, задачи, workspace и пакеты расчёта. Единственное состояние —
файлы workspace; каждый инструмент MCP разбирает аргументы и вызывает одну функцию
пакета или запускает задачу. Устройство целиком — в
[../../docs/architecture.md](../../docs/architecture.md).

| Модуль или пакет | Что делает |
|---|---|
| `mcp.py` | Сервер FastMCP с инструментами `jobs`, `series`, `model` ([../../docs/tools.md](../../docs/tools.md)); `python -m mst.mcp --workspace DIR` |
| `jobs.py` | Задачи: дочерний процесс `python <argv>` с логом в `jobs/<id>/`, блокировка ресурса (`series`, `models`, `corpus`, `code`), отмена дерева процессов |
| `store.py` | Каталоги workspace (`Workspace`), атомарная запись JSON, канонический хеш `sha256_of`, снимок `workspace.json` |
| `contracts.py` | Базовый класс контрактов pydantic (`extra="forbid"`, без NaN и бесконечностей) и общие типы: идентификатор, хеш, горизонт, сид |
| `importer.py` | Разовый перенос рядов старого проекта (`<old>/data/series/*.csv`) в workspace |
| [data/](data/README.md) | Загрузка и хранение месячных рядов |
| [macro/](macro/README.md) | Макромодель: оценка, идентификация, условный прогноз, тяжесть, бэктест |
| [drivers/](drivers/README.md) | Драйверы: каталоги, сателлиты, их проверка и пути |
| [corpus/](corpus/README.md) | Корпус источников: пока только манифест |
| [scenario/](scenario/README.md) | Контракт сценария |

Зависимости между пакетами идут в одну сторону: `data` → `macro` → `drivers`; `scenario`
нужен `macro` и `drivers`, `corpus` — бэктесту `macro`.
