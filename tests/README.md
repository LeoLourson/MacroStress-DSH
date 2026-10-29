# tests

Тесты pytest; ни один не ходит в сеть: ответы источников подменяются, а модели проверяются
на данных, сгенерированных из известных параметров.

```powershell
.venv\Scripts\python.exe -m pytest -q
# без тяжёлой оценки BVAR
.venv\Scripts\python.exe -m pytest -q --ignore=tests/macro/test_estimate.py `
    --ignore=tests/macro/test_forecast.py --ignore=tests/macro/test_identify.py
```

| Файл или папка | Что проверяет |
|---|---|
| `test_store.py` | Канонический хеш, атомарная запись JSON, снимок workspace |
| `test_jobs.py` | Задачи: запуск, лог, отказ занятого ресурса, освобождение после конца, отмена |
| `test_mcp.py` | Инструмент `jobs`: снимок, обновление `workspace.json`, status, log, cancel, отказ без `id` |
| `test_series_tool.py` | Инструмент `series`, импорт рядов старого проекта, задача загрузки |
| [data/](data/README.md) | Загрузчики и хранилище рядов |
| [macro/](macro/README.md) | Реестр, панель, оценка, идентификация, прогноз, версии, бэктест, инструмент `model` |
| [drivers/](drivers/README.md) | Каталоги, сателлиты, их проверка и пути, инструмент `model(kind="satellites")` |
