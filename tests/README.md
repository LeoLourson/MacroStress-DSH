# tests

Тесты pytest; ни один не ходит в сеть: ответы источников подменяются, а модели проверяются
на данных, сгенерированных из известных параметров.

```powershell
.venv\Scripts\python.exe -m pytest -q
# без тяжёлой оценки BVAR
.venv\Scripts\python.exe -m pytest -q -m "not slow"
```

`test_estimate.py`, `test_forecast.py` и `test_identify.py` гоняют сэмплер и идут минуты; они
помечены `slow`, и их пропускают, если модель не менялась.

| Файл или папка | Что проверяет |
|---|---|
| `conftest.py` | Общие фикстуры: макроверсия с сателлитами на сеанс, workspace с версиями и сценарием |
| `test_store.py` | Канонический хеш, атомарная запись JSON, снимок workspace |
| `test_versions.py` | Хеш версий не изменился, чтение и отказ на изменённой версии, имя не хеш, черновики |
| `test_jobs.py` | Задачи: запуск, лог, код выхода, отказ занятого ресурса, освобождение после конца, отмена |
| `test_mcp.py` | Инструмент `jobs`: снимок, обновление `workspace.json`, status, log, cancel, отказ без `id` |
| `test_series_tool.py` | Инструмент `series`, импорт рядов из каталога CSV, задача загрузки |
| `test_corpus_tool.py` | Инструмент `corpus`: `search`, `get` и `list` читают индекс, `index` и `fetch` запускают задачу |
| `test_runs.py` | `mst.runs`: запрос, мета, пути и результаты в `runs/<id>/`, инструмент `run` |
| [data/](data/README.md) | Загрузчики и хранилище рядов |
| [macro/](macro/README.md) | Реестр, панель, оценка, идентификация, прогноз, версии, бэктест, инструмент `model` |
| [drivers/](drivers/README.md) | Каталоги, сателлиты, их проверка и пути, инструмент `model(kind="satellites")` |
| [corpus/](corpus/README.md) | Манифест, загрузка, конвертация, эпизоды, индекс и поиск |
| [scenario/](scenario/README.md) | Контракт сценария, сборка путей, инструмент `scenario` |
| [engine/](engine/README.md) | Диспетчер, модели позиций, статусы и набор результатов |
| [plugins/](plugins/README.md) | Обнаружение плагинов, реестр, конформанс и границы ядра |
| [portfolio/](portfolio/README.md) | Портфель, его проверка, отчёт и расчёт портфеля |
