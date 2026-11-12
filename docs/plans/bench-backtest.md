# Вводные бэктеста моделей в `bench/`

Цель: всё, что задаёт проверку моделей на истории, лежит в `bench/model-backtest/` одним файлом,
код бэктеста читает его по явному пути, а эпизод бэктеста — отдельная сущность со своим месяцем
начала и ссылкой на карточку корпуса. Копий файла нет ни в пакете, ни в workspace.

## Что сейчас задаёт бэктест

| Вводная | Где лежит | Кто читает | Решение |
|---|---|---|---|
| Эпизоды R13–R32: `constrained`, `driving`, `half`, `source`, `reason`, `sources` | `src/mst/metrics/episodes.toml` | `metrics.backtest.run`, `metrics.rolling.run` через `load_episodes()` без пути | в `bench/model-backtest/episodes.toml` |
| `draws`, `seed`, `after_months` | там же | backtest, rolling | вместе с эпизодами: это параметры той же проверки |
| `[rolling]`: шаг сетки, `validation_from`, `price_events` | там же | rolling, цена в backtest | вместе с эпизодами |
| `[[exclusions]]`: спреды MOEX до 2021-01 не сравнимы | там же | backtest, при подсчёте ячеек | вместе с эпизодами: правило проверки, какие месяцы не судить |
| Месяц начала эпизода | `corpus/manifest.json`, поле `start` карточки | backtest через `--corpus FILE` | в запись эпизода, поле `start`; карточка остаётся ссылкой |
| Каталог драйверов, портфель хвоста | `workspace/drivers/catalog.json`, `workspace/portfolios/` | backtest с `--satellites`, `--portfolio` | остаются в workspace: это данные, а не вводные проверки |
| Пороги ворот G2–G5 и S1–S6 | `mst.models.gates`, `scripts/satellite_gates.py` | скрипты ворот | остаются: это стоп-условия планов ремонта, у них нет набора с ответами |

Все макромодели (`bvar.sv`, `msvar.markov`, `rw.joint`, `expert.elicited`) и сателлиты
проверяются на одном плане, поэтому файл один на все модели, а модель выбирает аргумент
`MODEL_ID`.

## Решения, которые фиксируются до кода

1. Файл переносится целиком под прежним именем: `bench/model-backtest/episodes.toml`. Имя
   остаётся, потому что его называют отчёты (`meta.episodes_toml`), README и тесты.
2. Эпизод получает поле `start = "YYYY-MM"` и необязательное поле `card` — ключ карточки
   корпуса. Для R13–R32 `card` равен `id`, а `start` переписывается из манифеста один раз.
   Бэктест перестаёт читать манифест, и аргумент `--corpus` у `python -m mst.metrics backtest`
   уходит.
3. Связь с корпусом проверяет тест: у эпизода с `card` карточка есть в `corpus/manifest.json`
   и её `start` совпадает с `start` эпизода. Расхождение — ошибка теста, а не предупреждение:
   иначе агент, читающий карточку, и бэктест говорят о разных окнах.
4. Путь к плану обязателен: `load_episodes(path)` без значения по умолчанию, у `backtest` и
   `rolling` аргумент `--episodes FILE`. MCP-инструмент `model(action="backtest")` передаёт
   `<корень репозитория>/bench/model-backtest/episodes.toml`, корень он берёт по расположению
   исходников пакета. Если файла там нет (пакет поставлен из PyPI без репозитория), инструмент
   отвечает ошибкой с текстом «запустите `python -m mst.metrics backtest --episodes FILE`».
5. Хеш плана. Перенос без правки содержимого сохраняет хеш, поэтому шаг 1 делается отдельным
   коммитом. Поле `start` меняет содержимое и хеш: `--against` со старыми отчётами после шага 2
   откажет («different episode plan»), и эталонный отчёт для сравнения надо пересчитать.

## Шаги

### Шаг 1. Перенос файла и явный путь

- `git mv src/mst/metrics/episodes.toml bench/model-backtest/episodes.toml`, содержимое не
  меняется.
- `src/mst/metrics/episodes.py`: удалить `EPISODES_PATH`, `load_episodes(path: Path)` без
  значения по умолчанию.
- `src/mst/metrics/backtest.py`: `run(..., episodes: Path, ...)` вызывает
  `load_episodes(episodes)`; в `_decision_block` вместо `plan.path or EPISODES_PATH` — `plan.path`.
- `src/mst/metrics/rolling.py`: то же для `run`.
- `src/mst/metrics/cli.py`: `--episodes FILE` обязателен у `backtest` и `rolling`.
- `src/mst/mcp.py`, `_backtest_argv`: добавить `--episodes` с путём из решения 4.
- `tests/metrics/test_backtest.py`: вместо `monkeypatch.setattr(runner, "load_episodes", ...)`
  передавать путь плана в `run`.
- Проверка: новый тест `tests/metrics/test_cli.py::test_backtest_needs_the_episode_plan` падает
  до правки и проходит после; быстрый набор тестов зелёный; `meta.episodes_toml` отчёта на
  перенесённом файле равен хешу до переноса (тест сравнивает `_file_digest` двух путей одного
  содержимого).
- Готово, если `git grep EPISODES_PATH` пуст и `python -m mst.metrics backtest --only R31 ...
  --episodes bench/model-backtest/episodes.toml` пишет отчёт с прежним `episodes_toml`.

### Шаг 2. Эпизод со своим месяцем начала

- `EpisodeCheck`: поля `start: date` (первое число месяца, строка `YYYY-MM` разбирается при
  чтении) и `card: Identifier | None = None`.
- `bench/model-backtest/episodes.toml`: у R13–R32 дописать `start` и `card`; значения берёт
  одноразовый скрипт из `corpus/manifest.json`, скрипт не коммитится.
- `backtest.run`: `starts` берётся из плана; удалить чтение манифеста и аргумент `corpus`;
  `--corpus` уходит из CLI и из `_backtest_argv`.
- `tests/test_bench.py`: тест решения 3 и подстановка дефекта (неверный `start` ловится).
- Проверка: быстрый набор зелёный; `backtest --only R31` даёт те же даты расчёта, что до шага 2.
- Готово, если `git grep "manifest" src/mst/metrics` пуст.

### Шаг 3. Документы

- `bench/README.md` и `.en.md`: строка `model-backtest/`, абзац «Что осталось вне `bench/`»
  удаляется.
- `src/mst/metrics/README.md` и `.en.md`: ссылки на `episodes.toml`, команды с `--episodes`
  вместо `--corpus`.
- `corpus/README.md` и `.en.md`: эпизодный бэктест больше не читает манифест.
- `docs/concept/architecture.md` и `.en.md`: строка «`corpus/manifest.json` нужен эпизодному
  бэктесту» убирается.
- `CHANGELOG.md` и `.en.md`: запись о переносе, новом аргументе и смене хеша плана.
- Проверка: `tests/test_docs_translation.py` зелёный, `git grep -n "metrics/episodes.toml"`
  находит только CHANGELOG.
- Обновление 'README.md' и '.en.md' для показательной ссылки на результаты bench для повышения доверия к проекту.

## Риски и стоп-условия

- Если после шага 1 хеш отчёта на одном и том же содержимом изменился, останавливаемся:
  значит, хеш зависит от пути, и решение 5 неверно.
- Если `backtest --only R31` после шага 2 даёт другую дату расчёта, останавливаемся и сверяем
  `start` с карточкой.
- Копии `episodes.toml` в `experiments/macro-models/trees/*` вне git и планом не трогаются; их
  код читает собственный пакет.

## Что план не делает

- Не меняет состав эпизодов, их половины и разметку.
- Не переносит пороги ворот и не трогает `scripts/`.
- Не вводит набор ожидаемых решений для `scenario-from-corpus/`.
