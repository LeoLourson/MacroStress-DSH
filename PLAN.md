# План переноса на DSH стек: шаги 8–11

Состояние на 29.10.2026: шаги 0–9 в `main` (`mst` собирается; series, macro, drivers,
сателлиты, корпус с индексом и поиском, сценарий и его пути, расчёт позиций и портфель с
отчётом работают). Остались шаги 10–11.
Один шаг — одна ветка, зелёные проверки, коммит с телом «Зачем / Что / Как проверено /
Что не сделано», слияние `git merge --no-ff` в `main`.

Шаги 6 и 7 слиты: ветки `corpus` (ce55175) и `scenario` (5f04ec5), слияния `78cd6b3` и
`9c17c47`; тогда же в `main` вошли ветки другой сессии `docs` и `fixes`. Разделы 6 и 7 ниже
оставлены как запись того, что и почему сделано: команды и проверки в них описывают уже
собранное и повторять их не нужно.

## Как начинать и проверять любой шаг

```powershell
cd C:\Users\LeoLeon\Documents\MacroStress\macrostress-dsh
git switch -c <шаг> main

# лёгкий набор тестов (201): тяжёлая оценка BVAR гоняется только после правки модели
.\.venv\Scripts\python.exe -m pytest -q --ignore=tests/macro/test_estimate.py `
    --ignore=tests/macro/test_forecast.py --ignore=tests/macro/test_identify.py
.\.venv\Scripts\python.exe -m ruff check --no-cache .
```

- `ruff` только с `--no-cache`: на этой машине он не может создать свой кэш.
- Перенос — механический: `msa.` → `mst.`, затем правка имён каталогов и путей. Сверка со
  старым репозиторием (`..\macrostress-core`) — диффом на одинаковых данных.
- Данные живут в `workspace/` (вне git): `series/`, `drivers/`, `models/`, `corpus/`,
  `portfolios/`, `scenarios/`, `runs/`. Каталоги уже объявлены в `mst.store.Workspace`.
- Макро-версия для прогонов лежит в `workspace/models/macro/a3233197…` (скопирована из
  старого `.data/models`), сателлиты — в `workspace/models/satellites/`.
- Инструменты MCP добавляются в `src/mst/mcp.py` рядом с `jobs`, `series`, `model`; каждая
  долгая операция запускается задачей (`_start`) на своём ресурсе, а не выполняется в сервере.
- Оценка моделей запускается только через `python -m mst.*`: точка входа `__main__` ставит
  `OPENBLAS_NUM_THREADS=1`. Без этого мелкие плотные solve идут в 20 потоков и оценка
  сателлитов занимает часы вместо минут.

## Шаг 6. corpus: документы, фрагменты, индекс, поиск — сделано

**Зачем.** Сценарий собирается из источников: агент ищет фрагменты корпуса по нарративу и
ссылкам, а эпизоды проверяют макромодель. Без корпуса шаги 7 и 8 нечем наполнить.

**Что переносится.** `src/msa/corpus/{fetch,convert,index,series,episodes,embed}.py` →
`src/mst/corpus/`; `manifest.py` уже перенесён. `src/msa/cli/corpus.py` →
`src/mst/corpus/cli.py`; `corpus/README.md`, `corpus/manifest.json`, `corpus/series.toml`,
`corpus/eval/` — в корень нового репозитория (данные — в `workspace/corpus/`).

**Что меняется.**
- Импорты `msa.*` → `mst.*`; `--corpus DIR` вместо `--data`.
- CLI: `python -m mst.corpus check|fetch|add|convert|series|episodes|index|search|eval|all`,
  обязательные `--corpus` и (для `search`) `--as-of`.
- Инструмент `corpus(action=...)`: `search` (query, as_of), `get` (chunk id, as_of),
  `list` (что в манифесте), `add` (ключ и файл по URL). `index` и `fetch` — задачи на
  `corpus` (`mst.corpus index --corpus …`).
- `store.state()` уже читает `corpus/index/state.json` — формат состояния не менять.

**Проверка.** Перенесённые `tests/corpus/` зелёные; `index` дважды подряд даёт тот же
`chunks.jsonl` (идемпотентность); `search --as-of` не возвращает фрагменты, опубликованные
позже; инструмент `corpus` отвечает на search/get без сети.

**Коммит.** «Port the corpus: documents, chunks, the index and its search».

**Сделано.** `src/mst/corpus/` (fetch, convert, series, episodes, index, embed, tool, cli),
`python -m mst.corpus <cmd> --corpus DIR` и инструмент `corpus` (search, get, list, add, index,
fetch). В git — `corpus/{README.md,manifest.json,series.toml,eval/,templates/}`; тексты,
оригиналы и индекс — данные `workspace/corpus`, как модели и ряды. `index` пишет
`chunks.jsonl`, векторы и `index/state.json` (`as_of`, `chunks`, `sha256`) — это читает
`store.state`. На реальном корпусе: 7036 фрагментов, второй `index` даёт тот же файл и 0 новых
векторов, recall@5 = 0,271.

## Шаг 7. scenario: нарратив, ограничения, шоки, надбавки — сделано

**Зачем.** Прогон — чистая функция от сценария $(N, C, D, O)$, версий моделей, даты $t=0$ и
сида. Шаг вводит сценарий и сборку путей.

**Что переносится.** `src/msa/engine/scenarios.py` (распределения факторов и шоков),
`src/msa/engine/scenario_set.py` (розыгрыши, Соболь, сид) → `src/mst/scenario/set.py` и
`src/mst/scenario/shocks.py`; `src/msa/contracts/scenario.py` уже в
`src/mst/scenario/contracts.py` — расширить до `Scenario`, `Constraint`, `Overlay`.

**Что меняется.**
- Сценарий — файл `workspace/scenarios/<id>.json`: `narrative`, `sources`,
  `constraints` (веса по переменным и месяцам, границы в шкале $y$), `driving_shocks`,
  `overlays`, `horizon`, `macro`, `satellites`.
- `mst.scenario.build(spec, macro_version, satellites_version, corpus)` → `Paths`:
  условный прогноз `mst.macro` с ограничениями, затем `mst.drivers.simulate` на тех же
  розыгрышах; тяжесть $\delta$ — из безусловного прогноза той же версии.
- Проверки: неизвестная переменная, месяц вне горизонта, `low > high`, ранг $Z$, надбавка
  на необслуживаемый драйвер — отказ с именем причины.
- Инструмент `scenario(action=list|get|save|validate)`; `run` (шаг 8) берёт сценарий по id.

**Проверка.** Перенесённые тесты розыгрышей (соболевские точки, повтор с тем же сидом —
те же числа); сборка путей на синтетической макро-версии даёт заданные ограничениям пути
до машинной точности; `validate` называет каждую невыполнимую связку.

**Коммит.** «Port the scenario: constraints, shocks, overlays and the paths they draw».

**Сделано.** `src/mst/scenario/{spec,build,tool}.py`: файл `scenarios/<id>.json` — контракт
сценария плюс хеши макроверсии и сателлитов; `build(spec, model, satellites, draws=…, seed=…)`
даёт `Paths` — условный прогноз с ограничениями, переменные обратно в исходные единицы, пути
всех обслуживаемых драйверов на тех же розыгрышах с надбавками, $\delta$ из безусловного
прогноза (без ограничений — `None`). `validate` называет неизвестную переменную и шок,
невыполнимое ограничение, надбавку на необслуженный драйвер с причиной, сателлиты другой
макроверсии и фрагмент корпуса позже даты расчёта. Инструмент `scenario` — list, get, save,
validate. На реальных данных (`a3233197` / `3121fdc2`): 23 переменные × 205 драйверов,
ограничение на сумму ставки за 12 месяцев держится до 2,1e-12, повтор побитно, $\delta$ = 6,984.

Соболевские розыгрыши `engine/scenarios.py` и `ScenarioSet` не переносились: они разыгрывают
подтверждённые распределения факторов, а здесь розыгрыш идёт из постериора макромодели
(`mst.macro.forecast`); их id, граф и `plugin_sdk` относятся к шагам 8–10.

## Шаг 8. assets-run: модели позиций и расчёт на путях

**Зачем.** Ядро расчёта: тип актива объявляет, какие входы ему нужны, платформа их находит
и считает позицию на путях $R \times (H+1)$.

**Что переносится.** `src/msa/engine/{dispatch,model_catalog,quantities,status}.py` →
`src/mst/engine/`; `src/msa/plugin_sdk/{model,paths,calc,results,specs}.py` →
`src/mst/plugin_sdk/`; модели позиций из `src/msa/plugins/` — по одной на шаг: акция
(`equity`), облигация (`bond`), депозит/кредит, недвижимость.

**Что меняется.**
- `Paths` — протокол из `plugin_sdk/paths.py` (макропеременные и драйверы в исходных
  единицах, пометки); платформа передаёт один и тот же объект всем моделям.
- `engine/dispatch.py` готовит входы по `DataRequirements`, при отсутствии драйвера позиция
  не считается, а результат называет недостающий драйвер.
- Инструмент `run(action=start|status|result|list)` — задача на `runs`; результат в
  `workspace/runs/<id>/` (мета, пути, вклад позиций).

**Проверка.** Модель акции на синтетических путях даёт ту же цену, что формула в тесте;
позиция без драйвера не считается и называет драйвер; повтор прогона с тем же сидом даёт
те же числа побитно.

**Коммит.** «Port the calculation of positions: the engine, the plugin SDK and the first
asset types».

**Сделано.** `src/mst/plugin_sdk/` — протокол `Paths` (макропеременные и драйверы в своих
единицах, пометки) и `FixedPaths` для тестов, `Position` (тип, валюта, номинал в миллионах,
коды объекта, `terms`), `DataRequirements`/`DriverRequirement`, `RunOptions`,
`PreparedInputs`, `CalculationContext`, `CalculationResult` и хелперы перцентилей;
`src/mst/engine/` — каталог моделей, `dispatch` (схема `terms`, разрешение драйверов
`Registry.match` среди обслуживаемых, проверка того, что модель вернула), `run` и `status`;
`src/mst/plugins/` — четыре модели: акция (`share_price`), облигация (`bond_spread`,
дюрация), кредит и депозит (`overdue_share`, LGD), недвижимость (`housing_price_*`, ОКАТО).
`src/mst/runs.py` пишет `runs/<id>/` (request, meta, paths.npz, results, changes) и
запускается как `python -m mst.runs <id>`; инструмент `run` — start, status, result, list;
у задач появился ресурс `runs`. На реальных данных (`a3233197` / `3121fdc2`) портфель из
четырёх позиций прошёл `check` без замечаний и дал отчёт; пометки `approximate_data`
дошли до результата. Не сделано: каталог моделей статичен (обнаружение — шаг 10); отчёт,
портфель и его инструменты — шаг 9; значения снэпшота (цена, дивиденды) не читаются —
позицию задаёт номинал, а модель читает пути.

## Шаг 9. portfolio: портфель, сопоставление, отчёт

**Зачем.** Прогон запускается для портфеля: позиции, их объёмы, сопоставление с объектами
реестра драйверов и отчёт с $P_5$, $P_{50}$, $P_{95}$ и тяжестью $\delta$.

**Что переносится.** `src/msa/db/repositories/portfolios.py` (формат и проверки),
`src/msa/engine/{portfolio,exposure,aggregation}.py`, `src/msa/api/portfolio_draft.py`
(разбор черновика) → `src/mst/portfolio/`.

**Что меняется.**
- Портфель — `workspace/portfolios/<id>.json`: позиции с типом актива, номиналом и кодами
  объекта (`secid`, `isin`, `inn`, `okved`, `okato`); драйвер находится `Registry.match`.
- `mst.portfolio.run(portfolio, scenario, …)` — полный прогон: пути → позиции → агрегат →
  перцентили и $\delta$; всё пишется в `workspace/runs/<id>/`, повтор даёт те же числа.
- Инструменты `portfolio(action=list|get|save|delete|check|exposure)` и `run` из шага 8;
  `jobs(state)` показывает число портфелей, сценариев и прогонов.

**Проверка.** Портфель из трёх позиций на синтетическом сценарии даёт те же числа, что
старый `run_batch`; повтор побитно совпадает; позиция с несуществующим объектом отклоняется
с именем кода; отчёт содержит $P_5$, $P_{50}$, $P_{95}$ и $\delta$.

**Коммит.** «Port the portfolio: positions, matching, aggregation and the run report».

**Сделано.** `src/mst/portfolio/` — `Portfolio` (имя, базовая валюта, позиции уникальных id),
`tool` (list, get, save, delete, check, exposure), `report` (агрегат). `check` грузит
сценарий и обе версии и называет каждую причину: неизвестный тип, `terms`, макропеременную,
которой нет в версии, объект, который не покрывает ни один драйвер (с кодом), отклонённый
драйвер с причиной проверки сателлитов. `exposure` считает стоимость на месяц расчёта по
панели макроверсии. Запрос прогона принимает `portfolio` или `positions`; прогон портфеля
пишет `summary.json`: потери $P_5/P_{50}/P_{95}$ и ES95 по сумме внутри розыгрыша,
$\delta$ и вероятности из тяжести сценария, покрытие с исключёнными позициями и причиной.
Статус портфеля — `ok`, `needs_review` или `incomplete` (исключённая позиция сверх 5 %
стоимости или неизмеримая стоимость). `runs.result` возвращает отчёт рядом с позициями.
Проверено: 258 тестов (включая 27 новых), ruff чист; отчёт на синтетическом сценарии
сходится с формулой по путям, повтор побитно тот же, ограниченный сценарий даёт
$\delta > 0$; настоящие `a3233197` / `3121fdc2` — портфель из четырёх типов активов,
`check` без замечаний, отчёт с P5/P50/P95 и пометками `approximate_data`. Расхождение
с планом: сравнение со старым `run_batch` невозможно — у него другой вход (наборы
сценариев и снэпшот), поэтому числа сверены с формулой на тех же путях.

## Шаг 10. extension: тип актива как расширение

**Зачем.** Новый тип актива добавляется пакетом, не правкой ядра: `msa-plugin new|check|demo-run`,
документация и контрактные тесты.

**Что переносится.** `src/msa/cli/plugin.py` → `src/mst/plugin.py`;
`src/msa/platform/registry.py` → `src/mst/plugins.py` (обнаружение пакетов
`mst.plugins.<id>` с `plugin = Plugin(...)`); `src/msa/plugin_sdk/testing.py`,
`src/msa/plugin_sdk/version.py`; `docs/plugins.md` и `docs/platform.md`.

**Что меняется.** Шаблон `msa-plugin new <id>` создаёт пакет с `plugin.py`, `model.py`,
`tests/`; `check` проверяет контракт и `ui-manifest` не нужен (веба нет); `demo-run` считает
позицию на демо-путях. Ядро не ветвится по типу актива.

**Проверка.** `new` → `check` → `demo-run` проходит на свежем пакете; сломанный пакет
отклоняется с именем ошибки; `pytest tests/plugins` зелёный; ни один модуль `mst.engine`
не импортирует конкретный тип актива.

**Коммит.** «Make an asset type an extension: discovery, the plugin command and its checks».

**Сделано.** `src/mst/plugin_sdk/{version,plugin,testing}.py` — версия SDK `1.0.0` и
диапазоны совместимости, объявление `Plugin` с моделями и демо-кейсом (`DemoCase`:
позиция, `FixedPaths`, карта «величина → драйвер»), `check_plugin` (контракт, прогон демо
через `engine.run`, детерминизм) и `calculate_demo`; `src/mst/plugins/__init__.py` —
обнаружение пакетов `mst.plugins.<id>`, проверка объявлений вместе и реестр, из которого
`mst.engine.model_catalog` строит каталог; у четырёх встроенных моделей появились `plugin.py`
с демо-кейсами; `src/mst/plugin.py` и `src/mst/templates/plugin/` — `python -m mst.plugin
new|check|demo-run`. `check` называет каждую проблему по имени, `demo-run` печатает
P5/P50/P95 и ES95; свежий пакет проходит `new → check → demo-run`, сломанный отклоняется.
Расхождение с планом: модуль `src/mst/plugins.py` не может существовать рядом с пакетом
`mst.plugins`, поэтому обнаружение и реестр лежат в `mst/plugins/__init__.py`; `ui-manifest`
не переносился — веба нет. Проверено: 313 лёгких тестов (45 новых в `tests/plugins/`), ruff
чист; `tests/plugins/test_boundary.py` не даёт `mst.engine` импортировать конкретный тип.

## Шаг 11. skills: скиллы и промпт DSH

**Зачем.** Работа идёт из окна DSH: агент должен знать, как собрать сценарий из корпуса,
как запустить прогон и как прочитать отчёт.

**Что переносится.** `src/msa/prompts/`, `src/msa/agents/*` (описания шагов и правила, не
LangGraph) → `dsh/skills/`; `dsh/index.js` (раздел промпта), `dsh/locale/{ru,en}.json`,
карточка плагина в Settings → Plugins.

**Что меняется.** Три скилла: `scenario-from-corpus` (как искать фрагменты и собирать
$N$, $C$, $D$, $O$), `run-report` (как читать $P_5$/$P_{50}$/$P_{95}$ и пометки), `drivers`
(что значит «сателлит отвергнут» и как обновить каталог). Промпт-раздел показывает
`workspace.json`: макро- и сателлитную версию, индекс корпуса, число портфелей, сценариев и
прогонов, незавершённые задачи.

**Проверка.** После перезапуска DSH в карточке видны инструменты, раздел промпта и список
скиллов; `jobs(state)` и `workspace.json` совпадают; скилл открывается и называет команды.

**Коммит.** «Teach DSH the workflow: the skills, the prompt section and the bundle card».

**Сделано.** `dsh/skills/{scenario-from-corpus,run-report,drivers}.md` — три скилла с
командами и правилами: сборка $N$, $C$, $D$, $O$ из фрагментов корпуса и разница
`save`/`validate`; чтение $P_5$/$P_{50}$/$P_{95}$, ES95, $\delta$, покрытия и пометок;
каталог драйверов, оценка сателлитов и причины отказа проверки. `dsh/index.js` показывает в
промпте пять строк из `workspace.json`: четвёртая — число портфелей, сценариев, прогонов и
задач в работе, пятая — с чего начать; скиллы регистрируются на время жизни плагина.
Карточка в Settings → Plugins подписана в обеих локалях. `dsh/test/smoke.mjs` без запуска
DSH подставляет фейковый контекст, регистрирует скиллы, строит пять строк раздела из
временного `workspace.json` и сверяет карточку; проверка `node dsh/test/smoke.mjs` —
`ok: 3 skills, 5 prompt lines, fallback, re-read, disposal, both locales`. Обновлены
`dsh/README.md`, `docs/install.md` и раздел про промпт в `docs/architecture.md`. `jobs(state)` пишет тот же
словарь, который читает раздел: `workspace/workspace.json`.

## Что уже решено по сателлитам (не переоткрывать)

- Доля прошлого значения в цепи драйвера ограничена `RHO_MAX = 0.99`: взрывной корень
  ($\rho > 1$) не имеет конечного пути, и матрица точности перестаёт быть положительно
  определённой. Усечение — по условному нормальному распределению, не обрезкой.
- Первый месяц цепи берётся из стационарного распределения, а не из диффузного априори:
  диффузное априори теряется в округлении на длинных путях (141 месяц и больше).
- Путь заканчивается последним месяцем, чей `x` известен панели: квартальная цена,
  опубликованная раньше своих регрессоров, оценивается по последний полный месяц, а не
  отбрасывается целиком.
- `python -m mst.drivers` ставит `OPENBLAS_NUM_THREADS=1` до импорта numpy.

## Открытые вопросы

1. На сколько месяцев вперёд хватает данных для `housing_price_*`: история начинается
   около 2020-12, проверка вне выборки съедает 24 месяца — стоит ли просить Росстат
   глубже или принять короткую историю.
2. Нужен ли флаг `--spec` у `python -m mst.drivers`, чтобы гонять проверку на малом
   сэмплере без правки `spec.toml`.
3. `mst.macro` тоже считает много мелких матриц: проверить, нужен ли ему тот же
   `OPENBLAS_NUM_THREADS=1` (сейчас не ставится).
