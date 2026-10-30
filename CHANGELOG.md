# Changelog

## 0.2.0 — модели как плагины

Репозиторий приведён к схеме системы стресс-тестирования: выбор сателлитной модели
существенно обобщён — теперь это подключаемая модель на каждую величину, а текущие модели
(BVAR и иерархический сателлит) стали обычными плагинами и подключаются наравне с
разрабатываемыми отдельно.

- **`mst.model_sdk`** — контракты макро- и сателлитных моделей: `ScenarioPlan` (горизонт,
  ограничения $C$, движущие шоки $D$), `AdditiveImpact` (прямое воздействие $o_l$ на любой
  уровень), контексты оценки и три шага модели (`estimate` → версия, `load` → модель
  версии, `simulate` → пути). Условная механика ограничений (усечённое нормальное,
  box-Gibbs, Антолин-Диас и др.) вынесена в `model_sdk.conditional` как общая.
- **Плагины трёх видов.** `Plugin` объявляет `models` (позиции), `macro` и `satellites`;
  обнаружение — пакеты `mst.plugins.<id>`, entry points группы `mst.plugins` и каталоги
  `MST_PLUGINS_PATH`. `python -m mst.plugin new --kind position|satellite|macro` создаёт
  рабочий пакет любого вида.
- **`mst.plugins.bvar`** — BVAR со стохастической волатильностью переезжает в плагин
  `bvar.sv`; ядро `mst.macro` — реестр переменных, панель и общее хранилище версий.
  Таблица шоков (множество $J$) принадлежит модели (`signs.toml`); сезонность переменных —
  свойство реестра данных, а не спецификации модели.
- **`mst.plugins.hier` + бандл.** Иерархический сателлит — плагин `hier.gibbs`; ядро
  `mst.drivers` — реестр величин, каталоги, `history` (подготовка данных) и `bundle`:
  сателлитная версия собирается из частей, своя модель на каждую величину, с назначением
  по декларации и явным `--model ВЕЛИЧИНА=МОДЕЛЬ` при двусмысленности. Симуляция
  диспетчеризует драйверы их моделям с их воздействиями и их срезом корреляции.
- **Сценарий.** `Overlay` → `Impact` с адресом: макропеременная или драйвер; воздействие
  на уровень 0 применяется до симуляции сателлитов. Прогон пишет условный прогноз целевых
  показателей `forecast.json` ($P_5/P_{50}/P_{95}$ по месяцам); `run` получает `targets`.
- **`mst.metrics`** — каркас качества раздела 5 схемы: ячейки $(e,t,q)$, CRPS с
  нормировкой на $\sigma_q(t)$, калибровка $u=\hat F(y)$, резкость, допуск (область
  значений, размах истории, вероятность ограничения в базовом расчёте), эталоны (случайное
  блуждание с дрейфом, наивный годовой), бутстрэп по эпизодам и эпизодный бэктест
  `python -m mst.metrics backtest` с кэшем версий эпизодов. Эпизоды — данные: конфиг в
  `mst/metrics/episodes.toml`. BVAR-специфичный бэктест покрытия удалён.
- **MCP.** `model` получает `model_id`, `assignment` (величина → сателлитная модель) и
  `kind=catalog`; `backtest` идёт в `mst.metrics`; `run` — параметр `targets`. Команды
  `python -m mst.macro` и `python -m mst.drivers estimate|check` заменены на
  `python -m mst.models estimate-macro|estimate-satellites`.

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-29

### Added

- Series store with release dates and loading from MOEX ISS, the Bank of Russia, Rosstat, and
  FRED.
- Macro model: BVAR with stochastic volatility, sign restrictions, conditional forecasts,
  scenario severity, and checks on historical episodes.
- Drivers: object catalogs, hierarchical satellites, out-of-sample checks, and driver paths on
  the same draws as the macro forecast.
- Corpus of sources: manifest with hashes, downloads, Markdown conversion, episode cards, index,
  and search that respects the publication date.
- Scenario file $(N, C, D, O)$ with validation and paths.
- Asset models for equities, bonds, credit and deposits, and housing.
- Portfolio checks and a run over scenario paths with $P_5$, $P_{50}$, $P_{95}$, ES95, and
  $\delta$ in the report.
- Asset types as plugins: `python -m mst.plugin new|check|demo-run`, contract tests, and a
  registry that keeps `mst.engine` free of asset-specific branches.
- MCP server with the tools `jobs`, `series`, `corpus`, `scenario`, `model`, `portfolio`, and
  `run`.
- DeepSeek Harness plugin: three skills, the prompt section built from `workspace.json`, and
  Russian and English locales.
