# mst.plugin_sdk

Поверхность, по которой пишется модель позиции: что она объявляет, что читает из расчёта и
что возвращает. Модель импортирует только этот пакет и `mst.contracts`; workspace, сеть и
сами модели макро и сателлитов ей недоступны. Порядок работы с плагином —
[plugins.md](../../../docs/howto/plugins.md), контракты платформы — [platform.md](../../../docs/reference/platform.md).

| Файл | Что делает |
|---|---|
| `model.py` | `CalculationModel` (требования → проверка → расчёт), `Position`, `RunOptions`, `PreparedInputs`, `DataRequirements`, `DriverRequirement`, `Diagnostic`, `CalculationContext` |
| `paths.py` | Протокол `Paths`: `R` розыгрышей на месяцы `t = 0 … H`, исходные единицы; `FixedPaths` для тестов; `relative` и `change` — отношение и разность к `t = 0` |
| `results.py` | `CalculationResult`, `MetricValue`, `Quantiles` и хелперы `money`, `scalar`, `distribution`, `quantiles`, `envelope`, `expected_shortfall`; P5/P50/P95 идут вместе, у потерь ещё ES95 |
| `fx.py` | Курс валюты позиции к базовой валюте из макропутей: `exposure`, `spot`, `ratio`, `variable` |
| `plugin.py` | `Plugin` — объявление пакета; `DemoCase` — самодостаточный кейс для `check` и `demo-run` |
| `testing.py` | `check_plugin` — конформанс-проверка через диспетчер платформы; `calculate_demo` — демо-расчёт для тестов плагина |
| `version.py` | `SDK_VERSION` и `compatible`: диапазон версий, который объявляет плагин |

Модель чистая: она не читает файлы и сеть и не разыгрывает случайные числа, а берёт готовые
реализации из `context.paths`. Пометки драйверов — надбавка сценария, параметры от объекта
уровнем выше, приблизительная дата публикации — доходят до результата через `envelope`, а
не через код модели. Возможность, которой модели не хватает, добавляют в SDK отдельным
изменением с повышением минорной версии; плагин объявляет новый диапазон `sdk`.

Тесты — [../../../tests/plugins](../../../tests/plugins/README.md) (конформанс и реестр) и
[../../../tests/engine](../../../tests/engine/README.md) (диспетчер и модели).
