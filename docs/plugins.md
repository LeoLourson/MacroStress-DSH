# Плагины: новый тип актива

Платформа расширяется плагинами. Плагин — пакет `src/mst/plugins/<plugin_id>/` с
объявлением `plugin = Plugin(...)`, его моделью, демо-кейсом, тестами и README. Платформа
находит пакет при старте и проверяет объявление целиком. Ядро при этом не меняется: если
для нового типа актива пришлось править `mst.engine`, значит, в SDK не хватает точки
расширения — это отдельное изменение SDK, а не часть плагина.

Контракты, на которые опирается плагин, описаны в [platform.md](platform.md).

## 1. Быстрый старт

```powershell
python -m mst.plugin new gold --label "Золото"      # src/mst/plugins/gold/
python -m mst.plugin check gold                     # контракт и демо-расчёт
python -m mst.plugin demo-run gold                  # P5/P50/P95 на демо-путях
python -m mst.plugin check                          # все установленные плагины
```

`new` создаёт пакет, который **сразу проходит** `check`:

```
src/mst/plugins/<id>/
  __init__.py        экспорт MODEL и plugin
  plugin.py          plugin = Plugin(...) и демо-кейс
  model.py           модель: requirements → validate → calculate
  tests/             тесты пакета
  README.md          что даёт плагин и что в нём менять
```

Пакет в `src/mst/plugins/` платформа подключает при следующем запуске. Пакет вне
репозитория создаётся с `--dest DIR` (в `DIR/<id>`) и проверяется там же:
`python -m mst.plugin check --path DIR/<id>`; `demo-run --path DIR/<id>` считает его
демо-кейс без установки.

## 2. Объявление

```python
from mst.plugin_sdk import DemoCase, FixedPaths, Plugin, Position

plugin = Plugin(
    plugin_id="gold",
    version="0.1.0",
    sdk=">=1.0,<2.0",
    label="Золото",
    description="Позиция в золоте на путях его цены.",
    models=(MODEL,),
    demo=(DemoCase(...),),
)
```

| Поле | Что значит |
|---|---|
| `plugin_id` | Имя пакета и пространство идентификаторов: `gold.paths.v1` |
| `version` | Версия плагина `x.y.z`; попадает в описание прогона |
| `sdk` | Диапазон версий SDK; несовместимый плагин блокирует старт с его именем |
| `label`, `description` | Как плагин называется и что даёт |
| `models` | Модели `CalculationModel`; их id начинаются с `<plugin_id>.` |
| `demo` | Демо-кейсы, на которых `check` и `demo-run` считают позицию |

Идентификатор модели — `<plugin_id>.<имя>`; тип актива глобален и не может обслуживаться
двумя моделями. Эти правила проверяет реестр при старте.

## 3. Модель

Модель — наследник `mst.plugin_sdk.CalculationModel` из трёх шагов:

```python
class GoldModel(CalculationModel):
    model_id = "gold.paths.v1"
    version = "0.1.0"
    label = "Золото на путях цены"
    instrument_types = ("gold_position",)
    terms_schema = Terms
    metrics = ("gold.value_current", "gold.pnl", "gold.loss")
    required_metrics = metrics

    def requirements(self, position, options):
        """Что нужно из путей: драйверы и макропеременные. Пути ещё не прочитаны."""

    def validate(self, prepared):
        """Суждение о подготовленных входах; error останавливает позицию."""

    def calculate(self, prepared, context):
        """CalculationResult на общих путях прогона; только чистые вычисления."""
```

- **`requirements`** возвращает `DataRequirements`: `DriverRequirement` (величина и, если
  нужно, объект или коды позиции) и имена макропеременных. Платформа находит драйвер в
  каталоге сателлитной версии; драйвер, которого нет, оставляет позицию без расчёта и
  называет величину.
- **`validate`** возвращает `Diagnostic`-и. `error` превращает позицию в `invalid` с
  причиной; она остаётся в прогоне и в покрытии отчёта.
- **`calculate`** читает только `prepared` (разрешённые драйверы, стоимость на `t = 0`,
  `terms`) и `context` (пути, дата, горизонт, сид, базовая валюта). Модель не читает файлы,
  сеть, workspace и не разыгрывает случайные числа. Результат собирается `envelope`;
  обязательные метрики должны быть в результате, длина `value_change` — числу розыгрышей.

Схема `terms` — pydantic-модель с `extra="forbid"`. Её поля — входы аналитика (дюрация,
LGD, рынок недвижимости), и у них нет молчаливых умолчаний: отсутствующее обязательное
поле — отказ с именем поля.

## 4. Демо-кейс

```python
DEMO = FixedPaths(
    drivers={"gold_price:demo": np.array([[100.0, 102.0, 104.0], [100.0, 98.0, 96.0]])},
    start=date(2026, 9, 1),
)

DemoCase(
    name="demo",
    position=Position(id="demo", type="gold_position", notional=100.0),
    paths=DEMO,
    drivers={"gold_price": "gold_price:demo"},
)
```

`drivers` — карта «величина требования → id драйвера в путях». `FixedPaths` — обычный
`Paths`: у него есть `draws`, `horizon`, `months`, `variables`, `macro`, `driver`, `flags`.
Демо-кейс самодостаточен: ни workspace, ни оценённых моделей, ни сети он не требует.

`check` считает каждый кейс через диспетчер платформы дважды: повтор должен дать те же
числа, позиция из `completes` — рассчитаться, обязательные метрики — быть на месте.
`demo-run` печатает P5/P50/P95, ES95 для потерь и пометки результата.

## 5. Проверка и приёмка

`python -m mst.plugin check` проверяет:

- идентификатор плагина, версии плагина и моделей, диапазон SDK, непустые `label` и
  `models`;
- имена моделей и их принадлежность пространству плагина, уникальность типов актива;
- `metrics`: нет повторов, `required_metrics` входят в `metrics`;
- схему `terms`: у неё есть JSON Schema;
- демо-кейс: его тип обслуживает модель, все обязательные драйверы отображены в пути,
  макропеременные есть в путях;
- расчёт демо-кейса через `engine.run`: статус, обязательные метрики, детерминизм.

Плагин без демо-кейса не проходит проверку: без него `demo-run` нечем проверить. Тесты
самого плагина лежат в `src/mst/plugins/<id>/tests/` и запускаются после установки пакета:

```powershell
pytest src/mst/plugins/gold/tests
```

Общая проверка установленных плагинов — `tests/plugins/test_conformance.py`; она же
падает, если ядро начнёт импортировать конкретный тип актива
(`tests/plugins/test_boundary.py`).

## 6. Границы

Плагин импортирует `mst.plugin_sdk`, `mst.contracts` и собственный пакет. Модель не
импортирует `mst.engine`, `mst.plugins`, `mst.runs`, `mst.portfolio` и другие плагины.
Ядро не называет ни одного конкретного типа актива: `mst.engine` спрашивает реестр, а
реестр собирается из пакетов `mst.plugins.<id>`.

Если модели не хватает возможности SDK (функции путей, поля результата, пометки), её
добавляют в SDK отдельным изменением ядра и повышают минорную версию SDK. Плагин,
которому нужна новая возможность, объявляет новый диапазон `sdk`.
