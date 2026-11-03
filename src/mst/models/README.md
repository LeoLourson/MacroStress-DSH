# mst.models

Жизненный цикл версий моделей: то, что инструмент `model` запускает задачей. Сами модели —
плагины (`mst.plugins`); здесь они оцениваются на истории workspace, публикуются версиями,
перечисляются и описываются.

```powershell
python -m mst.models estimate-macro [MODEL_ID] --as-of DATE --store DIR --models DIR
python -m mst.models estimate-satellites --macro SHA256 --store DIR --models DIR --drivers FILE [--model QUANTITY=MODEL_ID]
python -m mst.models list --models DIR
python -m mst.models describe --version SHA256 --models DIR
python -m mst.models diagnose --version SHA256 --models DIR
python -m mst.models catalog
```

| Действие | Что делает |
|---|---|
| `estimate-macro` | Оценивает макромодель (именованную или единственную установленную) на панели, известной на дату, и публикует версию |
| `estimate-satellites` | Собирает бандл сателлитной версии макроверсии: каждая величина с драйверами оценивается назначенной ей моделью — по декларации, а при двусмысленности явным `--model ВЕЛИЧИНА=МОДЕЛЬ`; печатает проверку каждого драйвера |
| `list`, `describe` | Версии обоих видов: модель, дата, обслуживаемые драйверы |
| `diagnose` | Диагностика сохранённой макроверсии без новой оценки: спектральный радиус каждой версии параметров по режимам с долями выше 1, 1.005, 1.01 и 1.02, выжившие версии параметров, доли режимов по годам и последнему месяцу, степени свободы по переменным, вес приора `P_0/(X'X+P_0)` по уравнениям |
| `catalog` | Установленные плагины моделей: макро-, сателлитные, позиционные; какая модель обслуживает каждую величину |

Назначение величины: модель, объявившая её, оценивает по умолчанию; величина, объявленная
несколькими моделями (плагин в разработке рядом с установленным), требует явного имени.
Величина без модели записывается в бандл как `unserved` и не оценивается.

Тесты — [../../../tests/drivers/test_bundle.py](../../../tests/drivers/test_bundle.py),
[../../../tests/macro/test_version.py](../../../tests/macro/test_version.py) и
[../../../tests/models/test_diagnose.py](../../../tests/models/test_diagnose.py).
