---
name: drivers
description: Обновить каталог драйверов, оценить сателлиты и понять отказ проверки
whenToUse: когда позиция не считается из-за драйвера, когда нужен новый объект или новая версия сателлитов
---

# Драйверы и сателлиты

Драйвер — величина объекта, `<величина>:<объект>`. Конкретные id берите из
`model(kind="satellites", action="describe")` или `workspace/drivers/catalog.json`: в
текущем workspace это, например, `share_price:sber`, `bond_spread:ru000a0jxpg2`,
`overdue_share:stroitelstvo`, `housing_price_primary:okato_77000000000`. Сателлит — модель
драйвера от макропеременных; версия сателлитов оценивается на панели конкретной макроверсии.

## 1. Обновить каталог

`series(action="update", quantity="share_price")` запускает задачу на `series`: каталог
находит объекты величины, загружает их ряды и пишет `workspace/drivers/catalog.json`.
Так же для `bond_spread`, `overdue_share`, `housing_price_primary`,
`housing_price_secondary`. Новый объект источника появляется сам при следующем `update`;
упавший каталог оставляет прошлую запись величины.

`jobs(action="status", id=<job>)` и `jobs(action="log", id=<job>)` — ход задачи.

## 2. Оценить сателлиты

`model(kind="satellites", action="estimate", version=<макроверсия>)` — задача на `models`:
оценка на панели этой макроверсии, проверка вне выборки и запись версии в
`workspace/models/satellites/<hash>/`. `model(kind="satellites", action="list")` — версии.

## 3. Прочитать отказ

`model(kind="satellites", action="describe", version=<сателлиты>)` возвращает `served` и
`rejected`. У отклонённого драйвера есть `reason`, `satellite`, `baseline`, `forecasts`.
Причины:

- `too few observations before the holdout` — истории мало для проверки;
- `no observation of the holdout forecast` — в проверочном окне нечего прогнозировать;
- `not more accurate than without macro` — сателлит не точнее авторегрессии без
  макропеременных;
- `no checked driver above it` — у драйвера нет своей истории и нет проверенного объекта выше;
- `the driver above, <driver>, is not served` — параметры берутся у объекта, который сам
  отклонён.

Отказ — не ошибка данных и не ноль. Платформа не симулирует такой драйвер; позиция, которой
он нужен, остаётся `invalid` с `driver_missing:<величина>`, а отчёт исключает её с причиной и
показывает покрытие. Не выдавайте исключённую позицию за нулевой убыток.

## 4. Команды

Оценка запускается только через `python -m` из корня репозитория и тем же интерпретатором,
что в `dsh/cordis.patch.yml`: точка входа ставит `OPENBLAS_NUM_THREADS=1`, иначе мелкие
системы решаются в 20 потоков и оценка идёт часы вместо минут.

```powershell
python -m mst.drivers update share_price --store workspace/series --drivers workspace/drivers/catalog.json
python -m mst.models estimate-satellites --macro <hash> --store workspace/series --models workspace/models --drivers workspace/drivers/catalog.json
```

`estimate` пишет версию, `check` делает то же без записи. Новая величина — правка
`src/mst/drivers/registry.toml` и, если у источника нет каталога, класс в
`src/mst/drivers/catalogs/`; это изменение кода платформы, а не работа агента в workspace.
