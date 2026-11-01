# Первый сеанс

Путь первого сеанса повторяет схему: сначала заготавливаются модели по истории, затем
формируется сценарий, затем идёт расчёт. После подключения сервера клиенту доступны семь
инструментов: `jobs`, `series`, `corpus`, `scenario`, `run`, `portfolio`, `model`; полный
справочник — [../reference/tools.md](../reference/tools.md).

Долгие операции запускаются задачами: инструмент сразу возвращает `job`, ход работы виден через
`jobs(action="status", id=...)` и `jobs(action="log", id=...)`.

## 1. История: ряды

1. `jobs(action="state")` — что лежит в workspace: версии моделей, индекс корпуса, число задач.
2. `series(action="update")` — загрузить все макроряды. Для FRED нужен ключ в переменной
   окружения `FRED_API_KEY`.
3. `jobs(action="status", id=...)` и `jobs(action="log", id=...)` — ждать конца задачи.

## 2. Макромодель, уровень 0

4. `model(kind="macro", action="estimate", as_of="2026-10-29")` — оценить макромодель на данных,
   известных к этой дате. Установлены две, `bvar.sv` и `msvar.markov`, поэтому `model_id`
   называет нужную. Получится версия с хешем в имени.

## 3. Сателлиты, уровни выше

5. `series(action="update", quantity="share_price")` — найти объекты величины и загрузить их
   ряды; так же для `bond_spread`, `overdue_share`, `housing_price_primary`,
   `housing_price_secondary`.
6. `model(kind="satellites", action="estimate", version=<хеш макроверсии>)` — собрать
   бандл сателлитов этой макроверсии: каждая величина оценивается назначенной ей моделью
   (по умолчанию `hier.gibbs`; `assignment` назовёт другую).
7. `model(kind="satellites", action="describe", version=<хеш>)` — какие величины прошли
   проверку вне выборки и почему остальные отклонены.

## 4. Сценарий

8. `corpus(action="search", query=..., as_of=...)` — фрагменты корпуса, известные на дату;
   `corpus(action="index")` пересобирает индекс задачей.
9. `scenario(action="save", id=..., spec={...})` — сохранить сценарий: нарратив, причины,
   условия, воздействия и того, кто его утвердил; `scenario(action="validate", id=...)` — назвать всё, что
   мешает его посчитать.

## 5. Расчёт и результат

10. `portfolio(action="save", id=..., spec={...})` — сохранить портфель;
    `portfolio(action="check", id=..., scenario=...)` — проверить его против сценария.
11. `run(action="start", scenario=..., portfolio=..., draws=..., seed=..., targets=...)` —
    запустить расчёт; `run(action="result", id=...)` — прочитать условный прогноз целевых
    показателей ($P_5$/$P_{50}$/$P_{95}$ по месяцам) и отчёт портфеля с ES95, ценой
    сценария $\mathrm{dist}(s)$ и покрытием.

## 6. Метрики качества

12. `model(kind="macro", action="backtest", as_of=...)` — проверка на исторических эпизодах:
    CRPS, покрытие полос прогноза и допуск моделей (`mst.metrics`).

## Что читать дальше

Метод за шагами расчёта — [../concept/formalization.md](../concept/formalization.md),
контракты чисел — [../reference/platform.md](../reference/platform.md).
