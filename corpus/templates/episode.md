---
id: R16
kind: episode
title: Ставка 17 % и обвал рубля
region: ru
country: RU
start: 2014-12
window: {before_months: 3, after_months: 12}
known_from: 2015-12-31
factors: [rate.policy, fx, rate.sovereign, spread.credit, bank.liquidity]
data: [D1, D2]
flags: []
sources: []
# Заполняет msa-corpus episodes по рядам D1–D5, руками не править.
movements:
  - factor: fx
    series: <ряд из series.toml>
    unit: <п. п. для ставок и спредов, % для курсов, индексов и цен>
    resolution: <day, week, month или quarter>
    base: {date: , value: }
    extreme: {date: , value: }
    change:
    days_to_extreme:
    days_to_recover:
unmeasured:
  bank.liquidity: нет рыночного ряда, описывается в хронологии
---

# R16. Ставка 17 % и обвал рубля

## Хронология

Даты и решения. Каждое утверждение со ссылкой на источник того времени: `[S1/2014 p. 5]`.

## Движения факторов

<!-- Генерируется скриптом из movements. -->

| Фактор | Ряд | База | Экстремум | Изменение | Дней до экстремума | Дней до восстановления |
|---|---|---|---|---|---|---|

## Механизм

Через какие каналы шок дошёл до рынков и балансов. Со ссылками.

## Чем полезен как аналогия

Какие условия делают эпизод похожим на новое событие и где аналогия перестаёт работать.

## Источники

- 
