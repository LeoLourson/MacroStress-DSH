# Forecast check

Portfolio `fifth-test`, data to 2026-10-29, macro `48092ef56382`, satellites `1c6a184e72e3`, commit `b9b2e78`, 2000 paths, seed 20261115. History from 2015-01-01; band 90%. Verdict: **fail**.

## Gates

| gate | rule | verdict |
|---|---|---|
| macro width, 1 m | at least 80% of 22 rows pass: 19 pass | ok |
| macro width, 3 m | at least 80% of 23 rows pass: 17 pass | FAIL |
| macro width, 12 m | at least 80% of 23 rows pass: 21 pass | ok |
| shares width, 1 m | at least 80% of 9 rows pass: 9 pass | ok |
| shares width, 3 m | at least 80% of 9 rows pass: 9 pass | ok |
| shares width, 12 m | at least 80% of 9 rows pass: 8 pass | ok |
| shares correlation, mean pair, 12 m | |model - history| <= 0.15 | ok |
| portfolio width, q95 - q05, %, 1 m | 0.75 <= model / history <= 1.4 | ok |
| portfolio tail, ES95 loss, %, 1 m | 0.75 <= model / history <= 1.4 | ok |
| portfolio width, q95 - q05, %, 3 m | 0.75 <= model / history <= 1.4 | ok |
| portfolio tail, ES95 loss, %, 3 m | 0.75 <= model / history <= 1.4 | ok |
| portfolio width, q95 - q05, %, 12 m | 0.75 <= model / history <= 1.4 | ok |
| portfolio tail, ES95 loss, %, 12 m | 0.75 <= model / history <= 1.4 | ok |
| macro coverage out of sample, all series, 1 m | 0.8 <= value <= 0.97 | ok |
| macro coverage out of sample, all series, 3 m | 0.8 <= value <= 0.97 | FAIL |
| macro coverage out of sample, all series, 12 m | 0.8 <= value <= 0.97 | FAIL |
| shares coverage out of sample, all series, 1 m | 0.8 <= value <= 0.97 | ok |
| shares coverage out of sample, all series, 3 m | 0.8 <= value <= 0.97 | ok |
| shares coverage out of sample, all series, 12 m | 0.8 <= value <= 0.97 | FAIL |
| portfolio coverage out of sample, return, 1 m | 0.8 <= value <= 0.97 | ok |
| portfolio coverage out of sample, return, 3 m | 0.8 <= value <= 0.97 | ok |
| portfolio coverage out of sample, return, 12 m | 0.8 <= value <= 0.97 | FAIL |

## Macro: width of the h-month change, model / history

| variable | 1 m | 3 m | 12 m |
|---|---:|---:|---:|
| gl.brent | 1.05 ok | 0.76 ok | 1.10 ok |
| gl.fed_rate | 0.95 ok | 0.68 FAIL | 0.56 FAIL |
| gl.ust_10y | 0.86 ok | 0.95 ok | 0.91 ok |
| gl.vix | 0.86 ok | 0.75 FAIL | 1.02 ok |
| ru.key_rate | 1.72 FAIL | 1.15 ok | 0.94 ok |
| ru.ruonia | 1.44 FAIL | 1.16 ok | 0.94 ok |
| ru.ofz_1y | 0.98 ok | 0.98 ok | 0.97 ok |
| ru.ofz_5y | 0.86 ok | 0.92 ok | 0.96 ok |
| ru.ofz_10y | 0.98 ok | 0.93 ok | 0.98 ok |
| ru.spread_aa | 1.15 ok | 0.56 FAIL | 0.92 ok |
| ru.spread_a | 1.34 ok | 0.88 ok | 0.96 ok |
| ru.spread_bbb | 1.04 ok | 0.67 FAIL | 0.93 ok |
| ru.usdrub | 1.11 ok | 0.75 FAIL | 0.91 ok |
| ru.cnyrub | 1.16 ok | 0.81 ok | 0.93 ok |
| ru.cpi | 1.54 FAIL | 1.46 FAIL | 0.97 ok |
| ru.output | 1.08 ok | 0.81 ok | 1.45 FAIL |
| ru.unemployment | 1.04 ok | 1.11 ok | 0.79 ok |
| ru.real_wage | 0.82 ok | 0.85 ok | 1.24 ok |
| ru.retail | 1.13 ok | 0.82 ok | 1.01 ok |
| ru.credit | 0.79 ok | 0.92 ok | 0.83 ok |
| ru.mortgage_rate | 1.08 ok | 0.90 ok | 0.94 ok |
| ru.housing_price | - short | 1.13 ok | 1.07 ok |
| ru.imoex | 0.92 ok | 1.01 ok | 0.88 ok |

## Shares

| check | item | h | model | reference | n | verdict |
|---|---|---:|---:|---:|---:|---|
| width | share_price:bspb | 1 | 27.79 | 25.16 | 141 | ok |
| width | share_price:gazp | 1 | 24.46 | 24.74 | 141 | ok |
| width | share_price:lkoh | 1 | 17.94 | 18.56 | 141 | ok |
| width | share_price:ozon | 1 | 34.95 | 38.68 | 68 | ok |
| width | share_price:plzl | 1 | 33.39 | 29.50 | 141 | ok |
| width | share_price:rosn | 1 | 20.79 | 20.93 | 141 | ok |
| width | share_price:sibn | 1 | 19.60 | 19.27 | 141 | ok |
| width | share_price:sngs | 1 | 23.77 | 21.87 | 141 | ok |
| width | share_price:vtbr | 1 | 29.85 | 25.89 | 141 | ok |
| correlation | mean pair | 1 | 0.43 | 0.41 | 68 |  |
| width | share_price:bspb | 3 | 58.93 | 56.28 | 139 | ok |
| width | share_price:gazp | 3 | 49.74 | 52.64 | 139 | ok |
| width | share_price:lkoh | 3 | 38.02 | 41.44 | 139 | ok |
| width | share_price:ozon | 3 | 70.95 | 86.62 | 66 | ok |
| width | share_price:plzl | 3 | 71.25 | 60.35 | 139 | ok |
| width | share_price:rosn | 3 | 40.40 | 44.55 | 139 | ok |
| width | share_price:sibn | 3 | 43.00 | 40.07 | 139 | ok |
| width | share_price:sngs | 3 | 45.94 | 51.50 | 139 | ok |
| width | share_price:vtbr | 3 | 61.40 | 52.08 | 139 | ok |
| correlation | mean pair | 3 | 0.43 | 0.44 | 66 |  |
| width | share_price:bspb | 12 | 133.95 | 117.64 | 130 | ok |
| width | share_price:gazp | 12 | 108.86 | 93.50 | 130 | ok |
| width | share_price:lkoh | 12 | 86.72 | 88.96 | 130 | ok |
| width | share_price:ozon | 12 | 158.13 | 211.99 | 58 | FAIL |
| width | share_price:plzl | 12 | 152.49 | 153.32 | 130 | ok |
| width | share_price:rosn | 12 | 88.35 | 94.18 | 130 | ok |
| width | share_price:sibn | 12 | 91.41 | 87.27 | 130 | ok |
| width | share_price:sngs | 12 | 95.19 | 82.15 | 130 | ok |
| width | share_price:vtbr | 12 | 134.87 | 135.18 | 130 | ok |
| correlation | mean pair | 12 | 0.43 | 0.47 | 58 | ok |
| holdout RMSE vs AR(1) | share_price:bspb | - | 13.56 | 39.93 | 222 |  |
| holdout RMSE vs AR(1) | share_price:gazp | - | 6.56 | 19.19 | 222 |  |
| holdout RMSE vs AR(1) | share_price:lkoh | - | 10.43 | 16.86 | 222 |  |
| holdout RMSE vs AR(1) | share_price:ozon | - | 19.73 | 22.43 | 199 |  |
| holdout RMSE vs AR(1) | share_price:plzl | - | 31.71 | 32.38 | 222 |  |
| holdout RMSE vs AR(1) | share_price:rosn | - | 11.37 | 15.36 | 222 |  |
| holdout RMSE vs AR(1) | share_price:sibn | - | 11.57 | 17.41 | 222 |  |
| holdout RMSE vs AR(1) | share_price:sngs | - | 7.95 | 26.88 | 222 |  |
| holdout RMSE vs AR(1) | share_price:vtbr | - | 14.90 | 21.76 | 222 |  |
| coverage out of sample | all series | 1 | 0.91 | 0.90 | 216 | ok |
| coverage out of sample | all series | 3 | 0.84 | 0.90 | 215 | ok |
| coverage out of sample | all series | 12 | 0.77 | 0.90 | 215 | FAIL |
| coverage out of sample | share_price:bspb | 12 | 0.91 | 0.90 | 22 |  |
| coverage out of sample | share_price:gazp | 12 | 0.59 | 0.90 | 27 |  |
| coverage out of sample | share_price:lkoh | 12 | 0.66 | 0.90 | 35 |  |
| coverage out of sample | share_price:ozon | 12 | 1.00 | 0.90 | 3 |  |
| coverage out of sample | share_price:plzl | 12 | 1.00 | 0.90 | 21 |  |
| coverage out of sample | share_price:rosn | 12 | 0.71 | 0.90 | 35 |  |
| coverage out of sample | share_price:sibn | 12 | 0.75 | 0.90 | 28 |  |
| coverage out of sample | share_price:sngs | 12 | 0.89 | 0.90 | 18 |  |
| coverage out of sample | share_price:vtbr | 12 | 0.81 | 0.90 | 26 |  |

## Portfolio

| check | item | h | model | reference | n | verdict |
|---|---|---:|---:|---:|---:|---|
| width | q95 - q05, % | 1 | 14.97 | 13.68 | 130 | ok |
| tail | ES95 loss, % | 1 | 9.76 | 11.74 | 130 | ok |
| quantile | q05, % | 1 | -6.02 | -6.28 | 130 |  |
| quantile | median, % | 1 | 1.01 | 1.08 | 130 |  |
| quantile | q95, % | 1 | 8.95 | 7.40 | 130 |  |
| mean | mean, % | 1 | 1.21 | 0.99 | 130 |  |
| width | q95 - q05, % | 3 | 32.72 | 29.41 | 130 | ok |
| tail | ES95 loss, % | 3 | 18.67 | 19.75 | 130 | ok |
| quantile | q05, % | 3 | -12.77 | -11.76 | 130 |  |
| quantile | median, % | 3 | 2.33 | 3.60 | 130 |  |
| quantile | q95, % | 3 | 19.95 | 17.65 | 130 |  |
| mean | mean, % | 3 | 2.88 | 3.08 | 130 |  |
| width | q95 - q05, % | 12 | 78.99 | 80.36 | 130 | ok |
| tail | ES95 loss, % | 12 | 30.86 | 31.10 | 130 | ok |
| quantile | q05, % | 12 | -22.69 | -26.25 | 130 |  |
| quantile | median, % | 12 | 7.60 | 10.11 | 130 |  |
| quantile | q95, % | 12 | 56.30 | 54.11 | 130 |  |
| mean | mean, % | 12 | 11.40 | 12.59 | 130 |  |
| coverage out of sample | return | 1 | 0.91 | 0.90 | 35 | ok |
| mean PIT | return | 1 | 0.54 | 0.50 | 35 |  |
| CRPS vs historical simulation | return | 1 | 2.38 | 2.40 | 35 |  |
| coverage out of sample | return | 3 | 0.80 | 0.90 | 35 | ok |
| mean PIT | return | 3 | 0.50 | 0.50 | 35 |  |
| CRPS vs historical simulation | return | 3 | 6.86 | 6.83 | 35 |  |
| coverage out of sample | return | 12 | 0.69 | 0.90 | 35 | FAIL |
| mean PIT | return | 12 | 0.49 | 0.50 | 35 |  |
| CRPS vs historical simulation | return | 12 | 17.04 | 16.14 | 35 |  |
| Acerbi-Szekely Z | loss | 12 | 6.09 | 0.00 | 35 |  |
| coverage of the historical simulation | return | 1 | 0.91 | 0.90 | 35 |  |
| coverage of the historical simulation | return | 3 | 0.89 | 0.90 | 35 |  |
| coverage of the historical simulation | return | 12 | 0.69 | 0.90 | 35 |  |

## Macro out of sample

| level | h | coverage | n | verdict |
|---|---:|---:|---:|---|
| macro | 1 | 0.88 | 770 | ok |
| macro | 3 | 0.80 | 770 | FAIL |
| macro | 12 | 0.77 | 770 | FAIL |
| shares | 1 | 0.91 | 216 | ok |
| shares | 3 | 0.84 | 215 | ok |
| shares | 12 | 0.77 | 215 | FAIL |
