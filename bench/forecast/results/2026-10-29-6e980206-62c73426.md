# Forecast check

Portfolio `fifth-test`, data to 2026-10-29, macro `6e98020647ce`, satellites `62c734262eab`, commit `92aa8d7`, 2000 paths, seed 20261115. History from 2015-01-01; band 90%. Verdict: **fail**.

## Gates

| gate | rule | verdict |
|---|---|---|
| macro width, 1 m | at least 80% of 22 rows pass: 20 pass | ok |
| macro width, 3 m | at least 80% of 23 rows pass: 18 pass | FAIL |
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
| macro coverage out of sample, all series, 3 m | 0.8 <= value <= 0.97 | ok |
| macro coverage out of sample, all series, 12 m | 0.8 <= value <= 0.97 | ok |
| shares coverage out of sample, all series, 1 m | 0.8 <= value <= 0.97 | ok |
| shares coverage out of sample, all series, 3 m | 0.8 <= value <= 0.97 | ok |
| shares coverage out of sample, all series, 12 m | 0.8 <= value <= 0.97 | FAIL |
| portfolio coverage out of sample, return, 1 m | 0.8 <= value <= 0.97 | ok |
| portfolio coverage out of sample, return, 3 m | 0.8 <= value <= 0.97 | FAIL |
| portfolio coverage out of sample, return, 12 m | 0.8 <= value <= 0.97 | FAIL |
| scenario admitted, cbr-cut-5pp-random, None m | the run is admitted | ok |
| scenario admitted, cbr-cut-5pp-market-reaction-random, None m | the run is admitted | ok |
| scenario admitted, cbr-target-6-7-random, None m | the run is admitted | ok |
| scenario admitted, bank-admin-top5-random, None m | the run is admitted | ok |
| scenario admitted, cn-bank-payments-stop-random, None m | the run is admitted | ok |
| scenario admitted, dividends-a-random, None m | the run is admitted | ok |
| scenario admitted, dividends-b-random, None m | the run is admitted | ok |
| scenario admitted, fed-zero-gold-random, None m | the run is admitted | ok |
| scenario admitted, fuel-export-ban-random, None m | the run is admitted | ok |
| scenario admitted, hormuz-closure-2m-random, None m | the run is admitted | ok |
| scenario admitted, migrant-ban-random, None m | the run is admitted | ok |
| scenario admitted, sbp-outage-10d-random, None m | the run is admitted | ok |

## Macro: width of the h-month change, model / history

| variable | 1 m | 3 m | 12 m |
|---|---:|---:|---:|
| gl.brent | 0.98 ok | 0.77 ok | 1.12 ok |
| gl.fed_rate | 0.85 ok | 0.61 FAIL | 0.51 FAIL |
| gl.ust_10y | 0.82 ok | 0.91 ok | 0.86 ok |
| gl.vix | 0.80 ok | 0.70 FAIL | 0.95 ok |
| ru.key_rate | 1.62 FAIL | 1.09 ok | 0.91 ok |
| ru.ruonia | 1.42 FAIL | 1.12 ok | 0.89 ok |
| ru.ofz_1y | 0.95 ok | 0.95 ok | 0.92 ok |
| ru.ofz_5y | 0.86 ok | 0.92 ok | 0.95 ok |
| ru.ofz_10y | 0.96 ok | 0.92 ok | 0.97 ok |
| ru.spread_aa | 1.09 ok | 0.54 FAIL | 0.87 ok |
| ru.spread_a | 1.24 ok | 0.81 ok | 0.88 ok |
| ru.spread_bbb | 0.98 ok | 0.61 FAIL | 0.85 ok |
| ru.usdrub | 1.09 ok | 0.86 ok | 1.09 ok |
| ru.cnyrub | 1.17 ok | 0.90 ok | 1.09 ok |
| ru.cpi | 1.32 ok | 1.44 FAIL | 1.07 ok |
| ru.output | 1.03 ok | 0.78 ok | 1.39 ok |
| ru.unemployment | 1.18 ok | 1.11 ok | 0.74 FAIL |
| ru.real_wage | 0.83 ok | 0.86 ok | 1.22 ok |
| ru.retail | 1.11 ok | 0.78 ok | 1.00 ok |
| ru.credit | 0.75 ok | 0.89 ok | 0.83 ok |
| ru.mortgage_rate | 0.92 ok | 0.81 ok | 0.91 ok |
| ru.housing_price | - short | 1.15 ok | 1.08 ok |
| ru.imoex | 0.93 ok | 0.88 ok | 0.80 ok |

## Shares

| check | item | h | model | reference | n | verdict |
|---|---|---:|---:|---:|---:|---|
| width | share_price:bspb | 1 | 28.09 | 25.16 | 141 | ok |
| width | share_price:gazp | 1 | 25.11 | 24.74 | 141 | ok |
| width | share_price:lkoh | 1 | 18.21 | 18.56 | 141 | ok |
| width | share_price:ozon | 1 | 35.14 | 38.68 | 68 | ok |
| width | share_price:plzl | 1 | 33.67 | 29.50 | 141 | ok |
| width | share_price:rosn | 1 | 20.61 | 20.93 | 141 | ok |
| width | share_price:sibn | 1 | 19.83 | 19.27 | 141 | ok |
| width | share_price:sngs | 1 | 23.38 | 21.87 | 141 | ok |
| width | share_price:vtbr | 1 | 29.08 | 25.89 | 141 | ok |
| correlation | mean pair | 1 | 0.39 | 0.41 | 68 |  |
| width | share_price:bspb | 3 | 58.81 | 56.28 | 139 | ok |
| width | share_price:gazp | 3 | 48.22 | 52.64 | 139 | ok |
| width | share_price:lkoh | 3 | 36.35 | 41.44 | 139 | ok |
| width | share_price:ozon | 3 | 70.98 | 86.62 | 66 | ok |
| width | share_price:plzl | 3 | 70.47 | 60.35 | 139 | ok |
| width | share_price:rosn | 3 | 38.61 | 44.55 | 139 | ok |
| width | share_price:sibn | 3 | 40.23 | 40.07 | 139 | ok |
| width | share_price:sngs | 3 | 42.90 | 51.50 | 139 | ok |
| width | share_price:vtbr | 3 | 58.65 | 52.08 | 139 | ok |
| correlation | mean pair | 3 | 0.39 | 0.44 | 66 |  |
| width | share_price:bspb | 12 | 127.73 | 117.64 | 130 | ok |
| width | share_price:gazp | 12 | 103.08 | 93.50 | 130 | ok |
| width | share_price:lkoh | 12 | 80.76 | 88.96 | 130 | ok |
| width | share_price:ozon | 12 | 156.21 | 211.99 | 58 | FAIL |
| width | share_price:plzl | 12 | 153.89 | 153.32 | 130 | ok |
| width | share_price:rosn | 12 | 84.03 | 94.18 | 130 | ok |
| width | share_price:sibn | 12 | 90.48 | 87.27 | 130 | ok |
| width | share_price:sngs | 12 | 93.54 | 82.15 | 130 | ok |
| width | share_price:vtbr | 12 | 126.59 | 135.18 | 130 | ok |
| correlation | mean pair | 12 | 0.39 | 0.47 | 58 | ok |
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
| coverage out of sample | all series | 3 | 0.86 | 0.90 | 215 | ok |
| coverage out of sample | all series | 12 | 0.77 | 0.90 | 215 | FAIL |
| coverage out of sample | share_price:bspb | 12 | 0.91 | 0.90 | 22 |  |
| coverage out of sample | share_price:gazp | 12 | 0.63 | 0.90 | 27 |  |
| coverage out of sample | share_price:lkoh | 12 | 0.63 | 0.90 | 35 |  |
| coverage out of sample | share_price:ozon | 12 | 1.00 | 0.90 | 3 |  |
| coverage out of sample | share_price:plzl | 12 | 1.00 | 0.90 | 21 |  |
| coverage out of sample | share_price:rosn | 12 | 0.69 | 0.90 | 35 |  |
| coverage out of sample | share_price:sibn | 12 | 0.75 | 0.90 | 28 |  |
| coverage out of sample | share_price:sngs | 12 | 0.89 | 0.90 | 18 |  |
| coverage out of sample | share_price:vtbr | 12 | 0.81 | 0.90 | 26 |  |

## Portfolio

| check | item | h | model | reference | n | verdict |
|---|---|---:|---:|---:|---:|---|
| width | q95 - q05, % | 1 | 15.07 | 13.68 | 130 | ok |
| tail | ES95 loss, % | 1 | 9.16 | 11.74 | 130 | ok |
| quantile | q05, % | 1 | -6.23 | -6.28 | 130 |  |
| quantile | median, % | 1 | 0.73 | 1.08 | 130 |  |
| quantile | q95, % | 1 | 8.84 | 7.40 | 130 |  |
| mean | mean, % | 1 | 1.00 | 0.99 | 130 |  |
| width | q95 - q05, % | 3 | 30.19 | 29.41 | 130 | ok |
| tail | ES95 loss, % | 3 | 16.68 | 19.75 | 130 | ok |
| quantile | q05, % | 3 | -11.56 | -11.76 | 130 |  |
| quantile | median, % | 3 | 2.15 | 3.60 | 130 |  |
| quantile | q95, % | 3 | 18.64 | 17.65 | 130 |  |
| mean | mean, % | 3 | 2.61 | 3.08 | 130 |  |
| width | q95 - q05, % | 12 | 73.86 | 80.36 | 130 | ok |
| tail | ES95 loss, % | 12 | 28.12 | 31.10 | 130 | ok |
| quantile | q05, % | 12 | -21.00 | -26.25 | 130 |  |
| quantile | median, % | 12 | 8.19 | 10.11 | 130 |  |
| quantile | q95, % | 12 | 52.86 | 54.11 | 130 |  |
| mean | mean, % | 12 | 10.98 | 12.59 | 130 |  |
| coverage out of sample | return | 1 | 0.91 | 0.90 | 35 | ok |
| mean PIT | return | 1 | 0.54 | 0.50 | 35 |  |
| CRPS vs historical simulation | return | 1 | 2.42 | 2.40 | 35 |  |
| coverage out of sample | return | 3 | 0.77 | 0.90 | 35 | FAIL |
| mean PIT | return | 3 | 0.50 | 0.50 | 35 |  |
| CRPS vs historical simulation | return | 3 | 6.89 | 6.83 | 35 |  |
| coverage out of sample | return | 12 | 0.71 | 0.90 | 35 | FAIL |
| mean PIT | return | 12 | 0.49 | 0.50 | 35 |  |
| CRPS vs historical simulation | return | 12 | 16.92 | 16.14 | 35 |  |
| Acerbi-Szekely Z | loss | 12 | 5.71 | 0.00 | 35 |  |
| coverage of the historical simulation | return | 1 | 0.91 | 0.90 | 35 |  |
| coverage of the historical simulation | return | 3 | 0.89 | 0.90 | 35 |  |
| coverage of the historical simulation | return | 12 | 0.69 | 0.90 | 35 |  |

## Events: shift against the base, % of the portfolio

| scenario | admitted | median 1 m | median 3 m | median 12 m | peak median | ES95 loss 12 m | path ES95 loss |
|---|---|---:|---:|---:|---:|---:|---:|
| cbr-cut-5pp-random | ok | -0.00 | 7.60 | 8.09 | 8.51 (10 m) | -4.83 | -4.84 |
| cbr-cut-5pp-market-reaction-random | ok | -0.00 | 8.42 | 8.04 | 9.07 (10 m) | -5.73 | -6.41 |
| cbr-target-6-7-random | ok | 0.13 | -0.08 | -0.70 | -0.72 (11 m) | 0.90 | -0.29 |
| bank-admin-top5-random | ok | -0.00 | -0.12 | -0.17 | -0.25 (11 m) | 0.12 | 0.11 |
| cn-bank-payments-stop-random | ok | -0.00 | -0.18 | -0.06 | -0.51 (5 m) | -0.01 | 0.04 |
| dividends-a-random | ok | -0.00 | 1.06 | 1.09 | 1.30 (6 m) | -0.87 | -0.86 |
| dividends-b-random | ok | -0.00 | -1.13 | -1.03 | -1.13 (3 m) | 0.77 | 0.71 |
| fed-zero-gold-random | ok | -0.00 | -5.88 | -9.63 | -9.63 (12 m) | 5.92 | 5.69 |
| fuel-export-ban-random | ok | -0.00 | -2.31 | -0.00 | -2.31 (3 m) | 0.00 | 0.38 |
| hormuz-closure-2m-random | ok | -0.00 | 2.30 | -1.81 | 2.30 (3 m) | 0.84 | 0.41 |
| migrant-ban-random | ok | -0.00 | -0.00 | -0.00 | -0.00 (1 m) | 0.00 | 0.00 |
| sbp-outage-10d-random | ok | -0.00 | -0.00 | -0.00 | -1.04 (2 m) | 0.00 | 0.00 |

## Macro out of sample

| level | h | coverage | n | verdict |
|---|---:|---:|---:|---|
| macro | 1 | 0.88 | 770 | ok |
| macro | 3 | 0.83 | 770 | ok |
| macro | 12 | 0.81 | 770 | ok |
| shares | 1 | 0.91 | 216 | ok |
| shares | 3 | 0.86 | 215 | ok |
| shares | 12 | 0.77 | 215 | FAIL |
