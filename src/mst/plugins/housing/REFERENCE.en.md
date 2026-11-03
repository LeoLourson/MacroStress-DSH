# Real estate position model `housing.paths.v1`: reference

Written from the design documents, not from the code: `docs/reference/platform.md`, `docs/howto/plugins.md`, `docs/concept/formalization.md`, `src/mst/plugin_sdk/README.md`, `src/mst/drivers/registry.toml`, `plans/canon-alignment.md`, `plans/satellite-models.md` and the commit messages of the plugin. Where these sources are silent, the gap is listed under open questions and is not filled.

## 1. Purpose and place in the system

The model values a position of type `real_estate` (residential property) along the scenario paths of a driver `housing_price_*`: the average price of a square metre in a Russian region on the primary market (`housing_price_primary`) or the secondary market (`housing_price_secondary`), from Rosstat. The value change in each draw feeds the portfolio loss $\Delta_t$, its quantiles P5/P50/P95 and $\mathrm{ES}_{95\%}$ at the end of the horizon, by month and over the path, and through them the reverse stress test. The model is pure: it reads only the paths the scenario has already drawn.

## 2. Inputs

| Input | Meaning | Unit |
|---|---|---|
| `notional` | Current value of the position, $N$ | millions of the position currency |
| `currency` | Position currency $c$; the base currency is the rouble | ISO code |
| `codes.okato` | Region of the property; the platform matches it to a regional driver | — |
| `terms.market` | Housing market $m\in\lbrace\text{primary},\text{secondary}\rbrace$, named by `docs/reference/platform.md` as a model input; it selects the driver `housing_price_m` | — |
| driver `housing_price_m` | Path $h_{r,t}$ | roubles per square metre |
| macro path of the exchange rate | $X_{r,t}$ for $c\ne\text{RUB}$ | roubles per unit of $c$ |

$R$ is the number of draws, $H$ the horizon in months, $t=0$ the calculation month. A missing `market` is a refusal naming the field; `terms` has no silent defaults.

## 3. Valuation

No source gives the housing formula explicitly. The commit message "Port the calculation of positions" states that a position sets its notional and the model reads paths, and the SDK offers `relative`, the ratio of a path to its value at $t=0$; the equity model uses that ratio. The valuation consistent with these sources, marked as an open question below, is

$$
V_{r,t}=N\,X_{r,t}\,\frac{h_{r,t}}{h_{r,0}},
$$

with $X\equiv1$ for a rouble position. The value change at the end of the horizon is $\delta_r=V_{r,H}-V_{r,0}$ and the loss $L_r=-\delta_r$. The property is marked to the regional average price per square metre, so only the relative move of the market matters.

The model reads the last column of the paths as the end of the horizon, so the loss by month $t$ is the same calculation on paths cut at $t$ (`plans/canon-alignment.md`). Property has no maturity. A missing driver, for example a region without a series, leaves the position without a calculation, status `invalid`, with the quantity and code named.

## 4. Parameters

| Symbol | Meaning | Value | Source |
|---|---|---|---|
| $N$ | Notional | from the position | `Position.notional` |
| $m$ | Market | from `terms.market`; no default | `docs/reference/platform.md` |
| $h_{r,t}$ | Price per square metre path | from the satellite version | driver `housing_price_m` |
| $X_{r,t}$ | Exchange rate path | from the macro version | `fx.py` |
| sale discount, rent income, depreciation | — | not set by the design | — |

## 5. Outputs

A `CalculationResult` per the SDK contract: status `ok`, `needs_review` or `invalid`; `value_change` of length $R$ with $\delta_r$; metrics with id `housing.<name>`, as a value or P5/P50/P95, the loss metric also with ES95; driver flags added by the platform through `envelope`.

## 6. Assumptions and limits

- The property moves with the regional average: object quality, location within the region and liquidity are absent.
- The driver is quarterly (`span = 3` in `registry.toml`): the months between quarters are drawn by the satellite, and the series starts in 2021, so the history is short.
- Per `plans/satellite-models.md`, more than 90 % of housing paths leave the training range of a node, so the results are mostly extrapolation.
- The series is published with a lag of about 35 days and revised, so $h_{r,0}$ is an estimate.

## 7. Checks the code must pass

1. With `market = primary` the model requires `housing_price_primary`, with `secondary` it requires `housing_price_secondary`.
2. A missing or other `market` value is refused with the field named.
3. For a rouble position on fixed paths, $\delta_r=N\,(h_{r,H}/h_{r,0}-1)$, if the formula of the valuation section is confirmed.
4. Constant paths give $\delta_r=0$.
5. `value_change` has $R$ entries; repeat on the same paths is bit-identical.
6. Paths cut at month $t$ give the value change at month $t$.
7. A region (`okato`) without a driver gives `invalid` with the quantity and code named.
8. $h_{r,0}=0$ is refused with a diagnostic.
9. The loss metric carries P5, P50, P95 and ES95.

## 8. Open questions and contradictions

1. The valuation formula is not stated in any source; the price ratio is inferred from the SDK `relative` helper and the equity model.
2. Allowed values of `market`: the driver ids suggest `primary` and `secondary`; the exact spelling in `terms` is not stated.
3. Regional matching: whether `okato` is the only code used, and whether a missing region falls back to the national level (`registry.toml` says a driver without a series takes the parameters of the level above), is not stated for positions.
4. Transaction costs and the time to sell in stress are not set.
5. Currency: a non-rouble position with a rouble-denominated driver has the same double-counting question as for equity.
6. Sign convention of the loss and the metric names: not stated explicitly.
