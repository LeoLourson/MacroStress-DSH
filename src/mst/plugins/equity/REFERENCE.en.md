# Equity position model `equity.paths.v1`: reference

Written from the design documents, not from the code: `docs/reference/platform.md`, `docs/howto/plugins.md`, `docs/concept/formalization.md`, `src/mst/plugin_sdk/README.md`, `src/mst/drivers/registry.toml`, `plans/canon-alignment.md`, `plans/satellite-models.md` and the commit messages of the plugin. Where these sources are silent, the gap is listed under open questions and is not filled.

## 1. Purpose and place in the system

The model values a position of type `common_stock` (a share listed on the Moscow Exchange) along the scenario paths of its driver `share_price`. The position value change in each draw feeds the portfolio loss $\Delta_t$, its quantiles P5/P50/P95 and $\mathrm{ES}_{95\%}$ at the end of the horizon, by month and over the path, and through them the reverse stress test, which relates scenario severity to losses. The model is pure: it reads only the paths the scenario has already drawn, so two positions in one share see the same draws and the order of positions does not change the result.

## 2. Inputs

| Input | Meaning | Unit |
|---|---|---|
| `notional` | Current value of the position, $N$ | millions of the position currency |
| `currency` | Position currency $c$; the portfolio base currency is the rouble | ISO code |
| `codes` | `secid`, `isin`, `inn`, `okved`, `okato`; the platform matches the share to a driver by them | — |
| `terms` | Model-specific inputs; the design names none for this model | — |
| driver `share_price` | Path $S_{r,t}$, $r=1,\dots,R$, $t=0,\dots,H$ | roubles per share |
| macro path of the exchange rate | $X_{r,t}$, roubles per unit of $c$, for $c\ne\text{RUB}$; for USD the design names `ru.usdrub` | roubles per unit of $c$ |

Here $R$ is the number of draws, $H$ the horizon in months, and $t=0$ the calculation month, the last month of history. All paths come in their source units with shape $(R, H+1)$.

## 3. Valuation

The position value in roubles in draw $r$ at month $t$ is

$$
V_{r,t}=N\,X_{r,t}\,\frac{S_{r,t}}{S_{r,0}},
$$

with $X_{r,t}\equiv 1$ for a rouble position. Notional is the value at $t=0$, so $V_{r,0}=N X_{r,0}$, and only the relative change of the price matters: the design does not read the price level or dividends from a snapshot (commit message "Port the calculation of positions"). The value change at the end of the horizon is

$$
\delta_r=V_{r,H}-V_{r,0},
$$

and the loss is $L_r=-\delta_r$. The model reads the last column of the paths as the end of the horizon, so the loss by month $t$ is the same calculation on paths cut at $t$ (`plans/canon-alignment.md`).

Currency conversion multiplies by the ratio of the rate path to its value at $t=0$ through `plugin_sdk/fx.py`; a currency without a rate path is named by `portfolio check` and the position is not calculated.

A share has no maturity, so the horizon end is the only terminal date. Issuer default is not modelled separately: it acts only through the driver path. A missing driver leaves the position without a calculation, with status `invalid` and the name of the quantity and code; the position stays in the coverage of the report.

## 4. Parameters

| Symbol | Meaning | Value | Source |
|---|---|---|---|
| $N$ | Notional | from the position | `Position.notional` |
| $S_{r,t}$ | Share price path | from the satellite version | driver `share_price` |
| $X_{r,t}$ | Exchange rate path | from the macro version | `fx.py`, `ru.usdrub` for USD |
| other | Dividends, lot size, liquidity haircut | not set by the design | — |

## 5. Outputs

A `CalculationResult` per the SDK contract: status `ok`, `needs_review` or `invalid`; `value_change` of length $R$ with $\delta_r$; metrics with an id of the form `equity.<name>`, each as a value or as P5/P50/P95, and the loss metric also with ES95; flags of the driver (`parameters_from_above`, `approximate_data`, `impact`), which the platform adds through `envelope`, not the model.

## 6. Assumptions and limits

- The value is linear in the price ratio: the model ignores dividends and corporate actions inside the horizon.
- The share price path comes from a satellite fitted in monthly log changes with an elasticity to macro variables; per `plans/satellite-models.md`, about 29 % of paths leave the training range of a node, so tail values extrapolate the satellite.
- A price of zero at $t=0$ makes the ratio undefined; the SDK helper `relative` refuses zero.
- Liquidation cost and market depth are not modelled.

## 7. Checks the code must pass

1. On fixed paths, $\delta_r=N\,(S_{r,H}/S_{r,0}-1)$ for a rouble position.
2. For a USD position, $\delta_r=N\,(X_{r,H}S_{r,H}/S_{r,0}-X_{r,0})$ with $X$ taken from `ru.usdrub`.
3. `value_change` has exactly $R$ entries.
4. Constant paths give $\delta_r=0$ in every draw.
5. Cutting the paths at month $t$ and calculating gives the value change at month $t$; the plugin has no separate code path for months.
6. Positions with the same `secid` and the same notional give identical `value_change`.
7. The same paths give bit-identical results on repeat; the model draws no random numbers and reads no files.
8. An unresolved `share_price` driver gives status `invalid` with the quantity named; `value_change` is not used in portfolio arithmetic.
9. $S_{r,0}=0$ is refused with a diagnostic, not turned into infinity or zero.
10. The loss metric carries P5, P50, P95 and ES95, where ES95 is the mean of the worst 5 % of losses.
11. A `terms` field outside the model schema is refused by name (`extra="forbid"`).

## 8. Open questions and contradictions

1. Sign convention of the loss: the platform reports `value_change` and losses; that loss equals $-\delta_r$ follows from the report summing losses per draw, but no source states the sign explicitly.
2. Metric names: `docs/howto/plugins.md` shows `value_current`, `pnl`, `loss` for a template plugin; the list for `equity` is not set by the design.
3. Currencies other than USD: `fx.py` reads "the rate from the macro paths", but only `ru.usdrub` is named.
4. Whether $X$ applies to the whole value (rate at $t$ times price ratio) or the rouble price of a foreign share already includes the rate: the driver unit is roubles per share, so for a rouble-quoted share with `currency` other than RUB the conversion may count the rate twice. The sources do not resolve this.
5. Dividends and the price snapshot: the commit says snapshot values are not read; whether this is a permanent design decision is not stated.
