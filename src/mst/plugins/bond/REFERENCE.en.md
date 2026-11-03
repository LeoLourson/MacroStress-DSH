# Bond position model `bond.paths.v1`: reference

Written from the design documents, not from the code: `docs/reference/platform.md`, `docs/howto/plugins.md`, `docs/concept/formalization.md`, `src/mst/plugin_sdk/README.md`, `src/mst/drivers/registry.toml`, `plans/canon-alignment.md`, `plans/satellite-models.md` and the commit messages of the plugin. Where these sources are silent, the gap is listed under open questions and is not filled.

## 1. Purpose and place in the system

The model values a position of type `bond` (a corporate bond) along the scenario paths of its driver `bond_spread`, the spread of the bond to the OFZ curve. The value change in each draw feeds the portfolio loss $\Delta_t$, its quantiles P5/P50/P95 and $\mathrm{ES}_{95\%}$ at the end of the horizon, by month and over the path, and through them the reverse stress test. The model is pure: it reads only the paths the scenario has already drawn.

## 2. Inputs

| Input | Meaning | Unit |
|---|---|---|
| `notional` | Current value of the position, $N$ | millions of the position currency |
| `currency` | Position currency $c$; the base currency is the rouble | ISO code |
| `codes` | `secid`, `isin`, `inn`, ...; the platform matches the bond to a driver by them, and a bond without its own series takes the driver of its rating group (`group_aa`, `group_a`, `group_bbb`) | — |
| `terms.duration` | Duration $D$ of the bond, named by `docs/reference/platform.md` as a model input | years |
| driver `bond_spread` | Path $s_{r,t}$ | percentage points |
| macro path of the exchange rate | $X_{r,t}$ for $c\ne\text{RUB}$; for USD `ru.usdrub` | roubles per unit of $c$ |

$R$ is the number of draws, $H$ the horizon in months, $t=0$ the calculation month. `terms` has no silent defaults: a missing `duration` is a refusal naming the field (`docs/howto/plugins.md`).

## 3. Valuation

The commit message "Port the calculation of positions" fixes the loss of a bond as `duration*ds/100`: a first-order duration approximation of the price change to a yield change, where the change of yield is the change of spread (Fabozzi, *Bond Markets, Analysis and Strategies*, chapter on price volatility). With $\Delta s_{r,t}=s_{r,t}-s_{r,0}$ in percentage points, the value in the position currency is

$$
P_{r,t}=N\left(1-\frac{D\,\Delta s_{r,t}}{100}\right),
$$

and in roubles $V_{r,t}=P_{r,t}\,X_{r,t}$, with $X\equiv 1$ for a rouble position. The value change at the end of the horizon is $\delta_r=V_{r,H}-V_{r,0}$ and the loss is $L_r=-\delta_r$; for a rouble position $L_r=N D\,\Delta s_{r,H}/100$. The division by 100 converts percentage points to a fraction.

The model reads the last column of the paths as the end of the horizon, so the loss by month $t$ is the same calculation on paths cut at $t$ (`plans/canon-alignment.md`). Currency conversion goes through `plugin_sdk/fx.py`.

The design is silent on what happens at maturity before $H$, on coupons, on the shortening of duration over the horizon, on a move of the OFZ curve and on default; these are open questions. A missing driver leaves the position without a calculation, status `invalid`, with the quantity and code named.

## 4. Parameters

| Symbol | Meaning | Value | Source |
|---|---|---|---|
| $N$ | Notional | from the position | `Position.notional` |
| $D$ | Duration | from `terms.duration`; no default | `docs/reference/platform.md`, commit message |
| $s_{r,t}$ | Spread path | from the satellite version | driver `bond_spread`, unit p. p. |
| $X_{r,t}$ | Exchange rate path | from the macro version | `fx.py` |
| $C$ | Convexity | not set by the design | — |
| maturity, coupon, recovery | — | not set by the design | — |

## 5. Outputs

A `CalculationResult` per the SDK contract: status `ok`, `needs_review` or `invalid`; `value_change` of length $R$ with $\delta_r$; metrics with id `bond.<name>`, as a value or P5/P50/P95, the loss metric also with ES95; driver flags added by the platform through `envelope`.

## 6. Assumptions and limits

- First order in the spread: for large spread moves the duration approximation overstates losses on a rising spread and understates gains on a falling one, because convexity is positive for an option-free bond.
- The OFZ curve itself is fixed in the valuation: only the spread moves, although the satellite reads `ru.ofz_5y` and `ru.key_rate` as regressors.
- Duration is constant over the horizon.
- Bond spreads are modelled in levels; per `plans/satellite-models.md` about 56 % of paths leave the training range, and a bond in `group_aa` barely reacts to a shock to `ru.spread_bbb` (loading 0.04).
- A spread rise above $100/D$ p. p. makes the value negative; the design does not cap it.

## 7. Checks the code must pass

1. For a rouble position on fixed paths, $\delta_r=-N D\,(s_{r,H}-s_{r,0})/100$.
2. $D=5$, $N=100$, $\Delta s=2$ p. p. gives $\delta=-10$.
3. Constant spread paths give $\delta_r=0$.
4. The loss is linear in $D$ and in $\Delta s$: doubling either doubles $L_r$.
5. A missing `terms.duration` is refused with the field named; there is no default.
6. For a USD position, $\delta_r=P_{r,H}X_{r,H}-N X_{r,0}$ with $X$ from `ru.usdrub`.
7. `value_change` has $R$ entries; repeat on the same paths is bit-identical.
8. Paths cut at month $t$ give the value change at month $t$.
9. An unresolved `bond_spread` driver gives `invalid` with the quantity named.
10. The loss metric carries P5, P50, P95 and ES95.

## 8. Open questions and contradictions

1. Duration type (Macaulay or modified) is not stated; the formula `duration*ds/100` is exact to first order only for modified duration.
2. Convexity: standard literature adds $\tfrac12 C(\Delta y)^2$; the design has only the linear term and no `convexity` field.
3. Maturity before the horizon end: whether the position is redeemed at par, rolled or kept at duration $D$ is not set.
4. Coupons and carry over the horizon are not set; the formula gives zero value change on a constant spread.
5. Default and recovery of the issuer: not set; default enters only through the spread path.
6. Moves of the risk-free curve (`ru.ofz_5y`, `ru.key_rate`) are in the satellite regressors but not in the valuation; whether a rate shock should change bond value directly is not stated.
7. Sign convention of the loss and the metric names, as for the other position models: not stated explicitly.
8. Order of currency conversion (spread loss in currency $c$, then the rate) is inferred from the general `fx.py` rule, not stated for bonds.
