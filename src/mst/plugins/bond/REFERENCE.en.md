# Bond position model `bond.paths.v1`: reference

The valuation follows `model.py` (version 2.0.0). The first version moved the price by the spread alone; why that was replaced is in section 8.

## 1. Purpose and place in the system

The model values a position of type `bond` (a corporate bond) along the scenario paths of its yield: the risk-free curve at the bond's duration plus its spread to the OFZ curve, the driver `bond_spread`. The value change in each draw feeds the portfolio loss $\Delta_t$, its quantiles P5/P50/P95 and $\mathrm{ES}_{95\%}$ at the end of the horizon, by month and over the path, and through them the reverse stress test. The model is pure: it reads only the paths the scenario has already drawn.

## 2. Inputs

| Input | Meaning | Unit |
|---|---|---|
| `notional` | Current value of the position, $N$ | millions of the position currency |
| `currency` | Position currency $c$; the base currency is the rouble | ISO code |
| `codes` | `secid`, `isin`, `inn`, ...; the platform matches the bond to a driver by them | — |
| `terms.duration` | Modified duration $D$ at the calculation month | years |
| driver `bond_spread` | Path $s_{r,t}$ | percentage points |
| macro paths of the curve | `ru.key_rate` (tenor 0, required), `ru.ofz_1y`, `ru.ofz_5y`, `ru.ofz_10y` (tenors 1, 5, 10 years, read when the macro model carries them) | % |
| macro path of the exchange rate | $X_{r,t}$ for $c\ne\text{RUB}$; for USD `ru.usdrub` | roubles per unit of $c$ |

$R$ is the number of draws, $H$ the horizon in months, $t=0$ the calculation month. `terms` has no silent defaults: a missing `duration` is a refusal naming the field. The exchange quotes the Macaulay duration in days; the modified one is $D=\text{days}/365/(1+y/f)$ for $f$ coupons a year.

## 3. Valuation

Let $r_{r,t}(\tau)$ be the risk-free rate at tenor $\tau$ years: linear between the tenors of the curve and flat beyond them. The bond has $D_h=\max(D-h/12,0)$ years of duration left after $h$ months. Its yield now and at month $h$ is

$$
y_{r,0}=r_{r,0}(D)+s_{r,0},\qquad y_{r,h}=r_{r,h}(D_h)+s_{r,h},
$$

and its value in the position currency is

$$
P_{r,h}=N\,(1+y_{r,0}/100)^{h/12}\left(1-\frac{D_h\,(y_{r,h}-y_{r,0})}{100}\right).
$$

The first factor is the yield earned over the horizon: coupons and the pull to par, reinvested at $y_{r,0}$. The second is the first-order price move of the duration left (Fabozzi, *Bond Markets, Analysis and Strategies*, chapter on price volatility). Because $y_{r,h}$ is read at the shorter tenor $D_h$, the roll down the curve is in it too. A bond whose duration runs out before $h$ is redeemed and carries no rate risk after that. In roubles $V_{r,h}=P_{r,h}X_{r,h}/X_{r,0}$, the value change is $\delta_r=V_{r,H}-N$ and the loss $L_r=-\delta_r$.

The model reads the last column of the paths as the end of the horizon, so the loss by month $t$ is the same calculation on paths cut at $t$. A missing driver leaves the position without a calculation, status `invalid`, with the quantity and code named.

## 4. Parameters

| Symbol | Meaning | Value | Source |
|---|---|---|---|
| $N$ | Notional | from the position | `Position.notional` |
| $D$ | Modified duration now | from `terms.duration`; no default | analyst |
| $s_{r,t}$ | Spread path | from the satellite version | driver `bond_spread`, p. p. |
| $r_{r,t}(\tau)$ | Risk-free curve | from the macro version | `ru.key_rate`, `ru.ofz_1y`, `ru.ofz_5y`, `ru.ofz_10y` |
| $X_{r,t}$ | Exchange rate path | from the macro version | `fx.py` |

## 5. Outputs

A `CalculationResult` per the SDK contract: status `ok`, `needs_review` or `invalid`; `value_change` of length $R$ with $\delta_r$; metrics with id `bond.<name>`, as a value or P5/P50/P95, the loss metric also with ES95; driver flags added by the platform through `envelope`.

## 6. Assumptions and limits

- First order in the yield: no convexity. For an option-free bond the approximation overstates the loss on a rise and understates the gain on a fall; at $D_h\le 2$ years and moves of a few points the error is a fraction of a percent.
- The yield earned is reinvested at $y_{r,0}$, also after redemption.
- The spread is measured to the zero-coupon curve at the Macaulay duration, the curve of the valuation is the G-curve at the modified one; the gap is smaller than the error of the duration approximation at these tenors.
- Default and recovery enter only through the spread path.
- A spread rise above $100/D_h$ p. p. makes the value negative; at the durations of the portfolios that is a rise of over 50 p. p.

## 7. Checks the code passes

Tests `tests/engine/test_models.py`:

1. Constant curve and spread: the bond earns its yield, $\delta=N((1+y_0/100)^{h/12}-1)$.
2. A spread widening costs the duration left, $D-h/12$, times the widening, on top of the yield earned.
3. A rise of the curve costs as much as an equal spread widening.
4. The curve is read at the duration of the bond, now and at the end: $D=3$, tenors 1 and 5 years at 12 % and 16 % give 14 % now and 13 % a year later.
5. A bond with duration shorter than the horizon carries no spread risk.

`tests/test_runs.py` checks the formula on a run of the simulated macro model, which carries the key rate only.

## 8. What changed in version 2.0.0

Version 1.0.0 valued $N(1-D\,\Delta s/100)$. On the scenarios of 2026-11-11 that gave three errors. A rate cut from 14 % to 7.6 % moved the bond by +0.2 % at the median, since the curve did not enter. With nothing moving the bond earned nothing, although its yield was 17.9 %. The duration of 2 years stayed for the whole horizon, although the bond in the portfolio had 1.3 years to its put date and 0.45 a year later, so a spread widening cost it 4.5 times too much.
