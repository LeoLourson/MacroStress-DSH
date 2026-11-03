# Credit position model `credit.paths.v1`: reference

Written from the design documents, not from the code: `docs/reference/platform.md`, `docs/howto/plugins.md`, `docs/concept/formalization.md`, `src/mst/plugin_sdk/README.md`, `src/mst/drivers/registry.toml`, `plans/canon-alignment.md`, `plans/satellite-models.md` and the commit messages of the plugin. Where these sources are silent, the gap is listed under open questions and is not filled.

## 1. Purpose and place in the system

The model values positions of types `loan` (a loan to an organisation) and `deposit` along the scenario paths of the driver `overdue_share`: the share of overdue debt in loans to organisations of the borrower's industry, from the Bank of Russia table of loans by activity. The value change in each draw feeds the portfolio loss $\Delta_t$, its quantiles P5/P50/P95 and $\mathrm{ES}_{95\%}$ at the end of the horizon, by month and over the path, and through them the reverse stress test. The model is pure: it reads only the paths the scenario has already drawn.

## 2. Inputs

| Input | Meaning | Unit |
|---|---|---|
| `notional` | Current value of the position, $N$ | millions of the position currency |
| `currency` | Position currency $c$; the base currency is the rouble | ISO code |
| `codes.okved` (and `inn`) | Activity of the borrower; the driver catalog maps OKVED 2 codes to the rows of the Bank of Russia table | — |
| `terms.lgd` | Loss given default $\mathrm{LGD}$, named by `docs/reference/platform.md` as a model input | fraction |
| driver `overdue_share` | Path $o_{r,t}$ | fraction |
| macro path of the exchange rate | $X_{r,t}$ for $c\ne\text{RUB}$ | roubles per unit of $c$ |

$R$ is the number of draws, $H$ the horizon in months, $t=0$ the calculation month. A missing `lgd` is a refusal naming the field; `terms` has no silent defaults.

## 3. Valuation

The commit message "Port the calculation of positions" fixes the credit loss as `lgd*do`: the increase of the overdue share is read as the share of the portfolio that newly defaults, and each defaulted unit loses $\mathrm{LGD}$. This is the expected-loss decomposition $\mathrm{EL}=\mathrm{EAD}\cdot\mathrm{PD}\cdot\mathrm{LGD}$ (Basel Committee, *International Convergence of Capital Measurement*, 2006) with $\mathrm{EAD}=N$ and $\mathrm{PD}$ replaced by the change of the overdue share. With $\Delta o_{r,t}=o_{r,t}-o_{r,0}$, the value in the position currency is

$$
P_{r,t}=N\left(1-\mathrm{LGD}\cdot\Delta o_{r,t}\right),
$$

in roubles $V_{r,t}=P_{r,t}X_{r,t}$, with $X\equiv1$ for a rouble position. The value change at the end of the horizon is $\delta_r=V_{r,H}-V_{r,0}$ and the loss $L_r=-\delta_r$; for a rouble position $L_r=N\,\mathrm{LGD}\,\Delta o_{r,H}$.

The model reads the last column of the paths as the end of the horizon, so the loss by month $t$ is the same calculation on paths cut at $t$ (`plans/canon-alignment.md`). Currency conversion goes through `plugin_sdk/fx.py`.

The design does not set what happens at maturity, whether a fall of the overdue share raises the value above the notional, or how a `deposit` differs from a `loan`; these are open questions. A missing driver leaves the position without a calculation, status `invalid`, with the quantity and code named.

## 4. Parameters

| Symbol | Meaning | Value | Source |
|---|---|---|---|
| $N$ | Notional, exposure at default | from the position | `Position.notional` |
| $\mathrm{LGD}$ | Loss given default | from `terms.lgd`; no default | `docs/reference/platform.md`, commit message |
| $o_{r,t}$ | Overdue share path of the industry | from the satellite version | driver `overdue_share`, logit scale in the satellite, fraction in the paths |
| $X_{r,t}$ | Exchange rate path | from the macro version | `fx.py` |
| maturity, rate, collateral, borrower-specific PD | — | not set by the design | — |

## 5. Outputs

A `CalculationResult` per the SDK contract: status `ok`, `needs_review` or `invalid`; `value_change` of length $R$ with $\delta_r$; metrics with id `credit.<name>`, as a value or P5/P50/P95, the loss metric also with ES95; driver flags added by the platform through `envelope`.

## 6. Assumptions and limits

- The borrower defaults at the rate of its industry: the overdue share of an OKVED row is the model's only credit signal, so idiosyncratic risk and concentration are absent.
- The overdue share is a stock, so its change mixes new defaults with write-offs and repayments; reading it as a default rate is an approximation.
- The overdue share is published with a lag of about 37 days and revised (`registry.toml`), so the value at $t=0$ is an estimate.
- The satellite has a threshold term on `ru.key_rate` above 15 %, so the loss responds non-linearly to the rate through the driver, not through the valuation.

## 7. Checks the code must pass

1. For a rouble `loan` on fixed paths, $\delta_r=-N\,\mathrm{LGD}\,(o_{r,H}-o_{r,0})$.
2. $N=100$, $\mathrm{LGD}=0.45$, $\Delta o=0.02$ gives $\delta=-0.9$.
3. Constant paths give $\delta_r=0$.
4. $\mathrm{LGD}=0$ gives $\delta_r=0$ for any path.
5. A missing `terms.lgd` is refused with the field named.
6. Both `loan` and `deposit` are served by this one model; no other model claims these types.
7. `value_change` has $R$ entries; repeat on the same paths is bit-identical.
8. Paths cut at month $t$ give the value change at month $t$.
9. A borrower whose OKVED maps to no row of the catalog gives `invalid` with the quantity and code named.
10. The loss metric carries P5, P50, P95 and ES95.

## 8. Open questions and contradictions

1. `deposit`: the registry assigns it to this model, but no source says whether a deposit is money placed by the portfolio owner in a bank (then the counterparty is a bank, not an OKVED industry) or something else, nor whether the same formula applies.
2. Valid range of $\mathrm{LGD}$: a fraction in $[0,1]$ by the literature; the design does not state bounds or validation.
3. Fall of the overdue share: the formula gives a gain above notional; whether the value should be capped at $N$ is not set.
4. Maturity, amortisation and interest income within the horizon: not set.
5. Borrower matching: `registry.toml` maps OKVED 2 codes to table rows; whether `inn` is used and what happens with several matching rows is not stated.
6. Sign convention of the loss and the metric names: not stated explicitly.
