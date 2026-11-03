# Macro model from expert elicitation

The `expert` plugin provides one level-0 macro model, `expert.elicited`. It is chosen explicitly, alongside the other macro models; it does not replace the default model. This text was written before the code and states what the code must do, so where they disagree the code is fixed.

## Why this model

Macro models estimated from history give a law of the macro variables that agrees poorly with economics: responses of the wrong sign or scale, explosive paths, shocks without meaning. This model takes the law from an expert's judgement. The expert is the LLM agent that drives the MacroStress MCP server. The method is structured expert elicitation (the SHELF protocol, Cooke's classical model): the expert does not build a model but answers predefined questions about quantities with quantiles, and the plugin fixes the form of the model the answers enter. So the agent cannot swap in its own VAR: it fills numbers into a fixed form.

The model knows nothing of the scenario's event. In terms of the [general scheme](../../../../docs/concept/formalization.en.md) it sets the base measure $P$ from history and the information at the calculation date, and the scenario, the imposition, the price and the satellites do not depend on it.

## Model

Let $x_t\in\mathbb R^n$ be the vector of the $n=23$ macro variables of the registry `src/mst/macro/registry.toml` in month $t$ in the model scale, $t=0$ the last month of history, $t=1,\dots,H$ the months of the horizon, $H\le 12$. The model:

$$
x_t=b_t+\sum_{j=1}^{J}\sum_{\tau=1}^{t}r_j(t-\tau,\varepsilon_{j\tau})+u_t,
$$

where

- $b_t$ is the base path (section "Base path");
- $\varepsilon_{j\tau}$ is the size of shock $j$ in month $\tau$, zero if shock $j$ does not occur that month;
- $r_j(k,\varepsilon)\in\mathbb R^n$ is the response of all variables $k$ months after shock $j$ of size $\varepsilon$, $r_j(k,0)=0$;
- $u_t$ is the residual, the movement of the variables the shocks do not explain (section "Residual").

Responses to different shocks add up. The response to one shock depends on its size nonlinearly: $r_j$ is set by the expert's answers at several sizes, not by multiplying one response by $\varepsilon$.

The level-0 outcome in the sense of the general scheme is

$$
\omega^0=(\theta,\varepsilon,u),
$$

where $\theta$ holds the expert's answers and the estimates from history, $\varepsilon=(\varepsilon_{j\tau})$ the shocks of the horizon, $u=(u_1,\dots,u_H)$ the residuals. The expert answers once, so $\theta$ is constant for a given model version. The scenario's causes are chosen among the coordinates $\varepsilon_{j\tau}$: a scenario may name a shock together with its month.

## Shocks

The set of shocks is fixed and the same for every calculation date, so that scenarios of different dates name the same causes. $J=18$.

| Shock | Meaning |
|---|---|
| `gl.oil` | Oil price from supply: OPEC+, production outages |
| `gl.oil_demand` | Oil price from world demand: recession, China |
| `gl.gas` | Gas prices and gas exports |
| `gl.metals` | Prices of metals and fertilisers |
| `gl.grain` | World prices of grain and food |
| `gl.financial` | Flight from risk, widening spreads |
| `gl.us_rates` | Fed rate and US yields |
| `gl.dollar` | Dollar against world currencies, the yuan included |
| `ru.monetary` | Central bank decision beyond its rule |
| `ru.inflation` | Supply-side inflation: harvest, tariffs |
| `ru.inflation_expect` | Shift of inflation expectations without a change in costs |
| `ru.fx_risk` | Capital outflow, demand for foreign currency |
| `ru.sanctions_trade` | Export and import restrictions, price cap |
| `ru.sanctions_finance` | Payment restrictions, asset freezes, bank disconnections |
| `ru.fiscal` | Budget spending and taxes beyond plan |
| `ru.credit` | Credit impulse: subsidised programmes, macroprudential limits |
| `ru.labor` | Labour shortage, mobilisation, migration |
| `ru.capital_controls` | Currency controls and mandatory sale of export proceeds |

The shocks `gl.oil`, `gl.financial`, `ru.monetary`, `ru.inflation` and `ru.fx_risk` carry the same names as in the other macro models, so corpus scenarios that name them fit this model unchanged.

## Elicitation

At every calculation date the agent answers $J=18$ questions, one per shock. The answer about shock $j$ holds:

- the law of the shock: the frequency per year and the 5, 50 and 95 % quantiles of its size;
- the responses of all 23 variables 1, 3, 6 and 12 months after the shock, at three shock sizes, each as 5, 50 and 95 % quantiles.

All sizes of a shock are asked in one question: the answers are then consistent, and the response to a larger shock does not come out smaller than to a smaller one. The question says outright that responses need not be proportional to size and asks where the response departs from proportionality and why. If the ratios of the responses equal the ratios of the sizes for every variable and horizon, the answer counts as linear and the question is reworded.

The server hands out the question and corpus material through MCP tools, the agent writes the answer, the server checks its form and stores it. The model becomes a version only after a human approves the answers, as with scenarios. The approved answers and their hash enter the model version $m$ and are reused by every scenario of that date.

## Model corpus

The model has its own corpus, separate from the scenario corpus. The agent sees only material published before the calculation date. Four parts:

1. Historical episodes of the shocks with the actual responses of the series: calibration material for frequencies and responses.
2. The state at the date: the latest values of the 23 series, central bank decisions and forecasts, budget parameters.
3. Transmission reviews: central bank and IMF material on how the rate, the exchange rate and commodities act on the economy.
4. Market expectations: futures, forwards, the OFZ curve, the analysts' consensus from the central bank survey.

## Base path

The model comes in two variants that differ only in $b_t$:

- `expert.elicited.market` takes market expectations at the calculation date where they exist (oil from futures, the exchange rate from forwards or rate parity, the policy rate from the OFZ curve and ROISfix, inflation and GDP from the consensus), and a random walk with drift for the other variables;
- `expert.elicited.rw` takes a random walk with drift for every variable.

Random walk with drift: $b_t=x_0+t\mu$, where $\mu$ is the variable's mean monthly change over history up to the calculation date, as in the `random_walk` benchmark of `mst.metrics.baselines`.

Market expectations already price the chance of shocks: an oil future is roughly the mean including a possible fall. So in the `market` variant the base path is obtained by subtracting the expected contribution of the shocks:

$$
b_t=\mathbb E[x_t]_{\text{market}}-\mathbb E\sum_{j}\sum_{\tau\le t}r_j(t-\tau,\varepsilon_{j\tau}),
$$

where the second mean is taken over the law of the shocks from the expert's answers.

## Residual

$u_t$ is what moves the variables in a month when none of the 18 shocks occurs. Its covariance is estimated from history up to the calculation date:

$$
\operatorname{Cov}(u_t-u_{t-1})=\hat\Sigma_{\Delta x}-\Sigma_{\text{shocks}},
$$

where $\hat\Sigma_{\Delta x}$ is the sample covariance of the monthly changes of the 23 variables, and $\Sigma_{\text{shocks}}$ is the covariance of monthly changes the shocks produce according to the expert's answers (frequency, size law and one-month response). If the difference is not positive semidefinite, the expert attributed more movement to the shocks than history shows; such a version is not admitted, and the report names the variables where the difference is negative.

## Imposing a scenario

The imposition is the general case of section 4 of the general scheme. Under a support condition $R=(\theta,\varepsilon_{\bar D},u)$ keeps its base law, and the named shocks $\varepsilon_D$ are drawn from $P(d\varepsilon_D\mid R,S)$. The law of $\varepsilon_D$ given $S$ has no closed form, because the $r_j$ are nonlinear, and is drawn numerically by MCMC at every value of $R$. If no causes are named, $\varepsilon_D$ is every shock and residual of the horizon. The impact $O$ is added to the path, as for any model.

## Checks

The metrics are section 7 of the general scheme. Three comparisons on the same cells:

- `expert.elicited.rw` against the `random_walk` benchmark: what the expert's shocks and responses add;
- `expert.elicited.market` against `expert.elicited.rw`: what market expectations add;
- both models against the other macro models.

The agent knows the outcomes of past episodes, so the backtest on them is for reference, and only a forward calculation, on dates after the LLM's training date, counts as a clean check.

## Open questions

1. Responses are asked 1, 3, 6 and 12 months out. How are the months in between obtained: linear interpolation of the quantiles or a monotone spline?
2. How is a distribution built from the three quantiles of an answer: which family (normal with skew, skewed $t$, piecewise linear distribution function) and which tails beyond 5 and 95 %?
3. How is a shock spread over the months of the horizon: a Poisson stream with the frequency from the answer, at most one shock $j$ per month?
4. How is a response drawn between its 5, 50 and 95 % quantiles: one random variable per shock, shared by all variables and horizons, or one per variable?
5. Residual $u_t$: normal with the estimated covariance, or a bootstrap of historical changes cleaned of the shocks' contribution?
