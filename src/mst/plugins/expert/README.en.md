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

- the frequency $\lambda_j$: how many times a year the shock occurs;
- the 5, 50 and 95 % quantiles of the signed size of the shock, given that it occurs. A shock that goes both ways, such as a central bank decision beyond its rule, has its 5 % quantile below zero and its 95 % quantile above, so separate "up" and "down" frequencies are not needed;
- the responses of all 23 variables 1, 3, 6 and 12 months after the shock at three sizes, equal to the 5, 50 and 95 % quantiles of the size law; every response is given by its 5, 50 and 95 % quantiles;
- the correlation $\rho_j\in[0,1]$: how consistently the responses of different variables to one occurrence of the shock come out stronger or weaker than their medians.

The month of the shock counts as the first: the response "1 month after" is the change of the variable in the month of the shock itself, $k=0$ in the notation of the section "Model".

All sizes of a shock are asked in one question: the answers are then consistent, and the response to a larger shock does not come out smaller than to a smaller one. The question says outright that responses need not be proportional to size and asks where the response departs from proportionality and why. If the ratios of the responses equal the ratios of the sizes for every variable and horizon, the answer counts as linear and the question is reworded.

The server hands out the question and the model corpus material through MCP tools, the agent writes the answer, the server checks its form and stores it in `workspace/elicitation/<date>/<shock>.json`. No human approves the answers: the 18 answers hold about 15 thousand numbers, too many to check by hand. The answers are checked by their form and by the condition on the residual in the section "Residual". The model is estimated once all 18 answers of a date are written; the answers and their hash enter the model version $m$ and are reused by every scenario of that date.

## How the answers become a law

An answer gives three quantiles of every quantity, while the model needs the law of the whole path. Below is what the plugin adds and why.

Shock stream. In every month of the horizon shock $j$ occurs with probability $1-e^{-\lambda_j/12}$, at most once, independently of other months and shocks. One occurrence per month is needed so that a scenario naming a shock together with its month names one event. A Poisson stream with the same frequency would give two occurrences in a month with probability about 1 % at $\lambda_j=2$, so the restriction takes almost nothing from frequent shocks and nothing from rare ones.

Law from three quantiles. The size of a shock and every response are given by quantiles $q_5$, $q_{50}$, $q_{95}$, and the answer says nothing about what lies beyond 5 and 95 %. So the tails are set by our assumption, and there is one for the whole model: the tail index $\alpha$ from the plugin's specification, 4 by default. It exceeds 2 so that the responses have a variance: without it $\Sigma_{\text{shocks}}$ in the section "Residual" cannot be computed. The family is chosen in the specification and recorded in the version; the backtest compares versions with different families. There are two:

- `pareto`: the quantile function is linear in probability between 5, 50 and 95 %, with Pareto tails of index $\alpha$ beyond. For $p<0.05$ the quantile is $q_5-s_-\bigl((0.05/p)^{1/\alpha}-1\bigr)$, for $p>0.95$ it is $q_{95}+s_+\bigl((0.05/(1-p))^{1/\alpha}-1\bigr)$, where $s_-=\alpha(q_{50}-q_5)/9$ and $s_+=\alpha(q_{95}-q_{50})/9$ keep the density from jumping at 5 and 95 %. The answer is reproduced exactly, but the density is a step function;
- `skew_t`: Azzalini's skew $t$ distribution with $\nu=\alpha$ degrees of freedom, whose location, scale and skewness are fitted to the three quantiles by least squares. The distribution is smooth but passes through the quantiles only approximately; if the fit misses by more than 1 % of the width $q_{95}-q_5$, the estimate report names the quantity.

An answer with $q_5=q_{50}=q_{95}$ gives a number without uncertainty, such as a zero response.

Months in between. Every quantile of a response is interpolated linearly between $k=0,2,5,11$, that is between the answers 1, 3, 6 and 12 months out. Linear interpolation keeps the order $q_5\le q_{50}\le q_{95}$, while a spline may break it between the nodes.

Response between sizes. At a given quantile level the response as a function of the shock size is a broken line through the point $(0,0)$ and the three sizes asked; beyond the outermost points it is constant. The expert said nothing beyond the 5 and 95 % size quantiles, and a constant response invents no continuation: a scenario naming a shock larger than any asked gets the response to the largest one asked, and the rest of its condition falls on other coordinates. For a one-sided shock, whose three sizes have one sign, the outermost point on the other side is $(0,0)$, and a size of the opposite sign from the tail of the law gives no response.

Dependence of the responses. The expert gives the law of every response separately, and the answer does not say how their departures from the medians are related, so the model sets the dependence. At every occurrence of shock $j$ a level $v_i\in(0,1)$ is drawn for every variable $i$ from a $t$ copula with $\nu=\alpha$ degrees of freedom and the same correlation $\rho_j$ between all pairs of variables. The level means the strength of the response: a cell (horizon, size) with a nonnegative median takes the quantile $v_i$, one with a negative median takes the quantile $1-v_i$, so a high $v_i$ moves the whole response of the variable away from zero whichever way it goes. The level is shared by all horizons and sizes of one variable, so the response stays one curve. Occurrences and shocks are independent.

The $t$ copula is the standard replacement of the gaussian copula (McNeil, Frey, Embrechts, "Quantitative Risk Management"). The gaussian copula, the usual way to join an expert's marginal distributions (Clemen, Reilly, 1999), has no tail dependence: joint extreme responses become independent in it and come out rarer than the correlation suggests, which is a dangerous error for a stress test. The $t$ copula has tail dependence. At $\rho_j=1$ all variables take one level, and a strong transmission of the shock is strong through every channel at once; at $\rho_j=0$ the levels are uncorrelated, but joint extreme responses are still more frequent than under independence. The degrees of freedom equal $\alpha$, so that one model parameter sets the weight of the tails.

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

The same agent writes the market expectations in the same elicitation, because the series store holds no futures, forwards, OFZ curve or consensus. For every variable that has expectations, the answer holds the expected value 1, 3, 6 and 12 months out in the model scale and the fragment of the model corpus the number comes from; between horizons the values are interpolated linearly. The server refuses an answer without a fragment: a market number that is not in the corpus is the expert's judgement, and judgements enter the model only through the shocks.

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

where $\hat\Sigma_{\Delta x}$ is the sample covariance of the monthly changes of the 23 variables, and $\Sigma_{\text{shocks}}$ is the covariance of monthly changes the shocks produce according to the expert's answers; it is computed by simulating the shock stream under the laws of the section "How the answers become a law". The residual is normal: $u_t-u_{t-1}\sim N(0,\hat\Sigma_{\Delta x}-\Sigma_{\text{shocks}})$, independently across months. If the difference is not positive semidefinite, the expert attributed more movement to the shocks than history shows; such a version is not admitted, and the report names the variables where the difference is negative.

The formula follows from a decomposition of the variance: we know the full covariance of the changes from history and the share of the shocks from the expert's answers, and the background is the difference. The background is the sum of many small influences the expert does not name, and their sum is close to normal; large rare moves are assigned to the shocks, so the background needs no heavy tails. With $\lambda_j=0$ for every shock the model coincides with the `random_walk` benchmark, so everything that sets it apart from the benchmark comes from the expert. The positive semidefinite condition checks the answers without a human.

Two other options were rejected. A bootstrap of historical changes cleaned of the shocks needs to know when and how large all 18 shocks were in history, and those data do not exist. The full covariance $\hat\Sigma_{\Delta x}$ without the subtraction counts the movement from the shocks twice, in history and in the answers, and the fan comes out wider than the benchmark even with correct answers.

## Imposing a scenario

The imposition is the general case of section 4 of the general scheme. Under a support condition $R=(\theta,\varepsilon_{\bar D},u)$ keeps its base law, and the named shocks $\varepsilon_D$ are drawn from $P(d\varepsilon_D\mid R,S)$. The law of $\varepsilon_D$ given $S$ has no closed form, because the $r_j$ are nonlinear, and is drawn numerically by MCMC at every value of $R$. If no causes are named, $\varepsilon_D$ is every shock and residual of the horizon. The impact $O$ is added to the path, as for any model.

## Checks

The metrics are section 7 of the general scheme. Three comparisons on the same cells:

- `expert.elicited.rw` against the `random_walk` benchmark: what the expert's shocks and responses add;
- `expert.elicited.market` against `expert.elicited.rw`: what market expectations add;
- both models against the other macro models.

The agent knows the outcomes of past episodes, so the backtest on them is for reference, and only a forward calculation, on dates after the LLM's training date, counts as a clean check.
