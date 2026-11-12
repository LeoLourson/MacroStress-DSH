# Reference description of the model `bvar.sv`

This document states the intent of the model from the design documents, not from the code, so that the code can be checked against it. Sources: the general scheme of the system (`docs/concept/formalization.md`), the legacy description of the BVAR method (`docs/concept/bvar-method-legacy.md`), the macro model repair plan (`plans/macro-models.md`) and its results log (`plans/macro-models-results.md`), the canon alignment plan (`plans/canon-alignment.md`), the macro variable registry (`src/mst/macro/registry.toml`), the READMEs of `mst.plugins.bvar`, `mst.macro`, `mst.model_sdk`, `mst.metrics`, the plugin guide (`docs/howto/plugins.md`) and the platform reference (`docs/reference/platform.md`). The legacy method description is taken as intent where later plans do not change it; each change is marked. Where sources are silent or disagree, the question is listed in section 11.

## 1. Purpose

The system answers the question of how the target quantities of an object (portfolio losses) are distributed if a given event happens. Quantities are split into levels: level 0 holds the macro variables of the economy, higher levels hold drivers of industries, companies and bond issues, and the last level holds the target quantities. A *macro model* is the level-0 model: it gives the law of the macro variable path over the horizon given the information available on the calculation date.

`bvar.sv` is the platform's reference macro model, plugin `mst.plugins.bvar`. It is a Bayesian vector autoregression (BVAR) with stochastic volatility, Student-t tails, block exogeneity of the global block, and shocks named by sign restrictions. It answers two questions:

- *unconditional forecast*: the distribution of macro paths $H$ months ahead without a scenario (the base measure $P$ of the general scheme);
- *conditional forecast*: the distribution of these paths when the scenario's observations are imposed and its interventions executed (the overlay $P_s$ of the general scheme).

Besides paths the model returns the law of the constrained quantities given the remaining coordinates; from it the platform computes the scenario price $\mathrm{dist}(s)=\mathrm{KL}(P_s\Vert P)$.

The output is consumed by satellite models of higher levels (macro paths as regressors), position models (macro variables read directly, for example the OFZ curve), the run report (price, admission, diagnostics) and the quality checks of `mst.metrics` (rolling check on a date grid and episode backtest).

## 2. Data

### 2.1. Time

Time is counted in months, $t$ is the month index. Month $t=0$ is the last month for which at least one variable is known on the calculation date; months $t\le 0$ are history, months $t=1,\dots,H$ are the horizon of length $H$. The horizon is at most 360 months. Quarterly values, if a report needs them, are aggregated outside the model.

### 2.2. Variables

The registry `src/mst/macro/registry.toml` holds $n=23$ macro variables. $X_{i,t}$ is the value of variable $i$ in month $t$ in original units, $y_{i,t}=g_i(X_{i,t})$ is the value in the *model scale*. The transform $g_i$ is either the identity (`identity`, for rates, yields, spreads and unemployment) or $g_i(X)=100\ln X$ (`log100`, for prices, exchange rates, indices and volumes), so that a difference of two values approximates a percentage change. The vector $y_t=(y_{1,t},\dots,y_{n,t})^\top$ is ordered with the global block (`gl.`) first.

| Id | Variable | Unit | $g_i$ | Seasonal | Lower bound $\underline x_i$ |
|---|---|---|---|---|---|
| `gl.brent` | Brent oil | USD per barrel | $100\ln X$ | — | — |
| `gl.fed_rate` | Fed rate | % | $X$ | — | 0 |
| `gl.ust_10y` | 10-year UST yield | % | $X$ | — | 0 |
| `gl.vix` | VIX | points | $100\ln X$ | — | — |
| `ru.key_rate` | Bank of Russia key rate | % | $X$ | — | 0 |
| `ru.ruonia` | RUONIA | % | $X$ | — | 0 |
| `ru.ofz_1y` | OFZ 1-year yield, G-curve | % | $X$ | — | 0 |
| `ru.ofz_5y` | OFZ 5-year yield | % | $X$ | — | 0 |
| `ru.ofz_10y` | OFZ 10-year yield | % | $X$ | — | 0 |
| `ru.spread_aa` | AA group spread to G-curve | p.p. | $X$ | — | none (`negative = true`) |
| `ru.spread_a` | A group spread to G-curve | p.p. | $X$ | — | none (`negative = true`) |
| `ru.spread_bbb` | BBB group spread to G-curve | p.p. | $X$ | — | 0 |
| `ru.usdrub` | official USD rate | RUB per USD | $100\ln X$ | — | — |
| `ru.cnyrub` | official CNY rate | RUB per CNY | $100\ln X$ | — | — |
| `ru.cpi` | consumer price index | index | $100\ln X$ | yes | — |
| `ru.output` | output of basic industries | index | $100\ln X$ | yes | — |
| `ru.unemployment` | unemployment | % | $X$ | yes | 0 |
| `ru.real_wage` | real wage | index | $100\ln X$ | yes | — |
| `ru.retail` | retail turnover | index | $100\ln X$ | yes | — |
| `ru.credit` | credit to firms and households | mln RUB | $100\ln X$ | — | — |
| `ru.mortgage_rate` | average mortgage rate | % | $X$ | — | 0 |
| `ru.housing_price` | housing price | RUB per m² | $100\ln X$ | — | — |
| `ru.imoex` | MOEX index | points | $100\ln X$ | — | — |

Global variables affect Russian ones and Russian ones do not affect global ones: Russia is a small open economy whose shocks do not move oil or the Fed rate. GDP is quarterly, so monthly output is represented by the index of basic industries.

*Stationary variables* are `gl.vix`, `ru.spread_aa`, `ru.spread_a`, `ru.spread_bbb`; the other 19 are treated as nonstationary. The distinction sets the centre of coefficient shrinkage (section 4.1).

### 2.3. Seasonality

Rosstat series are stored without seasonal adjustment. For the five seasonal variables (`ru.cpi`, `ru.output`, `ru.unemployment`, `ru.real_wage`, `ru.retail`) panel assembly subtracts a constant calendar profile: the mean increment of the series in each calendar month, estimated only on data known on the calculation date. The profile is constant because a short history cannot support changing seasonality; it is estimated as of the date so that checks on past dates do not see future data.

### 2.4. Sample, publications, missing months

Estimation starts in month $T_0$ = January 2012, where the store's series start. Each value is stored with its publication date, and estimation as of date $d$ sees each value in the vintage published by $d$, so checks on past dates do not use revisions that did not yet exist. For macro series the publication date is approximate (`approximate`) and the history is already revised.

A month without a value (quarterly housing price, a series starting after $T_0$, the ragged edge near $t=0$) stays missing in the panel. The model does not fill gaps before estimation but draws them inside the sampler (section 4.6). The value of `ru.retail` in month $t=0$, empty in the panel, is also drawn by the model and differs between macro models; this is not a defect.

### 2.5. Bounds

Rates, yields, unemployment and the BBB spread cannot be negative, and a linear model does not know this. For a variable with bound $\underline x_i$ (registry field `lower`) the model simulates a *shadow value* $s_{i,t}$, and the observed value is

$$
x_{i,t}=\max\big(s_{i,t},\,\underline x_i\big)
$$

in the model scale. Lags, observations, interventions and the price work on the shadow: it follows the gaussian law the whole conditional machinery knows, and a later rise starts from the shadow, not from the bound. The AA and A spreads have no bound because their history went below zero. Estimation reads history as observed; this is an approximation, and a proper shadow-rate estimation is out of scope.

## 3. Model

### 3.1. Equation

Let $p$ be the number of lags, $\kappa\in\mathbb R^n$ the intercept vector, $A_1,\dots,A_p$ the $n\times n$ coefficient matrices, $d_t\in\mathbb R^{q}$ the vector of deterministic dummy regressors (section 3.4), $\Gamma$ their $n\times q$ coefficient matrix, and $u_t$ the error vector. Then

$$
y_t=\kappa+\sum_{\ell=1}^{p}A_\ell y_{t-\ell}+\Gamma d_t+u_t.
$$

Block exogeneity: split $y_t=(y_t^{g},y_t^{r})$ into the global and Russian blocks. For all $\ell$ the block $A_\ell^{gr}$ (Russian lags in global equations) is zero.

### 3.2. Errors, volatility, tails

Errors are written through a unit lower triangular matrix $C$ (ones on the diagonal, variable order of section 2.2):

$$
u_t=C\,e_t,\qquad e_{j,t}=e^{h_{j,t}/2}\sqrt{\omega_{j,t}}\,z_{j,t},\quad j=1,\dots,n.
$$

Here $z_{j,t}\sim\mathcal N(0,1)$. The log variance $h_{j,t}$ is a random walk

$$
h_{j,t}=h_{j,t-1}+s_j\xi_{j,t},\qquad \xi_{j,t}\sim\mathcal N(0,1),
$$

where $s_j>0$ is the speed of volatility change: in a crisis series move more, and a constant variance would understate the tail. The multiplier $\omega_{j,t}\sim\mathrm{IG}(\nu_j/2,\nu_j/2)$ (inverse gamma) makes $e_{j,t}$ given $h_{j,t}$ Student-t with $\nu_j$ degrees of freedom, so the model admits rare large moves like December 2014. The variables $z$, $\xi$, $\omega$ are independent of each other and over time. The triangular form is there for equation-by-equation estimation (section 4.3).

The policy rates (`jumps` of the specification: the key rate and RUONIA) carry a jump instead of the Student multiplier: $\omega_{j,t}=\kappa_j$ with probability $\pi_j$ and $\omega_{j,t}=1$ otherwise (Chib, Nardari, Shephard, 2002). The assumptions before data: $\pi_j\sim\mathrm{Beta}(1,19)$, $\kappa_j$ inverse gamma with shape 2 and mean 25, truncated above one. Such a rate sits still for months and moves at meetings. A random-walk volatility with a tight assumption on $s_j$ cannot follow the step, and with a loose one it falls to nought between decisions and explodes the paths of a year (2026-11-12: a 12-month sd of the key rate of 35 pp against 4.25 in the history). The impulse dummies of 2014 and 2022 do not enter the equations of the rates (`spares`): the jumps are estimated on those months.

Block exogeneity extends to the errors: the elements of $C$ through which the Russian component $e^r_t$ would enter the global errors $u^g_t$ are zero; with the global block first and $C$ lower triangular this holds by construction.

### 3.3. Structural shocks

The *structural shocks* $\varepsilon_t\in\mathbb R^n$ of month $t$ come from $e_t$ by an orthogonal rotation $Q$ ($Q^\top Q=I$):

$$
u_t=B_t\,\varepsilon_t,\qquad B_t=C\,\mathrm{diag}\big(e^{h_{t}/2}\sqrt{\omega_{t}}\big)\,Q,\qquad \varepsilon_t\sim\mathcal N(0,I),
$$

with root and exponent taken elementwise. $B_t$ is the *impact matrix*: column $j$ shows how shock $j$ moves each variable in the month it occurs. $Q$ is block diagonal, $Q=\mathrm{diag}(Q^{g},Q^{r})$, so a Russian shock does not move a global variable. Identification (section 5) uses the impact matrix of the last history month, $B_0=C\mathrm{diag}(e^{h_0/2})Q$.

### 3.4. Regime dummies

The vector $d_t$ holds impulses in 2014-11 (move to a floating rouble), 2022-02 and 2022-03, and a step from 2021-01 (spread splice: international rating scale up to 2020-12, national from 2021-01). An impulse removes a single outlier that would otherwise inflate volatility and bias the coefficients. Dummies whose month is after the calculation date are excluded from estimation: a model as of a past date cannot know a future regime. Over the forecast horizon impulses are zero and the step equals one if its month is not after the calculation date.

### 3.5. Parameters and representation

The parameters $\theta=(\kappa,A_{1:p},\Gamma,C,s_{1:n},\nu_{1:n},Q)$ together with the historical paths $h_{j,t}$, $t\le 0$, form one draw of the posterior sample. In the terms of the general scheme the level-0 outcomes are $\omega^0=(\theta,h_{1:H},\omega_{1:H},\varepsilon_{1:H})$, where $1:H$ denotes the horizon; the path $x^0=G^0(\omega^0)$ follows from the recursion of section 3.1 starting from history, then the bound of section 2.5 is applied.

### 3.6. Companion matrix

For lags $A_{1:p}$ the *companion matrix* is

$$
F=\begin{pmatrix}A_1&A_2&\cdots&A_{p-1}&A_p\\ I_n&0&\cdots&0&0\\ 0&I_n&\cdots&0&0\\ \vdots&&\ddots&&\vdots\\ 0&0&\cdots&I_n&0\end{pmatrix}\in\mathbb R^{np\times np},
$$

and the *radius* $\rho(A_{1:p})$ is its spectral radius, the largest modulus of an eigenvalue. A path simulated from a draw with $\rho>1$ grows without bound, and scenario constraints amplify this.

## 4. Estimation

### 4.1. Assumption about lag coefficients before data (Minnesota)

The assumption about coefficients before data pulls the model toward a simple one: a nonstationary variable stays at its last level, a stationary one reverts to its mean (Litterman, 1986; Doan, Litterman, Sims, 1984). Let $a_{ij,\ell}$ be the coefficient on $y_{j,t-\ell}$ in equation $i$. Its distribution before data is normal with mean

$$
\mathbb E\,a_{ij,\ell}=\begin{cases}1,&i=j,\ \ell=1,\ i\ \text{nonstationary},\\0&\text{otherwise},\end{cases}
$$

and a standard deviation that grows with the overall tightness $\lambda$ (`tightness`) and decays with the lag. The legacy method description sets $\lambda=0.2$; the repair plan selects $\lambda$ from $\lbrace 0.05, 0.1, 0.2\rbrace$ (section 4.2). The lag decay, the cross-variable factor and the scales $\sigma_i$ are not set by the design (section 11). The coefficients of block $A_\ell^{gr}$ are not estimated and equal zero. The assumption about $\kappa$ and $\Gamma$ is not set by the design.

### 4.2. Dummy observations

An audit on 2026-10-30 found that the Minnesota assumption carries little weight with regressors in levels: the share $P_0/(X^\top X+P_0)$ on the diagonal of the lag block is orders of magnitude below its nominal value. The repair plan adds two sets of dummy observations that act on levels and pull the system toward unit roots and away from explosive roots (Sims and Zha, 1998; Giannone, Lenza and Primiceri, 2015). Let $\bar y_0$ be the mean of the first $p$ observations of the sample.

*Sum-of-coefficients dummies* with parameter $\mu$ (`mu`): one row per variable $i$,

$$
y^{+}_{i}=\frac{\bar y_{0,i}}{\mu}e_i^\top,\qquad x^{+}_{i}=\Big(0,\ \underbrace{\tfrac{\bar y_{0,i}}{\mu}e_i^\top,\ \dots,\ \tfrac{\bar y_{0,i}}{\mu}e_i^\top}_{p\ \text{times}},\ 0\Big),
$$

where $e_i$ is the $i$-th unit vector, the first zero sits at the intercept and the last at the dummies $d_t$. They state that if all lags of variable $i$ equal $\bar y_{0,i}$, its forecast equals $\bar y_{0,i}$ too.

*Initial-observation dummy* with parameter $\delta$ (`delta`): one row

$$
y^{++}=\frac{\bar y_0^\top}{\delta},\qquad x^{++}=\Big(\frac1\delta,\ \underbrace{\tfrac{\bar y_0^\top}{\delta},\ \dots,\ \tfrac{\bar y_0^\top}{\delta}}_{p\ \text{times}},\ 0\Big).
$$

In the Giannone, Lenza and Primiceri parametrisation smaller $\mu$ and $\delta$ mean a stronger pull. The plan's candidates are $\mu=\delta\in\lbrace 1, 5\rbrace$ for each $\lambda$ in $\lbrace 0.05, 0.1, 0.2\rbrace$, plus the current assumption without dummies as the reference. The dummy-augmented regression must equal the closed-form posterior.

The candidate is selected only on the tuning half of the date grid of the rolling check, by the relative CRPS of the unconditional forecast against the random walk; the winner is then evaluated on the checking half (gate G6). There is no tuning method beyond this grid, because any other would fit the choice to the checking data.

### 4.3. Sampler

The posterior sample is drawn by MCMC with the triangular algorithm of Carriero, Clark and Marcellino (2019) in the corrected form of Carriero, Chan, Clark and Marcellino (2022): given volatilities and $\omega$, equation $i$ of the triangular form is estimated separately, with regressors from lags and from the current errors of preceding equations, so the cost of a step grows as $n^4$ instead of $n^6$. Blocks of one iteration:

1. coefficients $(\kappa,A_{1:p},\Gamma)$ equation by equation, with the assumption of section 4.1 and the dummies of section 4.2, block $A^{gr}$ fixed at zero;
2. the off-diagonal elements of $C$;
3. the paths $h_{j,t}$ for $t\le 0$, via the normal mixture of Kim, Shephard and Chib (1998);
4. the speeds $s_j^2$;
5. the multipliers $\omega_{j,t}$ and degrees of freedom $\nu_j$;
6. missing panel values (section 4.6).

The number of kept draws `sampler.draws` is raised by the repair plan to 4000, with `thin` kept; its value, the burn-in length and the starting values are not set by the design. Estimation is deterministic given the seed recorded in the specification.

### 4.4. Stochastic volatility

The distribution of $s_j^2$ before data is inverse gamma with mean 0.03 and 60 degrees of freedom. It is narrow on purpose: with 5 degrees of freedom the median $s_j$ for `gl.fed_rate` reached 4.0 per month, and paths diverged within three months. The distribution of the initial $h_{j}$ is not set by the design.

### 4.5. Tails

The degrees of freedom $\nu_j$ are drawn on the integer grid from $\nu_{\min}$ (`dof_min`) to 50 with a uniform distribution before data. The legacy method description starts the grid at 3; the repair plan starts it at `dof_min`, which the specification sets to 5, and keeps 3 as a candidate for comparison. The reason: variance is infinite at $\nu=2$ and kurtosis at $\nu=4$, and for rates that sit still and then step, the floor of the grid absorbed what volatility should carry (the median draw sat at 3 for five variables). The change is kept only if gate G6 holds against the winner of section 4.2.

### 4.6. Missing values

Missing panel values are drawn at each iteration from their conditional normal distribution given current parameters and observed values; estimation thus uses all months, and the uncertainty of a gap enters the posterior.

### 4.7. Stability filter

After sampling and before the rotation search only draws with

$$
\rho(A_{1:p})\le\rho_{\max},\qquad \rho_{\max}=1.005,
$$

are kept. A unit root is admissible and expected, because most variables are nonstationary; growth without bound is not. This is the posterior restricted to the stable region, by the same mechanism as the identification filter. The number of draws after the filter is recorded in diagnostics as `stable`, beside `estimated` (after the sampler) and `identified` (after identification). If no draw is stable, estimation fails with an error naming the model and $\rho_{\max}$. If fewer than 200 survive, $\rho_{\max}$ is not raised: this is a reason to stop and report.

## 5. Identification of named shocks

The data determine the error covariance but not the impact matrix: any orthogonal $Q$ gives the same covariance. Some shocks get an economic meaning through sign restrictions (Uhlig, 2005; Rubio-Ramírez, Waggoner and Zha, 2010): the shock must move the listed variables in the listed direction in the month it occurs and in the following months up to lag `sign_lag` = 2 inclusive, that is at response horizons 0, 1 and 2. The table is a hypothesis: it is stored in the model version (`signs.toml`) and checked with it on episodes.

| Shock | Block | Rise | Fall |
|---|---|---|---|
| `ru.monetary` (monetary) | Russian | `ru.key_rate`, `ru.ruonia`, `ru.ofz_1y` | `ru.usdrub`, `ru.imoex`, `ru.cpi` |
| `ru.inflation` (inflation) | Russian | `ru.cpi`, `ru.key_rate` | `ru.real_wage` |
| `ru.fx_risk` (exchange rate and risk premium) | Russian | `ru.usdrub`, `ru.ofz_10y`, `ru.spread_bbb` | `ru.imoex` |
| `gl.oil` (oil) | global | `gl.brent` | `ru.usdrub` |
| `gl.financial` (global financial) | global | `gl.vix` | `gl.ust_10y`, `ru.imoex` |

`ru.cpi` among the falls of `ru.monetary` is added by the repair plan: in the earlier version inflation fell below base for months after a rate cut, so the shock did not behave as a monetary one. The other $n-5$ shocks are unnamed: `gl.unnamed_k` in the global block, `ru.unnamed_k` in the Russian block.

Procedure for each stable draw: draw the blocks $Q^g$ and $Q^r$ from the uniform (Haar) distribution on orthogonal matrices, compute responses $\Psi_h B_0$ at horizons $h=0,\dots,$ `sign_lag`, where $\Psi_h$ are the moving-average coefficients from $A_{1:p}$, and check the signs. Repeat up to 20 000 rotations per draw; a draw with no admissible rotation is dropped. The first admissible rotation is a uniform choice from the admissible set. The share of kept draws is shown in the version description. If survival falls below 20 %, no restrictions are added: which restrictions are right is the owner's decision.

## 6. Forecast

### 6.1. Unconditional forecast

For each of $R$ replications $r$: take a draw $\theta^{(r)}$ with its volatility history; draw $h_{j,1:H}$ by the random walk from $h_{j,0}$, $\omega_{j,1:H}$ and $\varepsilon_{1:H}\sim\mathcal N(0,I)$; compute the path by the recursion of section 3.1 from the last $p$ months of history (gaps at the edge take drawn values); apply the bound of section 2.5; return to original units through $g_i^{-1}$. The result is a function of the seed: the same seed gives the same numbers bit for bit.

### 6.2. Scenario observations and interventions

The horizon path in model scale is the vector $\mathbf y=(y_1^\top,\dots,y_H^\top)^\top$ of length $nH$. A constraint $c$ is a triple $(w_c,a_c,b_c)$ with the requirement $a_c\le w_c^\top\mathbf y\le b_c$; it covers the value of a variable, its change, a quarterly mean and the difference of two variables. An exact value $v$ is widened to $[v-\tau,v+\tau]$ with *tolerance* $\tau>0$, because a continuous law hits a point with probability zero. All constraints: $\mathbf a\le W\mathbf y\le\mathbf b$, with rows $w_c^\top$. Constraints apply to the shadow: an exact condition at the bound ("Fed at zero") means the shadow within $\tau$ of the bound.

An intervention is a decision of a subject outside the models: `set` makes a quantity $k$ in months $T$ equal to a given value, `shift` moves it. The model executes the interventions itself, in the recursive structural form of the VAR: a set replaces its equation's value in its month, and the reduced-form shock of that variable is not drawn in that month; a shift adds to the equation. Later months read the set value through the lags, so it does not return to the base path. Other variables of the same month keep their reduced-form errors: within level 0 an intervention acts through the lags, not within the month. The platform does not apply the interventions again, and the run report names the form executed.

### 6.3. Observations and the conditional law

Given $\theta$, history and future $h$, $\omega$, the path is linear in the shocks:

$$
\mathbf y=\mu_\theta+\Phi\boldsymbol\varepsilon,\qquad \boldsymbol\varepsilon=(\varepsilon_1^\top,\dots,\varepsilon_H^\top)^\top\sim\mathcal N(0,I_{nH}),
$$

where $\mu_\theta$ is the path at zero shocks and $\Phi$ the cumulated responses with impact matrices $B_t$. After the interventions are substituted, the terms of the set cells leave $W\mathbf y$ for the right-hand side: the columns of the set variables in a row are zeroed, and their contribution becomes a constant; a shift adds a known number to the right-hand side. Let $\mu^\psi$ be the path at zero free shocks with the interventions executed, and $\Phi^\psi$ the responses to the free shocks — the set cells are not drawn. The observations are

$$
\mathbf a-W\mu^\psi\ \le\ W\Phi^\psi\boldsymbol\varepsilon_{\text{free}}\ \le\ \mathbf b-W\mu^\psi,
$$

and the free shocks are drawn from $\mathcal N(0,I)$ truncated to this box by a Gibbs sampler for the truncated normal (Geweke, 1991) with 50 sweeps; the approach is Antolín-Díaz, Petrella and Rubio-Ramírez (2021). $\Phi^\psi$ is never built in full; only the rows $W\Phi^\psi$ are computed.

A row that is fully determined by the set values leaves the system; if its value is outside the bounds, the scenario contradicts the model (`ContradictsModels`). A scenario is rejected also when a row is linearly dependent on the rows already fixed, at $10^{-9}$ of how much all shocks move it: such an observation adds no information, and the scenario names its number (`InfeasibleScenario`).

### 6.4. Latent coordinate: the volatility path

The future volatility path $h_{1:H}$ is the model's *latent coordinate* (`volatility_path`); a scenario does not name it. It must not be drawn before the observations: for each value of $R$, `LATENT_CANDIDATES` = 64 candidates are drawn, each gets a weight proportional to the box probability $P(S\mid\theta,h)$, and one is chosen by weight; without observations the path is drawn from its base law. The log box probability comes from the GHK sampler `log_box_probability`. The effective size of the weights goes to diagnostics. At each $R$ the path law then equals the base law conditioned on the event, as the corollary of the general scheme requires.

### 6.5. Law of the observations and price

Let $R=(\theta,\psi)$: the parameters and the realized interventions, whose law the scenario preserves. By the corollary of the general scheme, with a single support observation,

$$
\frac{dP_s}{dP}=\frac{\mathbf 1_S}{P(S\mid R)},\qquad \mathrm{dist}(s)=\mathbb E_P\big[-\log P(S\mid R)\big].
$$

The method `constraint_law` returns the law of $W\mathbf y$ given $R$ **after the interventions**: per value of $R$, the gaussian law of the rows (means and covariances), the latent candidates and their log weights. The platform turns it into $P(S\mid R)$ and the price, and computes the one-sided price and the standard error. Tilting for observations on means and probabilities is done by the platform (`mst.scenario.overlay`); the model does not know it. The distribution of $\theta$ under $P_s$ does not change.

### 6.6. Forecast diagnostics

The size of the shock the observations need, in standard deviations per horizon month, is written to the simulation `diagnostics` and reaches the run's `meta.json`: a 5-point rate cut through one shock needs a large shock, and the number must be visible.

## 7. Parameters

| Symbol | Key | Meaning | Value | Source of the value |
|---|---|---|---|---|
| $n$ | — | number of variables | 23 | variable registry |
| $T_0$ | — | first sample month | 2012-01 | legacy method description; `mst.metrics` README |
| $p$ | — | number of lags | 3 | legacy method description |
| $\lambda$ | `tightness` | overall Minnesota tightness | candidates 0.05, 0.1, 0.2 (previously 0.2) | repair plan; legacy method description |
| — | — | lag decay, cross-variable factor, $\sigma_i$ | not set by the design | — |
| $\mu$ | `mu` | sum-of-coefficients weight | candidates 1, 5, with $\mu=\delta$ | repair plan |
| $\delta$ | `delta` | initial-observation weight | candidates 1, 5 | repair plan |
| — | — | stationary variables | `gl.vix`, three spreads | legacy method description |
| $d_t$ | — | regime dummies | impulses 2014-11, 2022-02, 2022-03; step 2021-01 | legacy method description |
| $\mathbb E s_j^2$, df | — | distribution of $s_j^2$ before data | IG, mean 0.03, 60 degrees of freedom | legacy method description |
| $\nu_{\min}$ | `dof_min` | floor of the $\nu_j$ grid | 5 (candidate 3) | repair plan |
| $\nu_{\max}$ | — | top of the $\nu_j$ grid | 50 | legacy method description |
| — | `sampler.draws` | kept draws | 4000 | repair plan |
| — | `thin`, burn-in | thinning, burn-in | not set by the design | — |
| $\rho_{\max}$ | `rho_max` | radius limit | 1.005 | repair plan |
| — | — | minimum surviving draws before stopping | 200 | repair plan |
| — | `sign_lag` | last horizon of sign restrictions | 2 | repair plan |
| — | — | rotations per draw | 20 000 | legacy method description |
| — | — | minimum identification survival | 20 % | repair plan |
| $\underline x_i$ | `lower` | lower bound | 0 for 10 variables (section 2.2) | variable registry |
| $\tau$ | — | tolerance of an exact value | not set by the design | — |
| — | — | Gibbs sweeps | 50 | legacy method description |
| — | `LATENT_CANDIDATES` | latent candidates per $R$ | 64 | legacy method description |
| — | — | dependence threshold of a constraint | $10^{-9}$ | legacy method description |
| $H$ | — | longest horizon | 360 | legacy method description |

## 8. Inputs and outputs

The plugin declares one macro model: `plugin = Plugin(...)` with `macro=(MODEL,)`, model id `bvar.sv`, an id starting with `bvar.`. The core knows the model only through the `mst.plugins` registry. The model implements the abstract `MacroModel` of `mst.model_sdk` and imports only allowed modules (`mst.model_sdk`, `mst.contracts`, `mst.macro.panel`, `mst.macro.registry` and its own package).

| Step | Input | Output |
|---|---|---|
| `estimate(context)` | series store, registry, calculation date, specification | version metadata and arrays: draws of $\theta$ after the filters, historical $h$, shock names, sign table, specification, registry, panel, diagnostics `estimated`, `stable`, `identified`; the platform checks the required keys (dates, variables, shocks, panel) |
| `load(meta, arrays)` | a version estimated by this plugin (`meta["model"]` = `bvar.sv`) | the model of the version |
| `simulate(model, plan, draws, seed)` | `ScenarioPlan`: horizon, observations, realized interventions (one row per group of paths), `inner` | `MacroSimulation`: paths `(R, H+1, n)` in model scale with the interventions executed, groups of $R$, candidate diagnostics and the shock size |
| `constraint_law(...)` | model, plan | `ConstraintLaw`: the gaussian law of $W\mathbf y$ by groups of $R$, with latent candidates and weights |

The class attribute `latents` declares the latent coordinate `volatility_path`; a scenario does not name it, and observations reweight its candidates. A version is immutable and addressed by the `sha256` hash of its metadata and arrays; a changed specification gives a new hash, and old versions are not overwritten.

The platform, not the model, converts paths to original units, applies tilting, computes the price, admission and the report; the level-0 interventions are executed by the model in its structural form. Positions receive `Paths.macro(name)` of shape `(R, H+1)` in original units; month 0 is the last history month.

## 9. Assumptions and limits of applicability

- No feedback: the portfolio and drivers do not affect macro variables.
- Russia does not affect the global block, through lags or through errors.
- Coefficients are constant over time; regime breaks enter only through the dummies 2014-11, 2022-02, 2022-03 and the step 2021-01. The model does not foresee a new regime after the calculation date.
- Signs are a hypothesis about the economy, not a consequence of the data. Unnamed shocks carry no economic meaning.
- The zero bound is applied only to the output; estimation sees history as observed.
- The seasonal profile is constant; paths of seasonal variables are in the adjusted scale.
- Publication dates of macro series are approximate and history is revised, so checks on past dates are optimistic.
- The $\rho_{\max}$ filter restricts the posterior; with it the model does not describe explosive dynamics even if the data held them.
- The price of an exact condition depends on $\tau$; prices are comparable only at equal tolerances.
- Under a condition ES95 can fall, because conditioning narrows the law. This is not a model defect.

## 10. Checks the code must pass

1. The model reads exactly the 23 registry variables in the order "global block, then Russian", with the `identity` and `log100` transforms of section 2.2.
2. The seasonal profile is subtracted for the five variables with `seasonal = true` and only for them; the profile is estimated on data known on the calculation date.
3. Estimation as of date $d$ reads no value published after $d$ and uses no month before 2012-01.
4. Missing panel values are not filled before estimation and are drawn inside the sampler; on a panel without gaps this step leaves the data unchanged.
5. The number of lags is $p=3$; exactly `gl.vix`, `ru.spread_aa`, `ru.spread_a`, `ru.spread_bbb` are stationary (own first lag centred at zero).
6. In every kept draw the coefficients of Russian lags in global equations are exactly zero, not approximately.
7. In every draw $C$ is unit lower triangular, and the block "Russian component in a global error" is zero.
8. Regime dummies: impulses 2014-11, 2022-02, 2022-03 and the step 2021-01; dummies whose month is after the calculation date do not enter estimation.
9. The sum-of-coefficients and initial-observation dummies follow the formulas of section 4.2 from the mean of the first $p$ observations; on data with a known posterior the dummy-augmented regression equals the closed form.
10. The candidate without dummy observations (the tuning reference) remains available in the specification, and its estimate equals the estimate under the Minnesota assumption alone.
11. The distribution of $s_j^2$ before data is inverse gamma with mean 0.03 and 60 degrees of freedom.
12. The $\nu_j$ grid is the integers from `dof_min` to 50; `draw_dof` never returns a value below `dof_min`; with `dof_min` = 5 no draw has $\nu_j<5$.
13. On simulated data with known degrees of freedom the sampler recovers them.
14. The default number of kept draws is 4000.
15. Estimation with the same seed and data gives the same version bit for bit.
16. The radius of every kept draw is at most `rho_max` = 1.005; the radius is the spectral radius of the companion matrix of section 3.6, and `companion_radius` returns the known answer on matrices with known roots.
17. The stability filter is applied before the rotation search; an explosive draw inserted into a simulated sample is removed, and only that draw.
18. Zero stable draws raise an error naming the model and `rho_max`.
19. Version diagnostics contain `estimated`, `stable`, `identified`, and $\text{estimated}\ge\text{stable}\ge\text{identified}$.
20. The named shocks are exactly `ru.monetary`, `ru.inflation`, `ru.fx_risk`, `gl.oil`, `gl.financial` with the signs of the table in section 5, including `ru.cpi` among the falls of `ru.monetary`; unnamed shocks are called `gl.unnamed_k` and `ru.unnamed_k`.
21. For every kept draw all signs hold at response horizons 0, 1, 2 under the matrix $B_0=C\mathrm{diag}(e^{h_0/2})Q$.
22. $Q$ is orthogonal and block diagonal: the response of every global variable to a Russian shock is zero at all horizons.
23. Rotations are drawn from the Haar distribution; at most 20 000 attempts per draw, and a draw without an admissible rotation is dropped.
24. An unconditional simulation with the same seed gives the same paths bit for bit; paths have shape `(R, H+1, n)`, and month 0 equals the last history month.
25. Future $h_{j,t}$ are drawn by the random walk from the last historical value with the same $s_j$; $\omega_{j,t}\sim\mathrm{IG}(\nu_j/2,\nu_j/2)$.
26. No output value is below `lower` for variables with a bound; for variables without `lower` paths equal those without this step.
27. The shadow continues below the bound, and a later rise starts from the shadow: the recursion lags take $s_t$, not $\max(s_t,\underline x)$.
28. A condition on a bounded variable holds on the shadow; a condition "at zero" holds for the shadow within $\tau$ of zero.
29. Under observations every path satisfies all constraints $\mathbf a\le W\mathbf y\le\mathbf b$ (on the shadow, with the interventions executed).
30. Free shocks under observations keep their base law $\mathcal N(0,I)$ truncated to the bounds; shocks of set cells are not drawn.
31. A set puts the value on every path of its group in its month, a shift adds to it; later months read the set value through the lags, and other variables of the same month keep their own errors.
32. An observation row fully determined by the set values leaves the system; its value outside the bounds is `ContradictsModels`. A scenario with a linearly dependent row (threshold $10^{-9}$) is rejected with `InfeasibleScenario` and the constraint numbers.
33. Under observations the volatility path is chosen from 64 candidates weighted by $P(S\mid\theta,h)$; without observations it is drawn from its base law.
34. `latents` declares `volatility_path`.
35. `constraint_law` returns the gaussian law of the rows $W\mathbf y$ after the interventions, grouped by $R$, matching the mean and covariance of `simulate` paths at the same $R$ (checked on a sample).
36. Simulation `diagnostics` contain the size of the shock the observations need, by month in standard deviations.
37. The plugin declares one macro model with id `bvar.sv`; a version is read back only by the model named in `meta["model"]`.
38. The plugin imports only allowed platform modules.
39. The distribution of $\theta$ under observations equals the unconditional one: at each $R$ the same parameter draws are used as without observations.

## 11. Open questions and contradictions between sources

1. **Bound of the BBB spread.** The registry and the package README give `ru.spread_bbb` a bound of 0 (README: history minimum +0.99). The repair plan's results log says the AA, A and BBB spreads were left unbounded.
2. **Who applies the bound.** Repair plan: a shared helper in `model_sdk`, called by both `simulate` functions on the final paths. Package README: the platform, after interventions and tilting (`mst.model_sdk.bounds.observed`). The results log names the helper `lower_bounds`. The answer decides whether `MacroSimulation` holds the observed or the shadow value.
3. **Where the dummy candidate is tuned.** Repair plan: the tuning half of the rolling check's date grid, by the unconditional forecast. Results log: "backtest on the tuning half of the episodes".
4. **Tuning winners not chosen.** $\lambda$, $\mu=\delta$ and `dof_min` (5 or 3) depend on runs the results log marks as pending measurement.
5. **Direction of `mu`, `delta`.** In the Giannone, Lenza and Primiceri (2015) parametrisation smaller values are tighter; in Sims and Zha (1998) the weight multiplies the dummies and larger is tighter. The plan names only the candidates 1 and 5.
6. **Minnesota details.** Only the tightness $\lambda$ is set. The lag decay, the cross-variable factor, how the scales $\sigma_i$ are estimated, and the assumption about intercepts and regime dummy coefficients are not set by the design.
7. **Where shocks live: structural or triangular components.** The model section of the legacy description gives stochastic volatility and Student-t tails to the structural shocks $\varepsilon_{j,t}$; its implementation notes give them to the triangular components $e_{j,t}$ and obtain structural shocks by the rotation $Q$. Rotated independent components with different $h$ and $\omega$ are neither independent nor Student-t; and a rotation chosen at $h_0$ changes the meaning of a named shock over the horizon as $h_t$ moves away from $h_0$.
8. **Sign window.** Legacy description: the month of occurrence and the two following. Repair plan: "`sign_lag = 2` months". This document reads it as horizons 0, 1, 2; the reading "horizons 0 and 1" also fits the plan's wording.
9. **Unrestricted price signs under other shocks.** The repair plan adds `ru.cpi` only to `ru.monetary`; the legacy table leaves prices free under `ru.fx_risk` and `gl.oil`. Exchange-rate pass-through is left to the data, not restricted.
10. **Constant-volatility variant.** The legacy description names a variant `volatility = "constant"` (conjugate BVAR with block exogeneity after Zha, 1999); the package README mentions "two samplers". Whether it belongs to model id `bvar.sv` and whether the filter and dummies apply to it is not stated.
11. **Seasonality in the output.** Where the list of seasonal variables lives differs by source (`src/mst/macro/spec.toml` in the legacy description, the `seasonal` field in the registry, "seasonal adjustment" in the plugin's `spec.toml` per the package README). Whether the seasonal profile is added back to paths for positions and satellites is not stated.
12. **Unit of `ru.credit`.** Legacy description: roubles; registry: mln RUB. In the $100\ln X$ scale this is a level shift and does not affect dynamics.
13. **Tolerance $\tau$.** Its value, and whether the platform or the scenario sets it, is not given in the sources read.
14. **Initial volatility.** The distribution of $h_{j}$ in the first sample month, the thinning `thin` and the burn-in length are not set.
15. **Too few draws.** The repair plan expects about 30 % stable draws and says to stop when fewer than 200 survive all filters, without raising $\rho_{\max}$. Whether the code itself must fail when $0<\text{identified}<200$, or this is a working rule for the engineer, is not stated.
16. **Distribution of $\nu_j$ before data.** Legacy description: uniform on the integers 3 to 50. The plan changes only the floor; it does not confirm that uniformity and the top value 50 remain.
17. **Gates G1–G6.** The repair plan sets them for a version, not for the code; G1 (zero share of draws above $\rho_{\max}$) coincides with check 16, the rest depend on workspace data and are not in the list of code checks.
