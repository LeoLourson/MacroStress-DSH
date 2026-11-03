# Random walk with structural shocks

The `rw` plugin gives one level-0 macro model, `rw.structural`. It is chosen explicitly, on a par with `bvar.sv` and `msvar.markov`; it does not replace the default model. This text was written before the code and states what the code must do, so where they differ, the code is fixed.

## Why this model

The metrics compare every macro model with a baseline, the random walk with drift (`random_walk` in `mst.metrics.baselines`). The baseline continues each variable on its own: next month's value is the current one plus the mean monthly change plus normal noise with the sample spread of the changes. If a complex model is not more accurate than the baseline by CRPS, the baseline stays the best forecast, but a scenario cannot be put on it. Its variables are independent, so a condition on the key rate changes nothing in the law of the exchange rate. Besides, by the [general scheme](../../../../docs/concept/formalization.en.md) a scenario names the causes of the event, the coordinates of the model's random outcome whose law it changes, and the baseline has no coordinates a scenario could name: the corpus scenarios and the backtest episodes name the shocks `ru.monetary`, `gl.oil` and others.

`rw.structural` adds to the baseline the two things a scenario needs: a joint normal law of the monthly changes and a decomposition of the changes into named shocks. The model takes the drift and spread of every variable from the baseline unchanged. So without a scenario the law of a variable known in the last month of history is the same for the model and the baseline, and the difference of their CRPS on the episodes measures only the contribution of the scenario. The difference of the CRPS of `bvar.sv` and `rw.structural` under one scenario measures what the dynamics of `bvar.sv` add.

## Model

Denote by $y_t\in\mathbb R^n$ the vector of the $n=23$ macro variables of the registry `src/mst/macro/registry.toml` in month $t$ on the model scale: rates, spreads and unemployment in their own units, prices, exchange rates, indices and volumes as $100\ln X$. The variables follow the order of the panel: first $n_g=4$ global ones (`gl.`), then $n_r=19$ domestic ones (`ru.`). The model:

$$
y_t=y_{t-1}+\mu+Bz_t,\qquad z_t\sim N(0,I_n)\ \text{independent over }t,
$$

where $\mu$ is the drift, $B$ is an $n\times n$ matrix, and $\Sigma=BB^\top$ is the covariance of the monthly changes. Column $j$ of $B$ is the response of all variables to a unit shock $z_{t,j}$. The model has no lags, so a shock moves the level of variable $i$ by $B_{ij}$ in the month of impact, and the move stays in all later months.

The data determine $\Sigma$ but not $B$: for any orthogonal $Q$ the matrix $BQ$ gives the same covariance. Put

$$
B=CQ,\qquad Q=\begin{pmatrix}Q_g&0\\0&Q_r\end{pmatrix},
$$

where $C$ is the lower-triangular Cholesky factor of $\Sigma$, and $Q_g$ and $Q_r$ are orthogonal matrices of orders $n_g$ and $n_r$. Such a $B$ is block lower-triangular: domestic shocks do not move global variables, while global ones move all. This is the small open economy assumption, as in `bvar.sv`; the data do not test it, because every $\Sigma$ has such a decomposition.

The model parameters are $\theta=(\mu,\Sigma,Q)$. The level-0 outcome in the sense of the general scheme is

$$
\omega^0=(\theta,u,\mathbf z),\qquad \mathbf z=(z_1,\dots,z_H),
$$

where $u$ are the standard normal coordinates of the values of the last month of history not yet published at the calculation date (section "Initial value"). The causes of a scenario are chosen from the coordinates $z_{t,j}$ and $u$.

## Data and the estimate of the spread

The model reads the same panel as `bvar.sv` and `msvar.markov`: months from 2012-01 to the calculation date, only values published by that date, five series seasonally adjusted by the platform, a month without a value left missing. Month $t=0$ is the last month in which at least one variable is known; the horizon is months $t=1,\dots,H$.

Denote by $\mathcal T_i$ the months in which both $y_{i,t}$ and $y_{i,t-1}$ are known, by $N_i$ their number, by $\Delta y_{i,t}=y_{i,t}-y_{i,t-1}$ the change. For $N_i\ge3$

$$
\mu_i=\frac{1}{N_i}\sum_{t\in\mathcal T_i}\Delta y_{i,t},\qquad
\sigma_i^2=\frac{1}{N_i}\sum_{t\in\mathcal T_i}\big(\Delta y_{i,t}-\mu_i\big)^2 .
$$

These are `drift` and `noise` of `random_walk` with a one-month step on the same panel; the comparison with the baseline rests on this equality, so the code checks it to $10^{-12}$.

The housing price `ru.housing_price` is published less often than monthly: on the panel to 2026-09 it has 32 values and no monthly change. The baseline returns a missing value for it, but the model needs its law, because the satellites read it. Let $s_0<s_1<\dots<s_m$ be the months in which variable $i$ is known, $k_j=s_j-s_{j-1}$ and $D_j=y_{i,s_j}-y_{i,s_{j-1}}$. By the model $D_j\sim N(k_j\mu_i,k_j\sigma_i^2)$ independently over $j$, and the maximum likelihood estimates are

$$
\mu_i=\frac{\sum_{j=1}^{m}D_j}{\sum_{j=1}^{m}k_j},\qquad
\sigma_i^2=\frac{1}{m}\sum_{j=1}^{m}\frac{\big(D_j-k_j\mu_i\big)^2}{k_j}.
$$

With all $k_j=1$ they equal the monthly ones. If a variable has fewer than three monthly changes and fewer than three intervals between observations, or if $\sigma_i=0$, the estimation stops with an error that names the variable.

## Correlations

The panel has no month in which the changes of all 23 variables are known, because of the housing price, so $\Sigma$ is estimated by pairs. For two monthly variables $\hat\rho_{ij}$ is the sample Pearson correlation of their changes over the common months $\mathcal T_i\cap\mathcal T_j$. For a pair with the housing price, take the common observation months $r_0<\dots<r_q$, the intervals $k_l=r_l-r_{l-1}$ and the changes of both variables $D^i_l$, $D^j_l$ over these intervals; the maximum likelihood estimate is

$$
\hat\rho_{ij}=\frac{1}{q\sigma_i\sigma_j}\sum_{l=1}^{q}\frac{\big(D^i_l-k_l\mu_i\big)\big(D^j_l-k_l\mu_j\big)}{k_l}
$$

clipped to $[-1,1]$. With fewer than three common changes or intervals, $\hat\rho_{ij}=0$.

The matrix of pairwise estimates may fail to be positive definite. If its smallest eigenvalue is below $10^{-8}$, it is replaced by the nearest correlation matrix $\rho$ in the Frobenius norm with eigenvalues at least $10^{-8}$ (Higham, 2002). The correction keeps the unit diagonal, so in

$$
\Sigma=\operatorname{diag}(\sigma)\,\rho\,\operatorname{diag}(\sigma)
$$

the spread of every variable stays that of the baseline.

## Named shocks

The shocks are named and defined by the signs of the response as in `bvar.sv`, so that one scenario fits both models. The table `signs.toml`:

| Shock | Block | Up | Down |
|---|---|---|---|
| `ru.monetary` | domestic | `ru.key_rate`, `ru.ruonia`, `ru.ofz_1y` | `ru.usdrub`, `ru.imoex`, `ru.cpi` |
| `ru.inflation` | domestic | `ru.cpi`, `ru.key_rate` | `ru.real_wage` |
| `ru.fx_risk` | domestic | `ru.usdrub`, `ru.ofz_10y`, `ru.spread_bbb` | `ru.imoex` |
| `gl.oil` | global | `gl.brent` | `ru.usdrub` |
| `gl.financial` | global | `gl.vix` | `gl.ust_10y`, `ru.imoex` |

A matrix $Q$ is admissible if the column of $B$ of every named shock is strictly positive on the "up" variables and strictly negative on the "down" variables. The other columns are the unnamed shocks `gl.unnamed_1`, `gl.unnamed_2`, `ru.unnamed_1`, …, `ru.unnamed_16`. The set of shocks $J$ and its order match `bvar.sv`: the global block first, then the domestic one, within a block the named shocks in the order of `signs.toml`, then the unnamed ones. `bvar.sv` checks the signs in the month of impact and `sign_lag` months after it, while the response of the walk is constant, so the check in the month of impact is enough, and the sign asserts a permanent shift of the level.

The signs pick out a set of admissible $Q$. A version stores $M=1000$ of its elements with equal weights, and the law of $\theta$ is uniform on this set, so the identification uncertainty enters the cost of a scenario with named causes, as in `bvar.sv`.

Each block $b\in\lbrace g,r\rbrace$ is searched separately with the algorithm of Rubio-Ramírez, Waggoner, Zha (2010). Denote by $C_b$ the columns of $C$ of block $b$ and by $k_b$ the number of its named shocks ($k_g=2$, $k_r=3$). The first $k_b$ columns of a uniformly distributed orthogonal matrix of order $n_b$ are drawn by the QR decomposition of a standard normal $n_b\times k_b$ matrix, with the column signs fixed by the signs of the diagonal of $R$. A candidate is accepted if every response $C_bq_l$, as is or with the opposite sign, satisfies the signs of its shock; the sign of the column is then fixed, which keeps the law uniform. The accepted columns are completed to an orthogonal matrix by the same QR decomposition with $n_b-k_b$ new normal columns. If no candidate out of 20000 is accepted, the estimation stops with an error that names the block. All draws come from one generator seeded by the specification; a version has one $\Sigma$, so the set $Q^{(1)},\dots,Q^{(M)}$ is an independent sample from the uniform law on the admissible set.

## Initial value

Series are published with different delays, so in month 0 part of the variables is unknown. The general scheme requires the law of $x^0$ given all available information, and the known moves of other variables in the missing months carry information about the missing one through $\Sigma$. So the unknown components of $y_0$ get a Gaussian law given all observations of the panel. It comes from the Kalman filter for the state $y_t$ with the model's transition, exact observations of the known components of every month and a diffuse initial law (Durbin, Koopman, 2012). The law is written as $m_0+L_0u$ with $u\sim N(0,I_q)$, where $q$ is the number of unknown components, and does not depend on $Q$.

The baseline crosses the missing months with independent steps of its variable, so for variables unknown in month 0 the laws of the model and the baseline differ.

## Forecast without a scenario

$$
y_t=y_0+t\mu+B\sum_{s=1}^{t}z_s,\qquad t=1,\dots,H.
$$

With $y_0$ known, $y_t\sim N(y_0+t\mu,\ t\Sigma)$. The law does not depend on $Q$ and for every variable equals the law of `random_walk`.

Rates, unemployment and the BBB spread have a lower bound in the registry. As in `bvar.sv`, the model draws a shadow value $s_t$ without the bound, conditions and their cost are computed on the shadow, and after the impacts the platform reads the observed value $\max(s_t,\text{lower})$ (`mst.model_sdk.bounds.observed`). The baseline applies the same bound, so the observed laws match too.

## Scenario

A scenario reaches the model as a plan: the horizon $H$, the constraints $\mathbf a\le W\mathbf y\le\mathbf b$ on the path $\mathbf y=(y_1,\dots,y_H)$ and the causes $D$. The platform widens an exact value beforehand to an interval of half-width $\tau$. It subtracts the impacts $O$ from the bounds and adds them to the paths after the draw, so the model does not see them, and by the general scheme an impact on a variable does not move the other level-0 variables.

Denote $\varepsilon=(u,\mathbf z)$. The path is linear in $\varepsilon$:

$$
W\mathbf y=c+Z\varepsilon,
$$

where $c$ is the value of $W\mathbf y$ at $\varepsilon=0$. A term $wy_{i,t}$ of a constraint contributes $w(L_0)_{i\cdot}$ to the row of $Z$ in the columns of $u$ (zero if $y_{i,0}$ is known) and $wB_{ij}$ in the column of $z_{s,j}$ for every $s\le t$.

### Causes not named

Then by the general scheme the causes are all coordinates except $\theta$, and $R=\theta$. The covariance $ZZ^\top$ depends on $B$ only through $BB^\top=\Sigma$, and a version has one $\mu$ and one $\Sigma$, so $P(S\mid\theta)=P(S)$ for every $Q$. By the corollary of the superposition theorem $P_s=P(\cdot\mid S)$: the vector $\varepsilon$ has the standard normal law truncated to the box $\mathbf a-c\le Z\varepsilon\le\mathbf b-c$, and

$$
\mathrm{dist}(s)=-\log P(S),\qquad P(S)=\Pr\big(\mathbf a\le N(c,ZZ^\top)\le\mathbf b\big).
$$

So the check of the cost against history, which estimates $-\log P(S)$, here checks $\mathrm{dist}(s)$ itself, while for models with uncertain parameters history does not check the difference $\mathrm{dist}(s)+\log P(S)$. For the same reason the uniformity check on the episodes is exact: under the true model the forecast equals the law of the outcome. All paths form one group.

### Causes named

Let $D\subseteq J$. Denote by $\varepsilon_D$ the coordinates $z_{t,j}$ with $j\in D$ and $t=1,\dots,H$, by $\varepsilon_{\bar D}$ the other coordinates of $\varepsilon$ together with $u$. Then $R=(\theta,\varepsilon_{\bar D})$ and

$$
W\mathbf y=c+Z_{\bar D}\varepsilon_{\bar D}+Z_D\varepsilon_D .
$$

The rotation, $u$ and the shocks outside $D$ keep the base law, and given $R$ the vector $\varepsilon_D$ has the standard normal law truncated to the box

$$
\mathbf a-c-Z_{\bar D}\varepsilon_{\bar D}\le Z_D\varepsilon_D\le\mathbf b-c-Z_{\bar D}\varepsilon_{\bar D}.
$$

The cost $\mathrm{dist}(s)=\mathbb E_P[-\log P(S\mid R)]$ is at least $-\log P(S)$, depends on $Q$ and $\varepsilon_{\bar D}$, and history does not check it. A group of paths is one value of $R$: its paths share the rotation $Q^{(m)}$ and $\varepsilon_{\bar D}$ and differ only in $\varepsilon_D$.

A scenario contradicts the model if there is a vector $\nu$ with $\nu^\top Z_D=0$, $\nu^\top Z_{\bar D}\ne0$ and the box bounded along $\nu$. Then $\nu^\top W\mathbf y$ given $R$ is not random and lies outside the box with positive probability, so $P(S\mid R)=0$ with positive probability, and by the general scheme the scenario has no admissible measures. The model rejects such a scenario and names the constraints. Most often the constraints fix the variables of a set $I$ in every month of the horizon, as in the backtest episodes. The rows of month $t$ then fix $\Delta y_{I,t}=\mu_I+B_{I\cdot}z_t$, the matrix $Z_D$ is block triangular with diagonal blocks $B_{I,D}$, and the scenario is feasible if and only if $\operatorname{rank}B_{I,D}=\lvert I\rvert$. So $D$ needs at least as many shocks as fixed variables, and since domestic shocks do not move global variables, $D$ must hold at least as many global shocks as fixed global variables. An episode that fixes Brent and the key rate is served by $D=\lbrace$`gl.oil`, `ru.monetary`$\rbrace$, while $D=\lbrace$`ru.monetary`, `ru.fx_risk`$\rbrace$ does not fit it.

### Draw and the law of the conditions

The truncated normal law is drawn by the common mechanics `mst.model_sdk.conditional` after Antolín-Díaz, Petrella, Rubio-Ramírez (2021): it fixes exact rows, draws interval rows with a Gibbs sampler (Geweke, 1991) in 50 sweeps from a point inside the box, merges repeated rows and rejects linearly dependent ones with the threshold $10^{-9}$. With named causes the rotation number of a group is drawn uniformly from $1,\dots,M$. The paths are a function of the version, the plan, the number of paths and the seed, so one seed gives bitwise identical paths.

`constraint_law` returns the Gaussian law of $W\mathbf y$ given $R$, one row per group. Without named causes there is one row: mean $c$, covariance $ZZ^\top$. With named causes a group has mean $c+Z_{\bar D}\varepsilon_{\bar D}$ and covariance $Z_DZ_D^\top$ under its rotation. The log weight of a row is zero; the model has no hidden coordinates. The platform computes the probability of the box and the slope for conditions on means.

## Difference from the baselines

For a variable known in month 0 the law without a scenario equals the law of `random_walk`, and the CRPS of the model and the baseline differ only by Monte Carlo error. The laws differ for variables unknown in month 0 and for the housing price, which has no baseline.

The conditional forecast baseline `conditional_walk` regresses the change of a variable on the changes of the fixed variables and substitutes their actual path. If the causes are not named and the variables $c$ are fixed in all months of the horizon, the months' changes are independent, and the conditional law of the change of the other variables $y$ in month $t$ is, up to $\tau$,

$$
N\big(\mu_y+\Sigma_{yc}\Sigma_{cc}^{-1}(\Delta c_t-\mu_c),\ \Sigma_{yy}-\Sigma_{yc}\Sigma_{cc}^{-1}\Sigma_{cy}\big),
$$

so the model gives the same linear projection as the baseline. The differences: the baseline estimates the regression over the months where all the changes involved are known, the model over pairwise moments; the baseline divides the residual sum of squares by the degrees of freedom; in a month without an outcome the baseline substitutes the mean change, while the model leaves the month free and the next known level fixes the sum of changes; the initial values differ for the reason given above; the model gives a joint law of all variables, the baseline the law of each one separately. Scenarios with named causes have no baseline.

## Assumptions

There is no mean reversion: the spread at horizon $t$ grows as $\sqrt t$ for all variables, rates and spreads included, and the sample drift continues linearly. The changes are Gaussian with constant covariance, without stochastic volatility, regimes or heavy tails, so crisis months enter $\sigma_i$ and $\rho$ like any other. The jump of the spreads `ru.spread_aa`, `ru.spread_a`, `ru.spread_bbb` in 2021-01 (the splice of the MOEX indices) enters their $\sigma_i$ as one large change, as in the baseline. The uncertainty of $\mu$ and $\Sigma$ is not counted; the identification uncertainty is counted by the set of rotations. Domestic variables do not affect global ones. The correlations of pairs with the housing price rest on 31 intervals or fewer, and the correction to positive definiteness changes the pairwise estimates.

## Parameters

| Parameter | Value | Meaning |
|---|---|---|
| `start` | 2012-01-01 | the first month of the panel, as in `bvar.sv` and `msvar.markov` |
| `min_changes` | 3 | the smallest number of changes or intervals for $\mu_i$, $\sigma_i$, $\hat\rho_{ij}$, as in `random_walk` |
| `eig_floor` | $10^{-8}$ | the lower bound of the eigenvalues of $\rho$ |
| `rotations` | 1000 | the number $M$ of rotations of a version |
| `rotation_tries` | 20000 | the largest number of candidates for one $Q_b$, as in `bvar.sv` |
| `seed` | 20261029 | the seed of the rotation search |
| `signs.toml` | the table above | named shocks and signs |

## Plugin and version

`plugin = Plugin(plugin_id="rw", ..., model_sdk=">=3.0,<4.0", macro=(MODEL,))`, the variables are the whole registry. The plugin imports only `mst.model_sdk`, `mst.macro.panel`, `mst.macro.registry`, `mst.plugin_sdk` and its own package; the plugin rules forbid importing from `mst.plugins.bvar`.

`estimate` writes to the meta `as_of`, `ids`, `n_global`, `months`, `shocks` (the set $J$), `diagnostics` and `spec`, and to the arrays `panel_y`, `panel_approximate`, `drift` and `sigma` of length $n$, `corr` and `chol` of size $n\times n$, `rotation` of size $M\times n\times n$, `start_mean` of length $n$ (the known components of $y_0$ and $m_0$ in place of the unknown ones) and `start_chol` of size $n\times q$ ($L_0$ in the rows of the unknown components). `load` restores the model without re-estimating. `simulate` returns `MacroSimulation`: paths of shape $(R,H+1,n)$ on the shadow, where row $t=0$ is the path's $y_0$, `history` is the same $y_0$ because the model has one lag, and the group numbers.

In `diagnostics` a version writes $N_i$ and the estimation method of every variable, the smallest number of common changes over pairs and the pairs with $\hat\rho_{ij}=0$ for lack of data, $\lVert\rho-\hat\rho\rVert_F$ and the smallest eigenvalue of $\hat\rho$, the share of accepted rotation candidates by block. A calculation with named causes writes the median over paths of the largest $\lvert z_{t,j}\rvert$, $j\in D$, in every month: the size of the driving shocks in standard deviations.

## Checks for the code

1. For every monthly variable `drift` and `sigma` equal `drift` and `noise` of `random_walk` on the same panel to $10^{-12}$.
2. On a simulated walk with gaps the housing price estimates are unbiased within error, and with all $k_j=1$ equal the monthly ones.
3. `corr` is symmetric, has a unit diagonal and is positive definite; on a pairwise matrix with a negative eigenvalue the correction gives a smallest eigenvalue of at least $10^{-8}$ and writes the distance.
4. $CC^\top=\Sigma$ and $B^{(m)}B^{(m)\top}=\Sigma$ to $10^{-10}$ for every $m$.
5. $B^{(m)}_{ij}=0$ for a global $i$ and a domestic shock $j$; every $B^{(m)}$ strictly satisfies the sign table.
6. A block without an admissible rotation within `rotation_tries` candidates gives an error naming the block.
7. `shocks` of a version equals the set $J$ of the `bvar.sv` version of the same date, order included.
8. Without a scenario the sample mean and covariance of $y_t-y_0$ agree with $t\mu$ and $t\Sigma$, and replacing the set of rotations does not change the quantiles beyond Monte Carlo error.
9. Without a scenario the paths of a variable known in month 0, after the bound, and the paths of `random_walk` with the same number of paths are indistinguishable by the Kolmogorov–Smirnov test in every month of the horizon at the 1 % level with a Bonferroni correction over months.
10. On a panel with a ragged edge the law of the unknown components of $y_0$ equals direct Gaussian conditioning of the changes on the known values.
11. Without named causes, for one constraint on one variable the cost equals $-\log(\Phi(\beta)-\Phi(\alpha))$ for the standardized bounds $\alpha$, $\beta$, and the share of unconditional paths in the box agrees with $P(S)$.
12. Without named causes, under exact constraints the sample mean and covariance of the changes of the other variables agree with the projection formula of the section "Difference from the baselines".
13. With named causes the interval conditions hold on every path, the exact ones within $\tau$.
14. A scenario that fixes a global variable with $D$ of domestic shocks only, and a scenario with $\lvert D\rvert<\lvert I\rvert$ under constraints of every month, are rejected with the numbers of the constraints.
15. `constraint_law` agrees with `simulate`: the share of unconditional paths in the box equals the probability of the row within Monte Carlo error, with named causes within every group.
16. One seed gives bitwise identical paths, different seeds different ones.
17. An impact moves only its variable in its months, and a condition on that variable holds on the final path.
18. After the platform reads them, no path of a variable with `lower` lies below the bound; without `lower` the output equals the shadow.
19. `load` after `estimate` gives a model with the same paths as before saving.
20. The `rw` folder without `__init__.py` does not break plugin discovery, and with code the plugin declares one macro model `rw.structural`.

## Open questions

A set of rotations or one rotation. The text above takes a set of $M$ equally likely admissible $Q$, as in `bvar.sv`, and then the cost of a scenario with named causes is averaged over the identification uncertainty. One $Q$, the closest to the median of the responses over the admissible set (Fry, Pagan, 2011), simplifies the model, but under the same scenario the spread of results with named causes will be narrower than in `bvar.sv`. The laws without named causes do not depend on this choice.

Initial value. The text above conditions the unknown components of month 0 on the known moves of other variables, as the general scheme requires. If instead the ragged edge is crossed with independent steps of each variable, as in `random_walk`, the law without a scenario will equal the baseline for all variables, but the model will not use part of the published data.

Rotation search code. The functions `haar_columns`, `complete` and `rotate` of `mst.plugins.bvar.identify` are needed by this model too. They can move to `mst.model_sdk` and be imported by both models, or be copied into the `rw` package.

Housing price. The estimates of its spread and correlations are derived as maximum likelihood estimates of a walk with gaps, and there is no baseline to check them against.

## References

Antolín-Díaz J., Petrella I., Rubio-Ramírez J. (2021). Structural scenario analysis with SVARs. Journal of Monetary Economics 117.

Durbin J., Koopman S. J. (2012). Time Series Analysis by State Space Methods. 2nd ed. Oxford University Press.

Fry R., Pagan A. (2011). Sign restrictions in structural vector autoregressions: a critical review. Journal of Economic Literature 49(4).

Geweke J. (1991). Efficient simulation from the multivariate normal and Student-t distributions subject to linear constraints and the evaluation of constraint probabilities.

Higham N. (2002). Computing the nearest correlation matrix — a problem from finance. IMA Journal of Numerical Analysis 22(3).

Rubio-Ramírez J., Waggoner D., Zha T. (2010). Structural vector autoregressions: theory of identification and algorithms for inference. Review of Economic Studies 77(2).
