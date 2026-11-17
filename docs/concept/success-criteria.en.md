# Criteria of trust in a forecasting model

A forecasting model is trusted when, on past dates, it does not refuse to calculate, its 90 % forecast interval is plausible against the history lived through, and on the held-out half of the dates it beats a simple benchmark by more than two standard errors. Calibration of the interval, the use of the scenario and the tail of the portfolio do not enter the decision yet, so a model with a systematically narrow interval can formally pass. The criteria are implemented in the package [mst.metrics](<../../src/mst/metrics/README.en.md>) and apply in the same way to any model plugin.

## How the check works

The check places the model at a past date $e$ and gives it only the data published by that day. From them the model draws $N$ random paths of every quantity $q$ for the months $t=1,\dots,H$, with a horizon $H$ of up to 14 months, and the paths are compared with the fact $y_{e,q,t}$ known today. The dates come from 32 crisis episodes and from a regular grid of dates chosen without selection. In an episode the model receives the scenario, the realized values of a few quantities over the horizon, and forecasts the rest. The check covers the macro variables, market quantities such as share prices and bond spreads, the single securities of a portfolio and the portfolio as a whole.

Let $x^{(1)},\dots,x^{(N)}$ be the paths of quantity $q$ in month $t$ of window $e$, $\hat F$ their empirical distribution function and $Q_p$ its $p$-quantile. The triple $(e,q,t)$ is a cell. The band of a cell is the 90 % interval $B=[Q_{0.05},Q_{0.95}]$ of width $w=Q_{0.95}-Q_{0.05}$.

## Completeness

The grid must have no date where the data suffice and the model refused to forecast. One such refusal fails the whole grid, since a score over the surviving dates would hide the weakness of the model. When the model lacks the history of a single quantity, only the cells of that quantity drop out. An episode drops out as a whole when such a quantity belongs to its scenario. The report names every dropped quantity, episode and security with the reason.

## Plausibility of the band

This check rejects bands that cannot be taken seriously, before any accuracy is compared. A cell breaks the domain rule when the band leaves the admissible values of the quantity, for example falls below zero for a price. A cell breaks the history rule when

$$
w>\max_{s\le T}y_s-\min_{s\le T}y_s,
$$

where $s$ runs over the months of the whole history of the quantity known on date $e$ and $T$ is the last of them. The history rule reads only the cells whose history spans at least 120 months. A cell of month $t$ breaks the moves rule when

$$
w>\max_s\Delta_s-\min_s\Delta_s,\qquad \Delta_s=y_{s+k}-y_s,\qquad k=t+g,
$$

where $g$ is the number of months from the last known value of the quantity to the end of the history. The band of month $t$ forecasts a change over $k$ months, so it is compared with past changes of the same length. The moves rule applies when such changes cover at least 60 months.

Let $\pi_{q,r}$ be the share of the cells of quantity $q$ breaking rule $r$ among the cells the rule applies to. The model passes when $\pi_{q,r}<0.05$ for every quantity and every rule, and one quantity over the threshold keeps the whole model out. Episodes add a condition on the scenario: every interval condition $c$ of the scenario must have probability $P_0(c)>0.01$ in the forecast without the scenario, otherwise the scenario rests on an event the model considers almost impossible. Point conditions are not held to this threshold, since their probability is set by the precision the condition is given with.

## Accuracy against a benchmark

The error of a cell is the continuous ranked probability score (CRPS)

$$
\operatorname{CRPS}(\hat F,y)=\int_{-\infty}^{\infty}\bigl(\hat F(z)-\mathbf 1\lbrace z\ge y\rbrace\bigr)^2dz=\frac1N\sum_i\lvert x^{(i)}-y\rvert-\frac1{2N^2}\sum_{i,j}\lvert x^{(i)}-x^{(j)}\rvert .
$$

It is zero for a forecast concentrated on the fact and grows with a shifted centre and with an excess width of the band. Model $M$ is compared with benchmark $R$ on the same cells by

$$
\rho_{q,t}=\frac{\sum_e\operatorname{CRPS}^M_{e,q,t}}{\sum_e\operatorname{CRPS}^R_{e,q,t}},\qquad S_{M/R}=\exp\Bigl(\frac1J\sum_t\frac1{K_t}\sum_q\log\rho_{q,t}\Bigr),
$$

where the sums run over the windows scored by both forecasts, $K_t$ is the number of quantities so scored in month $t$ and $J$ the number of such months. Every quantity weighs the same, so the aggregate does not depend on units. $S_{M/R}<1$ means the model is more accurate than the benchmark. The 95 % interval of $S$ comes from a bootstrap of 1000 repetitions over whole windows; the grid dates are drawn in blocks one window long, because neighbouring windows overlap.

The benchmark is a random walk with drift on the known history. When the scenario fixes the path of some quantities, the benchmark is the same walk that knows that path. For the securities and the portfolio the benchmark is historical simulation: past price changes over the same span, valued by the same position models. For market quantities the report adds the naive forecast of a year earlier with the trend. Next to the score of every quantity the report prints the score of the level without it, which shows whether the aggregate rests on one series.

## Decision on a version

The windows are split in advance into a tuning half and a validation half. The model is tuned on the first, and the decision reads only the validation half of the target level. Let $L=\log S_{\text{new}/R}$ and $L_0=\log S_{\text{current}/R}$, and let $\sigma$ and $\sigma_0$ be the bootstrap standard errors of $L$ and of $L-L_0$. A version without admission is rejected whatever $L$. An admitted version is accepted when $L<-2\sigma$ and $L-L_0<-2\sigma_0$, provided no share $\pi_{q,r}$ of the worst quantity grew against the current version. A version is rejected when $L>2\sigma$ or $L-L_0>2\sigma_0$. In every other case the current version stays, because a difference below two standard errors is not shown.

## What the decision does not read

The report computes further quantities that do not enter the decision yet. Calibration is the shares of the facts below $Q_{0.05}$, between $Q_{0.05}$ and the median, between the median and $Q_{0.95}$ and above $Q_{0.95}$. For a correct forecast they are 0.05, 0.45, 0.45 and 0.05, and the band covers 90 % of the facts. On the episodes calibration is biased by selection, since the episodes were chosen as crises.

The use of the scenario is the ratio $S_{M/M_0}$, where $M_0$ is the same model without the scenario. The report gives it per level, per episode and as the share of episodes where it is below one. Scenarios written by a person and by a language model are counted apart, since a language model may remember how the event ended. The direction of the scenario is the share of cells where the scenario moved the median toward the fact. A move smaller than a tenth of the width of the band of $M_0$ counts as noise, and a share below one half means the scenario leads the forecast away from the fact.

The tail of the portfolio is scored by the Acerbi–Székely statistic

$$
Z=\frac1{0.05E}\sum_{e=1}^{E}\frac{\Delta_e\mathbf 1\lbrace\Delta_e>\mathrm{VaR}_e\rbrace}{\mathrm{ES}_e}-1,
$$

where $\Delta_e$ is the realized loss of the portfolio in window $e$, $\mathrm{VaR}_e$ the forecast 95th percentile of the loss, $\mathrm{ES}_e$ the mean loss over the worst 5 % of the paths and $E$ the number of windows. With a correct tail $\mathbb E Z=0$, and a positive $Z$ means the extreme losses are understated. For a model of market quantities the share of paths outside the range of the training sample must be below 1 %, otherwise its tail is marked unverified.

## Where in the code

[baselines.py](<../../src/mst/metrics/baselines.py>) checks the domain, history and moves rules and the condition on the scenario; [cells.py](<../../src/mst/metrics/cells.py>) computes the ranges of the history and of the moves and the CRPS. [sanity.py](<../../src/mst/metrics/sanity.py>) collects the shares per quantity and the admission, [quality.py](<../../src/mst/metrics/quality.py>) the relative score $S$ and the bootstrap. The benchmarks are built in [baselines.py](<../../src/mst/metrics/baselines.py>) and [levels.py](<../../src/mst/metrics/levels.py>), the decision is taken in [decision.py](<../../src/mst/metrics/decision.py>), and [es.py](<../../src/mst/metrics/es.py>) computes $Z$. The episodes, their halves and the grid of dates are set in [episodes.toml](<../../bench/model-backtest/episodes.toml>); running a check is described in the [package README](<../../src/mst/metrics/README.en.md>).

## What next

While calibration and the use of the scenario stay for reference, the decision admits a model with a systematically narrow band or with a scenario that worsens the forecast. The first option keeps the criteria as they are and costs nothing, and the decision then does not guarantee the right frequency of facts inside the band. The second option adds to the decision the coverage of the band on the grid, for example between 0.85 and 0.95, and the condition $S_{M/M_0}<1$ on the validation half of the episodes. It rejects more versions and needs more dates for the intervals of the scores to narrow. The third option adds the coverage alone and keeps the use of the scenario for reference, because on 32 episodes its interval is wide.
