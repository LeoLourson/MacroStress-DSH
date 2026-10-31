"""Bayesian estimation of the Markov-switching VAR of the scheme's section 1.

The observation equation and the Markov kernel are the analysis's (1) and (2):

    y_t = c_{s_t} + A_{1,s_t} y_{t-1} + ... + A_{p,s_t} y_{t-p} + u_t,  u_t | s_t ~ N(0, S_{s_t}),
    P(s_t = j | s_{t-1} = i) = p_ij,

with the per-regime parameters in theta_0's posterior. The sampler is a Gibbs chain that
alternates:

- the regime path s_{p..T-1} by forward-filtering backward-sampling (Carter and Kohn,
  1994, Biometrika 81(3)), given the per-regime parameters and the transition matrix;
- the coefficients of every regime equation by equation, given the path: the orthogonalised
  errors e = A_s u are independent across equations, so each equation's posterior is a
  regression on the months of its regime with Minnesota prior and the triangular weights of
  Carriero, Clark and Marcellino (2019, Journal of Econometrics 212(1));
- the covariance of every regime from the residuals of its months, conjugate
  inverse-Wishart, factored as ``C_s diag(exp(h_s)) C_s'`` with ``C_s`` unit lower
  triangular;
- the transition matrix row by row, Dirichlet prior plus the counts of the path;
- the unknown months of the panel by their conditional normal given the path and the
  parameters, through the banded precision of the whole history (data augmentation).

Months the panel does not know are drawn as part of every sweep; with a complete panel the
coefficients of the two steps above are drawn without conditioning on anything unobserved.

Regimes are labelled by total volatility: after every sweep the regimes are reordered so
the one declared first has the smallest trace of its covariance, which is what makes
``regime.stress`` of the signs table the stressful one (the label-switching fix of
Frühwirth-Schnatter, 2006, Finite Mixture and Markov Switching Models, Springer). The
chain starts from one VAR over the whole panel with its months split by residual size, so
the first sweeps already see two regimes.

The Minnesota prior pulls a variable's own first lag to 1 (to 0 if the spec calls it
stationary) and every other lag coefficient to 0, with standard deviation
``lambda sigma_i / (l sigma_k)`` for lag ``l`` of variable ``k`` in equation ``i``;
``sigma_k`` is the scale of the monthly changes of ``k``. The constant is left practically
free. The prior is the same in every regime: the months of a regime are few, and the prior,
not the regime, is where the shrinkage lives.
"""

from dataclasses import dataclass, field
from datetime import date

import numpy as np
from scipy import sparse
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
from scipy.stats import invwishart

from mst.macro.panel import Panel

from .spec import Spec
from .var import allowed, regressors, split

LOOSE = 1e6
"""Prior variance factor of the constant (times sigma_i^2)."""
COVARIANCE_DF = 2
"""Prior degrees of freedom of a regime's covariance beyond its size."""
TINY = 1e-300
"""The floor every probability is clipped at before its logarithm is taken."""


@dataclass(frozen=True)
class Posterior:
    """``S`` draws of the MS-VAR: the parameters of every regime and the path of history."""

    ids: list[str]
    n_global: int
    months: list[date]
    lags_count: int
    n_regimes: int
    const: np.ndarray  # (S, m, n)
    lags: np.ndarray  # (S, m, p, n, n)
    chol: np.ndarray  # (S, m, n, n) unit lower triangular C_s: u_t = C_s e_t
    logvar: np.ndarray  # (S, m, n) h_s of e_t in regime s
    transition: np.ndarray  # (S, m, m) p_ij; every row sums to 1
    states: np.ndarray  # (S, T - p) the sampled regime of each equation month
    history: np.ndarray  # (S, p, n) the last p months, drawn where unknown
    diagnostics: dict = field(default_factory=dict)
    # (S, m, n, n) Q of mst.plugins.msvar.identify: B_s = C_s diag(exp(h_s / 2)) Q_s
    rotation: np.ndarray | None = None
    shocks: list[str] = field(default_factory=list)  # the economic shock of each column of B
    regime_ids: list[str] = field(default_factory=list)  # the regime element of each regime

    @property
    def draws(self) -> int:
        return len(self.const)


@dataclass(frozen=True)
class Prior:
    mean: np.ndarray  # (n, k)
    var: np.ndarray  # (n, k), equation i scaled by sigma_i^2
    scale: np.ndarray  # (n,) sigma_i
    mask: np.ndarray  # (n, k) allowed coefficients


def scales(y: np.ndarray) -> np.ndarray:
    """``sigma_k``: the standard deviation of monthly changes, from consecutive observations
    ``(y_t - y_s) / sqrt(t - s)``, so sparse series count too."""
    result = np.empty(y.shape[1])
    for k in range(y.shape[1]):
        months = np.flatnonzero(~np.isnan(y[:, k]))
        if len(months) < 3:
            raise ValueError(f"variable {k} has fewer than three values")
        changes = np.diff(y[months, k]) / np.sqrt(np.diff(months))
        result[k] = changes.std()
    if np.any(result <= 0):
        raise ValueError("a variable never changes")
    return result


def minnesota(panel: Panel, spec: Spec) -> Prior:
    """The Minnesota prior, the same in every regime; see the module docstring."""
    n, p = len(panel.ids), spec.lags
    unknown = set(spec.stationary) - set(panel.ids)
    if unknown:
        raise ValueError(f"stationary names unknown variables: {', '.join(sorted(unknown))}")
    sigma = scales(panel.y)
    mean = np.zeros((n, 1 + n * p))
    var = np.full(mean.shape, LOOSE)
    for i, id_ in enumerate(panel.ids):
        mean[i, 1 + i] = 0.0 if id_ in spec.stationary else 1.0
    for lag in range(1, p + 1):
        block = slice(1 + (lag - 1) * n, 1 + lag * n)
        var[:, block] = (spec.tightness / (lag * sigma[None, :])) ** 2
    return Prior(mean, var * sigma[:, None] ** 2, sigma, allowed(n, panel.n_global, p))


def fill(y: np.ndarray) -> np.ndarray:
    """Linear interpolation of the unknown months, flat beyond the first and last value."""
    filled = y.copy()
    grid = np.arange(len(y))
    for k in range(y.shape[1]):
        known = ~np.isnan(y[:, k])
        filled[:, k] = np.interp(grid, grid[known], y[known, k])
    return filled


def unit_lower(sigma: np.ndarray):
    """``(C, log diag)`` with ``Sigma = C diag(exp(h)) C'`` and ``C`` unit lower triangular."""
    root = np.linalg.cholesky(sigma)
    diagonal = np.diag(root)
    return root / diagonal[None, :], 2 * np.log(diagonal)


def _keep(spec: Spec):
    total = spec.sampler.burn + spec.sampler.draws * spec.sampler.thin
    for step in range(total):
        keep = step >= spec.sampler.burn and (step - spec.sampler.burn) % spec.sampler.thin == 0
        yield keep


def _choice(weights: np.ndarray, rng) -> int:
    """One draw from a distribution given by weights; they may not sum exactly to one."""
    cumulated = np.cumsum(weights)
    return min(int(np.searchsorted(cumulated, rng.random() * cumulated[-1])), len(weights) - 1)


def stationary(transition: np.ndarray) -> np.ndarray:
    """The stationary distribution of a row-stochastic matrix; uniform when degenerate."""
    m = len(transition)
    values, vectors = np.linalg.eig(transition.T)
    close = np.isclose(values.real, 1.0, atol=1e-8)
    if not close.any():
        return np.full(m, 1.0 / m)
    weights = np.abs(vectors[:, int(np.argmax(close))].real)
    return weights / weights.sum()


def ffbs(
    target: np.ndarray,
    x: np.ndarray,
    coefficients: list[np.ndarray],
    sigmas: list[np.ndarray],
    transition: np.ndarray,
    initial: np.ndarray,
    rng,
) -> np.ndarray:
    """The regime path of the equation months by forward-filtering backward-sampling.

    Given the path each month is one of ``m`` regressions, so its density under regime ``j``
    is normal; the filtered probabilities run forward in log space and the path is sampled
    backward, month by month, each state given its successor (Carter and Kohn, 1994).
    """
    n, count = target.shape[1], len(target)
    m = len(sigmas)
    logf = np.empty((count, m))
    for j in range(m):
        difference = target - x @ coefficients[j].T
        root = np.linalg.cholesky(sigmas[j])
        whitened = np.linalg.solve(root, difference.T).T
        logf[:, j] = (
            -0.5 * (whitened**2).sum(axis=1)
            - np.log(root.diagonal()).sum()
            - 0.5 * n * np.log(2 * np.pi)
        )
    log_transition = np.log(np.clip(transition, TINY, None))
    log_xi = np.log(np.clip(initial, TINY, None))
    filtered = np.empty((count, m))
    for t in range(count):
        if t:
            log_predicted = logsumexp(log_xi[:, None] + log_transition, axis=0)
        else:
            log_predicted = log_xi
        log_updated = log_predicted + logf[t]
        log_xi = log_updated - logsumexp(log_updated)
        filtered[t] = np.exp(log_xi)
    states = np.empty(count, dtype=int)
    states[-1] = _choice(np.exp(log_xi), rng)
    for t in range(count - 2, -1, -1):
        weights = filtered[t] * np.clip(transition[:, states[t + 1]], TINY, None)
        states[t] = _choice(weights, rng)
    return states


def _coefficients(target, x, coefficients, a, variance, prior, rng):
    """Draws each equation's coefficients of one regime given the others.

    Equation ``i`` enters the orthogonalised errors ``e_j = sum_k a_jk u_k`` of ``j >= i``:
    ``e_j + a_ji x pi_i = a_ji x pi_i + e_j`` is a regression on ``x`` with weights
    ``a_ji^2 / variance_j`` — the triangular algorithm of Carriero, Clark and Marcellino
    (2019), on the months of the regime. An empty regime draws from the prior.
    """
    coefficients = coefficients.copy()
    n = len(a)
    u = target - x @ coefficients.T
    for i in range(n):
        columns = prior.mask[i]
        xi = x[:, columns]
        fitted = xi @ coefficients[i, columns]
        e = u @ a[i:].T
        pseudo = e + fitted[:, None] * a[i:, i][None, :]
        weight = a[i:, i] ** 2 / variance[:, i:]
        response = (pseudo * (a[i:, i] / variance[:, i:])).sum(axis=1)
        prec = np.diag(1 / prior.var[i, columns]) + xi.T @ (xi * weight.sum(axis=1)[:, None])
        root = np.linalg.cholesky(prec)
        rhs = prior.mean[i, columns] / prior.var[i, columns] + xi.T @ response
        mean = np.linalg.solve(root.T, np.linalg.solve(root, rhs))
        draw = mean + np.linalg.solve(root.T, rng.standard_normal(len(mean)))
        coefficients[i, columns] = draw
        u[:, i] = target[:, i] - xi @ draw
    return coefficients


def _covariance(u: np.ndarray, scale: np.ndarray, rng) -> np.ndarray:
    """One draw of a regime's covariance: conjugate inverse-Wishart on its months."""
    scatter = u.T @ u + np.diag(scale**2)
    draw = invwishart.rvs(
        len(u) + len(scale) + COVARIANCE_DF, (scatter + scatter.T) / 2, random_state=rng
    )
    return np.atleast_2d(draw)


def _transition(states: np.ndarray, spec: Spec, rng) -> np.ndarray:
    """One row-stochastic draw: a sticky Dirichlet prior plus the counts of the path."""
    m = int(states.max()) + 1
    counts = np.zeros((m, m))
    for i, j in zip(states[:-1], states[1:], strict=True):
        counts[i, j] += 1
    prior = np.full((m, m), spec.transition_switch)
    np.fill_diagonal(prior, spec.transition_stay)
    return np.array([rng.dirichlet(counts[i] + prior[i]) for i in range(m)])


def impute(
    y: np.ndarray,
    missing: np.ndarray,
    states: np.ndarray,
    coefficients: list[np.ndarray],
    a: list[np.ndarray],
    logvar: list[np.ndarray],
    p: int,
    rng,
) -> np.ndarray:
    """Draws the unknown months of ``y`` given the regime path and the per-regime parameters.

    Given the parameters and the path the whole history is Gaussian with a banded
    precision: the regime of each month picks its lag matrices and its error precision, so
    the conditional normal of the unknown months comes from one sparse solve. Values before
    the first equation carry a flat prior.
    """
    if not missing.any():
        return y
    count, n = y.shape[0] - p, y.shape[1]
    m = len(coefficients)
    lags = np.stack([split(coefficients[j], p)[1] for j in range(m)])  # (m, p, n, n)
    const = np.stack([split(coefficients[j], p)[0] for j in range(m)])  # (m, n)
    month_lags = lags[states]  # (count, p, n, n)
    intercept = const[states]  # (count, n)
    root = np.stack(a)[states]  # (count, n, n) A_{s_t}
    variance = np.exp(np.stack(logvar)[states])  # (count, n) exp(h_{s_t})
    precisions = np.einsum("tji,tj,tjk->tik", root, 1 / variance, root)
    # K: every equation month's y_t - sum A_{l,s_t} y_{t-l}, as one sparse matrix.
    blocks = np.concatenate(
        [np.broadcast_to(np.eye(n)[None, None], (count, 1, n, n)), -month_lags], axis=1
    )
    rows = np.repeat(np.arange(count) * n, (p + 1) * n * n).reshape(count, p + 1, n, n)
    rows = rows + np.arange(n)[None, None, :, None]
    columns = np.arange(count)[:, None, None, None] + p - np.arange(p + 1)[None, :, None, None]
    columns = (columns * n + np.arange(n)[None, None, None, :]).repeat(n, axis=2)
    equation = sparse.coo_matrix(
        (blocks.ravel(), (rows.ravel(), columns.ravel())),
        shape=(count * n, len(y) * n),
    ).tocsr()
    weight = sparse.block_diag(list(precisions), format="csr")
    precision = (equation.T @ weight @ equation).tocsr()
    linear = equation.T @ (weight @ intercept.ravel())
    flat_missing = np.flatnonzero(missing.ravel())
    known = np.nan_to_num(y.ravel())
    known[flat_missing] = 0.0
    of_missing = precision[flat_missing][:, flat_missing].toarray()
    rhs = linear[flat_missing] - precision[flat_missing] @ known
    root_mm = np.linalg.cholesky(of_missing)
    mean = np.linalg.solve(root_mm.T, np.linalg.solve(root_mm, rhs))
    draw = mean + np.linalg.solve(root_mm.T, rng.standard_normal(len(mean)))
    completed = known.copy()
    completed[flat_missing] = draw
    return completed.reshape(y.shape)


def _initial(panel: Panel, prior: Prior, spec: Spec, n_regimes: int):
    """The starting split: one VAR over the whole panel, its months ordered by residual size.

    The quietest months go to the first regime, so the starting labels already agree with the
    volatility order the sampler enforces.
    """
    p, n = spec.lags, len(panel.ids)
    filled = fill(panel.y)
    x, target = regressors(filled, p), filled[p:]
    coefficients = np.zeros((n, x.shape[1]))
    for i in range(n):
        columns = prior.mask[i]
        coefficients[i, columns] = np.linalg.lstsq(x[:, columns], target[:, i], rcond=None)[0]
    u = target - x @ coefficients.T
    order = np.argsort(np.linalg.norm(u, axis=1))
    states = np.empty(len(target), dtype=int)
    sigmas = []
    for j, chosen in enumerate(np.array_split(order, n_regimes)):
        states[chosen] = j
        scatter = u[chosen].T @ u[chosen] if len(chosen) else np.zeros((n, n))
        sigmas.append(np.atleast_2d(scatter) + np.diag(prior.scale**2) * 1e-4)
    transition = np.full((n_regimes, n_regimes), 0.1)
    np.fill_diagonal(transition, 0.9)
    return [coefficients.copy() for _ in range(n_regimes)], sigmas, transition, states


def estimate(panel: Panel, spec: Spec, n_regimes: int) -> Posterior:
    """The posterior sample of the MS-VAR under ``spec``; seeded, so it repeats exactly."""
    if [panel.months[0].year, panel.months[0].month] != [spec.start.year, spec.start.month]:
        raise ValueError(f"the panel starts in {panel.months[0]}, the spec in {spec.start}")
    if not 0 < panel.n_global < len(panel.ids):
        raise ValueError("the model needs both a global and a Russian block")
    if n_regimes < 2:
        raise ValueError("a switching model needs at least two regimes")
    prior = minnesota(panel, spec)
    rng = np.random.default_rng(spec.sampler.seed)
    p, n, m = spec.lags, len(panel.ids), n_regimes
    missing = np.isnan(panel.y)
    coefficients, sigmas, transition, states = _initial(panel, prior, spec, n_regimes)
    identity = np.eye(n)
    saved = []
    for keep in _keep(spec):
        factors = [unit_lower(sigma) for sigma in sigmas]
        a = [solve_triangular(c, identity, lower=True, unit_diagonal=True) for c, _ in factors]
        logvar = [h for _, h in factors]
        y = impute(panel.y, missing, states, coefficients, a, logvar, p, rng)
        x, target = regressors(y, p), y[p:]
        for j in range(m):
            rows = states == j
            variance = np.tile(np.exp(logvar[j]), (int(rows.sum()), 1))
            coefficients[j] = _coefficients(
                target[rows], x[rows], coefficients[j], a[j], variance, prior, rng
            )
        for j in range(m):
            rows = states == j
            u = target[rows] - x[rows] @ coefficients[j].T
            sigmas[j] = _covariance(u, prior.scale, rng)
        # Label switching: the declaration order is the volatility order.
        order = np.argsort([np.trace(sigma) for sigma in sigmas])
        if list(order) != list(range(m)):
            coefficients = [coefficients[k] for k in order]
            sigmas = [sigmas[k] for k in order]
            relabelled = np.empty(m, dtype=int)
            relabelled[order] = np.arange(m)
            states = relabelled[states]
            transition = transition[np.ix_(order, order)]
        factors = [unit_lower(sigma) for sigma in sigmas]
        transition = _transition(states, spec, rng)
        states = ffbs(
            target, x, coefficients, sigmas, transition, stationary(transition), rng
        )
        if keep:
            saved.append(
                {
                    "coefficients": np.stack(coefficients),
                    "chol": np.stack([c for c, _ in factors]),
                    "logvar": np.stack([h for _, h in factors]),
                    "transition": transition,
                    "states": states,
                    "history": y[-p:],
                }
            )
    share = np.mean(
        [[(draw["states"] == j).mean() for j in range(m)] for draw in saved], axis=0
    )
    diagnostics = {"missing": int(missing.sum()), "regime_share": list(share)}
    return _posterior(panel, spec, saved, diagnostics)


def _posterior(panel: Panel, spec: Spec, saved: list[dict], diagnostics: dict) -> Posterior:
    stacked = {name: np.array([draw[name] for draw in saved]) for name in saved[0]}
    p, n = spec.lags, len(panel.ids)
    coefficients = stacked["coefficients"]  # (S, m, n, 1 + n p)
    count, regimes = coefficients.shape[0], coefficients.shape[1]
    lags = (
        coefficients[:, :, :, 1 : 1 + n * p]
        .reshape(count, regimes, n, p, n)
        .transpose(0, 1, 3, 2, 4)
    )
    return Posterior(
        ids=panel.ids,
        n_global=panel.n_global,
        months=panel.months,
        lags_count=p,
        n_regimes=regimes,
        const=coefficients[:, :, :, 0],
        lags=lags,
        chol=stacked["chol"],
        logvar=stacked["logvar"],
        transition=stacked["transition"],
        states=stacked["states"],
        history=stacked["history"],
        diagnostics=diagnostics,
    )
