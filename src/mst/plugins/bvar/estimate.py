"""Bayesian estimation of the macro model of section 4.1.

Two samplers give the same ``Posterior``, so the constant-volatility model is the special case
``s_j = 0``, ``nu_j = infinity`` of the stochastic one:

``constant``: the BVAR with the conjugate Minnesota prior (normal-inverse-Wishart) and block
exogeneity. Block exogeneity breaks conjugacy of the whole system, so the likelihood is split
as in Zha (1999, Journal of Econometrics 90(2)): the global block regresses on global lags
only; the Russian block regresses on all lags, the dummies and the current global variables.
Each block is a conjugate regression and is drawn without MCMC; the reduced form is recovered
from the two.

``stochastic``: the triangular algorithm of Carriero, Clark and Marcellino (2019, Journal of
Econometrics 212(1)) in its corrected form (Carriero, Chan, Clark and Marcellino, 2022,
Journal of Econometrics 227(2)): ``A (y_t - Pi x_t) = e_t`` with ``A`` unit lower triangular
and each ``e_{j,t}`` carrying its own stochastic volatility and Student tails (module ``sv``).
Coefficients are drawn equation by equation, so each equation keeps its own regressors.

Months the panel does not know are drawn too (data augmentation): given the parameters the
whole history is Gaussian with a banded precision, and the missing values come from their
conditional normal. Values before the first equation carry a flat prior. With a complete panel
and constant volatility the draws are independent.

The Minnesota prior pulls a variable's own first lag to 1 (to 0 if the spec calls it
stationary) and every other lag coefficient to 0, with standard deviation ``lambda sigma_i /
(l sigma_k)`` for lag ``l`` of variable ``k`` in equation ``i``; ``sigma_k`` is the scale of
the monthly changes of ``k``. The constant and the dummies are left practically free.
"""

from dataclasses import dataclass, field
from datetime import date

import numpy as np
from scipy import sparse
from scipy.linalg import solve_triangular
from scipy.stats import invwishart

from mst.macro.panel import Panel

from . import sv
from .spec import Dummy, Spec
from .var import allowed, regressors, split

LOOSE = 1e6
"""Prior variance factor of the constant, the dummies (times sigma_i^2)."""
CONTEMPORANEOUS = 1.0
"""Prior variance factor of a current-global coefficient, times (sigma_i / sigma_k)^2."""
LOGVAR_PRIOR_VAR = 10.0
"""Prior variance of the first log variance around the log of the initial error variance."""


@dataclass(frozen=True)
class Posterior:
    """``S`` draws of theta with the last ``p`` months and the log variances of history."""

    ids: list[str]
    n_global: int
    months: list[date]
    lags_count: int
    dummies: list[Dummy]
    const: np.ndarray  # (S, n)
    lags: np.ndarray  # (S, p, n, n)
    dummy: np.ndarray  # (S, n, d)
    chol: np.ndarray  # (S, n, n) unit lower triangular C: u_t = C e_t
    logvar: np.ndarray  # (S, T - p, n) h_{j,t} of e_{j,t}, months p .. T-1
    volvol: np.ndarray  # (S, n) s_j; 0 for constant volatility
    dof: np.ndarray  # (S, n) nu_j; inf for normal errors
    history: np.ndarray  # (S, p, n) the last p months, drawn where unknown
    diagnostics: dict = field(default_factory=dict)
    rotation: np.ndarray | None = None  # (S, n, n) Q of mst.macro.identify; B = C diag Q
    shocks: list[str] = field(default_factory=list)  # the shock of each column of B

    @property
    def draws(self) -> int:
        return len(self.const)


@dataclass(frozen=True)
class Prior:
    mean: np.ndarray  # (n, k)
    var: np.ndarray  # (n, k), equation i scaled by sigma_i^2 (non-conjugate form)
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
    n, p, d = len(panel.ids), spec.lags, len(spec.dummies)
    unknown = set(spec.stationary) - set(panel.ids)
    if unknown:
        raise ValueError(f"stationary names unknown variables: {', '.join(sorted(unknown))}")
    sigma = scales(panel.y)
    mean = np.zeros((n, 1 + n * p + d))
    var = np.full(mean.shape, LOOSE)
    for i, id_ in enumerate(panel.ids):
        mean[i, 1 + i] = 0.0 if id_ in spec.stationary else 1.0
    for lag in range(1, p + 1):
        block = slice(1 + (lag - 1) * n, 1 + lag * n)
        var[:, block] = (spec.tightness / (lag * sigma[None, :])) ** 2
    return Prior(mean, var * sigma[:, None] ** 2, sigma, allowed(n, panel.n_global, p, d))


def dummy_matrix(months: list[date], dummies: list[Dummy]) -> np.ndarray:
    return np.array([[dummy.value(m) for dummy in dummies] for m in months]).reshape(
        len(months), len(dummies)
    )


def fill(y: np.ndarray) -> np.ndarray:
    """Linear interpolation of the unknown months, flat beyond the first and last value."""
    filled = y.copy()
    grid = np.arange(len(y))
    for k in range(y.shape[1]):
        known = ~np.isnan(y[:, k])
        filled[:, k] = np.interp(grid, grid[known], y[known, k])
    return filled


def niw(y, x, mean, prec, s0, nu0, rng):
    """One draw ``(B, Sigma)`` of ``y = x B + u``, ``u ~ N(0, Sigma)``, under the conjugate prior
    ``Sigma ~ IW(s0, nu0)``, ``vec B | Sigma ~ N(vec mean, Sigma (x) diag(1 / prec))``."""
    gram = x.T @ x + np.diag(prec)
    root = np.linalg.cholesky(gram)
    post = np.linalg.solve(gram, prec[:, None] * mean + x.T @ y)
    scale = s0 + y.T @ y + mean.T @ (prec[:, None] * mean) - post.T @ gram @ post
    sigma = np.atleast_2d(invwishart.rvs(nu0 + len(y), (scale + scale.T) / 2, random_state=rng))
    noise = np.linalg.solve(root.T, rng.standard_normal(post.shape))
    return post + noise @ np.linalg.cholesky(sigma).T, sigma


def conjugate_step(y: np.ndarray, dummies: np.ndarray, prior: Prior, p: int, n_global: int, rng):
    """Coefficients ``(n, k)`` and ``Sigma`` from the block-recursive conjugate posterior."""
    n, g = y.shape[1], n_global
    x = regressors(y, dummies, p)
    target = y[p:]
    sigma2 = prior.scale**2
    # The conjugate prior variance is the same in every equation up to sigma_i^2.
    relative = prior.var / sigma2[:, None]
    coefficients = np.zeros((n, x.shape[1]))
    columns = prior.mask[0]
    b_g, s_gg = niw(
        target[:, :g],
        x[:, columns],
        prior.mean[:g, columns].T,
        1 / relative[0, columns],
        np.diag(sigma2[:g]),
        g + 2,
        rng,
    )
    coefficients[:g, columns] = b_g.T
    b_r, s_v = niw(
        target[:, g:],
        np.hstack([x, target[:, :g]]),
        np.vstack([prior.mean[g:].T, np.zeros((g, n - g))]),
        np.concatenate([1 / relative[g], sigma2[:g] / CONTEMPORANEOUS]),
        np.diag(sigma2[g:]),
        n - g + 2,
        rng,
    )
    gamma = b_r[x.shape[1] :].T
    coefficients[g:] = b_r[: x.shape[1]].T + gamma @ coefficients[:g]
    sigma = np.block([[s_gg, s_gg @ gamma.T], [gamma @ s_gg, gamma @ s_gg @ gamma.T + s_v]])
    return coefficients, sigma


def impute(y, missing, coefficients, precisions, dummies, p, rng):
    """Draws the unknown months of ``y`` given the coefficients and each equation's error
    precision ``precisions`` (T - p, n, n). Returns a completed copy."""
    if not missing.any():
        return y
    count, n = y.shape
    const, lags, dummy = split(coefficients, p)
    intercept = const + dummies[p:] @ dummy.T
    blocks = np.concatenate([np.eye(n)[None], -lags])
    t = np.arange(p, count)[:, None, None, None]
    lag = np.arange(p + 1)[None, :, None, None]
    r, c = np.arange(n)[:, None], np.arange(n)[None, :]
    shape = (count - p, p + 1, n, n)
    k = sparse.csr_matrix(
        (
            np.broadcast_to(blocks, shape).ravel(),
            (
                np.broadcast_to((t - p) * n + r, shape).ravel(),
                np.broadcast_to((t - lag) * n + c, shape).ravel(),
            ),
        ),
        shape=((count - p) * n, count * n),
    )
    s = sparse.block_diag(precisions, format="csr")
    precision = (k.T @ s @ k).tocsr()
    linear = k.T @ (s @ intercept.ravel())
    flat_missing = np.flatnonzero(missing.ravel())
    known = np.nan_to_num(y.ravel())
    known[flat_missing] = 0.0
    p_mm = precision[flat_missing][:, flat_missing].toarray()
    rhs = linear[flat_missing] - precision[flat_missing] @ known
    root = np.linalg.cholesky(p_mm)
    mean = np.linalg.solve(root.T, np.linalg.solve(root, rhs))
    draw = mean + np.linalg.solve(root.T, rng.standard_normal(len(mean)))
    completed = known.copy()
    completed[flat_missing] = draw
    return completed.reshape(count, n)


def unit_lower(sigma: np.ndarray):
    """``(C, log diag)`` with ``Sigma = C diag(exp(h)) C'`` and ``C`` unit lower triangular."""
    root = np.linalg.cholesky(sigma)
    diagonal = np.diag(root)
    return root / diagonal[None, :], 2 * np.log(diagonal)


def estimate(panel: Panel, spec: Spec) -> Posterior:
    """The posterior sample of section 4.1 under ``spec``; seeded, so it repeats exactly."""
    if [panel.months[0].year, panel.months[0].month] != [spec.start.year, spec.start.month]:
        raise ValueError(f"the panel starts in {panel.months[0]}, the spec in {spec.start}")
    if not 0 < panel.n_global < len(panel.ids):
        raise ValueError("the model needs both a global and a Russian block")
    prior = minnesota(panel, spec)
    dummies = dummy_matrix(panel.months, spec.dummies)
    rng = np.random.default_rng(spec.sampler.seed)
    sampler = _constant if spec.volatility == "constant" else _stochastic
    return sampler(panel, spec, prior, dummies, rng)


def _keep(spec: Spec):
    total = spec.sampler.burn + spec.sampler.draws * spec.sampler.thin
    for step in range(total):
        keep = step >= spec.sampler.burn and (step - spec.sampler.burn) % spec.sampler.thin == 0
        yield keep


def _posterior(panel, spec, saved, diagnostics) -> Posterior:
    stacked = {name: np.array([draw[name] for draw in saved]) for name in saved[0]}
    const, lags, dummy = zip(*(split(c, spec.lags) for c in stacked["coefficients"]), strict=True)
    return Posterior(
        ids=panel.ids,
        n_global=panel.n_global,
        months=panel.months,
        lags_count=spec.lags,
        dummies=spec.dummies,
        const=np.array(const),
        lags=np.array(lags),
        dummy=np.array(dummy),
        chol=stacked["chol"],
        logvar=stacked["logvar"],
        volvol=stacked["volvol"],
        dof=stacked["dof"],
        history=stacked["history"],
        diagnostics=diagnostics,
    )


def _constant(panel, spec, prior, dummies, rng) -> Posterior:
    p, n = spec.lags, len(panel.ids)
    missing = np.isnan(panel.y)
    y = fill(panel.y)
    equations = len(y) - p
    saved = []
    for keep in _keep(spec):
        coefficients, sigma = conjugate_step(y, dummies, prior, p, panel.n_global, rng)
        y = impute(
            panel.y, missing, coefficients, [np.linalg.inv(sigma)] * equations, dummies, p, rng
        )
        if keep:
            chol, logvar = unit_lower(sigma)
            saved.append(
                {
                    "coefficients": coefficients,
                    "chol": chol,
                    "logvar": np.tile(logvar, (equations, 1)),
                    "volvol": np.zeros(n),
                    "dof": np.full(n, np.inf),
                    "history": y[-p:],
                }
            )
    return _posterior(panel, spec, saved, {"missing": int(missing.sum())})


def _stochastic(panel, spec, prior, dummies, rng) -> Posterior:
    p, n = spec.lags, len(panel.ids)
    missing = np.isnan(panel.y)
    y = fill(panel.y)
    coefficients, sigma = conjugate_step(y, dummies, prior, p, panel.n_global, rng)
    chol, h_start = unit_lower(sigma)
    a = np.linalg.inv(chol)
    equations = len(y) - p
    h = np.tile(h_start, (equations, 1))
    omega = np.ones((equations, n))
    s2 = np.full(n, spec.volvol_mean)
    nu = np.full(n, sv.DOF_GRID[-1])
    shape = spec.volvol_df / 2
    volvol_prior = (shape, spec.volvol_mean * (shape - 1))
    a_var = (prior.scale[:, None] / prior.scale[None, :]) ** 2
    saved = []
    for keep in _keep(spec):
        variance = np.exp(h) * omega
        precisions = np.einsum("ji,tj,jk->tik", a, 1 / variance, a)
        y = impute(panel.y, missing, coefficients, precisions, dummies, p, rng)
        x, target = regressors(y, dummies, p), y[p:]
        coefficients = _coefficients(target, x, coefficients, a, variance, prior, rng)
        u = target - x @ coefficients.T
        a = _triangle(u, variance, a_var, rng)
        e = u @ a.T
        for j in range(n):
            omega[:, j] = sv.draw_mixing(e[:, j] ** 2 / np.exp(h[:, j]), nu[j], rng)
            nu[j] = sv.draw_dof(omega[:, j], rng)
            log_e2 = np.log(e[:, j] ** 2 / omega[:, j] + sv.OFFSET)
            h[:, j] = sv.draw_logvar(log_e2, h[:, j], s2[j], (h_start[j], LOGVAR_PRIOR_VAR), rng)
            s2[j] = sv.draw_volvol(h[:, j], volvol_prior, rng)
        if keep:
            saved.append(
                {
                    "coefficients": coefficients.copy(),
                    # A triangular solve keeps the zeros above the diagonal exact.
                    "chol": solve_triangular(a, np.eye(n), lower=True, unit_diagonal=True),
                    "logvar": h.copy(),
                    "volvol": np.sqrt(s2),
                    "dof": nu.copy(),
                    "history": y[-p:],
                }
            )
    return _posterior(panel, spec, saved, {"missing": int(missing.sum())})


def _coefficients(target, x, coefficients, a, variance, prior, rng):
    """Draws each equation's coefficients given the others (corrected triangular algorithm).

    Equation ``i`` enters the orthogonalised errors ``e_j = sum_k a_jk u_k`` of ``j >= i``:
    ``e_j + a_ji x pi_i = a_ji x pi_i + e_j`` is a regression on ``x`` with weights
    ``a_ji^2 / variance_j``.
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


def _triangle(u, variance, prior_var, rng):
    """Draws the unit lower triangular ``A`` row by row: ``u_j = -sum_{k<j} a_jk u_k + e_j``."""
    n = u.shape[1]
    a = np.eye(n)
    for j in range(1, n):
        regressors_ = -u[:, :j]
        weight = 1 / variance[:, j]
        prec = np.diag(1 / prior_var[j, :j]) + regressors_.T @ (regressors_ * weight[:, None])
        root = np.linalg.cholesky(prec)
        mean = np.linalg.solve(root.T, np.linalg.solve(root, regressors_.T @ (u[:, j] * weight)))
        a[j, :j] = mean + np.linalg.solve(root.T, rng.standard_normal(j))
    return a
