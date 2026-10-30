"""Estimation of the satellite models (docs/concept/scenario-formalization.md, section 6.2).

Every driver ``k`` follows ``d_t = gamma_k' z_t + e_t`` with ``z_t = (1, d_{t-1}, x_t, ...,
x_{t-L})``. The parameters of an object are drawn around those of the object above it, ``gamma_k
= gamma_up + eta_k``, ``eta_k ~ N(0, Psi_c)``, where ``c`` is the class of the object; an object
with nothing above it is drawn around ``gamma_bar_c``, the mean of its class, which has a flat
prior. An object without a series of its own is a node of the same tree: its parameters come
from the level above and from the objects below it.

The Gibbs sampler draws, in turn, each node given its parent, its children and its own data,
the class means, the diagonal ``Psi_c`` from inverse gamma priors and the residual variance of
each driver with data. It works in a scale where every column of ``z`` but the constant is
centred and divided by its standard deviation within objects, so one prior spread fits all
coefficients (``spec.toml``), and returns the draws in the original scale.

Months a driver is not observed in are drawn inside the sampler (``Path``, ``draw_path``): a
stored value fixes the mean of the driver over the ``span`` months it covers (one for a monthly
value, three for a quarter's average), so a quarterly price or a bond that did not trade for a
month keeps every month of its history. ``x`` is not drawn: a node's months start after the last
month whose ``x`` is unknown and end at the last month whose ``x`` is known, so a price published
before its regressors is still estimated up to the last complete month. The chain is stationary:
``rho`` is drawn below one in absolute value, and the first month starts from the stationary
distribution, which keeps the tridiagonal precision well conditioned however long the path is.
``Psi_c`` is diagonal, a special case of the full covariance of section 6.2. The residual
correlation ``Sigma_e`` across drivers is estimated after the sampler, from the residuals at the
posterior mean over the months two drivers both observe.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_solve_banded, cholesky_banded, solve_banded
from scipy.special import ndtr, ndtri

from mst.drivers.history import History

from .spec import Spec

SHAPE = 3.0
"""The shape of the inverse gamma prior of each diagonal element of ``Psi_c``."""
COMMON = 12
"""Months two drivers must share for their residual correlation to be estimated."""


def lagged(x: np.ndarray, lags: int, t: int) -> np.ndarray:
    """The rows ``t - lags .. t`` of ``x``, nearest first: the ``x`` part of ``z_t``."""
    return np.concatenate([x[t - lag] for lag in range(lags + 1)])


def regressors(lags: int, x: np.ndarray, previous: float, t: int) -> np.ndarray:
    """``z_t`` from ``d_{t-1}`` and the rows ``t - lags .. t`` of ``x``."""
    return np.concatenate([[1.0, previous], lagged(x, lags, t)])


@dataclass
class Path:
    """The months a node is estimated on, its observations and its path over those months.

    The path runs from the first month the first observation covers to the last observation,
    after the last month whose ``x`` is unknown. An observation fixes the mean of the path over
    the ``span`` months it covers; months no observation fixes are drawn by the sampler.
    """

    first: int
    observed: np.ndarray
    """(o,) month indices of the observations."""
    values: np.ndarray
    span: int
    xpart: np.ndarray
    """(n, m (L + 1)) the ``x`` part of ``z_t`` for months ``first .. last``."""
    path: np.ndarray
    """(n,) the driver over months ``first .. last``; observed or drawn."""

    @property
    def complete(self) -> bool:
        return self.span == 1 and len(self.observed) == len(self.path)

    def rows(self) -> tuple[np.ndarray, np.ndarray]:
        """``(Z, y)`` of months ``first + 1 .. last`` on the current path."""
        n = len(self.path)
        z = np.column_stack([np.ones(n - 1), self.path[:-1], self.xpart[1:]])
        return z, self.path[1:]

    def constraints(self) -> np.ndarray:
        """(o, n) weights of the observations on the path."""
        weights = np.zeros((len(self.observed), len(self.path)))
        for k, month in enumerate(self.observed - self.first):
            weights[k, month - self.span + 1 : month + 1] = 1 / self.span
        return weights


def path_of(hist: History, node: int, until: int | None = None) -> Path | None:
    """The estimation months of a node up to ``until``; None without two usable observations.

    The window is the months whose ``x`` the panel knows: it starts after the last month with an
    unknown ``x`` and ends at the last one, so an observation that falls outside it is dropped.
    """
    span, lags, d = hist.quantity.span, hist.quantity.lags, hist.d[:, node]
    last = len(d) - 1 if until is None else until
    observed = np.flatnonzero(np.isfinite(d[: last + 1]))
    if len(observed) < 2:
        return None
    known = np.array(
        [t >= lags and bool(np.all(np.isfinite(hist.x[t - lags : t + 1]))) for t in range(last + 1)]
    )
    usable = np.flatnonzero(known)
    if not len(usable):
        return None
    end = int(usable[-1])
    observed = observed[observed <= end]
    if len(observed) < 2:
        return None
    first = max(int(observed[0]) - span + 1, lags)
    unknown = np.flatnonzero(~known[first : end + 1]) + first
    if len(unknown):
        first = int(unknown[-1]) + 1
    if first > end:
        return None
    observed = observed[observed - span + 1 >= first]
    if len(observed) < 2:
        return None
    months = np.arange(first, end + 1)
    return Path(
        first=first,
        observed=observed,
        values=d[observed],
        span=span,
        xpart=np.array([lagged(hist.x, lags, t) for t in months]),
        path=np.interp(months, observed, d[observed]),
    )


def draw_path(item: Path, gamma, centre, spread, sigma2: float, rng) -> np.ndarray:
    """The path given scaled coefficients and the residual variance, observations held exactly.

    Given the parameters ``d_t = a_t + rho d_{t-1} + e_t`` is a Gaussian chain with a
    tridiagonal precision, the first month taken from its stationary distribution. A draw of the
    chain is moved to the observations by ``z + Q^-1 W' (W Q^-1 W')^-1 (v - W z)``.
    """
    n = len(item.path)
    rho = gamma[1] / spread[1]
    intercept = (
        gamma[0]
        - gamma[1] * centre[1] / spread[1]
        + ((item.xpart - centre[2:]) / spread[2:]) @ gamma[2:]
    )
    a = intercept[1:]
    # Upper banded form of the tridiagonal precision: row 1 the diagonal, row 0 above it.
    banded = np.zeros((2, n))
    banded[1, 1:] += 1 / sigma2
    banded[1, :-1] += rho**2 / sigma2
    banded[1, 0] += (1 - rho**2) / sigma2
    banded[0, 1:] = -rho / sigma2
    shift = np.zeros(n)
    shift[1:] += a / sigma2
    shift[:-1] -= rho * a / sigma2
    shift[0] += (intercept[0] / (1 - rho)) * (1 - rho**2) / sigma2
    upper = cholesky_banded(banded)
    draw = cho_solve_banded((upper, False), shift) + solve_banded(
        (0, 1), upper, rng.standard_normal(n)
    )
    weights = item.constraints()
    gain = cho_solve_banded((upper, False), weights.T)
    return draw + gain @ np.linalg.solve(weights @ gain, item.values - weights @ draw)


@dataclass(frozen=True)
class Fit:
    history: History
    gamma: np.ndarray
    """(S, N, k) coefficients of every node in the original scale."""
    sigma: np.ndarray
    """(S, N) residual standard deviation; NaN at nodes without data."""
    rows: np.ndarray
    """(N,) observations each node was estimated on; 0 at nodes without data."""


def columns(quantity) -> list[str]:
    """The names of the elements of ``z_t``: constant, own lag, ``x_{t-l}`` for each lag."""
    return ["const", "own_lag"] + [
        f"{name}@{lag}" for lag in range(quantity.lags + 1) for name in quantity.inputs
    ]


def _scales(paths: dict[int, Path], names: list[str]):
    """Pooled centres and within-object spreads of the columns of ``z`` and of ``d``."""
    rows = {i: item.rows() for i, item in paths.items()}
    stacked = np.vstack([z for z, _ in rows.values()])
    centre = stacked.mean(axis=0)
    centre[0] = 0.0
    within = [z - z.mean(axis=0) for z, _ in rows.values()]
    spread = np.sqrt(np.vstack(within).var(axis=0))
    spread[0] = 1.0
    flat = [names[j] for j in range(1, len(names)) if not spread[j] > 0]
    if flat:
        raise ValueError(f"columns {', '.join(flat)} do not vary within objects")
    level = np.concatenate([y - y.mean() for _, y in rows.values()])
    return centre, spread, float(level.std())


def fit(hist: History, spec: Spec, *, until: int | None = None, seed: int | None = None) -> Fit:
    """Posterior draws of the satellite parameters of one quantity, months ``<= until``."""
    columns_ = columns(hist.quantity)
    k, n = len(columns_), len(hist.nodes)
    paths = {}
    for i in range(n):
        item = path_of(hist, i, until)
        if item is not None and len(item.path) - 1 > k:
            paths[i] = item
    if not paths:
        raise ValueError(f"no driver of {hist.quantity.id} has more than {k} usable months")
    centre, spread, level = _scales(paths, columns_)
    prior = np.full(k, (spec.spread * level) ** 2)
    prior[0] = (spec.constant_spread * level) ** 2
    b0 = (SHAPE - 1) * prior

    classes = sorted(set(hist.classes))
    cls = np.array([classes.index(c) for c in hist.classes])
    children = [np.flatnonzero(hist.parent == i) for i in range(n)]
    roots = np.flatnonzero(hist.parent < 0)

    def scaled(i):
        z, y = paths[i].rows()
        return (z - centre) / spread, y

    rng = np.random.default_rng(spec.sampler.seed if seed is None else seed)
    gamma = np.zeros((n, k))
    sigma2 = np.full(n, np.nan)
    for i in paths:
        z, y = scaled(i)
        gamma[i] = np.linalg.solve(z.T @ z + 1e-6 * np.eye(k), z.T @ y)
        sigma2[i] = max(np.mean((y - z @ gamma[i]) ** 2), 1e-12)
    pooled = np.mean([gamma[i] for i in paths], axis=0)
    for i in range(n):
        if i not in paths:
            gamma[i] = pooled
    psi = np.tile(prior, (len(classes), 1))
    top = np.tile(pooled, (len(classes), 1))

    kept_gamma = np.empty((spec.sampler.draws, n, k))
    kept_sigma = np.full((spec.sampler.draws, n), np.nan)
    for step in range(spec.sampler.burn + spec.sampler.draws):
        data = {i: scaled(i) for i in paths}
        for i in range(n):
            above = top[cls[i]] if hist.parent[i] < 0 else gamma[hist.parent[i]]
            precision = np.diag(1 / psi[cls[i]])
            shift = above / psi[cls[i]]
            for c in children[i]:
                precision += np.diag(1 / psi[cls[c]])
                shift += gamma[c] / psi[cls[c]]
            if i in data:
                z, y = data[i]
                precision += z.T @ z / sigma2[i]
                shift += z.T @ y / sigma2[i]
            gamma[i] = _stationary(
                _normal(precision, shift, rng), precision, shift, spread, spec.rho_max, rng
            )
        for c in range(len(classes)):
            members = roots[cls[roots] == c]
            if len(members):
                top[c] = gamma[members].mean(axis=0) + np.sqrt(
                    psi[c] / len(members)
                ) * rng.standard_normal(k)
            nodes = np.flatnonzero(cls == c)
            above = np.array(
                [top[c] if hist.parent[i] < 0 else gamma[hist.parent[i]] for i in nodes]
            )
            gap = ((gamma[nodes] - above) ** 2).sum(axis=0)
            psi[c] = (b0 + gap / 2) / rng.gamma(SHAPE + len(nodes) / 2, size=k)
        for i, (z, y) in data.items():
            residual = y - z @ gamma[i]
            sigma2[i] = (residual @ residual / 2) / rng.gamma(len(y) / 2)
        for i, item in paths.items():
            if not item.complete:
                item.path = draw_path(item, gamma[i], centre, spread, sigma2[i], rng)
        if step >= spec.sampler.burn:
            kept_gamma[step - spec.sampler.burn] = gamma
            kept_sigma[step - spec.sampler.burn] = np.sqrt(sigma2)

    original = kept_gamma / spread
    original[..., 0] = kept_gamma[..., 0] - (kept_gamma[..., 1:] * centre[1:] / spread[1:]).sum(-1)
    rows = np.array([len(paths[i].observed) if i in paths else 0 for i in range(n)])
    return Fit(history=hist, gamma=original, sigma=kept_sigma, rows=rows)


def _normal(precision: np.ndarray, shift: np.ndarray, rng) -> np.ndarray:
    """A draw of ``N(precision^-1 shift, precision^-1)``."""
    chol = np.linalg.cholesky(precision)
    mean = np.linalg.solve(chol.T, np.linalg.solve(chol, shift))
    return mean + np.linalg.solve(chol.T, rng.standard_normal(len(shift)))


def _stationary(draw: np.ndarray, precision, shift, spread, rho_max: float, rng) -> np.ndarray:
    """The draw with its share of the previous month below ``rho_max`` in absolute value.

    The conditional of that coefficient given the others is normal, and truncating it does not
    depend on their values, so drawing it again from the truncated conditional gives a draw of
    the truncated joint. An explosive driver has no path to simulate, so the tail is cut.
    """
    bound = rho_max * spread[1]
    if abs(draw[1]) <= bound:
        return draw
    variance = 1 / precision[1, 1]
    mean = (shift[1] - precision[1] @ draw + precision[1, 1] * draw[1]) * variance
    scale = np.sqrt(variance)
    low, high = ndtr((-bound - mean) / scale), ndtr((bound - mean) / scale)
    share = low + rng.random() * (high - low)
    draw[1] = np.clip(mean + scale * ndtri(min(max(share, 1e-300), 1 - 1e-16)), -bound, bound)
    return draw


def residuals(fit_: Fit, node: int) -> np.ndarray:
    """(T,) residuals of a node at the posterior mean; NaN where a month was not used."""
    hist, gamma = fit_.history, fit_.gamma[:, node].mean(axis=0)
    d, lags = hist.d[:, node], hist.quantity.lags
    out = np.full(len(d), np.nan)
    for t in range(max(1, lags), len(d)):
        z = regressors(lags, hist.x, d[t - 1], t)
        if np.isfinite(d[t]) and np.all(np.isfinite(z)):
            out[t] = d[t] - gamma @ z
    return out
