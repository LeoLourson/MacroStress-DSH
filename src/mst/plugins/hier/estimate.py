"""Estimation of the satellite models (docs/concept/bvar-method-legacy.md, section 6.2).

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
posterior mean over the months two drivers both observe. The fit also records the lowest and
highest observed value of every node inside the window it was estimated on, so the platform
checks simulated paths against what the node was trained on rather than against its whole
history column.
"""

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.linalg import lapack
from scipy.special import ndtr, ndtri

from mst.drivers.history import History
from mst.model_sdk import ShortHistory
from mst.model_sdk.checkpoint import report

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


def difference(columns: np.ndarray) -> np.ndarray:
    """Monthly changes of a ``(T, ...)`` array of the panel; month 0 has none.

    A month missing in the levels gives a missing change for two months — the change into it
    and the change out of it — which the sampler draws like any other unknown month of the
    path. Used on ``d`` and on ``x`` alike, the linear form of the model is unchanged: the
    sampler sees ``Δd`` against ``Δx`` and returns the coefficients in that scale.
    """
    out = np.full(columns.shape, np.nan)
    out[1:] = columns[1:] - columns[:-1]
    return out


@dataclass
class Path:
    """The months a node is estimated on, its observations and its path over those months.

    The path runs from the first month the first observation covers to the last observation,
    after the last month whose ``x`` is unknown. An observation fixes a weighted sum of the
    path (``weights``); months no observation fixes are drawn by the sampler.
    """

    first: int
    observed: np.ndarray
    """(o,) month indices of the observations; of the later level for a change."""
    values: np.ndarray
    weights: np.ndarray
    """(o, n) the weights of the observations on the path."""
    xpart: np.ndarray
    """(n, m (L + 1)) the ``x`` part of ``z_t`` for months ``first .. last``."""
    path: np.ndarray
    """(n,) the driver over months ``first .. last``; observed or drawn."""
    earlier: np.ndarray | None = None
    """(o,) for a path of changes, the month of the earlier level of every observation."""
    complete: bool = field(init=False)
    """Every month is an observation of its own: the sampler has nothing to draw."""

    def __post_init__(self) -> None:
        n = len(self.path)
        self.complete = self.weights.shape == (n, n) and np.array_equal(self.weights, np.eye(n))

    def rows(self) -> tuple[np.ndarray, np.ndarray]:
        """``(Z, y)`` of months ``first + 1 .. last`` on the current path."""
        n = len(self.path)
        z = np.column_stack([np.ones(n - 1), self.path[:-1], self.xpart[1:]])
        return z, self.path[1:]


def _weights(first: int, n: int, span: int, observed, earlier) -> np.ndarray:
    """The weights of the observations on the path of months ``first .. first + n - 1``.

    A level stored at month ``t`` is the mean of the ``span`` months up to ``t``. On a path of
    levels it weighs those months by ``1 / span``. On a path of monthly changes the observation
    is the difference of two stored levels at ``t'`` < ``t``: it is the sum, over the ``span``
    months ``j`` of the mean, of the changes from ``t' - j + 1`` to ``t - j`` — the weights
    1/3, 2/3, 1, 2/3, 1/3 for two neighbouring quarters, one month of weight one for two
    neighbouring months.
    """
    out = np.zeros((len(observed), n))
    for k, t in enumerate(observed - first):
        if earlier is None:
            out[k, t - span + 1 : t + 1] = 1 / span
            continue
        for j in range(span):
            out[k, earlier[k] - first - j + 1 : t - j + 1] += 1 / span
    return out


def path_of(
    hist: History, node: int, until: int | None = None, *, differenced: bool = False
) -> Path | None:
    """The estimation months of a node up to ``until``; None without two usable observations.

    The window is the months whose ``x`` the panel knows: it starts after the last month with an
    unknown ``x`` and ends at the last one, so an observation that falls outside it is dropped.
    With ``differenced`` the path is of monthly changes and ``hist.x`` holds the changes of
    ``x``; every stored level after the first is observed as its change from the stored level
    before it, so a quarterly average or a gap of a month keeps its information.
    """
    span, lags, d = hist.quantity.span, hist.quantity.lags, hist.d[:, node]
    last = len(d) - 1 if until is None else until
    observed = np.flatnonzero(np.isfinite(d[: last + 1]))
    known = np.array(
        [t >= lags and bool(np.all(np.isfinite(hist.x[t - lags : t + 1]))) for t in range(last + 1)]
    )
    usable = np.flatnonzero(known)
    if not len(usable):
        return None
    end = int(usable[-1])
    observed = observed[observed <= end]
    # The first month an observation covers: for a level the first of its span, for a change
    # the month after the first of the span of the earlier level.
    reach = span - 1 - int(differenced)
    if len(observed) < 2 + int(differenced):
        return None
    first = max(int(observed[0]) - reach, lags)
    unknown = np.flatnonzero(~known[first : end + 1]) + first
    if len(unknown):
        first = int(unknown[-1]) + 1
    if first > end:
        return None
    observed = observed[observed - reach >= first]
    earlier = None
    if differenced:
        observed, earlier = observed[1:], observed[:-1]
    if len(observed) < 2:
        return None
    months = np.arange(first, end + 1)
    if differenced:
        ends = np.r_[earlier[:1], observed]
        start = np.diff(np.interp(np.arange(first - 1, end + 1), ends, d[ends]))
        values = d[observed] - d[earlier]
    else:
        values, start = d[observed], np.interp(months, observed, d[observed])
    return Path(
        first=first,
        observed=observed,
        values=values,
        weights=_weights(first, len(months), span, observed, earlier),
        xpart=np.array([lagged(hist.x, lags, t) for t in months]),
        path=start,
        earlier=earlier,
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
    # The sampler calls this every sweep for every incomplete path, so LAPACK is called
    # directly: ``P = U'U`` once, then the mean and the gain in one solve with ``P``, and the
    # noise ``U^-1 z`` whose covariance is ``P^-1``.
    upper, info = lapack.dpbtrf(banded, lower=0)
    if info:
        raise np.linalg.LinAlgError("the precision of a driver's path is not definite")
    weights = item.weights
    solved, _ = lapack.dpbtrs(upper, np.column_stack([shift, weights.T]), lower=0)
    noise, _ = lapack.dtbtrs(upper, rng.standard_normal(n), uplo="U")
    draw, gain = solved[:, 0] + noise, solved[:, 1:]
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
    train_low: np.ndarray | None = None
    """(N,) the lowest observed value of every node inside its estimation window; ``None`` only
    before ``fit`` fills it in.

    A node without a series of its own takes the bounds of the nearest object above it with
    data, as it takes the residual spread of that object when simulated; NaN when no object
    above it was estimated on data.
    """
    train_high: np.ndarray | None = None
    """(N,) the highest observed value of every node inside its estimation window."""
    differenced: bool = False
    """The equation was fitted on monthly changes: ``gamma`` reads ``Δd`` and ``Δx`` and the
    simulation accumulates the level from the last observed one. The bounds stay in levels,
    so the outside-range flag keeps its meaning."""
    sources: dict[int, int] = field(default_factory=dict)
    """The node whose coefficients serve a driver its own equation does not (the check)."""
    flat: tuple[str, ...] = ()
    """The columns of ``z`` that did not vary within any object in the window: the data say
    nothing of their effect, so their coefficients are zero."""


def columns(quantity) -> list[str]:
    """The names of the elements of ``z_t``: constant, own lag, ``x_{t-l}`` for each lag."""
    return ["const", "own_lag"] + [
        f"{name}@{lag}" for lag in range(quantity.lags + 1) for name in quantity.inputs
    ]


def _scales(paths: dict[int, Path], names: list[str]):
    """Pooled centres and within-object spreads of the columns of ``z`` and of ``d``; and the
    columns that do not vary within any object, whose spread is set to one.

    A column constant within every object — a feature of the key rate above 15 % in a window
    of months below it — cannot be told from the objects' constants, so its coefficient is not
    estimated: ``fit`` holds it at zero."""
    rows = {i: item.rows() for i, item in paths.items()}
    stacked = np.vstack([z for z, _ in rows.values()])
    centre = stacked.mean(axis=0)
    centre[0] = 0.0
    within = [z - z.mean(axis=0) for z, _ in rows.values()]
    spread = np.sqrt(np.vstack(within).var(axis=0))
    spread[0] = 1.0
    flat = ~(spread > 0)
    spread[flat] = 1.0
    level = np.concatenate([y - y.mean() for _, y in rows.values()])
    return centre, spread, flat, float(level.std())


def training_bounds(
    hist: History, rows: np.ndarray, paths: dict[int, Path]
) -> tuple[np.ndarray, np.ndarray]:
    """The lowest and highest observed value of every node inside its estimation window.

    The window of a node is the one ``path_of`` chose: months whose ``x`` the panel knows, from
    the first observation to the last complete month. Months the sampler drew are not data, so
    they do not widen the range a simulated path is checked against, and a value an observation
    falls back to (a quarter's average) is not taken either. A node without a series of its own
    has no window; it takes the bounds of the nearest object above it with data, exactly the
    source ``simulate._nearest`` uses for its starting level and residual spread.
    """
    low = np.full(len(rows), np.nan)
    high = np.full(len(rows), np.nan)
    for i, item in paths.items():
        if len(item.values):
            low[i], high[i] = float(item.values.min()), float(item.values.max())
    for i in range(len(rows)):
        if rows[i] > 0:
            continue
        above = hist.parent[i]
        while above >= 0 and rows[above] == 0:
            above = hist.parent[above]
        if above >= 0:
            low[i], high[i] = low[above], high[above]
    return low, high


def level_bounds(
    hist: History, rows: np.ndarray, paths: dict[int, Path]
) -> tuple[np.ndarray, np.ndarray]:
    """The training bounds of a quantity fitted in differences: the levels it was trained on.

    Every observation of a change joins two stored levels, so the levels the estimation saw
    are the two ends of every observed change — a level no change touches is not data the model
    was trained on. The bounds stay in levels, so the outside-range flag
    keeps the meaning it has for a quantity in levels; a node without a series of its own
    inherits them from the nearest object above it with data, as ``training_bounds`` gives it.
    """
    low = np.full(len(rows), np.nan)
    high = np.full(len(rows), np.nan)
    for i, item in paths.items():
        values = hist.d[np.concatenate([item.observed, item.earlier]), i]
        if len(values):
            low[i], high[i] = float(values.min()), float(values.max())
    for i in range(len(rows)):
        if rows[i] > 0:
            continue
        above = hist.parent[i]
        while above >= 0 and rows[above] == 0:
            above = hist.parent[above]
        if above >= 0:
            low[i], high[i] = low[above], high[above]
    return low, high


def fit(
    hist: History,
    spec: Spec,
    *,
    until: int | None = None,
    seed: int | None = None,
    progress: str | None = None,
) -> Fit:
    """Posterior draws of the satellite parameters of one quantity, months ``<= until``.

    With ``progress`` the sampler reports every tenth of its steps under that label.

    A quantity named in ``spec.differences`` is fitted on monthly changes: ``d`` and ``x`` are
    differenced before the sampler, so ``gamma`` reads ``Δd`` against ``Δx`` with the same
    machinery, while the recorded training bounds stay in levels.
    """
    columns_ = columns(hist.quantity)
    k, n = len(columns_), len(hist.nodes)
    differenced = hist.quantity.id in spec.differences
    data = replace(hist, x=difference(hist.x)) if differenced else hist
    paths = {}
    for i in range(n):
        item = path_of(data, i, until, differenced=differenced)
        if item is not None and len(item.path) - 1 > k:
            paths[i] = item
    if not paths:
        reason = f"no driver of {hist.quantity.id} has more than {k} usable months"
        raise ShortHistory({driver: reason for driver in hist.driver if driver is not None})
    centre, spread, flat, level = _scales(paths, columns_)
    if flat.any() and progress is not None:
        names = ", ".join(columns_[j] for j in np.flatnonzero(flat))
        report(f"{progress}: {names} do not vary within objects, their coefficients are zero")
    prior = np.full(k, (spec.spread * level) ** 2)
    constant = spec.constant_spreads.get(hist.quantity.id, spec.constant_spread)
    prior[0] = (constant * level) ** 2
    b0 = (SHAPE - 1) * prior

    classes = sorted(set(hist.classes))
    cls = np.array([classes.index(c) for c in hist.classes])
    children = [np.flatnonzero(hist.parent == i) for i in range(n)]
    roots = np.flatnonzero(hist.parent < 0)

    def scaled(i):
        z, y = paths[i].rows()
        z = (z - centre) / spread
        z[:, flat] = 0.0
        return z, y

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
    steps = spec.sampler.burn + spec.sampler.draws
    for step in range(steps):
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
            # A flat column is out of the likelihood and its prior is independent of the
            # others, so holding it at zero leaves the draw of the others exact.
            gamma[i, flat] = 0.0
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
        if progress is not None and (step + 1) * 10 // steps > step * 10 // steps:
            report(f"{progress} {(step + 1) * 100 // steps}%")

    original = kept_gamma / spread
    original[..., 0] = kept_gamma[..., 0] - (kept_gamma[..., 1:] * centre[1:] / spread[1:]).sum(-1)
    rows = np.array([len(paths[i].observed) if i in paths else 0 for i in range(n)])
    bounds = level_bounds if differenced else training_bounds
    train_low, train_high = bounds(hist, rows, paths)
    return Fit(
        history=hist,
        gamma=original,
        sigma=kept_sigma,
        rows=rows,
        train_low=train_low,
        train_high=train_high,
        differenced=differenced,
        flat=tuple(columns_[j] for j in np.flatnonzero(flat)),
    )


def _normal(precision: np.ndarray, shift: np.ndarray, rng) -> np.ndarray:
    """A draw of ``N(precision^-1 shift, precision^-1)``: ``L^-T (L^-1 shift + z)`` with
    ``precision = L L'``.

    The sampler draws a dozen coefficients a million times per estimate, so LAPACK is called
    directly: the general solvers of numpy cost six times the arithmetic here.
    """
    root, info = lapack.dpotrf(precision, lower=1)
    if info:
        raise np.linalg.LinAlgError("the precision of a node's coefficients is not definite")
    half, _ = lapack.dtrtrs(root, shift, lower=1)
    draw, _ = lapack.dtrtrs(root, half + rng.standard_normal(len(shift)), lower=1, trans=1)
    return draw


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
    """(T,) residuals of a node at the posterior mean; NaN where a month was not used.

    A quantity fitted in differences reports the residuals of its own equation, the changes:
    those are the shocks the version correlates across drivers.
    """
    hist, gamma = fit_.history, fit_.gamma[:, node].mean(axis=0)
    d, lags = hist.d[:, node], hist.quantity.lags
    x = difference(hist.x) if fit_.differenced else hist.x
    y = difference(d) if fit_.differenced else d
    out = np.full(len(d), np.nan)
    for t in range(max(1, lags), len(d)):
        z = regressors(lags, x, y[t - 1], t)
        if np.isfinite(y[t]) and np.all(np.isfinite(z)):
            out[t] = y[t] - gamma @ z
    return out
