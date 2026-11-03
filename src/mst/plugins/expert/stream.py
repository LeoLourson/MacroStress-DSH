"""The shock stream and the responses: what the occurrences of one shock do to the path.

Every random quantity of a shock's month comes from standard normal inputs through
quantile transforms, so the imposition can move them by a Markov chain in the standard
normal space (``subset.py``). The inputs hold one column for the occurrence, one for the
size, one per variable for the copula and one for the copula's chi-square scale; the
equicorrelated gaussian vector of the t copula is built from the per-variable columns, so
at ``rho = 1`` every variable takes one level and at ``rho = 0`` the levels are
uncorrelated while joint extremes stay more frequent than under independence.

The response of one occurrence is read at the drawn level: the cell (horizon node, size
node) with a nonnegative median takes the quantile ``v_i`` of its table, one with a
negative median the quantile ``1 - v_i`` — a high level moves the whole response of the
variable away from zero whichever way it goes, and the level is shared by all horizons and
sizes of the variable, so the response stays one curve. Between the horizon nodes the
quantiles are interpolated linearly in ``k``; in the size the response is a broken line
through ``(0, 0)`` and the three asked sizes, constant beyond the outermost — a scenario
naming a shock larger than any asked gets the response to the largest one asked.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr
from scipy.stats import chi2 as chi2_distribution
from scipy.stats import t as student

#: The lags the answers give, in months: the answers 1, 3, 6 and 12 months out.
HORIZON_NODES = (0, 2, 5, 11)
#: How many lags a response profile covers; the response is constant beyond the last node.
PROFILE = 12
#: The weights of the linear interpolation in ``k`` between the horizon nodes, (12, 4).
LAG_WEIGHTS = np.zeros((PROFILE, 4))
for _k in range(PROFILE):
    if _k <= 2:
        LAG_WEIGHTS[_k, 0] = 1.0 - _k / 2.0
        LAG_WEIGHTS[_k, 1] = _k / 2.0
    elif _k <= 5:
        LAG_WEIGHTS[_k, 1] = (5.0 - _k) / 3.0
        LAG_WEIGHTS[_k, 2] = (_k - 2.0) / 3.0
    else:
        LAG_WEIGHTS[_k, 2] = (11.0 - _k) / 6.0
        LAG_WEIGHTS[_k, 3] = (_k - 5.0) / 6.0


@dataclass(frozen=True)
class ShockTables:
    """Everything the stream of one shock needs, built from its answer by the estimate.

    ``sizes`` are the three asked sizes (the quantiles of the size law), ``size_table`` the
    size law on the level grid, ``responses`` the quantile table of every cell
    ``(variable, horizon node, size node)`` on the same grid and ``medians`` the median of
    every cell, which decides the side of the level lookup.
    """

    frequency: float
    rho: float
    sizes: np.ndarray
    size_table: np.ndarray
    responses: np.ndarray
    medians: np.ndarray
    levels: np.ndarray
    alpha: float

    @property
    def variables(self) -> int:
        return self.responses.shape[0]


@dataclass(frozen=True)
class StreamStatistics:
    """What the stationary stream of one shock contributes to the changes of the path.

    ``mean_change`` and ``cov_change`` are the mean and the covariance of the monthly
    change the shock adds to the level of the path, and ``mean_response`` is the mean
    response of every variable at every lag 0 .. 11, non-occurrences counted as zero.
    """

    mean_change: np.ndarray
    cov_change: np.ndarray
    mean_response: np.ndarray


def draw(tables: ShockTables, rng, horizon: int, draws: int) -> np.ndarray:
    """The standard normal inputs of the shock's months, ``(draws, horizon, 2 + n + 1)``."""
    return rng.standard_normal((draws, horizon, 2 + tables.variables + 1))


def occurrences(tables: ShockTables, inputs: np.ndarray) -> np.ndarray:
    """Whether the shock occurs in a month: probability ``1 - e^{-lambda/12}`` a month."""
    rate = 1.0 - np.exp(-tables.frequency / 12.0)
    if rate == 0.0:
        return np.zeros(inputs.shape[:-1], dtype=bool)
    return ndtr(inputs[..., 0]) < rate


def drawn_sizes(tables: ShockTables, inputs: np.ndarray) -> np.ndarray:
    """The size of the month's occurrence, from the size law at the drawn level."""
    return np.interp(ndtr(inputs[..., 1]), tables.levels, tables.size_table)


def levels(inputs: np.ndarray, rho: float, alpha: float) -> np.ndarray:
    """The copula levels of every variable, ``(..., n)`` in ``(0, 1)``.

    The gaussian vector is equicorrelated at ``rho`` and built from the independent
    per-variable columns; its chi-square scale comes from the last column through the
    quantile transform, so every column of the inputs stays standard normal.
    """
    count = inputs.shape[-1] - 3
    base = inputs[..., 2 : 2 + count]
    # base·(sqrt(1-rho) I + a/n 11') has covariance (1-rho) I + rho 11' when
    # (sqrt(1-rho) + a)^2 = 1 - rho + n rho, so every column stays standard normal.
    common = base.mean(axis=-1, keepdims=True)
    loading = np.sqrt(1.0 - rho + count * rho) - np.sqrt(1.0 - rho)
    gaussian = loading * common + np.sqrt(1.0 - rho) * base
    scale = chi2_distribution.ppf(
        np.clip(ndtr(inputs[..., -1]), 1e-15, 1.0 - 1e-15), alpha
    )
    return student.cdf(gaussian / np.sqrt(scale / alpha)[..., None], alpha)


def _node_values(tables: ShockTables, inputs: np.ndarray) -> np.ndarray:
    """The response of every cell ``(variable, horizon node, size node)`` at the drawn level."""
    count = tables.variables
    v = levels(inputs, tables.rho, tables.alpha)
    values = np.empty((*np.shape(v), 4, 3))
    for variable in range(count):
        for node in range(4):
            for size_node in range(3):
                lookup = v[..., variable]
                if tables.medians[variable, node, size_node] < 0.0:
                    lookup = 1.0 - lookup
                values[..., variable, node, size_node] = np.interp(
                    lookup, tables.levels, tables.responses[variable, node, size_node]
                )
    return values


def _size_knots(tables: ShockTables) -> tuple[np.ndarray, list[int | None]]:
    """The knot sizes of the broken line in increasing order and the size node each reads.

    The origin ``(0, 0)`` is a knot, read as no response; a size node of zero size is
    dropped there, since the model puts the response at size zero at zero.
    """
    asked = [(float(size), node) for node, size in enumerate(tables.sizes) if size != 0.0]
    knots: list[float] = []
    nodes: list[int | None] = []
    for size, node in sorted([(0.0, None), *asked], key=lambda knot: knot[0]):
        if knots and size == knots[-1]:
            nodes[-1] = node  # an equal size: the later size node is read
            continue
        knots.append(size)
        nodes.append(node)
    return np.array(knots), nodes


def _response_profile(tables: ShockTables, inputs: np.ndarray) -> np.ndarray:
    """The response of one month's occurrence at every lag 0 .. 11, ``(..., n, 12)``.

    Linear in ``k`` between the horizon nodes, a broken line in the size through the
    origin and the three asked sizes, constant beyond the outermost knot.
    """
    nodes = _node_values(tables, inputs)
    at_nodes = np.einsum("kj,...ijm->...ikm", LAG_WEIGHTS, nodes)
    size = drawn_sizes(tables, inputs)
    knots, knot_nodes = _size_knots(tables)
    count = tables.variables
    profile = np.zeros((*np.shape(size), count, PROFILE))
    if len(knots) < 2:
        return profile
    knot_values = [
        np.zeros((*np.shape(size), count, PROFILE)) if node is None else at_nodes[..., node]
        for node in knot_nodes
    ]
    segment = np.clip(np.searchsorted(knots, size) - 1, 0, len(knots) - 2)
    for first in range(len(knots) - 1):
        chosen = segment == first
        if not chosen.any():
            continue
        share = np.clip(
            (size - knots[first]) / (knots[first + 1] - knots[first]), 0.0, 1.0
        )[chosen]
        profile[chosen] = (1.0 - share)[..., None, None] * knot_values[first][chosen] + (
            share[..., None, None] * knot_values[first + 1][chosen]
        )
    return profile


def contributions(tables: ShockTables, inputs: np.ndarray, horizon: int) -> np.ndarray:
    """The contribution of this shock to the path, ``(draws, horizon + 1, n)``.

    Month 0 is zero: the shocks of the horizon start at month 1. The inputs cover the
    horizon's months; every occurrence adds its response profile to the months that
    follow, and a month without an occurrence adds nothing.
    """
    months = inputs.shape[1]
    if months < horizon:
        raise ValueError("the inputs cover fewer months than the horizon asks")
    occur = occurrences(tables, inputs)
    # Only the months with an occurrence have a profile: most months have none, and the
    # profile is the cost of the whole contribution.
    profile = np.zeros((*occur.shape, tables.variables, PROFILE))
    if occur.any():
        profile[occur] = _response_profile(tables, inputs[occur])
    path = np.zeros((inputs.shape[0], horizon + 1, tables.variables))
    for month in range(min(months, horizon)):
        span = horizon - month
        path[:, month + 1 : month + 1 + span, :] += profile[:, month, :, :span].transpose(
            0, 2, 1
        )
    return path


def statistics(
    tables: ShockTables, months: int, rng, block: int = 2000
) -> StreamStatistics:
    """The mean and covariance of the monthly change the shock adds, from its stream.

    One path of ``months`` months with its first twelve left out of the change statistics
    (they miss the occurrences before the start); the response is constant beyond the last
    horizon node, so an occurrence shifts the level for good and only its first twelve
    monthly differences move the change. The per-lag mean responses count every month,
    non-occurrences included as zero.
    """
    count = tables.variables
    carry = np.zeros((PROFILE - 1, PROFILE, count))
    sum_change = np.zeros(count)
    sum_outer = np.zeros((count, count))
    counted = 0
    sum_response = np.zeros((PROFILE, count))
    seen = 0
    while seen < months:
        take = min(block, months - seen)
        inputs = draw(tables, rng, 1, take)[:, 0, :]
        occur = occurrences(tables, inputs)
        profile = _response_profile(tables, inputs)
        profile = profile * occur[:, None, None]
        sum_response += profile.sum(axis=0).T
        lag_profile = profile.transpose(0, 2, 1)
        # delta[i, k] is r(k) - r(k-1) of the month, r(-1) = 0; the change of month t
        # sums the delta profiles of the last twelve months, each at its own lag.
        delta = np.diff(
            np.concatenate([np.zeros((take, 1, count)), lag_profile], axis=1), axis=1
        )
        window = np.concatenate([carry, delta])
        change = np.zeros((take, count))
        for age in range(PROFILE):
            change += window[PROFILE - 1 - age : PROFILE - 1 - age + take, age, :]
        start = max(0, PROFILE - seen)
        counted += take - start
        sum_change += change[start:].sum(axis=0)
        sum_outer += change[start:].T @ change[start:]
        carry = window[-(PROFILE - 1) :]
        seen += take
    mean_change = sum_change / max(counted, 1)
    cov_change = sum_outer / max(counted, 1) - np.outer(mean_change, mean_change)
    return StreamStatistics(
        mean_change=mean_change,
        cov_change=cov_change,
        mean_response=sum_response / months,
    )
