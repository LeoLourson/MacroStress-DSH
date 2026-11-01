"""The conditions on means and probabilities: the exponential tilt of the causes.

A condition ``low <= E_Q f <= high`` — or ``Q(B) >= p``, which is the same with
``f = 1_B`` — cannot be met by truncating a box: it speaks about the average, not about
every path. The canon imposes it by tilting the causes: within every value of ``R`` the law
of ``omega_D`` turns proportional to ``e^{lambda' f}`` (section 4, the theorem), and the
price of that is the Kullback—Leibler divergence from the law the support conditions left.

``solve_lambda`` finds ``lambda`` by minimising the convex dual

.. math::

    \\\\mathrm{D}(\\\\lambda) = \\\\frac{1}{K} \\\\sum_g \\\\log \\\\Big( \\\\frac{1}{n_g}
    \\\\sum_{i \\\\in g} e^{\\\\lambda' f_i} \\\\Big)
    - \\\\sum_i \\\\big( \\\\lambda_i^+ \\\\ell_i - \\\\lambda_i^- r_i \\\\big),

whose minimum is minus the price and whose minimiser is the ``lambda`` of the theorem: the
first term is the log-partition of the tilt, the second swaps in the bound each multiplier
pushes towards — the lower one for a positive ``lambda``, the upper one for a negative.
A one-sided condition bounds the sign of its multiplier, and L-BFGS-B carries the bound.
An inactive condition — the plain average already meets it — keeps ``lambda = 0`` and
changes nothing.

``resample`` then turns the weights into a plain sample: systematic resampling inside every
group of ``R`` keeps the law of ``R`` and leaves the consumers of the sample — percentiles,
CRPS, the position engine — working with equal weights, while ``effective_size`` reports how
many paths carry the weight really.

The hidden coordinates — the regime path, the volatility path — are selected by the support
box before the tilt reads the paths; the tilt then reweights the draws of the causes inside
their group. When a scenario has no support conditions the tilt is the whole imposition.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

LAMBDA_BOUND = 30.0
"""The far bound of a multiplier: a tilt beyond e^{30} carries the answer on a single path,
and the report reads such a scenario as resting on a rare tail."""


@dataclass(frozen=True)
class Tilt:
    """The solved tilt: the multipliers, the weights and what they did."""

    lam: np.ndarray
    """``(k,)`` the multipliers of the conditions; zero where a condition is inactive."""
    weights: np.ndarray
    """``(N,)`` the weight of every path inside its group, summing to one per group."""
    price: float
    """``KL(Q* || P_s)``: the price of the moment conditions on top of the support ones."""
    effective_size: float
    """``(sum w)^2 / sum w^2`` over all paths: how many carry the tilted law."""
    means: np.ndarray
    """``(k,)`` the means of the functionals under the tilt, against their bounds."""
    feasible: bool
    """Whether the tilt meets every condition the sample can meet."""
    active: tuple[bool, ...]
    """Which conditions carry a non-zero multiplier."""


def solve_lambda(
    values: np.ndarray,
    group: np.ndarray,
    low,
    high,
    *,
    seed: int = 0,
) -> Tilt:
    """The tilt that meets ``low <= E_Q f <= high`` at the least price.

    ``values`` is ``(N, k)`` — the ``k`` functionals of the conditions on ``N`` paths;
    ``group`` is ``(N,)`` — which value of ``R`` every path belongs to, the tilt running
    inside each group. ``low`` and ``high`` are ``(k,)`` with ``-inf``/``+inf`` for the open
    sides. Paths enter with equal weight inside their group.
    """
    values = np.asarray(values, dtype=float)
    group = np.asarray(group, dtype=int)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    k = values.shape[1]
    counts = np.bincount(group)
    if np.any(counts == 0):
        raise ValueError("a group of the tilt holds no path")
    even = _weights(np.zeros(k), values, group)
    if k == 0:
        return _tilt(np.zeros(0), even, 0.0, values, group, True, ())
    plain = _means(even, values, group)
    if np.all(plain >= low - 1e-12) and np.all(plain <= high + 1e-12):
        # Every condition already holds: no tilt, no price, the sample as it is.
        return _tilt(
            np.zeros(k),
            even,
            0.0,
            values,
            group,
            True,
            tuple(False for _ in range(k)),
        )

    bounds = []
    for i in range(k):
        lower, upper = -LAMBDA_BOUND, LAMBDA_BOUND
        if np.isfinite(low[i]) and not np.isfinite(high[i]):
            lower = 0.0  # a lower bound alone is met by a non-negative multiplier
        if np.isfinite(high[i]) and not np.isfinite(low[i]):
            upper = 0.0  # an upper bound alone by a non-positive one
        bounds.append((lower, upper))

    def target(lam: np.ndarray) -> np.ndarray:
        """The bound each multiplier pushes towards: the lower one when positive."""
        return np.where(lam > 0, low, np.where(lam < 0, high, 0.0))

    def dual(lam: np.ndarray) -> float:
        partition = _log_partition(lam, values, group)
        return float(partition - np.sum(lam * target(lam)))

    def gradient(lam: np.ndarray) -> np.ndarray:
        # The subgradient of the dual: at a zero multiplier the goal is the violated bound
        # — the clip — so a condition the plain average breaks moves its multiplier away
        # from zero instead of resting on the kink.
        weights = _weights(lam, values, group)
        found = _means(weights, values, group)
        goal = np.where(
            lam > 0, low, np.where(lam < 0, high, np.clip(found, low, high))
        )
        return found - goal

    started = np.zeros(k)
    found = minimize(
        dual,
        started,
        jac=gradient,
        bounds=bounds,
        method="L-BFGS-B",
        options={"ftol": 1e-14, "gtol": 1e-12, "maxiter": 1000},
    )
    lam = np.clip(found.x, [b[0] for b in bounds], [b[1] for b in bounds])
    weights = _weights(lam, values, group)
    means = _means(weights, values, group)
    magnitude = np.maximum(
        np.abs(np.where(np.isfinite(low), low, 0.0)),
        np.abs(np.where(np.isfinite(high), high, 0.0)),
    )
    scale = 1e-6 + 1e-8 * magnitude
    feasible = bool(np.all(means >= low - scale) and np.all(means <= high + scale))
    price = -float(found.fun) if feasible else float("inf")
    active = tuple(bool(abs(component) > 1e-8) for component in lam)
    return _tilt(lam, weights, price, values, group, feasible, active)


def resample(weights: np.ndarray, group: np.ndarray, rng) -> np.ndarray:
    """``(N,)`` indices: a systematic resampling of the paths inside every group.

    Each group keeps its size and its law: the weights pick the paths, the systematic draw
    spreads them, and the consumers of the sample see equal weights again. Weight one in
    every group returns the paths in their order.
    """
    weights = np.asarray(weights, dtype=float)
    group = np.asarray(group, dtype=int)
    picked = np.empty(len(weights), dtype=int)
    for g in np.unique(group):
        rows = np.flatnonzero(group == g)
        share = weights[rows] / weights[rows].sum()
        positions = (rng.random() + np.arange(len(rows))) / len(rows)
        cumulative = np.cumsum(share)
        # The first cumulative share above the position: a path of zero weight is never picked.
        chosen = np.searchsorted(cumulative, positions, side="right")
        picked[rows] = rows[np.minimum(chosen, len(rows) - 1)]
    return picked


def _weights(lam: np.ndarray, values: np.ndarray, group: np.ndarray) -> np.ndarray:
    """The tilted weights: ``e^{lambda' f}`` normalised inside every group."""
    scores = values @ lam if len(lam) else np.zeros(len(values))
    weights = np.empty(len(values))
    for g in np.unique(group):
        rows = group == g
        shifted = scores[rows] - scores[rows].max()
        share = np.exp(shifted)
        weights[rows] = share / share.sum()
    return weights


def _log_partition(lam: np.ndarray, values: np.ndarray, group: np.ndarray) -> float:
    """``(1/K) sum_g log(mean_{i in g} e^{lambda' f_i})``, stable in the exponent."""
    scores = values @ lam if len(lam) else np.zeros(len(values))
    total = 0.0
    groups = np.unique(group)
    for g in groups:
        rows = group == g
        shifted = scores[rows] - scores[rows].max()
        total += np.log(np.mean(np.exp(shifted))) + scores[rows].max()
    return float(total / len(groups))


def _means(weights: np.ndarray, values: np.ndarray, group: np.ndarray) -> np.ndarray:
    """The weighted mean of the functionals: groups equal weight, paths weighted inside."""
    found = np.zeros(values.shape[1])
    groups = np.unique(group)
    for g in groups:
        rows = group == g
        found += (weights[rows] @ values[rows]) / len(groups)
    return found


def _tilt(lam, weights, price, values, group, feasible, active) -> Tilt:
    weights = np.asarray(weights, dtype=float)
    effective = float(weights.sum() ** 2 / np.sum(weights**2))
    return Tilt(
        lam=np.asarray(lam, dtype=float),
        weights=weights,
        price=float(price),
        effective_size=effective,
        means=_means(weights, values, group),
        feasible=bool(feasible),
        active=active,
    )
