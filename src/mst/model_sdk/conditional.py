"""Drawing shocks under linear constraints and the probability of a constraint box.

A macro model that reduces to ``y = mu + Phi eps`` with standard normal ``eps`` — a VAR with
any volatility model, a random walk with drift — enforces its scenario constraints the same
way. With ``Z = W Phi_D`` the image ``v = Z eps_D`` is normal; its exact rows are fixed, its
interval rows are drawn from the box-truncated normal by the Gibbs sampler of Geweke (1991),
and ``eps_D`` given ``v`` is ``eta + Z' (Z Z')^{-1} (v - Z eta)`` with standard normal ``eta``
(Antolin-Diaz, Petrella and Rubio-Ramirez, 2021, Journal of Monetary Economics 117). The
scenario is infeasible when a row of ``Z`` is a combination of the earlier ones or no driving
shock moves it; a row that merely repeats an earlier one is redundant — its range is
intersected with the range of the copy and the row is dropped.

The probability of the box itself — what the price of a scenario needs — is estimated by the
GHK sampler (Geweke 1991, Hajivassiliou and McFadden 1994, Keane 1994) in logarithms: the
coordinates are visited once in Cholesky order, each standard normal draw comes from its
truncated conditional through ``ndtri_exp``, and every step adds ``log(Phi(b) - Phi(a))`` of
that coordinate. In one dimension the sampler is exact whatever ``samples`` is: there is no
earlier coordinate to condition on. ``scipy.stats.multivariate_normal.cdf``, by contrast,
returns zero or noise in the far tail the price cares about, and the tests pin the difference.
"""

import numpy as np
from scipy.special import log_ndtr, logsumexp, ndtri_exp

SWEEPS = 50
"""Gibbs sweeps per repetition for interval constraints, started inside the box."""
RANK_TOLERANCE = 1e-9
"""A constraint row is dependent when the driving shocks move it, beyond what the earlier rows
already fix, by less than this share of what all shocks move it."""
DEGENERATE = 1e-14
"""A coordinate whose variance is this share of the largest one is read as fixed."""


class InfeasibleScenario(ValueError):
    """The driving shocks cannot meet the constraints; the numbers are the constraint indices."""

    def __init__(self, constraints: list[int]):
        self.constraints = constraints
        super().__init__(
            "the driving shocks cannot meet constraints "
            + ", ".join(str(c + 1) for c in constraints)
            + " (numbered from 1): each is a combination of earlier ones, repeats one with a "
            "disjoint range, or no driving shock moves it"
        )


def dependent_rows(driving: np.ndarray, every: np.ndarray) -> list[int]:
    """Constraints the driving shocks cannot move on their own.

    Row ``c`` of ``driving`` is dependent when its distance to the span of the earlier
    independent rows is negligible next to row ``c`` of ``every`` (the loadings of all shocks):
    a loading at rounding level is no loading.

    A row that repeats an earlier one is not reported: two copies of one constraint are
    consistent, and ``draw_driving`` intersects their ranges and draws from the intersection,
    refusing only a pair whose ranges do not meet. Only a full row of ``every`` repeats another:
    equal loadings on the driving shocks alone are not a repeated constraint.
    """

    kept, dependent = [], []
    for c, row in enumerate(driving):
        if any(np.array_equal(every[k], every[c]) for k in kept):
            continue
        if kept:
            basis = driving[kept].T
            residual = row - basis @ np.linalg.lstsq(basis, row, rcond=None)[0]
        else:
            residual = row
        if np.linalg.norm(residual) <= RANK_TOLERANCE * np.linalg.norm(every[c]):
            dependent.append(c)
        else:
            kept.append(c)
    return dependent


def truncated_normal(low: float, high: float, rng) -> float:
    """A standard normal draw restricted to ``[low, high]``, accurate far in the tails."""
    if low > 0:
        return -truncated_normal(-high, -low, rng)
    log_low, log_high = log_ndtr(low), log_ndtr(high)
    # log(Phi(low) + u (Phi(high) - Phi(low))), written so that nothing overflows.
    share = rng.random()
    value = ndtri_exp(log_high + np.log(share + (1 - share) * np.exp(log_low - log_high)))
    return float(np.clip(value, low, high))


def box_gibbs(mean, cov, low, high, rng, sweeps: int) -> np.ndarray:
    """A draw of ``N(mean, cov)`` restricted to the box ``[low, high]`` (Geweke, 1991)."""
    precision = np.linalg.inv(cov)
    x = np.clip(mean, low, high)
    for _ in range(sweeps):
        for i in range(len(x)):
            spread = 1 / np.sqrt(precision[i, i])
            centre = (
                mean[i]
                - (precision[i] @ (x - mean) - precision[i, i] * (x[i] - mean[i]))
                / (precision[i, i])
            )
            x[i] = centre + spread * truncated_normal(
                (low[i] - centre) / spread, (high[i] - centre) / spread, rng
            )
    return x


def draw_driving(z, low, high, rng, sweeps: int = SWEEPS, *, every=None) -> np.ndarray:
    """``eps ~ N(0, I)`` given ``low <= z eps <= high``; ``z`` has independent rows.

    A row that repeats an earlier one is redundant, not a contradiction: its range is intersected
    with the range of the copy and the row is dropped. A pair of copies with no common value is
    a genuine contradiction and raises ``InfeasibleScenario`` naming the later row. ``every`` —
    the loadings of all shocks, of which ``z`` is the driving slice — tells a repeated constraint
    from two constraints whose driving loadings happen to coincide; with it, only a row repeated
    in full counts as redundant.
    """
    z, low, high = _merge_redundant(z, low, high, every)
    gram = z @ z.T
    exact = low == high
    image = np.where(exact, low, 0.0)
    free = ~exact
    if free.any():
        if exact.any():
            gain = np.linalg.solve(gram[np.ix_(exact, exact)], gram[np.ix_(exact, free)]).T
            mean = gain @ low[exact]
            cov = gram[np.ix_(free, free)] - gain @ gram[np.ix_(exact, free)]
        else:
            mean, cov = np.zeros(free.sum()), gram
        image[free] = box_gibbs(mean, cov, low[free], high[free], rng, sweeps)
    eta = rng.standard_normal(z.shape[1])
    return eta + z.T @ np.linalg.solve(gram, image - z @ eta)


def _merge_redundant(z, low, high, every=None):
    """Drop rows that repeat an earlier one, intersecting their ranges.

    Returns the reduced ``(z, low, high)``; raises ``InfeasibleScenario`` when two copies of a
    row demand disjoint ranges. ``every``, when given, is the full loading matrix: a row counts
    as a copy only if its full row repeats one, not just its driving slice.
    """
    low = np.array(low, dtype=float).copy()
    high = np.array(high, dtype=float).copy()
    kept: list[int] = []
    for c, row in enumerate(z):
        same = next(
            (
                k
                for k in kept
                if np.array_equal(z[k], row)
                and (every is None or np.array_equal(every[k], every[c]))
            ),
            None,
        )
        if same is None:
            kept.append(c)
            continue
        overlap = (max(low[same], low[c]), min(high[same], high[c]))
        if overlap[0] > overlap[1]:
            raise InfeasibleScenario([c])
        low[same], high[same] = overlap
    index = np.array(kept, dtype=int)
    return z[index], low[index], high[index]


def pick_latent(log_probabilities, rng) -> tuple[int, float]:
    """``(index, ess)``: one candidate of a hidden coordinate drawn by its weight.

    The candidates of a hidden coordinate — a regime path, a volatility path — are weighted
    by the probability each gives the event; the imposition of the canon keeps the law of
    ``R`` and reweights everything else, so a candidate is picked with probability
    proportional to ``exp(log p)``. The effective sample size ``(sum p)^2 / sum p^2`` tells
    how many candidates carry the weight: a size near one means the event rides a single
    candidate, and the plan's stopping rule reads the median of these sizes on a real
    scenario.
    """
    weights = np.asarray(log_probabilities, dtype=float)
    if len(weights) == 0:
        raise ValueError("a hidden coordinate needs at least one candidate")
    shift = weights - weights.max()
    share = np.exp(shift)
    total = share.sum()
    if not np.isfinite(total) or total <= 0.0:
        # Every candidate gives the event no probability: any pick is as good, and the
        # caller sees the effective size of zero — the scenario contradicts the models.
        return int(rng.integers(len(weights))), 0.0
    share = share / total
    picked = int(np.searchsorted(np.cumsum(share), rng.random()))
    picked = min(picked, len(weights) - 1)
    ess = float(1.0 / np.sum(share**2))
    return picked, ess


def log_box_probability(mean, cov, low, high, rng, samples: int) -> float:
    """``log P(low <= X <= high)`` for ``X ~ N(mean, cov)`` by the GHK sampler.

    One dimension is exact: the sampler's only step has no earlier coordinate to condition on,
    so any ``samples`` gives ``log(Phi(b) - Phi(a))`` of the standardised interval. A coordinate
    the covariance does not move is checked against the box directly, and a violation there
    returns ``-inf``: the box has no probability at all, the case lemma 2 of the canon names.
    """
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    m = len(mean)
    if m == 0:
        return 0.0
    if not np.all(low <= high):
        return float("-inf")
    variance = np.diag(cov)
    scale = float(variance.max()) if m else 0.0
    fixed = variance <= DEGENERATE * max(scale, np.finfo(float).tiny)
    if np.any((mean[fixed] < low[fixed]) | (mean[fixed] > high[fixed])):
        return float("-inf")
    moving = np.flatnonzero(~fixed)
    if not len(moving):
        return 0.0
    centred_low = low[moving] - mean[moving]
    centred_high = high[moving] - mean[moving]
    factor = _cholesky(cov[np.ix_(moving, moving)], scale)
    log_weights = _ghk_row(factor, centred_low, centred_high, rng, samples)
    return float(logsumexp(log_weights) - np.log(samples))


def log_box_probabilities(
    means, covs, low, high, rng, samples: int, *, chunk: int = 0
) -> np.ndarray:
    """``(N,)`` log box probabilities of a batch of gaussian laws, one GHK pass each.

    ``means`` is ``(N, m)`` and ``covs`` is ``(N, m, m)``; ``low`` and ``high`` are the one
    box every row is measured against, in the order of the coordinates. Rows are processed in
    chunks so the ``(rows, samples, m)`` work array stays small; a row that violates its box
    on a coordinate its covariance does not move is ``-inf`` without a sampler pass.
    """
    means = np.asarray(means, dtype=float)
    covs = np.asarray(covs, dtype=float)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    count, m = means.shape
    result = np.zeros(count)
    if m == 0:
        return result
    if count == 0:
        return result
    if not chunk:
        chunk = max(1, int(4_000_000 // (max(samples, 1) * m)))
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        block = slice(start, stop)
        rows = stop - start
        variance = np.diagonal(covs[block], axis1=1, axis2=2)
        scale = variance.max(axis=1)
        fixed = variance <= (DEGENERATE * np.maximum(scale, np.finfo(float).tiny))[:, None]
        inside = (means[block] >= low[None, :]) & (means[block] <= high[None, :])
        result[block] = np.where(
            np.any(fixed & ~inside, axis=1), float("-inf"), 0.0
        )
        alive = np.flatnonzero(
            ~np.any(fixed & ~inside, axis=1) & (scale > 0.0)
        )
        if not len(alive):
            continue
        # Every row keeps its own moving coordinates; the union of them is processed at once,
        # and a coordinate fixed in one row only is jittered to the scale of that row.
        moving = np.flatnonzero(np.any(~fixed[alive], axis=0))
        if not len(moving):
            continue
        picked = alive
        sub_means = means[block][picked][:, moving]
        sub_covs = covs[block][picked][:, moving][:, :, moving]
        sub_fixed = fixed[picked][:, moving]
        centred_low = low[moving][None, :] - sub_means
        centred_high = high[moving][None, :] - sub_means
        jitter = np.where(
            sub_fixed,
            1e-12 * np.maximum(scale[picked], np.finfo(float).tiny)[:, None],
            0.0,
        )
        factors = _cholesky(
            sub_covs + jitter[:, :, None] * np.eye(len(moving))[None], 1.0
        )
        weights = _ghk_batch(
            factors, centred_low, centred_high, rng, samples
        )
        result[start + picked] = logsumexp(weights, axis=1) - np.log(samples)
    return result


def _cholesky(cov, scale: float):
    """The Cholesky factor of a covariance, with a relative jitter when it is singular."""
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        jitter = 1e-12 * max(scale, np.finfo(float).tiny)
        return np.linalg.cholesky(cov + jitter * np.eye(cov.shape[-1]))


def _ghk_row(factor, low, high, rng, samples: int) -> np.ndarray:
    """``(samples,)`` log weights of one GHK pass over a single law.

    ``factor`` is the lower Cholesky factor of the covariance; ``low`` and ``high`` are the
    centred bounds the standard normal ``z`` of the factor must meet.
    """
    m = factor.shape[0]
    z = np.zeros((samples, m))
    log_weights = np.zeros(samples)
    with np.errstate(divide="ignore"):
        for i in range(m):
            partial = z[:, :i] @ factor[i, :i] if i else 0.0
            a = (low[i] - partial) / factor[i, i]
            b = (high[i] - partial) / factor[i, i]
            la, lb = log_ndtr(a), log_ndtr(b)
            share = rng.random(samples)
            log_target = np.logaddexp(la + np.log1p(-share), lb + np.log(share))
            z[:, i] = np.clip(ndtri_exp(log_target), a, b)
            log_weights = log_weights + lb + np.log1p(-np.exp(np.minimum(la - lb, 0.0)))
    return log_weights


def _ghk_batch(factors, low, high, rng, samples: int) -> np.ndarray:
    """``(rows, samples)`` log weights of GHK passes over a batch of laws.

    ``factors`` is ``(rows, m, m)`` lower Cholesky factors; ``low`` and ``high`` are the
    centred bounds ``(rows, m)``. The coordinates are visited in order, every row drawing its
    own standard normals, so one pass costs ``rows * samples * m^2 / 2`` products.
    """
    rows, m = factors.shape[0], factors.shape[1]
    z = np.zeros((rows, samples, m))
    log_weights = np.zeros((rows, samples))
    with np.errstate(divide="ignore"):
        for i in range(m):
            if i:
                partial = np.einsum("rsj,rj->rs", z[:, :, :i], factors[:, i, :i])
            else:
                partial = np.zeros((rows, samples))
            a = (low[:, i][:, None] - partial) / factors[:, i, i][:, None]
            b = (high[:, i][:, None] - partial) / factors[:, i, i][:, None]
            la, lb = log_ndtr(a), log_ndtr(b)
            share = rng.random((rows, samples))
            log_target = np.logaddexp(la + np.log1p(-share), lb + np.log(share))
            z[:, :, i] = np.clip(ndtri_exp(log_target), a, b)
            log_weights += lb + np.log1p(-np.exp(np.minimum(la - lb, 0.0)))
    return log_weights
