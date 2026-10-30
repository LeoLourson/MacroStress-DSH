"""Drawing shocks under linear constraints: the machinery every macro model can reuse.

A macro model that reduces to ``y = mu + Phi eps`` with standard normal ``eps`` — a VAR with
any volatility model, a random walk with drift — enforces its scenario constraints the same
way. With ``Z = W Phi_D`` the image ``v = Z eps_D`` is normal; its exact rows are fixed, its
interval rows are drawn from the box-truncated normal by the Gibbs sampler of Geweke (1991),
and ``eps_D`` given ``v`` is ``eta + Z' (Z Z')^{-1} (v - Z eta)`` with standard normal ``eta``
(Antolin-Diaz, Petrella and Rubio-Ramirez, 2021, Journal of Monetary Economics 117). The
scenario is infeasible when a row of ``Z`` is a combination of the earlier ones or no driving
shock moves it; a row that merely repeats an earlier one is redundant — its range is
intersected with the range of the copy and the row is dropped.
"""

import numpy as np
from scipy.special import log_ndtr, ndtri_exp

SWEEPS = 50
"""Gibbs sweeps per repetition for interval constraints, started inside the box."""
RANK_TOLERANCE = 1e-9
"""A constraint row is dependent when the driving shocks move it, beyond what the earlier rows
already fix, by less than this share of what all shocks move it."""


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
