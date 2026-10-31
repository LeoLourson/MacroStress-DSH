"""The platform measure of scenario implausibility: ``dist(P_s, P) = -log P(C)``.

Section 4 of the scheme measures how far a scenario reaches from the unconditional run by the
distance ``dist(P_s, P)``. Under the conditional scheme the scenario ``P_s`` differs from ``P``
only in the marginal distribution of the driving shocks, and the conditional forecast ``Q_s``
is the distribution closest to the original one that satisfies every constraint, so inside the
support the two densities differ by a constant: this module uses the plan's definition
``dist(P_s, P) = -log P(C)``, where ``P(C)`` is the joint probability that the *unconditional*
run of the model satisfies every constraint at once. The Mahalanobis ``delta`` of the plugin
stays beside it as a model-specific severity; this distance is available to every scenario and
every macro model.

``P(C)`` is estimated two ways. The Monte-Carlo way counts the share of the base run's paths
that satisfy every constraint together — the same paths the constraint gate reads, and the
joint share, not the product of the per-constraint shares: correlated constraints make the
product wrong, and the tests pin the difference. When no drawn path satisfies everything, the
share is zero and the exact way takes over: per draw of the parameters the constraint
functionals are normal, the probability of the box follows from the Gaussian cdf, and the
mixture average over the draws is the answer — the same structure the BVAR severity uses.
"""

import numpy as np
from scipy.stats import multivariate_normal, norm

from .contracts import LinearConstraint


def constraint_matrix(constraints: tuple[LinearConstraint, ...], ids: list[str]) -> np.ndarray:
    """``(m, n)`` The weights of every constraint functional over the *variables of one month*.

    A term addresses ``variable`` at ``month``; the caller applies the returned rows to the
    paths of that month — ``paths[:, month, :ids.index(variable)]`` is not how the map is used,
    because one constraint may span several variables of the same month only. A constraint with
    terms at different months cannot be a single row: raise, and let the caller fall back to
    the Monte-Carlo estimate.
    """
    index = {id_: k for k, id_ in enumerate(ids)}
    rows = []
    for constraint in constraints:
        months = {term.month for term in constraint.terms}
        if len(months) > 1:
            raise ValueError(
                "the mixture estimate needs one month per constraint; "
                "a constraint across months keeps the Monte-Carlo estimate"
            )
        row = np.zeros(len(ids))
        for term in constraint.terms:
            row[index[term.variable]] += term.weight
        rows.append(row)
    return np.asarray(rows, dtype=float)


def satisfied(base_paths: np.ndarray, ids: list[str], constraints) -> np.ndarray:
    """``(R,)`` Whether every path satisfies every constraint at once.

    ``base_paths`` is ``(R, H + 1, n)`` in the model scale. The joint share of this indicator —
    not the product of the per-constraint shares — is the Monte-Carlo estimate of ``P(C)``:
    correlated constraints satisfy jointly more or less often than independently.
    """
    index = {id_: k for k, id_ in enumerate(ids)}
    total = np.ones(base_paths.shape[0], dtype=bool)
    for constraint in constraints:
        value = np.zeros(base_paths.shape[0])
        for term in constraint.terms:
            value += term.weight * base_paths[:, term.month, index[term.variable]]
        total &= (constraint.low <= value) & (value <= constraint.high)
    return total


def box_probability_mc(base_paths: np.ndarray, ids: list[str], constraints) -> float:
    """``P(C)`` as the share of the base run's paths that satisfy every constraint.

    A scenario without constraints has no ``dist`` at all rather than a zero distance, so an
    empty list raises; the caller reports the distance as absent.
    """
    if not constraints:
        raise ValueError("a scenario without constraints has no dist to estimate")
    return float(np.mean(satisfied(base_paths, ids, constraints)))


def box_probability_mixture(
    means: np.ndarray, covs: np.ndarray, constraints: tuple[LinearConstraint, ...]
) -> float:
    """``P(C)`` for a mixture of gaussians over the constraint functionals.

    ``means`` is ``(R, m)`` and ``covs`` is ``(R, m, m)``: the mean and covariance of the ``m``
    constraint functionals — in the order of ``constraints`` — under each of the ``R`` mixture
    components. Per component the box probability is the multivariate normal integral from
    ``low`` to ``high``; the mixture average is the answer. One constraint in one dimension
    goes through the univariate cdf, which is exact where the multidimensional quadrature only
    approximates.
    """
    if not constraints:
        raise ValueError("a scenario without constraints has no dist to estimate")
    means = np.asarray(means, dtype=float)
    covs = np.asarray(covs, dtype=float)
    if means.shape != (len(covs), len(constraints)):
        raise ValueError(
            f"means must be (R, {len(constraints)}), got {means.shape}; "
            "the functionals follow the order of the constraints"
        )
    if covs.shape != (len(covs), len(constraints), len(constraints)):
        raise ValueError(
            f"covs must be (R, {len(constraints)}, {len(constraints)}), got {covs.shape}"
        )
    low = np.array([constraint.low for constraint in constraints])
    high = np.array([constraint.high for constraint in constraints])
    probabilities = np.empty(len(covs))
    for r in range(len(covs)):
        if len(constraints) == 1:
            sd = float(np.sqrt(max(covs[r, 0, 0], 0.0)))
            probabilities[r] = norm.cdf(high[0], loc=means[r, 0], scale=sd) - norm.cdf(
                low[0], loc=means[r, 0], scale=sd
            )
            continue
        probabilities[r] = multivariate_normal(
            mean=means[r], cov=covs[r], allow_singular=True
        ).cdf(high, lower_limit=low)
    return float(np.mean(probabilities))


def dist_from_probability(pc: float) -> float:
    """``-log P(C)``; a probability of zero — no unconditional path works — is an infinite dist."""
    if pc <= 0.0:
        return float("inf")
    return float(-np.log(pc))


def dist_mc(base_paths: np.ndarray, ids: list[str], constraints) -> float:
    """``dist`` from the Monte-Carlo share of the base run; the way the report labels ``mc``."""
    return dist_from_probability(box_probability_mc(base_paths, ids, constraints))


def dist_mixture(
    means: np.ndarray, covs: np.ndarray, constraints: tuple[LinearConstraint, ...]
) -> float:
    """``dist`` from the exact box probability of the mixture; the report labels it ``exact``."""
    return dist_from_probability(box_probability_mixture(means, covs, constraints))
