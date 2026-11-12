"""The price of a scenario: ``dist(s) = KL(P_s || P) = E_P[-log P(S | R)]``.

Section 4 of the canon (docs/concept/formalization.md) imposes a scenario by conditioning the
base measure ``P`` on the event ``S`` while the law of ``R`` — the parameters and every
coordinate the causes do not name — stays as it was. With no moment conditions the imposed
measure is ``P`` conditioned on ``S``, and its Kullback—Leibler divergence from ``P`` is the
mean of ``-log P(S | R)`` over the law of ``R``. The older code measured ``-log P(S)``, which
reads the same only when ``R`` is constant: with named causes the price rises, because
explaining the event through fewer coordinates costs more (section 5 of the canon).

``P(S | R)`` is not counted on simulated paths: a share of a few thousand paths cannot tell a
rare box from an impossible one. The macro model hands the law of the constraint functionals
given ``R`` and the hidden coordinates (``MacroModel.constraint_law``), and the kernel here
mixes the rows of every group into

.. math::

    \\mathrm{dist} = -\\frac{1}{K} \\sum_g \\log \\Big( \\frac{1}{n_g}
    \\sum_{i \\in g} e^{\\log w_i} P_i \\Big),

with ``P_i`` the gaussian box probability of row ``i`` by the GHK sampler
(``mst.model_sdk.conditional``) and ``e^{log w_i}`` the importance weight of the row inside
its group — a plain draw carries weight one, a forced regime path carries its prior. The
groups are one draw of ``R`` each, so the spread of ``-log P(S | R)`` over them is the
standard error of the estimate. A group of probability zero — the box is unreachable at that
``R``, the case lemma 2 names — makes the distance infinite and sets the message.

``log P(S)`` is reported beside the distance: averaged over the same groups, it separates the
price of the event itself from the part the parameter uncertainty adds.
"""

import numpy as np
from scipy.special import logsumexp

from .conditional import log_box_probabilities
from .contracts import ConstraintLaw

GHK_SAMPLES = 64
"""GHK draws per row of the law; with the 64 latent candidates of a group this averages the
box probability over 4096 draws before the logarithm is taken."""
DIST_STREAM = 15
"""The stream of the GHK draws inside a run's seed, away from the model's own streams."""

ONE_SIDED_STREAM = 16
"""The stream of the median draws and the GHK draws of the one-sided price, apart from
the two-sided price's own stream."""

PRICE_ERROR_LIMIT = 0.10
"""A price whose standard error exceeds this share of it is treated as not estimated."""


def price_estimated(distance: float, standard_error: float) -> bool:
    """Whether a price with this error reads as an estimate (the canon, section 7).

    The price is a mean over values of ``R`` and its error is the spread over the groups;
    when the error exceeds ``PRICE_ERROR_LIMIT`` of the estimate, noise would set the bound
    the price is used for, so the report shows the number only with the mark ``not
    estimated``. A price that is not finite — infinite or missing — is not an estimate
    either; a price of exactly zero with no error is (nothing to price was kept).
    """
    if not (np.isfinite(distance) and np.isfinite(standard_error)):
        return False
    return bool(standard_error <= PRICE_ERROR_LIMIT * distance)


def median_of_law(law: ConstraintLaw, rng, samples: int = GHK_SAMPLES) -> np.ndarray:
    """The median of every coordinate of the law's mixture of rows, one draw set per row.

    For each coordinate, ``samples`` draws from the marginal normal of every row, each draw
    carrying its row's importance weight ``e**log_weight``; the weighted median of the draws
    is the median of the mixture. The medians are of the unshifted functionals — the
    plan's bounds already have the level-0 impacts taken out, so the medians and the bounds
    stand on the same footing.
    """
    means = np.asarray(law.means, dtype=float)
    covs = np.asarray(law.covs, dtype=float)
    log_weight = np.asarray(law.log_weight, dtype=float)
    rows, m = means.shape
    if rows == 0 or m == 0:
        return np.zeros(m)
    share = np.exp(log_weight - log_weight.max())
    spread = np.sqrt(np.clip(np.diagonal(covs, axis1=1, axis2=2), 0.0, None))
    draws = means[:, None, :] + spread[:, None, :] * rng.standard_normal((rows, samples, m))
    weights = np.repeat(share, samples)
    medians = np.empty(m)
    for coordinate in range(m):
        values = draws[:, :, coordinate].ravel()
        order = np.argsort(values)
        cumulative = np.cumsum(weights[order])
        medians[coordinate] = values[order][
            np.searchsorted(cumulative, 0.5 * cumulative[-1])
        ]
    return medians


def one_sided_box(low, high, medians) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The rays of the one-sided event ``A+``: ``(low, high, kept)`` of the kept coordinates.

    An interval below the median of its functional — ``high < median`` — becomes the ray
    down from its upper bound ``(-inf, high)``; an interval above the median — ``low >
    median`` — becomes the ray up from its lower bound ``(low, +inf)``; a coordinate whose
    interval contains the median is dropped, and its number lands in the caller's dropped
    list. The width of the interval does not enter the rays, which is what lets events of
    unequal boxes be compared.
    """
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    medians = np.asarray(medians, dtype=float)
    below = high < medians
    above = low > medians
    kept = np.flatnonzero(below | above)
    side_low = np.where(below, -np.inf, low)[kept]
    side_high = np.where(below, high, np.inf)[kept]
    return side_low, side_high, kept


class ContradictsModels(ValueError):
    """Some group of the law gives the event no probability: the scenario cannot be imposed."""

    def __init__(self, constraints: tuple[int, ...] = (), *, group: int | None = None):
        self.constraints = tuple(constraints)
        self.group = group
        details = ""
        if constraints:
            details += "; constraint " + ", ".join(str(number) for number in constraints)
        if group is not None:
            details += f" in group {group}"
        super().__init__(
            "the scenario contradicts the models: at some value of R the event has no "
            "probability" + details
        )


def price_of_law(
    law: ConstraintLaw,
    low,
    high,
    *,
    rng,
    samples: int = GHK_SAMPLES,
) -> tuple[float, float, float]:
    """``(distance, log_probability, standard_error)`` of a scenario's law against its box.

    ``low`` and ``high`` are the bounds of every constraint, in the order of the law's
    coordinates; they are the bounds the plan carries, impacts included. The distance is
    infinite and the message is left to the caller when a group has no probability.
    """
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    groups = np.asarray(law.group)
    count = int(groups.max()) + 1 if len(groups) else 0
    if count == 0:
        raise ValueError("a scenario without groups has no price to estimate")
    log_box = log_box_probabilities(law.means, law.covs, low, high, rng, samples)
    log_weight = np.asarray(law.log_weight, dtype=float)
    per_group = np.full(count, -np.inf)
    for g in range(count):
        rows = groups == g
        if not rows.any():
            continue
        per_group[g] = logsumexp(log_weight[rows] + log_box[rows]) - np.log(rows.sum())
    if np.any(np.isneginf(per_group)):
        return float("inf"), float("-inf"), float("inf")
    terms = -per_group
    distance = float(terms.mean())
    standard_error = float(terms.std(ddof=1) / np.sqrt(count)) if count > 1 else 0.0
    log_probability = float(logsumexp(per_group) - np.log(count))
    return distance, log_probability, standard_error
