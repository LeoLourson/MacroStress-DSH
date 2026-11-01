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

    \\mathrm{dist} = -\\frac{1}{K} \\sum_g \\log \\Big( \\frac{1}{n_g} \\sum_{i \\in g} e^{\\log w_i} P_i \\Big),

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


class ContradictsModels(ValueError):
    """Some group of the law gives the event no probability: the scenario cannot be imposed."""

    def __init__(self):
        super().__init__(
            "the scenario contradicts the models: at some value of R the event has no "
            "probability, the case lemma 2 of the canon names"
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
