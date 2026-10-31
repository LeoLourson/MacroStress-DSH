"""The wild cluster bootstrap with Webb weights: the check of every few-cluster interval.

The quality metrics take the 95 % interval from an episode bootstrap, but with few
episodes the ordinary cluster bootstrap understates the interval: resampling ``n`` clusters
with replacement cannot break a dominant cluster. The wild cluster bootstrap (Cameron &
Miller 2015) keeps every cluster in every repetition and instead reweights its centred
contribution by an independent weight; the Webb weights ``±sqrt(3/2), ±sqrt(2/2), ±sqrt(1/2)``
with probability 1/6 each keep the variance of the weighted statistic equal to the variance of
the original one and work with as few as one cluster per arm.

The statistic here is a mean over the cells, and the clusters are the episodes: the cells of
one episode are dependent, so its centred contribution is reweighted as a whole. For a plain
mean the null is the point estimate itself; for a paired difference — the model less the
random walk — the null is zero, because the bootstrap answers whether the difference is
distinguishable from nothing.
"""

import numpy as np

WEBB_WEIGHTS = np.array(
    [
        np.sqrt(3.0 / 2.0),
        -np.sqrt(3.0 / 2.0),
        np.sqrt(2.0 / 2.0),
        -np.sqrt(2.0 / 2.0),
        np.sqrt(1.0 / 2.0),
        -np.sqrt(1.0 / 2.0),
    ]
)
"""The Webb weights, each with probability 1/6: mean zero, variance one."""

WILD_REPETITIONS = 1000
"""The repetitions of the wild cluster bootstrap behind every reported wild interval."""


def wild_interval(
    contributions: dict[str, float],
    *,
    repetitions: int = WILD_REPETITIONS,
    seed: int,
    null: float | None = None,
) -> tuple[float, float]:
    """The 95 % wild cluster bootstrap interval of the mean over the episode contributions.

    ``contributions`` maps an episode to the mean of its cells — the same per-episode mean the
    cluster bootstrap resamples. ``null`` is the value the statistic is tested against: the
    point estimate for a plain mean, zero for a paired difference. Every episode stays in every
    repetition; its centred contribution is multiplied by an independent Webb weight, and the
    2.5/97.5 percentiles of the reweighted statistic are the interval.
    """
    if not contributions:
        return float("nan"), float("nan")
    values = np.asarray(list(contributions.values()), dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    centre = float(np.mean(values))
    mu = centre if null is None else null
    rng = np.random.default_rng([seed, 11])
    weights = WEBB_WEIGHTS[rng.integers(0, len(WEBB_WEIGHTS), size=(repetitions, len(values)))]
    statistics = mu + (weights * (values - mu)[None, :]).mean(axis=1)
    low, high = np.percentile(statistics, [2.5, 97.5])
    return float(low), float(high)


def disagrees(pair: tuple[float, float], wild: tuple[float, float]) -> bool:
    """Whether the two intervals disagree about the significance of a difference from zero.

    The verdicts diverge when zero is inside one interval and outside the other: the pair
    bootstrap finds a difference the wild bootstrap does not confirm, or the other way round.
    """
    return (pair[0] <= 0.0 <= pair[1]) != (wild[0] <= 0.0 <= wild[1])
