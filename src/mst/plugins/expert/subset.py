"""Subset simulation (Au and Beck, 2001): a small box probability, level by level.

When the causes name shocks without the residual, the constraint functionals given ``R``
are a deterministic nonlinear function of the named shocks' standard normal inputs, and
their box probability can be far below what a plain Monte Carlo of a few thousand draws
sees. Subset simulation estimates it by a Markov chain in the standard normal space:
every level keeps the samples closest to the box and seeds chains from them, so each level
needs only a conditional probability of about ``p0``, and the estimate is ``p0^m`` times
the final share, with ``m`` the levels completed.

The sampler is the modified Metropolis of Au and Beck: every component is proposed from a
uniform of unit spread and accepted by the ratio of the standard normal densities, and a
step that leaves the level's region is rejected whole, so the product measure of the
independent inputs survives. The chains are batched: one pass of the performance function
covers every chain of a step.
"""

from dataclasses import dataclass

import numpy as np

#: The conditional probability every level asks for.
LEVEL_PROBABILITY = 0.1
#: The most levels before the estimate gives up: ``p0 ** levels`` of probability.
MAX_LEVELS = 20


@dataclass(frozen=True)
class SubsetResult:
    """What one subset run measured: the log probability, its spread and the samples.

    ``inside`` holds the standard normal inputs whose performance function is not positive
    — the samples inside the box — in rows; ``log_probability`` is ``-inf`` when none were
    found within the levels, and ``coefficient_of_variation`` is the spread of the
    estimate as a share of itself, read from the chains (Au and Beck, correlated chains).
    """

    log_probability: float
    coefficient_of_variation: float
    inside: np.ndarray


def subset(
    function,
    dimension: int,
    *,
    samples: int,
    rng,
    p0: float = LEVEL_PROBABILITY,
    levels: int = MAX_LEVELS,
) -> SubsetResult:
    """Estimate ``log P(function(x) <= 0)`` for ``x`` standard normal, ``dimension`` wide.

    ``function`` maps a batch ``(N, dimension)`` of inputs to their violations ``(N,)``,
    positive outside the box; it is called once per level and once per chain step, so a
    batched implementation keeps the whole run cheap.
    """
    seeds = max(2, int(round(p0 * samples)))
    steps = max(1, int(round(1.0 / p0)))
    x = rng.standard_normal((samples, dimension))
    g = function(x)
    done = 0
    for _ in range(levels):
        count = int((g <= 0).sum())
        if count >= seeds:
            return _reached(x, g, count, samples, done, p0, seeds)
        # The level's threshold: the seeds are the ``seeds`` samples closest to the box.
        order = np.argsort(g, kind="stable")[:seeds]
        threshold = float(g[order][-1])
        done += 1
        current = x[order].copy()
        current_g = g[order]
        pool = [current]
        pool_g = [current_g]
        for _ in range(steps - 1):
            proposal = current + rng.uniform(-1.0, 1.0, current.shape)
            accept = rng.random(current.shape) < np.exp(
                0.5 * (current**2 - proposal**2)
            )
            candidate = np.where(accept, proposal, current)
            candidate_g = function(candidate)
            keep = candidate_g <= threshold
            current = np.where(keep[:, None], candidate, current)
            current_g = np.where(keep, candidate_g, current_g)
            pool.append(current)
            pool_g.append(current_g)
        x = np.concatenate(pool)
        g = np.concatenate(pool_g)
    return SubsetResult(
        log_probability=float("-inf"),
        coefficient_of_variation=float("inf"),
        inside=np.empty((0, dimension)),
    )


def _reached(x, g, count, samples, done, p0, seeds) -> SubsetResult:
    """The estimate of the level whose share reached ``p0``: ``p0^done`` times the share.

    The spread reads the chains at the final level (the direct level has no chains, so its
    share is binomial); the samples inside the box are what the imposition resamples.
    """
    share = count / samples
    log_probability = float(done * np.log(p0) + np.log(share))
    inside = x[g <= 0]
    if done == 0:
        cov = float(np.sqrt((1.0 - share) / (share * samples))) if share > 0 else float("inf")
    else:
        # The pool is laid out (step, chain): the states of one chain sit ``seeds`` apart.
        states = (g <= 0).reshape(-1, seeds).mean(axis=0)
        cov = (
            float(np.std(states, ddof=1) / (seeds**0.5 * share))
            if seeds > 1 and share > 0
            else float("inf")
        )
    return SubsetResult(
        log_probability=log_probability,
        coefficient_of_variation=cov,
        inside=inside,
    )
