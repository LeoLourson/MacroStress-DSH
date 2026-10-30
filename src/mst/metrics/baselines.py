"""The admission rules of section 5 and the baselines the scores are compared with.

A version is admitted when fewer than five percent of its cells break each of two rules
separately — the shares are pooled over every scored cell, not taken per quantity and not
unioned: the ``[p5, p95]`` band lies inside the admissible domain of the quantity, and the
width of the band does not exceed the range of the quantity over its history. A scenario also
has to be reachable: the share of the *base* run's paths that satisfy each constraint must be
above one percent, or the scenario rests on a rare tail of the distribution. A constraint
reconstructed as an exact point (``low == high``) cannot be met by a continuous draw, so its
probability is not applicable rather than zero; the report says so and leaves it out of the
gate.

The random walk with drift is the benchmark of every quantity, computed on the same cells and
with the same number of draws as the model, from the history known at the episode's calculation
date. The naive forecast of a target month ``T + t`` is the value a year earlier, ``y[T + t -
12]``, carried by ``t`` monthly drifts; the two must not be confused with the unconditional run
of the model itself, which is the other comparison of the scheme.
"""

import numpy as np

DOMAIN = "band_outside_domain"
WIDTH = "band_wider_than_history"
CONSTRAINT = "constraint_probability_below_one_percent"
CONSTRAINT_THRESHOLD = 0.01
"""The share of the base run's paths that must satisfy each informative constraint."""


def band_problems(
    draws: np.ndarray, fact: float, *, domain: tuple[float, float] | None, history_range: float
) -> list[str]:
    """The admission rules a cell breaks: its band against the domain and the history range."""
    p05, _, p95 = np.percentile(np.asarray(draws, dtype=float), [5, 50, 95])
    problems = []
    if domain is not None and (p05 < domain[0] or p95 > domain[1]):
        problems.append(DOMAIN)
    if np.isfinite(history_range) and (p95 - p05) > history_range:
        problems.append(WIDTH)
    return problems


def constraint_probabilities(
    paths: np.ndarray, ids: list[str], constraints
) -> dict[int, float | None]:
    """The share of ``paths`` that satisfy each constraint, numbered from zero.

    ``paths`` is ``(R, H + 1, n)`` in the model scale; a constraint's terms name variables and
    months of the horizon. A constraint whose ``low`` equals its ``high`` fixes a point, which
    a continuous draw meets with probability zero: its share is reported as ``None`` (not
    applicable) instead of a meaningless zero.
    """
    index = {id_: k for k, id_ in enumerate(ids)}
    shares: dict[int, float | None] = {}
    for number, constraint in enumerate(constraints):
        if constraint.low == constraint.high:
            shares[number] = None
            continue
        total = np.zeros(paths.shape[0])
        for term in constraint.terms:
            total += term.weight * paths[:, term.month, index[term.variable]]
        shares[number] = float(np.mean((constraint.low <= total) & (total <= constraint.high)))
    return shares


def random_walk(
    history: np.ndarray, horizon: int, *, draws: int, seed: int, months: int = 1
) -> np.ndarray:
    """A random walk with drift: the last value plus the trend and the noise of the changes.

    ``history`` is the quantity in the model scale, one value a month; the drift and the noise
    come from its monthly changes. ``months`` is the step of the walk.
    """
    y = np.asarray(history, dtype=float)
    known = y[np.isfinite(y)]
    if len(known) < 3:
        return np.full((draws, horizon), np.nan)
    changes = known[months:] - known[:-months]
    drift, noise = float(np.mean(changes)) / months, float(np.std(changes)) / np.sqrt(months)
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift, noise, size=(draws, horizon))
    if months > 1:
        # A year's move spread over its months, so every horizon month has a draw.
        steps = np.repeat(steps / months, months, axis=1)[:, :horizon]
    return known[-1] + np.cumsum(steps, axis=1)


def naive_annual(
    history: np.ndarray, horizon: int, *, draws: int, seed: int
) -> np.ndarray | None:
    """The naive forecast: the year-ago value of every target month plus ``t`` drifts.

    For target month ``T + t`` the year-ago month is ``T + t - 12``; the history ends at ``T``,
    so the base value advances with ``t`` and a horizon beyond twelve months reuses the last
    finite value. Where the year-ago month is missing the last finite value at or before it
    stands in; the deterministic mean is "year-ago value + t * drift". The drift is the mean
    twelve-month change per month and the noise is its spread over the months. Returns None
    when no base value is known.
    """
    y = np.asarray(history, dtype=float)
    if len(y) < 13:
        return None
    changes = y[12:] - y[:-12]
    finite = changes[np.isfinite(changes)]
    if not len(finite):
        return None
    drift, noise = float(np.mean(finite)) / 12, float(np.std(finite)) / np.sqrt(12)
    last = len(y) - 1
    starts = np.empty(horizon)
    for step in range(1, horizon + 1):
        target = min(last + step - 12, last)
        while target >= 0 and not np.isfinite(y[target]):
            target -= 1
        if target < 0:
            return None
        starts[step - 1] = y[target]
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, noise, size=(draws, horizon))
    drift_path = np.arange(1, horizon + 1, dtype=float) * drift
    return starts[None, :] + drift_path[None, :] + np.cumsum(steps, axis=1)
