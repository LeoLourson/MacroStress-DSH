"""The admission rules of section 5 and the baselines the scores are compared with.

A version is admitted when fewer than five percent of its cells break each of two rules: the
``[p5, p95]`` band lies inside the admissible domain of the quantity, and the width of the
band does not exceed the range of the quantity over its whole history. A scenario also has to
be reachable: the share of the *base* run's paths that satisfy each constraint must be above
one percent, or the scenario rests on a rare tail of the distribution.

The baseline of every quantity is a random walk with drift — the last value carried by the
average monthly change, with the variance of the monthly changes; a target quantity is also
compared with the naive forecast, the value a year earlier carried by the same trend.
"""

import numpy as np

DOMAIN = "band_outside_domain"
WIDTH = "band_wider_than_history"
CONSTRAINT = "constraint_probability_below_one_percent"


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
) -> dict[int, float]:
    """The share of ``paths`` that satisfy each constraint, numbered from zero.

    ``paths`` is ``(R, H + 1, n)`` in the model scale; a constraint's terms name variables and
    months of the horizon.
    """
    index = {id_: k for k, id_ in enumerate(ids)}
    shares = {}
    for number, constraint in enumerate(constraints):
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
    come from its monthly changes. ``months`` is the step of the walk — the baseline of the
    naive forecast moves a year at a time.
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
    """The naive forecast of a target: the value a year earlier, carried by the same trend.

    Returns None where the history is shorter than a year. The noise is the spread of the
    twelve-month changes of the history.
    """
    y = np.asarray(history, dtype=float)
    if len(y) < 13:
        return None
    changes = y[12:] - y[:-12]
    finite = changes[np.isfinite(changes)]
    if not len(finite):
        return None
    drift, noise = float(np.mean(finite)) / 12, float(np.std(finite)) / np.sqrt(12)
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift, noise, size=(draws, horizon))
    start = np.where(np.isfinite(y[-12]), y[-12], y[-1])
    return start + np.cumsum(steps, axis=1)
