"""The admission rules and the baselines the scores are compared with.

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

The conditional walk is the benchmark of a conditional forecast: the walk also regresses the
monthly change of the quantity on the changes of the variables the scenario fixes, and reads
their realized path, so it knows what the model knows and has only the links of the history
to add. Both walks start from the last known value of the quantity, however many months before
the first month of the horizon it lies, and read a floor of the registry the way the model's
reported paths do.
"""

import numpy as np

DOMAIN = "band_outside_domain"
WIDTH = "band_wider_than_history"


def macro_domain(variable) -> tuple[float, float] | None:
    """The admissible range of a macro variable in the model scale; None when unbounded.

    Rates and unemployment are non-negative. A ``log100`` value is ``100 ln X``, any real
    number for a positive ``X``, and a variable registered as ``negative`` has been below
    nought in its history. The upper side is unbounded.
    """
    if variable.transform == "log100" or variable.negative:
        return None
    return (0.0, float("inf"))


def quantity_domain(declared) -> tuple[float, float] | None:
    """The admissible range of a driver quantity in the model scale; None when unbounded.

    A declared domain is mapped through the transform, its natural endpoints becoming the
    natural boundaries instead of disabling the check: both ``log100`` and the ``logit`` map
    a boundary they never reach to an infinity, so ``log100`` sends a price of 0 to ``-inf``
    rather than requiring a price above one. Without a declared domain every transform is
    unbounded.
    """
    declared_domain = declared.domain
    if declared_domain is not None and len(declared_domain) == 2:
        low = _endpoint(declared.transform, float(declared_domain[0]))
        high = _endpoint(declared.transform, float(declared_domain[1]))
        if low == -float("inf") and high == float("inf"):
            return None
        return (low, high)
    return None


def _endpoint(transform: str, value: float) -> float:
    """A domain endpoint in the model scale; the natural infinity where the transform diverges."""
    if transform == "logit":
        if value <= 0.0:
            return -float("inf")
        if value >= 1.0:
            return float("inf")
        return float(np.log(value / (1.0 - value)))
    if transform == "log100":
        if value <= 0.0:
            return -float("inf")
        return 100.0 * float(np.log(value))
    return value
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
    months of the horizon. A constraint that fixes a point — the platform expanded it to
    ``value ± tau`` — is not applicable: its share would measure ``tau`` rather than the
    scenario, so it is reported as ``None`` and left out of the gate.
    """
    index = {id_: k for k, id_ in enumerate(ids)}
    shares: dict[int, float | None] = {}
    for number, constraint in enumerate(constraints):
        if getattr(constraint, "point", False):
            shares[number] = None
            continue
        total = np.zeros(paths.shape[0])
        for term in constraint.terms:
            total += term.weight * paths[:, term.month, index[term.variable]]
        shares[number] = float(np.mean((constraint.low <= total) & (total <= constraint.high)))
    return shares


def random_walk(
    history: np.ndarray,
    horizon: int,
    *,
    draws: int,
    seed: int,
    months: int = 1,
    lower: float | None = None,
) -> np.ndarray:
    """A random walk with drift: the last value plus the trend and the noise of the changes.

    ``history`` is the quantity in the model scale, one value a month; the drift and the noise
    come from the changes between months exactly ``months`` apart on the calendar grid, so a
    gap does not join distant observations — a pair counts only when both of its ends are
    known, and fewer than three pairs leave the walk NaN. ``months`` is the step of the walk.
    A history that ends in NaN walks through its unknown months, so the first month of the
    horizon is as many steps ahead as it really is. ``lower`` is the floor of the quantity.
    """
    y = np.asarray(history, dtype=float)
    changes = y[months:] - y[:-months]
    changes = changes[np.isfinite(changes)]
    if len(changes) < 3:
        return np.full((draws, horizon), np.nan)
    drift, noise = float(np.mean(changes)) / months, float(np.std(changes)) / np.sqrt(months)
    seen = np.flatnonzero(np.isfinite(y))
    gap = len(y) - 1 - int(seen[-1])
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift, noise, size=(draws, horizon + gap))
    if months > 1:
        # A year's move spread over its months, so every horizon month has a draw.
        steps = np.repeat(steps / months, months, axis=1)[:, : horizon + gap]
    return _floored(y[seen[-1]] + np.cumsum(steps, axis=1)[:, gap:], lower)


def conditional_walk(
    history: np.ndarray,
    given_history: np.ndarray,
    given_path: np.ndarray,
    *,
    draws: int,
    seed: int,
    lower: float | None = None,
) -> np.ndarray:
    """The walk that knows the scenario: ``dy = a + b' dc + e``, on the history known then.

    ``history`` is ``(T,)``, ``given_history`` ``(T, k)`` the fixed variables over the same
    months and ``given_path`` ``(H, k)`` their realized path, NaN where unknown. The regression
    runs on the months with every change known, and needs three more of them than it has
    coefficients, or the walk is NaN. The path of the quantity adds up the fitted changes and
    gaussian noise of the residual spread; an unknown change of a fixed variable reads its mean.
    """
    y = np.asarray(history, dtype=float)
    given = np.vstack([np.asarray(given_history, dtype=float), np.asarray(given_path, dtype=float)])
    horizon = len(given_path)
    changes = given[1:] - given[:-1]
    target = y[1:] - y[:-1]
    usable = np.isfinite(target) & np.isfinite(changes[: len(target)]).all(axis=1)
    columns = given.shape[1] + 1
    seen = np.flatnonzero(np.isfinite(y))
    if usable.sum() < columns + 3 or not len(seen):
        return np.full((draws, horizon), np.nan)
    known = changes[: len(target)][usable]
    design = np.column_stack([np.ones(len(known)), known])
    coefficients, *_ = np.linalg.lstsq(design, target[usable], rcond=None)
    residuals = target[usable] - design @ coefficients
    noise = float(np.sqrt(residuals @ residuals / (len(known) - columns)))
    start = int(seen[-1])
    ahead = np.where(np.isfinite(changes[start:]), changes[start:], known.mean(axis=0))
    fitted = coefficients[0] + ahead @ coefficients[1:]
    rng = np.random.default_rng(seed)
    steps = fitted + rng.normal(0.0, noise, size=(draws, len(fitted)))
    return _floored(y[start] + np.cumsum(steps, axis=1)[:, -horizon:], lower)


def _floored(paths: np.ndarray, lower: float | None) -> np.ndarray:
    """The paths read through the floor of the quantity; ``None`` leaves them as they are."""
    return paths if lower is None else np.maximum(paths, lower)


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
