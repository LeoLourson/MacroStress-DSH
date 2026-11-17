"""The observation cells of the backtest and the numbers read from them: CRPS, PIT, bands.

A cell is one thing the forecast is judged on: a window ``e`` — an episode or a date of the
rolling grid — a month ``t`` of its horizon and a quantity ``q`` of any level. The cell holds
the fact ``y``, the ``R`` draws of the forecast and the draws of every comparison on the same
cell: the random walk with drift (``baseline``), the walk that knows the scenario's path
(``conditional``), the naive forecast of a target (``naive``) and the unconstrained run of the
model (``base``). It also holds the range of the quantity's changes over as many months as
the cell is ahead of the last known value, and the admissible domain, which the admission rules
read.

CRPS is the integral of the squared distance between the forecast distribution and the fact;
the sample estimator over the draws is ``E|X - y| - E|X - X'| / 2``, which penalises both bias
and needless width. It keeps the unit of the quantity, so two quantities are compared only
through ratios to a benchmark on the same cells (``mst.metrics.quality``). Calibration reads the
probability integral transform ``u = F(y)``. The implementation uses the mid-PIT
``(below + tied / 2) / R``: for continuous draws that is ``F(y)``, but a forecast with an atom at
the fact is not uniform, so the bucket shares 0.05 / 0.45 / 0.45 / 0.05 are the target for a
continuous forecast only. Sharpness is the median width of the ``[p5, p95]`` band, meaningful
only once calibration passes.

``typical_movement`` — the median absolute change of a history — is not a score here: the
scenario reads it as the tolerance of a point it fixes (``mst.scenario.build``).
"""

from dataclasses import dataclass

import numpy as np

BUCKETS = (5.0, 50.0, 95.0)
"""The quantile edges of the calibration band, in percent."""
BUCKET_EDGES = (0.05, 0.50, 0.95)
"""The same edges as shares, where the probability integral transform falls."""
MIN_MOVES = 200
"""The fewest known changes of one length the moves rule reads (``move_ranges``).

The range of a few changes is narrower than a calibrated 90 % band: for a Student walk with
four degrees of freedom, a band right by construction breaks the rule in 7 % of its cells
with 120 monthly values and in 27-66 % with annual ones. With two hundred changes of each
length the share is under 0.5 %."""


@dataclass(frozen=True)
class Cell:
    """One observation the forecast is judged on: window, month, quantity, fact and draws."""

    unit: str
    """The window the cell belongs to: the episode id or the grid date."""
    month: int
    """The month of the horizon, 1..H."""
    quantity: str
    fact: float
    draws: np.ndarray
    """(R,) the forecast of the quantity in the model scale."""
    level: str = "macro"
    """The level the quantity belongs to: ``macro``, ``driver/<quantity>`` and further."""
    constrained: bool = False
    """The scenario fixes this cell exactly; it is reported, never scored."""
    baseline: np.ndarray | None = None
    """(R,) the random walk with drift on the same cell, scored alongside."""
    naive: np.ndarray | None = None
    """(R,) the naive forecast (a year earlier with the trend), for target quantities."""
    base: np.ndarray | None = None
    """(R,) the unconstrained run of the model, the second comparison of the scheme."""
    conditional: np.ndarray | None = None
    """(R,) the walk that knows the path the scenario fixes: the like-for-like benchmark."""
    historical: np.ndarray | None = None
    """The historical simulation of a return: its values over the past windows of the same
    length the calculation date knows. The benchmark of a position and of the portfolio."""
    move_range: float = float("nan")
    """The range of the history's changes over the months from the last known value to the
    cell's month (``move_ranges``); for a return, the range of the past returns of the same
    length (``return_range``). The moves rule reads it."""
    domain: tuple[float, float] | None = None
    """The admissible range in the model scale; None when unbounded."""
    unverified: bool = False
    """The forecast left the range of the model's own training history: an unverified tail."""


def crps(draws: np.ndarray, fact: float) -> float:
    """The continuous ranked probability score of ``draws`` against ``fact``.

    ``E|X - X'|`` over all pairs is computed from the sorted draws and their prefix sums, so
    the cost is ``R log R``, not ``R^2``.
    """
    x = np.sort(np.asarray(draws, dtype=float))
    if not len(x):
        return float("nan")
    count = len(x)
    first = np.abs(x - fact).mean()
    ranks = np.arange(count)
    # sum_i sum_j |x_i - x_j| = 2 * sum_i (i * x_(i) - prefix(x_0 .. x_(i-1))).
    prefix = np.concatenate([[0.0], np.cumsum(x)[:-1]])
    pairwise = 2.0 * float(np.sum(ranks * x - prefix)) / count**2
    return float(first - pairwise / 2)


def pit(draws: np.ndarray, fact: float) -> float:
    """The mid-PIT of the sample distribution: the share below the fact, ties halved."""
    x = np.asarray(draws, dtype=float)
    if not len(x):
        return float("nan")
    below = np.sum(x < fact)
    tied = np.sum(x == fact)
    return float((below + 0.5 * tied) / len(x))


def quantiles(draws: np.ndarray) -> np.ndarray:
    """``[p5, p50, p95]`` of the sample distribution."""
    return np.percentile(np.asarray(draws, dtype=float), BUCKETS, axis=0)


def finite_draws(draws: np.ndarray) -> np.ndarray:
    """The finite draws of a cell; a non-finite draw says nothing about the forecast."""
    x = np.asarray(draws, dtype=float)
    return x[np.isfinite(x)]


def typical_movement(history: np.ndarray, months: int) -> float:
    """``sigma_q(t)``: the median absolute change of the history over ``months`` months.

    ``history`` is the quantity in the model scale, one value a month, NaN where unknown.
    A zero median — a history that does not move over ``months`` months — is no scale either,
    so it is NaN as well.
    """
    y = np.asarray(history, dtype=float)
    if len(y) <= months:
        return float("nan")
    changes = np.abs(y[months:] - y[:-months])
    changes = changes[np.isfinite(changes)]
    if not len(changes):
        return float("nan")
    sigma = float(np.median(changes))
    return sigma if sigma > 0 else float("nan")


def return_range(past: np.ndarray) -> float:
    """The range of the finite past returns of one length; NaN with fewer than ``MIN_MOVES``.

    A return already is a change over the cell's months, so its past values are the moves the
    moves rule compares the band with, under the same least count."""
    known = np.asarray(past, dtype=float)
    known = known[np.isfinite(known)]
    if len(known) < MIN_MOVES:
        return float("nan")
    return float(known.max() - known.min())


def move_ranges(history: np.ndarray, horizon: int) -> np.ndarray:
    """(horizon,) per month ``t`` of the horizon, the range of the history's changes over the
    months between the last known value and ``t``; NaN where fewer than ``MIN_MOVES`` changes of
    that length are known.

    The band of month ``t`` is a forecast of the change from the last known value, so its
    yardstick is how far the quantity moved over as many months, not the range of its levels:
    a mean-reverting series moves less over a year than a walk would, and a short history has
    seen many moves of a year but few of its levels. A series last known two months before the
    end of the history is three months from month 1. The changes are counted per length because
    a series can change its frequency: one annual until 2020 and quarterly since has many
    twelve-month changes but only the few three-month ones after the switch.
    """
    y = np.asarray(history, dtype=float)
    found = np.full(horizon, np.nan)
    seen = np.flatnonzero(np.isfinite(y))
    if not len(seen):
        return found
    gap = len(y) - 1 - seen[-1]
    for t in range(horizon):
        steps = t + 1 + gap
        moves = y[steps:] - y[:-steps]
        known = moves[np.isfinite(moves)]
        if len(known) >= MIN_MOVES:
            found[t] = known.max() - known.min()
    return found
