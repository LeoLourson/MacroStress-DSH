"""The observation cells of the backtest and their scores: CRPS, calibration, sharpness.

A cell is one thing the forecast is judged on: an episode ``e``, a month ``t`` of its horizon
and a quantity ``q`` of any level — a macro variable or a driver. The cell holds the fact
``y`` and the ``R`` draws of the forecast; every score below is computed from them.

CRPS is the integral of the squared distance between the forecast distribution and the fact;
the sample estimator over the draws is ``E|X - y| - E|X - X'| / 2``, which penalises both bias
and needless width. To compare quantities measured in different units, CRPS is divided by
``sigma_q(t)`` — the typical movement of the quantity over ``t`` months, the median absolute
change over the history (section 5 of the scheme). Calibration reads the probability
integral transform ``u = F(y)``: with a calibrated forecast the fact falls below p5, between
p5 and p50, between p50 and p95 and above p95 in shares 0.05 / 0.45 / 0.45 / 0.05. Sharpness
is the median width of the ``[p5, p95]`` band, meaningful only once calibration passes.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

BUCKETS = (5.0, 50.0, 95.0)
"""The quantile edges of the calibration band, in percent."""
BUCKET_EDGES = (0.05, 0.50, 0.95)
"""The same edges as shares, where the probability integral transform falls."""
BUCKET_SHARES = (0.05, 0.45, 0.45, 0.05)
"""Where a calibrated forecast puts the facts: below p5, p5..p50, p50..p95, above p95."""
ADMISSION_SHARE = 0.05
"""A version is admitted when fewer than this share of cells break each admission rule."""


@dataclass(frozen=True)
class Cell:
    """One observation the forecast is judged on: episode, month, quantity, fact and draws."""

    episode: str
    month: int
    """The month of the horizon, 1..H."""
    quantity: str
    fact: float
    draws: np.ndarray
    """(R,) the forecast of the quantity in the model scale."""
    constrained: bool = False
    """The scenario fixes this cell exactly; it is reported, never scored."""
    baseline: np.ndarray | None = None
    """(R,) the random walk with drift on the same cell, scored alongside."""
    naive: np.ndarray | None = None
    """(R,) the naive forecast (a year earlier with the trend), for target quantities."""


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
    """``F(y)`` of the sample distribution: the share of draws below the fact, ties halved."""
    x = np.asarray(draws, dtype=float)
    if not len(x):
        return float("nan")
    below = np.sum(x < fact)
    tied = np.sum(x == fact)
    return float((below + 0.5 * tied) / len(x))


def quantiles(draws: np.ndarray) -> np.ndarray:
    """``[p5, p50, p95]`` of the sample distribution."""
    return np.percentile(np.asarray(draws, dtype=float), BUCKETS, axis=0)


def typical_movement(history: np.ndarray, months: int) -> float:
    """``sigma_q(t)``: the median absolute change of the history over ``months`` months.

    ``history`` is the quantity in the model scale, one value a month, NaN where unknown.
    """
    y = np.asarray(history, dtype=float)
    if len(y) <= months:
        return float("nan")
    changes = np.abs(y[months:] - y[:-months])
    changes = changes[np.isfinite(changes)]
    return float(np.median(changes)) if len(changes) else float("nan")


def sigma_of_month(cells: list[Cell], history: np.ndarray) -> dict[int, float]:
    """``sigma_q(t)`` for every month the cells name, from the history in the model scale."""
    return {
        month: typical_movement(history, month)
        for month in sorted({cell.month for cell in cells})
    }


def score_cells(cells: list[Cell], sigma: Mapping[int, float]) -> dict:
    """The scores of the cells of one quantity: CRPS, calibration and sharpness.

    A constrained cell is counted but never scored: the scenario fixes it exactly, so it says
    nothing about the forecast. A cell's ``baseline`` and ``naive`` draws, when present, are
    scored on the same cell alongside the forecast.
    """
    scored = [cell for cell in cells if not cell.constrained and np.isfinite(cell.fact)]
    constrained = [cell for cell in cells if cell.constrained]
    if not scored:
        return {
            "cells": len(cells),
            "constrained": len(constrained),
            "scored": 0,
            "crps": None,
            "baseline_crps": None,
            "normalized_crps": None,
            "calibration": None,
            "sharpness": None,
            "months": [],
        }
    rows = []
    buckets = [0, 0, 0, 0]
    for cell in scored:
        band = quantiles(cell.draws)
        u = pit(cell.draws, cell.fact)
        place = int(np.searchsorted(BUCKET_EDGES, u, side="right"))
        buckets[place] += 1
        movement = sigma.get(cell.month, float("nan"))
        row = {
            "episode": cell.episode,
            "month": cell.month,
            "crps": crps(cell.draws, cell.fact),
            "normalized": (
                crps(cell.draws, cell.fact) / movement
                if np.isfinite(movement) and movement > 0
                else None
            ),
            "p05": float(band[0]),
            "p50": float(band[1]),
            "p95": float(band[2]),
            "pit": u,
            "width": float(band[2] - band[0]),
        }
        if cell.baseline is not None:
            row["baseline_crps"] = crps(cell.baseline, cell.fact)
            row["baseline_normalized"] = (
                row["baseline_crps"] / movement
                if np.isfinite(movement) and movement > 0
                else None
            )
        if cell.naive is not None:
            row["naive_crps"] = crps(cell.naive, cell.fact)
            row["naive_normalized"] = (
                row["naive_crps"] / movement
                if np.isfinite(movement) and movement > 0
                else None
            )
        rows.append(row)
    count = len(rows)
    normalized = [row["normalized"] for row in rows if row["normalized"] is not None]
    baseline = [row["baseline_crps"] for row in rows if "baseline_crps" in row]
    baseline_normalized = [
        row["baseline_normalized"]
        for row in rows
        if row.get("baseline_normalized") is not None
    ]
    naive = [row["naive_crps"] for row in rows if "naive_crps" in row]
    naive_normalized = [
        row["naive_normalized"] for row in rows if row.get("naive_normalized") is not None
    ]
    return {
        "cells": len(cells),
        "constrained": len(constrained),
        "scored": count,
        "crps": float(np.mean([row["crps"] for row in rows])),
        "baseline_crps": float(np.mean(baseline)) if baseline else None,
        "normalized_crps": float(np.mean(normalized)) if normalized else None,
        "baseline_normalized_crps": float(np.mean(baseline_normalized))
        if baseline_normalized
        else None,
        "naive_crps": float(np.mean(naive)) if naive else None,
        "naive_normalized_crps": float(np.mean(naive_normalized)) if naive_normalized else None,
        "calibration": {
            "shares": [share / count for share in buckets],
            "expected": list(BUCKET_SHARES),
        },
        "sharpness": float(np.median([row["width"] for row in rows])),
        "months": rows,
    }


def bootstrap(
    values: list[tuple[str, float]], repetitions: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """The 95% interval of the mean of ``values``, resampling the episodes with replacement.

    ``values`` maps an episode to its mean score; the cells of one episode are dependent, so
    the resampling unit is the episode, not the cell.
    """
    if not values:
        return float("nan"), float("nan")
    means = np.array([value for _, value in values], dtype=float)
    rng = np.random.default_rng(seed)
    drawn = rng.choice(means, size=(repetitions, len(means)), replace=True).mean(axis=1)
    low, high = np.percentile(drawn, [2.5, 97.5])
    return float(low), float(high)
