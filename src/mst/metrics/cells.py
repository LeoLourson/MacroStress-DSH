"""The observation cells of the backtest and their scores: CRPS, calibration, sharpness.

A cell is one thing the forecast is judged on: an episode ``e``, a month ``t`` of its horizon
and a quantity ``q`` of any level — a macro variable or a driver. The cell holds the fact
``y``, the ``R`` draws of the forecast and the numbers its scores need: the random walk with
drift on the same cell (``baseline``), the naive forecast of a target (``naive``), the
unconstrained run of the model (``base``), the typical movement ``sigma_q(t)`` and the range of
the quantity's history, the last two as known at the episode's calculation date. Admission also
reads the admissible domain of the quantity.

CRPS is the integral of the squared distance between the forecast distribution and the fact;
the sample estimator over the draws is ``E|X - y| - E|X - X'| / 2``, which penalises both bias
and needless width. To compare quantities measured in different units, CRPS is divided by
``sigma_q(t)`` — the typical movement of the quantity over ``t`` months, the median absolute
change over the history (section 5 of the scheme). Calibration reads the probability integral
transform ``u = F(y)``. The implementation uses the mid-PIT ``(below + tied / 2) / R``: for
continuous draws that is ``F(y)``, but a forecast with an atom at the fact is not uniform, so
the bucket shares 0.05 / 0.45 / 0.45 / 0.05 are the target for a continuous forecast only.
Sharpness is the median width of the ``[p5, p95]`` band, meaningful only once calibration
passes; coverage is reported for reference only, because it grows with width.

The scores are per quantity. The equal-weight aggregate across quantities, the paired decision
rule of section 5 and the tail statistics (Acerbi-Székely ``Z``, ES) are not in this module:
they live in ``aggregate``, ``decision``, ``es`` and the runner ``backtest``.
"""

from collections.abc import Mapping, Sequence
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
    base: np.ndarray | None = None
    """(R,) the unconstrained run of the model, the second comparison of the scheme."""
    sigma: float = float("nan")
    """``sigma_q(t)`` from the history known at the episode's calculation date."""
    sigma_source: str = "none"
    """Where ``sigma`` came from: ``t`` its own month, ``t*``/``t+`` a neighbour, ``pooled``
    the peers of the level, ``none`` no scale at all (``mst.metrics.cells.sigma_with_fallback``)."""
    sigma_pairs: int = 0
    """How many finite pairs the value of ``sigma`` rests on."""
    history_range: float = float("nan")
    """The range of the quantity's history known at the same date; the width rule reads it."""
    domain: tuple[float, float] | None = None
    """The admissible range in the model scale; None when unbounded."""


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
    A zero median — a history that does not move over ``months`` months — is no scale either:
    the normalisation it feeds would be degenerate, so it is NaN as well.
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


def movement_pairs(history: np.ndarray, months: int) -> int:
    """How many finite pairs the absolute change over ``months`` months rests on."""
    y = np.asarray(history, dtype=float)
    if len(y) <= months:
        return 0
    return int(np.isfinite(y[months:] - y[:-months]).sum())


def sigma_with_fallback(
    history: np.ndarray,
    months_list: Sequence[int],
    *,
    peers: dict[str, np.ndarray] | None = None,
) -> dict[int, tuple[float, str, int]]:
    """``sigma_q(t)`` for every month, with a fallback when the history gives no scale.

    A month whose own ``sigma_q(t)`` is valid — finite and positive — is the source ``"t"``.
    Otherwise the scale is borrowed from the nearest month of ``months_list`` that has one:
    the largest ``t*`` below (``"t*"``), then the smallest ``t+`` above (``"t+"``), then the
    median of the peers' ``sigma_r(t)`` over the valid peer scales — the histories of the
    same level's quantities (``"pooled"``) — and otherwise ``(NaN, "none", 0)``.

    The third element counts the pairs behind the value: the pairs of the month it came from
    for ``"t"``, ``"t*"`` and ``"t+"``, the number of peer scales for ``"pooled"``, and zero
    when no scale was found. A scale resting on one pair is reported with it rather than
    silently trusted.
    """
    months = sorted({int(t) for t in months_list})
    own = {t: typical_movement(history, t) for t in months}
    pairs = {t: movement_pairs(history, t) for t in months}
    valid = {t: s for t, s in own.items() if np.isfinite(s) and s > 0}
    result: dict[int, tuple[float, str, int]] = {}
    for t in months:
        if t in valid:
            result[t] = (valid[t], "t", pairs[t])
            continue
        below = [u for u in valid if u < t]
        if below:
            best = max(below)
            result[t] = (valid[best], "t*", pairs[best])
            continue
        above = [u for u in valid if u > t]
        if above:
            best = min(above)
            result[t] = (valid[best], "t+", pairs[best])
            continue
        pooled, scales = _pooled_sigma(peers, t)
        if np.isfinite(pooled) and pooled > 0:
            result[t] = (pooled, "pooled", scales)
        else:
            result[t] = (float("nan"), "none", 0)
    return result


def _pooled_sigma(peers: dict[str, np.ndarray] | None, months: int) -> tuple[float, int]:
    """The median of the peers' valid ``sigma_r(months)`` and how many scales it used."""
    if not peers:
        return float("nan"), 0
    scales = []
    for other in peers.values():
        sigma = typical_movement(other, months)
        if np.isfinite(sigma) and sigma > 0:
            scales.append(sigma)
    return (float(np.median(scales)) if scales else float("nan")), len(scales)


def history_range(history: np.ndarray) -> float:
    """The range of the finite values of a history; NaN without a finite value."""
    y = np.asarray(history, dtype=float)
    known = y[np.isfinite(y)]
    return float(known.max() - known.min()) if len(known) else float("nan")


def sigma_of_month(cells: list[Cell], history: np.ndarray) -> dict[int, float]:
    """``sigma_q(t)`` for every month the cells name, from the history in the model scale."""
    return {
        month: typical_movement(history, month)
        for month in sorted({cell.month for cell in cells})
    }


def score_cells(cells: list[Cell]) -> dict:
    """The scores of the cells of one quantity: CRPS, calibration and sharpness.

    A constrained cell is counted but never scored: the scenario fixes it exactly, so it says
    nothing about the forecast. A cell's ``baseline``, ``naive`` and ``base`` draws, when
    present, are scored on the same cell alongside the forecast; ``difference`` is the model's
    normalised CRPS less the random walk's, the paired quantity the decision reads. A cell
    whose draws are entirely non-finite is counted in ``dropped`` and left out.
    """
    scored = [cell for cell in cells if not cell.constrained and np.isfinite(cell.fact)]
    constrained = [cell for cell in cells if cell.constrained]
    if not scored:
        return _empty_scores(len(cells), len(constrained))
    rows = []
    buckets = [0, 0, 0, 0]
    dropped = 0
    for cell in scored:
        draws = finite_draws(cell.draws)
        if not len(draws):
            dropped += 1
            continue
        band = quantiles(draws)
        u = pit(draws, cell.fact)
        buckets[int(np.searchsorted(BUCKET_EDGES, u, side="right"))] += 1
        movement = cell.sigma
        known_scale = bool(np.isfinite(movement) and movement > 0)
        value = crps(draws, cell.fact)
        row = {
            "episode": cell.episode,
            "month": cell.month,
            "crps": value,
            "normalized": value / movement if known_scale else None,
            "sigma_source": cell.sigma_source,
            "sigma_pairs": cell.sigma_pairs,
            "p05": float(band[0]),
            "p50": float(band[1]),
            "p95": float(band[2]),
            "pit": u,
            "coverage": bool(band[0] <= cell.fact <= band[2]),
            "width": float(band[2] - band[0]),
        }
        for name, column in (
            ("baseline", cell.baseline),
            ("base", cell.base),
            ("naive", cell.naive),
        ):
            if column is None:
                continue
            known = finite_draws(column)
            if not len(known):
                continue
            score = crps(known, cell.fact)
            row[f"{name}_crps"] = score
            row[f"{name}_normalized"] = score / movement if known_scale else None
        baseline = row.get("baseline_normalized")
        if baseline is not None and row["normalized"] is not None:
            row["difference"] = row["normalized"] - baseline
        rows.append(row)
    count = len(rows)
    if not count:
        return _empty_scores(len(cells), len(constrained), dropped=dropped)
    normalized = [row["normalized"] for row in rows if row["normalized"] is not None]
    baseline = [row["baseline_crps"] for row in rows if row.get("baseline_crps") is not None]
    baseline_normalized = [
        row["baseline_normalized"]
        for row in rows
        if row.get("baseline_normalized") is not None
    ]
    base = [row["base_crps"] for row in rows if row.get("base_crps") is not None]
    base_normalized = [
        row["base_normalized"] for row in rows if row.get("base_normalized") is not None
    ]
    naive = [row["naive_crps"] for row in rows if row.get("naive_crps") is not None]
    naive_normalized = [
        row["naive_normalized"] for row in rows if row.get("naive_normalized") is not None
    ]
    return {
        "cells": len(cells),
        "constrained": len(constrained),
        "scored": count,
        "dropped": dropped,
        "normalized_cells": len(normalized),
        "crps": float(np.mean([row["crps"] for row in rows])),
        "base_crps": float(np.mean(base)) if base else None,
        "baseline_crps": float(np.mean(baseline)) if baseline else None,
        "normalized_crps": float(np.mean(normalized)) if normalized else None,
        "base_normalized_crps": float(np.mean(base_normalized)) if base_normalized else None,
        "baseline_normalized_crps": float(np.mean(baseline_normalized))
        if baseline_normalized
        else None,
        "naive_crps": float(np.mean(naive)) if naive else None,
        "naive_normalized_crps": float(np.mean(naive_normalized)) if naive_normalized else None,
        "coverage": float(np.mean([row["coverage"] for row in rows])),
        "calibration": {
            "shares": [share / count for share in buckets],
            "expected": list(BUCKET_SHARES),
        },
        "sharpness": float(np.median([row["width"] for row in rows])),
        "months": rows,
    }


def _empty_scores(cells: int, constrained: int, *, dropped: int = 0) -> dict:
    return {
        "cells": cells,
        "constrained": constrained,
        "scored": 0,
        "dropped": dropped,
        "normalized_cells": 0,
        "crps": None,
        "base_crps": None,
        "baseline_crps": None,
        "normalized_crps": None,
        "base_normalized_crps": None,
        "baseline_normalized_crps": None,
        "naive_crps": None,
        "naive_normalized_crps": None,
        "coverage": None,
        "calibration": None,
        "sharpness": None,
        "months": [],
    }


def bootstrap(
    samples: Mapping[str, Sequence[float]], repetitions: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """The 95% interval of the mean over cells, resampling the episodes with replacement.

    ``samples`` maps an episode to the per-cell values it contributes. The resampled statistic
    is the same cell-weighted mean the report shows, not the mean of the episode means: the
    cells of one episode are dependent, so the episodes are the resampling unit, while the
    weights stay those of the report.
    """
    groups = []
    for values in samples.values():
        known = np.asarray(values, dtype=float)
        known = known[np.isfinite(known)]
        if len(known):
            groups.append(known)
    if not groups:
        return float("nan"), float("nan")
    totals = np.array([group.sum() for group in groups])
    sizes = np.array([len(group) for group in groups], dtype=float)
    rng = np.random.default_rng(seed)
    drawn = rng.integers(0, len(groups), size=(repetitions, len(groups)))
    counts = np.stack(
        [
            (drawn == column).sum(axis=1).astype(float)
            for column in range(len(groups))
        ],
        axis=1,
    )
    means = (counts * totals).sum(axis=1) / (counts * sizes).sum(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)
