"""The relative score: how much better a forecast is than its benchmark, free of any scale.

For a quantity at a horizon month the score is the ratio of the mean CRPS of the forecast to
the mean CRPS of the benchmark on the same cells; below one the forecast wins. Ratios are
pooled in logarithms, with equal weights over the quantities at a month and then over the
months, so 0.5 on one quantity and 2 on another cancel out, and a quantity that rarely moves
weighs as much as any other: its small CRPS sits in both parts of its own ratio. The CRPS
divided by the typical movement has no such property, since a cell of a sticky rate with a
near-zero median change can outweigh dozens of others.

The interval resamples whole units of the rows, dates or episodes, in moving blocks
(``mst.metrics.block``): the windows of neighbouring dates overlap, and the cells of one unit
are dependent, so neither may be read as independent draws.
"""

from collections.abc import Mapping, Sequence

import numpy as np

from .block import block_weights

Rows = Sequence[Mapping[str, object]]
REPETITIONS = 1000


def relative(rows: Rows, model: str, benchmark: str) -> dict:
    """The score over ``rows``, with its values per horizon month and per quantity.

    ``model`` and ``benchmark`` name the row keys holding the two CRPS. A ratio needs both on
    the same cell and a positive mean of each; a group without one is left out and counted.
    """
    cube = _cube(rows, (model, benchmark), None)
    if cube is None:
        return {"score": None, "by_horizon": {}, "by_quantity": {}, "cells": 0, "left_out": 0}
    logs = _log_ratios(cube["values"][model], cube["values"][benchmark], np.ones(cube["units"]))
    known = np.isfinite(logs)
    return {
        "score": _score_of_logs(logs),
        "by_horizon": {
            month: float(np.exp(np.nanmean(logs[m])))
            for m, month in enumerate(cube["months"])
            if known[m].any()
        },
        "by_quantity": {
            quantity: float(np.exp(np.nanmean(logs[:, q])))
            for q, quantity in enumerate(cube["quantities"])
            if known[:, q].any()
        },
        "cells": int(
            (np.isfinite(cube["values"][model]) & np.isfinite(cube["values"][benchmark])).sum()
        ),
        "left_out": int((~known).sum()),
    }


def interval(
    rows: Rows,
    model: str,
    benchmark: str,
    *,
    unit: str,
    block: int = 1,
    repetitions: int = REPETITIONS,
    seed: int,
) -> dict:
    """The score of ``relative`` with its sampling error: point, SE of the log, 95 % bounds."""
    cube = _cube(rows, (model, benchmark), unit)
    if cube is None:
        return {"point": None, "log_se": None, "low": None, "high": None}
    first, second = cube["values"][model], cube["values"][benchmark]
    point = _score_of_logs(_log_ratios(first, second, np.ones(cube["units"])))
    if point is None:
        return {"point": None, "log_se": None, "low": None, "high": None}
    return {"point": point, **_spread(lambda w: _log_ratios(first, second, w), cube, block,
                                      repetitions, seed)}


def paired(
    rows: Rows,
    other: Rows,
    key: str,
    *,
    unit: str,
    block: int = 1,
    repetitions: int = REPETITIONS,
    seed: int,
) -> dict:
    """The ratio of two versions' CRPS on the same rows: below one the first version wins."""
    joined = []
    by_key = {(row[unit], row["month"], row["quantity"]): row for row in other}
    for row in rows:
        twin = by_key.get((row[unit], row["month"], row["quantity"]))
        if twin is None:
            continue
        joined.append({**row, "_first": row.get(key), "_second": twin.get(key)})
    return interval(
        joined, "_first", "_second", unit=unit, block=block, repetitions=repetitions, seed=seed
    )


def _spread(logs_of, cube: dict, block: int, repetitions: int, seed: int) -> dict:
    """The SE of the log score and the 95 % bounds of the score, by moving blocks of units."""
    rng = np.random.default_rng(seed)
    samples = np.full(repetitions, np.nan)
    for repetition in range(repetitions):
        weights = block_weights(cube["units"], block, rng)
        logs = logs_of(weights)
        if np.isfinite(logs).any():
            samples[repetition] = _mean_of_logs(logs)
    known = samples[np.isfinite(samples)]
    if len(known) < 2:
        return {"log_se": None, "low": None, "high": None}
    low, high = np.percentile(known, [2.5, 97.5])
    return {
        "log_se": float(np.std(known, ddof=1)),
        "low": float(np.exp(low)),
        "high": float(np.exp(high)),
    }


def _cube(rows: Rows, columns: Sequence[str], unit: str | None) -> dict | None:
    """The rows as ``(units, months, quantities)`` arrays; ``None`` without a value."""
    chosen = [
        row
        for row in rows
        if all(row.get(column) is not None and np.isfinite(row[column]) for column in columns)
    ]
    if not chosen:
        return None
    units = sorted({str(row[unit]) for row in chosen}) if unit else ["all"]
    months = sorted({int(row["month"]) for row in chosen})
    quantities = sorted({str(row["quantity"]) for row in chosen})
    u_index = {name: k for k, name in enumerate(units)}
    m_index = {month: k for k, month in enumerate(months)}
    q_index = {quantity: k for k, quantity in enumerate(quantities)}
    if unit:
        shape = (len(units), len(months), len(quantities))
        values = {column: np.full(shape, np.nan) for column in columns}
        for row in chosen:
            at = (
                u_index[str(row[unit])],
                m_index[int(row["month"])],
                q_index[str(row["quantity"])],
            )
            for column in columns:
                values[column][at] = float(row[column])
    else:
        # Without a unit the rows of a group are pooled: only their mean enters the ratio.
        totals = {column: np.zeros((len(months), len(quantities))) for column in columns}
        counts = np.zeros((len(months), len(quantities)))
        for row in chosen:
            at = (m_index[int(row["month"])], q_index[str(row["quantity"])])
            counts[at] += 1
            for column in columns:
                totals[column][at] += float(row[column])
        values = {}
        for column in columns:
            values[column] = np.full((1, *counts.shape), np.nan)
            np.divide(totals[column], counts, out=values[column][0], where=counts > 0)
    return {"units": len(units), "months": months, "quantities": quantities, "values": values}


def _log_ratios(first: np.ndarray, second: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``log(mean first / mean second)`` per month and quantity, the units weighted."""
    both = np.isfinite(first) & np.isfinite(second)
    weight = np.where(both, weights[:, None, None], 0.0)
    top = (np.where(both, first, 0.0) * weight).sum(axis=0)
    bottom = (np.where(both, second, 0.0) * weight).sum(axis=0)
    logs = np.full(top.shape, np.nan)
    positive = (top > 0) & (bottom > 0)
    logs[positive] = np.log(top[positive] / bottom[positive])
    return logs


def _mean_of_logs(logs: np.ndarray) -> float:
    """Equal weights over the quantities at a month, then over the months."""
    known = np.isfinite(logs)
    sizes = known.sum(axis=1)
    totals = np.where(known, logs, 0.0).sum(axis=1)
    return float((totals[sizes > 0] / sizes[sizes > 0]).mean())


def _score_of_logs(logs: np.ndarray) -> float | None:
    if not np.isfinite(logs).any():
        return None
    return float(np.exp(_mean_of_logs(logs)))
