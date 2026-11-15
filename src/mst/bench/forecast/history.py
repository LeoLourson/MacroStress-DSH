"""What happened: monthly series, their h-month changes and the windows a forecast is read on.

Every historical number of the set comes from here, on the same rule: a window of ``h``
months starts at month ``t`` and ends at ``t + h``; its changes are the months ``t + 1 .. t +
h``, all of them on or after the plan's ``history_start``, none of them a splice of the
series. Windows overlap, so their count is not a count of independent observations.
"""

from datetime import date

import numpy as np

from mst.data.store import SeriesStore
from mst.drivers.objects import series_id


def month_index(month: date) -> int:
    return month.year * 12 + month.month - 1


def from_index(index: int) -> date:
    return date(index // 12, index % 12 + 1, 1)


def grid(first: date, last: date) -> list[date]:
    return [from_index(i) for i in range(month_index(first), month_index(last) + 1)]


def monthly(store: SeriesStore, series: str, as_of: date) -> dict[date, float]:
    """The values of a series by month, as published on or before ``as_of``."""
    return {row.month: row.value for row in store.as_of(series, as_of)}


def driver_series(registry, driver: str) -> str:
    """The store series of a driver: its own, or ``drv.<quantity>.<object>``."""
    quantity, object_ = driver.split(":", 1)
    for item in registry.drivers:
        if item.id == driver and item.series:
            return item.series
    return series_id(quantity, object_)


def windows(months: list[date], horizon: int, start: date, spliced: set[date]) -> list[int]:
    """The start positions ``t`` of the ``horizon``-month windows the history may read."""
    first = month_index(start)
    found = []
    for t in range(len(months) - horizon):
        changes = range(month_index(months[t]) + 1, month_index(months[t]) + horizon + 1)
        if changes.start >= first and not any(from_index(m) in spliced for m in changes):
            found.append(t)
    return found


def changes(values: np.ndarray, horizon: int, starts: list[int]) -> np.ndarray:
    """``values[t + horizon] - values[t]`` of the windows, the missing ones left out."""
    starts = np.asarray(starts, dtype=int)
    found = values[starts + horizon] - values[starts] if len(starts) else np.zeros(0)
    return found[np.isfinite(found)]


def width(sample: np.ndarray, quantiles: tuple[float, float]) -> float:
    """``q_high - q_low`` of a sample; NaN when it is empty."""
    sample = np.asarray(sample, dtype=float)
    sample = sample[np.isfinite(sample)]
    if not len(sample):
        return float("nan")
    low, high = np.quantile(sample, quantiles)
    return float(high - low)


def mean_pair_correlation(columns: np.ndarray) -> tuple[float, int]:
    """The mean correlation over pairs of columns, each pair on the rows where both are known;
    and the fewest rows a pair had."""
    count = columns.shape[1]
    found, fewest = [], len(columns)
    for i in range(count):
        for j in range(i + 1, count):
            known = np.isfinite(columns[:, i]) & np.isfinite(columns[:, j])
            fewest = min(fewest, int(known.sum()))
            if known.sum() > 2:
                found.append(np.corrcoef(columns[known, i], columns[known, j])[0, 1])
    return (float(np.mean(found)) if found else float("nan")), fewest
