"""The history the macro model is estimated on, as it was known on a given day.

``build_panel`` reads every registered variable from the series store as of a day, takes
``y = g(X)`` and lays the values on a month grid from ``start`` to the last month any variable
reached. A month a variable has no value for stays ``NaN``: the model draws it (quarterly and
annual prices, a series that starts later, the ragged end of slower releases). Global
variables come first, so the block structure of section 4.1 is the upper-left corner of every
matrix.
"""

from dataclasses import dataclass
from datetime import date

import numpy as np

from mst.data.store import SeriesStore

from .registry import Registry


@dataclass(frozen=True)
class Panel:
    ids: list[str]
    n_global: int
    months: list[date]
    y: np.ndarray
    """(T, n) in the model scale; NaN where the month is unknown."""
    approximate: np.ndarray
    """(T, n) True where the value's publication date is estimated."""


def months_between(first: date, last: date) -> list[date]:
    count = (last.year - first.year) * 12 + last.month - first.month + 1
    return [shift(first, k) for k in range(count)]


def shift(month: date, k: int) -> date:
    index = month.year * 12 + month.month - 1 + k
    return date(index // 12, index % 12 + 1, 1)


def deseasonalize(y: np.ndarray, months: list[date]) -> np.ndarray:
    """Removes a fixed calendar-month pattern estimated from month-on-month changes."""
    calendar = np.array([m.month - 1 for m in months])
    return y - seasonal_profile(y, months)[calendar]


def seasonal_profile(y: np.ndarray, months: list[date]) -> np.ndarray:
    """The seasonal level of each calendar month, January first.

    The mean change into each calendar month, less the mean over the twelve, accumulates into a
    level that repeats every year and averages zero.
    """
    calendar = np.array([m.month - 1 for m in months])
    steps = np.diff(y)
    known = ~np.isnan(steps)
    means = np.zeros(12)
    for month in range(12):
        chosen = known & (calendar[1:] == month)
        if not chosen.any():
            raise ValueError(f"no change into calendar month {month + 1} to estimate seasonality")
        means[month] = steps[chosen].mean()
    level = np.cumsum(means - means.mean())
    return level - level.mean()


def build_panel(
    store: SeriesStore,
    registry: Registry,
    *,
    start: date,
    as_of: date,
    seasonal: list[str] = (),
) -> Panel:
    variables = sorted(registry.variables, key=lambda v: v.block != "gl")
    unknown = set(seasonal) - {v.id for v in variables}
    if unknown:
        raise ValueError(f"seasonal names unknown variables: {', '.join(sorted(unknown))}")
    rows = {v.id: [o for o in store.as_of(v.id, as_of) if o.month >= start] for v in variables}
    last = max((r[-1].month for r in rows.values() if r), default=None)
    if last is None:
        raise ValueError(f"no variable has a value known on {as_of}")
    months = months_between(start, last)
    index = {month: t for t, month in enumerate(months)}
    y = np.full((len(months), len(variables)), np.nan)
    approximate = np.zeros(y.shape, dtype=bool)
    for i, variable in enumerate(variables):
        for row in rows[variable.id]:
            y[index[row.month], i] = variable.to_model(row.value)
            approximate[index[row.month], i] = row.approximate
        if variable.id in seasonal:
            y[:, i] = deseasonalize(y[:, i], months)
    return Panel(
        ids=[v.id for v in variables],
        n_global=sum(v.block == "gl" for v in variables),
        months=months,
        y=y,
        approximate=approximate,
    )
