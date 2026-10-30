"""The history of a quantity on the macro panel: what any satellite model estimates on.

``history`` assembles the months, the object tree and the observations of one quantity in the
model scale of its transform, with the ``x`` inputs the quantity declares (its regressors and
its ``max(0, y - c)`` features) taken from the panel. A satellite model that wants the standard
linear form uses this; one that needs something else reads the series store and the panel
itself. ``x_from_paths`` builds the same ``x`` from the months of a macro simulation, and
``correlation`` turns the residual series of the drivers into their correlation matrix.
"""

from dataclasses import dataclass
from datetime import date

import numpy as np

from mst.data.store import SeriesStore
from mst.macro.panel import Panel

from .registry import Quantity, Registry, to_model

COMMON = 12
"""Months two drivers must share for their residual correlation to be estimated."""


@dataclass(frozen=True)
class History:
    """One quantity on the month grid of the macro panel."""

    quantity: Quantity
    months: list[date]
    nodes: list[str]
    """Objects with a driver of the quantity and every object above them, parents first."""
    parent: np.ndarray
    """(N,) the index of each node's parent, -1 at the top."""
    classes: list[str]
    driver: list[str | None]
    """The driver id of each node; None for a node that only links drivers below it."""
    d: np.ndarray
    """(T, N) the driver in the model scale; NaN where unknown and at nodes without a series."""
    approximate: np.ndarray
    """(T, N) True where the value's publication date is estimated."""
    x: np.ndarray
    """(T, m) the elements of x_t; NaN where a macro variable is unknown."""


def inputs(quantity: Quantity, panel: Panel) -> np.ndarray:
    """``x_t`` on the panel months: the regressors, then the features."""
    index = {id_: i for i, id_ in enumerate(panel.ids)}
    missing = [v for v in quantity.regressors + [f.variable for f in quantity.features]
               if v not in index]  # fmt: skip
    if missing:
        raise ValueError(f"{quantity.id} needs {', '.join(missing)}, which the panel lacks")
    columns = [panel.y[:, index[v]] for v in quantity.regressors]
    columns += [np.maximum(0.0, panel.y[:, index[f.variable]] - f.above) for f in quantity.features]
    return np.column_stack(columns)


def history(
    registry: Registry, quantity: str, panel: Panel, store: SeriesStore, as_of: date
) -> History:
    """The history of ``quantity`` on the panel months, as known on ``as_of``."""
    q = registry.quantity(quantity)
    drivers = {d.object: d for d in registry.drivers if d.quantity == quantity}
    if not drivers:
        raise ValueError(f"no driver of {quantity} is registered")
    nodes = set(drivers)
    for object_ in drivers:
        nodes.update(registry.ancestors(object_))
    order = [o.id for o in registry.objects if o.id in nodes]
    order.sort(key=lambda id_: len(registry.ancestors(id_)))
    position = {id_: i for i, id_ in enumerate(order)}
    month = {m: t for t, m in enumerate(panel.months)}
    d = np.full((len(panel.months), len(order)), np.nan)
    approximate = np.zeros(d.shape, dtype=bool)
    for object_, driver in drivers.items():
        if driver.series is None:
            continue
        rows = [row for row in store.as_of(driver.series, as_of) if row.month in month]
        if not rows:
            raise ValueError(f"{driver.id}: series {driver.series} has no value known on {as_of}")
        i = position[object_]
        for row in rows:
            d[month[row.month], i] = to_model(q.transform, row.value)
            approximate[month[row.month], i] = row.approximate
    return History(
        quantity=q,
        months=list(panel.months),
        nodes=order,
        parent=np.array(
            [position.get(registry.object(id_).parent, -1) for id_ in order], dtype=int
        ),
        classes=[registry.object(id_).class_ for id_ in order],
        driver=[drivers[id_].id if id_ in drivers else None for id_ in order],
        d=d,
        approximate=approximate,
        x=inputs(q, panel),
    )


def x_from_paths(quantity: Quantity, ids: list[str], months: np.ndarray) -> np.ndarray:
    """``x`` of a quantity from macro months ``(..., n)`` in the order of ``ids``."""
    index = {id_: i for i, id_ in enumerate(ids)}
    parts = [months[..., index[v]] for v in quantity.regressors]
    parts += [np.maximum(0.0, months[..., index[f.variable]] - f.above) for f in quantity.features]
    return np.stack(parts, axis=-1)


def correlation(series: list[np.ndarray]) -> np.ndarray:
    """The residual correlation of drivers, pairwise over common months, made positive definite.

    A pair with fewer than ``COMMON`` common months gets zero. The pairwise matrix is moved to
    the nearest positive definite one by clipping its eigenvalues, then rescaled to a unit
    diagonal.
    """
    k = len(series)
    corr = np.eye(k)
    for a in range(k):
        for b in range(a + 1, k):
            both = np.isfinite(series[a]) & np.isfinite(series[b])
            if both.sum() >= COMMON:
                corr[a, b] = corr[b, a] = np.corrcoef(series[a][both], series[b][both])[0, 1]
    values, vectors = np.linalg.eigh(corr)
    fixed = vectors @ np.diag(np.maximum(values, 1e-6)) @ vectors.T
    scale = np.sqrt(np.diag(fixed))
    return fixed / np.outer(scale, scale)
