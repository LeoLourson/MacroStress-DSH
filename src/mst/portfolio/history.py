"""The portfolio held over the past: every position valued by its own model on what happened.

The observed macro series and driver series of a past window make one path, and the position
models value it exactly as they value a forecast. A position is valued on the windows where
everything it reads was observed; the portfolio's return of a window is the sum of the changes
over the sum of the current values of those positions, so a share listed later does not turn a
window into a loss. The returns of the windows that ended before a date are the historical
simulation of that date: the benchmark of a position's or the portfolio's forecast.
"""

from dataclasses import dataclass
from datetime import date

import numpy as np

from mst.data.store import SeriesStore
from mst.drivers.objects import series_id
from mst.engine.model_catalog import catalog
from mst.engine.run import run as engine_run
from mst.plugin_sdk.model import RunOptions
from mst.plugin_sdk.paths import FixedPaths


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


def inputs(position, satellites, as_of: date, horizon: int, base_currency: str):
    """The macro variables and the drivers one position reads."""
    model = catalog.model_for(position)
    options = RunOptions(as_of=as_of, horizon=horizon, base_currency=base_currency)
    needs = model.requirements(position, options)
    drivers = []
    for requirement in needs.drivers:
        driver = satellites.registry.resolve(
            requirement.quantity,
            object_=requirement.object,
            codes=requirement.codes or position.codes,
            among=satellites.served(),
        )
        if driver is None:
            raise ValueError(f"{position.id}: no served driver of {requirement.quantity}")
        drivers.append(driver)
    return list(needs.macro), drivers


@dataclass(frozen=True)
class Replay:
    """Every position held over every past window, kept apart so any subset can be summed."""

    starts: list[date]
    """The first month (``t = 0``) of every window."""
    positions: list[str]
    changes: np.ndarray
    """``(P, W, H)`` the change of value of every position by months ``1 .. H``; NaN where
    something the position reads was not observed through that month."""
    values: np.ndarray
    """``(P, W)`` the current value of every position at the start of every window."""

    def returns(self, chosen: list[str] | None = None) -> np.ndarray:
        """``(W, H)`` the return of the chosen positions, %: the sum of the changes of those
        observed over the sum of their current values; NaN where none was observed."""
        rows = [self.positions.index(p) for p in (chosen or self.positions)]
        changes = self.changes[rows]
        observed = np.isfinite(changes)
        values = np.where(observed, self.values[rows][:, :, None], 0.0).sum(axis=0)
        total = np.where(observed, changes, 0.0).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(values > 0, 100 * total / values, np.nan)

    def window(self, start: date) -> int:
        return self.starts.index(start)


def replay(macro, satellites, portfolio, store, *, horizon: int, as_of: date) -> Replay:
    """The portfolio held over every window of ``horizon`` months of the macro panel."""
    from mst.macro.registry import load_registry

    months = macro.panel.months
    macro_registry = load_registry()
    natural = {
        variable: np.asarray(
            macro_registry.variable(variable).from_model(macro.panel.y[:, k]), dtype=float
        )
        for k, variable in enumerate(macro.panel.ids)
    }
    starts = np.arange(len(months) - horizon)
    ids = [position.id for position in portfolio.positions]
    changes = np.full((len(ids), len(starts), horizon), np.nan)
    values = np.full((len(ids), len(starts)), np.nan)
    for p, position in enumerate(portfolio.positions):
        variables, drivers = inputs(position, satellites, as_of, horizon, portfolio.base_currency)
        driver_rows = {}
        for driver in drivers:
            found = monthly(store, driver_series(satellites.registry, driver), as_of)
            driver_rows[driver] = np.array([found.get(m, np.nan) for m in months])
        for t in range(1, horizon + 1):
            span = starts[:, None] + np.arange(t + 1)[None, :]
            known = np.ones(len(starts), dtype=bool)
            for row in [*(natural[v] for v in variables), *driver_rows.values()]:
                known &= np.isfinite(row[span]).all(axis=1)
            if not known.any():
                continue
            chosen = span[known]
            # Every macro variable observed through these windows: a model may read more than
            # it declares (the bond reads the whole curve it finds).
            observed = {
                v: row[chosen] for v, row in natural.items() if np.isfinite(row[chosen]).all()
            }
            result = engine_run(
                [position],
                FixedPaths(
                    macros=observed,
                    drivers={d: row[chosen] for d, row in driver_rows.items()},
                    driver_flags={d: frozenset() for d in driver_rows},
                    start=months[0],
                    seed=0,
                ),
                drivers=satellites.registry,
                served=satellites.served(),
                base_currency=portfolio.base_currency,
                as_of=as_of,
            ).results[0]
            if result.status == "invalid":
                raise ValueError(
                    f"{position.id} could not be valued on history: "
                    + "; ".join(d.message for d in result.diagnostics)
                )
            changes[p, known, t - 1] = np.asarray(result.value_change, dtype=float)
            values[p, known] = result.value_current
    return Replay([months[t] for t in starts], ids, changes, values)
