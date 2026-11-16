"""The positions valued on the paths of a window: the forecast, the base run and the facts.

The macro variables and the drivers of a window come back in the own units the position models
read (``conditional_paths``); the facts of the window make one path of the same shape
(``realized_paths``); the engine values the portfolio's positions on either (``portfolio_run``),
and a position it could not calculate stays in the batch with its reason
(``portfolio_exclusions``). ``returns`` turns a batch into the return of every position and of
the portfolio, in percent of the current value, as the cells of those levels read it.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np

from ..data.store import SeriesStore, shift
from ..drivers.bundle import SatelliteBundle
from ..drivers.registry import from_model
from ..engine.run import Batch
from ..engine.run import run as run_positions
from ..macro.panel import Panel
from ..macro.registry import load_registry
from ..plugin_sdk import FixedPaths
from ..portfolio.spec import Portfolio


def conditional_paths(
    simulation, driver_paths, bundle: SatelliteBundle, *, seed: int = 0
) -> FixedPaths:
    """The episode's conditional forecast in the own units the position models read.

    ``simulation`` is the macro forecast in the model scale and ``driver_paths`` the drivers the
    bundle simulated on it, also in the model scale. Every macro variable is mapped back through
    its registry transform (``variable.from_model``) and every driver through the transform of its
    quantity, exactly as ``mst.scenario.build`` reports a run; the drivers keep the flags their
    simulation set, and ``seed`` is the run's seed, recorded on the paths.
    """
    if len(simulation.months) != simulation.paths.shape[1]:
        raise ValueError(
            f"the simulation names {len(simulation.months)} months for "
            f"{simulation.paths.shape[1]} months of paths"
        )
    if driver_paths.paths.shape[:2] != simulation.paths.shape[:2]:
        raise ValueError(
            f"the drivers are {driver_paths.paths.shape[:2]}, the macro paths "
            f"{simulation.paths.shape[:2]}"
        )
    registry = load_registry()
    macros = {}
    for k, id_ in enumerate(simulation.ids):
        try:
            variable = registry.variable(id_)
        except KeyError:
            raise ValueError(
                f"the paths carry {id_}, which the macro registry does not know"
            ) from None
        macros[id_] = variable.from_model(simulation.paths[..., k])
    drivers = {}
    flags = {}
    for k, driver in enumerate(driver_paths.ids):
        try:
            quantity = bundle.registry.quantity(bundle.quantity_of(driver))
        except KeyError:
            raise ValueError(f"the satellite version knows no quantity of {driver}") from None
        drivers[driver] = from_model(quantity.transform, driver_paths.paths[..., k])
        flags[driver] = driver_paths.flags.get(driver, frozenset())
    return FixedPaths(
        macros=macros,
        drivers=drivers,
        driver_flags=flags,
        start=simulation.months[0],
        seed=seed,
    )


def realized_paths(
    store: SeriesStore,
    *,
    as_of: date,
    known_on: date,
    months: Sequence[date],
    panel: Panel,
    realized: np.ndarray,
    bundle: SatelliteBundle,
) -> tuple[FixedPaths | None, list[str]]:
    """The facts of the episode window in the own units the position models read.

    Month ``t = 0`` is the last month of ``panel``, the month of ``as_of`` or, when that month's
    values are not out yet, an earlier one, and holds what the calculation date knows: the
    panel's last row for a macro variable and the latest value of the driver's series known on
    ``as_of`` for a month up to ``t = 0``. Months ``t = 1 .. H`` are ``months`` exactly, the
    months after ``t = 0`` as the window builds them, and hold the facts known on ``known_on``; the macro values come from
    ``realized`` ``(H, n)`` in the model scale, in the order of ``panel.ids``, and are mapped back
    through their registry transforms. Drivers in their own units are read from ``store``.

    ``missing`` names every ``<series or id>/<month>`` gap of the window, including gaps in the
    middle of it and at its ends, and every served driver without a series of its own. A gap
    stays NaN in the path rather than voiding the window: a position that reads that value
    cannot be valued, and the caller drops the episode by name — a driver nobody reads must not
    drop it. ``None`` — there is nothing to build: no macro variable and no driver at all.
    """
    start = _window_start(as_of, panel)
    window = list(months)
    expected = [shift(start, k) for k in range(1, len(window) + 1)]
    if window != expected:
        raise ValueError(
            f"the window {window[:1]}{'...' if window else ''} is not the months t = 1 .. H "
            f"after {start:%Y-%m}"
        )
    facts = np.asarray(realized, dtype=float)
    if facts.shape != (len(window), len(panel.ids)):
        raise ValueError(
            f"the realized values are {facts.shape}, the window {len(window)} months of "
            f"{len(panel.ids)} macro variables"
        )
    registry = load_registry()
    missing: list[str] = []
    macros = {}
    for i, id_ in enumerate(panel.ids):
        try:
            variable = registry.variable(id_)
        except KeyError:
            raise ValueError(
                f"the panel carries {id_}, which the macro registry does not know"
            ) from None
        column = np.concatenate([[panel.y[-1, i]], facts[:, i]])
        _record_missing(missing, id_, [start, *window], column)
        macros[id_] = variable.from_model(column)[None, :]
    series_of = {driver.id: driver.series for driver in bundle.registry.drivers}
    drivers = {}
    for driver in bundle.served():
        series = series_of.get(driver)
        column = np.full(len(window) + 1, np.nan)
        if series is None:
            missing.append(f"{driver}/no series")
        else:
            known_today = [row for row in store.as_of(series, as_of) if row.month <= start]
            if known_today:
                column[0] = float(known_today[-1].value)
            known_facts = {row.month: float(row.value) for row in store.as_of(series, known_on)}
            for t, month in enumerate(window, start=1):
                if month in known_facts:
                    column[t] = known_facts[month]
            _record_missing(missing, series, [start, *window], column)
        drivers[driver] = column[None, :]
    if not macros and not drivers:
        return None, missing
    return (
        FixedPaths(
            macros=macros,
            drivers=drivers,
            driver_flags={driver: frozenset() for driver in drivers},
            start=start,
        ),
        missing,
    )


def portfolio_run(portfolio: Portfolio, paths, *, drivers, served, as_of) -> Batch:
    """The engine's results of the portfolio's positions on ``paths``, every reason kept.

    ``drivers`` is the driver registry of the episode's satellite version and ``served`` the
    drivers it serves, exactly as ``mst.engine.run`` reads them. A position the engine could not
    calculate stays in the batch as ``invalid`` with its diagnostics; ``portfolio_exclusions``
    names it.
    """
    return run_positions(
        portfolio.positions,
        paths,
        drivers=drivers,
        served=served,
        base_currency=portfolio.base_currency,
        as_of=as_of,
    )


def portfolio_exclusions(batch: Batch) -> list[dict]:
    """The positions of the batch that were not calculated, each with its reason."""
    found = []
    for result in batch.results:
        if result.status != "invalid":
            continue
        reason = (
            "; ".join(item.message for item in result.diagnostics if item.severity == "error")
            or "; ".join(result.flags)
            or "the position was not calculated"
        )
        found.append({"position": result.position_id, "reason": reason})
    return found


def _window_start(as_of: date, panel: Panel) -> date:
    """Month ``t = 0`` of the window: the last month of the panel, not after the month of
    ``as_of``; an earlier month when the values of the calculation month are not out yet."""
    if not panel.months:
        raise ValueError("the panel has no month: month t = 0 of the window is unknown")
    start = panel.months[-1]
    if start > as_of.replace(day=1):
        raise ValueError(
            f"the panel ends at {start:%Y-%m}, after the calculation date {as_of:%Y-%m-%d}: "
            "the panel holds a month the date cannot know"
        )
    return start


def _record_missing(missing: list[str], name: str, months: Sequence[date], values) -> None:
    """Append every ``<name>/<month>`` of the window whose value is not finite."""
    for month, value in zip(months, values, strict=True):
        if not np.isfinite(value):
            missing.append(f"{name}/{month:%Y-%m}")


@dataclass(frozen=True)
class Valued:
    """The portfolio's positions on one set of paths: what moved, what it was worth, what failed."""

    changes: dict[str, np.ndarray]
    """``(R,)`` the change of value of every calculated position, in the base currency."""
    values: dict[str, float]
    """The current value of every calculated position."""
    excluded: dict[str, str]
    """The positions the engine could not calculate, with the reason."""

    def returns(self, chosen: Sequence[str]) -> np.ndarray:
        """``(R,)`` the return of the chosen positions together, % of their current value."""
        total = sum(self.values[name] for name in chosen)
        if total <= 0:
            return np.full(len(next(iter(self.changes.values()))), np.nan)
        return 100.0 * np.sum([self.changes[name] for name in chosen], axis=0) / total


def value(portfolio: Portfolio, paths, *, bundle: SatelliteBundle, as_of: date) -> Valued:
    """The positions of ``portfolio`` valued on ``paths`` by the engine."""
    batch = portfolio_run(
        portfolio, paths, drivers=bundle.registry, served=bundle.served(), as_of=as_of
    )
    return Valued(
        changes={
            result.position_id: np.asarray(result.value_change, dtype=float)
            for result in batch.calculated
        },
        values={result.position_id: float(result.value_current) for result in batch.calculated},
        excluded={item["position"]: item["reason"] for item in portfolio_exclusions(batch)},
    )
