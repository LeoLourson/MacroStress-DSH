"""The cells of every level a window is judged on, and the rows the scores read.

A window (``mst.metrics.windows``) gives one cell per quantity and month of its horizon where
the fact is known: the macro variables from the macro paths, the served drivers from the driver
paths. A cell carries the forecast and every comparison drawn on the same cell — the random
walk with drift from the history known on the calculation date, the walk that also knows the
path the scenario fixes, the base run of the same models under the empty scenario and, for a
driver, the naive forecast of a year earlier with the trend. A comparison that does not exist
for a window is absent: the empty scenario has no base run apart from itself and fixes no path.

A cell the scenario fixes — a point observation or an intervention — is kept and marked
constrained: its value was told to the model, so it is reported and never scored.

``rows`` turns the scored cells into flat rows: the CRPS of every forecast, the band of the
model, and the numbers the sanity of the quantities reads (``mst.metrics.sanity``).
"""

from collections.abc import Mapping
from datetime import date

import numpy as np

from mst.macro.registry import Registry
from mst.model_sdk.bounds import lower_bounds, observed

from .baselines import (
    DOMAIN,
    WIDTH,
    band_problems,
    conditional_walk,
    macro_domain,
    naive_annual,
    quantity_domain,
    random_walk,
)
from .cells import Cell, crps, finite_draws, history_range, pit, quantiles
from .episodes import Episodes
from .windows import Window

MACRO = "macro"
"""The level of the macro variables."""


def cells(window: Window, registry: Registry, plan: Episodes) -> list[Cell]:
    """Every cell of the window: the macro variables, then the drivers when a bundle is on."""
    found = macro_cells(window, registry, plan)
    if window.bundle is not None:
        found += driver_cells(window, registry, plan)
    return found


def macro_cells(window: Window, registry: Registry, plan: Episodes) -> list[Cell]:
    """The cells of the window's macro variables, with the walks and the base run alongside."""
    panel, draws = window.panel, window.macro_paths
    base = None if window.base_shadow is None else observed(window.base_shadow, registry)
    column_of = {id_: i for i, id_ in enumerate(draws.ids)}
    pinned = _pinned(window.scenario.constraints) | _installed(window.scenario.interventions)
    given = [panel.ids.index(id_) for id_ in window.constrained]
    found = []
    for j, id_ in enumerate(panel.ids):
        i = column_of.get(id_)
        if i is None:
            continue
        history = panel.y[:, j]
        floors = lower_bounds(registry, [id_])
        lower = None if floors is None else float(floors[0])
        walk = random_walk(history, window.horizon, draws=plan.draws, seed=plan.seed, lower=lower)
        knowing = (
            conditional_walk(
                history,
                panel.y[:, given],
                window.realized[:, given],
                draws=plan.draws,
                seed=plan.seed,
                lower=lower,
            )
            if given and j not in given
            else None
        )
        domain = macro_domain(registry.variable(id_))
        spread = history_range(history)
        for k in range(window.horizon):
            fact = window.realized[k, j]
            if np.isnan(fact):
                continue
            found.append(
                Cell(
                    unit=window.id,
                    month=k + 1,
                    quantity=id_,
                    level=MACRO,
                    fact=float(fact),
                    draws=draws.paths[:, k + 1, i],
                    baseline=walk[:, k],
                    base=None if base is None else base.paths[:, k + 1, i],
                    conditional=None if knowing is None else knowing[:, k],
                    constrained=(id_, k + 1) in pinned,
                    history_range=spread,
                    domain=domain,
                )
            )
    return found


def driver_cells(window: Window, registry: Registry, plan: Episodes) -> list[Cell]:
    """The cells of the served drivers, with the walk, the naive forecast and the base run.

    The base run of a driver is the bundle on the macro base run, drawn with the same seed. A
    driver whose paths leave the range of its own training history carries an unverified tail.
    """
    bundle, paths = window.bundle, window.driver_paths
    base = None
    if window.base_shadow is not None:
        base = bundle.simulate(
            observed(window.base_shadow, registry), bundle.served(), (), seed=plan.seed
        )
    series_of = {driver.id: driver.series for driver in bundle.registry.drivers}
    found = []
    for k, driver in enumerate(paths.ids):
        quantity, node = bundle.node(driver)
        declared = bundle.registry.quantity(quantity)
        level = (
            f"driver/{quantity}" if declared.level <= 1 else f"driver{declared.level}/{quantity}"
        )
        d = bundle.arrays.get(f"{quantity}.d")
        own = d[:, node] if d is not None else np.array([])
        walk = random_walk(own, window.horizon, draws=plan.draws, seed=plan.seed)
        naive = naive_annual(own, window.horizon, draws=plan.draws, seed=plan.seed)
        unverified = "outside_training_range" in paths.flags.get(driver, frozenset())
        domain = quantity_domain(declared)
        spread = history_range(own)
        for t, month in enumerate(window.months):
            fact = _driver_fact(window, series_of.get(driver), month, declared)
            if fact is None:
                continue
            found.append(
                Cell(
                    unit=window.id,
                    month=t + 1,
                    quantity=driver,
                    level=level,
                    fact=fact,
                    draws=paths.paths[:, t + 1, k],
                    baseline=walk[:, t] if walk.size else None,
                    naive=naive[:, t] if naive is not None else None,
                    base=None if base is None else base.paths[:, t + 1, k],
                    history_range=spread,
                    domain=domain,
                    unverified=unverified,
                )
            )
    return found


def rows(found: list[Cell], unit: str, half_of: Mapping[str, str]) -> list[dict]:
    """The scored cells as flat rows; ``unit`` names the key of the window id, ``half_of`` maps
    every window to its half.

    A constrained cell says nothing about the forecast and is left out; so is a cell whose
    draws are all non-finite, which ``non_finite`` counts. The band columns feed the sanity of
    the quantities: the width against the history range, the admission rules and, beside a
    base run, the move of the median the scenario made and the miss of the base run's median.
    """
    result = []
    for cell in found:
        if cell.constrained or not np.isfinite(cell.fact):
            continue
        draws = finite_draws(cell.draws)
        if not len(draws):
            continue
        band = quantiles(draws)
        broken = band_problems(
            draws, cell.fact, domain=cell.domain, history_range=cell.history_range
        )
        row = {
            unit: cell.unit,
            "half": half_of[cell.unit],
            "month": cell.month,
            "quantity": cell.quantity,
            "level": cell.level,
            "model": crps(draws, cell.fact),
            "walk": None,
            "conditional_walk": None,
            "base": None,
            "naive": None,
            "pit": pit(draws, cell.fact),
            "inside": bool(band[0] <= cell.fact <= band[2]),
            "width": float(band[2] - band[0]),
            "history_range": cell.history_range,
            DOMAIN: DOMAIN in broken,
            WIDTH: WIDTH in broken,
            "shift": None,
            "miss": None,
            "base_width": None,
        }
        for name, column in (
            ("walk", cell.baseline),
            ("conditional_walk", cell.conditional),
            ("base", cell.base),
            ("naive", cell.naive),
        ):
            known = np.array([]) if column is None else finite_draws(column)
            if len(known):
                row[name] = crps(known, cell.fact)
        base = np.array([]) if cell.base is None else finite_draws(cell.base)
        if len(base):
            base_band = quantiles(base)
            row["shift"] = float(band[1] - base_band[1])
            row["miss"] = float(cell.fact - base_band[1])
            row["base_width"] = float(base_band[2] - base_band[0])
        result.append(row)
    return result


def non_finite(found: list[Cell]) -> dict[str, int]:
    """The unconstrained cells with a fact whose draws are all non-finite, per quantity."""
    counts: dict[str, int] = {}
    for cell in found:
        if cell.constrained or not np.isfinite(cell.fact) or len(finite_draws(cell.draws)):
            continue
        counts[cell.quantity] = counts.get(cell.quantity, 0) + 1
    return counts


def _pinned(constraints) -> set[tuple[str, int]]:
    """The ``(variable, month)`` cells a single-term constraint fixes.

    A constraint of one term pins its cell when it comes from a point, or when its band is
    narrower than the tolerance a point of the same terms would expand to: a narrow interval
    of that scale carries no scoring information either. The mark decides for a point, since
    ``value ± tau`` need not round back to a band of ``2 tau``.
    """
    pinned = set()
    for constraint in constraints:
        if len(constraint.terms) != 1:
            continue
        term = constraint.terms[0]
        if constraint.point or constraint.high - constraint.low <= 2 * constraint.tolerance:
            pinned.add((term.variable, term.month))
    return pinned


def _installed(interventions) -> set[tuple[str, int]]:
    """The ``(variable, month)`` cells an intervention sets: the scenario asserts the decision,
    so they are reported and never scored, as a point observation's cell is."""
    return {(item.target, month) for item in interventions for month in item.months}


def _driver_fact(window: Window, series, month: date, declared):
    """The realized value of a driver in the model scale, or None when it is not known."""
    from mst.drivers.registry import to_model

    if series is None:
        return None
    rows_ = [row for row in window.store.as_of(series, window.known_on) if row.month == month]
    if not rows_:
        return None
    try:
        return float(to_model(declared.transform, rows_[-1].value))
    except ValueError:
        return None
