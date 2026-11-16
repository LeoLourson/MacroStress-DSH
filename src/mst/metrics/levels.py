"""The cells of every level a window is judged on, and the rows the scores read.

A window (``mst.metrics.windows``) gives one cell per quantity and month of its horizon where
the fact is known: the macro variables from the macro paths, the served drivers from the driver
paths, and — when the check names a portfolio — every position and the portfolio itself from
the position models on those paths. A cell carries the forecast and every comparison drawn on
the same cell: the random walk with drift from the history known on the calculation date, the
walk that also knows the path the scenario fixes, the base run of the same models under the
empty scenario, for a driver the naive forecast of a year earlier with the trend, and for a
position or the portfolio the historical simulation — its return over every past window of the
same length the calculation date knows, valued by the same models. A comparison that does not
exist for a window is absent: the empty scenario has no base run apart from itself and fixes no
path.

The return of a position is its change of value over its current value, in percent; the fact
is the same valuation on the facts of the window. The portfolio's return reads only the
positions valued on both, so a position the facts cannot value does not turn into a loss.

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
from .valuation import conditional_paths, realized_paths, value
from .windows import Window

MACRO = "macro"
"""The level of the macro variables."""
POSITION = "position"
"""The level of the positions of a portfolio."""
PORTFOLIO = "portfolio"
"""The level of the portfolio as a whole."""


def cells(window: Window, registry: Registry, plan: Episodes) -> list[Cell]:
    """Every cell of the window: the macro variables, the drivers when a bundle is on, and the
    positions and the portfolio when a portfolio is."""
    found = macro_cells(window, registry, plan)
    if window.bundle is not None:
        found += driver_cells(window, registry, plan)
    if window.portfolio is not None:
        found += position_cells(window, registry, plan)
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
    bundle, paths, base = window.bundle, window.driver_paths, window.base_driver_paths
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


def position_cells(window: Window, registry: Registry, plan: Episodes) -> list[Cell]:
    """The cells of every position and of the portfolio, month by month of the horizon.

    The forecast, the base run and the facts are valued by the same position models on the
    paths cut at the month; the benchmark is the historical simulation of the same month. The
    positions the engine cannot value on the forecast are kept on the window with the reason.
    """
    portfolio, bundle, as_of = window.portfolio, window.bundle, window.as_of
    forecast = conditional_paths(window.macro_paths, window.driver_paths, bundle, seed=plan.seed)
    base = None
    if window.base_shadow is not None:
        base = conditional_paths(
            observed(window.base_shadow, registry), window.base_driver_paths, bundle, seed=plan.seed
        )
    realized, _ = realized_paths(
        window.store,
        as_of=as_of,
        known_on=window.known_on,
        months=window.months,
        panel=window.panel,
        realized=window.realized,
        bundle=bundle,
    )
    if realized is None:
        return []
    found = []
    for t in range(1, window.horizon + 1):
        drawn = value(portfolio, forecast.up_to(t), bundle=bundle, as_of=as_of)
        window.excluded.update(drawn.excluded)
        facts = value(portfolio, realized.up_to(t), bundle=bundle, as_of=as_of)
        based = (
            None if base is None else value(portfolio, base.up_to(t), bundle=bundle, as_of=as_of)
        )
        kept = [
            name
            for name in drawn.changes
            if name in facts.changes and np.isfinite(facts.changes[name]).all()
        ]
        for name in kept:
            found.append(
                _return_cell(
                    window, t, name, POSITION, [name], drawn, facts, based, _past(window, [name], t)
                )
            )
        if kept:
            found.append(
                _return_cell(
                    window,
                    t,
                    window.portfolio_id,
                    PORTFOLIO,
                    kept,
                    drawn,
                    facts,
                    based,
                    _past(window, kept, t),
                )
            )
    return found


def _return_cell(window, t, quantity, level, chosen, drawn, facts, based, past) -> Cell:
    """One cell of a return: the chosen positions together on the forecast, the facts, the
    base run and the past."""
    has_base = based is not None and all(name in based.changes for name in chosen)
    return Cell(
        unit=window.id,
        month=t,
        quantity=quantity,
        level=level,
        fact=float(facts.returns(chosen)[0]),
        draws=drawn.returns(chosen),
        base=based.returns(chosen) if has_base else None,
        historical=past if len(past) else None,
        history_range=history_range(past),
    )


def _past(window: Window, chosen: list[str], t: int) -> np.ndarray:
    """The returns of the chosen positions over the past windows of ``t`` months, the ones the
    calculation date knows."""
    found = window.past.returns(chosen)[:, t - 1]
    return found[np.isfinite(found)]


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
            "historical": None,
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
            ("historical", cell.historical),
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


def fell_out(
    windows: list[Window],
    found: list[Cell],
    *,
    refused: Mapping[str, str],
    skipped: Mapping[str, str] | None = None,
) -> dict:
    """What a check could not judge, with the reasons: the windows refused or left out for too
    little history, the variables a model left out as too short, the positions the engine could
    not value and the cells without a finite draw."""
    return {
        "refused": dict(refused),
        "skipped": dict(skipped or {}),
        "left_out": {window.id: window.left_out for window in windows if window.left_out},
        "excluded": {window.id: dict(window.excluded) for window in windows if window.excluded},
        "non_finite": non_finite(found),
    }
