"""Build scenario paths from translated observations and interventions."""

from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from ..corpus.index import sources_missing
from ..drivers.registry import from_model
from ..macro.registry import load_registry
from ..macro.version import MacroVersion
from ..metrics.cells import typical_movement
from ..model_sdk import dist as dist_sdk
from ..model_sdk.bounds import observed
from ..model_sdk.contracts import (
    ConstraintLaw,
    Dist,
    DriverPaths,
    LinearConstraint,
    LinearTerm,
    OneSided,
    ScenarioPlan,
)
from ..model_sdk.dist import ContradictsModels
from ..model_sdk.interventions import apply_default
from . import overlay, translate
from .context import context_from_versions
from .spec import Spec

#: The stream of the overlay's resampling draw, away from the model's own streams.
OVERLAY_STREAM = 4
MIXED_CONDITION = (
    "a condition mixes macro variables and drivers: split it, the macro model meets one part "
    "and the overlay the other"
)


@dataclass(frozen=True)
class Paths:
    """``R`` paths of ``H`` months: the macro variables and the drivers, in their own units."""

    months: list[date]
    """Months ``t = 0 .. H``; ``t = 0`` is the calculation month."""
    ids: list[str]
    """The macro variables of the paths."""
    macro: np.ndarray
    """``(R, H + 1, n)`` in each variable's own unit."""
    drivers: DriverPaths
    """``(R, H + 1, k)`` in each driver's own unit, with the flags of every path."""
    dist: Dist | None
    effective_size: float | None
    """How many paths carry the weight of the conditions on means: ``None`` without them."""
    checks: dict
    """What the sample says about the imposed conditions: the share of paths inside the
    event and the means against their bounds."""
    seed: int
    draws: int
    shadow: np.ndarray | None = None
    """``(R, H + 1, n)`` the macro paths in the model scale before the floors of the registry:
    what the conditions read. ``None`` when the paths did not come from a macro model."""
    diagnostics: dict = field(default_factory=dict)
    """What the macro model reports about the imposition: the size of the driving shocks the
    conditions need and the spread of the hidden-coordinate candidates."""


def tolerance_of(terms, macro: MacroVersion) -> float:
    """The half-width ``tau`` a point of these terms expands to: ``0.1 sigma(1)``.

    ``sigma_q(1)`` is the median absolute monthly change of the variable over the version's
    panel — the same scale the metrics normalise by. A sum of terms scales with the absolute
    weights: ``tau = 0.1 sum |w_q| sigma_q(1)``. A variable the panel never sees move takes
    the median ``sigma`` of the variables that do, and a panel where nothing moves falls back
    to ``1e-6``, so the expansion always stays a strict interval.
    """
    index = {id_: k for k, id_ in enumerate(macro.panel.ids)}
    total = 0.0
    for term in terms:
        if term.variable in index:
            history = macro.panel.y[:, index[term.variable]]
            movement = typical_movement(history, 1)
        else:
            # A variable the version does not know: validation names it separately, and the
            # tolerance falls back to the panel scale so the plan stays constructible.
            movement = float("nan")
        if not np.isfinite(movement) or movement <= 0:
            movement = _panel_movement(macro)
        total += abs(term.weight) * 0.1 * movement
    return float(total)


def point_constraint(
    terms: tuple[LinearTerm, ...], value: float, macro: MacroVersion
) -> LinearConstraint:
    """The constraint of a value fixed exactly: the interval ``value ± tau`` with the mark.

    This is the one place a point becomes an interval — the scenario's ``value`` constraints
    and the facts of a backtest episode both pass through it, so both expand the same way.
    """
    tolerance = tolerance_of(terms, macro)
    return LinearConstraint(
        terms=terms,
        low=value - tolerance,
        high=value + tolerance,
        point=True,
        tolerance=tolerance,
    )


def _panel_movement(macro: MacroVersion) -> float:
    """The median of the finite ``sigma(1)`` of the panel's variables; ``1e-6`` when none."""
    movements = [typical_movement(macro.panel.y[:, k], 1) for k in range(len(macro.panel.ids))]
    finite = [value for value in movements if np.isfinite(value) and value > 0]
    return float(np.median(finite)) if finite else 1e-6


def plan_of(spec, macro, *, draws=1, seed=0, horizon=12, inner=1, satellites=None):
    data = context_from_versions(macro, satellites)
    plan = translate.plan_of(spec, data, horizon, groups=draws, inner=inner, seed=seed)
    return replace(
        plan, interventions=tuple(i for i in plan.interventions if i.target in macro.ids)
    )


def dist_of(spec, macro, satellites=None, *, draws, seed, horizon=12, inner=1):
    plan = plan_of(
        spec, macro, draws=draws, seed=seed, horizon=horizon, inner=inner, satellites=satellites
    )
    if not plan.constraints:
        return None
    low = np.array([c.low for c in plan.constraints])
    high = np.array([c.high for c in plan.constraints])
    law = macro.plugin.constraint_law(macro.model, plan, draws=draws, seed=seed)
    if law is None:
        logs = macro.plugin.log_box_probability(
            macro.model, plan, low, high, draws=draws, seed=seed
        )
        if logs is None:
            return None
        distance, log_probability, error = _price_of_logs(logs)
        side = None
    else:
        rng = np.random.default_rng([seed, dist_sdk.DIST_STREAM])
        distance, log_probability, error = dist_sdk.price_of_law(law, low, high, rng=rng)
        side = _one_sided_of(spec, plan, law, low, high, seed=seed)
    return Dist(
        distance=distance,
        log_probability=log_probability,
        standard_error=error,
        tolerances=tuple(c.tolerance for c in plan.constraints if c.point),
        one_sided=side,
        message="the event has no probability at some value of R"
        if not np.isfinite(distance)
        else None,
    )


def _price_of_logs(logs) -> tuple[float, float, float]:
    """``(distance, log_probability, standard_error)`` of a model's own log probabilities."""
    logs = np.asarray(logs, dtype=float)
    count = len(logs)
    if np.any(np.isneginf(logs)):
        return float("inf"), float("-inf"), float("inf")
    terms = -logs
    return (
        float(terms.mean()),
        float(logsumexp(logs) - np.log(count)),
        float(terms.std(ddof=1) / np.sqrt(count)) if count > 1 else 0.0,
    )


def _one_sided_of(spec: Spec, plan: ScenarioPlan, law: ConstraintLaw, low, high, *, seed: int):
    """``dist+(s)``: the price of the event with every interval condition opened to a ray.

    The medians of the law's coordinates — weighted over its rows, one draw set per row —
    decide the side: an interval below the median opens downwards, above it upwards, and
    one that holds the median is dropped, because the one-sided event does not restrict
    that coordinate at all. The sliced law and the ray bounds go through the same
    ``price_of_law`` on a stream of their own; with nothing kept the price is zero.
    """
    if not plan.constraints or any(
        o.kind != "constraint" or any(t.on_driver for t in o.terms) for o in spec.observations
    ):
        return None
    rng = np.random.default_rng([seed, dist_sdk.ONE_SIDED_STREAM])
    medians = dist_sdk.median_of_law(law, rng)
    side_low, side_high, kept = dist_sdk.one_sided_box(low, high, medians)
    dropped = tuple(int(k) + 1 for k in range(len(low)) if k not in set(kept.tolist()))
    if not len(kept):
        return OneSided(distance=0.0, log_probability=0.0, standard_error=0.0, dropped=dropped)
    sliced = ConstraintLaw(
        means=law.means[:, kept],
        covs=law.covs[:, kept][:, :, kept],
        log_weight=law.log_weight,
        group=law.group,
    )
    distance, log_probability, standard_error = dist_sdk.price_of_law(
        sliced, side_low, side_high, rng=rng
    )
    return OneSided(
        distance=distance,
        log_probability=log_probability,
        standard_error=standard_error,
        dropped=dropped,
    )


def build(spec, macro, satellites, *, draws, seed, horizon=12, inner=1):
    data = context_from_versions(macro, satellites)
    all_plan = translate.plan_of(spec, data, horizon, groups=draws, inner=inner, seed=seed)
    plan = replace(
        all_plan, interventions=tuple(i for i in all_plan.interventions if i.target in macro.ids)
    )
    driver_interventions = tuple(i for i in all_plan.interventions if i.target not in macro.ids)
    unknown = {i.target for i in driver_interventions} - set(satellites.served())
    if unknown:
        raise ValueError(f"intervention targets not served: {', '.join(sorted(unknown))}")
    rows = translate.observations_of(spec, data, horizon)
    simulation = macro.plugin.simulate(macro.model, plan, draws=draws, seed=seed)
    if macro.plugin.structural_form(macro.model) is None:
        simulation = replace(
            simulation,
            paths=apply_default(simulation.paths, simulation.ids, plan.interventions, inner=inner),
        )
    moments = [row for row in rows if row.kind != "constraint" and not row.on_driver]
    tilt, picked = _tilt(moments, simulation, simulation.paths, simulation.ids, seed)
    if picked is not None:
        simulation = _resample_simulation(simulation, picked)
    registry = load_registry()
    reported = observed(simulation, registry)
    own = np.stack(
        [
            registry.variable(name).from_model(reported.paths[..., k])
            for k, name in enumerate(reported.ids)
        ],
        axis=-1,
    )
    drivers = satellites.simulate(
        reported, sorted(satellites.served()), driver_interventions, seed=seed, inner=inner
    )
    found = dist_of(spec, macro, satellites, draws=draws, seed=seed, horizon=horizon, inner=inner)
    if tilt is not None:
        found = _add_price(found, tilt.price, tilt.standard_error)
    driver_rows = [row for row in rows if row.on_driver]
    if driver_rows:
        simulation, own, drivers, found = _below(
            driver_rows, data, satellites, simulation, own, drivers, found, seed, inner
        )
    checks = {}
    if plan.constraints:
        index = {name: k for k, name in enumerate(simulation.ids)}
        inside = np.ones(len(simulation.paths), dtype=bool)
        for constraint in plan.constraints:
            value = sum(
                t.weight * simulation.paths[:, t.month, index[t.variable]] for t in constraint.terms
            )
            inside &= (constraint.low <= value) & (value <= constraint.high)
        checks["support_share"] = float(inside.mean())
    if tilt is not None:
        checks["means"] = [
            {"mean": float(m), "active": bool(a)}
            for m, a in zip(tilt.means, tilt.active, strict=True)
        ]
        checks["effective_size"] = tilt.effective_size
    return Paths(
        simulation.months,
        simulation.ids,
        own,
        _own_units(satellites, drivers),
        found,
        None if tilt is None else tilt.effective_size,
        checks,
        seed,
        draws,
        simulation.paths,
        simulation.diagnostics,
    )


def _resample_simulation(simulation, picked):
    return replace(
        simulation,
        paths=simulation.paths[picked],
        history=None if simulation.history is None else simulation.history[picked],
        group=None if simulation.group is None else simulation.group[picked],
    )


def _moments(rows, paths, ids):
    values, lows, highs = [], [], []
    index = {name: k for k, name in enumerate(ids)}
    for row in rows:
        value = sum(t.weight * paths[:, t.month, index[t.variable]] for t in row.terms)
        if row.kind == "probability":
            value = ((value >= row.low) & (value <= row.high)).astype(float)
            lows.append(row.at_least)
            highs.append(np.inf)
        else:
            lows.append(-np.inf if row.low is None else row.low)
            highs.append(np.inf if row.high is None else row.high)
        values.append(value)
    return np.column_stack(values), np.asarray(lows), np.asarray(highs)


def _tilt(rows, simulation, paths, ids, seed):
    if not rows:
        return None, None
    group = simulation.group if simulation.group is not None else np.arange(len(paths))
    values, low, high = _moments(rows, paths, ids)
    tilt = overlay.solve_lambda(values, group, low, high)
    if not tilt.feasible:
        raise ValueError("the moment conditions cannot be met on this sample")
    picked = overlay.resample(tilt.weights, group, np.random.default_rng([seed, OVERLAY_STREAM]))
    return tilt, picked


def _add_price(found, price, error, by_level=None):
    if found is None:
        return Dist(price, 0.0, error, by_level=by_level or {})
    levels = {0: found.distance, **by_level} if by_level else found.by_level
    return replace(
        found,
        distance=found.distance + price,
        standard_error=float(np.hypot(found.standard_error, error)),
        by_level=levels,
    )


def _below(rows, data, satellites, simulation, own, drivers, found, seed, inner):
    if inner < 2:
        raise ValueError("driver observations require inner >= 2")
    group = simulation.group if simulation.group is not None else np.arange(len(simulation.paths))
    groups = np.unique(group)
    inside = np.ones(len(group), dtype=bool)
    by_level, terms = {}, np.zeros(len(groups))
    constraints = [row for row in rows if row.kind == "constraint"]
    levels = {
        satellites.registry.quantity(row.terms[0].variable.split(":", 1)[0]).level
        for row in constraints
    }
    for level in sorted(levels):
        before = inside.copy()
        for row in constraints:
            if satellites.registry.quantity(row.terms[0].variable.split(":", 1)[0]).level != level:
                continue
            value = sum(
                t.weight * drivers.paths[:, t.month, drivers.ids.index(t.variable)]
                for t in row.terms
            )
            tolerance = translate.tolerance_of(row.terms, data)
            low = row.value - tolerance if row.point else row.low
            high = row.value + tolerance if row.point else row.high
            inside &= (low <= value) & (value <= high)
        level_terms = []
        for k, g in enumerate(groups):
            kept = inside[group == g].sum()
            if not kept:
                raise ContradictsModels()
            cost = -np.log(kept / before[group == g].sum())
            terms[k] += cost
            level_terms.append(cost)
        by_level[level] = float(np.mean(level_terms))
    weights = inside.astype(float)
    moments = [row for row in rows if row.kind != "constraint"]
    if moments:
        values, low, high = _moments(moments, drivers.paths, drivers.ids)
        alive = np.flatnonzero(inside)
        tilt = overlay.solve_lambda(
            values[alive], np.unique(group[alive], return_inverse=True)[1], low, high
        )
        if not tilt.feasible:
            raise ValueError("the driver moment conditions cannot be met on this sample")
        weights[alive] = tilt.weights
        terms += tilt.by_group
        deepest = max(
            satellites.registry.quantity(t.variable.split(":", 1)[0]).level
            for row in moments
            for t in row.terms
        )
        by_level[deepest] = by_level.get(deepest, 0.0) + tilt.price
    picked = overlay.resample(weights, group, np.random.default_rng([seed, OVERLAY_STREAM, 1]))
    found = _add_price(found, sum(by_level.values()), overlay.error_over_groups(terms), by_level)
    return (
        _resample_simulation(simulation, picked),
        own[picked],
        DriverPaths(drivers.ids, drivers.paths[picked], drivers.flags),
        found,
    )


def validate(spec, macro, satellites, *, as_of: date, corpus: Path | None = None):
    problems, warnings = [], []
    if not spec.approved_by:
        problems.append("the scenario names no approver: a human asserts it before it is computed")
    try:
        plan = plan_of(spec, macro, satellites=satellites)
    except (ValueError, KeyError) as error:
        problems.append(str(error))
    else:
        if plan.constraints:
            try:
                macro.plugin.simulate(macro.model, plan, draws=1, seed=0)
            except ValueError as error:  # ContradictsModels among them: the bands never meet
                problems.append(str(error))
    if satellites.macro != macro.name:
        problems.append(
            f"the satellites were estimated on macro version {satellites.macro}, not {macro.name}"
        )
    marked = [*spec.observations, *spec.interventions]
    if corpus is not None:
        cited = list(spec.sources) + [
            source
            for element in marked
            for source in (element.basis if isinstance(element.basis, list) else ())
        ]
        missing = sources_missing(corpus / "index", cited, as_of)
        if missing:
            problems.append(f"no corpus fragment {', '.join(missing)} visible on {as_of}")
    assumed = sum(item.basis == "assumption" for item in marked)
    stated = sum(item.basis == "event" for item in marked)
    if not spec.sources:
        warnings.append("the narrative cites no source")
    if marked and assumed == len(marked):
        warnings.append("every observation and intervention is an assumption")
    if not marked:
        warnings.append("no observation and no intervention: this is the base scenario")
    return {
        "ok": not problems,
        "problems": problems,
        "labels": {
            "event": stated,
            "sourced": len(marked) - assumed - stated,
            "assumption": assumed,
        },
        "warnings": warnings,
    }


def _own_units(satellites, paths: DriverPaths) -> DriverPaths:
    """The driver paths back in each driver's own unit, as the positions read them."""
    if not paths.ids:
        return DriverPaths(ids=[], paths=np.empty(paths.paths.shape[:2] + (0,)), flags={})
    transformed = [
        from_model(
            satellites.registry.quantity(satellites.quantity_of(driver)).transform,
            paths.paths[..., k],
        )
        for k, driver in enumerate(paths.ids)
    ]
    return DriverPaths(ids=list(paths.ids), paths=np.stack(transformed, axis=-1), flags=paths.flags)
