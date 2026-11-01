"""The paths of a scenario: the conditional forecast, then the drivers under it.

``build`` asks the macro plugin of the scenario's macro version to draw ``draws`` paths of
the macro variables under the constraints, adds the scenario's impacts on the macro variables
(level 0, before anything below it is simulated), turns the paths back into each variable's
own unit, and drives the satellites of the same macro version over exactly those draws with
the impacts on the drivers. The macro month ``t = 0`` is the last month of the version's
panel, so a run is a function of the scenario, the two model versions and the seed alone.

A level-0 impact is added after the macro model has drawn the paths, while the canon puts the
constraints on ``F(omega, O)``; so the plan the model solves has its bounds shifted by the
impact of every term, and the constraint holds on the paths the run finally reports
(``plan_with_impacts``).

A constraint that fixes a value exactly — a point — is expanded to ``value ± tau`` with
``tau`` the tenth of the typical monthly movement of its terms (``tolerance_of``), so every
plan carries strict intervals and a point keeps a probability the price can measure. The
price itself — ``dist(s) = E_P[-log P(S | R)]`` of the canon — is measured on the law of the
constraint functionals the model returns (``constraint_law``), against the impact-shifted
bounds: the event the paths are built on is the event the price measures. A scenario whose
event has no probability at some value of ``R`` contradicts the models, and its price is
infinite with the message saying so.
"""

from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import numpy as np

from ..corpus.index import sources_missing
from ..drivers.bundle import SatelliteBundle
from ..drivers.registry import from_model
from ..macro.registry import load_registry
from ..macro.version import MacroVersion
from ..metrics.cells import typical_movement
from ..model_sdk import dist as dist_sdk
from ..model_sdk.conditional import InfeasibleScenario
from ..model_sdk.contracts import (
    AdditiveImpact,
    Dist,
    DriverPaths,
    LinearConstraint,
    LinearTerm,
    MacroSimulation,
    ScenarioPlan,
)
from .contracts import Impact
from .spec import Spec


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
    seed: int
    draws: int


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


def plan_of(spec: Spec, macro: MacroVersion) -> ScenarioPlan:
    """The scenario's constraints and driving shocks as the macro model reads them.

    A constraint that fixes a value exactly arrives as ``value`` and leaves as the interval
    ``value ± tau`` with the point mark, so the model always sees strict bounds and the gates
    can tell a point from a range.
    """
    constraints = []
    for constraint in spec.constraints:
        terms = tuple(
            LinearTerm(variable=term.variable, month=term.month, weight=term.weight)
            for term in constraint.terms
        )
        if constraint.value is None:
            constraints.append(
                LinearConstraint(
                    terms=terms,
                    low=constraint.low,
                    high=constraint.high,
                    point=False,
                    tolerance=tolerance_of(terms, macro),
                )
            )
        else:
            constraints.append(point_constraint(terms, constraint.value, macro))
    return ScenarioPlan(
        horizon=spec.horizon,
        constraints=tuple(constraints),
        causes=tuple(spec.causes),
        regime_event=spec.regime_event,
    )


def plan_with_impacts(spec: Spec, macro: MacroVersion) -> ScenarioPlan:
    """The plan the macro model solves: ``C`` moved by the level-0 impacts of ``spec``.

    An impact ``o_0`` is added to the paths after ``G_0`` has drawn them, so the model must
    meet ``low <= w'y <= high`` on the unshifted path with the impact taken out:
    ``low - w'o_0 <= w'y <= high - w'o_0`` for every term on an impacted variable and month.
    The run then adds ``o_0`` to the paths and ``C`` holds on ``F(omega, O)`` — and the price
    reads the same shifted bounds, so it measures the event the paths carry.
    """
    plan = plan_of(spec, macro)
    of_macro = [impact for impact in spec.impacts if not impact.on_driver]
    if not of_macro:
        return plan
    shifted = []
    for constraint in plan.constraints:
        shift = _impact_shift(constraint, of_macro)
        shifted.append(
            replace(constraint, low=constraint.low - shift, high=constraint.high - shift)
        )
    return replace(plan, constraints=tuple(shifted))


def _impact_shift(constraint: LinearConstraint, impacts: list[Impact]) -> float:
    """``w'o_0``: what the level-0 impacts add to the left side of one constraint."""
    return sum(
        term.weight * impact.delta
        for impact in impacts
        for term in constraint.terms
        if term.variable == impact.target and impact.start <= term.month <= impact.end
    )


def dist_of(
    spec: Spec, macro: MacroVersion, *, draws: int, seed: int
) -> Dist | None:
    """The price of the scenario: ``dist(s) = E_P[-log P(S | R)]`` of the canon.

    The law of the constraint functionals given ``R`` and the hidden coordinates comes from
    the macro model (``constraint_law``); the bounds are the ones the plan carries with the
    level-0 impacts taken out — the same event the paths are built on. ``None`` when the
    scenario has nothing to price (no constraints, no event of a hidden coordinate) or the
    model does not measure the law. An event some value of ``R`` cannot reach gives an
    infinite distance with the message naming lemma 2; the standard error reads the spread
    of ``-log P(S | R)`` over the groups, so a price with a large error is seen as such.
    """
    plan = plan_with_impacts(spec, macro)
    if not plan.constraints and not plan.causes and plan.regime_event is None:
        return None
    law = macro.plugin.constraint_law(macro.model, plan, draws=draws, seed=seed)
    if law is None:
        return None
    low = np.array([constraint.low for constraint in plan.constraints])
    high = np.array([constraint.high for constraint in plan.constraints])
    rng = np.random.default_rng([seed, dist_sdk.DIST_STREAM])
    distance, log_probability, standard_error = dist_sdk.price_of_law(
        law, low, high, rng=rng
    )
    return Dist(
        distance=distance,
        log_probability=log_probability,
        standard_error=standard_error,
        causes=tuple(spec.causes),
        tolerances=tuple(
            constraint.tolerance for constraint in plan.constraints if constraint.point
        ),
        message=(
            "the event has no probability at some value of R: the scenario contradicts "
            "the models (lemma 2 of the canon)"
            if not np.isfinite(distance)
            else None
        ),
    )


def build(
    spec: Spec,
    macro: MacroVersion,
    satellites: SatelliteBundle,
    *,
    draws: int,
    seed: int,
) -> Paths:
    """The paths of ``spec`` from its macro version and the satellites of that version.

    Raises ``InfeasibleScenario`` when the driving shocks cannot meet the constraints, and
    ``ValueError`` naming an impact on a macro variable the version lacks or on a driver the
    satellites do not serve.
    """
    problems = _impact_problems(spec, macro, satellites)
    if problems:
        raise ValueError("; ".join(problems))
    plan = plan_of(spec, macro)
    simulation = macro.plugin.simulate(
        macro.model, plan_with_impacts(spec, macro), draws=draws, seed=seed
    )
    simulation = _with_macro_impacts(simulation, spec.impacts)
    registry = load_registry()
    own = np.stack(
        [
            registry.variable(id_).from_model(simulation.paths[..., k])
            for k, id_ in enumerate(simulation.ids)
        ],
        axis=-1,
    )
    drivers = satellites.simulate(
        simulation,
        _drivers(satellites, spec, macro),
        [
            AdditiveImpact(
                target=impact.target, start=impact.start, end=impact.end, delta=impact.delta
            )
            for impact in spec.impacts
            if impact.on_driver
        ],
        seed=seed,
    )
    return Paths(
        months=simulation.months,
        ids=simulation.ids,
        macro=own,
        drivers=_own_units(satellites, drivers),
        dist=dist_of(spec, macro, draws=draws, seed=seed),
        seed=seed,
        draws=draws,
    )


def validate(
    spec: Spec,
    macro: MacroVersion,
    satellites: SatelliteBundle,
    *,
    as_of: date,
    corpus: Path | None = None,
) -> dict:
    """Every problem that keeps the scenario from being computed, each one named.

    ``as_of`` is the calculation day of the macro version: a corpus fragment cited by the
    scenario or by one of its impacts must be published by then. ``corpus`` is the corpus root;
    without it the sources are not checked.
    """
    problems = []
    try:
        load_registry().check(spec)
    except ValueError as error:
        problems.append(str(error))
    allowed = tuple(macro.shocks) + tuple(macro.latents)
    problems += [
        f"unknown cause {cause}: the macro version knows {', '.join(allowed) or 'none'}"
        for cause in spec.causes
        if cause not in allowed
    ]
    if spec.regime_event is not None:
        if spec.regime_event not in macro.shocks:
            problems.append(
                f"unknown regime event {spec.regime_event}: the macro version knows "
                f"{', '.join(macro.shocks)}"
            )
        elif spec.causes and not (set(macro.latents) & set(spec.causes)):
            problems.append(
                f"the regime event {spec.regime_event} needs a hidden coordinate "
                f"({', '.join(macro.latents) or 'none declared'}) among its causes, or no "
                "causes at all: the event fixes the regime, so its law cannot stay as it "
                "was (lemma 2 of the canon)"
            )
    if not spec.approved_by:
        problems.append(
            "the scenario names no approver: a human asserts it before it is computed"
        )
    problems += _impact_problems(spec, macro, satellites)
    problems += _version_problems(macro, satellites)
    if spec.constraints:
        try:
            macro.plugin.simulate(
                macro.model, plan_with_impacts(spec, macro), draws=1, seed=0
            )
        except (InfeasibleScenario, ValueError) as error:
            problems.append(str(error))
    if corpus is not None:
        cited = list(spec.sources) + [
            source
            for impact in spec.impacts
            for source in (impact.basis if impact.basis != "assumption" else ())
        ]

        missing = sources_missing(corpus / "index", cited, as_of)
        if missing:
            problems.append(f"no corpus fragment {', '.join(missing)} visible on {as_of}")
    return {"ok": not problems, "problems": problems}


def _drivers(satellites: SatelliteBundle, spec: Spec, macro: MacroVersion) -> list[str]:
    """The drivers the run simulates: everything the satellites serve, in a stable order."""
    served = sorted(satellites.served())
    problems = _impact_problems(spec, macro, satellites)
    if problems:
        raise ValueError("; ".join(problems))
    return served


def _impact_problems(spec: Spec, macro: MacroVersion, satellites: SatelliteBundle) -> list[str]:
    """Impacts on a variable the macro version lacks or a driver the satellites do not serve."""
    served = set(satellites.served())
    problems = []
    for impact in spec.impacts:
        if not impact.on_driver:
            if impact.target not in macro.ids:
                problems.append(
                    f"impact on {impact.target}: the macro version has no such variable"
                )
            continue
        if impact.target in served:
            continue
        try:
            quantity, node = satellites.node(impact.target)
        except KeyError:
            problems.append(
                f"impact on {impact.target}: the satellite version knows no such driver"
            )
            continue
        reason = satellites.checks[quantity][node].reason
        problems.append(f"impact on {impact.target}, which is not served: {reason}")
    return problems


def _with_macro_impacts(simulation: MacroSimulation, impacts: list[Impact]) -> MacroSimulation:
    """The macro paths with the scenario's impacts on the macro variables added.

    The impact is in the model scale of its variable; the satellites below see the shifted
    paths, as the scheme prescribes: an impact on a level enters before the levels under it.
    """
    of_macro = [impact for impact in impacts if not impact.on_driver]
    if not of_macro:
        return simulation
    paths = simulation.paths.copy()
    for impact in of_macro:
        k = simulation.ids.index(impact.target)
        paths[:, impact.start : impact.end + 1, k] += impact.delta
    return MacroSimulation(
        ids=simulation.ids,
        months=simulation.months,
        paths=paths,
        history=simulation.history,
    )


def _version_problems(macro: MacroVersion, satellites: SatelliteBundle) -> list[str]:
    """The satellites of another macro version are not the ones these paths were drawn on."""
    last = satellites.part_months_end()
    if last is None or last == macro.months[-1]:
        return []
    return [
        f"the satellites are estimated up to {last:%Y-%m}, "
        f"the macro version ends at {macro.months[-1]:%Y-%m}"
    ]


def _own_units(satellites: SatelliteBundle, paths: DriverPaths) -> DriverPaths:
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
