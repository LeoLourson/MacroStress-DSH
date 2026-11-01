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
from ..model_sdk.dist import ContradictsModels
from . import overlay
from .contracts import Impact
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
    """The scenario's macro conditions and driving shocks as the macro model reads them.

    A constraint that fixes a value exactly arrives as ``value`` and leaves as the interval
    ``value ± tau`` with the point mark, so the model always sees strict bounds and the gates
    can tell a point from a range. A constraint with a term on a driver — a quantity of a
    level above — never reaches the macro model: the overlay carries it, because the macro
    model does not know the driver's law.
    """
    constraints = []
    for constraint in spec.constraints:
        if any(term.on_driver for term in constraint.terms):
            continue
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
        inner=spec.inner,
    )


def driver_constraints(spec: Spec, macro: MacroVersion) -> tuple[LinearConstraint, ...]:
    """The scenario's conditions on the quantities of the levels above the macro variables.

    A mixed condition — macro terms and driver terms together — is refused: the macro part is
    met by the macro model and the driver part by the overlay, and a condition split between
    two imposers is not a condition at all. The bounds follow the same tolerance rule as the
    macro ones.
    """
    rows = []
    for constraint in spec.constraints:
        on_driver = any(term.on_driver for term in constraint.terms)
        on_macro = any(not term.on_driver for term in constraint.terms)
        if on_driver and on_macro:
            raise ValueError(MIXED_CONDITION)
        if not on_driver:
            continue
        terms = tuple(
            LinearTerm(variable=term.variable, month=term.month, weight=term.weight)
            for term in constraint.terms
        )
        if constraint.value is None:
            rows.append(
                LinearConstraint(
                    terms=terms,
                    low=constraint.low,
                    high=constraint.high,
                    point=False,
                    tolerance=tolerance_of(terms, macro),
                )
            )
        else:
            rows.append(point_constraint(terms, constraint.value, macro))
    return tuple(rows)


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

    Raises ``InfeasibleScenario`` when the driving shocks cannot meet the constraints,
    ``ValueError`` when the conditions on means cannot be met by any tilt of the causes,
    and ``ValueError`` naming an impact on a macro variable the version lacks or on a
    driver the satellites do not serve.
    """
    problems = _impact_problems(spec, macro, satellites)
    if problems:
        raise ValueError("; ".join(problems))
    plan = plan_of(spec, macro)
    simulation = macro.plugin.simulate(
        macro.model, plan_with_impacts(spec, macro), draws=draws, seed=seed
    )
    simulation = _with_macro_impacts(simulation, spec.impacts)
    tilt, simulation = _tilted(spec, plan, simulation, seed=seed)
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
    found = dist_of(spec, macro, draws=draws, seed=seed)
    if tilt is not None and found is None:
        # No support conditions: the event is everything, its price is zero and the tilt
        # carries the whole price of the scenario.
        found = Dist(
            distance=tilt.price,
            log_probability=0.0,
            standard_error=0.0,
            causes=tuple(spec.causes),
        )
    elif tilt is not None:
        found = replace(found, distance=found.distance + tilt.price)
    # The conditions on the quantities of the levels above ride the overlay: the support
    # ones zero the paths that break them inside their group of R, and the moment ones tilt
    # the same weights. A group without a surviving path contradicts the models.
    below = _below_levels(spec, macro, satellites, drivers, simulation, seed)
    if below is not None:
        price, picked, by_level, ess = below
        simulation = MacroSimulation(
            ids=simulation.ids,
            months=simulation.months,
            paths=simulation.paths[picked],
            history=None if simulation.history is None else simulation.history[picked],
            diagnostics=simulation.diagnostics,
            group=simulation.group[picked] if simulation.group is not None else None,
        )
        own = own[picked]
        drivers = DriverPaths(
            ids=drivers.ids,
            paths=drivers.paths[picked],
            flags=drivers.flags,
        )
        if found is None:
            # No macro condition to price: the levels above carry the whole price.
            found = Dist(
                distance=price,
                log_probability=0.0,
                standard_error=0.0,
                causes=tuple(spec.causes),
                by_level=by_level,
            )
        else:
            found = replace(
                found, distance=found.distance + price, by_level={0: found.distance, **by_level}
            )
    checks = _checks(spec, plan, simulation, tilt)
    if below is not None and below[3] is not None:
        checks["effective_size"] = below[3]
    return Paths(
        months=simulation.months,
        ids=simulation.ids,
        macro=own,
        drivers=_own_units(satellites, drivers),
        dist=found,
        effective_size=None if tilt is None else tilt.effective_size,
        checks=checks,
        seed=seed,
        draws=draws,
    )


def _names_drivers(spec: Spec) -> bool:
    """Whether any condition of the scenario reads a quantity of a level above."""
    terms = [term for element in (*spec.constraints, *spec.expectations) for term in element.terms]
    terms += [term for condition in spec.probabilities for term in condition.terms]
    return any(term.on_driver for term in terms)


def _mixes_levels(spec: Spec) -> bool:
    """Whether a condition sums macro variables and drivers: no single imposer can meet it."""
    elements = (*spec.constraints, *spec.expectations, *spec.probabilities)
    return any(
        any(term.on_driver for term in element.terms)
        and not all(term.on_driver for term in element.terms)
        for element in elements
    )


def _driver_condition_problems(spec: Spec, macro, satellites) -> list[str]:
    """What keeps the conditions on the quantities above from being carried."""
    problems = []
    served = set(satellites.served())
    terms = [term for element in (*spec.constraints, *spec.expectations) for term in element.terms]
    terms += [term for condition in spec.probabilities for term in condition.terms]
    for term in terms:
        if not term.on_driver:
            continue
        if term.variable not in served:
            problems.append(
                f"the condition names {term.variable}, which the satellites do not serve"
            )
    if _mixes_levels(spec):
        problems.append(MIXED_CONDITION)
    if spec.inner < 2:
        problems.append(
            "the conditions on the quantities of the levels above need inner >= 2: a "
            "single draw of the causes per value of R leaves the overlay nothing to "
            "reweight"
        )
    return problems


def _below_levels(
    spec: Spec,
    macro: MacroVersion,
    satellites: SatelliteBundle,
    drivers: DriverPaths,
    simulation: MacroSimulation,
    seed: int,
):
    """``(price, picked rows, price by level, effective size)`` of the driver conditions.

    The support conditions on drivers zero the paths that break them inside every group of
    ``R`` — a group without a survivor contradicts the models, the case lemma 2 names — and
    the conditions on means tilt the survivors. Both resample the macro and the driver paths
    together, the rows being one draw of the whole system. The support price is the mean of
    ``-log`` of the surviving share over the groups; going through the levels in order splits
    it by the chain rule, every level paying for what it removes from the survivors of the
    levels before it, and the tilt's price goes to the deepest level it reads.
    """
    constraints = driver_constraints(spec, macro)
    moments = [
        element
        for element in (*spec.expectations, *spec.probabilities)
        if any(term.on_driver for term in element.terms)
    ]
    if not constraints and not moments:
        return None
    if _mixes_levels(spec):
        raise ValueError(MIXED_CONDITION)
    if simulation.group is None:
        group = np.arange(simulation.paths.shape[0])
    else:
        group = simulation.group
    if np.all(np.bincount(group) == 1):
        raise ValueError(
            "the conditions on the quantities of the levels above need inner >= 2: a "
            "single draw of the causes per value of R leaves the overlay nothing to "
            "reweight, and every broken path would read as a contradiction"
        )
    groups = np.unique(group)
    inside = np.ones(simulation.paths.shape[0], dtype=bool)
    by_level: dict[int, float] = {}
    levels = {_level_of_term(satellites, c.terms[0].variable) for c in constraints}
    for level in sorted(levels):
        survivors = inside.copy()
        for constraint in constraints:
            if _level_of_term(satellites, constraint.terms[0].variable) != level:
                continue
            value = _functional(constraint, drivers.paths, drivers.ids)
            inside &= (constraint.low <= value) & (value <= constraint.high)
        lost = 0.0
        for g in groups:
            kept = inside[group == g].sum()
            if not kept:
                raise ContradictsModels()
            lost -= np.log(kept / survivors[group == g].sum())
        by_level[level] = float(lost / len(groups))
    price = sum(by_level.values())
    weights = inside.astype(float)
    ess = None
    if moments:
        values, lows, highs = _moment_conditions(
            moments,
            lambda term: _term_of(term, drivers.paths, drivers.ids),
            simulation.paths.shape[0],
        )
        alive = np.flatnonzero(inside)
        tilt = overlay.solve_lambda(
            values[alive], np.unique(group[alive], return_inverse=True)[1], lows, highs
        )
        if not tilt.feasible:
            raise ValueError(
                "the conditions on the means of the drivers cannot be met: no tilt of the "
                "causes brings them inside their bounds on this sample"
            )
        weights[alive] = tilt.weights
        deepest = max(
            _level_of_term(satellites, term.variable)
            for element in moments
            for term in element.terms
        )
        by_level[deepest] = by_level.get(deepest, 0.0) + tilt.price
        price += tilt.price
        ess = tilt.effective_size
    picked = overlay.resample(
        weights, group, np.random.default_rng([seed, OVERLAY_STREAM, 1])
    )
    return price, picked, by_level, ess


def _moment_conditions(chosen, term_paths, rows: int):
    """``(values, lows, highs)``: the functionals of the conditions on means and probabilities.

    ``values`` is ``(rows, k)``. A condition ``Q(B) >= p`` is the mean of ``1_B`` above ``p``.
    """
    values, lows, highs = [], [], []
    for element in chosen:
        value = np.zeros(rows)
        for term in element.terms:
            value += term.weight * term_paths(term)
        if hasattr(element, "at_least"):
            values.append(((value >= element.low) & (value <= element.high)).astype(float))
            lows.append(element.at_least)
            highs.append(np.inf)
        else:
            values.append(value)
            lows.append(element.low if element.low is not None else -np.inf)
            highs.append(element.high if element.high is not None else np.inf)
    return np.column_stack(values), np.array(lows), np.array(highs)


def _functional(constraint, paths: np.ndarray, ids: list[str]) -> np.ndarray:
    """The value of a condition's functional on every path."""
    return sum(term.weight * _term_of(term, paths, ids) for term in constraint.terms)


def _term_of(term, paths: np.ndarray, ids: list[str]) -> np.ndarray:
    """The paths of one driver term, in the model scale the conditions read."""
    if term.variable not in ids:
        raise ValueError(
            f"the condition names {term.variable}, which the run does not simulate"
        )
    return paths[:, term.month, ids.index(term.variable)]


def _level_of_term(satellites: SatelliteBundle, driver: str) -> int:
    """The level of the quantity a driver belongs to."""
    quantity, _ = driver.split(":", 1)
    return satellites.registry.quantity(quantity).level


def _tilted(
    spec: Spec, plan: ScenarioPlan, simulation: MacroSimulation, *, seed: int
):
    """``(tilt, simulation)``: the conditions on means and probabilities imposed on the paths.

    The functionals are measured on the final paths — impacts included, model scale — and
    the tilt reweights the draws of the causes inside every group of ``R``. The weights
    then turn into a plain sample by systematic resampling inside the groups, so the
    percentiles, the scores and the position engine read equal weights again; the
    effective size and the checks tell what the weights did. The hidden coordinates are
    chosen by the support box once per group, and the tilt reweights the draws of the
    causes — the approximation the overlay lives with.
    """
    # The conditions on drivers wait for the drivers: the overlay of the levels above takes them.
    chosen = [
        element
        for element in (*spec.expectations, *spec.probabilities)
        if not any(term.on_driver for term in element.terms)
    ]
    if not chosen:
        return None, simulation
    index = {id_: k for k, id_ in enumerate(simulation.ids)}
    values, lows, highs = _moment_conditions(
        chosen,
        lambda term: simulation.paths[:, term.month, index[term.variable]],
        simulation.paths.shape[0],
    )
    group = (
        simulation.group
        if simulation.group is not None
        else np.arange(simulation.paths.shape[0])
    )
    tilt = overlay.solve_lambda(values, group, lows, highs)
    if not tilt.feasible:
        raise ValueError(
            "the conditions on means cannot be met: no tilt of the causes brings them "
            "inside their bounds on this sample"
        )
    picked = overlay.resample(
        tilt.weights, group, np.random.default_rng([seed, OVERLAY_STREAM])
    )
    resampled = MacroSimulation(
        ids=simulation.ids,
        months=simulation.months,
        paths=simulation.paths[picked],
        history=None if simulation.history is None else simulation.history[picked],
        diagnostics=simulation.diagnostics,
        group=group[picked],
    )
    return tilt, resampled


def _checks(spec: Spec, plan: ScenarioPlan, simulation: MacroSimulation, tilt) -> dict:
    """What the sample says about the imposed conditions (the canon, section 7).

    The share of paths inside the support event, the means of the conditioned functionals
    against their bounds, and the multipliers against the theorem: a positive one rests on
    the lower bound, a negative one on the upper, and the effective size says how many
    paths carry the tilt.
    """
    checks: dict = {}
    if plan.constraints:
        index = {id_: k for k, id_ in enumerate(simulation.ids)}
        inside = np.ones(simulation.paths.shape[0], dtype=bool)
        for constraint in plan.constraints:
            value = np.zeros(simulation.paths.shape[0])
            for term in constraint.terms:
                value += term.weight * simulation.paths[:, term.month, index[term.variable]]
            inside &= (constraint.low <= value) & (value <= constraint.high)
        checks["support_share"] = float(inside.mean())
    if tilt is not None:
        checks["means"] = [
            {
                "mean": float(mean),
                "active": bool(active),
            }
            for mean, active in zip(tilt.means, tilt.active, strict=True)
        ]
        checks["effective_size"] = tilt.effective_size
    return checks


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
    if _names_drivers(spec):
        problems += _driver_condition_problems(spec, macro, satellites)
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
            for element in _marked(spec)
            for source in (element.basis if not isinstance(element.basis, str) else ())
        ]
        missing = sources_missing(corpus / "index", cited, as_of)
        if missing:
            problems.append(f"no corpus fragment {', '.join(missing)} visible on {as_of}")
    marks = _labels(spec)
    return {
        "ok": not problems,
        "problems": problems,
        "labels": marks["labels"],
        "warnings": marks["warnings"] + _reach(spec, satellites),
    }


def _marked(spec: Spec) -> list:
    """The elements of ``O`` and ``C``: the canon marks each with the event, a source or an
    assumption."""
    return [*spec.constraints, *spec.expectations, *spec.probabilities, *spec.impacts]


def _reach(spec: Spec, satellites: SatelliteBundle) -> list[str]:
    """Where the impacts of the scenario do not go: the macro level, and the drivers below."""
    registry = satellites.registry
    served = set(satellites.served())
    targets = {impact.target for impact in spec.impacts}
    warnings = []
    for target in sorted(targets):
        if ":" not in target:
            readers = [
                quantity.id
                for quantity in registry.quantities
                if target in quantity.regressors
                or any(feature.variable == target for feature in quantity.features)
            ]
            warnings.append(
                f"impact on {target}: the other macro variables do not react to it, and "
                f"{'only ' + ', '.join(readers) + ' read' if readers else 'no satellite reads'} "
                "it; if the effect must spread, write it as a condition"
            )
            continue
        quantity, _, object_ = target.partition(":")
        below = sorted(
            driver.id
            for driver in registry.drivers
            if driver.quantity == quantity
            and driver.id in served
            and driver.id not in targets
            and object_ in registry.ancestors(driver.object)
        )
        if below:
            warnings.append(
                f"impact on {target} does not reach the {len(below)} served drivers below it: "
                f"{', '.join(below)}; to move them, list them"
            )
    return warnings


def _labels(spec: Spec) -> dict:
    """How the elements are marked, and what a valid scenario still leaves to the reader.

    Nothing here keeps a scenario from being computed: an assumption is an honest mark, and
    only the human who approves can tell whether a cited fragment gives the number.
    """
    marked = _marked(spec)
    assumed = sum(element.basis == "assumption" for element in marked)
    stated = sum(element.basis == "event" for element in marked)
    warnings = []
    if not spec.sources:
        warnings.append("the narrative cites no source")
    if marked and assumed == len(marked):
        warnings.append("every condition and impact is an assumption: none rests on the corpus")
    conditioned = spec.constraints or spec.expectations or spec.probabilities or spec.regime_event
    if not conditioned and not spec.impacts:
        warnings.append("no condition and no impact: this is the base s_0, it records no event")
    elif not conditioned:
        warnings.append(
            "no condition: Q = P, so the price is zero and says nothing about the impacts"
        )
    return {
        "labels": {
            "event": stated,
            "sourced": len(marked) - assumed - stated,
            "assumption": assumed,
        },
        "warnings": warnings,
    }


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
    return replace(simulation, paths=paths)


def _version_problems(macro: MacroVersion, satellites: SatelliteBundle) -> list[str]:
    """The satellites of another macro version are not the ones these paths were drawn on."""
    if satellites.macro != macro.name:
        return [
            f"the satellites were estimated on the macro version {satellites.macro}, "
            f"the scenario names {macro.name}"
        ]
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
