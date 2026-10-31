"""The paths of a scenario: the conditional forecast, then the drivers under it.

``build`` asks the macro plugin of the scenario's macro version to draw ``draws`` paths of
the macro variables under the constraints, adds the scenario's impacts on the macro variables
(level 0, before anything below it is simulated), turns the paths back into each variable's
own unit, and drives the satellites of the same macro version over exactly those draws with
the impacts on the drivers. The macro month ``t = 0`` is the last month of the version's
panel, so a run is a function of the scenario, the two model versions and the seed alone.

A level-0 impact is added after the macro model has drawn the paths, while the scheme puts the
constraints on ``F(omega, O)``; so the plan the model solves has its bounds shifted by the
impact of every term, and the constraint holds on the paths the run finally reports
(``plan_with_impacts``).

Severity ``delta`` of the scenario comes from the unconditional forecast of the same model:
the distance of the constrained paths from what the model expects without them, measured on
``C`` alone, without the level-0 impacts. A scenario without constraints has no severity, and
a macro model that does not measure it leaves the severity empty.

The platform's ``dist(P_s, P)`` lives beside it (``mst.model_sdk.dist``): the same unconditional
forecast, but the probability of the whole constraint box, not the Mahalanobis distance of the
midpoints. The unconditional plan is empty (``s_0``), so the same ``draws`` and ``seed`` that
the constraint gate reads draw its paths. The estimate is the Monte-Carlo share of the paths
that satisfy every constraint at once; when no path does, the exact probability of the
parameter mixture takes over for a model that measures the per-draw moments of the constraint
functionals (``constraint_moments``).
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
    Severity,
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
    severity: Severity | None
    dist: Dist | None
    seed: int
    draws: int


def plan_of(spec: Spec) -> ScenarioPlan:
    """The scenario's constraints and driving shocks as the macro model reads them."""
    return ScenarioPlan(
        horizon=spec.horizon,
        constraints=tuple(
            LinearConstraint(
                terms=tuple(
                    LinearTerm(variable=term.variable, month=term.month, weight=term.weight)
                    for term in constraint.terms
                ),
                low=constraint.low,
                high=constraint.high,
            )
            for constraint in spec.constraints
        ),
        driving_shocks=tuple(spec.driving_shocks),
    )


def plan_with_impacts(spec: Spec) -> ScenarioPlan:
    """The plan the macro model solves: ``C`` moved by the level-0 impacts of ``spec``.

    An impact ``o_0`` is added to the paths after ``G_0`` has drawn them, so the model must
    meet ``low <= w'y <= high`` on the unshifted path with the impact taken out:
    ``low - w'o_0 <= w'y <= high - w'o_0`` for every term on an impacted variable and month.
    The run then adds ``o_0`` to the paths and ``C`` holds on ``F(omega, O)``.
    """
    plan = plan_of(spec)
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
    """``dist(P_s, P)`` of the scenario: ``-log P(C)`` on the unconditional run ``s_0``.

    The plan is empty, so the paths are the pure model without constraints or impacts — the
    same ``draws`` and ``seed`` the constraint gate ``pi_c`` reads. The estimate is the
    Monte-Carlo share of the paths that satisfy every constraint at once; when no drawn path
    does, the share cannot tell a rare box from an impossible one, and the exact probability
    of the mixture takes over — per drawn parameter set the constraint functionals are
    normal (``constraint_moments``), the box probability follows from the gaussian cdf.
    A model without that ability leaves the Monte-Carlo estimate of zero, an infinite dist.
    ``None`` — the scenario has no constraints, so it has no dist at all rather than a zero
    distance.
    """
    plan = plan_of(spec)
    if not plan.constraints:
        return None
    base = macro.plugin.simulate(
        macro.model, ScenarioPlan(horizon=spec.horizon), draws=draws, seed=seed
    )
    share = dist_sdk.box_probability_mc(base.paths, base.ids, plan.constraints)
    if share > 0.0:
        return _dist(share, "mc")
    moments = macro.plugin.constraint_moments(macro.model, plan, draws=draws, seed=seed)
    if moments is not None:
        means, covs = moments
        try:
            return _dist(dist_sdk.box_probability_mixture(means, covs, plan.constraints), "exact")
        except (ValueError, np.linalg.LinAlgError):
            pass  # a box the mixture cannot integrate: the Monte-Carlo zero stands
    return _dist(share, "mc")


def _dist(probability: float, way: str) -> Dist:
    """The ``Dist`` of one estimate; a zero probability is named, not silently passed as 0."""
    return Dist(
        probability=probability,
        distance=dist_sdk.dist_from_probability(probability),
        way=way,
        message=(
            "the unconditional model gives the constraint box no probability; the constraint "
            "gate pi_c is failed"
            if probability <= 0.0
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
    plan = plan_of(spec)
    simulation = macro.plugin.simulate(macro.model, plan_with_impacts(spec), draws=draws, seed=seed)
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
        severity=(
            macro.plugin.severity(macro.model, plan, draws=draws, seed=seed)
            if spec.constraints
            else None
        ),
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
    problems += [
        f"unknown shock {shock}: the macro version knows {', '.join(macro.shocks)}"
        for shock in spec.driving_shocks
        if shock not in macro.shocks
    ]
    problems += _impact_problems(spec, macro, satellites)
    problems += _version_problems(macro, satellites)
    if spec.constraints:
        try:
            macro.plugin.simulate(macro.model, plan_with_impacts(spec), draws=1, seed=0)
        except (InfeasibleScenario, ValueError) as error:
            problems.append(str(error))
    if corpus is not None:
        cited = list(spec.sources) + [
            source for impact in spec.impacts for source in impact.sources
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
