"""The paths of a scenario: the conditional forecast, then the drivers under it.

``build`` asks the macro plugin of the scenario's macro version to draw ``draws`` paths of
the macro variables under the constraints, turns them back into each variable's own unit,
and drives the satellites of the same macro version over exactly those draws with the
overlays of the scenario. The macro month ``t = 0`` is the last month of the version's panel,
so a run is a function of the scenario, the two model versions and the seed alone.

Severity ``delta`` of the scenario comes from the unconditional forecast of the same model:
the distance of the constrained paths from what the model expects without them. A scenario
without constraints has no severity, and a macro model that does not measure it leaves the
severity empty.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from ..corpus.index import sources_missing
from ..drivers.bundle import SatelliteBundle
from ..drivers.registry import from_model
from ..macro.registry import load_registry
from ..macro.version import MacroVersion
from ..model_sdk.conditional import InfeasibleScenario
from ..model_sdk.contracts import (
    AdditiveImpact,
    DriverPaths,
    LinearConstraint,
    LinearTerm,
    ScenarioPlan,
    Severity,
)
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
    ``ValueError`` naming an overlay on a driver the satellites do not serve.
    """
    plan = plan_of(spec)
    simulation = macro.plugin.simulate(macro.model, plan, draws=draws, seed=seed)
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
        _drivers(satellites, spec),
        [
            AdditiveImpact(
                target=overlay.driver,
                start=overlay.start,
                end=overlay.end,
                delta=overlay.delta,
            )
            for overlay in spec.overlays
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
    scenario must be published by then. ``corpus`` is the corpus root; without it the sources
    are not checked.
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
    problems += _overlay_problems(satellites, spec)
    problems += _version_problems(macro, satellites)
    if spec.constraints:
        try:
            macro.plugin.simulate(macro.model, plan_of(spec), draws=1, seed=0)
        except (InfeasibleScenario, ValueError) as error:
            problems.append(str(error))
    if corpus is not None and spec.sources:
        missing = sources_missing(corpus / "index", list(spec.sources), as_of)
        if missing:
            problems.append(f"no corpus fragment {', '.join(missing)} visible on {as_of}")
    return {"ok": not problems, "problems": problems}


def _drivers(satellites: SatelliteBundle, spec: Spec) -> list[str]:
    """The drivers the run simulates: everything the satellites serve, in a stable order."""
    served = sorted(satellites.served())
    problems = _overlay_problems(satellites, spec)
    if problems:
        raise ValueError("; ".join(problems))
    return served


def _overlay_problems(satellites: SatelliteBundle, spec: Spec) -> list[str]:
    """Overlays on a driver the satellite version does not simulate, with the reason."""
    served = set(satellites.served())
    problems = []
    for overlay in spec.overlays:
        if overlay.driver in served:
            continue
        try:
            quantity, node = satellites.node(overlay.driver)
        except KeyError:
            problems.append(
                f"overlay on {overlay.driver}: the satellite version knows no such driver"
            )
            continue
        reason = satellites.checks[quantity][node].reason
        problems.append(f"overlay on {overlay.driver}, which is not served: {reason}")
    return problems


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
    return DriverPaths(
        ids=list(paths.ids), paths=np.stack(transformed, axis=-1), flags=paths.flags
    )
