"""The paths of a scenario: the conditional forecast, then the drivers under it.

``build`` draws ``draws`` paths of the macro variables under the constraints (``mst.macro``),
turns them back into each variable's own unit, and drives the satellites of the same macro
version over exactly those draws (``mst.drivers``) with the overlays of the scenario. The macro
month ``t = 0`` is the last month of the version's panel, so a run is a function of the
scenario, the two model versions and the seed alone.

Severity ``delta`` of the scenario comes from the unconditional forecast of the same version:
the distance of the constrained paths from what the model expects without them. A scenario
without constraints has no severity.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from ..corpus.index import sources_missing
from ..drivers.registry import from_model
from ..drivers.simulate import DriverPaths
from ..drivers.simulate import simulate as simulate_drivers
from ..drivers.version import Satellites
from ..macro.estimate import Posterior
from ..macro.forecast import InfeasibleScenario, Severity, severity
from ..macro.forecast import simulate as simulate_macro
from ..macro.registry import load_registry
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


def build(
    spec: Spec, model: Posterior, satellites: Satellites, *, draws: int, seed: int
) -> Paths:
    """The paths of ``spec`` from the macro version ``model`` and its ``satellites``.

    Raises ``InfeasibleScenario`` when the driving shocks cannot meet the constraints, and
    ``ValueError`` naming an overlay on a driver the satellites do not serve.
    """
    simulation = simulate_macro(model, spec, draws=draws, seed=seed)
    registry = load_registry()
    macro = np.stack(
        [
            registry.variable(id_).from_model(simulation.paths[..., k])
            for k, id_ in enumerate(simulation.ids)
        ],
        axis=-1,
    )
    drivers = simulate_drivers(
        satellites, simulation, _drivers(satellites, spec), spec.overlays, seed=seed
    )
    return Paths(
        months=simulation.months,
        ids=simulation.ids,
        macro=macro,
        drivers=_own_units(satellites, drivers),
        severity=severity(model, spec, draws=draws, seed=seed) if spec.constraints else None,
        seed=seed,
        draws=draws,
    )


def validate(
    spec: Spec,
    model: Posterior,
    satellites: Satellites,
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
    problems += _overlay_problems(satellites, spec)
    problems += _version_problems(model, satellites)
    if spec.constraints:
        try:
            simulate_macro(model, spec, draws=1, seed=0)
        except (InfeasibleScenario, ValueError) as error:
            problems.append(str(error))
    if corpus is not None and spec.sources:
        missing = sources_missing(corpus / "index", list(spec.sources), as_of)
        if missing:
            problems.append(f"no corpus fragment {', '.join(missing)} visible on {as_of}")
    return {"ok": not problems, "problems": problems}


def _drivers(satellites: Satellites, spec: Spec) -> list[str]:
    """The drivers the run simulates: everything the satellites serve, in a stable order."""
    served = sorted(satellites.served())
    problems = _overlay_problems(satellites, spec)
    if problems:
        raise ValueError("; ".join(problems))
    return served


def _overlay_problems(satellites: Satellites, spec: Spec) -> list[str]:
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


def _version_problems(model: Posterior, satellites: Satellites) -> list[str]:
    """The satellites of another macro version are not the ones these paths were drawn on."""
    last = next(iter(satellites.fits.values()), None)
    if last is None or last.history.months[-1] == model.months[-1]:
        return []
    return [
        f"the satellites are estimated up to {last.history.months[-1]:%Y-%m}, "
        f"the macro version ends at {model.months[-1]:%Y-%m}"
    ]


def _own_units(satellites: Satellites, paths: DriverPaths) -> DriverPaths:
    """The driver paths back in each driver's own unit, as the positions read them."""
    if not paths.ids:
        return DriverPaths(ids=[], paths=np.empty(paths.paths.shape[:2] + (0,)), flags={})
    transformed = [
        from_model(
            satellites.registry.quantity(satellites.node(driver)[0]).transform,
            paths.paths[..., k],
        )
        for k, driver in enumerate(paths.ids)
    ]
    return DriverPaths(
        ids=list(paths.ids), paths=np.stack(transformed, axis=-1), flags=paths.flags
    )
