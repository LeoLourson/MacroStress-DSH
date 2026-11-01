"""The conformance check of the model SDK's shock contract, for a plugin's own tests.

A satellite model receives the standardized residuals of its correlated drivers from the
platform (``simulate(..., shocks=...)``) and must use them instead of drawing those residuals
itself; otherwise the measured cross-model correlation never reaches the joint paths.
``check_shocks`` calls ``simulate`` twice with different platform shocks on the same seed and
stream and reports a finding when the paths come out equal: the model ignored the shocks. A
model that returns other drivers or other shapes is reported too.

The plan of the metrics work named ``mst.plugin_sdk.testing`` for this check, but that module
conforms position plugins (declarations and the demo calculation); the shock contract belongs
to the model SDK, so the helper lives here.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .contracts import AdditiveImpact, MacroSimulation, SatelliteModel


@dataclass(frozen=True)
class Finding:
    """One problem the shock conformance check found."""

    check: str
    message: str

    def __str__(self) -> str:
        return f"{self.check}: {self.message}"


def check_shocks(
    model: SatelliteModel,
    loaded,
    macro: MacroSimulation,
    drivers: Sequence[str],
    *,
    correlated: Sequence[str] | None = None,
    history: int | None = None,
    impacts: Sequence[AdditiveImpact] = (),
    seed: int = 1,
    stream: int = 1,
) -> list[Finding]:
    """Every problem with how ``model`` uses the platform's ``shocks``; empty means it conforms.

    ``drivers`` is the argument the platform passes to ``simulate``; ``correlated`` names the
    drivers with a history of their own, in that order (default: every driver) — only those are
    columns of ``shocks``. ``history`` is the number of months the model's part records, which
    fixes the platform's row count ``horizon + max(history - 1, 0)``, as
    ``mst.drivers.bundle`` draws it; without it the helper supplies the horizon alone.

    A deterministic model whose paths do not depend on any residual is reported as ignoring the
    shocks: it cannot honour the contract either.
    """
    names = set(drivers) if correlated is None else set(correlated)
    columns = [driver for driver in drivers if driver in names]
    if not columns:
        return []
    horizon = macro.paths.shape[1] - 1
    rows = horizon + max((history or 1) - 1, 0)
    first_shocks = np.random.default_rng([seed, 0, 0]).standard_normal(
        (macro.paths.shape[0], rows, len(columns))
    )
    second_shocks = np.random.default_rng([seed, 0, 1]).standard_normal(
        (macro.paths.shape[0], rows, len(columns))
    )
    try:
        first = model.simulate(
            loaded,
            macro,
            list(drivers),
            list(impacts),
            shocks=first_shocks,
            seed=seed,
            stream=stream,
        )
    except Exception as error:  # noqa: BLE001 - a broken model is a finding
        return [Finding("simulate", f"{type(error).__name__}: {error}")]
    try:
        second = model.simulate(
            loaded,
            macro,
            list(drivers),
            list(impacts),
            shocks=second_shocks,
            seed=seed,
            stream=stream,
        )
    except Exception as error:  # noqa: BLE001 - a model that rejects shocks is a finding
        return [
            Finding(
                "shocks",
                f"the model did not accept the platform shocks: {type(error).__name__}: {error}",
            )
        ]
    findings: list[Finding] = []
    if sorted(second.ids) != sorted(first.ids):
        findings.append(Finding("shocks", "the model returned other drivers with the shocks"))
    if second.paths.shape != first.paths.shape:
        findings.append(
            Finding(
                "shocks",
                "the paths changed shape: "
                f"{first.paths.shape} on the first shocks, {second.paths.shape} on the second",
            )
        )
    elif np.array_equal(second.paths, first.paths):
        findings.append(
            Finding(
                "shocks",
                "the model ignored the platform shocks: its paths on two different shocks are "
                "equal",
            )
        )
    return findings
