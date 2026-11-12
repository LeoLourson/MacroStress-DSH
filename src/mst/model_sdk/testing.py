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

from .contracts import (
    Intervention,
    MacroSimulation,
    SatelliteModel,
    ScenarioPlan,
    UpperPaths,
)
from .interventions import apply_default


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
    interventions: Sequence[Intervention] = (),
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
    upper = UpperPaths(macro=macro, levels={})
    try:
        first = model.simulate(
            loaded,
            upper,
            list(drivers),
            list(interventions),
            shocks=first_shocks,
            seed=seed,
            stream=stream,
        )
    except Exception as error:  # noqa: BLE001 - a broken model is a finding
        return [Finding("simulate", f"{type(error).__name__}: {error}")]
    try:
        second = model.simulate(
            loaded,
            upper,
            list(drivers),
            list(interventions),
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


def check_interventions(
    model, loaded, *, horizon: int = 3, draws: int = 2, inner: int = 2, seed: int = 1
) -> list[Finding]:
    """Check group layout and the macro platform's default/recursive dispatch boundary.

    A default-form macro plugin returns pre-intervention paths; the platform applies set or
    shift once. A recursive plugin returns the intervened paths itself. Both must set exact
    values on every path, preserve month zero, and return contiguous preserved groups.
    """
    try:
        baseline = model.simulate(
            loaded, ScenarioPlan(horizon=horizon, inner=inner), draws=draws, seed=seed
        )
        findings = _group_findings(baseline, draws, inner, horizon)
        if findings or not baseline.ids:
            return findings or [Finding("interventions", "the model returned no quantities")]
        form = model.structural_form(loaded)
        if form is not None and set(form.order_within_month) != set(baseline.ids):
            return [Finding("structural_form", "the order is not a permutation of model ids")]
        for kind in ("set", "shift"):
            findings.extend(
                _intervention_findings(
                    model, loaded, baseline, kind, form, horizon, draws, inner, seed
                )
            )
        return findings
    except Exception as error:  # noqa: BLE001 - a broken model is a finding
        return [Finding("interventions", f"{type(error).__name__}: {error}")]


def _group_findings(simulation, draws, inner, horizon):
    expected = (draws * inner, horizon + 1, len(simulation.ids))
    if simulation.paths.shape != expected:
        return [Finding("groups", f"expected paths {expected}, got {simulation.paths.shape}")]
    group = simulation.group
    if group is None and inner == 1:
        return []
    if group is None or not np.array_equal(group, np.repeat(np.arange(draws), inner)):
        return [Finding("groups", "paths do not use contiguous groups of inner members")]
    return []


def _intervention_findings(model, loaded, baseline, kind, form, horizon, draws, inner, seed):
    values = np.arange(draws, dtype=float)[:, None] + 7.0
    item = Intervention(baseline.ids[0], (1,), kind, values)
    plan = ScenarioPlan(horizon=horizon, interventions=(item,), inner=inner)
    simulated = model.simulate(loaded, plan, draws=draws, seed=seed)
    findings = _group_findings(simulated, draws, inner, horizon)
    if findings:
        return findings
    if simulated.ids != baseline.ids:
        return [Finding("interventions", "interventions changed the returned quantity ids")]
    paths = (
        apply_default(simulated.paths, simulated.ids, (item,), inner=inner)
        if form is None else simulated.paths
    )
    expected = np.repeat(values[:, 0], inner)
    if kind == "set" and not np.array_equal(paths[:, 1, 0], expected):
        findings.append(Finding("set", "the installed value is not exact on every path"))
    if not np.array_equal(paths[:, 0], baseline.paths[:, 0]):
        findings.append(Finding("interventions", "an intervention changed month zero"))
    if form is None:
        intended = baseline.paths.copy()
        if kind == "set":
            intended[:, 1, 0] = expected
        else:
            intended[:, 1, 0] += expected
        if not np.array_equal(paths, intended):
            findings.append(Finding(kind, "the default form changed unmentioned cells"))
    return findings
