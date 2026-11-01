"""The conformance check every plugin passes: declarations and the demo calculation."""

from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from mst import plugins
from mst.model_sdk import DriverPaths, MacroSimulation, SatelliteModel
from mst.model_sdk.testing import check_shocks
from mst.plugin_sdk import DataRequirements, DriverRequirement
from mst.plugin_sdk.testing import check_plugin
from tests.plugins import fakes


@pytest.mark.parametrize("plugin_id", sorted(plugins.registry().plugins))
def test_each_installed_plugin_conforms(plugin_id):
    assert check_plugin(plugins.registry().plugins[plugin_id]) == []


def test_a_plugin_without_a_model_is_rejected():
    findings = check_plugin(fakes.plugin(models=()))
    assert any("no model is declared" in str(finding) for finding in findings)


def test_a_plugin_without_a_demo_is_rejected():
    findings = check_plugin(fakes.plugin(demo=()))
    assert any("demo case" in str(finding) for finding in findings)


def test_a_demo_driver_must_exist_in_the_paths():
    case = replace(fakes.CASE, drivers={"gold_price": "gold_price:missing"})
    findings = check_plugin(fakes.plugin(demo=(case,)))
    assert any("do not carry" in str(finding) for finding in findings)


def test_a_demo_without_paths_is_a_finding():
    case = replace(fakes.CASE, paths=None)
    findings = check_plugin(fakes.plugin(demo=(case,)))
    assert any("demo case" in str(finding) for finding in findings)


def test_a_required_metric_must_be_declared():
    class Undeclared(fakes.DemoModel):
        required_metrics = ("gold.pnl",)

    findings = check_plugin(fakes.plugin(models=(Undeclared(),)))
    assert any("required metric" in str(finding) for finding in findings)


def test_a_demo_that_cannot_complete_is_a_finding():
    class Broken(fakes.DemoModel):
        def calculate(self, prepared, context):
            raise RuntimeError("boom")

    findings = check_plugin(fakes.plugin(models=(Broken(),)))
    assert any("boom" in str(finding) for finding in findings)


def test_a_demo_that_changes_between_calls_is_a_finding():
    class Changing(fakes.DemoModel):
        calls = 0

        def calculate(self, prepared, context):
            Changing.calls += 1
            result = super().calculate(prepared, context)
            return result.model_copy(
                update={"value_change": [float(Changing.calls)] * len(result.value_change)}
            )

    findings = check_plugin(fakes.plugin(models=(Changing(),)))
    assert any(finding.check == "determinism" for finding in findings)


def test_every_required_driver_must_be_mapped_to_a_demo_path():
    case = replace(fakes.CASE, drivers={})
    findings = check_plugin(fakes.plugin(demo=(case,)))
    assert any("gold_price" in str(finding) for finding in findings)


def test_every_demo_macro_variable_must_be_in_the_demo_paths():
    class Needy(fakes.DemoModel):
        def requirements(self, position, options):
            return DataRequirements(
                drivers=[DriverRequirement(key="price", quantity="gold_price")],
                macro=["ru.key_rate"],
            )

    findings = check_plugin(fakes.plugin(models=(Needy(),)))
    assert any("ru.key_rate" in str(finding) for finding in findings)


def test_a_demo_needs_a_position_its_model_serves():
    case = replace(fakes.CASE, position=fakes.CASE.position.model_copy(update={"type": "silver"}))
    findings = check_plugin(fakes.plugin(demo=(case,)))
    assert any("silver" in str(finding) for finding in findings)


DRIVER = "share_price:sber"


def simulated_macro(draws: int = 3, horizon: int = 2) -> MacroSimulation:
    months = [date(2026, 1 + k, 1) for k in range(horizon + 1)]
    return MacroSimulation(ids=["gl.brent"], months=months, paths=np.zeros((draws, horizon + 1, 1)))


class ShocklessSatellite(SatelliteModel):
    """A satellite that draws its own residuals whatever the platform passes."""

    model_id = "shockless.sat.v1"
    version = "0.1.0"
    label = "Shockless"
    quantities = ("share_price",)

    def estimate(self, context): ...

    def load(self, parts): ...

    def simulate(
        self, model, macro, drivers, impacts, *, correlation=None, shocks=None, seed, stream
    ) -> DriverPaths:
        return DriverPaths(
            ids=list(drivers),
            paths=np.zeros((macro.paths.shape[0], macro.paths.shape[1], len(drivers))),
            flags={driver: frozenset() for driver in drivers},
        )


class UsingSatellite(ShocklessSatellite):
    """A satellite that uses the platform's shocks column by column."""

    model_id = "using.sat.v1"
    label = "Using"

    def simulate(
        self, model, macro, drivers, impacts, *, correlation=None, shocks=None, seed, stream
    ) -> DriverPaths:
        horizon = macro.paths.shape[1] - 1
        paths = np.zeros((macro.paths.shape[0], horizon + 1, len(drivers)))
        if shocks is not None:
            paths[:, 1:, :] = shocks[:, :horizon, :]
        return DriverPaths(
            ids=list(drivers), paths=paths, flags={driver: frozenset() for driver in drivers}
        )


class OldSignature(ShocklessSatellite):
    """A satellite written before model SDK 1.1: no ``shocks`` parameter at all."""

    model_id = "old.sat.v1"
    label = "Old"

    def simulate(self, model, macro, drivers, impacts, *, correlation=None, seed, stream):
        return DriverPaths(
            ids=list(drivers),
            paths=np.zeros((macro.paths.shape[0], macro.paths.shape[1], len(drivers))),
            flags={driver: frozenset() for driver in drivers},
        )


def _shock_findings(model: SatelliteModel):
    return check_shocks(
        model,
        None,
        simulated_macro(),
        [DRIVER],
        correlated=[DRIVER],
        history=4,
    )


def test_a_satellite_that_uses_the_platform_shocks_conforms():
    assert _shock_findings(UsingSatellite()) == []


def test_a_satellite_that_ignores_the_platform_shocks_is_a_finding():
    findings = _shock_findings(ShocklessSatellite())
    assert len(findings) == 1 and "ignored the platform shocks" in str(findings[0])


def test_a_satellite_written_before_the_shocks_argument_is_a_finding():
    # The platform always passes shocks now, so a model without the parameter fails the call
    # itself; the finding names the TypeError it raised.
    findings = _shock_findings(OldSignature())
    assert len(findings) == 1
    assert findings[0].check == "simulate"
    assert "shocks" in str(findings[0])
