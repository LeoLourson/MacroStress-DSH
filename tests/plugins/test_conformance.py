"""The conformance check every plugin passes: declarations and the demo calculation."""

from dataclasses import replace

import pytest

from mst import plugins
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
