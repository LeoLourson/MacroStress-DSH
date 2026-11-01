"""mst.macro.registry: macro variables and their transforms."""

import math

import pytest
from pydantic import ValidationError

from mst.macro.registry import Registry, load_registry
from mst.scenario.contracts import Scenario


def variable(id_, transform="identity"):
    return {"id": id_, "label": id_, "unit": "%", "transform": transform, "source": "test"}


def registry(variables):
    return Registry.model_validate({"variables": variables})


@pytest.fixture(scope="module")
def shipped():
    return load_registry()


def test_shipped_registry_holds_the_formalization_tables(shipped):
    assert len(shipped.variables) == 23
    assert shipped.variable("ru.key_rate").block == "ru"
    assert shipped.variable("gl.brent").transform == "log100"


def test_log100_turns_a_ratio_into_a_difference(shipped):
    usdrub = shipped.variable("ru.usdrub")
    change = usdrub.to_model(110.0) - usdrub.to_model(100.0)
    assert change == pytest.approx(100 * math.log(1.1))
    assert usdrub.from_model(usdrub.to_model(97.5)) == pytest.approx(97.5)


def test_identity_keeps_rates_in_percent(shipped):
    assert shipped.variable("ru.key_rate").to_model(21.0) == 21.0


def test_log100_rejects_a_non_positive_value(shipped):
    with pytest.raises(ValueError, match="positive"):
        shipped.variable("ru.cpi").to_model(0.0)


def test_variable_id_names_its_block():
    with pytest.raises(ValidationError, match="String should match pattern"):
        registry([variable("key_rate")])


def test_variable_ids_are_unique():
    with pytest.raises(ValidationError, match=r"ru\.key_rate is registered twice"):
        registry([variable("ru.key_rate"), variable("ru.key_rate")])


def test_check_names_unknown_variables(shipped):
    unknown = Scenario.model_validate(
        {
            "narrative": "x",
            "horizon": 12,
            "constraints": [
                {
                    "terms": [{"variable": "ru.gdp", "month": 1, "weight": 1.0}],
                    "value": 0.0,
                }
            ],
        }
    )
    with pytest.raises(ValueError, match=r"unknown variable ru\.gdp"):
        shipped.check(unknown)


def test_check_accepts_a_scenario_on_registered_names(shipped):
    known = Scenario.model_validate(
        {
            "narrative": "x",
            "horizon": 12,
            "constraints": [
                {
                    "terms": [{"variable": "ru.key_rate", "month": 3, "weight": 1.0}],
                    "value": 21.0,
                }
            ],
        }
    )
    shipped.check(known)
