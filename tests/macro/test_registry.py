"""mst.macro.registry: macro variables, their transforms and the named shocks."""

import math

import pytest
from pydantic import ValidationError

from mst.macro.registry import Registry, load_registry
from mst.scenario.contracts import Scenario


def variable(id_, transform="identity"):
    return {"id": id_, "label": id_, "unit": "%", "transform": transform, "source": "test"}


def registry(variables, shocks=()):
    return Registry.model_validate({"variables": variables, "shocks": list(shocks)})


def shock(id_, rises=(), falls=()):
    return {"id": id_, "label": id_, "rises": list(rises), "falls": list(falls)}


@pytest.fixture(scope="module")
def shipped():
    return load_registry()


def test_shipped_registry_holds_the_formalization_tables(shipped):
    assert len(shipped.variables) == 23
    assert [s.id for s in shipped.shocks] == [
        "ru.monetary",
        "ru.inflation",
        "ru.fx_risk",
        "gl.oil",
        "gl.financial",
    ]
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


def test_shock_restricts_only_registered_variables():
    with pytest.raises(ValidationError, match=r"ru\.monetary restricts unknown ru\.ruonia"):
        registry([variable("ru.key_rate")], [shock("ru.monetary", rises=["ru.ruonia"])])


def test_shock_cannot_move_a_variable_both_ways():
    with pytest.raises(ValidationError, match=r"both raises and lowers ru\.key_rate"):
        registry(
            [variable("ru.key_rate")],
            [shock("ru.monetary", rises=["ru.key_rate"], falls=["ru.key_rate"])],
        )


def test_russian_shock_cannot_move_a_global_variable():
    with pytest.raises(ValidationError, match=r"Russian shock ru\.monetary cannot move gl\.vix"):
        registry([variable("gl.vix", "log100")], [shock("ru.monetary", falls=["gl.vix"])])


def test_shock_needs_a_sign():
    with pytest.raises(ValidationError, match=r"gl\.oil restricts no variable"):
        registry([variable("gl.brent", "log100")], [shock("gl.oil")])


def test_check_names_unknown_variables_and_shocks(shipped):
    unknown = Scenario.model_validate(
        {
            "narrative": "x",
            "horizon": 12,
            "constraints": [
                {
                    "terms": [{"variable": "ru.gdp", "month": 1, "weight": 1.0}],
                    "low": 0.0,
                    "high": 0.0,
                }
            ],
            "driving_shocks": ["ru.demand"],
        }
    )
    with pytest.raises(ValueError, match=r"unknown variable ru\.gdp; unknown shock ru\.demand"):
        shipped.check(unknown)


def test_check_accepts_a_scenario_on_registered_names(shipped):
    known = Scenario.model_validate(
        {
            "narrative": "x",
            "horizon": 12,
            "constraints": [
                {
                    "terms": [{"variable": "ru.key_rate", "month": 3, "weight": 1.0}],
                    "low": 21.0,
                    "high": 21.0,
                }
            ],
            "driving_shocks": ["ru.inflation"],
        }
    )
    shipped.check(known)


def test_shock_ids_are_unique():
    with pytest.raises(ValidationError, match=r"shock gl\.oil is registered twice"):
        registry(
            [variable("gl.brent", "log100")],
            [shock("gl.oil", rises=["gl.brent"]), shock("gl.oil", rises=["gl.brent"])],
        )
