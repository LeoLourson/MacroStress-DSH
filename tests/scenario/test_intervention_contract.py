"""Calendar scenario contracts independent of calculation versions."""

import pytest
from pydantic import ValidationError

from mst.scenario.contracts import (
    Constraint,
    Expectation,
    Intervention,
    ProbabilityCondition,
    ProbabilityValue,
    Scenario,
    UniformValue,
)
from mst.scenario.spec import Spec


def intervention(**changes):
    return {
        "quantity": "ru.key_rate",
        "months": ["2026-11"],
        "kind": "set",
        "values": [11.5],
        "unit": "level",
        "reason": "Policy decision",
        "basis": "event",
    } | changes


def observation(**changes):
    return {
        "kind": "constraint",
        "terms": [{"quantity": "ru.key_rate", "month": "2026-11", "weight": 1.0}],
        "value": 11.5,
        "unit": "level",
        "basis": "event",
    } | changes


def test_scenario_schema_has_only_the_event_contract():
    assert set(Scenario.model_json_schema()["properties"]) == {
        "narrative",
        "sources",
        "interventions",
        "observations",
        "approved_by",
    }
    assert Spec.model_json_schema() == Scenario.model_json_schema()
    for field, value in (("horizon", 12), ("macro", "a" * 64), ("causes", [])):
        with pytest.raises(ValidationError, match=field):
            Scenario.model_validate({"narrative": "Event", field: value})


def test_observations_are_discriminated_and_keep_their_units():
    rows = [
        observation(),
        observation(kind="expectation", value=None, low=10.0),
        observation(kind="probability", value=None, low=10.0, high=12.0, at_least=0.8),
    ]
    rows[1].pop("value")
    rows[2].pop("value")
    scenario = Scenario(narrative="Event", observations=rows)
    assert [type(item) for item in scenario.observations] == [
        Constraint,
        Expectation,
        ProbabilityCondition,
    ]
    assert scenario.model_dump(mode="json")["observations"][0]["unit"] == "level"
    with pytest.raises(ValidationError, match="kind"):
        Scenario(
            narrative="Event", observations=[{k: v for k, v in rows[0].items() if k != "kind"}]
        )


@pytest.mark.parametrize("month", ["2026-00", "2026-13", "2026-1", "2026-11-01", 2])
def test_months_are_calendar_months(month):
    with pytest.raises(ValidationError):
        Intervention.model_validate(intervention(months=[month]))


@pytest.mark.parametrize(
    "changes",
    [
        {"months": ["2026-11", "2026-11"], "values": [11.5, 12.0]},
        {"values": []},
        {"months": ["2026-11", "2026-12"]},
        {"kind": "add"},
        {"unit": "model"},
        {"values": [float("inf")]},
    ],
)
def test_invalid_interventions_are_rejected(changes):
    with pytest.raises(ValidationError):
        Intervention.model_validate(intervention(**changes))


def test_one_intervention_per_quantity_and_calendar_month():
    with pytest.raises(ValidationError, match="ru.key_rate.*2026-11"):
        Scenario(narrative="Event", interventions=[intervention(), intervention(kind="shift")])
    built = Scenario(
        narrative="Event", interventions=[intervention(), intervention(months=["2026-12"])]
    )
    assert len(built.interventions) == 2


def test_units_are_required_at_the_wire_boundary():
    for cls, raw in ((Intervention, intervention()), (Constraint, observation())):
        del raw["unit"]
        with pytest.raises(ValidationError, match="unit"):
            cls.model_validate(raw)


@pytest.mark.parametrize(
    "law", [{"uniform": [10.0, 12.0]}, {"with_probability": 0.5, "value": 11.5}]
)
def test_stochastic_values_enter_an_intervention(law):
    cls = UniformValue if "uniform" in law else ProbabilityValue
    assert cls.model_validate(law).model_dump(mode="json") == law
    built = Intervention.model_validate(intervention(values=[law]))
    assert built.values[0] == cls.model_validate(law)


@pytest.mark.parametrize(
    "law", [{"uniform": [12.0, 10.0]}, {"uniform": [1.0]}, {"with_probability": 1.1, "value": 1.0}]
)
def test_stochastic_value_contract_rejects_invalid_laws(law):
    cls = UniformValue if "uniform" in law else ProbabilityValue
    with pytest.raises(ValidationError):
        cls.model_validate(law)
