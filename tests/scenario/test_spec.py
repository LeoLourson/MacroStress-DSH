"""mst.scenario.spec: the event file and the statements it carries."""

import pytest
from pydantic import ValidationError

from mst.scenario.contracts import (
    Constraint,
    Expectation,
    ProbabilityCondition,
    Scenario,
)
from mst.scenario.spec import Spec


def term(quantity="ru.key_rate", month="2026-11", weight=1.0):
    return {"quantity": quantity, "month": month, "weight": weight}


def constraint(**changes):
    return {
        "kind": "constraint",
        "terms": [term()],
        "value": 11.5,
        "unit": "level",
        "basis": "event",
    } | changes


def expectation(**changes):
    return {
        "kind": "expectation",
        "terms": [term()],
        "low": 11.0,
        "unit": "level",
        "basis": "event",
    } | changes


def probability(**changes):
    return {
        "kind": "probability",
        "terms": [term()],
        "low": 10.0,
        "high": 12.0,
        "at_least": 0.8,
        "unit": "level",
        "basis": "event",
    } | changes


def test_a_stored_scenario_is_the_event_contract_itself():
    assert Spec is Scenario
    raw = {
        "narrative": "ставка растёт",
        "sources": ["S1/2020#1/1"],
        "interventions": [
            {
                "quantity": "ru.key_rate",
                "months": ["2026-11"],
                "kind": "set",
                "values": [11.5],
                "unit": "level",
                "reason": "Решение ЦБ",
                "basis": "event",
            }
        ],
        "observations": [constraint()],
        "approved_by": "И. Иванов",
    }
    built = Spec.model_validate(raw)
    dumped = built.model_dump(mode="json")
    assert dumped["narrative"] == raw["narrative"] and dumped["sources"] == raw["sources"]
    assert dumped["observations"][0]["value"] == 11.5
    assert dumped["interventions"] == raw["interventions"]
    assert Spec.model_validate(dumped) == built


def test_a_constraint_names_either_an_interval_or_a_value():
    with pytest.raises(ValidationError, match="not both"):
        Constraint.model_validate(constraint(low=10.0, high=12.0))
    with pytest.raises(ValidationError, match="low and high, or a value"):
        Constraint.model_validate(constraint(value=None))
    with pytest.raises(ValidationError, match="both low and high"):
        Constraint.model_validate(constraint(value=None, low=10.0))


def test_a_constraint_interval_needs_strict_bounds():
    with pytest.raises(ValidationError, match="low < high"):
        Constraint.model_validate(constraint(value=None, low=12.0, high=12.0))
    with pytest.raises(ValidationError, match="low < high"):
        Constraint.model_validate(constraint(value=None, low=13.0, high=12.0))
    assert Constraint.model_validate(constraint(value=None, low=10.0, high=12.0)).point is False


def test_a_point_is_marked_and_keeps_its_value():
    point = Constraint.model_validate(constraint())
    assert point.point and point.value == 11.5 and point.low is None and point.high is None
    interval = Constraint.model_validate(constraint(value=None, low=10.0, high=12.0))
    assert not interval.point and interval.value is None


def test_an_expectation_accepts_one_or_two_bounds_and_orders_them():
    assert Expectation.model_validate(expectation()).high is None
    assert Expectation.model_validate(expectation(low=None, high=12.0)).low is None
    assert Expectation.model_validate(expectation(high=12.0)).high == 12.0
    with pytest.raises(ValidationError, match="low, high, or both"):
        Expectation.model_validate(expectation(low=None))
    with pytest.raises(ValidationError, match="low < high"):
        Expectation.model_validate(expectation(high=11.0))


def test_a_probability_condition_needs_a_band_and_a_level():
    assert ProbabilityCondition.model_validate(probability()).at_least == 0.8
    with pytest.raises(ValidationError, match="low < high"):
        ProbabilityCondition.model_validate(probability(low=12.0, high=12.0))
    for level in (-0.1, 1.1):
        with pytest.raises(ValidationError):
            ProbabilityCondition.model_validate(probability(at_least=level))


def test_a_term_needs_a_calendar_month_and_a_non_zero_weight():
    for month in ("2026-00", "2026-13", "2026-1", "2026-11-01", 11):
        with pytest.raises(ValidationError):
            Constraint.model_validate(constraint(terms=[term(month=month)]))
    with pytest.raises(ValidationError, match="non-zero weight"):
        Constraint.model_validate(constraint(terms=[term(weight=0.0)]))
    with pytest.raises(ValidationError):
        Constraint.model_validate(constraint(terms=[term(weight=float("inf"))]))


def test_one_statement_names_a_quantity_and_a_month_at_most_once():
    repeated = [term(), term(weight=-2.0)]
    with pytest.raises(ValidationError, match=r"ru.key_rate in month 2026-11 appears twice"):
        Constraint.model_validate(constraint(terms=repeated))
    # The same quantity in two months is a change, not a repetition.
    months = [term(), term(month="2026-12")]
    assert len(Constraint.model_validate(constraint(terms=months)).terms) == 2


def test_every_statement_needs_a_basis_and_the_basis_is_a_word_or_fragments():
    for cls, raw in ((Constraint, constraint()), (Expectation, expectation())):
        del raw["basis"]
        with pytest.raises(ValidationError, match="basis"):
            cls.model_validate(raw)
        assert cls.model_validate(raw | {"basis": "assumption"}).basis == "assumption"
        assert cls.model_validate(raw | {"basis": ["S1/2020#1/1"]}).basis == ["S1/2020#1/1"]
        with pytest.raises(ValidationError, match="at least 1"):
            cls.model_validate(raw | {"basis": []})


def test_a_word_inside_a_list_of_fragments_is_refused_by_name():
    # Left as a list, the word would pass the contract and fail later as a missing fragment.
    for word in ("event", "assumption"):
        with pytest.raises(ValidationError, match=f"the word {word} inside a list"):
            Constraint.model_validate(constraint(basis=[word, "S1/2020#1/1"]))
    # A word alone in a list is still that word.
    assert Constraint.model_validate(constraint(basis=["event"])).basis == "event"


def test_the_narrative_and_sources_follow_the_wire_contract():
    with pytest.raises(ValidationError, match="narrative"):
        Scenario.model_validate({"narrative": ""})
    with pytest.raises(ValidationError):
        Scenario.model_validate({"narrative": "  "})
    with pytest.raises(ValidationError):
        Scenario.model_validate({"narrative": "Event", "sources": [""]})
    with pytest.raises(ValidationError):
        Scenario(narrative="Event", approved_by=2)
    assert Scenario(narrative="Event").sources == []
