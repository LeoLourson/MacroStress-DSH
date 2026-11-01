"""mst.scenario.spec: the scenario file and the contracts it carries."""

import pytest
from pydantic import ValidationError

from mst.scenario.contracts import Constraint, Impact, Scenario
from mst.scenario.spec import Spec

MACRO = "a" * 64
SATELLITES = "b" * 64


def rate_at(month, low, high=None):
    """A constraint on the key rate: a point when only the value is given, an interval
    otherwise."""
    row = {
        "terms": [{"variable": "ru.key_rate", "month": month, "weight": 1.0}],
        "basis": ["S1/2020#1/1"],
    }
    if high is None:
        return row | {"value": low}
    return row | {"low": low, "high": high}


def impact(**changes):
    return {
        "target": "price:alpha",
        "start": 1,
        "end": 6,
        "delta": -10.0,
        "reason": "перебой поставок",
        "basis": ["S1/2020#1/1"],
    } | changes


def spec(**changes):
    values = {
        "narrative": "ставка растёт",
        "horizon": 12,
        "macro": MACRO,
        "satellites": SATELLITES,
        "approved_by": "И. Иванов",
    }
    values.update(changes)
    return values


def test_a_scenario_file_names_its_versions_and_the_contract_holds():
    built = Spec.model_validate(
        spec(constraints=[rate_at(3, 21.0), rate_at(12, 15.0, 18.0)], impacts=[impact()])
    )
    assert built.macro == MACRO and built.satellites == SATELLITES
    assert [c.value for c in built.constraints] == [21.0, None]
    assert [(c.low, c.high) for c in built.constraints] == [(None, None), (15.0, 18.0)]
    assert built.constraints[0].point and not built.constraints[1].point
    assert built.impacts[0].target == "price:alpha"


def test_both_versions_are_required_and_are_hashes():
    for key in ("macro", "satellites"):
        for value in (None, "", "short"):
            with pytest.raises(ValidationError):
                Spec.model_validate(spec() | {key: value})
        values = spec()
        del values[key]
        with pytest.raises(ValidationError, match=key):
            Spec.model_validate(values)


def test_unknown_fields_are_refused():
    with pytest.raises(ValidationError):
        Spec.model_validate(spec() | {"seed": 1})


def test_constraint_requires_low_below_high():
    with pytest.raises(ValidationError, match="low < high"):
        Constraint.model_validate(rate_at(3, 21.0, 20.0))


def test_a_constraint_of_zero_width_is_a_point_not_an_interval():
    with pytest.raises(ValidationError, match="low < high"):
        Constraint.model_validate(rate_at(3, 21.0, 21.0))


def test_a_constraint_names_either_an_interval_or_a_value():
    with pytest.raises(ValidationError, match="not both"):
        Constraint.model_validate(rate_at(3, 20.0, 22.0) | {"value": 21.0})
    with pytest.raises(ValidationError, match="low and high, or a value"):
        Constraint.model_validate(
            {"terms": rate_at(3, 21.0)["terms"], "basis": ["S1/2020#1/1"]}
        )
    with pytest.raises(ValidationError, match="both low and high"):
        Constraint.model_validate(rate_at(3, None, 20.0))


def test_a_constraint_month_must_lie_in_the_horizon():
    with pytest.raises(ValidationError, match="month 13 is beyond the horizon of 12"):
        Spec.model_validate(spec(constraints=[rate_at(13, 21.0)]))


def test_an_impact_must_end_in_the_horizon_and_go_forwards():
    with pytest.raises(ValidationError, match="month 13 is beyond the horizon"):
        Spec.model_validate(spec(impacts=[impact(end=13)]))
    with pytest.raises(ValidationError, match="start <= end"):
        Impact.model_validate(impact(start=7, end=6))


def test_an_impact_needs_a_reason_and_a_basis():
    with pytest.raises(ValidationError, match="at least 1"):
        Impact.model_validate(impact(basis=[]))


def test_a_constraint_needs_a_basis():
    row = rate_at(3, 21.0)
    with pytest.raises(ValidationError, match="basis"):
        Constraint.model_validate({key: value for key, value in row.items() if key != "basis"})
    # The word of the author's own is a basis too, and so is a list of fragments.
    assert Constraint.model_validate(row | {"basis": "assumption"}).basis == "assumption"
    assert Impact.model_validate(impact(basis="assumption")).basis == "assumption"


def test_causes_are_unique():
    with pytest.raises(ValidationError, match=r"ru.inflation is named twice"):
        Spec.model_validate(spec(causes=["ru.inflation", "ru.inflation"]))


def test_the_old_overlays_field_is_refused():
    old = {
        "driver": "price:alpha",
        "start": 1,
        "end": 6,
        "delta": -10.0,
        "reason": "перебой поставок",
        "basis": ["S1/2020#1/1"],
    }
    with pytest.raises(ValidationError, match="overlays"):
        Spec.model_validate(spec(overlays=[old]))


def test_the_old_driving_shocks_field_is_refused():
    # The causes have their own name now; the old one is an unknown field, refused as such.
    with pytest.raises(ValidationError, match="driving_shocks"):
        Spec.model_validate(spec(driving_shocks=["ru.monetary"]))


def test_the_contract_reads_the_fields_the_forecast_uses():
    built = Scenario.model_validate(
        {
            "narrative": "ставка растёт",
            "horizon": 12,
            "causes": ["ru.monetary"],
            "sources": ["S1/2020#1/1"],
        }
    )
    assert built.horizon == 12 and built.causes == ["ru.monetary"]
    assert built.sources == ["S1/2020#1/1"]
