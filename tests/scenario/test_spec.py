"""mst.scenario.spec: the scenario file and the contracts it carries."""

import pytest
from pydantic import ValidationError

from mst.scenario.contracts import Constraint, Impact, Scenario
from mst.scenario.spec import Spec

MACRO = "a" * 64
SATELLITES = "b" * 64


def rate_at(month, low, high=None):
    return {
        "terms": [{"variable": "ru.key_rate", "month": month, "weight": 1.0}],
        "low": low,
        "high": low if high is None else high,
    }


def impact(**changes):
    return {
        "target": "price:alpha",
        "start": 1,
        "end": 6,
        "delta": -10.0,
        "reason": "перебой поставок",
        "sources": ["S1/2020#1/1"],
    } | changes


def spec(**changes):
    values = {"narrative": "ставка растёт", "horizon": 12, "macro": MACRO, "satellites": SATELLITES}
    values.update(changes)
    return values


def test_a_scenario_file_names_its_versions_and_the_contract_holds():
    built = Spec.model_validate(
        spec(constraints=[rate_at(3, 21.0), rate_at(12, 15.0, 18.0)], impacts=[impact()])
    )
    assert built.macro == MACRO and built.satellites == SATELLITES
    assert [(c.low, c.high) for c in built.constraints] == [(21.0, 21.0), (15.0, 18.0)]
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


def test_constraint_requires_low_not_above_high():
    with pytest.raises(ValidationError, match="low <= high"):
        Constraint.model_validate(rate_at(3, 21.0, 20.0))


def test_a_constraint_month_must_lie_in_the_horizon():
    with pytest.raises(ValidationError, match="month 13 is beyond the horizon of 12"):
        Spec.model_validate(spec(constraints=[rate_at(13, 21.0)]))


def test_an_impact_must_end_in_the_horizon_and_go_forwards():
    with pytest.raises(ValidationError, match="month 13 is beyond the horizon"):
        Spec.model_validate(spec(impacts=[impact(end=13)]))
    with pytest.raises(ValidationError, match="start <= end"):
        Impact.model_validate(impact(start=7, end=6))


def test_an_impact_needs_a_reason_and_a_source():
    with pytest.raises(ValidationError, match="at least 1"):
        Impact.model_validate(impact(sources=[]))


def test_driving_shocks_are_unique():
    with pytest.raises(ValidationError, match=r"ru.inflation is named twice"):
        Spec.model_validate(spec(driving_shocks=["ru.inflation", "ru.inflation"]))


def test_the_old_overlays_field_is_read_as_impacts():
    old = {
        "driver": "price:alpha",
        "start": 1,
        "end": 6,
        "delta": -10.0,
        "reason": "перебой поставок",
        "sources": ["S1/2020#1/1"],
    }
    built = Spec.model_validate(spec(overlays=[old]))
    assert built.impacts[0].target == "price:alpha"
    stored = built.model_dump(mode="json")
    assert stored["impacts"][0]["target"] == "price:alpha" and "overlays" not in stored


def test_the_old_overlays_field_is_refused_next_to_impacts():
    old = {
        "driver": "price:alpha",
        "start": 1,
        "end": 6,
        "delta": -10.0,
        "reason": "перебой поставок",
        "sources": ["S1/2020#1/1"],
    }
    with pytest.raises(ValidationError, match="both `overlays`"):
        Spec.model_validate(spec(overlays=[old], impacts=[impact()]))


def test_the_contract_reads_the_fields_the_forecast_uses():
    built = Scenario.model_validate(
        {
            "narrative": "ставка растёт",
            "horizon": 12,
            "driving_shocks": ["ru.monetary"],
            "sources": ["S1/2020#1/1"],
        }
    )
    assert built.horizon == 12 and built.driving_shocks == ["ru.monetary"]
    assert built.sources == ["S1/2020#1/1"]
