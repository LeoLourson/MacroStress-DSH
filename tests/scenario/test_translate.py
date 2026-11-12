"""Calendar and scalar conversion uses the data panel, not a model version."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from mst.scenario.contracts import Scenario
from mst.scenario.translate import (
    QuantityData,
    TranslationData,
    month_index,
    plan_of,
    shift_to_scale,
    to_scale,
)


def data():
    return TranslationData(
        origin="2026-09",
        quantities={
            "ru.usdrub": QuantityData("ru.usdrub", "log100"),
            "ru.key_rate": QuantityData("ru.key_rate", "identity"),
            "gl.brent": QuantityData("gl.brent", "log100"),
        },
        last_model={"ru.usdrub": 444.39, "ru.key_rate": 16.5, "gl.brent": 430.0},
        history={
            "ru.usdrub": np.array([440.0, 442.0, 444.39]),
            "ru.key_rate": np.array([14.5, 15.5, 16.5]),
            "gl.brent": np.array([426.0, 428.0, 430.0]),
        },
    )


def test_percent_is_added_to_the_log_coordinate_once():
    q = QuantityData("ru.usdrub", "log100")
    result = to_scale(q, 25.0, "percent", 444.39)
    assert result == pytest.approx(466.70435513142097)
    assert shift_to_scale(q, 25.0, "percent", 444.39) == pytest.approx(22.314355131420975)


def test_rate_change_is_from_last_known_level():
    q = QuantityData("ru.key_rate", "identity")
    assert to_scale(q, -5.0, "change", 16.5) == 11.5
    assert shift_to_scale(q, -5.0, "change", 16.5) == -5.0
    with pytest.raises(ValueError, match="percent.*log100"):
        to_scale(q, 25.0, "percent", 16.5)


def test_decimal_scalar_conversion_preserves_a_legacy_shift_exactly():
    q = QuantityData("share_price:vtbr", "log100")
    with localcontext() as context:
        context.prec = 80
        percent = float(100 * ((Decimal(-3) / 100).exp() - 1))
    assert shift_to_scale(q, percent, "percent", 400.0) == -3.0
    assert (
        shift_to_scale(
            QuantityData("gl.vix", "log100"), 5.520771983549331, "change", 275.87433154177285
        )
        == 30.0
    )


@pytest.mark.parametrize("month,index", [("2026-10", 1), ("2026-11", 2), ("2027-01", 4)])
def test_month_index_uses_panel_origin(month, index):
    assert month_index(month, "2026-09") == index


@pytest.mark.parametrize("month", ["2026-09", "2026-08"])
def test_historical_month_is_rejected_by_name(month):
    with pytest.raises(ValueError, match=month):
        month_index(month, "2026-09")


def test_plan_converts_dates_and_interventions_for_every_group():
    scenario = Scenario.model_validate(
        {
            "narrative": "A rate decision",
            "interventions": [
                {
                    "quantity": "ru.key_rate",
                    "months": ["2026-11"],
                    "kind": "set",
                    "values": [-5.0],
                    "unit": "change",
                    "reason": "Decision",
                    "basis": "event",
                }
            ],
            "observations": [
                {
                    "kind": "constraint",
                    "unit": "level",
                    "basis": "event",
                    "value": 11.5,
                    "terms": [{"quantity": "ru.key_rate", "month": "2026-11", "weight": 1.0}],
                }
            ],
        }
    )
    plan = plan_of(scenario, data(), 12, groups=3, inner=2)
    assert plan.horizon == 12 and plan.inner == 2
    (impact,) = plan.interventions
    assert impact.target == "ru.key_rate" and impact.months == (2,)
    np.testing.assert_array_equal(impact.values, np.full((3, 1), 11.5))
    (condition,) = plan.constraints
    assert condition.point and condition.low < 11.5 < condition.high
    assert condition.terms[0].variable == "ru.key_rate"
    assert condition.terms[0].month == 2


def test_log_ratio_observation_percent_translates_the_scalar_functional():
    scenario = Scenario.model_validate(
        {
            "narrative": "Brent rises between two future months",
            "observations": [
                {
                    "kind": "constraint",
                    "unit": "percent",
                    "basis": "event",
                    "low": 10.0,
                    "high": 20.0,
                    "terms": [
                        {"quantity": "gl.brent", "month": "2026-10", "weight": -1.0},
                        {"quantity": "gl.brent", "month": "2026-12", "weight": 1.0},
                    ],
                }
            ],
        }
    )
    (condition,) = plan_of(scenario, data(), 12).constraints
    assert condition.low == pytest.approx(100 * np.log1p(0.1))
    assert condition.high == pytest.approx(100 * np.log1p(0.2))
    assert [term.month for term in condition.terms] == [1, 3]


def test_plan_refuses_dates_outside_the_requested_horizon():
    scenario = Scenario.model_validate(
        {
            "narrative": "Late decision",
            "interventions": [
                {
                    "quantity": "ru.key_rate",
                    "months": ["2027-10"],
                    "kind": "set",
                    "values": [10.0],
                    "unit": "level",
                    "reason": "Decision",
                    "basis": "event",
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="2027-10"):
        plan_of(scenario, data(), 12)
