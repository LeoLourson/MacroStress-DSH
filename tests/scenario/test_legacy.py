"""The legacy adapter reads old data without writing or retaining model references."""

import copy

import numpy as np
import pytest

from mst.scenario.legacy import LegacyTranslationError, read_legacy
from mst.scenario.translate import QuantityData, TranslationData, shift_to_scale, to_scale


def context():
    return TranslationData(
        origin="2026-09",
        quantities={
            "ru.key_rate": QuantityData("ru.key_rate", "identity"),
            "ru.usdrub": QuantityData("ru.usdrub", "log100"),
            "share_price:vtbr": QuantityData("share_price:vtbr", "log100"),
            "gl.vix": QuantityData("gl.vix", "log100"),
        },
        last_model={
            "ru.key_rate": 16.5,
            "ru.usdrub": 444.39,
            "share_price:vtbr": 300.0,
            "gl.vix": 275.87433154177285,
        },
        history={"ru.key_rate": np.array([14.5, 15.5, 16.5])},
    )


def old(**changes):
    return {
        "narrative": "An event",
        "horizon": 12,
        "macro": "a" * 64,
        "satellites": "b" * 64,
    } | changes


def test_old_empty_scenario_needs_no_calendar_context():
    result = read_legacy(old())
    assert result.narrative == "An event" and not result.interventions and not result.observations
    assert "macro" not in result.model_dump()


def test_legacy_reader_resolves_actual_panel_origin_and_preserves_approval():
    raw = old(
        constraints=[
            {
                "terms": [{"variable": "ru.key_rate", "month": 2, "weight": 1.0}],
                "value": 9.0,
                "basis": "event",
            }
        ],
        approved_by="A human",
    )
    seen = []

    def resolver(value):
        seen.append(value["macro"])
        return context()

    result = read_legacy(raw, resolver=resolver)
    assert seen == ["a" * 64]
    assert result.observations[0].terms[0].month == "2026-11"
    assert result.observations[0].value == 9.0
    assert result.approved_by == "A human"
    with pytest.raises(LegacyTranslationError, match="context"):
        read_legacy(raw)


def test_legacy_delta_end_and_values_are_read_without_mutating_the_input():
    raw = old(
        impacts=[
            {
                "target": "ru.key_rate",
                "start": 2,
                "end": 4,
                "delta": -5.0,
                "reason": "Decision",
                "basis": "event",
            }
        ]
    )
    original = copy.deepcopy(raw)
    result = read_legacy(raw, context=context())
    (impact,) = result.interventions
    assert impact.months == ["2026-11", "2026-12", "2027-01"]
    assert impact.kind == "shift" and impact.values == [-5.0] * 3
    assert raw == original
    raw["impacts"][0] = {
        k: v for k, v in raw["impacts"][0].items() if k not in ("delta", "end")
    } | {"values": [-5.0, -3.0]}
    assert read_legacy(raw, context=context()).interventions[0].values == [-5.0, -3.0]


def test_legacy_causes_are_kept_as_text_not_as_runtime_coordinates():
    result = read_legacy(old(causes=["ru.monetary"], regime_event="crisis", approved_by="A human"))
    assert "прежние причины:" in result.narrative
    assert "ru.monetary" in result.narrative and "crisis" in result.narrative
    assert "causes" not in result.model_dump()
    assert result.approved_by is None


def test_legacy_log_point_roundtrips_exactly():
    raw = old(
        constraints=[
            {
                "terms": [{"variable": "ru.usdrub", "month": 2, "weight": 1.0}],
                "value": 478.7492,
                "basis": "event",
            }
        ]
    )
    result = read_legacy(raw, context=context())
    (condition,) = result.observations
    assert (
        to_scale(context().quantities["ru.usdrub"], condition.value, condition.unit, 444.39)
        == 478.7492
    )


@pytest.mark.parametrize(
    "quantity,values", [("share_price:vtbr", [-3.0]), ("gl.vix", [61.2, 30.0, 8.0])]
)
def test_legacy_log_shift_selects_one_exact_unit_for_its_whole_path(quantity, values):
    raw = old(
        impacts=[
            {
                "target": quantity,
                "start": 2,
                "values": values,
                "reason": "Event channel",
                "basis": "assumption",
            }
        ]
    )
    (impact,) = read_legacy(raw, context=context()).interventions
    converted = [
        shift_to_scale(
            context().quantities[quantity], value, impact.unit, context().last_model[quantity]
        )
        for value in impact.values
    ]
    assert converted == values
    if quantity == "gl.vix":
        assert impact.unit == "change"


def test_overlapping_old_impacts_stop_instead_of_summing():
    impact = {
        "target": "ru.key_rate",
        "start": 2,
        "values": [-5.0],
        "reason": "Decision",
        "basis": "event",
    }
    with pytest.raises(LegacyTranslationError, match="ru.key_rate.*2026-11"):
        read_legacy(old(impacts=[impact, impact]), context=context())


def test_old_log_expectation_stops_instead_of_becoming_an_arithmetic_mean():
    raw = old(
        expectations=[
            {
                "terms": [{"variable": "ru.usdrub", "month": 2, "weight": 1.0}],
                "low": 450.0,
                "basis": "assumption",
            }
        ]
    )
    with pytest.raises(LegacyTranslationError, match="expectations.*ru.usdrub"):
        read_legacy(raw, context=context())
