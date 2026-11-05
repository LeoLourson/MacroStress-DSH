"""Incremental answers preserve every judgement of the complete expert answer."""

import json
from copy import deepcopy
from dataclasses import replace
from datetime import date

import pytest

from mst.data.store import Observation
from mst.macro.registry import Registry, load_registry
from mst.model_sdk.contracts import MacroEstimateContext, MacroModel
from mst.plugins.expert.answers import HORIZONS, NODES, ShockAnswer
from mst.plugins.expert.model import MARKET, ExpertModel, load_shocks
from tests.expert import fakes


class StateStore:
    def as_of(self, series_id, day):
        return [Observation(date(2026, 10, 1), 71.25, date(2026, 11, 1), False)]


@pytest.fixture(scope="module")
def ctx():
    return MacroEstimateContext(
        store=StateStore(), registry=load_registry(), as_of=fakes.AS_OF,
        fragment=fakes.visible,
    )


def split_answer(payload):
    parts = {"law": {key: value for key, value in payload.items()
                     if key not in ("shock", "responses")}}
    for variable, by_horizon in payload["responses"].items():
        for horizon, by_node in by_horizon.items():
            parts[f"response:{variable}:{horizon}"] = {
                f"at_{node}": quantiles for node, quantiles in by_node.items()
            }
    return deepcopy(parts)


@pytest.mark.parametrize("base", ["rw", "market"])
@pytest.mark.parametrize("shock", load_shocks(), ids=lambda shock: shock.id)
def test_every_shock_has_one_law_and_four_steps_per_variable(ctx, base, shock):
    steps = ExpertModel(base).input_steps(ctx, shock.id)
    expected = ["law"] + [
        f"response:{variable.id}:{horizon}"
        for variable in ctx.registry.variables for horizon in HORIZONS
    ]
    assert [step["id"] for step in steps] == expected
    assert len(steps) == 1 + 4 * len(ctx.registry.variables)
    assert all(set(step) == {"id", "text", "schema"} for step in steps)


def test_steps_follow_the_context_registry_not_a_fixed_variable_count(ctx):
    small = replace(ctx, registry=Registry(variables=ctx.registry.variables[:2]))
    assert len(ExpertModel("rw").input_steps(small, "gl.oil")) == 9


@pytest.mark.parametrize("key,payload", list(fakes.answers().items()))
def test_full_fixture_answer_round_trips_identically(ctx, key, payload):
    model = ExpertModel("rw")
    parts = split_answer(payload)
    before = deepcopy(parts)
    assembled = model.assemble_input(ctx, key, parts)
    assert assembled == payload
    assert parts == before
    assert ShockAnswer.model_validate(assembled).model_dump(mode="json") == payload
    assert model.check_input(ctx, key, assembled) == payload


def test_step_schemas_are_small_explicit_and_have_no_defaults(ctx):
    law, *responses = ExpertModel("rw").input_steps(ctx, "gl.oil")
    law_fields = {"frequency", "size", "rho", "nonlinearity", "sources"}
    assert set(law["schema"]["properties"]) == law_fields
    assert set(law["schema"]["required"]) == law_fields
    assert law["schema"]["additionalProperties"] is False
    for step in responses:
        schema = step["schema"]
        fields = {"at_q5", "at_q50", "at_q95"}
        assert set(schema["properties"]) == fields
        assert set(schema["required"]) == fields
        assert schema["additionalProperties"] is False
        quantiles = schema["$defs"]["Quantiles"]
        assert set(quantiles["properties"]) == set(NODES)
        assert set(quantiles["required"]) == set(NODES)
        assert quantiles["additionalProperties"] is False
    assert '"default"' not in json.dumps([step["schema"] for step in [law, *responses]])


def test_response_question_names_shock_scale_state_and_month_convention(ctx):
    shock = load_shocks()[0]
    steps = ExpertModel("rw").input_steps(ctx, shock.id)
    assert shock.meaning in steps[0]["text"]
    assert shock.size_unit in steps[0]["text"]
    for step in steps[1:]:
        _, variable_id, horizon = step["id"].split(":")
        variable = ctx.registry.variable(variable_id)
        text = step["text"]
        for value in (shock.meaning, shock.size_unit, variable.id, variable.unit,
                      variable.transform, "71.25", "2026-10-01", str(horizon)):
            assert value in text
        assert "model scale" in text and "response" in text
        assert "month of the shock" in text and "first" in text
        assert all(f"at_{node}" in text for node in NODES)


@pytest.mark.parametrize("missing", ["law", "response:gl.brent:1", "response:ru.cpi:12"])
def test_missing_parts_are_refused_without_inventing_numbers(ctx, missing):
    parts = split_answer(fakes.shock_payload("gl.oil"))
    del parts[missing]
    with pytest.raises(ValueError, match=missing):
        ExpertModel("rw").assemble_input(ctx, "gl.oil", parts)


def test_extra_parts_are_refused(ctx):
    parts = split_answer(fakes.shock_payload("gl.oil"))
    parts["response:gl.brent:2"] = parts["response:gl.brent:1"]
    with pytest.raises(ValueError, match="response:gl.brent:2"):
        ExpertModel("rw").assemble_input(ctx, "gl.oil", parts)


@pytest.mark.parametrize("part,field", [("law", "frequency"), ("law", "sources"),
                                        ("response:gl.brent:1", "at_q95")])
def test_missing_part_fields_never_get_default_answers(ctx, part, field):
    parts = split_answer(fakes.shock_payload("gl.oil"))
    del parts[part][field]
    with pytest.raises(ValueError, match=field):
        ExpertModel("rw").assemble_input(ctx, "gl.oil", parts)


@pytest.mark.parametrize("part", ["law", "response:gl.brent:1"])
def test_extra_part_fields_are_refused(ctx, part):
    parts = split_answer(fakes.shock_payload("gl.oil"))
    parts[part]["invented"] = 1.0
    with pytest.raises(ValueError, match="invented"):
        ExpertModel("rw").assemble_input(ctx, "gl.oil", parts)


def test_missing_quantile_is_refused_without_inventing_a_value(ctx):
    parts = split_answer(fakes.shock_payload("gl.oil"))
    del parts["response:gl.brent:1"]["at_q50"]["q50"]
    with pytest.raises(ValueError, match="at_q50.q50"):
        ExpertModel("rw").assemble_input(ctx, "gl.oil", parts)


def test_crossed_response_quantiles_are_refused(ctx):
    parts = split_answer(fakes.shock_payload("gl.oil"))
    parts["response:gl.brent:1"]["at_q50"]["q5"] = 99.0
    with pytest.raises(ValueError, match="quantiles"):
        ExpertModel("rw").assemble_input(ctx, "gl.oil", parts)


def test_missing_state_is_reported_not_filled(ctx):
    class EmptyStore:
        def as_of(self, series_id, day):
            return []

    empty = replace(ctx, store=EmptyStore())
    steps = ExpertModel("rw").input_steps(empty, "gl.oil")
    assert "| none |" in steps[0]["text"]
    assert "Last known state in the observed unit: none." in steps[1]["text"]


def test_source_fragments_are_checked_by_the_full_check(ctx):
    model = ExpertModel("rw")
    payload = fakes.shock_payload("gl.oil", sources=["not-visible"])
    assembled = model.assemble_input(ctx, "gl.oil", split_answer(payload))
    assert assembled == payload
    with pytest.raises(ValueError, match="sources"):
        model.check_input(ctx, "gl.oil", assembled)


def test_market_remains_the_unchanged_full_question_and_answer(ctx):
    model = ExpertModel("market")
    question = model.question(ctx, MARKET)
    steps = model.input_steps(ctx, MARKET)
    assert steps == [{"id": "answer", "text": question["text"], "schema": question["schema"]}]
    payload = fakes.market_payload()
    assert model.assemble_input(ctx, MARKET, {"answer": payload}) == payload
    for parts in ({}, {"answer": payload, "law": {}}):
        with pytest.raises(ValueError):
            model.assemble_input(ctx, MARKET, parts)


def test_unknown_and_unavailable_market_keys_are_refused(ctx):
    model = ExpertModel("rw")
    for key in ("unknown", MARKET):
        with pytest.raises(KeyError):
            model.input_steps(ctx, key)
        with pytest.raises(KeyError):
            model.assemble_input(ctx, key, {})


def test_sdk_defaults_keep_existing_single_answer_plugins_compatible(ctx):
    model = ExpertModel("rw")
    question = model.question(ctx, "gl.oil")
    assert MacroModel.input_steps(model, ctx, "gl.oil") == [
        {"id": "answer", "text": question["text"], "schema": question["schema"]}
    ]
    payload = fakes.shock_payload("gl.oil")
    assert MacroModel.assemble_input(model, ctx, "gl.oil", {"answer": payload}) is payload
