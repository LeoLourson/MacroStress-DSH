"""Small elicitation steps through the same MCP boundary an agent uses."""

import asyncio
from datetime import date

import pytest
from fastmcp import Client

from mst import inputs, plugins, store
from mst.mcp import build
from tests.plugins import fakes


class TwoSteps(fakes.InputMacroModel):
    def input_keys(self, as_of):
        return ("q1",)

    def input_steps(self, context, key):
        return [
            {"id": name, "text": f"Your judgement for {name}?", "schema": {"type": "object"}}
            for name in ("first", "second")
        ]

    def check_input_step(self, context, key, step, payload):
        if set(payload) != {"value"} or not isinstance(payload["value"], float | int):
            raise ValueError("value must be a number")
        return payload

    def assemble_input(self, context, key, parts):
        return {"value": parts["first"]["value"] + parts["second"]["value"]}


@pytest.fixture
def ws(tmp_path, monkeypatch):
    registry = plugins.build([fakes.model_plugin(macro=(TwoSteps(),))])
    monkeypatch.setattr(plugins, "registry", lambda: registry)
    return store.Workspace(tmp_path)


def call(ws, arguments):
    async def invoke():
        async with Client(build(ws)) as client:
            return (await client.call_tool("elicit", arguments)).data

    return asyncio.run(invoke())


def test_next_answer_resume_and_publish_only_when_complete(ws):
    first = call(ws, {"action": "next", "as_of": "2026-11-05"})
    assert first["step"] == "first" and first["progress"] == {"answered": 0, "total": 2}
    assert "ticket" in first["answer_call"] and "model_id" not in first["answer_call"]
    second = call(ws, {**first["answer_call"], "payload": {"value": 2}})
    assert second["step"] == "second"
    assert inputs.read(ws.elicitation, date(2026, 11, 5)) == {}
    resumed = call(ws, {"action": "next", "as_of": "2026-11-05"})
    assert resumed == second
    done = call(ws, {**second["answer_call"], "payload": {"value": 3}})
    assert done["complete"] is True
    assert inputs.read(ws.elicitation, date(2026, 11, 5)) == {"q1": {"value": 5.0}}
    assert call(ws, {**second["answer_call"], "payload": {"value": 3}}) == done


def test_invalid_and_stale_answers_do_not_change_saved_parts(ws):
    first = call(ws, {"action": "next", "as_of": "2026-11-05"})
    with pytest.raises(Exception, match="value must be a number"):
        call(ws, {**first["answer_call"], "payload": {"wrong": 2}})
    assert call(ws, {"action": "next", "as_of": "2026-11-05"}) == first
    second = call(ws, {**first["answer_call"], "payload": {"value": 2}})
    with pytest.raises(Exception, match="already answered"):
        call(ws, {**first["answer_call"], "payload": {"value": 9}})
    assert call(ws, {"action": "next", "as_of": "2026-11-05"}) == second


def test_ticket_cannot_escape_workspace_or_name_unknown_steps(ws):
    with pytest.raises(Exception, match="ticket"):
        call(ws, {"action": "answer", "ticket": "../../elsewhere", "payload": {"value": 1}})


def test_interrupted_final_publication_recovers_on_next(ws, monkeypatch):
    first = call(ws, {"action": "next", "as_of": "2026-11-05"})
    second = call(ws, {**first["answer_call"], "payload": {"value": 2}})
    original = inputs.write

    def interrupted(*args):
        raise OSError("interrupted publication")

    monkeypatch.setattr(inputs, "write", interrupted)
    with pytest.raises(Exception, match="interrupted publication"):
        call(ws, {**second["answer_call"], "payload": {"value": 3}})
    monkeypatch.setattr(inputs, "write", original)
    assert call(ws, {"action": "next", "as_of": "2026-11-05"})["complete"]
    assert inputs.read(ws.elicitation, date(2026, 11, 5))["q1"]["value"] == 5


def test_rejected_final_check_preserves_earlier_parts_only(ws, monkeypatch):
    model = plugins.registry().macro["gmodel.inputs.v1"]
    original = model.check_input

    def refuse(context, key, payload):
        if payload["value"] > 5:
            raise ValueError("total too large")
        return original(context, key, payload)

    monkeypatch.setattr(model, "check_input", refuse)
    first = call(ws, {"action": "next", "as_of": "2026-11-05"})
    second = call(ws, {**first["answer_call"], "payload": {"value": 2}})
    with pytest.raises(Exception, match="total too large"):
        call(ws, {**second["answer_call"], "payload": {"value": 8}})
    assert call(ws, {"action": "next", "as_of": "2026-11-05"}) == second
    assert inputs.read(ws.elicitation, date(2026, 11, 5)) == {}
    assert call(ws, {**second["answer_call"], "payload": {"value": 1}})["complete"]


def test_real_expert_steps_publish_exact_full_answer_and_refuse_bad_part(tmp_path):
    from tests.expert import fakes as expert_fakes

    ws = store.Workspace(tmp_path)
    payload = expert_fakes.shock_payload("gl.oil")

    async def interview():
        async with Client(build(ws)) as client:
            async def invoke(arguments):
                return (await client.call_tool("elicit", arguments)).data

            current = await invoke({"action": "next", "as_of": "2026-11-05",
                                    "model_id": "expert.elicited.rw", "key": "gl.oil"})
            law = {k: v for k, v in payload.items() if k not in {"shock", "responses"}}
            current = await invoke({**current["answer_call"], "payload": law})
            assert current["shock_law"] == law
            with pytest.raises(Exception, match="at_q5"):
                await invoke({**current["answer_call"], "payload": {"q5": 1}})
            for variable, by_horizon in payload["responses"].items():
                for horizon, by_node in by_horizon.items():
                    assert current["step"] == f"response:{variable}:{horizon}"
                    part = {f"at_{node}": value for node, value in by_node.items()}
                    current = await invoke({**current["answer_call"], "payload": part})
            return current

    current = asyncio.run(interview())
    assert current["key"] != "gl.oil"
    assert inputs.read(ws.elicitation, date(2026, 11, 5)) == {"gl.oil": payload}


def test_single_step_models_keep_existing_validation_and_storage(ws, monkeypatch):
    registry = plugins.build([fakes.model_plugin(macro=(fakes.InputMacroModel(),))])
    monkeypatch.setattr(plugins, "registry", lambda: registry)
    question = call(ws, {"action": "next", "as_of": "2026-11-05"})
    assert question["step"] == "answer"
    with pytest.raises(Exception, match="value"):
        call(ws, {**question["answer_call"], "payload": {"wrong": 1}})
    after = call(ws, {**question["answer_call"], "payload": {"value": 4}})
    assert after["key"] == "q2"
