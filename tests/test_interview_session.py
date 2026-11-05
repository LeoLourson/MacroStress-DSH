"""The interview session of a model with declared inputs: tickets and their refusals."""

import asyncio
import json
from datetime import datetime, timedelta

import pytest
from fastmcp import Client

from mst import plugins, store
from mst.mcp import build
from mst.plugin_sdk.interview import address, tickets
from tests.plugins import fakes


@pytest.fixture
def ws(tmp_path, monkeypatch):
    registry = plugins.build([fakes.model_plugin(macro=(fakes.InputMacroModel(),))])
    monkeypatch.setattr(plugins, "registry", lambda: registry)
    return store.Workspace(tmp_path)


def call(ws, arguments):
    async def invoke():
        async with Client(build(ws)) as client:
            return (await client.call_tool("elicit", arguments)).data

    return asyncio.run(invoke())


def test_a_ticket_addresses_the_model_date_key_and_step(ws):
    first = call(ws, {"action": "next", "as_of": "2026-11-05"})
    assert address(ws.elicitation, first["answer_call"]["ticket"]) == {
        "model": "gmodel.inputs.v1",
        "date": "2026-11-05",
        "key": "q1",
        "step": "answer",
    }


def test_an_answer_without_payload_is_refused(ws):
    first = call(ws, {"action": "next", "as_of": "2026-11-05"})
    with pytest.raises(Exception, match="payload"):
        call(ws, {"action": "answer", "ticket": first["answer_call"]["ticket"]})


def test_tickets_are_one_map_pruned_after_a_week(ws):
    first = call(ws, {"action": "next", "as_of": "2026-11-05"})
    call(ws, {**first["answer_call"], "payload": {"value": 1}})
    path = ws.elicitation / ".steps" / "tickets.json"
    known = json.loads(path.read_text(encoding="utf-8"))
    assert len(known) == 2
    assert {entry["key"] for entry in known.values()} == {"q1", "q2"}
    assert not (ws.elicitation / ".steps" / "tickets").exists()
    assert set(next(iter(known.values()))) == {"model", "date", "key", "step", "at"}

    old = (datetime.now() - timedelta(days=8)).isoformat(timespec="seconds")
    known["0" * 16] = {
        "model": "gone",
        "date": "2026-01-01",
        "key": "q2",
        "step": "answer",
        "at": old,
    }
    path.write_text(json.dumps(known), encoding="utf-8")
    tickets.encode(ws.elicitation, "gmodel.inputs.v1", datetime(2026, 11, 5).date(), "q2", "a")
    assert "0" * 16 not in json.loads(path.read_text(encoding="utf-8"))
