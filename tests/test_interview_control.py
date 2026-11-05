"""Ownership and progress of an interview through the same MCP boundary an agent uses."""

import asyncio
import json

import pytest
from fastmcp import Client

from mst import plugins, store
from mst.mcp import build
from tests.plugins import fakes


class TwoSteps(fakes.InputMacroModel):
    def input_keys(self, as_of):
        return ("q1", "q2")

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
        return {"first": parts["first"]["value"], "second": parts["second"]["value"]}

    def input_parts(self, context, key, payload):
        return {
            "first": {"value": payload["first"]},
            "second": {"value": payload["second"]},
        }

    def check_input(self, context, key, payload):
        if set(payload) != {"first", "second"}:
            raise ValueError("the answer needs first and second")
        return payload


class Warned(TwoSteps):
    """A model whose published answers always carry one advisory finding."""

    def audit_input(self, context, key, payload):
        return [{"code": "rule", "message": "a test finding", "cells": []}]


@pytest.fixture
def ws(tmp_path, monkeypatch):
    registry = plugins.build([fakes.model_plugin(macro=(TwoSteps(),))])
    monkeypatch.setattr(plugins, "registry", lambda: registry)
    return store.Workspace(tmp_path)


@pytest.fixture
def warned(tmp_path, monkeypatch):
    registry = plugins.build([fakes.model_plugin(macro=(Warned(),))])
    monkeypatch.setattr(plugins, "registry", lambda: registry)
    return store.Workspace(tmp_path)


def call(ws, tool, arguments):
    async def invoke():
        async with Client(build(ws)) as client:
            return (await client.call_tool(tool, arguments)).data

    return asyncio.run(invoke())


def interview(ws, action, **arguments):
    return call(
        ws,
        "interview",
        {"action": action, "as_of": "2026-11-05", "model_id": "gmodel.inputs.v1", **arguments},
    )


def elicit(ws, **arguments):
    return call(ws, "elicit", {"as_of": "2026-11-05", "model_id": "gmodel.inputs.v1", **arguments})


def test_a_second_agent_cannot_take_a_claimed_key(ws):
    assert interview(ws, "claim", key="q1", agent="oil")["claim"]["agent"] == "oil"
    with pytest.raises(Exception, match="claimed by oil"):
        interview(ws, "claim", key="q1", agent="demand")
    with pytest.raises(Exception, match="claimed by oil"):
        elicit(ws, action="next", key="q1", agent="demand")


def test_only_the_owner_releases_a_claim(ws):
    interview(ws, "claim", key="q1", agent="oil")
    with pytest.raises(Exception, match="only its owner"):
        interview(ws, "release", key="q1", agent="demand")
    assert interview(ws, "release", key="q1", agent="oil")["released"] is True
    assert interview(ws, "claim", key="q1", agent="demand")["claim"]["agent"] == "demand"


def test_an_expired_claim_does_not_block(ws):
    interview(ws, "claim", key="q1", agent="oil", minutes=0)
    assert interview(ws, "claim", key="q1", agent="demand")["claim"]["agent"] == "demand"


def test_progress_counts_the_parts_and_the_owner(ws):
    interview(ws, "claim", key="q1", agent="oil")
    current = elicit(ws, action="next", key="q1", agent="oil")
    elicit(ws, action="answer", ticket=current["answer_call"]["ticket"],
           payload={"value": 1}, agent="oil")

    rows = {row["key"]: row for row in interview(ws, "progress")["keys"]}
    assert rows["q1"]["state"] == "claimed"
    assert (rows["q1"]["answered"], rows["q1"]["total"]) == (1, 2)
    assert rows["q1"]["claim"]["agent"] == "oil"
    assert rows["q2"]["state"] == "missing"

    second = elicit(ws, action="next", key="q1", agent="oil")
    elicit(ws, action="answer", ticket=second["answer_call"]["ticket"],
           payload={"value": 2}, agent="oil")
    rows = {row["key"]: row for row in interview(ws, "progress")["keys"]}
    assert rows["q1"]["state"] == "accepted" and rows["q1"]["hash"]
    assert interview(ws, "progress")["counts"] == {"accepted": 1, "missing": 1}


def test_the_journal_records_the_flow_without_the_numbers(ws):
    interview(ws, "claim", key="q1", agent="oil")
    current = elicit(ws, action="next", key="q1", agent="oil")
    second = elicit(ws, action="answer", ticket=current["answer_call"]["ticket"],
                    payload={"value": 1}, agent="oil")
    elicit(ws, action="answer", ticket=second["answer_call"]["ticket"],
           payload={"value": 2}, agent="oil")

    events = interview(ws, "journal", key="q1")["events"]
    assert [event["action"] for event in events] == [
        "claim", "next", "answer", "next", "answer", "publish", "accept",
    ]
    assert events[0]["agent"] == "oil" and events[-1]["hash"]
    assert "value" not in json.dumps(events)


def test_a_publication_with_findings_needs_a_review_and_an_acceptance(warned):
    current = elicit(warned, action="next", key="q1", agent="oil")
    answered = elicit(warned, action="answer", ticket=current["answer_call"]["ticket"],
                      payload={"value": 1}, agent="oil")
    second = elicit(warned, action="answer", ticket=answered["answer_call"]["ticket"],
                    payload={"value": 2}, agent="oil")
    assert second["published"][0]["needs_review"] is True
    rows = {row["key"]: row for row in interview(warned, "progress")["keys"]}
    assert rows["q1"]["state"] == "needs_review"
    assert rows["q1"]["warnings"] == 1 and rows["q1"]["codes"] == ["rule"]

    with pytest.raises(Exception, match="reason"):
        interview(warned, "accept", key="q1", agent="lead", reason="")
    accepted = interview(warned, "accept", key="q1", agent="lead", reason="checked by hand")
    assert accepted["accepted"]["hash"] == rows["q1"]["hash"]
    rows = {row["key"]: row for row in interview(warned, "progress")["keys"]}
    assert rows["q1"]["state"] == "accepted" and rows["q1"]["accepted_by"] == "lead"


def test_reopen_keeps_the_old_answer_and_asks_the_named_step_again(ws):
    current = elicit(ws, action="next", key="q1", agent="oil")
    first = elicit(ws, action="answer", ticket=current["answer_call"]["ticket"],
                   payload={"value": 1}, agent="oil")
    done = elicit(ws, action="answer", ticket=first["answer_call"]["ticket"],
                  payload={"value": 2}, agent="oil")
    old_hash = done["published"][0]["hash"]

    reopened = interview(ws, "reopen", key="q1", agent="lead", steps=["first"],
                         reason="the first judgement was copied")
    assert reopened["step"] == "first"
    rows = {row["key"]: row for row in interview(ws, "progress")["keys"]}
    assert rows["q1"]["state"] == "in_progress" and rows["q1"]["answered"] == 1
    assert rows["q2"]["state"] == "missing"

    again = elicit(ws, action="answer",
                   ticket=reopened["answer_call"]["ticket"], payload={"value": 5},
                   agent="oil")
    new_hash = again["published"][0]["hash"]
    assert new_hash != old_hash

    from datetime import date

    from mst import inputs

    assert inputs.read(ws.elicitation, date(2026, 11, 5))["q1"]["first"] == 5
    history = interview(ws, "history", key="q1")
    assert history["current"] == new_hash
    assert {row["hash"] for row in history["versions"]} == {old_hash, new_hash}


def test_reopen_needs_a_published_answer_and_known_steps(ws):
    with pytest.raises(Exception, match="nothing to reopen"):
        interview(ws, "reopen", key="q1", agent="lead", steps=["first"], reason="x")
    current = elicit(ws, action="next", key="q1", agent="oil")
    first = elicit(ws, action="answer", ticket=current["answer_call"]["ticket"],
                   payload={"value": 1}, agent="oil")
    elicit(ws, action="answer", ticket=first["answer_call"]["ticket"],
           payload={"value": 2}, agent="oil")
    with pytest.raises(Exception, match="has no steps"):
        interview(ws, "reopen", key="q1", agent="lead", steps=["nowhere"], reason="x")
