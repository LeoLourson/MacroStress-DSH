"""Control-plane guarantees at the MCP boundary, including existing publications."""

from datetime import date

import pytest

from mst import inputs
from tests import test_interview_control as control
from tests.test_interview_control import elicit, interview


@pytest.fixture
def ws(tmp_path, monkeypatch):
    return control.ws.__wrapped__(tmp_path, monkeypatch)


@pytest.fixture
def warned(tmp_path, monkeypatch):
    return control.warned.__wrapped__(tmp_path, monkeypatch)


def publish(workspace, agent=None):
    current = elicit(workspace, action="next", key="q1", agent=agent)
    second = elicit(workspace, **current["answer_call"], payload={"value": 1})
    return elicit(workspace, **second["answer_call"], payload={"value": 2})


def test_a_claim_cannot_be_bypassed_by_omitting_agent(ws):
    current = elicit(ws, action="next", key="q1")
    interview(ws, "claim", key="q1", agent="oil")
    with pytest.raises(Exception, match="claimed by oil"):
        elicit(ws, action="next", key="q1")
    with pytest.raises(Exception, match="claimed by oil"):
        elicit(ws, **current["answer_call"], payload={"value": 1})


def test_claimed_agent_stops_at_its_key_and_answer_call_keeps_identity(ws):
    interview(ws, "claim", key="q1", agent="oil")
    done = publish(ws, "oil")
    assert done["complete"] and done["scope"] == "q1"
    assert interview(ws, "progress")["counts"] == {"accepted": 1, "missing": 1}
    assert not interview(ws, "journal", key="q2")["events"]


def test_legacy_publication_is_audited_by_progress(warned):
    inputs.write(warned.elicitation, date(2026, 11, 5), "q1", {"first": 1, "second": 2})
    row = interview(warned, "progress")["keys"][0]
    assert row["state"] == "needs_review"
    assert row["answered"] == row["total"] == 2
    assert row["codes"] == ["rule"]


def test_reopen_of_claimed_publication_and_stale_ticket(ws):
    interview(ws, "claim", key="q1", agent="oil")
    current = elicit(ws, action="next", key="q1", agent="oil")
    second = elicit(ws, **{**current["answer_call"], "agent": "oil"}, payload={"value": 1})
    elicit(ws, **{**second["answer_call"], "agent": "oil"}, payload={"value": 2})
    reopened = interview(ws, "reopen", key="q1", agent="lead", steps=["first"], reason="review")
    with pytest.raises(Exception, match="no longer current"):
        elicit(ws, **{**current["answer_call"], "agent": "oil"}, payload={"value": 1})
    assert reopened["step"] == "first"


def test_history_includes_current_and_reviewed_versions(warned):
    old_hash = publish(warned)["published"][0]["hash"]
    interview(warned, "accept", key="q1", agent="lead", reason="reviewed")
    reopened = interview(warned, "reopen", key="q1", agent="lead", steps=["first"], reason="revise")
    result = elicit(warned, **reopened["answer_call"], payload={"value": 5})
    new_hash = result["published"][0]["hash"]
    history = interview(warned, "history", key="q1")["versions"]
    assert {row["hash"] for row in history} == {old_hash, new_hash}
    old = next(row for row in history if row["hash"] == old_hash)
    assert old["accepted"] and old["reason"] == "reviewed"
    assert next(row for row in history if row["hash"] == new_hash)["current"]


def test_journal_is_sequenced_utc_and_reports_corrupt_lines(ws):
    publish(ws)
    path = ws.elicitation / ".steps" / "gmodel.inputs.v1" / "2026-11-05" / "_journal.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write("broken\n")
    from mst.plugin_sdk.interview import journal

    with pytest.warns(UserWarning, match="journal"):
        events = journal.tail(path.parent)
    assert interview(ws, "journal")["events"] == events
    assert [row["seq"] for row in events] == list(range(1, len(events) + 1))
    assert all(row["at"].endswith("+00:00") for row in events)
