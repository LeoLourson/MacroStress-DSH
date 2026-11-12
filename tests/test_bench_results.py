"""The results of the check sets are kept beside them: the scenarios of scenario-from-corpus with
the journal of the agent's corpus calls and an empty review the expert fills by hand."""

import asyncio
import json
import tomllib

import pytest
from fastmcp import Client

from mst import bench, store
from mst.corpus import tool as corpus_tool
from mst.mcp import build

EVENTS = '''[[events]]
id = "E001"
text = "ЦБ поднимает ставку на 3 п. п."

[[events]]
id = "E002"
text = "Нефть дешевеет вдвое."
'''


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A bench/ with two events, put where the server looks for it, and a corpus whose list
    answers without a manifest."""
    folder = tmp_path / "bench"
    (folder / "scenario-from-corpus").mkdir(parents=True)
    (folder / "scenario-from-corpus" / "events.toml").write_text(EVENTS, encoding="utf-8")
    monkeypatch.setattr(bench, "ROOT", folder)
    monkeypatch.setattr(corpus_tool, "catalogue", lambda corpus: {"sources": ["S1"]})
    return folder


def spec(**changes):
    return {"narrative": "ставка растёт"} | changes


def call(ws, steps):
    """Tool calls of one server, in order, as one agent session makes them."""

    async def run():
        async with Client(build(ws)) as client:
            return [(await client.call_tool(name, arguments)).data for name, arguments in steps]

    return asyncio.run(run())


def test_the_agent_gets_the_text_of_the_event_and_nothing_else(root, tmp_path):
    [found] = call(store.Workspace(tmp_path / "ws"), [("scenario", {"action": "bench",
                                                                    "event": "E002"})])

    assert found["event"] == "E002"
    assert found["text"] == "Нефть дешевеет вдвое."
    assert "E001" not in json.dumps(found, ensure_ascii=False)


def test_the_saved_scenario_keeps_the_corpus_calls_and_an_empty_review(root, tmp_path):
    ws = store.Workspace(tmp_path / "ws")
    call(ws, [
        ("scenario", {"action": "bench", "event": "E001"}),
        ("corpus", {"action": "list"}),
        ("scenario", {"action": "bench_save", "event": "E001", "spec": spec(),
                      "notes": "ставка из текста события"}),
    ])

    folder = root / "scenario-from-corpus" / "results" / "E001"
    assert store.read_json(folder / "scenario.json")["narrative"] == "ставка растёт"
    journal = store.read_json(folder / "journal.json")
    assert journal["text"] == "ЦБ поднимает ставку на 3 п. п."
    assert [(entry["action"], entry["result"]) for entry in journal["calls"]] == [
        ("list", {"sources": ["S1"]})
    ]
    assert journal["notes"] == "ставка из текста события"
    review = tomllib.loads((folder / "review.toml").read_text(encoding="utf-8"))
    assert review == {"verdict": "", "comment": ""}


def test_corpus_calls_outside_a_bench_session_are_not_journaled(root, tmp_path):
    ws = store.Workspace(tmp_path / "ws")
    call(ws, [
        ("corpus", {"action": "list"}),
        ("scenario", {"action": "bench", "event": "E001"}),
        ("scenario", {"action": "bench_save", "event": "E001", "spec": spec()}),
    ])

    journal = store.read_json(root / "scenario-from-corpus" / "results" / "E001" / "journal.json")
    assert journal["calls"] == []


def test_a_scenario_is_saved_only_for_the_event_the_session_started(root, tmp_path):
    ws = store.Workspace(tmp_path / "ws")
    with pytest.raises(Exception, match="E001"):
        call(ws, [
            ("scenario", {"action": "bench", "event": "E001"}),
            ("scenario", {"action": "bench_save", "event": "E002", "spec": spec()}),
        ])
    assert not (root / "scenario-from-corpus" / "results").exists()


def test_a_bench_scenario_is_not_approved_by_the_agent(root):
    with pytest.raises(ValueError, match="approved_by"):
        bench.save_scenario("E001", spec(approved_by="агент"), [])


def test_an_unknown_event_is_refused(root):
    with pytest.raises(ValueError, match="E404"):
        bench.event("E404")


def test_the_summary_counts_the_verdicts_the_expert_wrote(root):
    for id in ("E001", "E002"):
        bench.save_scenario(id, spec(), [])
    review = root / "scenario-from-corpus" / "results" / "E001" / "review.toml"
    review.write_text('verdict = "partial"\ncomment = "шок ставки без курса"\n', encoding="utf-8")

    summary = bench.scenario_summary()

    assert "| partial | 1 | 0.50 |" in summary
    assert "| not reviewed | 1 | 0.50 |" in summary
    assert "| E001 | partial |" in summary


def test_a_verdict_outside_the_scale_is_named(root):
    bench.save_scenario("E001", spec(), [])
    review = root / "scenario-from-corpus" / "results" / "E001" / "review.toml"
    review.write_text('verdict = "good"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="good"):
        bench.scenario_summary()


def test_a_result_is_written_over_the_last_one(tmp_path):
    bench.save_result(tmp_path, "m", {"value": 1}, "first\n")
    bench.save_result(tmp_path, "m", {"value": 2}, "second\n")

    assert store.read_json(tmp_path / "m.json") == {"value": 2}
    assert (tmp_path / "m.md").read_bytes() == b"second\n"
