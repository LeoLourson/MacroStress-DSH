"""The scenario tool at the server: the file it writes, what it lists and what validate tells."""

import asyncio
import json
import shutil

import pytest
from fastmcp import Client

from mst import store
from mst.data.store import shift
from mst.macro.registry import load_registry
from mst.mcp import build


def call(ws, name, arguments):
    """One tool call against a server built for ``ws``."""

    async def run():
        async with Client(build(ws)) as client:
            return await client.call_tool(name, arguments)

    return asyncio.run(run())


def install(ws, versions):
    """The two model versions of the fixture, in the workspace the server reads."""
    for kind, name in (("macro", versions.macro), ("satellites", versions.satellites)):
        shutil.copytree(versions.models / kind / name, ws.models / kind / name)


def month(versions, index):
    return shift(versions.loaded.panel.months[-1], index).strftime("%Y-%m")


def level(versions, offset=0.0):
    """The last oil price of the panel in its own unit, moved by ``offset`` in model scale."""
    registry = load_registry()
    panel = versions.loaded.panel
    column = panel.ids.index("gl.brent")
    return float(registry.variable("gl.brent").from_model(panel.y[-1, column] + offset))


def observation(versions, **changes):
    return {
        "kind": "constraint",
        "terms": [{"quantity": "gl.brent", "month": month(versions, 2), "weight": 1.0}],
        "value": level(versions),
        "unit": "level",
        "basis": "assumption",
    } | changes


def spec(versions, **changes):
    values = {
        "narrative": "нефть дорожает",
        "observations": [observation(versions)],
        "approved_by": "И. Иванов",
    }
    values.update(changes)
    return values


def test_save_writes_the_event_file_and_get_reads_it_back(tmp_path, versions):
    ws = store.Workspace(tmp_path)
    saved = call(ws, "scenario", {"action": "save", "id": "oil", "spec": spec(versions)}).data
    assert saved["id"] == "oil" and saved["spec"]["narrative"] == "нефть дорожает"
    written = json.loads((ws.scenarios / "oil.json").read_text(encoding="utf-8"))
    assert set(written) == {"narrative", "sources", "interventions", "observations", "approved_by"}
    assert "macro" not in written and "horizon" not in written
    assert call(ws, "scenario", {"action": "get", "id": "oil"}).data["spec"] == written


def test_list_shows_every_event_and_names_a_broken_file(tmp_path, versions):
    ws = store.Workspace(tmp_path)
    call(ws, "scenario", {"action": "save", "id": "oil", "spec": spec(versions)})
    (ws.scenarios / "broken.json").write_text('{"sources": []}', encoding="utf-8")
    listed = call(ws, "scenario", {"action": "list"}).data
    assert [item["id"] for item in listed["scenarios"]] == ["oil"]
    assert listed["scenarios"][0]["observations"] == 1
    assert [item["id"] for item in listed["broken"]] == ["broken"]


def test_an_old_scenario_naming_versions_is_listed_as_broken(tmp_path, versions):
    ws = store.Workspace(tmp_path)
    ws.scenarios.mkdir(parents=True)
    old = {
        "narrative": "нефть дорожает",
        "horizon": 6,
        "macro": versions.macro,
        "satellites": versions.satellites,
        "approved_by": "И. Иванов",
    }
    (ws.scenarios / "old.json").write_text(json.dumps(old), encoding="utf-8")
    listed = call(ws, "scenario", {"action": "list"}).data
    assert [item["id"] for item in listed["scenarios"]] == []
    assert [item["id"] for item in listed["broken"]] == ["old"]


def test_validate_accepts_a_scenario_the_versions_can_compute(tmp_path, versions):
    ws = store.Workspace(tmp_path)
    install(ws, versions)
    intervention = {
        "quantity": "ru.key_rate",
        "months": [month(versions, 2)],
        "kind": "set",
        "values": [11.5],
        "unit": "level",
        "reason": "Решение ЦБ",
        "basis": "event",
    }
    call(
        ws,
        "scenario",
        {"action": "save", "id": "oil", "spec": spec(versions, interventions=[intervention])},
    )
    report = call(ws, "scenario", {"action": "validate", "id": "oil"}).data
    assert report["id"] == "oil" and report["ok"] is True and report["problems"] == []


def test_validate_names_the_unknown_quantity(tmp_path, versions):
    ws = store.Workspace(tmp_path)
    install(ws, versions)
    unknown = spec(
        versions,
        observations=[
            observation(
                versions,
                terms=[
                    {
                        "quantity": "gl.nowhere",
                        "month": month(versions, 2),
                        "weight": 1.0,
                    }
                ],
            )
        ],
    )
    call(ws, "scenario", {"action": "save", "id": "nowhere", "spec": unknown})
    report = call(ws, "scenario", {"action": "validate", "id": "nowhere"}).data
    assert report["ok"] is False
    assert "unknown quantity gl.nowhere" in report["problems"][0]


def test_validate_labels_what_the_event_states_and_assumes(tmp_path, versions):
    ws = store.Workspace(tmp_path)
    install(ws, versions)
    assumed = spec(
        versions,
        observations=[observation(versions, basis="event")],
        interventions=[
            {
                "quantity": "ru.key_rate",
                "months": [month(versions, 2)],
                "kind": "shift",
                "values": [1.0],
                "unit": "level",
                "reason": "Решение ЦБ",
                "basis": "assumption",
            }
        ],
    )
    call(ws, "scenario", {"action": "save", "id": "event", "spec": assumed})
    report = call(ws, "scenario", {"action": "validate", "id": "event"}).data
    assert report["labels"] == {"event": 1, "sourced": 0, "assumption": 1}


def test_validate_checks_the_corpus_only_when_it_is_indexed(tmp_path, versions):
    ws = store.Workspace(tmp_path)
    install(ws, versions)
    sourced = spec(versions, sources=["S1#1/1"])
    call(ws, "scenario", {"action": "save", "id": "oil", "spec": sourced})
    assert call(ws, "scenario", {"action": "validate", "id": "oil"}).data["ok"] is True
    (ws.corpus / "index").mkdir(parents=True)
    (ws.corpus / "index" / "chunks.jsonl").write_text("", encoding="utf-8")
    report = call(ws, "scenario", {"action": "validate", "id": "oil"}).data
    assert report["ok"] is False and "S1#1/1" in report["problems"][0]


def test_reverse_takes_the_versions_from_the_workspace(tmp_path, versions):
    # The scenario file names no versions: the reverse branch selects the pair the workspace
    # holds and then fails on the portfolio, not on a missing version of the scenario.
    ws = store.Workspace(tmp_path)
    install(ws, versions)
    call(ws, "scenario", {"action": "save", "id": "oil", "spec": spec(versions)})
    with pytest.raises(Exception, match="no portfolio 'missing'"):
        call(
            ws,
            "scenario",
            {
                "action": "reverse",
                "id": "oil",
                "portfolio": "missing",
                "kappa": 1.0,
                "draws": 2,
                "seed": 1,
            },
        )


def test_save_refuses_a_broken_contract_and_an_id_that_is_not_a_file_name(tmp_path, versions):
    ws = store.Workspace(tmp_path)
    with pytest.raises(Exception, match="not a scenario id"):
        call(ws, "scenario", {"action": "save", "id": "Bad Id", "spec": spec(versions)})
    with pytest.raises(Exception, match="no scenario"):
        call(ws, "scenario", {"action": "get", "id": "missing"})
    broken = spec(versions, observations=[observation(versions, low=2.0, high=1.0, value=None)])
    with pytest.raises(Exception, match="low < high"):
        call(ws, "scenario", {"action": "save", "id": "broken", "spec": broken})
