"""The scenario tool: the file it writes, what it lists and what validate tells."""

import asyncio
import json
import shutil

from fastmcp import Client

from mst import store
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


def spec(versions, **changes):
    values = {
        "narrative": "нефть дорожает",
        "horizon": 6,
        "macro": versions.macro,
        "satellites": versions.satellites,
        "approved_by": "И. Иванов",
    }
    values.update(changes)
    return values


def fixed(variable, month, value):
    return {
        "terms": [{"variable": variable, "month": month, "weight": 1.0}],
        "value": value,
        "basis": "assumption",
    }


def scenario(versions, **changes):
    """A scenario file whose one constraint fixes ``gl.brent`` in month 2."""

    def at(variable, month):
        return {
            "terms": [{"variable": variable, "month": month, "weight": 1.0}],
            "value": 1.0,
            "basis": "assumption",
        }

    changes.setdefault("constraints", [at("gl.brent", 2)])
    return spec(versions, **changes)


def test_save_writes_the_file_and_get_reads_it_back(tmp_path, versions):
    ws = store.Workspace(tmp_path)
    saved = call(ws, "scenario", {"action": "save", "id": "oil", "spec": spec(versions)}).data
    assert saved["id"] == "oil" and saved["spec"]["horizon"] == 6
    written = json.loads((ws.scenarios / "oil.json").read_text(encoding="utf-8"))
    assert written["macro"] == versions.macro
    assert call(ws, "scenario", {"action": "get", "id": "oil"}).data["spec"] == written


def test_list_shows_every_scenario_and_names_a_broken_one(tmp_path, versions):
    ws = store.Workspace(tmp_path)
    call(ws, "scenario", {"action": "save", "id": "oil", "spec": spec(versions)})
    (ws.scenarios / "broken.json").write_text('{"narrative": "x"}', encoding="utf-8")
    listed = call(ws, "scenario", {"action": "list"}).data
    assert [item["id"] for item in listed["scenarios"]] == ["oil"]
    assert listed["scenarios"][0]["macro"] == versions.macro
    assert [item["id"] for item in listed["broken"]] == ["broken"]


def test_an_old_scenario_with_overlays_is_listed_as_broken(tmp_path, versions):
    ws = store.Workspace(tmp_path)
    ws.scenarios.mkdir(parents=True)
    old = {
        "narrative": "нефть дорожает",
        "horizon": 6,
        "macro": versions.macro,
        "satellites": versions.satellites,
        "overlays": [
            {
                "driver": "price:alpha",
                "start": 1,
                "end": 2,
                "delta": -3.0,
                "reason": "перебой",
                "sources": ["S1#1/1"],
            }
        ],
    }
    (ws.scenarios / "old.json").write_text(json.dumps(old), encoding="utf-8")
    listed = call(ws, "scenario", {"action": "list"}).data
    assert [item["id"] for item in listed["scenarios"]] == [] 
    assert [item["id"] for item in listed["broken"]] == ["old"]


def test_validate_accepts_a_scenario_the_versions_can_compute(tmp_path, versions):
    ws = store.Workspace(tmp_path)
    install(ws, versions)
    call(ws, "scenario", {"action": "save", "id": "oil", "spec": scenario(versions)})
    report = call(ws, "scenario", {"action": "validate", "id": "oil"}).data
    assert report["id"] == "oil" and report["ok"] is True and report["problems"] == []


def test_validate_names_the_unknown_variable(tmp_path, versions):
    ws = store.Workspace(tmp_path)
    install(ws, versions)
    unknown = scenario(versions, constraints=[fixed("gl.nowhere", 2, 1.0)])
    call(ws, "scenario", {"action": "save", "id": "nowhere", "spec": unknown})
    report = call(ws, "scenario", {"action": "validate", "id": "nowhere"}).data
    assert report["ok"] is False and "unknown variable gl.nowhere" in report["problems"][0]


def test_validate_checks_the_corpus_only_when_it_is_indexed(tmp_path, versions):
    ws = store.Workspace(tmp_path)
    install(ws, versions)
    sources = spec(versions, sources=["S1#1/1"])
    call(ws, "scenario", {"action": "save", "id": "oil", "spec": sources})
    assert call(ws, "scenario", {"action": "validate", "id": "oil"}).data["ok"] is True
    (ws.corpus / "index").mkdir(parents=True)
    (ws.corpus / "index" / "chunks.jsonl").write_text("", encoding="utf-8")
    report = call(ws, "scenario", {"action": "validate", "id": "oil"}).data
    assert report["ok"] is False and "S1#1/1" in report["problems"][0]


def test_save_refuses_a_broken_contract_and_an_id_that_is_not_a_file_name(tmp_path, versions):
    ws = store.Workspace(tmp_path)
    late = scenario(versions, horizon=2, constraints=[fixed("gl.brent", 3, 1.0)])
    for arguments, message in (
        ({"action": "save", "id": "oil", "spec": late}, "beyond the horizon"),
        ({"action": "save", "id": "Bad Id", "spec": spec(versions)}, "not a scenario id"),
        ({"action": "get", "id": "missing"}, "no scenario"),
    ):
        try:
            call(ws, "scenario", arguments)
        except Exception as error:
            assert message in str(error)
        else:
            raise AssertionError(f"{arguments} must not be accepted")
