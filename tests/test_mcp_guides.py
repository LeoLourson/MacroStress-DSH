"""The server carries the way of working itself: a client needs nothing but its protocol."""

import asyncio
import json
from datetime import date

import pytest
from fastmcp import Client

from mst import guides, store
from mst.mcp import build
from mst.scenario import tool as scenario_tool
from mst.scenario.build import _labels


def session(ws, work):
    """Run ``work`` on a client connected to a server built for ``ws``."""

    async def run():
        async with Client(build(ws)) as client:
            return await work(client)

    return asyncio.run(run())


def test_any_client_is_told_to_read_the_guide_when_it_connects(tmp_path):
    async def work(client):
        return client.initialize_result.instructions

    told = session(store.Workspace(tmp_path), work)
    assert 'guide(name="scenario-from-corpus")' in told
    assert "corpus" in told and "approved_by" in told and "records no event" in told


def test_the_guide_tool_lists_and_returns_every_guide(tmp_path):
    async def work(client):
        listed = await client.call_tool("guide", {})
        one = await client.call_tool("guide", {"name": "scenario-from-corpus"})
        return listed.data, one.data

    listed, one = session(store.Workspace(tmp_path), work)
    assert {item["name"] for item in listed["guides"]} == {
        "scenario-from-corpus", "run-report", "drivers",
    }
    assert all(item["description"] and "content" not in item for item in listed["guides"])
    assert 'corpus(action="search"' in one["content"] and not one["content"].startswith("---")


def test_an_unknown_guide_names_the_guides_that_exist():
    with pytest.raises(ValueError, match="scenario-from-corpus"):
        guides.get("nowhere")


def test_the_prompt_puts_the_event_after_the_guide(tmp_path):
    async def work(client):
        prompts = await client.list_prompts()
        got = await client.get_prompt("scenario_from_event", {"event": "ставка выросла"})
        return [item.name for item in prompts], got.messages[0].content.text

    names, text = session(store.Workspace(tmp_path), work)
    assert names == ["scenario_from_event"]
    assert 'corpus(action="search"' in text and text.endswith("ставка выросла")


def test_describe_gives_what_a_scenario_is_written_against(workspace, versions):
    async def work(client):
        macro = await client.call_tool(
            "model", {"kind": "macro", "action": "describe", "version": versions.macro}
        )
        satellites = await client.call_tool(
            "model", {"kind": "satellites", "action": "describe", "version": versions.satellites}
        )
        return macro.data, satellites.data

    macro, satellites = session(workspace, work)
    assert macro["origin"]["month"] == macro["months"][-1]
    assert set(macro["origin"]["last"]) <= set(macro["ids"])
    assert isinstance(macro["latents"], list)
    assert {"id", "transform", "unit", "seasonal"} <= set(macro["variables"][0])
    assert {"id", "transform", "regressors"} <= set(satellites["quantities"][0])
    # What the tool returned crossed the protocol, so a missing value must not be a NaN.
    assert "NaN" not in json.dumps(macro["origin"])


def test_validate_reports_the_marks_and_what_stays_without_support(workspace, versions):
    term = {"terms": [{"variable": "gl.brent", "month": 1, "weight": 1.0}], "low": 0.0, "high": 1e3}
    spec = {
        "narrative": "нефть", "horizon": 3, "macro": versions.macro,
        "satellites": versions.satellites,
    }
    scenario = scenario_tool.Spec.model_validate(
        {**spec, "constraints": [{**term, "basis": "assumption"}]}
    )
    found = _labels(scenario)
    assert found["labels"] == {"event": 0, "sourced": 0, "assumption": 1}
    assert any("assumption" in warning for warning in found["warnings"])

    bare = scenario_tool.Spec.model_validate(spec)
    assert any("records no event" in warning for warning in _labels(bare)["warnings"])

    impact = {"target": "gl.brent", "start": 1, "end": 1, "delta": 1.0, "reason": "r"}
    moved = scenario_tool.Spec.model_validate(
        {**spec, "impacts": [{**impact, "basis": "assumption"}]}
    )
    assert any("Q = P" in warning for warning in _labels(moved)["warnings"])

    sourced = scenario_tool.Spec.model_validate(
        {**spec, "sources": ["S1"], "constraints": [{**term, "basis": ["S1"]}]}
    )
    assert _labels(sourced) == {
        "labels": {"event": 0, "sourced": 1, "assumption": 0},
        "warnings": [],
    }

    stated = scenario_tool.Spec.model_validate(
        {**spec, "sources": ["S1"], "constraints": [{**term, "basis": "event"}]}
    )
    assert _labels(stated) == {
        "labels": {"event": 1, "sourced": 0, "assumption": 0},
        "warnings": [],
    }


@pytest.mark.parametrize("word", ["assumption", "event"])
def test_a_word_in_a_list_is_still_the_word(versions, word):
    scenario = scenario_tool.Spec.model_validate(
        {
            "narrative": "нефть", "horizon": 3, "macro": versions.macro,
            "satellites": versions.satellites,
            "impacts": [
                {
                    "target": "gl.brent", "start": 1, "end": 1, "delta": 1.0, "reason": "r",
                    "basis": [word],
                }
            ],
        }
    )
    assert scenario.impacts[0].basis == word
    assert _labels(scenario)["labels"][word] == 1


def test_the_base_path_needs_no_calculation_and_is_in_the_model_scale(workspace, versions):
    async def work(client):
        base = await client.call_tool(
            "model",
            {
                "kind": "macro", "action": "base", "version": versions.macro,
                "horizon": 3, "only": ["gl.brent"],
            },
        )
        macro = await client.call_tool(
            "model", {"kind": "macro", "action": "describe", "version": versions.macro}
        )
        return base.data, macro.data

    base, macro = session(workspace, work)
    assert list(base["paths"]) == ["gl.brent"]
    assert len(base["months"]) == 3 and base["months"][0] > macro["origin"]["month"]
    path = base["paths"]["gl.brent"]
    assert all(low <= middle <= high for low, middle, high in zip(*path.values(), strict=True))
    assert not list(workspace.runs.glob("*"))


def test_the_base_path_refuses_a_variable_the_version_lacks(workspace, versions):
    async def work(client):
        return await client.call_tool(
            "model",
            {"kind": "macro", "action": "base", "version": versions.macro, "only": ["ru.nowhere"]},
        )

    with pytest.raises(Exception, match="ru.nowhere"):
        session(workspace, work)


def test_validate_says_where_an_impact_does_not_go(workspace, versions):
    from mst import plugins
    from mst.drivers import bundle
    from mst.scenario.build import _reach

    satellites = bundle.load(workspace.models, versions.satellites, plugins.registry().satellites)
    read = next(iter(satellites.registry.quantities[0].regressors))
    spec = {
        "narrative": "нефть", "horizon": 3, "macro": versions.macro,
        "satellites": versions.satellites,
    }
    impact = {"start": 1, "end": 1, "delta": 1.0, "reason": "r", "basis": "assumption"}
    moved = scenario_tool.Spec.model_validate(
        {**spec, "impacts": [{**impact, "target": read}, {**impact, "target": "ru.nowhere"}]}
    )
    first, second = sorted(_reach(moved, satellites), key=lambda text: "nowhere" in text)
    assert read in first and satellites.registry.quantities[0].id in first
    assert "no satellite reads" in second

    below = _driver_under_a_parent(satellites)
    if below is not None:
        parent, driver = below
        sector = scenario_tool.Spec.model_validate(
            {**spec, "impacts": [{**impact, "target": parent}]}
        )
        assert driver in _reach(sector, satellites)[0]
        listed = scenario_tool.Spec.model_validate(
            {**spec, "impacts": [{**impact, "target": parent}, {**impact, "target": driver}]}
        )
        assert all(driver not in text for text in _reach(listed, satellites))


def test_validate_refuses_satellites_of_another_macro_version(workspace, versions):
    from dataclasses import replace

    from mst import plugins
    from mst.drivers import bundle
    from mst.macro import version as macro_version
    from mst.scenario.build import _version_problems

    installed = plugins.registry()
    macro = macro_version.load(workspace.models, versions.macro, installed.macro)
    satellites = bundle.load(workspace.models, versions.satellites, installed.satellites)
    assert _version_problems(macro, satellites) == []
    other = replace(satellites, macro="f" * 64)
    assert "f" * 64 in _version_problems(macro, other)[0]


def test_the_state_names_satellites_of_the_macro_version_it_names(workspace, versions):
    import os
    import shutil

    other = workspace.models / "satellites" / ("e" * 64)
    shutil.copytree(workspace.models / "satellites" / versions.satellites, other)
    meta = json.loads((other / "meta.json").read_text(encoding="utf-8"))
    (other / "meta.json").write_text(json.dumps({**meta, "macro": "f" * 64}), encoding="utf-8")
    newest = (other / "meta.json").stat().st_mtime + 100
    os.utime(other, (newest, newest))

    found = store.state(workspace)
    assert found["macro"] == versions.macro and found["satellites"] == versions.satellites


def test_series_reads_the_history_of_a_driver_by_its_id(tmp_path):
    from mst.data.store import Observation, SeriesStore
    from mst.mcp import _rows

    ws = store.Workspace(tmp_path)
    SeriesStore(ws.series).write(
        "drv.share_price.sibn",
        [Observation(date(2020, 1, 1), 5.0, date(2020, 2, 1), approximate=False)],
    )
    found = _rows(ws, "share_price:sibn", "2021-01-01")
    assert [row["value"] for row in found["rows"]] == [5.0]


def _driver_under_a_parent(satellites) -> tuple[str, str] | None:
    """A served driver and the served driver of its quantity one object above it, if any."""
    registry, served = satellites.registry, set(satellites.served())
    for driver in registry.drivers:
        parents = registry.ancestors(driver.object)
        if driver.id in served and parents:
            parent = f"{driver.quantity}:{parents[0]}"
            if parent in served:
                return parent, driver.id
    return None


def test_validate_refuses_a_condition_on_a_fragment_published_later(workspace, versions):
    index = workspace.corpus / "index"
    index.mkdir(parents=True)
    (index / "chunks.jsonl").write_text(
        json.dumps({"chunk_id": "S1", "visible_from": "2999-01-01"}) + "\n", encoding="utf-8"
    )
    scenario_tool.save(
        workspace.scenarios,
        "cited",
        {
            "narrative": "нефть", "horizon": 3, "macro": versions.macro,
            "satellites": versions.satellites, "approved_by": "И. Иванов",
            "constraints": [
                {
                    "terms": [{"variable": "gl.brent", "month": 1, "weight": 1.0}],
                    "low": 0.0, "high": 1e3, "basis": ["S1"],
                }
            ],
        },
    )
    report = scenario_tool.validate(
        workspace.scenarios, "cited", workspace.models, workspace.corpus
    )
    assert any("no corpus fragment S1" in problem for problem in report["problems"])
