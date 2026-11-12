"""The server carries the way of working itself: a client needs nothing but its protocol."""

import asyncio
import json
from datetime import date

import pytest
from fastmcp import Client

from mst import guides, store
from mst.mcp import build
from mst.scenario import tool as scenario_tool
from mst.scenario.build import dist_of
from mst.scenario.build import validate as validate_scenario


def calendar(versions, index: int) -> str:
    """The calendar month of horizon month ``index`` (from one) of the test panel."""
    from mst.scenario.translate import calendar_month

    return calendar_month(index, versions.data.months[-1].strftime("%Y-%m"))


def loaded(workspace, versions):
    """The macro version and the satellites bundle of the workspace, loaded as a run does."""
    from mst import plugins
    from mst.drivers import bundle
    from mst.macro import version as macro_version

    installed = plugins.registry()
    macro = macro_version.load(workspace.models, versions.macro, installed.macro)
    satellites = bundle.load(workspace.models, versions.satellites, installed.satellites)
    return macro, satellites


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
        "scenario-from-corpus", "run-report", "drivers", "model-inputs",
        "elicitation-orchestration",
    }
    assert all(item["description"] and "content" not in item for item in listed["guides"])
    assert 'corpus(action="search"' in one["content"] and not one["content"].startswith("---")


def test_the_orchestration_guide_is_served_and_named_by_the_instructions(tmp_path):
    async def work(client):
        told = client.initialize_result.instructions
        orchestration = await client.call_tool(
            "guide", {"name": "elicitation-orchestration"}
        )
        return told, orchestration.data

    told, orchestration = session(store.Workspace(tmp_path), work)
    assert 'guide(name="elicitation-orchestration")' in told
    assert 'interview(action="progress"' in orchestration["content"]
    assert "model-inputs" in orchestration["content"]


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
    macro, satellites = loaded(workspace, versions)
    bare = {"narrative": "нефть"}
    term = {
        "kind": "constraint",
        "terms": [{"quantity": "gl.brent", "month": calendar(versions, 1), "weight": 1.0}],
        "unit": "level",
        "low": 1.0,
        "high": 1e3,
    }
    one = scenario_tool.Spec.model_validate(
        {**bare, "observations": [{**term, "basis": "assumption"}]}
    )
    found = validate_scenario(one, macro, satellites, as_of=macro.as_of)
    assert found["labels"] == {"event": 0, "sourced": 0, "assumption": 1}
    assert any("assumption" in warning for warning in found["warnings"])

    empty = scenario_tool.Spec.model_validate(bare)
    assert any(
        "base scenario" in warning
        for warning in validate_scenario(empty, macro, satellites, as_of=macro.as_of)["warnings"]
    )

    # An intervention alone is not an event: there is no condition to price, so Q = P.
    moved = scenario_tool.Spec.model_validate(
        {
            **bare,
            "interventions": [
                {
                    "quantity": "gl.brent",
                    "months": [calendar(versions, 1)],
                    "kind": "shift",
                    "values": [0.1],
                    "unit": "level",
                    "reason": "r",
                    "basis": "assumption",
                }
            ],
        }
    )
    assert dist_of(moved, macro, satellites, draws=1, seed=0) is None

    sourced = scenario_tool.Spec.model_validate(
        {
            **bare,
            "approved_by": "И. Иванов",
            "sources": ["S1"],
            "observations": [{**term, "basis": ["S1"]}],
        }
    )
    assert validate_scenario(sourced, macro, satellites, as_of=macro.as_of) == {
        "ok": True,
        "problems": [],
        "labels": {"event": 0, "sourced": 1, "assumption": 0},
        "warnings": [],
    }

    stated = scenario_tool.Spec.model_validate(
        {
            **bare,
            "approved_by": "И. Иванов",
            "sources": ["S1"],
            "observations": [{**term, "basis": "event"}],
        }
    )
    assert validate_scenario(stated, macro, satellites, as_of=macro.as_of) == {
        "ok": True,
        "problems": [],
        "labels": {"event": 1, "sourced": 0, "assumption": 0},
        "warnings": [],
    }


@pytest.mark.parametrize("word", ["assumption", "event"])
def test_a_word_in_a_list_is_still_the_word(workspace, versions, word):
    macro, satellites = loaded(workspace, versions)
    scenario = scenario_tool.Spec.model_validate(
        {
            "narrative": "нефть",
            "interventions": [
                {
                    "quantity": "gl.brent",
                    "months": [calendar(versions, 1)],
                    "kind": "shift",
                    "values": [0.1],
                    "unit": "level",
                    "reason": "r",
                    "basis": [word],
                }
            ],
        }
    )
    assert scenario.interventions[0].basis == word
    assert validate_scenario(scenario, macro, satellites, as_of=macro.as_of)["labels"][word] == 1


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


def test_validate_names_an_intervention_on_a_quantity_no_model_knows(workspace, versions):
    # The reach warnings of an intervention went with the impact list: what remains of that
    # check is that a quantity no registry knows is refused instead of silently dropped.
    macro, satellites = loaded(workspace, versions)
    spec = {
        "narrative": "нефть",
        "interventions": [
            {
                "quantity": "ru.nowhere",
                "months": [calendar(versions, 1)],
                "kind": "shift",
                "values": [0.1],
                "unit": "level",
                "reason": "r",
                "basis": "assumption",
            }
        ],
    }
    moved = scenario_tool.Spec.model_validate(spec)
    problems = validate_scenario(moved, macro, satellites, as_of=macro.as_of)["problems"]
    assert any("ru.nowhere" in problem for problem in problems)


def test_validate_refuses_satellites_of_another_macro_version(workspace, versions):
    from dataclasses import replace

    macro, satellites = loaded(workspace, versions)
    spec = scenario_tool.Spec.model_validate({"narrative": "нефть", "approved_by": "И. Иванов"})
    assert validate_scenario(spec, macro, satellites, as_of=macro.as_of)["problems"] == []
    other = replace(satellites, macro="f" * 64)
    problems = validate_scenario(spec, macro, other, as_of=macro.as_of)["problems"]
    assert "f" * 64 in problems[0]


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
            "narrative": "нефть",
            "approved_by": "И. Иванов",
            "observations": [
                {
                    "kind": "constraint",
                    "terms": [
                        {"quantity": "gl.brent", "month": calendar(versions, 1), "weight": 1.0}
                    ],
                    "unit": "level",
                    "low": 1.0,
                    "high": 1e3,
                    "basis": ["S1"],
                }
            ],
        },
    )
    report = scenario_tool.validate(
        workspace.scenarios, "cited", workspace.models, workspace.corpus
    )
    assert any("no corpus fragment S1" in problem for problem in report["problems"])
