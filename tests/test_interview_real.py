"""Installed expert models through MCP, with independent test data and no live workspace."""

import asyncio
from datetime import date

from fastmcp import Client

from mst import inputs, models, store
from mst.mcp import _macro_context, build
from tests.expert import fakes

DAY = date(2026, 11, 5)
MODEL = "expert.elicited.rw"


def published_expert(tmp_path):
    """A legacy publication: normal answer only, without interview session metadata."""
    ws = store.Workspace(tmp_path)
    fakes.series_store(ws.series)
    context = _macro_context(ws, DAY)
    model = models.selected(MODEL)
    full = model.check_input(context, "gl.oil", fakes.shock_payload("gl.oil"))
    inputs.write(ws.elicitation, DAY, "gl.oil", full)
    return ws, model, context, full


async def interview(client, action, *, model_id=MODEL, **arguments):
    return (
        await client.call_tool(
            "interview",
            {"action": action, "as_of": DAY.isoformat(), "model_id": model_id, **arguments},
        )
    ).data


def test_installed_expert_progress_audits_a_legacy_publication(tmp_path):
    ws, _, _, full = published_expert(tmp_path)

    async def exercise():
        async with Client(build(ws)) as client:
            progress = await interview(client, "progress")
            oil = next(row for row in progress["keys"] if row["key"] == "gl.oil")
            audited = await interview(client, "audit", key="gl.oil")
            findings = audited["keys"][0]["findings"]
            assert findings
            assert oil["hash"] == inputs.hash_of(full)
            assert oil["state"] == "needs_review"
            assert oil["answered"] == oil["total"] == 93
            assert oil["warnings"] == len(findings)
            assert set(oil["codes"]) == {row["code"] for row in findings}

    asyncio.run(exercise())


def test_installed_expert_round_trip_reopen_and_republish(tmp_path):
    ws, model, context, full = published_expert(tmp_path)
    old_hash = inputs.hash_of(full)
    parts = model.input_parts(context, "gl.oil", full)
    assert len(parts) == 93
    assert inputs.hash_of(model.assemble_input(context, "gl.oil", parts)) == old_hash
    step = "response:ru.imoex:12"
    changed = {
        node: {quantile: value + 0.01 for quantile, value in cell.items()}
        for node, cell in parts[step].items()
    }

    async def exercise():
        async with Client(build(ws)) as client:
            reopened = await interview(
                client, "reopen", key="gl.oil", steps=[step], agent="reviewer",
                reason="Recheck one equity horizon in the isolated test workspace",
            )
            assert reopened["key"] == "gl.oil" and reopened["step"] == step
            assert reopened["progress"] == {"answered": 92, "total": 93}
            assert "gl.oil" not in inputs.read(ws.elicitation, DAY)
            result = (
                await client.call_tool(
                    "elicit", {**reopened["answer_call"], "payload": changed, "agent": "reviewer"},
                )
            ).data
            new_hash = result["published"][0]["hash"]
            assert new_hash != old_hash
            current = inputs.read(ws.elicitation, DAY)["gl.oil"]
            current_parts = model.input_parts(context, "gl.oil", current)
            assert current_parts[step] == changed
            assert {key: value for key, value in current_parts.items() if key != step} == {
                key: value for key, value in parts.items() if key != step
            }
            history = await interview(client, "history", key="gl.oil")
            assert history["current"] == new_hash
            assert {version["hash"] for version in history["versions"]} == {old_hash, new_hash}
            assert next(row for row in history["versions"] if row["current"])["hash"] == new_hash

    asyncio.run(exercise())


def test_installed_expert_market_audit_does_not_read_shock_responses(tmp_path, monkeypatch):
    import mst.mcp as mcp

    ws = store.Workspace(tmp_path)
    monkeypatch.setattr(mcp, "_model_fragment", lambda workspace, day: fakes.visible)

    async def exercise():
        async with Client(build(ws)) as client:
            written = (
                await client.call_tool(
                    "model", {
                        "kind": "inputs", "action": "write", "as_of": DAY.isoformat(),
                        "model_id": "expert.elicited.market", "key": "market",
                        "payload": fakes.market_payload(),
                    },
                )
            ).data
            assert written
            audited = await interview(
                client, "audit", model_id="expert.elicited.market", key="market",
            )
            assert audited["keys"][0]["findings"] == []
            progress = await interview(client, "progress", model_id="expert.elicited.market")
            market = next(row for row in progress["keys"] if row["key"] == "market")
            assert market["answered"] == market["total"] == 1
            assert market["warnings"] == 0

    asyncio.run(exercise())
