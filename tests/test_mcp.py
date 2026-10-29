"""The server answers a call in memory, so the MCP layer is checked without a transport."""

import asyncio

from fastmcp import Client

from mst import store
from mst.mcp import build


def call(ws, name, arguments):
    """One tool call against a server built for ``ws``."""

    async def run():
        async with Client(build(ws)) as client:
            return await client.call_tool(name, arguments)

    return asyncio.run(run())


def test_jobs_state_returns_the_workspace_summary(tmp_path):
    ws = store.Workspace(tmp_path)
    result = call(ws, "jobs", {"action": "state"})
    assert result.data == {
        "workspace": str(tmp_path.resolve()),
        "macro": None,
        "satellites": None,
        "corpus_index": None,
        "portfolios": 0,
        "scenarios": 0,
        "runs": 0,
    }


def test_jobs_state_refreshes_the_file_the_prompt_section_reads(tmp_path):
    ws = store.Workspace(tmp_path)
    (ws.portfolios / "demo").mkdir(parents=True)
    assert store.read_json(ws.state_file) is None
    call(ws, "jobs", {"action": "state"})
    assert store.read_json(ws.state_file)["portfolios"] == 1


def test_jobs_rejects_an_unknown_action(tmp_path):
    ws = store.Workspace(tmp_path)
    try:
        call(ws, "jobs", {"action": "start"})
    except Exception as error:  # fastmcp raises ToolError with the message of the tool
        assert "unknown action" in str(error)
    else:
        raise AssertionError("an unknown action must not be accepted")
