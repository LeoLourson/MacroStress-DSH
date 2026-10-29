"""The server answers a call in memory, so the MCP layer is checked without a transport."""

import asyncio

from fastmcp import Client

from mst import jobs, store
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
        "jobs": 0,
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


def test_jobs_status_log_and_cancel_go_through_the_tool(tmp_path):
    ws = store.Workspace(tmp_path)
    job_id = jobs.start(ws, "code", ["-c", "import time; time.sleep(30)"])
    try:
        assert call(ws, "jobs", {"action": "status", "id": job_id}).data["running"] is True
        assert call(ws, "jobs", {"action": "log", "id": job_id}).data == {"id": job_id, "log": ""}
    finally:
        cancelled = call(ws, "jobs", {"action": "cancel", "id": job_id}).data
    assert cancelled["cancelled"] is True
    assert cancelled["running"] is False


def test_jobs_without_an_id_is_refused(tmp_path):
    ws = store.Workspace(tmp_path)
    try:
        call(ws, "jobs", {"action": "status"})
    except Exception as error:
        assert "needs id" in str(error)
    else:
        raise AssertionError("status without an id must not be accepted")
