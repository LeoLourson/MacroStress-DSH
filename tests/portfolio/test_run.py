"""mst.runs over a portfolio: its report, its determinism and what it says about a bad position."""

import asyncio
import time

import pytest
from fastmcp import Client

from mst import jobs, runs
from mst.mcp import build as server
from mst.portfolio import tool as portfolio_tool
from mst.scenario import tool as scenario_tool

REQUEST = {"scenario": "oil", "draws": 5, "seed": 11}


def start(ws, request=None):
    run_id = runs.new_id()
    runs.begin(ws, run_id, request or REQUEST | {"portfolio": "demo"})
    return run_id, runs.execute(ws, run_id)


def call(ws, name, arguments):
    """One tool call against a server built for ``ws``."""

    async def run():
        async with Client(server(ws)) as client:
            return await client.call_tool(name, arguments)

    return asyncio.run(run())


def wait(ws, job_id, timeout=120.0):
    """Wait until the job's process is gone."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not jobs.status(ws, job_id)["running"]:
            return
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} did not finish in {timeout} s")


def test_a_portfolio_run_writes_its_report(workspace, portfolio):
    run_id, summary = start(workspace)
    stored = runs.result(workspace, run_id)
    report = stored["summary"]
    assert report["loss"] is not None
    assert report["loss"]["p05"] <= report["loss"]["p50"] <= report["loss"]["p95"]
    assert report["loss"]["es95"] >= report["loss"]["p95"]
    assert report["severity"] is None
    assert report["status"] == "ok" and report["coverage"]["share"] == 1.0
    assert report["portfolio"] == {"name": "Демо", "base_currency": "RUB", "positions": 4}
    assert len(report["positions"]) == 4
    assert stored["run"]["portfolio"] == portfolio
    assert stored["run"]["status"] == "ok"
    assert summary["status"] == "ok"


def test_two_runs_of_one_portfolio_give_the_same_numbers(workspace, portfolio):
    first, _ = start(workspace)
    second, _ = start(workspace)
    one = runs.result(workspace, first)["summary"]["loss"]
    two = runs.result(workspace, second)["summary"]["loss"]
    assert one == two


def test_the_report_of_a_constrained_scenario_carries_delta(workspace, portfolio, versions):
    target = float(versions.data.y[-1, 0] + 1.0)
    scenario_tool.save(
        workspace.scenarios,
        "oil-fixed",
        {
            "narrative": "нефть",
            "horizon": 3,
            "macro": versions.macro,
            "satellites": versions.satellites,
            "constraints": [
                {
                    "terms": [{"variable": "gl.brent", "month": 1, "weight": 1.0}],
                    "low": target,
                    "high": target,
                }
            ],
        },
    )
    run_id, _ = start(workspace, REQUEST | {"scenario": "oil-fixed", "portfolio": "demo"})
    report = runs.result(workspace, run_id)["summary"]
    assert report["severity"]["distance"] > 0
    assert report["severity"]["exceeded"] > 0
    assert len(report["severity"]["probabilities"]) == 1


def test_a_portfolio_with_an_unknown_code_is_incomplete_with_its_reason(workspace, positions):
    positions[0]["codes"] = {"secid": "NOWHERE"}
    portfolio_tool.save(workspace.portfolios, "typo", {"positions": positions[:1]})
    run_id, _ = start(workspace, REQUEST | {"portfolio": "typo"})
    stored = runs.result(workspace, run_id)
    report = stored["summary"]
    assert report["status"] == "incomplete"
    assert report["loss"] is None
    assert "driver_missing:share_price" in report["coverage"]["excluded"][0]["reason"]
    assert stored["run"]["status"] == "incomplete"


def test_a_request_names_one_source_of_positions():
    with pytest.raises(ValueError, match="not both and not neither"):
        runs.RunRequest.model_validate(
            {"scenario": "oil", "draws": 5, "seed": 11, "portfolio": "demo", "positions": []}
        )


def test_the_run_tool_starts_a_portfolio_run(workspace, portfolio):
    started = call(
        workspace,
        "run",
        {"action": "start", "scenario": "oil", "draws": 3, "seed": 5, "portfolio": portfolio},
    ).data
    wait(workspace, started["job"])
    assert jobs.status(workspace, started["job"])["exit_code"] == 0
    stored = call(workspace, "run", {"action": "result", "id": started["run"]}).data
    assert stored["summary"]["loss"] is not None
