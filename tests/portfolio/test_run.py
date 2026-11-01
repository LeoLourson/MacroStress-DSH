"""mst.runs over a portfolio: its report, its determinism and what it says about a bad position."""

import asyncio
import time

import numpy as np
import pytest
from fastmcp import Client

from mst import jobs, runs, store
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


def test_the_report_of_a_constrained_scenario_carries_the_price(workspace, portfolio, versions):
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
                    "value": target,
                    "basis": "assumption",
                }
            ],
        },
    )
    run_id, _ = start(workspace, REQUEST | {"scenario": "oil-fixed", "portfolio": "demo"})
    report = runs.result(workspace, run_id)["summary"]
    assert report["dist"]["distance"] > 0
    assert np.isfinite(report["dist"]["distance"])
    assert report["dist"]["standard_error"] >= 0
    assert report["dist"]["tolerances"]


def test_the_forecast_carries_the_empty_scenario_beside_it(workspace, portfolio):
    run_id, _ = start(workspace)
    stored = runs.result(workspace, run_id)
    forecast_rows = stored["forecast"]
    assert set(forecast_rows) == {"targets", "months", "drivers", "s0"}
    # The empty scenario answers the same months and targets without any condition.
    assert forecast_rows["s0"]["months"] == forecast_rows["months"]
    assert forecast_rows["s0"]["targets"] == forecast_rows["targets"]


def test_the_es_of_the_loss_carries_its_interval_and_its_trimming(workspace, portfolio):
    run_id, _ = start(workspace)
    loss = runs.result(workspace, run_id)["summary"]["loss"]
    assert loss["es95"] >= loss["p95"]
    low, high = loss["es95_interval"]
    assert low <= loss["es95"] <= high
    assert loss["trimmed_es95"] <= loss["es95"]
    assert loss["trim"] == 0.01
    assert loss["horizon_month"]


def test_the_meta_carries_the_basis_the_checks_and_the_commit(workspace, portfolio):
    run_id, _ = start(workspace)
    stored = runs.result(workspace, run_id)["run"]
    assert stored["basis"] == []  # the plain scenario has no condition to rest on anything
    assert "paths" in stored["checks"]
    assert stored["git_commit"]


def test_the_contribution_of_one_condition_is_the_difference_with_the_empty(
    workspace, portfolio, versions
):
    target = float(versions.data.y[-1, 0] + 1.0)
    scenario_tool.save(
        workspace.scenarios,
        "oil-fixed",
        {
            "narrative": "нефть",
            "horizon": 3,
            "macro": versions.macro,
            "satellites": versions.satellites,
            "approved_by": "И. Иванов",
            "constraints": [
                {
                    "terms": [{"variable": "gl.brent", "month": 1, "weight": 1.0}],
                    "value": target,
                    "basis": "assumption",
                }
            ],
        },
    )
    run_id, _ = start(
        workspace, REQUEST | {"scenario": "oil-fixed", "portfolio": "demo", "contributions": True}
    )
    stored = runs.result(workspace, run_id)
    rows = store.read_json(runs.folder(workspace, run_id) / "contributions.json")["contributions"]
    assert len(rows) == 1 and rows[0]["element"] == "constraint#1"
    # With one condition, taking it out is the empty scenario: the difference of the
    # contribution is the difference against the s0 run, on the same random numbers.
    empty = start(workspace, REQUEST | {"portfolio": "demo"})
    empty_loss = runs.result(workspace, empty[0])["summary"]["loss"]
    full_loss = stored["summary"]["loss"]
    assert rows[0]["difference"]["es95"] == pytest.approx(
        empty_loss["es95"] - full_loss["es95"], abs=1e-6
    )


def test_a_portfolio_with_an_unknown_code_is_incomplete_with_its_reason(workspace, positions):
    positions[0]["codes"] = {"secid": "NOWHERE"}
    portfolio_tool.save(workspace.portfolios, "typo", {"positions": positions[:1]})
    run_id, _ = start(workspace, REQUEST | {"portfolio": "typo"})
    stored = runs.result(workspace, run_id)
    report = stored["summary"]
    assert report["status"] == "incomplete"
    assert report["loss"] is None
    reason = report["coverage"]["excluded"][0]["reason"]
    assert "share_price" in reason and "secid NOWHERE" in reason
    assert stored["run"]["status"] == "incomplete"


def test_a_request_names_one_source_of_positions():
    with pytest.raises(ValueError, match="not both and not neither"):
        runs.RunRequest.model_validate(
            {"scenario": "oil", "draws": 5, "seed": 11, "portfolio": "demo", "positions": []}
        )
    with pytest.raises(ValueError, match="at least one position"):
        runs.RunRequest.model_validate(
            {"scenario": "oil", "draws": 5, "seed": 11, "positions": []}
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
