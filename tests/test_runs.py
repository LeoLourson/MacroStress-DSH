"""mst.runs: a run of a scenario writes its paths, its results and its meta in the workspace."""

import asyncio
import time
from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest
from fastmcp import Client

from mst import jobs, runs
from mst.mcp import build

POSITIONS = [
    {
        "id": "sber",
        "name": "Сбербанк",
        "type": "common_stock",
        "notional": 10.0,
        "codes": {"secid": "SBER"},
    },
    {
        "id": "corp",
        "type": "bond",
        "notional": 5.0,
        "codes": {"isin": "RU000A0JX0J2"},
        "terms": {"duration": 2.0},
    },
    {
        "id": "mining",
        "type": "loan",
        "notional": 3.0,
        "codes": {"okved": "05"},
        "terms": {"lgd": 0.45},
    },
    {
        "id": "flat",
        "type": "real_estate",
        "notional": 1.0,
        "codes": {"okato": "14000000000"},
        "terms": {"market": "primary"},
    },
]
REQUEST = {"scenario": "oil", "draws": 5, "seed": 11, "positions": POSITIONS}


def start(ws, request=None, run_id=None):
    run_id = run_id or runs.new_id()
    runs.begin(ws, run_id, request or REQUEST)
    return run_id, runs.execute(ws, run_id)


def arrays(ws, run_id):
    meta = runs.meta(ws, run_id)
    with np.load(runs.folder(ws, run_id) / "paths.npz") as stored:
        macro, drivers = stored["macro"], stored["drivers"]
    with np.load(runs.folder(ws, run_id) / "changes.npz") as stored:
        changes = stored["changes"]
    return meta, macro, drivers, changes


def driver(meta, drivers, name):
    return drivers[:, :, meta["drivers"].index(name)]


def test_a_run_writes_meta_paths_results_and_changes(workspace):
    run_id, summary = start(workspace)
    assert summary["status"] == "ok" and summary["positions"] == len(POSITIONS)
    meta, macro, drivers, changes = arrays(workspace, run_id)
    assert meta["status"] == "ok"
    assert meta["scenario"] == "oil" and meta["seed"] == 11 and meta["draws"] == 5
    assert meta["horizon"] == 3 and len(meta["months"]) == 4
    assert meta["severity"] is None
    assert macro.shape == (5, 4, 3) and drivers.shape == (5, 4, 5)
    assert changes.shape == (5, len(POSITIONS))
    stored = runs.result(workspace, run_id)
    assert [item["position_id"] for item in stored["positions"]] == [
        position["id"] for position in POSITIONS
    ]
    assert all(item["status"] == "ok" for item in stored["positions"])
    assert all("value_change" not in item for item in stored["positions"])

    # The conditional forecast of the target indicators: quantiles by month, own units.
    forecast = stored["forecast"]
    assert forecast["targets"] == meta["drivers"]
    assert forecast["months"] == meta["months"][1:]
    for name in meta["drivers"]:
        row = forecast["drivers"][name]
        column = driver(meta, drivers, name)[:, 1:]
        low, mid, high = np.percentile(column, [5, 50, 95], axis=0)
        assert row["p05"] == [round(float(x), 6) for x in low]
        assert row["p50"] == [round(float(x), 6) for x in mid]
        assert row["p95"] == [round(float(x), 6) for x in high]
        assert "unit_months" not in row  # the horizon is the top-level months


def test_named_targets_are_the_forecast_of_the_run(workspace):
    request = dict(REQUEST, targets=["share_price:sber"])
    run_id, summary = start(workspace, request)
    stored = runs.result(workspace, run_id)
    assert stored["forecast"]["targets"] == ["share_price:sber"]
    assert list(stored["forecast"]["drivers"]) == ["share_price:sber"]
    assert stored["run"]["targets"] == ["share_price:sber"]


def test_a_target_the_paths_do_not_carry_is_refused(workspace):
    request = dict(REQUEST, targets=["share_price:nowhere"])
    with pytest.raises(ValueError, match="the paths carry no drivers"):
        start(workspace, request)


def test_a_request_refuses_duplicate_targets(workspace):
    request = dict(REQUEST, targets=["share_price:sber", "share_price:sber"])
    with pytest.raises(ValueError, match="share_price:sber is named twice"):
        runs.begin(workspace, runs.new_id(), request)


def test_the_forecast_refuses_duplicate_targets():
    paths = SimpleNamespace(
        drivers=SimpleNamespace(
            ids=["share_price:sber"],
            paths=np.zeros((2, 3, 1)),
            flags={"share_price:sber": frozenset()},
        ),
        months=[date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1)],
    )
    with pytest.raises(ValueError, match="share_price:sber is named twice"):
        runs.forecast(paths, ["share_price:sber", "share_price:sber"])


def test_every_position_is_the_formula_of_its_model(workspace):
    run_id, _ = start(workspace)
    meta, _, drivers, changes = arrays(workspace, run_id)
    price = driver(meta, drivers, "share_price:sber")
    spread = driver(meta, drivers, "bond_spread:corp")
    overdue = driver(meta, drivers, "overdue_share:mining")
    housing = driver(meta, drivers, "housing_price_primary:belgorod")
    expected = {
        "sber": 10.0 * (price[:, -1] / price[:, 0] - 1),
        "corp": 5.0 * (1 - 2.0 * (spread[:, -1] - spread[:, 0]) / 100 - 1),
        "mining": 3.0 * (1 - 0.45 * (overdue[:, -1] - overdue[:, 0]) - 1),
        "flat": 1.0 * (housing[:, -1] / housing[:, 0] - 1),
    }
    for index, item in enumerate(meta["position_ids"]):
        assert np.allclose(changes[:, index], expected[item], rtol=0, atol=1e-12), item


def test_the_same_request_gives_the_same_numbers(workspace):
    first_id, _ = start(workspace)
    second_id, _ = start(workspace)
    _, _, first_drivers, first = arrays(workspace, first_id)
    _, _, second_drivers, second = arrays(workspace, second_id)
    # The meta differs by id and times; the paths, the changes and the metrics do not.
    assert np.array_equal(first_drivers, second_drivers)
    assert np.array_equal(first, second)
    assert (
        runs.result(workspace, first_id)["positions"]
        == runs.result(workspace, second_id)["positions"]
    )
    other_id, _ = start(workspace, REQUEST | {"seed": 12})
    _, _, _, other = arrays(workspace, other_id)
    assert not np.array_equal(first, other)


def test_a_position_without_a_served_driver_is_invalid_in_the_files(workspace):
    positions = [dict(POSITIONS[0], codes={"secid": "NOWHERE"})]
    run_id, summary = start(workspace, REQUEST | {"positions": positions})
    assert summary["status"] == "invalid"
    stored = runs.result(workspace, run_id)
    item = stored["positions"][0]
    assert item["status"] == "invalid"
    assert [error["code"] for error in item["diagnostics"] if error["severity"] == "error"] == [
        "driver_missing:share_price"
    ]
    assert "secid NOWHERE" in item["diagnostics"][0]["message"]
    _, _, _, changes = arrays(workspace, run_id)
    assert np.all(np.isnan(changes[:, 0]))


def test_a_run_folder_is_not_overwritten(workspace):
    run_id = runs.new_id()
    runs.begin(workspace, run_id, REQUEST)
    with pytest.raises(FileExistsError):
        runs.begin(workspace, run_id, REQUEST)


def test_a_run_refuses_another_base_currency(workspace):
    with pytest.raises(ValueError, match="base currency is RUB"):
        runs.begin(workspace, runs.new_id(), REQUEST | {"base_currency": "USD"})


def test_a_run_that_cannot_start_leaves_its_reason_in_the_meta(workspace):
    run_id = runs.new_id()
    runs.begin(workspace, run_id, REQUEST | {"scenario": "nowhere"})
    with pytest.raises(FileNotFoundError, match="nowhere"):
        runs.execute(workspace, run_id)
    meta = runs.meta(workspace, run_id)
    assert meta["status"] == "failed" and "nowhere" in meta["error"]
    stored = runs.result(workspace, run_id)
    assert stored["run"]["status"] == "failed"
    assert stored["positions"] == [] and stored["summary"] is None


def test_the_catalogue_names_every_run(workspace):
    run_id, _ = start(workspace)
    found = runs.catalogue(workspace)["runs"]
    assert [item["id"] for item in found] == [run_id]
    assert found[0]["status"] == "ok" and found[0]["positions"] == len(POSITIONS)


def test_a_request_needs_a_scenario_its_draws_and_its_seed(workspace):
    for missing in ("scenario", "draws", "seed", "positions"):
        with pytest.raises(ValueError):
            runs.begin(workspace, runs.new_id(), {k: v for k, v in REQUEST.items() if k != missing})


def call(ws, name, arguments):
    """One tool call against a server built for ``ws``."""

    async def run():
        async with Client(build(ws)) as client:
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


def test_the_run_tool_starts_a_job_and_reports_its_result(workspace):
    started = call(
        workspace,
        "run",
        {"action": "start", "scenario": "oil", "draws": 3, "seed": 5, "positions": POSITIONS},
    ).data
    wait(workspace, started["job"])
    assert jobs.status(workspace, started["job"])["exit_code"] == 0
    status = call(workspace, "run", {"action": "status", "id": started["run"]}).data
    assert status["meta"]["status"] == "ok"
    assert status["job"]["exit_code"] == 0
    stored = call(workspace, "run", {"action": "result", "id": started["run"]}).data
    assert len(stored["positions"]) == len(POSITIONS)
    listed = call(workspace, "run", {"action": "list"}).data["runs"]
    assert [item["id"] for item in listed] == [started["run"]]


def test_the_run_tool_forecasts_the_targets_it_is_given(workspace):
    started = call(
        workspace,
        "run",
        {
            "action": "start",
            "scenario": "oil",
            "draws": 3,
            "seed": 5,
            "positions": POSITIONS,
            "targets": ["share_price:sber"],
        },
    ).data
    wait(workspace, started["job"])
    stored = call(workspace, "run", {"action": "result", "id": started["run"]}).data
    assert stored["run"]["targets"] == ["share_price:sber"]
    assert stored["forecast"]["targets"] == ["share_price:sber"]
    assert list(stored["forecast"]["drivers"]) == ["share_price:sber"]


def test_the_run_tool_names_the_argument_it_lacks(workspace):
    with pytest.raises(Exception, match="seed is required"):
        call(
            workspace,
            "run",
            {"action": "start", "scenario": "oil", "draws": 3, "positions": POSITIONS},
        )


def test_a_run_that_cannot_get_the_resource_is_marked_failed(workspace):
    blocker = jobs.start(workspace, "runs", ["-c", "import time; time.sleep(30)"])
    try:
        with pytest.raises(Exception, match="busy"):
            call(
                workspace,
                "run",
                {
                    "action": "start",
                    "scenario": "oil",
                    "draws": 3,
                    "seed": 5,
                    "positions": POSITIONS,
                },
            )
    finally:
        jobs.cancel(workspace, blocker)
    failed = [item for item in runs.catalogue(workspace)["runs"] if item["status"] == "failed"]
    assert len(failed) == 1
    assert "busy" in runs.meta(workspace, failed[0]["id"])["error"]
