"""The model tool: versions of a workspace are listed and described."""

import asyncio
from datetime import date

import numpy as np
from fastmcp import Client

from mst import jobs, store
from mst.macro import version
from mst.macro.estimate import estimate
from mst.macro.identify import identify
from mst.macro.registry import Shock
from mst.mcp import build

from . import simulated as sim


def call(ws, name, arguments):
    async def run():
        async with Client(build(ws)) as client:
            return await client.call_tool(name, arguments)

    return asyncio.run(run())


def saved_version(tmp_path):
    """A tiny identified version under ``<tmp_path>/models``."""
    y = sim.simulate(80, np.random.default_rng(0))
    panel = sim.panel(y)
    spec = sim.spec(draws=10)
    shocks = [Shock(id="ru.t", label="t", rises=["ru.b"])]
    model = identify(estimate(panel, spec), shocks, sign_lag=0, tries=100, seed=1)
    return version.save(model, spec, panel, date(2010, 1, 1), tmp_path / "models")


def test_a_saved_version_is_listed_and_described(tmp_path):
    ws = store.Workspace(tmp_path)
    name = saved_version(tmp_path)

    listed = call(ws, "model", {"kind": "macro", "action": "list"}).data["versions"]
    assert [row["version"] for row in listed] == [name]
    assert listed[0]["as_of"] == "2010-01-01"

    described = call(ws, "model", {"kind": "macro", "action": "describe", "version": name}).data
    assert described["as_of"] == "2010-01-01"
    assert "ru.t" in described["shocks"]
    assert 0 < described["identification_share"] <= 1


def test_estimate_and_backtest_become_tasks_on_the_models_resource(tmp_path):
    ws = store.Workspace(tmp_path)
    for action in ("estimate", "backtest"):
        answer = call(ws, "model", {"kind": "macro", "action": action, "as_of": "2010-01-01"}).data
        try:
            assert answer["argv"][:3] == ["-m", "mst.macro", action]
            assert jobs.status(ws, answer["job"])["resource"] == "models"
        finally:
            jobs.cancel(ws, answer["job"])


def test_another_kind_is_refused(tmp_path):
    ws = store.Workspace(tmp_path)
    try:
        call(ws, "model", {"kind": "satellites", "action": "list"})
    except Exception as error:
        assert "unknown kind" in str(error)
    else:
        raise AssertionError("a kind with no tool yet must not be accepted")
