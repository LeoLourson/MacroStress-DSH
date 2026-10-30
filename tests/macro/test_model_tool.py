"""The model tool: versions of a workspace are listed and described."""

import asyncio
from datetime import date

import numpy as np
from fastmcp import Client

from mst import jobs, store
from mst.macro.version import load, save
from mst.mcp import build
from mst.plugins.bvar import MODEL
from mst.plugins.bvar.estimate import estimate
from mst.plugins.bvar.identify import identify
from mst.plugins.bvar.model import parts
from mst.plugins.bvar.signs import Shock
from tests.bvar import simulated as sim


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
    estimate_ = parts(model, spec, panel, date(2010, 1, 1))
    return save("bvar.sv", estimate_.meta, estimate_.arrays, tmp_path / "models")


def test_a_saved_version_is_listed_and_described(tmp_path):
    ws = store.Workspace(tmp_path)
    name = saved_version(tmp_path)

    listed = call(ws, "model", {"kind": "macro", "action": "list"}).data["versions"]
    assert [row["version"] for row in listed] == [name]
    assert listed[0]["as_of"] == "2010-01-01"
    assert listed[0]["model"] == "bvar.sv"

    described = call(ws, "model", {"kind": "macro", "action": "describe", "version": name}).data
    assert described["as_of"] == "2010-01-01"
    assert described["model"] == "bvar.sv"
    assert "ru.t" in described["shocks"]
    assert 0 < described["identification_share"] <= 1


def test_estimate_and_backtest_become_tasks_on_the_models_resource(tmp_path):
    ws = store.Workspace(tmp_path)
    answer = call(ws, "model", {"kind": "macro", "action": "estimate", "as_of": "2010-01-01"}).data
    try:
        assert answer["argv"][:4] == ["-m", "mst.models", "estimate-macro", "bvar.sv"]
        assert jobs.status(ws, answer["job"])["resource"] == "models"
    finally:
        jobs.cancel(ws, answer["job"])

    answer = call(ws, "model", {"kind": "macro", "action": "backtest", "as_of": "2010-01-01"}).data
    try:
        assert answer["argv"][:4] == ["-m", "mst.metrics", "backtest", "bvar.sv"]
        assert "--corpus" in answer["argv"] and "--drivers" in answer["argv"]
        assert jobs.status(ws, answer["job"])["resource"] == "models"
    finally:
        jobs.cancel(ws, answer["job"])


def test_an_unknown_model_id_is_refused(tmp_path):
    ws = store.Workspace(tmp_path)
    try:
        call(
            ws,
            "model",
            {"kind": "macro", "action": "estimate", "as_of": "2010-01-01", "model_id": "nope"},
        )
    except Exception as error:  # fastmcp raises ToolError with the message of the tool
        assert "no installed macro model 'nope'" in str(error)
    else:
        raise AssertionError("an unknown model id must not be accepted")


def test_the_catalog_lists_the_model_plugins(tmp_path):
    ws = store.Workspace(tmp_path)
    answer = call(ws, "model", {"kind": "catalog", "action": "list"}).data
    assert "bvar" in [row["plugin"] for row in answer["plugins"]]
    bvar = next(row for row in answer["plugins"] if row["plugin"] == "bvar")
    assert [model["model_id"] for model in bvar["macro"]] == ["bvar.sv"]
    assert answer["macro"] == ["bvar.sv"]


def test_a_loaded_version_comes_from_its_plugin(tmp_path):
    ws = store.Workspace(tmp_path)
    name = saved_version(tmp_path)
    from mst import plugins

    version = load(ws.models, name, plugins.registry().macro)
    assert version.plugin is MODEL
    assert version.as_of == date(2010, 1, 1)


def test_another_kind_is_refused(tmp_path):
    ws = store.Workspace(tmp_path)
    try:
        call(ws, "model", {"kind": "corpus", "action": "list"})
    except Exception as error:  # fastmcp raises ToolError with the message of the tool
        assert "unknown kind" in str(error)
    else:
        raise AssertionError("a kind with no tool yet must not be accepted")
