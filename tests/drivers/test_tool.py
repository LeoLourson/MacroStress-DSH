"""The satellites kind of the model tool, and the driver catalog of ``series update``."""

import asyncio
from datetime import date

from fastmcp import Client

from mst import jobs, store
from mst.drivers import bundle as satellite_bundle
from mst.mcp import build as build_server
from mst.model_sdk.contracts import SatelliteEstimateContext
from mst.plugins.hier.model import HierModel
from mst.plugins.hier.spec import Sampler, Spec

from . import simulated

AS_OF = date(2030, 1, 1)


def call(ws, name, arguments):
    async def run():
        async with Client(build_server(ws)) as client:
            return await client.call_tool(name, arguments)

    return asyncio.run(run())


def small_spec() -> Spec:
    """The estimation of ``spec.toml`` with a sampler small enough for a test."""
    return Spec(
        spread=0.1,
        constant_spread=10.0,
        rho_max=0.99,
        holdout=24,
        reach=12,
        sampler=Sampler(draws=10, burn=5, seed=7),
    )


def saved_version(tmp_path):
    """A tiny satellite bundle under ``<tmp_path>/models``."""
    panel = simulated.panel(240)
    series = simulated.store(tmp_path / "data", panel)
    registry = simulated.registry()
    model = HierModel(spec=small_spec())
    estimate = model.estimate(
        SatelliteEstimateContext(
            store=series,
            registry=registry,
            panel=panel,
            as_of=AS_OF,
            macro="macro-version",
            quantities=("price",),
        )
    )
    meta, arrays = satellite_bundle.assemble(
        registry, "macro-version", AS_OF, {model.model_id: estimate}
    )
    name = satellite_bundle.save(meta, arrays, tmp_path / "models")
    return name, satellite_bundle.load(tmp_path / "models", name, {model.model_id: model})


def test_a_saved_satellite_version_is_listed_and_described(tmp_path):
    ws = store.Workspace(tmp_path)
    name, bundle = saved_version(tmp_path)

    listed = call(ws, "model", {"kind": "satellites", "action": "list"}).data["versions"]
    assert [row["version"] for row in listed] == [name]
    assert listed[0]["macro"] == "macro-version"
    assert listed[0]["models"] == ["hier.gibbs"]
    assert listed[0]["checked"] == sum(len(item) for item in bundle.checks.values())

    described = call(
        ws, "model", {"kind": "satellites", "action": "describe", "version": name}
    ).data
    assert described["macro"] == "macro-version"
    assert described["models"] == {"price": "hier.gibbs"}
    assert described["served"] == bundle.served()
    assert {item["driver"] for item in described["rejected"]} == (
        {d for d in bundle.parts["price"]["driver"] if d} - set(bundle.served())
    )


def test_estimate_becomes_a_task_on_the_models_resource(tmp_path):
    ws = store.Workspace(tmp_path)
    answer = call(
        ws, "model", {"kind": "satellites", "action": "estimate", "version": "macro-sha"}
    ).data
    try:
        assert answer["argv"][:3] == ["-m", "mst.models", "estimate-satellites"]
        assert answer["argv"][answer["argv"].index("--macro") + 1] == "macro-sha"
        assert jobs.status(ws, answer["job"])["resource"] == "models"
    finally:
        jobs.cancel(ws, answer["job"])


def test_an_assignment_names_the_model_of_a_quantity(tmp_path):
    ws = store.Workspace(tmp_path)
    answer = call(
        ws,
        "model",
        {
            "kind": "satellites",
            "action": "estimate",
            "version": "macro-sha",
            "assignment": {"share_price": "other.v1"},
        },
    ).data
    try:
        argv = answer["argv"]
        assert argv[:3] == ["-m", "mst.models", "estimate-satellites"]
        assert argv[argv.index("--model") + 1] == "share_price=other.v1"
    finally:
        jobs.cancel(ws, answer["job"])


def test_describe_without_a_version_is_refused(tmp_path):
    ws = store.Workspace(tmp_path)
    try:
        call(ws, "model", {"kind": "satellites", "action": "describe"})
    except Exception as error:
        assert "version is required" in str(error)
    else:
        raise AssertionError("describe without a version must not be accepted")


def test_series_update_with_a_quantity_runs_the_driver_catalog(tmp_path):
    ws = store.Workspace(tmp_path)
    answer = call(ws, "series", {"action": "update", "quantity": "overdue_share"}).data
    try:
        assert answer["argv"][:4] == ["-m", "mst.drivers", "update", "overdue_share"]
        assert jobs.status(ws, answer["job"])["resource"] == "series"
    finally:
        jobs.cancel(ws, answer["job"])


def test_ids_and_a_quantity_together_are_refused(tmp_path):
    ws = store.Workspace(tmp_path)
    try:
        call(ws, "series", {"action": "update", "ids": ["gl.vix"], "quantity": "overdue_share"})
    except Exception as error:
        assert "not both" in str(error)
    else:
        raise AssertionError("a series update with ids and a quantity must not be accepted")
