"""Fixtures shared by the tests of the calculation layers."""

import shutil

import pytest

from mst import store
from mst.scenario import tool as scenario_tool
from tests.engine import simulated


@pytest.fixture(scope="session")
def versions(tmp_path_factory):
    """One macro version and its satellites for every test that runs positions on paths."""
    return simulated.build(tmp_path_factory.mktemp("models"), tmp_path_factory.mktemp("series"))


@pytest.fixture
def workspace(tmp_path, versions):
    """A workspace with the two versions of the session and one scenario."""
    ws = store.Workspace(tmp_path)
    for kind, name in (("macro", versions.macro), ("satellites", versions.satellites)):
        shutil.copytree(versions.models / kind / name, ws.models / kind / name)
    scenario_tool.save(
        ws.scenarios,
        "oil",
        {
            "narrative": "нефть",
            "horizon": 3,
            "macro": versions.macro,
            "satellites": versions.satellites,
            "approved_by": "И. Иванов",
        },
    )
    return ws
