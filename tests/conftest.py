"""Fixtures shared by the tests of the calculation layers."""

import pytest

from tests.engine import simulated


@pytest.fixture(scope="session")
def versions(tmp_path_factory):
    """One macro version and its satellites for every test that runs positions on paths."""
    return simulated.build(tmp_path_factory.mktemp("models"), tmp_path_factory.mktemp("series"))
