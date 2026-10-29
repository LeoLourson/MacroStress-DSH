import pytest

from tests.scenario import simulated


@pytest.fixture(scope="session")
def versions(tmp_path_factory):
    """One macro version and its satellites for every test of the scenario layer."""
    return simulated.build(tmp_path_factory.mktemp("models"), tmp_path_factory.mktemp("series"))
