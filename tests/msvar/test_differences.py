"""Variables whose equation runs on monthly changes: the version stores the levels form."""

from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest

from mst.model_sdk.stability import companion_radius
from mst.plugins.msvar import estimate, identify
from mst.plugins.msvar.model import MSVarModel, parts
from mst.plugins.msvar.signs import Signs
from tests.msvar import simulated as sim

SIGNS = Signs.model_validate(
    {
        "shocks": [
            {"id": "gl.s", "label": "g", "rises": ["gl.a"]},
            {"id": "ru.t", "label": "t", "rises": ["ru.b", "ru.c"]},
        ],
        "regimes": [
            {"id": "regime.calm", "label": "спокойный"},
            {"id": "regime.stress", "label": "стрессовый"},
        ],
    }
)


def _walk(count: int, rng) -> np.ndarray:
    """gl.a and ru.c from the simulated MS-VAR; ru.b replaced by a random walk with drift
    0.4 whose change loads on gl.a."""
    y, _ = sim.simulate_common(count, rng)
    steps = 0.4 + 0.5 * (y[:, 0] - y[:, 0].mean()) + rng.normal(size=count)
    y[:, 1] = 100.0 + np.cumsum(steps)
    return y


def _spec(**changes):
    values = {
        "stationary": ["gl.a", "ru.c"],
        "differences": ["ru.b"],
        "switching": "mean_covariance",
        "draws": 20,
        "burn": 20,
    }
    values.update(changes)
    return sim.spec(**values)


@pytest.mark.parametrize("switching", ["all", "mean_covariance"])
def test_a_differenced_variable_keeps_an_exact_unit_root(switching):
    y = _walk(160, np.random.default_rng(2))
    y[60:63, 1] = np.nan  # an unknown stretch is still drawn in levels
    posterior = estimate(sim.panel(y), _spec(switching=switching), 2)
    # One lag on changes is two in levels, and the history holds as many months.
    assert posterior.lags_count == 2
    assert posterior.lags.shape[2:] == (2, 3, 3)
    assert posterior.history.shape[1:] == (2, 3)
    assert posterior.states.shape[1] == len(y) - 2
    # The level of ru.b carries its previous level one for one in every regime: the lag sum
    # on itself is one exactly and on every other variable nought.
    sums = posterior.lags.sum(axis=2)
    assert np.allclose(sums[:, :, 1, 1], 1.0)
    assert np.allclose(sums[:, :, 0, 1], 0.0) and np.allclose(sums[:, :, 2, 1], 0.0)
    assert np.all(companion_radius(posterior.lags) >= 1.0 - 1e-9)
    assert np.all(np.isfinite(posterior.history))


@pytest.mark.parametrize("rows", [{"mu": 1.0}, {"delta": 1.0}])
def test_the_dummy_rows_written_for_levels_are_refused_with_differences(rows):
    y = _walk(80, np.random.default_rng(3))
    with pytest.raises(ValueError, match="dummy rows"):
        estimate(sim.panel(y), _spec(**rows), 2)


def test_a_differenced_version_round_trips_through_parts_and_load():
    y = _walk(160, np.random.default_rng(4))
    spec = _spec(draws=6, burn=6)
    identified = identify(estimate(sim.panel(y), spec, 2), SIGNS, sign_lag=2, tries=2000, seed=3)
    version = parts(identified, spec, sim.panel(y), date(2013, 4, 1))
    loaded = MSVarModel().load(version.meta, version.arrays)
    assert loaded.lags_count == 2
    assert np.array_equal(loaded.lags, identified.lags)


def test_a_variable_the_platform_left_out_may_stay_in_the_differences(monkeypatch):
    """A window whose history is too short for ru.x has no ru.x in its panel: the spec may
    still difference it, the version stores the differences it was estimated with, and a
    name neither in the panel nor left out is a typo."""
    from mst.plugins.msvar import model as msvar_model

    y = _walk(120, np.random.default_rng(5))
    spec = _spec(differences=["ru.b", "ru.x"], draws=4, burn=4)
    monkeypatch.setattr(msvar_model, "load_spec", lambda: spec)
    monkeypatch.setattr(msvar_model, "load_signs", lambda: SIGNS)
    monkeypatch.setattr(msvar_model, "build_panel", lambda *args, **kwargs: sim.panel(y))

    def context(left_out):
        return SimpleNamespace(
            store=None,
            registry=SimpleNamespace(variables=[]),
            as_of=date(2009, 12, 1),
            left_out=left_out,
        )

    version = MSVarModel().estimate(context({"ru.x": "too short"}))
    assert version.meta["spec"]["differences"] == ["ru.b"]
    with pytest.raises(ValueError, match="unknown variables: ru.x"):
        MSVarModel().estimate(context({}))
