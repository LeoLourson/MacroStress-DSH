"""The stability filter of the MS-VAR: the posterior restricted to stable draws."""

from dataclasses import replace

import numpy as np
import pytest

from mst.model_sdk.stability import companion_radius
from mst.plugins.msvar import estimate, identify
from mst.plugins.msvar.forecast import simulate
from mst.plugins.msvar.identify import stable
from tests.msvar import simulated as sim

pytestmark = pytest.mark.slow

SIGNS = {
    "shocks": [
        {"id": "gl.s", "label": "g", "rises": ["gl.a"]},
        {"id": "ru.t", "label": "t", "rises": ["ru.b", "ru.c"]},
    ],
    "regimes": [
        {"id": "regime.calm", "label": "спокойный"},
        {"id": "regime.stress", "label": "стрессовый"},
    ],
}


@pytest.fixture(scope="module")
def posterior():
    y, _ = sim.simulate(240, np.random.default_rng(0))
    return estimate(sim.panel(y), sim.spec(), 2)


def test_an_inserted_explosive_draw_is_lost_and_only_it(posterior):
    # The calm regime of every draw stays as it was; the stress regime of one draw is made
    # explosive by hand, and the filter loses exactly that draw.
    lags = posterior.lags.copy()
    at = 2
    lags[at, 1, 0] = 5.0 * np.eye(3)
    mixed = replace(posterior, lags=lags)
    radius = companion_radius(posterior.lags)
    limit = float(radius.max() + 1e-9)
    kept = stable(mixed, limit)
    assert kept.draws == posterior.draws - 1
    assert np.array_equal(kept.lags, np.delete(posterior.lags, at, axis=0))
    assert np.array_equal(kept.const, np.delete(posterior.const, at, axis=0))
    assert np.array_equal(kept.states, np.delete(posterior.states, at, axis=0))
    assert kept.diagnostics["stable"] == posterior.draws - 1


def test_a_kept_draw_never_has_a_radius_above_the_limit(posterior):
    kept = stable(posterior, 1.05)
    radius = companion_radius(kept.lags)
    assert np.all(radius <= 1.05)  # every regime of every kept draw
    expected = np.flatnonzero((companion_radius(posterior.lags) <= 1.05).all(axis=1))
    assert kept.draws == len(expected)
    assert np.array_equal(kept.lags, posterior.lags[expected])


def test_a_regime_explosive_in_one_draw_alone_sinks_that_draw(posterior):
    # The stress regime of the true model is the less persistent one, but still stable; a
    # ceiling the stress regime of every draw breaks leaves nothing, and the error names
    # the model and the limit.
    explosive = replace(
        posterior,
        lags=np.tile(
            np.stack([np.zeros((1, 3, 3)), 2.0 * np.eye(3)[None]]),
            (posterior.draws, 1, 1, 1, 1),
        ),
    )
    with pytest.raises(
        ValueError,
        match=r"no draw of msvar\.markov keeps the companion radius of every regime "
        r"at or below rho_max 1\.005",
    ):
        stable(explosive, 1.005)


def test_identify_filters_before_the_rotation_search(posterior):
    from mst.plugins.msvar.signs import Signs

    signs = Signs.model_validate(SIGNS)
    model = identify(posterior, signs, sign_lag=1, tries=200, seed=0, rho_max=1.05)
    assert np.all(companion_radius(model.lags) <= 1.05)
    assert model.diagnostics["estimated"] == posterior.draws
    assert model.diagnostics["stable"] <= posterior.draws
    assert model.diagnostics["identified"] <= model.diagnostics["stable"]


def test_a_filtered_model_still_simulates(posterior):
    # The filter keeps the shapes the forecast reads: (draws, regimes, p, n, n).
    from mst.model_sdk.contracts import ScenarioPlan
    from mst.plugins.msvar.signs import Signs

    signs = Signs.model_validate(SIGNS)
    model = identify(posterior, signs, sign_lag=1, tries=200, seed=0, rho_max=1.05)
    run = simulate(model, ScenarioPlan(horizon=3), draws=4, seed=1)
    assert run.paths.shape == (4, 4, 3)
