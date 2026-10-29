"""mst.macro.identify: sign restrictions by uniform rotations."""

from dataclasses import replace

import numpy as np
import pytest
from scipy import stats

from mst.macro.estimate import estimate
from mst.macro.identify import identify, impact, rotate
from mst.macro.registry import Shock
from mst.macro.var import responses

from . import simulated as sim

pytestmark = pytest.mark.slow


def shock(id_, rises=(), falls=()):
    return Shock(id=id_, label=id_, rises=list(rises), falls=list(falls))


@pytest.fixture(scope="module")
def posterior():
    y = sim.simulate(200, np.random.default_rng(0))
    return estimate(sim.panel(y), sim.spec(draws=60))


def test_every_kept_draw_moves_the_restricted_variables_the_required_way(posterior):
    shocks = [shock("gl.s", rises=["gl.a"], falls=["ru.c"]), shock("ru.t", rises=["ru.b", "ru.c"])]
    model = identify(posterior, shocks, sign_lag=2, tries=5000, seed=1)

    assert model.shocks == ["gl.s", "ru.t", "ru.unnamed_1"]
    assert model.diagnostics["identified"] > 0
    for draw in range(model.draws):
        b = impact(model, draw) @ model.rotation[draw]
        moved = responses(model.lags[draw], 3) @ b
        assert np.all(moved[:, 0, 0] > 0) and np.all(moved[:, 2, 0] < 0)
        assert np.all(moved[:, 1, 1] > 0) and np.all(moved[:, 2, 1] > 0)
        assert np.all(b[0, 1:] == 0)  # Russian shocks leave the global variable alone
        assert np.allclose(model.rotation[draw] @ model.rotation[draw].T, np.eye(3))


def test_admissible_rotations_are_uniform():
    # Uncorrelated unit errors and no lags: the impact of the shock is its rotation column,
    # which must be uniform on the quarter circle where both variables rise.
    rng = np.random.default_rng(0)
    angles = []
    for _ in range(2000):
        q = rotate(np.eye(2)[None], np.array([[1.0, 1.0]]), 100, rng)
        angles.append(np.arctan2(q[1, 0], q[0, 0]))
    assert stats.kstest(angles, stats.uniform(0, np.pi / 2).cdf).pvalue > 0.01


def test_draws_without_an_admissible_rotation_are_dropped(posterior):
    # A lag of -1 flips every response in the next month, so no shock raises b two months.
    flipping = replace(posterior, lags=-np.eye(3)[None, None].repeat(posterior.draws, axis=0))
    with pytest.raises(ValueError, match="no draw admits"):
        identify(flipping, [shock("ru.t", rises=["ru.b"])], sign_lag=1, tries=200, seed=0)
