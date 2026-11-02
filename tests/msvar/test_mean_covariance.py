"""The coefficient step under ``switching = "mean_covariance"``: common lags, two laws."""

import numpy as np
import pytest

from mst.model_sdk.contracts import ScenarioPlan
from mst.plugins.msvar import estimate, identify
from mst.plugins.msvar.forecast import simulate
from mst.plugins.msvar.signs import Signs
from mst.plugins.msvar.spec import load_spec
from tests.msvar import simulated as sim

pytestmark = pytest.mark.slow

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

FREE = [(0, 0), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
"""The entries of the lag matrix the truth sets: the global equation reads only its own
lags, so the zeros beside (0, 0) are structural, not estimated."""


@pytest.fixture(scope="module")
def posterior():
    y, _ = sim.simulate_common(300, np.random.default_rng(7))
    return estimate(sim.panel(y), sim.spec(switching="mean_covariance"), 2)


def traces(posterior, draw: int) -> np.ndarray:
    columns = (posterior.chol[draw] ** 2).sum(axis=1)
    return np.einsum("mn,mn->m", columns, np.exp(posterior.logvar[draw]))


def test_the_lag_block_is_common_to_every_regime(posterior):
    assert posterior.lags.shape == (posterior.draws, 2, 1, 3, 3)
    assert np.array_equal(posterior.lags[:, 0], posterior.lags[:, 1])


def test_the_common_lags_recover_the_truth(posterior):
    lag = posterior.lags[:, 0, 0]
    assert np.allclose(lag.mean(axis=0), sim.LAGS[0], atol=0.1)
    low, high = np.quantile(lag, [0.05, 0.95], axis=0)
    covered = [(low[i, k] <= sim.LAGS[0][i, k] <= high[i, k]) for i, k in FREE]
    assert np.mean(covered) >= 0.7


def test_the_regimes_separate_by_intercept_and_covariance(posterior):
    # The intercepts recover the truth of each regime; the stress covariance is the larger
    # one in every draw, which is what keeps the labels apart.
    assert np.allclose(posterior.const[:, 0].mean(axis=0), sim.CONST[0], atol=0.4)
    assert np.allclose(posterior.const[:, 1].mean(axis=0), sim.CONST[1], atol=0.4)
    assert not np.allclose(sim.CONST[0], sim.CONST[1])  # the truth separates the means
    for draw in range(posterior.draws):
        assert traces(posterior, draw)[0] < traces(posterior, draw)[1]
    mean = posterior.transition.mean(axis=0)
    assert np.allclose(mean, sim.TRANSITION, atol=0.12)


def test_the_shape_contract_holds_through_identification_and_simulation(posterior):
    model = identify(posterior, SIGNS, sign_lag=1, tries=200, seed=0, rho_max=1.05)
    assert np.array_equal(model.lags[:, 0], model.lags[:, 1])
    run = simulate(model, ScenarioPlan(horizon=3), draws=4, seed=1)
    assert run.paths.shape == (4, 4, 3)


def test_the_shipped_specification_switches_the_mean_and_covariance():
    assert load_spec().switching == "mean_covariance"


def test_with_all_the_existing_results_are_unchanged():
    # The default of the spec is "all" — the behaviour the versions estimated before the
    # field carried — and an explicit "all" repeats it bit for bit.
    assert sim.spec().switching == "all"
    y, _ = sim.simulate(240, np.random.default_rng(1))
    first = estimate(sim.panel(y), sim.spec(draws=8, burn=8, seed=5), 2)
    second = estimate(sim.panel(y), sim.spec(draws=8, burn=8, seed=5, switching="all"), 2)
    for name in ("const", "lags", "chol", "logvar", "transition", "states", "history"):
        assert np.array_equal(getattr(first, name), getattr(second, name))
    # The lags of the two regimes are not forced equal under "all".
    assert not np.allclose(first.lags[:, 0], first.lags[:, 1])


def test_the_prior_carries_the_dummy_rows():
    from mst.plugins.msvar.estimate import minnesota

    y, _ = sim.simulate(60, np.random.default_rng(0))
    prior = minnesota(sim.panel(y), sim.spec(mu=2.0, delta=4.0))
    average = y[0]  # p = 1, so the average of the first p months is the first month
    for i in range(len(sim.IDS)):
        assert prior.observation_y[i, i] == pytest.approx(average[i] / 2.0)
        assert np.all(np.delete(prior.observation_y[i], i) == 0.0)
        assert prior.observation_x[i, 0] == 0.0
        assert prior.observation_x[i, 1 + i] == pytest.approx(average[i] / 2.0)
    assert np.allclose(prior.observation_y[-1], average / 4.0)
    assert np.allclose(prior.observation_x[-1, 1:], np.tile(average / 4.0, 1))
    plain = minnesota(sim.panel(y), sim.spec())
    assert plain.observation_y is None and plain.observation_x is None


def test_mean_covariance_with_the_dummy_rows_pulls_the_own_lags_up():
    y, _ = sim.simulate_common(300, np.random.default_rng(7))
    plain = estimate(sim.panel(y), sim.spec(switching="mean_covariance"), 2)
    pulled = estimate(
        sim.panel(y), sim.spec(switching="mean_covariance", mu=1.0, delta=1.0), 2
    )
    # The shape contract holds with the dummy rows in, and the own-lag medians rise toward
    # one, as the rows pull: the stationary prior pulls them to nought.
    assert np.array_equal(pulled.lags[:, 0], pulled.lags[:, 1])
    for i in range(3):
        plain_own = np.median(plain.lags[:, 0, 0, i, i])
        pulled_own = np.median(pulled.lags[:, 0, 0, i, i])
        assert pulled_own >= plain_own
    # The regimes still separate and the transition is still recovered.
    assert np.allclose(pulled.transition.mean(axis=0), sim.TRANSITION, atol=0.12)


def test_all_with_the_dummy_rows_estimates():
    y, _ = sim.simulate(240, np.random.default_rng(1))
    found = estimate(sim.panel(y), sim.spec(switching="all", mu=1.0, delta=1.0), 2)
    assert found.lags.shape[0] == sim.spec().sampler.draws
    assert not np.allclose(found.lags[:, 0], found.lags[:, 1])


def test_the_initial_observation_row_reads_the_intercept_of_every_regime():
    from dataclasses import replace

    from mst.plugins.msvar.estimate import _mean_covariance, minnesota, regressors

    # A stack of regimes whose intercepts differ by a hundred: with no months in either regime
    # the prior rows alone speak to the intercepts, and each regime's copy of the
    # initial-observation row pulls its own back to what the lag block leaves of the level.
    y, _ = sim.simulate_common(60, np.random.default_rng(3))
    panel = sim.panel(y)
    prior = minnesota(panel, sim.spec(mu=None, delta=0.01))
    prior = replace(prior, observation_y=prior.observation_y, observation_x=prior.observation_x)
    x, target = regressors(y, 1)[:0], y[1:][:0]
    n, k = 3, prior.mean.shape[1]
    start = [np.zeros((n, k)), np.zeros((n, k))]
    start[0][:, 0], start[1][:, 0] = 0.0, 100.0
    logvar = [np.zeros(n), np.zeros(n)]
    found = _mean_covariance(
        target,
        x,
        np.zeros(0, dtype=int),
        start,
        [np.eye(n), np.eye(n)],
        logvar,
        prior,
        np.random.default_rng(4),
    )
    # The row reads c_j + sum(A) ybar = ybar, tight at this delta, so both intercepts come out
    # near the same value whatever they started from, and the lag block stays common.
    assert np.array_equal(found[0][:, 1:], found[1][:, 1:])
    assert np.allclose(found[0][:, 0], found[1][:, 0], atol=1.0)


def test_a_regime_the_path_left_empty_keeps_its_row_of_transitions():
    from mst.plugins.msvar.estimate import _transition

    states = np.zeros(30, dtype=int)  # the top regime holds no month at all
    found = _transition(states, 2, sim.spec(), np.random.default_rng(0))
    assert found.shape == (2, 2)
    assert np.allclose(found.sum(axis=1), 1.0)
