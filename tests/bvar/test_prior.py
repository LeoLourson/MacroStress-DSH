"""The dummy rows of Sims-Zha and Giannone-Lenza-Primiceri in the priors of both models."""

import numpy as np
import pytest

from mst.plugins.bvar.estimate import conjugate_step, dummy_matrix, fill, minnesota
from tests.bvar import simulated as sim

FAST = 120


def test_the_dummy_rows_are_the_sims_zha_ones():
    y = sim.simulate(60, np.random.default_rng(0))
    panel = sim.panel(y)
    prior = minnesota(panel, sim.spec(mu=2.0, delta=4.0))
    # p = 1, so the average of the first p observations is the first month.
    average = y[0]
    assert prior.observation_y.shape == (len(sim.IDS) + 1, len(sim.IDS))
    # The sum-of-coefficients rows: the own-lag regressor and the own response carry the
    # average over mu, everything else is nought.
    for i in range(len(sim.IDS)):
        assert prior.observation_y[i, i] == pytest.approx(average[i] / 2.0)
        assert np.all(np.delete(prior.observation_y[i], i) == 0.0)
        assert prior.observation_x[i, 0] == 0.0  # the constant stays free
        assert prior.observation_x[i, 1 + i] == pytest.approx(average[i] / 2.0)
        assert np.all(np.delete(prior.observation_x[i, 1:], i) == 0.0)
    # The initial-observation row: every lag column and every response carries the average
    # over delta, and the constant column one over delta.
    assert np.allclose(prior.observation_y[-1], average / 4.0)
    assert prior.observation_x[-1, 0] == pytest.approx(1 / 4.0)
    assert np.allclose(prior.observation_x[-1, 1:], np.tile(average / 4.0, 1))


def test_without_mu_and_delta_there_are_no_rows():
    y = sim.simulate(60, np.random.default_rng(0))
    prior = minnesota(sim.panel(y), sim.spec())
    assert prior.observation_y is None
    assert prior.observation_x is None
    # One hyperparameter alone brings its own rows alone.
    only_mu = minnesota(sim.panel(y), sim.spec(mu=1.0))
    assert only_mu.observation_y.shape == (len(sim.IDS), len(sim.IDS))
    only_delta = minnesota(sim.panel(y), sim.spec(delta=1.0))
    assert only_delta.observation_y.shape == (1, len(sim.IDS))


def test_a_late_series_reads_its_first_observed_value():
    y = sim.simulate(60, np.random.default_rng(0))
    holed = y.copy()
    holed[:12, 2] = np.nan  # the series starts after the first p months
    prior = minnesota(sim.panel(holed), sim.spec(mu=1.0, delta=None))
    assert prior.observation_y[2, 2] == pytest.approx(holed[12, 2])


def test_the_dummy_augmented_regression_equals_the_closed_form():
    # The global block of the conjugate posterior is a plain conjugate regression, so its
    # posterior mean has the closed form of the dummy-augmented data — the draws of
    # conjugate_step average to it.
    y = sim.simulate(80, np.random.default_rng(1))
    panel = sim.panel(y)
    spec = sim.spec(mu=1.0, delta=1.0)
    prior = minnesota(panel, spec)
    filled = fill(y)
    dummies = dummy_matrix(panel.months, spec.dummies)
    from mst.plugins.bvar.var import regressors

    x = regressors(filled, dummies, spec.lags)
    target = filled[spec.lags :]
    columns = prior.mask[0]
    draws = np.array(
        [
            conjugate_step(filled, dummies, prior, spec.lags, panel.n_global, rng)[0]
            for rng in (np.random.default_rng(k) for k in range(400))
        ]
    )
    found = draws[:, :1, :][:, :, columns].mean(axis=0)
    error = draws[:, :1, :][:, :, columns].std(axis=0) / np.sqrt(len(draws))
    # The closed form: the augmented regression on top of the conjugate prior.
    star_x = np.vstack([x[:, columns], prior.observation_x[:, columns]])
    star_y = np.vstack([target[:, :1], prior.observation_y[:, :1]])
    relative = prior.var / prior.scale[:, None] ** 2
    prec = 1 / relative[0, columns]
    closed = np.linalg.solve(
        star_x.T @ star_x + np.diag(prec),
        prec[:, None] * prior.mean[:1, columns].T + star_x.T @ star_y,
    ).T
    assert np.all(np.abs(found - closed) < 5 * error + 1e-9)


@pytest.mark.slow
def test_the_dummies_pull_the_own_lag_sums_toward_one():
    from mst.plugins.bvar.estimate import estimate

    rng = np.random.default_rng(2)
    y = sim.simulate(240, rng)
    plain = estimate(sim.panel(y), sim.spec(draws=200, burn=50))
    pulled = estimate(sim.panel(y), sim.spec(draws=200, burn=50, mu=1.0, delta=1.0))
    own = np.array([np.median(d.lags[:, 0, i, i]) for d in (plain, pulled) for i in range(3)])
    plain_own, pulled_own = own[:3], own[3:]
    # The stationary prior pulls the own lags to nought; the dummy rows pull them toward
    # one, so every own-lag median rises.
    assert np.all(pulled_own > plain_own)
    from mst.model_sdk.stability import companion_radius

    assert np.median(companion_radius(pulled.lags)) > np.median(companion_radius(plain.lags))
