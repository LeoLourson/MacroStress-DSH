"""Posterior intervals cover the parameters of a known model."""

import numpy as np
import pytest

from mst.plugins.bvar.estimate import estimate, impute
from tests.bvar import simulated as sim

pytestmark = pytest.mark.slow


def covered(draws: np.ndarray, truth: np.ndarray) -> np.ndarray:
    low, high = np.quantile(draws, [0.05, 0.95], axis=0)
    return (low <= truth) & (truth <= high)


@pytest.mark.parametrize("hidden", [0.0, 0.2])
def test_constant_volatility_intervals_cover_the_truth_nine_times_in_ten(hidden):
    rng = np.random.default_rng(1)
    hits = []
    for _ in range(25):
        y = sim.simulate(240, rng)
        y[rng.random(y.shape) < hidden] = np.nan
        y[0] = sim.simulate(1, rng)[0]  # some month must be known in every column
        posterior = estimate(
            sim.panel(y),
            sim.spec(draws=300, burn=50, thin=1, seed=int(rng.integers(1e6))),
        )
        free = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (0, 0)]
        lag = posterior.lags[:, 0]
        hits += [covered(lag[:, i, k], sim.LAG[i, k]) for i, k in free]
        hits += list(covered(posterior.const, sim.CONST))
        sigma = np.einsum(
            "sij,sj,skj->sik", posterior.chol, np.exp(posterior.logvar[:, -1]), posterior.chol
        )
        truth = sim.CHOL @ np.diag(np.exp(sim.LOGVAR)) @ sim.CHOL.T
        hits += list(covered(sigma, truth)[np.tril_indices(3)])
    assert 0.84 <= np.mean(hits) <= 0.96


def test_russian_lags_never_enter_global_equations():
    y = sim.simulate(120, np.random.default_rng(2))
    for volatility in ("constant", "stochastic"):
        posterior = estimate(
            sim.panel(y),
            sim.spec(volatility=volatility, draws=20, burn=5, thin=1, seed=3),
        )
        assert np.all(posterior.lags[:, :, :1, 1:] == 0)
        assert np.all(posterior.chol[:, :1, 1:] == 0)
        assert np.all(np.triu(posterior.chol, 1) == 0)


def test_same_seed_gives_the_same_draws_bit_for_bit():
    y = sim.simulate(100, np.random.default_rng(4))
    y[50:60, 2] = np.nan
    first = estimate(sim.panel(y), sim.spec(draws=30, burn=5, thin=2, seed=9))
    second = estimate(sim.panel(y), sim.spec(draws=30, burn=5, thin=2, seed=9))
    assert np.array_equal(first.lags, second.lags)
    assert np.array_equal(first.history, second.history)


def test_imputed_months_are_drawn_from_their_conditional_distribution():
    rng = np.random.default_rng(5)
    y = sim.simulate(150, rng)
    missing = np.zeros(y.shape, dtype=bool)
    missing[:12, 2] = True  # the series starts late
    missing[20:140:3, 1] = True  # observed only in some months
    missing[-2:, 0] = True  # ragged end
    hidden = np.where(missing, np.nan, y)
    coefficients = np.hstack([sim.CONST[:, None], sim.LAG])
    sigma = sim.CHOL @ np.diag(np.exp(sim.LOGVAR)) @ sim.CHOL.T
    precisions = [np.linalg.inv(sigma)] * (len(y) - 1)
    draws = np.array(
        [
            impute(hidden, missing, coefficients, precisions, np.zeros((len(y), 0)), 1, rng)[
                missing
            ]
            for _ in range(400)
        ]
    )
    assert 0.85 <= covered(draws, y[missing]).mean() <= 0.95
    assert np.array_equal(draws[0][..., None].shape, (missing.sum(), 1))


def test_stochastic_volatility_tracks_a_known_volatility_path():
    rng = np.random.default_rng(6)
    count = 400
    logvar = np.tile(sim.LOGVAR, (count, 1))
    logvar += np.cumsum(0.15 * rng.standard_normal((count, 3)), axis=0)
    y = sim.simulate(count, rng, logvar=logvar, dof=8.0)
    posterior = estimate(
        sim.panel(y),
        sim.spec(volatility="stochastic", draws=400, burn=400, thin=1, seed=11),
    )
    assert covered(posterior.logvar, logvar[1:]).mean() >= 0.8
    lag = posterior.lags[:, 0]
    assert covered(lag, sim.LAG)[1:].mean() >= 0.6
    assert np.all(posterior.volvol > 0)
    assert np.isfinite(posterior.dof).all()
