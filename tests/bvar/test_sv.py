"""The Gibbs steps of stochastic volatility with Student tails."""

import numpy as np
import pytest

from mst.plugins.bvar import sv


def test_mixture_has_the_mean_and_variance_of_log_chi_square_one():
    mean = sv.MIX_WEIGHT @ sv.MIX_MEAN
    second = sv.MIX_WEIGHT @ (sv.MIX_VAR + sv.MIX_MEAN**2)
    # E log chi2_1 = -(Euler gamma) - ln 2, Var = pi^2 / 2.
    assert sv.MIX_WEIGHT.sum() == pytest.approx(1.0, abs=1e-4)
    assert mean == pytest.approx(-np.euler_gamma - np.log(2), abs=1e-3)
    assert second - mean**2 == pytest.approx(np.pi**2 / 2, abs=1e-2)


def test_degrees_of_freedom_are_recovered_from_mixing_variables():
    rng = np.random.default_rng(0)
    omega = 4.0 / rng.gamma(4.0, size=5000)  # InvGamma(nu/2, nu/2) with nu = 8
    draws = [sv.draw_dof(omega, rng) for _ in range(200)]
    assert 7 <= np.median(draws) <= 9


def test_the_grid_starts_at_its_floor_and_draws_never_fall_below():
    rng = np.random.default_rng(3)
    grid = sv.dof_grid(5.0)
    assert grid[0] == pytest.approx(5.0)
    assert grid[-1] == pytest.approx(sv.DOF_TOP)
    # Mixing variables of a truth at 3 — the heaviest the old grid allowed — pull the
    # posterior to the floor of the shipped grid, never through it.
    omega = 1.5 / rng.gamma(1.5, size=5000)
    draws = [sv.draw_dof(omega, rng, grid) for _ in range(100)]
    assert min(draws) >= 5.0
    assert np.median(draws) == pytest.approx(5.0)


def test_a_floor_at_three_keeps_the_earlier_grid():
    assert np.array_equal(sv.dof_grid(), np.arange(3.0, 51.0))


def test_logvar_path_follows_the_true_one():
    rng = np.random.default_rng(1)
    h = np.cumsum(0.2 * rng.standard_normal(600))
    log_e2 = np.log((np.exp(h / 2) * rng.standard_normal(600)) ** 2)
    path = h * 0
    draws = []
    for step in range(600):
        path = sv.draw_logvar(log_e2, path, 0.04, (0.0, 10.0), rng)
        if step >= 200:
            draws.append(path)
    low, high = np.quantile(draws, [0.05, 0.95], axis=0)
    assert np.mean((low <= h) & (h <= high)) >= 0.8
    assert np.corrcoef(np.mean(draws, axis=0), h)[0, 1] > 0.9


def test_a_jump_mixture_recovers_its_probability_and_scale():
    # Errors with variance kappa = 50 in a tenth of the months and 1 elsewhere: alternating
    # the jump indicators and the parameters finds both.
    rng = np.random.default_rng(4)
    jumps = rng.random(3000) < 0.1
    scaled_e2 = (np.where(jumps, np.sqrt(50.0), 1.0) * rng.standard_normal(3000)) ** 2
    pi, kappa, kept = 0.05, 10.0, []
    for step in range(400):
        omega = sv.draw_jumps(scaled_e2, pi, kappa, rng)
        pi, kappa = sv.draw_jump_parameters(scaled_e2, omega, rng)
        if step >= 100:
            kept.append((pi, kappa))
    pi_draws, kappa_draws = np.array(kept).T
    assert 0.08 <= np.median(pi_draws) <= 0.12
    assert 40.0 <= np.median(kappa_draws) <= 62.0
    assert np.all(kappa_draws > 1.0)


def test_without_jumps_in_the_data_the_jump_probability_stays_small():
    rng = np.random.default_rng(5)
    scaled_e2 = rng.standard_normal(3000) ** 2
    pi, kappa, kept = 0.05, 10.0, []
    for step in range(300):
        omega = sv.draw_jumps(scaled_e2, pi, kappa, rng)
        pi, kappa = sv.draw_jump_parameters(scaled_e2, omega, rng)
        if step >= 100:
            kept.append(pi)
    assert np.median(kept) < 0.02
