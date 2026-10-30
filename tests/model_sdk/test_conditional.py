"""mst.model_sdk.conditional: drawing shocks under linear constraints."""

import numpy as np
import pytest
from scipy import stats

from mst.model_sdk import (
    InfeasibleScenario,
    box_gibbs,
    dependent_rows,
    draw_driving,
    truncated_normal,
)


def test_the_infeasible_scenario_names_its_constraints_from_one():
    error = InfeasibleScenario([0, 2])
    assert error.constraints == [0, 2]
    assert "constraints 1, 3" in str(error)


def test_a_loading_at_rounding_level_counts_as_none():
    # Driving shocks move the first row by 1e-13 where all shocks move it by about 1.
    z_driving = np.array([[1e-13, 0.0], [0.0, 1.0]])
    z_all = np.array([[1e-13, 0.0, 1.0], [0.0, 1.0, 0.5]])
    assert dependent_rows(z_driving, z_all) == [0]
    assert dependent_rows(np.array([[1.0, 2.0], [2.0, 4.0]]), np.eye(2) * 3) == [1]


def test_a_repeated_row_is_redundant_not_infeasible():
    # Two copies of one constraint are consistent: they are drawn from the intersection of
    # their ranges, not refused as linearly dependent rows.
    z = np.array([[1.0, 0.0], [1.0, 0.0]])
    every = np.array([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    assert dependent_rows(z, every) == []
    rng = np.random.default_rng(7)
    low, high = np.array([-10.0, -1.0]), np.array([10.0, 1.0])
    draws = np.array(
        [draw_driving(z, low, high, rng, sweeps=5, every=every) for _ in range(50)]
    )
    assert np.all(np.abs(draws @ z[1].T) <= 1.0 + 1e-9)


def test_equal_driving_loadings_alone_are_not_a_repeated_constraint():
    # The driving slice of two different constraints may coincide; the full rows do not, so
    # this is still a dependent row the sampler cannot enforce, not a redundant copy.
    driving = np.array([[1.0, 0.0], [1.0, 0.0]])
    every = np.array([[1.0, 0.0, 1.0], [1.0, 0.0, 2.0]])
    assert dependent_rows(driving, every) == [1]


def test_two_copies_of_one_row_with_disjoint_ranges_are_infeasible():
    z = np.array([[1.0, 0.0], [1.0, 0.0]])
    low, high = np.array([1.0, 2.0]), np.array([1.0, 2.0])
    with pytest.raises(InfeasibleScenario, match="constraints 2") as caught:
        draw_driving(z, low, high, np.random.default_rng(0), sweeps=2)
    assert caught.value.constraints == [1]


def test_truncated_normal_far_in_the_tail_stays_in_its_interval():
    rng = np.random.default_rng(0)
    draws = [truncated_normal(40.0, 41.0, rng) for _ in range(200)]
    assert all(40.0 <= d <= 41.0 for d in draws) and np.mean(draws) < 40.1


def test_truncated_normal_over_a_wide_interval_is_finite():
    rng = np.random.default_rng(0)
    draws = [truncated_normal(-1e3, 0.5, rng) for _ in range(2000)]
    assert np.all(np.isfinite(draws)) and max(draws) <= 0.5
    assert stats.kstest(draws, stats.truncnorm(-1e3, 0.5).cdf).pvalue > 0.01


def test_gibbs_matches_rejection_sampling_on_two_variables():
    z = np.array([[1.0, 0.5, 0.0], [0.8, 0.6, 0.2]])
    low, high = np.array([0.5, -0.4]), np.array([1.5, 0.6])
    rng = np.random.default_rng(1)
    gibbs = np.array([draw_driving(z, low, high, rng, sweeps=20) for _ in range(4000)])
    raw = rng.standard_normal((400000, 3))
    image = raw @ z.T
    inside = raw[np.all((image >= low) & (image <= high), axis=1)][:4000]
    assert len(inside) == 4000
    for k in range(3):
        assert stats.ks_2samp(gibbs[:, k], inside[:, k]).pvalue > 0.01
    for found, truth in ((gibbs @ z.T, inside @ z.T), (gibbs, inside)):
        assert np.allclose(np.corrcoef(found.T), np.corrcoef(truth.T), atol=0.05)


def test_box_gibbs_stays_inside_the_box_and_covers_it():
    mean = np.zeros(2)
    cov = np.array([[1.0, 0.3], [0.3, 2.0]])
    low, high = np.array([-0.5, -1.0]), np.array([0.5, 1.0])
    rng = np.random.default_rng(3)
    draws = np.array([box_gibbs(mean, cov, low, high, rng, sweeps=10) for _ in range(2000)])
    assert np.all((draws >= low) & (draws <= high))
    assert abs(draws[:, 0].mean()) < 0.1  # the box is symmetric about the mean


def test_draw_driving_meets_exact_and_interval_rows():
    z = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.5]])
    low, high = np.array([2.0, -1.0]), np.array([2.0, 1.0])
    rng = np.random.default_rng(4)
    eps = draw_driving(z, low, high, rng, sweeps=10)
    assert np.allclose(eps @ z[0].T, 2.0, atol=1e-9)
    assert np.all((eps @ z[1].T >= -1.0 - 1e-9) & (eps @ z[1].T <= 1.0 + 1e-9))
