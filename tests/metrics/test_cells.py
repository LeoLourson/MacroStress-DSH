"""The scores of the cells: CRPS, calibration, sharpness and the admission baselines."""

import numpy as np
import pytest

from mst.metrics.baselines import (
    band_problems,
    constraint_probabilities,
    naive_annual,
    random_walk,
)
from mst.metrics.cells import (
    BUCKET_SHARES,
    Cell,
    bootstrap,
    crps,
    pit,
    quantiles,
    score_cells,
    sigma_of_month,
    typical_movement,
)


def draws_normal(count: int, seed: int = 0, loc: float = 0.0, scale: float = 1.0):
    return np.random.default_rng(seed).normal(loc, scale, count)


def test_crps_of_a_standard_normal_at_its_mean():
    # CRPS(N(0,1), 0) = E|X| - E|X - X'| / 2 = sqrt(2/pi) - 1/sqrt(pi).
    x = draws_normal(200_000, seed=1)
    expected = np.sqrt(2 / np.pi) - 1 / np.sqrt(np.pi)
    assert crps(x, 0.0) == pytest.approx(expected, abs=0.005)


def test_crps_penalises_a_shifted_forecast_more_than_a_wide_one():
    tight = draws_normal(50_000, seed=2, loc=1.0, scale=0.1)
    wide = draws_normal(50_000, seed=3, loc=0.0, scale=1.0)
    assert crps(tight, 0.0) > crps(wide, 0.0)


def test_pit_is_the_share_of_draws_below_the_fact():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert pit(x, 0.0) == 0.0
    assert pit(x, 5.0) == 1.0
    assert pit(x, 2.5) == 0.5
    assert pit(x, 2.0) == pytest.approx(0.375)  # a tie counts half


def test_quantiles_give_the_band():
    x = np.arange(101, dtype=float)
    band = quantiles(x)
    assert band[0] == pytest.approx(5.0)
    assert band[1] == pytest.approx(50.0)
    assert band[2] == pytest.approx(95.0)


def test_typical_movement_is_the_median_absolute_change():
    y = np.array([0.0, 1.0, 3.0, 6.0, 10.0])
    assert typical_movement(y, 1) == pytest.approx(2.5)  # median of 1, 2, 3, 4
    assert typical_movement(y, 2) == pytest.approx(5.0)  # median of 3, 5, 7
    assert np.isnan(typical_movement(y[:1], 1))


def test_sigma_of_month_covers_every_month_of_the_cells():
    cells = [
        Cell("e", 1, "q", 0.0, np.zeros(3)),
        Cell("e", 2, "q", 0.0, np.zeros(3)),
    ]
    history = np.array([0.0, 1.0, 3.0, 6.0, 10.0, 15.0])
    sigma = sigma_of_month(cells, history)
    assert sigma[1] == pytest.approx(3.0)  # median of 1, 2, 3, 4, 5
    assert sigma[2] == pytest.approx(6.0)  # median of 3, 5, 7, 9


def test_score_cells_aggregates_and_skips_constrained_cells():
    cells = [
        Cell(
            "e1", 1, "q", 0.0,
            draws_normal(500, seed=4), baseline=draws_normal(500, seed=5, loc=2.0),
        ),
        Cell(
            "e1", 2, "q", 0.0,
            draws_normal(500, seed=6, loc=0.5),
            baseline=draws_normal(500, seed=7, loc=2.0),
        ),
        Cell("e1", 1, "q", 0.0, np.zeros(2), constrained=True),
        Cell(
            "e2", 1, "q", 0.0,
            draws_normal(500, seed=8), baseline=draws_normal(500, seed=9, loc=2.0),
        ),
    ]
    row = score_cells(cells, {1: 1.0, 2: 2.0})
    assert row["cells"] == 4 and row["constrained"] == 1 and row["scored"] == 3
    assert row["crps"] > 0
    # The baseline is shifted by two sigmas, so its CRPS is larger.
    assert row["baseline_crps"] > row["crps"]
    # The month-2 cell divides by sigma 2, so the normalised mean is below the raw one.
    assert 0 < row["normalized_crps"] < row["crps"]
    shares = row["calibration"]["shares"]
    assert len(shares) == 4 and sum(shares) == pytest.approx(1.0)
    assert row["calibration"]["expected"] == list(BUCKET_SHARES)


def test_score_cells_without_scored_cells_is_empty_but_complete():
    row = score_cells([Cell("e", 1, "q", 0.0, np.zeros(2), constrained=True)], {})
    assert row["scored"] == 0 and row["crps"] is None and row["calibration"] is None


def test_bootstrap_resamples_the_episodes():
    values = [("e1", 1.0), ("e2", 2.0), ("e3", 3.0), ("e4", 4.0)]
    low, high = bootstrap(values, repetitions=2000, seed=3)
    assert low < 2.5 - 0.4 and high > 2.5 + 0.4
    assert low < np.mean([v for _, v in values]) < high


def test_band_problems_name_the_rule_they_break():
    draws = np.array([-1.0, 0.0, 2.0, 3.0])
    assert band_problems(draws, 0.0, domain=(0.0, 1.0), history_range=100.0) == [
        "band_outside_domain"
    ]
    wide = np.array([-50.0, 0.0, 10.0, 60.0])
    assert band_problems(wide, 0.0, domain=None, history_range=10.0) == [
        "band_wider_than_history"
    ]
    assert band_problems(draws, 0.0, domain=None, history_range=100.0) == []


def test_constraint_probabilities_count_the_paths_inside():
    paths = np.array([[[0.0], [1.0]], [[0.0], [3.0]], [[0.0], [5.0]]])  # (3, 2, 1)
    from mst.model_sdk import LinearConstraint, LinearTerm

    constraint = LinearConstraint(
        terms=(LinearTerm(variable="x", month=1, weight=1.0),), low=0.5, high=3.5
    )
    assert constraint_probabilities(paths, ["x"], [constraint]) == {0: 2 / 3}


def test_random_walk_carries_the_last_value_with_the_trend():
    history = np.arange(50.0) + np.random.default_rng(20).normal(0, 0.05, 50)
    draws = random_walk(history, 4, draws=500, seed=11)
    assert draws.shape == (500, 4)
    # The trend is about one a month, so the mean path follows 50, 51, 52, 53.
    assert np.allclose(draws.mean(axis=0), [50.0, 51.0, 52.0, 53.0], atol=0.2)
    assert draws.std(axis=0).mean() > 0


def test_naive_annual_starts_from_a_year_earlier():
    history = np.arange(30.0)
    draws = naive_annual(history, 3, draws=200, seed=12)
    assert draws.shape == (200, 3)
    # A year earlier is 18, the trend is one a month.
    assert np.allclose(draws.mean(axis=0), [19.0, 20.0, 21.0], atol=0.3)
    assert naive_annual(np.arange(5.0), 3, draws=2, seed=1) is None
