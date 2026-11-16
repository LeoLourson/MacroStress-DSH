"""The scores of the cells: CRPS, calibration, sharpness, intervals and the admission baselines."""

import numpy as np
import pytest

from mst.metrics.baselines import (
    band_problems,
    constraint_probabilities,
    naive_annual,
    random_walk,
)
from mst.metrics.cells import (
    crps,
    finite_draws,
    history_range,
    pit,
    quantiles,
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


def test_pit_is_a_mid_pit_at_an_atom():
    # The documented convention: (below + tied / 2) / R, so an atom at the fact is 0.5.
    assert pit(np.zeros(4), 0.0) == pytest.approx(0.5)


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


def test_history_range_ignores_the_gaps():
    assert history_range(np.array([1.0, np.nan, 4.0, 2.0])) == pytest.approx(3.0)
    assert np.isnan(history_range(np.array([np.nan, np.nan])))


def test_finite_draws_drops_the_non_finite_values():
    assert finite_draws(np.array([1.0, np.nan, np.inf, 2.0])).tolist() == [1.0, 2.0]


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
    # The conservative macro domain: a band below nought breaks it.
    assert band_problems(draws, 0.0, domain=(0.0, np.inf), history_range=100.0) == [
        "band_outside_domain"
    ]


def test_constraint_probabilities_count_the_paths_inside():
    paths = np.array([[[0.0], [1.0]], [[0.0], [3.0]], [[0.0], [5.0]]])  # (3, 2, 1)
    from mst.model_sdk import LinearConstraint, LinearTerm

    constraint = LinearConstraint(
        terms=(LinearTerm(variable="x", month=1, weight=1.0),), low=0.5, high=3.5
    )
    assert constraint_probabilities(paths, ["x"], [constraint]) == {0: 2 / 3}


def test_constraint_probabilities_are_not_applicable_to_a_point():
    paths = np.array([[[0.0], [1.0]], [[0.0], [3.0]], [[0.0], [5.0]]])
    from mst.model_sdk import LinearConstraint, LinearTerm

    exact = LinearConstraint(
        terms=(LinearTerm(variable="x", month=1, weight=1.0),),
        low=0.9,
        high=1.1,
        point=True,
        tolerance=0.1,
    )
    # A point constraint expanded to ±tau: its share would measure tau rather than the
    # scenario, so it is None, not a number.
    assert constraint_probabilities(paths, ["x"], [exact]) == {0: None}
    interval = LinearConstraint(
        terms=(LinearTerm(variable="x", month=1, weight=1.0),), low=1.0, high=2.0
    )
    assert constraint_probabilities(paths, ["x"], [interval]) == {0: 1 / 3}


def test_random_walk_gappy_history_uses_only_true_distance_pairs():
    # The gap must not join distant observations: only grid pairs at true distance three
    # count, both ends known — [7, 6, 6, 7, 10], 7.2 a quarter. The compressed series of the
    # old code read the changes [15, 15, 19] and a drift of 5.44.
    history = np.array([10.0, 12.0, 14.0, 17.0, np.nan, 20.0, 23.0, 25.0, 27.0, 33.0])
    draws = random_walk(history, 6, draws=50_000, seed=0, months=3)
    chg = np.array([7.0, 6.0, 6.0, 7.0, 10.0])
    change, noise = chg.mean(), chg.std(ddof=0)
    # A group of three horizon months carries one 3-month change: after j months the mean is
    # last + j monthly shares of the change; month 3 holds one whole change, month 6 two.
    assert np.allclose(draws.mean(axis=0), 33 + np.arange(1, 7) * change / 3, atol=0.05)
    assert draws.mean(axis=0)[2] == pytest.approx(33 + change, abs=0.05)
    assert draws.std(axis=0)[2] == pytest.approx(noise, abs=0.05)
    assert draws.std(axis=0)[5] == pytest.approx(noise * np.sqrt(2), abs=0.05)


def test_random_walk_gappy_history_with_two_pairs_is_nan():
    # Only (16-10) and (20-14) sit on the 3-grid with both ends known: two pairs are fewer
    # than three, so the walk is NaN. The compressed series of the old code had seven known
    # points and would have walked.
    history = np.array([10.0, 12.0, 14.0, 16.0, np.nan, 20.0, np.nan, 24.0, np.nan, 30.0])
    draws = random_walk(history, 2, draws=1, seed=0, months=3)
    assert draws.shape == (1, 2)
    assert np.all(np.isnan(draws))


def test_the_walk_is_deterministic():
    history = np.array([0.0, 1.0, 3.0, 6.0, 10.0, 15.0])
    walk = random_walk(history, 4, draws=8, seed=5, months=2)
    assert np.array_equal(walk, random_walk(history, 4, draws=8, seed=5, months=2))


def test_random_walk_carries_the_last_value_with_the_trend():
    history = np.arange(50.0) + np.random.default_rng(20).normal(0, 0.05, 50)
    draws = random_walk(history, 4, draws=500, seed=11)
    assert draws.shape == (500, 4)
    # The trend is about one a month, so the mean path follows 50, 51, 52, 53.
    assert np.allclose(draws.mean(axis=0), [50.0, 51.0, 52.0, 53.0], atol=0.2)
    assert draws.std(axis=0).mean() > 0


def test_naive_annual_advances_the_year_ago_value_with_the_horizon():
    history = np.arange(30.0)
    draws = naive_annual(history, 4, draws=200, seed=12)
    assert draws.shape == (200, 4)
    # The targets are months 30..33; a year earlier they are 18..21, plus 1..4 drifts.
    assert np.allclose(draws.mean(axis=0), [19.0, 21.0, 23.0, 25.0], atol=0.05)
    # At t = 12 the year-ago month is the last observation, 29: 29 + 12.
    assert naive_annual(history, 12, draws=20, seed=12).mean(axis=0)[-1] == pytest.approx(
        41.0, abs=0.1
    )
    assert naive_annual(np.arange(5.0), 3, draws=2, seed=1) is None


def test_naive_annual_falls_back_to_the_last_known_value():
    history = np.arange(30.0)
    history[18] = np.nan
    draws = naive_annual(history, 2, draws=200, seed=12)
    # The t = 1 target year-ago month is 18, unknown; the last known value at or before is 17.
    assert draws.mean(axis=0)[0] == pytest.approx(18.0, abs=0.05)
