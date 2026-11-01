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
    BUCKET_SHARES,
    Cell,
    bootstrap,
    crps,
    finite_draws,
    history_range,
    movement_pairs,
    pit,
    quantiles,
    score_cells,
    sigma_of_month,
    sigma_with_fallback,
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


def test_sigma_of_month_covers_every_month_of_the_cells():
    cells = [
        Cell("e", 1, "q", 0.0, np.zeros(3)),
        Cell("e", 2, "q", 0.0, np.zeros(3)),
    ]
    history = np.array([0.0, 1.0, 3.0, 6.0, 10.0, 15.0])
    sigma = sigma_of_month(cells, history)
    assert sigma[1] == pytest.approx(3.0)  # median of 1, 2, 3, 4, 5
    assert sigma[2] == pytest.approx(6.0)  # median of 3, 5, 7, 9


def test_score_cells_keeps_the_random_walk_and_the_base_run_apart():
    model = draws_normal(500, seed=4)
    cells = [
        Cell(
            "e1",
            1,
            "q",
            0.0,
            model,
            baseline=draws_normal(500, seed=5, loc=2.0),
            base=model,
            sigma=1.0,
        ),
        Cell(
            "e1",
            2,
            "q",
            0.0,
            draws_normal(500, seed=6, loc=0.5),
            baseline=draws_normal(500, seed=7, loc=2.0),
            sigma=2.0,
        ),
        Cell("e1", 1, "q", 0.0, np.zeros(2), constrained=True),
        Cell(
            "e2",
            1,
            "q",
            0.0,
            draws_normal(500, seed=8),
            baseline=draws_normal(500, seed=9, loc=2.0),
            sigma=1.0,
        ),
    ]
    row = score_cells(cells)
    assert row["cells"] == 4 and row["constrained"] == 1 and row["scored"] == 3
    assert row["normalized_cells"] == 3 and row["dropped"] == 0
    assert row["crps"] > 0
    # The base run is the model itself, so its score is the model's; the random walk is not.
    assert row["months"][0]["base_crps"] == pytest.approx(row["months"][0]["crps"])
    assert row["base_crps"] is not None
    assert row["baseline_crps"] > row["crps"]
    # The month-2 cell divides by sigma 2, so the normalised mean is below the raw one.
    assert 0 < row["normalized_crps"] < row["crps"]
    shares = row["calibration"]["shares"]
    assert len(shares) == 4 and sum(shares) == pytest.approx(1.0)
    assert row["calibration"]["expected"] == list(BUCKET_SHARES)
    assert row["coverage"] == pytest.approx(1.0)
    assert row["months"][0]["difference"] == pytest.approx(
        row["months"][0]["normalized"] - row["months"][0]["baseline_normalized"]
    )


def test_score_cells_without_a_scale_leaves_the_normalised_mean_empty():
    row = score_cells(
        [
            Cell("e", 1, "q", 0.0, draws_normal(50, seed=1), sigma=float("nan")),
            Cell("e", 2, "q", 0.0, draws_normal(50, seed=2), sigma=0.0),
        ]
    )
    assert row["scored"] == 2 and row["normalized_cells"] == 0
    assert row["normalized_crps"] is None and row["crps"] is not None


def test_score_cells_drops_a_cell_without_finite_draws():
    row = score_cells([Cell("e", 1, "q", 0.0, np.array([np.nan, np.nan]))])
    assert row["dropped"] == 1 and row["scored"] == 0 and row["crps"] is None


def test_score_cells_without_scored_cells_is_empty_but_complete():
    row = score_cells([Cell("e", 1, "q", 0.0, np.zeros(2), constrained=True)])
    assert row["scored"] == 0 and row["crps"] is None and row["calibration"] is None


def test_bootstrap_resamples_the_same_cell_weighted_statistic():
    # One cell in e1 and ten in each of e2..e4: the report shows 38/31 = 1.2258, while the mean
    # of the episode means is 2.75. The interval must be the one of the reported statistic.
    samples = {"e1": [8.0], "e2": [1.0] * 10, "e3": [1.0] * 10, "e4": [1.0] * 10}
    low, high = bootstrap(samples, repetitions=2000, seed=3)
    assert low < 38 / 31 < high
    assert high < 2.75


def test_bootstrap_with_balanced_clusters_contains_the_mean():
    low, high = bootstrap({"e1": [1.0, 3.0], "e2": [2.0, 4.0]}, repetitions=2000, seed=3)
    assert low < 2.5 < high
    empty = bootstrap({}, repetitions=10, seed=0)
    assert np.isnan(empty[0]) and np.isnan(empty[1])


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
    # count, both ends known — [7, 6, 6, 7, 10], drift 2.4 a month. The compressed series of
    # the old code read the changes [15, 15, 19] and a drift of 5.44.
    history = np.array([10.0, 12.0, 14.0, 17.0, np.nan, 20.0, 23.0, 25.0, 27.0, 33.0])
    draws = random_walk(history, 6, draws=50_000, seed=0, months=3)
    chg = np.array([7.0, 6.0, 6.0, 7.0, 10.0])
    drift, noise = chg.mean() / 3, chg.std(ddof=0) / np.sqrt(3)
    # A group of three horizon months carries one 3-month change: after j months the mean is
    # last + j monthly shares of the change; month 3 holds one whole change, month 6 two.
    assert np.allclose(draws.mean(axis=0), 33 + np.arange(1, 7) * drift / 3, atol=0.05)
    assert draws.mean(axis=0)[2] == pytest.approx(33 + drift, abs=0.05)
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


def test_movement_pairs_count_only_the_pairs_with_two_known_ends():
    history = np.array([0.0, np.nan, 3.0, 6.0, np.nan, 15.0])
    assert movement_pairs(history, 1) == 1
    assert movement_pairs(history, 2) == 2
    assert movement_pairs(history, 3) == 2
    assert movement_pairs(history, 6) == 0


def test_sigma_with_fallback_borrows_the_nearest_valid_month():
    # A six-point history: sigma(3) = median of [6, 9, 12] = 9, sigma(4) = 12, sigma(5) = 15;
    # months 6 and 7 are past the history and borrow the largest valid month below, 5.
    history = np.array([0.0, 1.0, 3.0, 6.0, 10.0, 15.0])
    fallback = sigma_with_fallback(history, [3, 4, 5, 6, 7])
    assert fallback[3] == (9.0, "t", 3)
    assert fallback[4] == (12.0, "t", 2)
    assert fallback[5] == (15.0, "t", 1)
    assert fallback[6] == (15.0, "t*", 1)
    assert fallback[7] == (15.0, "t*", 1)


def test_sigma_with_fallback_takes_the_nearest_month_above_when_below_is_empty():
    # Adjacent changes are mostly zeros (sigma(1) = NaN), while the pairs of months two and
    # three apart move (sigma(2) = 0.5, sigma(3) = 1): month 1 borrows the smallest valid
    # month above, month 4 — which does not move over four months — the largest below.
    history = np.array([5.0, 5.0, 5.0, 6.0, 5.0, 5.0, 5.0, 6.0])
    fallback = sigma_with_fallback(history, [1, 2, 4])
    assert fallback[1] == (0.5, "t+", 6)
    assert fallback[2] == (0.5, "t", 6)
    assert fallback[4] == (0.5, "t*", 6)


def test_sigma_with_fallback_pools_the_peers_for_the_last_resort():
    # The own history is too short for month 6 and no neighbour month helps, so the median
    # of the peers' valid sigma(6) stands in; without a valid peer the source is "none".
    history = np.array([0.0, 1.0, 3.0, 6.0, 10.0, 15.0])
    peers = {
        "r1": np.array([0.0, 2.0, 4.0, 6.0]),  # four points: no sigma(6)
        "r2": np.array([5.0, 7.0, 8.0]),  # three points: no sigma(6)
        "r3": np.array([0.0, 1.0, 5.0, 3.0, 9.0, 2.0, 4.0]),  # sigma(6) = 4.0
    }
    fallback = sigma_with_fallback(history, [6], peers=peers)
    assert fallback[6] == (4.0, "pooled", 1)
    peers["r1"] = np.array([0.0, 2.0, 4.0, 6.0, 1.0, 3.0, 5.0, 2.0])  # sigma(6) = 2.5
    fallback = sigma_with_fallback(history, [6], peers=peers)
    assert fallback[6] == (3.25, "pooled", 2)  # the median of [2.5, 4.0]
    none = sigma_with_fallback(history, [6], peers={"r2": peers["r2"]})[6]
    assert np.isnan(none[0]) and none[1] == "none" and none[2] == 0
    none = sigma_with_fallback(history, [6], peers={})[6]
    assert np.isnan(none[0]) and none[1] == "none" and none[2] == 0
    none = sigma_with_fallback(history, [6])[6]
    assert np.isnan(none[0]) and none[1] == "none" and none[2] == 0


def test_sigma_with_fallback_zero_sigma_falls_back_to_the_peers():
    # A constant history does not move: the median absolute change is zero, which is no
    # scale — typical_movement is NaN and the fallback goes to the peers.
    constant = np.full(8, 5.0)
    assert np.isnan(typical_movement(constant, 3))
    peers = {"r1": np.arange(10.0)}  # sigma(3) = 3
    fallback = sigma_with_fallback(constant, [3], peers=peers)
    assert fallback[3] == (3.0, "pooled", 1)
    none = sigma_with_fallback(constant, [3], peers=None)[3]
    assert np.isnan(none[0]) and none[1] == "none" and none[2] == 0


def test_sigma_with_fallback_and_the_walk_are_deterministic():
    history = np.array([0.0, 1.0, 3.0, 6.0, 10.0, 15.0])
    peers = {"r1": np.arange(9.0), "r2": np.arange(12.0)}
    months = [2, 5, 8]
    assert sigma_with_fallback(history, months, peers=peers) == sigma_with_fallback(
        history, months, peers=peers
    )
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
