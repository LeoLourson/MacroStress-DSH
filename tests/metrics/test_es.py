"""The Acerbi-Székely Z: calibration, a compressed forecast, strictness and the window tail.

The core statistics are checked on synthetic ``(loss, VaR, ES)`` terms; ``tail`` on synthetic
windows whose realized loss is drawn from the predictive law itself, or from a wider one.
"""

import numpy as np
import pytest

from mst.metrics.es import (
    contributions,
    paired_difference,
    tail,
    trim_terms,
    value_at_risk,
    z_statistic,
)
from mst.plugin_sdk import expected_shortfall

DRAWS = 2000
"""The repetitions of one episode's predictive loss distribution."""
EPISODES = 200
SEEDS = tuple(range(20))
"""Enough seeds that the mean of Z is tight: one episode's contribution fires rarely."""


def terms(rng, *, episodes: int = EPISODES, compress: float = 1.0):
    """(loss, VaR, ES) of every episode: the realized loss from the predictive law itself.

    With ``compress`` below one the predictive tail is understated — VaR and ES are both
    scaled down, as if the forecast had been squeezed towards the centre.
    """
    predictive = rng.standard_normal((episodes, DRAWS))
    losses = rng.standard_normal(episodes)
    var = np.array([value_at_risk(row) for row in predictive]) * compress
    es = np.array([expected_shortfall(row) for row in predictive]) * compress
    return list(zip(losses.tolist(), var.tolist(), es.tolist(), strict=True))


def test_a_calibrated_tail_averages_zero_over_seeds():
    # One episode's contribution is zero with probability 0.95 and ~20 with probability 0.05,
    # so a single Z wanders by ~0.3/sqrt(episodes); the mean over seeds reads the bias only.
    found = [z_statistic(terms(np.random.default_rng(seed))) for seed in SEEDS]
    assert abs(float(np.mean(found))) < 0.15
    assert all(abs(z) < 1.0 for z in found)


def test_a_tail_compressed_towards_the_centre_reads_positive():
    zs = [z_statistic(terms(np.random.default_rng(seed), compress=0.5)) for seed in SEEDS[:5]]
    assert all(z > 0.5 for z in zs)


def test_a_loss_exactly_at_var_contributes_nothing():
    # The comparison is strict: l_e > VaR_e, not >=.
    assert contributions([(5.0, 5.0, 1.0)]) == [0.0]
    assert z_statistic([(5.0, 5.0, 1.0)]) == pytest.approx(-1.0)
    assert z_statistic([(5.0, 4.0, 1.0)]) == pytest.approx(99.0)


def test_z_is_the_mean_of_the_contributions_less_one():
    found = terms(np.random.default_rng(7))
    assert z_statistic(found) == pytest.approx(np.mean(contributions(found)) - 1.0)


def test_trimming_the_heaviest_draws_lowers_var_and_es():
    # A 1 % cut of 200 draws drops exactly the two heaviest losses of the tail.
    predictive = [np.arange(200.0), np.arange(200.0) * 2.0]
    trimmed = trim_terms([7.0, 9.0], predictive, share=0.01)
    assert trimmed[0][0] == 7.0 and trimmed[1][0] == 9.0  # the realized losses are untouched
    for (loss, var, es), draws in zip(trimmed, predictive, strict=True):
        kept = draws[:-2]
        # ES of the kept sample at level 0.95, its weighted upper tail — not the mean of kept.
        assert (loss, var, es) == (
            loss,
            value_at_risk(kept),
            expected_shortfall(kept),
        )
        assert value_at_risk(kept) < value_at_risk(draws)
        assert expected_shortfall(kept) < expected_shortfall(draws)


def test_trimming_below_one_repetition_keeps_the_terms():
    # Fewer than 1/share draws: nothing is cut, the documented boundary — the trimmed
    # VaR and ES equal the untrimmed ones.
    draws = np.arange(99.0)
    found = trim_terms([3.0], [draws], share=0.01)
    assert found == [(3.0, value_at_risk(draws), expected_shortfall(draws))]
    with pytest.raises(ValueError, match="share"):
        trim_terms([1.0], [draws], share=0.0)


def test_value_at_risk_is_the_quantile_of_the_losses():
    grid = np.arange(101, dtype=float)
    assert value_at_risk(grid) == pytest.approx(95.0)
    assert value_at_risk(grid, level=0.9) == pytest.approx(90.0)
    with pytest.raises(ValueError, match="no repetition"):
        value_at_risk(np.empty(0))


def test_paired_difference_separates_the_shift_of_the_centre_from_the_change_of_width():
    rng = np.random.default_rng(4)
    centre = rng.standard_normal(400)
    narrow = centre + 10.0 + 0.1 * rng.standard_normal(400)
    wide = centre + 10.0 + 2.0 * rng.standard_normal(400)
    found = paired_difference(narrow, wide, seed=5)
    # The same centre: the difference of medians is near zero with an interval around it.
    assert abs(found["centre"]["value"]) < 0.2
    low, high = found["centre"]["interval"]
    assert low <= found["centre"]["value"] <= high
    # The first sample is the narrower one, so the difference of widths is below zero.
    assert found["width"]["value"] < -1.0
    assert found["width"]["interval"][1] < 0.0
    assert found["p95"]["value"] < 0.0
    assert set(found) == {"centre", "width", "p95", "es95"}
    for block in found.values():
        assert set(block) == {"value", "interval"} and len(block["interval"]) == 2


def test_paired_difference_is_deterministic_and_refuses_mismatched_samples():
    first = np.random.default_rng(9).standard_normal(60)
    second = np.random.default_rng(10).standard_normal(60)
    assert paired_difference(first, second, seed=2) == paired_difference(first, second, seed=2)
    assert paired_difference(first, second, seed=2) != paired_difference(first, second, seed=3)
    with pytest.raises(ValueError, match="two samples"):
        paired_difference(first, second[:10], seed=2)
    with pytest.raises(ValueError, match="two samples"):
        paired_difference(np.empty(0), second, seed=2)


def test_paired_difference_of_a_pure_shift_matches_it():
    # The two samples differ by a constant: the centre, P95 and ES95 move by it exactly and
    # the width does not move at all, while the paired resampling keeps every interval tight.
    first = np.random.default_rng(6).standard_normal(500)
    found = paired_difference(first, first - 3.0, seed=7)
    for name in ("centre", "p95", "es95"):
        assert found[name]["value"] == pytest.approx(3.0, abs=1e-9)
        assert found[name]["interval"][0] == pytest.approx(3.0, abs=1e-9)
        assert found[name]["interval"][1] == pytest.approx(3.0, abs=1e-9)
    assert found["width"]["value"] == pytest.approx(0.0, abs=1e-9)
    assert found["width"]["interval"][0] == pytest.approx(0.0, abs=1e-9)


def test_bad_terms_are_refused_by_name():
    with pytest.raises(ValueError, match="no episodes"):
        z_statistic([])
    with pytest.raises(ValueError, match="episode 1"):
        z_statistic([(1.0, 1.0, 1.0), (1.0, 1.0, 0.0)])
    with pytest.raises(ValueError, match="episode 0"):
        z_statistic([(np.nan, 1.0, 1.0)])
    with pytest.raises(ValueError, match="triple"):
        z_statistic([(1.0, 1.0), (2.0, 2.0)])


def windows_of(rng, *, count: int = 200, draws: int = 1000, compress: float = 1.0):
    """Windows whose realized loss comes from the predictive law; ``compress`` below one
    squeezes the predictive law towards the centre while the facts keep theirs."""
    return [
        (f"w{number:03d}", float(rng.standard_normal()), rng.standard_normal(draws) * compress)
        for number in range(count)
    ]


def test_the_tail_of_calibrated_windows_averages_zero_over_seeds():
    found = [
        tail(windows_of(np.random.default_rng(seed)), block=1, seed=seed, repetitions=200)["z"]
        for seed in range(20)
    ]
    assert abs(float(np.mean(found))) < 0.25


def test_the_tail_of_a_compressed_forecast_reads_positive_and_trimming_raises_it():
    found = tail(
        windows_of(np.random.default_rng(1), compress=0.5), block=1, seed=1, repetitions=200
    )
    assert found["z"] > 0.5
    low, high = found["interval"]
    assert low < found["z"] < high
    assert found["trimmed_z"] > found["z"]


def test_a_window_without_a_positive_expected_shortfall_is_named():
    good = windows_of(np.random.default_rng(2), count=3)
    found = tail([*good, ("flat", 1.0, np.full(1000, -5.0))], block=1, seed=0, repetitions=50)
    assert [item["id"] for item in found["dropped"]] == ["flat"]
    assert [item["id"] for item in found["windows"]] == ["w000", "w001", "w002"]
    nothing = tail([("flat", 1.0, np.full(10, -5.0))], block=1, seed=0, repetitions=50)
    assert nothing["z"] is None and nothing["interval"] is None


def test_the_tail_is_deterministic_and_the_block_moves_only_the_interval():
    windows = windows_of(np.random.default_rng(6), count=120, compress=0.7)
    one = tail(windows, block=1, seed=7, repetitions=300)
    assert one == tail(windows, block=1, seed=7, repetitions=300)
    blocked = tail(windows, block=12, seed=7, repetitions=300)
    assert blocked["z"] == one["z"] and blocked["trimmed_z"] == one["trimmed_z"]
    assert blocked["interval"] != one["interval"]
