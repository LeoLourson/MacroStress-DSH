"""The wild cluster bootstrap with Webb weights: calibration, heavy tails, reproducibility."""

import numpy as np
import pytest

from mst.metrics.cells import bootstrap
from mst.metrics.wild import WEBB_WEIGHTS, disagrees, wild_interval


def test_the_weights_have_mean_zero_and_variance_one():
    assert WEBB_WEIGHTS.shape == (6,)
    assert np.isclose(WEBB_WEIGHTS.mean(), 0.0)
    assert np.isclose((WEBB_WEIGHTS**2).mean(), 1.0)
    assert set(np.round(WEBB_WEIGHTS**2, 12)) == {0.5, 1.0, 1.5}


def test_reproducible_by_seed():
    contributions = {"a": 1.0, "b": -2.0, "c": 3.0, "d": 0.5, "e": -0.5, "f": 2.5}
    first = wild_interval(contributions, repetitions=200, seed=5)
    second = wild_interval(contributions, repetitions=200, seed=5)
    assert first == second


def test_empty_and_all_nan_contributions():
    low, high = wild_interval({}, repetitions=10, seed=1)
    assert np.isnan(low) and np.isnan(high)
    low, high = wild_interval({"a": float("nan")}, repetitions=10, seed=1)
    assert np.isnan(low) and np.isnan(high)


def test_heavy_tailed_contributions_widen_the_wild_interval():
    # One dominant episode: ordinary resampling cannot break it, the wild weights can.
    contributions = {"calm": 0.1, "calm2": -0.1, "calm3": 0.2, "wild": 25.0}
    pair = bootstrap({key: [value] for key, value in contributions.items()}, seed=3)
    wild = wild_interval(contributions, repetitions=1000, seed=3)
    assert wild[1] - wild[0] > pair[1] - pair[0]


def test_a_null_equal_to_the_contributions_degenerates_to_itself():
    # Contributions equal to the null: the centred contributions are zero, every reweighted
    # statistic is the null itself, and the interval degenerates to the point.
    contributions = {"a": 2.0, "b": 2.0, "c": 2.0, "d": 2.0}
    low, high = wild_interval(contributions, repetitions=100, seed=2, null=2.0)
    assert low == pytest.approx(2.0) and high == pytest.approx(2.0)


def test_disagrees_reads_zero_inside_one_interval_only():
    assert disagrees((-1.0, 1.0), (0.5, 2.0)) is True
    assert disagrees((-1.0, 1.0), (-2.0, 2.0)) is False
    assert disagrees((0.5, 1.0), (0.2, 2.0)) is False
    assert disagrees((0.5, 1.0), (-2.0, 0.2)) is True


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_zero_is_covered_about_95_percent_of_the_time(seed):
    # A calibrated statistic: centred normal contributions, the null at the true mean.
    rng = np.random.default_rng(seed)
    contributions = {f"e{k}": float(value) for k, value in enumerate(rng.normal(0, 1, 16))}
    low, high = wild_interval(contributions, repetitions=500, seed=seed)
    assert low <= 0.0 <= high
