"""The relative score and the benchmarks that make a comparison like for like."""

import os

import numpy as np
import pytest

from mst import versions
from mst.metrics.baselines import conditional_walk, random_walk
from mst.metrics.block import block_weights
from mst.metrics.cells import crps
from mst.metrics.quality import interval, paired, relative


def rows_of(scores: dict[str, tuple[float, float]], *, units: int = 4) -> list[dict]:
    """Rows of the given model and walk CRPS, per quantity, over some dates and two months."""
    return [
        {"date": f"d{u}", "month": month, "quantity": quantity, "model": model, "walk": walk}
        for quantity, (model, walk) in scores.items()
        for month in (1, 2)
        for u in range(units)
    ]


def test_the_score_is_the_geometric_mean_of_the_ratios():
    found = relative(rows_of({"a": (1.0, 2.0), "b": (2.0, 1.0)}), "model", "walk")
    assert found["score"] == pytest.approx(1.0)
    assert found["by_quantity"] == pytest.approx({"a": 0.5, "b": 2.0})
    assert found["by_horizon"] == pytest.approx({1: 1.0, 2: 1.0})


def test_a_quantity_that_barely_moves_weighs_as_any_other():
    plain = relative(rows_of({"a": (1.0, 2.0), "b": (3.0, 3.0)}), "model", "walk")
    sticky = relative(rows_of({"a": (1.0, 2.0), "b": (0.003, 0.003)}), "model", "walk")
    assert sticky["score"] == pytest.approx(plain["score"])


def test_the_ratio_of_a_group_is_of_its_means_not_a_mean_of_ratios():
    rows = [
        {"date": "d1", "month": 1, "quantity": "a", "model": 1.0, "walk": 0.001},
        {"date": "d2", "month": 1, "quantity": "a", "model": 1.0, "walk": 3.999},
    ]
    # One cell where the walk was lucky does not decide: 2 / 4, not the mean of 1000 and 0.25.
    assert relative(rows, "model", "walk")["score"] == pytest.approx(0.5)


def test_a_cell_without_the_benchmark_leaves_the_comparison():
    rows = [
        {"date": "d1", "month": 1, "quantity": "a", "model": 1.0, "walk": 2.0},
        {"date": "d2", "month": 1, "quantity": "a", "model": 9.0, "walk": None},
    ]
    assert relative(rows, "model", "walk")["score"] == pytest.approx(0.5)
    assert relative(rows, "model", "other")["score"] is None


def test_the_interval_belongs_to_the_score_and_tightens_with_agreement():
    rng = np.random.default_rng(1)
    rows = []
    for u in range(60):
        for quantity in ("a", "b"):
            rows.append(
                {
                    "date": f"d{u:02d}",
                    "month": 1,
                    "quantity": quantity,
                    "model": float(rng.lognormal(0.0, 0.5)),
                    "walk": float(rng.lognormal(0.0, 0.5)),
                }
            )
    found = interval(rows, "model", "walk", unit="date", seed=3)
    assert found["low"] < found["point"] < found["high"]
    assert found["point"] == pytest.approx(relative(rows, "model", "walk")["score"])
    same = interval(rows, "model", "model", unit="date", seed=3)
    assert same["point"] == pytest.approx(1.0)
    assert same["log_se"] == pytest.approx(0.0, abs=1e-12)


def test_blocks_widen_the_interval_of_neighbours_that_move_together():
    # Long runs of a good and a bad stretch: single dates look independent, blocks do not.
    rows = [
        {
            "date": f"d{u:03d}",
            "month": 1,
            "quantity": "a",
            "model": 2.0 if (u // 24) % 2 else 0.5,
            "walk": 1.0,
        }
        for u in range(240)
    ]
    plain = interval(rows, "model", "walk", unit="date", block=1, seed=5)
    blocked = interval(rows, "model", "walk", unit="date", block=24, seed=5)
    assert blocked["log_se"] > 1.5 * plain["log_se"]


def test_the_versions_are_compared_on_their_own_cells():
    mine = rows_of({"a": (1.0, 2.0)})
    other = rows_of({"a": (2.0, 2.0)})
    found = paired(mine, other, "model", unit="date", seed=1)
    assert found["point"] == pytest.approx(0.5)


@pytest.mark.parametrize("count,block", [(10, 1), (10, 3), (10, 10), (7, 20), (1, 1)])
def test_block_weights_cover_the_sample(count, block):
    rng = np.random.default_rng(2)
    for _ in range(20):
        weights = block_weights(count, block, rng)
        assert len(weights) == count
        assert weights.sum() == count


def test_a_block_draws_neighbours_together():
    weights = block_weights(100, 100, np.random.default_rng(0))
    assert np.array_equal(weights, np.ones(100))


def history_with_a_link(months=200, *, seed=4):
    """A quantity whose monthly change is twice the change of the given one, plus small noise."""
    rng = np.random.default_rng(seed)
    given = np.cumsum(rng.normal(0.0, 1.0, months + 12))
    noise = rng.normal(0.0, 0.1, months + 12)
    y = np.cumsum(2.0 * np.diff(given, prepend=0.0) + noise)
    return y[:months], given[:months, None], y[months:], given[months:, None]


def test_the_conditional_walk_reads_the_path_it_is_given():
    y, given, facts, path = history_with_a_link()
    knowing = conditional_walk(y, given, path, draws=2000, seed=1)
    plain = random_walk(y, 12, draws=2000, seed=1)
    # It knows the shock and the link, so it beats the plain walk by far at every month.
    for k in range(12):
        assert crps(knowing[:, k], facts[k]) < 0.5 * crps(plain[:, k], facts[k])
    assert np.median(knowing[:, -1]) == pytest.approx(facts[-1], abs=1.5)


def test_the_conditional_walk_is_a_walk_when_the_given_path_says_nothing():
    rng = np.random.default_rng(6)
    y = np.cumsum(rng.normal(0.5, 1.0, 400))
    unrelated = np.cumsum(rng.normal(0.0, 1.0, 412))
    knowing = conditional_walk(y, unrelated[:400, None], unrelated[400:, None], draws=4000, seed=1)
    plain = random_walk(y, 12, draws=4000, seed=1)
    assert knowing[:, -1].std() == pytest.approx(plain[:, -1].std(), rel=0.1)
    assert knowing[:, -1].mean() == pytest.approx(plain[:, -1].mean(), abs=1.0)


def test_the_conditional_walk_needs_enough_months_for_its_coefficients():
    y = np.arange(5.0)
    given = np.arange(5.0)[:, None]
    found = conditional_walk(y, given, np.full((3, 1), 1.0), draws=4, seed=1)
    assert found.shape == (4, 3)
    assert np.isnan(found).all()


def test_an_unknown_month_of_the_given_path_reads_the_mean_change():
    y, given, _, path = history_with_a_link()
    gappy = path.copy()
    gappy[3] = np.nan
    gappy[4] = np.nan
    found = conditional_walk(y, given, gappy, draws=50, seed=1)
    assert np.isfinite(found).all()


def test_the_walks_start_from_the_last_known_value_and_count_its_months():
    y = np.cumsum(np.random.default_rng(8).normal(0.0, 1.0, 300))
    late = y.copy()
    late[-3:] = np.nan
    far = random_walk(late, 4, draws=4000, seed=1)
    near = random_walk(y[:-3], 4, draws=4000, seed=1)
    # Three more unknown months: the first month of the horizon is four steps away, not one.
    assert far[:, 0].std() == pytest.approx(near[:, 3].std(), rel=0.1)
    assert np.median(far[:, 0]) == pytest.approx(y[-4], abs=1.0)
    given = np.cumsum(np.random.default_rng(9).normal(0.0, 1.0, 304))[:, None]
    knowing = conditional_walk(late, given[:300], given[300:], draws=20, seed=1)
    assert knowing.shape == (20, 4)
    assert np.isfinite(knowing).all()


def test_the_walks_read_the_floor_as_the_model_does():
    y = np.abs(np.random.default_rng(3).normal(0.0, 0.2, 100))
    assert random_walk(y, 12, draws=500, seed=1).min() < 0.0
    assert random_walk(y, 12, draws=500, seed=1, lower=0.0).min() == 0.0
    given = np.cumsum(np.random.default_rng(4).normal(0.0, 1.0, 112))[:, None]
    assert conditional_walk(y, given[:100], given[100:], draws=500, seed=1, lower=0.0).min() == 0.0


def test_a_rename_held_for_a_moment_passes(tmp_path, monkeypatch):
    calls = []
    real = os.replace

    def held(source, target):
        calls.append(source)
        if len(calls) < 3:
            raise PermissionError("held by a scanner")
        real(source, target)

    monkeypatch.setattr(versions.os, "replace", held)
    monkeypatch.setattr(versions.time, "sleep", lambda seconds: None)
    name = versions.save(tmp_path, {"a": 1}, {"x": np.zeros(2)})
    assert len(calls) == 3
    assert (tmp_path / name / "meta.json").exists()


def test_another_process_publishing_the_same_version_is_not_an_error(tmp_path, monkeypatch):
    def beaten(source, target):
        os.makedirs(target)
        raise PermissionError("the target is there")

    monkeypatch.setattr(versions.os, "replace", beaten)
    name = versions.save(tmp_path, {"a": 1}, {"x": np.zeros(2)})
    assert (tmp_path / name).is_dir()
    assert [path.name for path in tmp_path.iterdir()] == [name]


def test_a_rename_that_never_passes_raises_and_leaves_no_draft(tmp_path, monkeypatch):
    def refused(source, target):
        raise PermissionError("refused")

    monkeypatch.setattr(versions.os, "replace", refused)
    monkeypatch.setattr(versions.time, "sleep", lambda seconds: None)
    with pytest.raises(PermissionError):
        versions.save(tmp_path, {"a": 1}, {"x": np.zeros(2)})
    assert list(tmp_path.iterdir()) == []
