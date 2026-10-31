"""The equal-weight aggregate: manual checks against a hand-built fixture."""

import numpy as np
import pytest

from mst.metrics.aggregate import (
    aggregate_bootstrap,
    episode_contributions,
    equal_weight,
    mc_interval,
    utility_by_episode,
)


def rows():
    """Two quantities x two episodes x two months, with hand-checked numbers.

    The model scores: alpha 0.5/0.7 (even) and 0.3/0.4 (odd months), beta 1.0 everywhere;
    the baselines: alpha 1.0 everywhere, beta 0.6/0.8. The paired differences and the columns
    of the aggregate are read off these numbers in the tests.
    """
    rows = []
    for episode in ("e1", "e2"):
        for month in (1, 2):
            model_alpha = 0.5 if month == 1 else 0.7
            model_beta = 1.0
            baseline_alpha = 1.0
            baseline_beta = 0.6 if month == 1 else 0.8
            rows.append(
                {
                    "episode": episode,
                    "month": month,
                    "quantity": "alpha",
                    "level": "macro",
                    "normalized": model_alpha,
                    "baseline_normalized": baseline_alpha,
                    "base_normalized": model_alpha + 0.1,
                    "naive_normalized": None,
                }
            )
            rows.append(
                {
                    "episode": episode,
                    "month": month,
                    "quantity": "beta",
                    "level": "macro",
                    "normalized": model_beta,
                    "baseline_normalized": baseline_beta,
                    "base_normalized": None,
                    "naive_normalized": 2.0,
                }
            )
    return rows


def test_step_one_pools_inside_a_quantity():
    found = equal_weight(rows(), target_level="macro")
    macro = found["levels"]["macro"]
    # The equal weight is over the quantities: alpha 0.5 + beta 1.0 at t=1, alpha 0.7 + beta
    # 1.0 at t=2.
    assert macro["by_horizon"][1]["model"] == pytest.approx(0.75)
    assert macro["by_horizon"][2]["model"] == pytest.approx(0.85)
    # Baselines: alpha 1.0 + beta 0.6 at t=1, alpha 1.0 + beta 0.8 at t=2.
    assert macro["by_horizon"][1]["baseline"] == pytest.approx(0.8)
    assert macro["by_horizon"][2]["baseline"] == pytest.approx(0.9)


def test_a_quantity_with_ten_cells_weighs_as_a_quantity_with_two():
    weighted = rows()
    ten = [
        {
            "episode": "e9",
            "month": 1,
            "quantity": "alpha",
            "level": "macro",
            "normalized": 0.5,
            "baseline_normalized": 1.0,
            "base_normalized": None,
            "naive_normalized": None,
        }
        for _ in range(8)  # eight extra cells: alpha now has 10 at t=1, beta still 2
    ]
    without = equal_weight(weighted, target_level="macro")
    with_extra = equal_weight(weighted + ten, target_level="macro")
    # The mean inside the quantity is unchanged (0.5 again), so the level aggregate cannot move.
    assert with_extra["levels"]["macro"]["by_horizon"][1]["model"] == pytest.approx(
        without["levels"]["macro"]["by_horizon"][1]["model"]
    )


def test_overall_is_the_mean_across_horizons():
    found = equal_weight(rows(), target_level="macro")
    macro = found["levels"]["macro"]
    t1, t2 = macro["by_horizon"][1], macro["by_horizon"][2]
    for column in ("model", "baseline", "base", "naive"):
        if t1[column] is None or t2[column] is None:
            continue
        assert macro["overall"][column] == pytest.approx((t1[column] + t2[column]) / 2)


def test_the_paired_differences_are_per_cell_then_aggregated():
    found = equal_weight(rows(), target_level="macro")
    macro = found["levels"]["macro"]
    # alpha t=1: 0.5 - 1.0 = -0.5; beta t=1: 1.0 - 0.6 = 0.4 -> equal-weight 0.05
    assert macro["by_horizon"][1]["diff_vs_baseline"] == pytest.approx((-0.5 + 0.4) / 2)
    both = macro["by_horizon"][1]["diff_vs_baseline"] + macro["by_horizon"][2]["diff_vs_baseline"]
    assert macro["overall"]["diff_vs_baseline"] == pytest.approx(both / 2)
    # diff_vs_base exists only where both parts exist: alpha yes, beta no.
    assert macro["counters"]["diff_vs_base"]["normalized"] == 4
    assert macro["counters"]["diff_vs_baseline"]["normalized"] == 8


def test_none_cells_are_counted_but_not_averaged():
    thin = [
        {
            "episode": "e1",
            "month": 1,
            "quantity": "q",
            "level": "driver/share_price",
            "normalized": None,
            "baseline_normalized": 1.0,
            "base_normalized": None,
            "naive_normalized": None,
        }
    ]
    found = equal_weight(thin, target_level="driver/share_price")
    level = found["levels"]["driver/share_price"]
    assert level["overall"]["model"] is None
    assert level["overall"]["baseline"] == 1.0
    assert level["counters"]["model"]["scored"] == 1
    assert level["counters"]["model"]["normalized"] == 0


def test_a_target_level_without_cells_is_an_error():
    with pytest.raises(ValueError, match="target_level"):
        equal_weight(rows(), target_level="driver/absent")


def test_utility_by_episode_counts_negative_differences():
    found = utility_by_episode(
        rows(), {"e1": "manual", "e2": "manual"}, target_level="macro"
    )
    manual = found["by_source"]["manual"]["target"]
    # e1 diffs vs base: alpha -0.1, -0.1 (base = model + 0.1), beta none -> mean -0.1.
    assert manual["share_negative"] == 1.0
    assert manual["episodes"] == 2
    assert manual["mean_diff"] == pytest.approx(-0.1)


def test_utility_splits_manual_and_llm():
    split = utility_by_episode(
        rows(), {"e1": "manual", "e2": "llm"}, target_level="macro"
    )
    assert set(split["by_source"]) == {"manual", "llm"}
    for source in ("manual", "llm"):
        assert split["by_source"][source]["target"]["episodes"] == 1


def test_utility_demands_a_source_for_every_episode():
    with pytest.raises(ValueError, match="without a source"):
        utility_by_episode(rows(), {"e1": "manual"}, target_level="macro")


def test_utility_rejects_an_unknown_source():
    with pytest.raises(ValueError, match="outside"):
        utility_by_episode(rows(), {"e1": "human", "e2": "manual"}, target_level="macro")


def test_mc_interval_shrinks_with_the_draw_count():
    # An approximate check: 10x more draws give a visibly narrower Monte-Carlo interval.
    rng = np.random.default_rng(7)
    for draws in (200, 2000):
        cells = [
            {
                "quantity": "alpha",
                "month": 1,
                "level": "macro",
                "draws": rng.normal(0.0, 1.0, draws),
                "fact": 0.3,
            },
            {
                "quantity": "beta",
                "month": 1,
                "level": "macro",
                "draws": rng.normal(1.0, 2.0, draws),
                "fact": 1.4,
            },
        ]
        low, high = mc_interval(cells, repetitions=200, seed=5)
        width = high - low
        if draws == 200:
            wide = width
        else:
            narrow = width
    assert narrow < wide
    assert wide / narrow == pytest.approx(np.sqrt(10), rel=1.0)  # order only: abs=100%


def test_the_bootstrap_aggregates_the_column_the_report_shows():
    found = aggregate_bootstrap(
        rows(),
        ("model", "diff_vs_baseline"),
        level="macro",
        repetitions=200,
        seed=3,
    )
    overall = equal_weight(rows(), target_level="macro")["levels"]["macro"]["overall"]
    assert found["model"]["point"] == pytest.approx(overall["model"])
    assert found["diff_vs_baseline"]["point"] == pytest.approx(overall["diff_vs_baseline"])
    assert found["model"]["low"] <= found["model"]["high"]
    assert found["model"]["se"] >= 0.0
    # The same seed gives the same numbers: the sampling error is reproducible.
    again = aggregate_bootstrap(
        rows(),
        ("model", "diff_vs_baseline"),
        level="macro",
        repetitions=200,
        seed=3,
    )
    assert again == found
    with pytest.raises(ValueError, match="levels present"):
        aggregate_bootstrap(rows(), ("model",), level="driver/absent", repetitions=5, seed=1)


def test_episode_contributions_sum_to_the_aggregate():
    overall = equal_weight(rows(), target_level="macro")["levels"]["macro"]["overall"]
    for column in ("model", "diff_vs_baseline"):
        contributions = episode_contributions(rows(), column, level="macro")
        assert set(contributions) == {"e1", "e2"}
        assert sum(contributions.values()) == pytest.approx(overall[column])
    assert episode_contributions(rows(), "diff_vs_incumbent", level="macro") == {}
