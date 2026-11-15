"""The forecast check set: its windows, verdicts, sums and plan behave as the plan says."""

from datetime import date
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mst.bench.forecast import history, load_plan, rolling
from mst.bench.forecast.pipeline import Replay
from mst.bench.forecast.report import gates
from mst.bench.forecast.rows import distance, inside, ratio, share_gate

ROOT = Path(__file__).resolve().parents[2]


def months(first: date, count: int) -> list[date]:
    start = history.month_index(first)
    return [history.from_index(start + k) for k in range(count)]


def test_windows_start_after_the_history_start_and_skip_a_splice():
    grid = months(date(2014, 11, 1), 8)  # 2014-11 .. 2015-06
    starts = history.windows(grid, 2, date(2015, 1, 1), {date(2015, 3, 1)})
    # A window's changes are its months 1..h: 2014-12 is not history, 2015-03 is a splice.
    assert [grid[t] for t in starts] == [date(2014, 12, 1), date(2015, 3, 1), date(2015, 4, 1)]


def test_changes_leave_out_missing_values():
    values = np.array([0.0, 1.0, np.nan, 4.0, 9.0])
    assert_allclose(history.changes(values, 1, [0, 1, 2, 3]), [1.0, 5.0])


def test_width_is_the_central_quantile_range():
    sample = np.arange(101, dtype=float)
    assert history.width(sample, (0.05, 0.95)) == pytest.approx(90.0)
    assert np.isnan(history.width(np.array([np.nan]), (0.05, 0.95)))


def test_mean_pair_correlation_uses_the_rows_each_pair_shares():
    a = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    columns = np.column_stack([a, 2 * a, -a])
    found, fewest = history.mean_pair_correlation(columns)
    assert found == pytest.approx((1 - 1 - 1) / 3)
    assert fewest == 4


def test_a_ratio_passes_inside_its_bounds_and_is_short_on_few_windows():
    assert ratio("macro", "width", "x", 12, 1.2, 1.0, 50, (0.75, 1.4)).verdict == "pass"
    assert ratio("macro", "width", "x", 12, 1.5, 1.0, 50, (0.75, 1.4)).verdict == "fail"
    assert ratio("macro", "width", "x", 12, 1.0, 1.0, 3, (0.75, 1.4)).verdict == "short"
    assert distance("shares", "c", "m", 12, 0.3, 0.5, 40, 0.15).verdict == "fail"
    assert inside("portfolio", "c", "r", 12, 0.9, 0.9, 30, (0.8, 0.97)).verdict == "pass"


def test_a_share_gate_counts_only_judged_rows():
    rows = [
        ratio("macro", "width", "a", 1, 1.0, 1.0, 50, (0.75, 1.4)),
        ratio("macro", "width", "b", 1, 2.0, 1.0, 50, (0.75, 1.4)),
        ratio("macro", "width", "c", 1, 1.0, 1.0, 2, (0.75, 1.4)),
    ]
    found = share_gate(rows, 0.5, "macro width")
    assert (found["passed"], found["judged"], found["verdict"]) == (1, 2, "pass")
    assert share_gate(rows, 0.8, "macro width")["verdict"] == "fail"


def test_replay_sums_the_observed_positions_of_a_window():
    changes = np.array(
        [
            [[10.0], [20.0]],  # a: 1000, both windows
            [[np.nan], [-40.0]],  # b: 1000, listed from the second window
        ]
    )
    found = Replay([date(2020, 1, 1), date(2020, 2, 1)], ["a", "b"], changes,
                   np.array([[1000.0, 1000.0], [np.nan, 1000.0]]))
    assert_allclose(found.returns()[:, 0], [1.0, -1.0])
    assert_allclose(found.returns(["a"])[:, 0], [1.0, 2.0])
    assert found.window(date(2020, 2, 1)) == 1


def test_reduced_catalog_keeps_the_objects_above_and_their_drivers():
    catalog = {
        "updated": "2026-10-29",
        "quantities": {
            "share_price": {
                "drivers": ["oil", "lkoh", "gazp", "sber", "banks"],
                "objects": [
                    {"id": "shares"},
                    {"id": "oil", "parent": "shares"},
                    {"id": "banks", "parent": "shares"},
                    {"id": "lkoh", "parent": "oil"},
                    {"id": "gazp", "parent": "oil"},
                    {"id": "sber", "parent": "banks"},
                ],
                "sources": {k: {} for k in ("oil", "lkoh", "gazp", "sber", "banks")},
            },
            "bond_spread": {"drivers": ["x"], "objects": [{"id": "x"}], "sources": {"x": {}}},
        },
    }
    found = rolling.reduced_catalog(catalog, ["share_price:lkoh"])
    part = found["quantities"]["share_price"]
    assert set(found["quantities"]) == {"share_price"}
    assert part["drivers"] == ["oil", "lkoh"]
    assert [item["id"] for item in part["objects"]] == ["shares", "oil", "lkoh"]
    assert set(part["sources"]) == {"oil", "lkoh"}


def test_the_forecast_of_a_month_is_made_mid_next_month():
    assert rolling.as_of_of(date(2025, 12, 1)) == date(2026, 1, 15)


def test_the_tail_statistic_is_zero_when_the_losses_match_the_forecast():
    rng = np.random.default_rng(3)
    sample = rng.normal(0, 10, 4000)
    pairs = [(rng.normal(0, 10), sample) for _ in range(4000)]
    assert abs(rolling._acerbi(pairs)) < 0.15


def test_the_crps_of_an_empirical_sample_matches_the_normal_formula():
    sample = np.random.default_rng(5).normal(0, 1, 200000)
    # CRPS of N(0, 1) at 0 is 2 phi(0) - 1 / sqrt(pi).
    expected = 2 / np.sqrt(2 * np.pi) - 1 / np.sqrt(np.pi)
    assert rolling._crps(sample, 0.0) == pytest.approx(expected, abs=0.003)


def test_the_plan_of_the_set_loads_and_every_gate_names_its_horizon():
    plan = load_plan(ROOT / "bench" / "forecast" / "plan.toml")
    assert plan.horizon == 12
    assert plan.quantiles == pytest.approx((0.05, 0.95))
    assert plan.rolling.origins()[-1] <= date(2025, 9, 1)
    rows = [ratio("macro", "width", "x", h, 1.0, 1.0, 50, plan.gates.width) for h in plan.horizons]
    assert [g["gate"] for g in gates(plan, rows)] == [f"macro width, {h} m" for h in plan.horizons]


def test_a_driver_listed_later_is_not_in_the_catalog_of_an_earlier_date():
    class Store:
        def as_of(self, series, day):
            if series in ("drv.share_price.it", "drv.bond_spread.x") and day < date(2020, 1, 1):
                return ["value"] * (rolling.LEAST_MONTHS - 1)
            return ["value"] * rolling.LEAST_MONTHS

    catalog = {"quantities": {
        "share_price": {"drivers": ["oil", "it"], "objects": [], "sources": {}},
        "bond_spread": {"drivers": ["x"], "objects": [], "sources": {}},
    }}
    early = rolling.known_catalog(catalog, Store(), date(2017, 2, 15))
    late = rolling.known_catalog(catalog, Store(), date(2021, 2, 15))
    assert early["quantities"] == {"share_price": {"drivers": ["oil"], "objects": [],
                                                   "sources": {}}}
    assert late["quantities"]["share_price"]["drivers"] == ["oil", "it"]
    assert late["quantities"]["bond_spread"]["drivers"] == ["x"]
