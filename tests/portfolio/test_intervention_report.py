"""Intervention remoteness is measured on the empty scenario, independently of price."""

from types import SimpleNamespace

import numpy as np
import pytest

from mst.portfolio import report


def test_set_remoteness_is_base_fraction_at_or_below_the_imposed_value():
    base = SimpleNamespace(
        ids=["ru.key_rate"],
        months=["2026-09", "2026-10", "2026-11"],
        shadow=np.array([
            [[10.0], [11.0], [12.0]],
            [[10.0], [13.0], [14.0]],
            [[10.0], [15.0], [16.0]],
        ]),
        drivers=SimpleNamespace(ids=[], paths=np.empty((3, 3, 0))),
    )
    intervention = SimpleNamespace(
        target="ru.key_rate", months=(1,), kind="set", values=np.full((3, 1), 13.0), active=None
    )
    (row,) = report.interventions(base, [intervention], inner=1)
    assert row["quantity"] == "ru.key_rate"
    assert row["month"] == "2026-10"
    assert row["kind"] == "set"
    assert row["alpha"] == pytest.approx(2 / 3)
    assert row["active_share"] == 1.0


def test_no_interventions_produce_an_empty_diagnostic_block():
    assert report.interventions(SimpleNamespace(), [], inner=1) == []


def test_a_random_intervention_reports_its_active_share_and_the_law_of_its_alpha():
    """Values that differ across groups give the distribution of remoteness, not one number."""
    base = SimpleNamespace(
        ids=["ru.key_rate"],
        months=["2026-09", "2026-10"],
        shadow=np.array([[[10.0], [11.0]], [[10.0], [13.0]], [[10.0], [15.0]], [[10.0], [17.0]]]),
        drivers=SimpleNamespace(ids=[], paths=np.empty((4, 2, 0))),
    )
    intervention = SimpleNamespace(
        target="ru.key_rate",
        months=(1,),
        kind="set",
        values=np.array([[11.0], [13.0], [15.0], [17.0]]),
        active=np.array([[True], [False], [True], [True]]),
    )
    (row,) = report.interventions(base, [intervention], inner=1)
    assert row["active_share"] == pytest.approx(0.75)
    assert row["alpha"] is None
    assert set(row["alpha_distribution"]) == {"p05", "p50", "p95"}
    assert 0.0 <= row["alpha_distribution"]["p05"] <= row["alpha_distribution"]["p95"] <= 1.0
