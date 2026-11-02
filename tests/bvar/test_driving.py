"""The size of the driving shocks the conditions need, in standard deviations."""

from mst.model_sdk.contracts import ScenarioPlan
from mst.plugins.bvar.forecast import simulate
from tests.bvar.test_bounds import bounded, plan


def test_a_far_condition_needs_a_visible_shock():
    # The path starts at four with unit noise; a condition at twenty in the first month
    # asks about sixteen standard deviations of the driving shock, and the diagnostics say
    # so, month by month.
    run = simulate(bounded(0.0), plan([(1, 20.0)]), draws=60, seed=4)
    sizes = run.diagnostics["driving_sd"]
    assert len(sizes["median_by_month"]) == 6
    assert sizes["median_by_month"][0] > 10.0
    assert sizes["max"] >= sizes["median_by_month"][0]
    # The months after the condition draw freely again: about one standard deviation.
    assert sizes["median_by_month"][2] < 2.0


def test_a_plain_run_reports_no_driving_size():
    run = simulate(bounded(0.0), ScenarioPlan(horizon=4), draws=10, seed=1)
    assert "driving_sd" not in run.diagnostics
