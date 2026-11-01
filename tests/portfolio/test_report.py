"""mst.portfolio.report: the loss of the portfolio at P5/P50/P95 and the price of a scenario."""

from dataclasses import replace

import numpy as np
import pytest

from mst.engine import run
from mst.metrics.es import es_block
from mst.model_sdk.contracts import Dist
from mst.plugin_sdk import Position
from mst.portfolio.report import build, path_block
from mst.portfolio.spec import Portfolio
from tests.engine import simulated
from tests.engine.fixed import fan, paths

SERVED = {
    "share_price:sber",
    "share_price:gazp",
    "bond_spread:corp",
    "overdue_share:mining",
    "housing_price_primary:belgorod",
}


def portfolio(*positions):
    return Portfolio.model_validate(
        {"name": "тест", "positions": [Position.model_validate(item) for item in positions]}
    )


def batch(*positions, paths_=None):
    checked = [Position.model_validate(item) for item in positions]
    return run(checked, paths_ or paths(), drivers=simulated.registry(), served=SERVED)


def equity(**changes):
    values = {"id": "sber", "type": "common_stock", "notional": 10.0, "codes": {"secid": "SBER"}}
    values.update(changes)
    return values


def test_the_report_sums_losses_inside_each_repetition():
    first = equity()
    second = equity(id="gazp", notional=5.0, codes={"secid": "GAZP"})
    found = batch(first, second)
    report = build(portfolio(first, second), found.results)
    losses = -np.sum([result.value_change for result in found.results], axis=0)
    assert report["loss"]["p05"] == pytest.approx(float(np.percentile(losses, 5)))
    assert report["loss"]["p50"] == pytest.approx(float(np.percentile(losses, 50)))
    assert report["loss"]["p95"] == pytest.approx(float(np.percentile(losses, 95)))
    assert report["loss"]["es95"] >= report["loss"]["p95"]
    assert report["status"] == "ok" and report["flags"] == []
    assert [row["position"] for row in report["positions"]] == ["sber", "gazp"]


def test_the_price_of_the_scenario_stands_beside_the_loss():
    found = Dist(
        distance=2.659,
        log_probability=-3.1,
        standard_error=0.2,
        causes=("ru.monetary",),
        tolerances=(0.4,),
        by_level={0: 2.0, 2: 0.659},
    )
    report = build(portfolio(equity()), batch(equity()).results, dist_found=found)
    assert report["dist"] == {
        "distance": 2.659,
        "log_probability": -3.1,
        "standard_error": 0.2,
        "estimated": True,
        "causes": ["ru.monetary"],
        "tolerances": [0.4],
        "by_level": {"0": 2.0, "2": 0.659},
        "message": None,
    }
    assert build(portfolio(equity()), batch(equity()).results)["dist"] is None


def test_a_price_with_a_large_error_is_not_estimated_but_stays_visible():
    # The canon's ten-percent rule: the number stays in the report, marked as not an
    # estimate, and the flag says so without touching the status.
    noisy = Dist(distance=2.0, log_probability=-3.0, standard_error=0.5)
    report = build(portfolio(equity()), batch(equity()).results, dist_found=noisy)
    assert report["dist"]["estimated"] is False
    assert report["dist"]["distance"] == 2.0
    assert "price_not_estimated" in report["flags"]
    assert report["status"] == "ok"
    calm = Dist(distance=2.0, log_probability=-3.0, standard_error=0.1)
    quiet = build(portfolio(equity()), batch(equity()).results, dist_found=calm)
    assert quiet["dist"]["estimated"] is True
    assert "price_not_estimated" not in quiet["flags"]


def test_an_infinite_price_is_passed_as_is_with_its_message():
    # At some value of R the event has no probability: the report names the contradiction
    # instead of passing a bare infinity. Such a price is not an estimate either, and the
    # flag says so while the status stays what the positions make it.
    found = Dist(
        distance=float("inf"),
        log_probability=float("-inf"),
        standard_error=float("inf"),
        message="the event has no probability at some value of R: the scenario contradicts "
        "the models (lemma 2 of the canon)",
    )
    report = build(portfolio(equity()), batch(equity()).results, dist_found=found)
    assert report["dist"]["distance"] == float("inf")
    assert report["dist"]["estimated"] is False
    assert "price_not_estimated" in report["flags"]
    assert "contradicts the models" in report["dist"]["message"]


def test_an_invalid_position_is_excluded_with_its_reason():
    good = equity()
    broken = equity(id="gbp", currency="GBP", notional=40.0)
    report = build(portfolio(good, broken), batch(good, broken).results)
    assert report["status"] == "incomplete"
    # The unpriceable position has no value in the base currency, so the coverage of the
    # portfolio's value is unknown; the report names the position it could not measure.
    assert report["coverage"]["share"] is None
    assert report["coverage"]["excluded"][0]["position"] == "gbp"
    assert "no macro path from GBP to RUB" in report["coverage"]["excluded"][0]["reason"]
    assert "portfolio_exclusions" in report["flags"]
    assert "exposure_unknown:gbp" in report["flags"]
    assert report["loss"] is not None


def test_nothing_calculated_has_no_loss():
    broken = equity(currency="GBP")
    report = build(portfolio(broken), batch(broken).results)
    assert report["loss"] is None
    assert report["status"] == "incomplete"
    assert report["coverage"]["share"] is None


def test_a_position_that_needs_review_makes_the_portfolio_need_review():
    result = batch(equity()).results[0]
    review = result.model_copy(update={"status": "needs_review"})
    report = build(portfolio(equity()), [review])
    assert report["status"] == "needs_review"
    assert report["loss"] is not None


def test_es95_is_the_mean_of_the_worst_five_percent():
    paths_ = paths(draws=100)
    falling = dict(paths_.drivers)
    falling["share_price:sber"] = fan(100.0, 70.0, draws=100)
    result = batch(equity(), paths_=replace(paths_, drivers=falling)).results[0]
    losses = np.sort(-np.asarray(result.value_change))
    assert result.metrics["equity.loss"].quantiles.es95 == pytest.approx(losses[-5:].mean())


MONTHS = ["2027-01", "2027-02", "2027-03"]


def test_path_block_reports_every_month_and_the_maximum_over_the_path():
    monthly = np.array(
        [
            [1.0, 5.0, 2.0],
            [2.0, 6.0, 3.0],
            [0.0, 1.0, 4.0],
            [3.0, 2.0, 5.0],
        ]
    )
    block = path_block(monthly, MONTHS, seed=7)
    assert [row["month"] for row in block["by_month"]] == MONTHS
    for t, row in enumerate(block["by_month"]):
        p05, p50, p95 = np.percentile(monthly[:, t], [5, 50, 95])
        assert row["p05"] == pytest.approx(float(p05))
        assert row["p50"] == pytest.approx(float(p50))
        assert row["p95"] == pytest.approx(float(p95))
        assert row["es95"] >= row["p95"]
        assert set(row) == {"month", "p05", "p50", "p95", "es95"}
    worst = monthly.max(axis=1)
    tail = es_block(worst, seed=7)
    assert block["path"]["p95"] == pytest.approx(float(np.percentile(worst, 95)))
    assert block["path"]["es95"] == pytest.approx(max(tail["es95"], np.percentile(worst, 95)))
    assert block["path"]["es95_interval"] == pytest.approx(tail["interval"])
    assert block["path"]["es95"] >= max(row["es95"] for row in block["by_month"])
    assert set(block["path"]) == {"p05", "p50", "p95", "es95", "es95_interval"}


def test_the_row_of_a_month_sums_losses_inside_a_draw():
    # The row is the quantile of the sum over the positions within one draw, not the sum of
    # the positions' own quantiles: the difference is diversification.
    first = np.tile(np.array([[1.0], [2.0], [3.0], [4.0]]), (1, 3))
    second = np.tile(np.array([[40.0], [10.0], [30.0], [20.0]]), (1, 3))
    block = path_block(first + second, MONTHS, seed=1)
    row = block["by_month"][0]
    within = first[:, 0] + second[:, 0]
    assert row["p95"] == pytest.approx(float(np.percentile(within, 95)))
    assert row["p95"] != pytest.approx(
        float(np.percentile(first[:, 0], 95)) + float(np.percentile(second[:, 0], 95))
    )


def test_path_block_refuses_months_that_do_not_match_the_losses():
    monthly = np.zeros((4, 3))
    with pytest.raises(ValueError, match="not .4, 2."):
        path_block(monthly, MONTHS[:2], seed=1)
    with pytest.raises(ValueError, match="no repetition"):
        path_block(np.zeros((0, 3)), MONTHS, seed=1)


NOT_ADMITTED = {
    "admitted": False,
    "limit": 0.05,
    "shares": {"band_outside_domain": 0.0, "band_wider_than_history": 0.2},
    "broken": [
        {"condition": "band_wider_than_history", "quantity": "driver/price:alpha", "share": 0.2}
    ],
}


def test_a_report_that_is_not_admitted_names_the_quantity_and_keeps_the_loss():
    report = build(
        portfolio(equity()), batch(equity()).results, admission=NOT_ADMITTED
    )
    assert report["status"] == "not_admitted"
    assert report["admission"] == NOT_ADMITTED
    assert "not_admitted" in report["flags"]
    assert report["loss"] is not None  # the numbers stay, for diagnostics
    # An admitted calculation changes nothing.
    admitted = {**NOT_ADMITTED, "admitted": True}
    plain = build(portfolio(equity()), batch(equity()).results, admission=admitted)
    assert plain["status"] == "ok" and "not_admitted" not in plain["flags"]


def test_not_admitted_wins_over_incomplete_and_needs_review():
    good, broken = equity(), equity(id="gbp", currency="GBP", notional=40.0)
    excluded = build(
        portfolio(good, broken), batch(good, broken).results, admission=NOT_ADMITTED
    )
    assert excluded["status"] == "not_admitted"
    assert "portfolio_incomplete" in excluded["flags"]  # the coverage flags stay
    review = batch(equity()).results[0].model_copy(update={"status": "needs_review"})
    flagged = build(portfolio(equity()), [review], admission=NOT_ADMITTED)
    assert flagged["status"] == "not_admitted"


def test_an_unverified_tail_and_a_rare_condition_flag_the_report_without_changing_it():
    flagged = batch(equity()).results[0].model_copy(
        update={"flags": ["outside_training_range:share_price:sber"]}
    )
    report = build(portfolio(equity()), [flagged], rare_constraints=[2])
    assert report["status"] == "ok"
    assert "tail_unverified" in report["flags"]
    assert "constraint_rare:2" in report["flags"]
