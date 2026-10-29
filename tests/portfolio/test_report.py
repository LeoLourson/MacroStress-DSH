"""mst.portfolio.report: the loss of the portfolio at P5/P50/P95 and the severity of a scenario."""

from dataclasses import dataclass, replace

import numpy as np
import pytest

from mst.engine import run
from mst.plugin_sdk import Position
from mst.portfolio.report import build
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


@dataclass(frozen=True)
class Severity:
    probabilities: np.ndarray
    distance: float
    exceeded: float


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


def test_the_severity_is_what_macro_computed():
    found = Severity(probabilities=np.array([0.2, 0.8]), distance=6.984, exceeded=0.01)
    report = build(portfolio(equity()), batch(equity()).results, severity_found=found)
    assert report["severity"] == {
        "distance": 6.984,
        "exceeded": 0.01,
        "probabilities": [0.2, 0.8],
    }
    plain = build(portfolio(equity()), batch(equity()).results)
    assert plain["severity"] is None


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
