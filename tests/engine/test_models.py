"""The position models: the number each one returns on known paths."""

from dataclasses import replace

import pytest

from mst.engine import run
from mst.plugin_sdk import Position
from tests.engine.fixed import fan, paths
from tests.engine.simulated import registry

SERVED = {
    "share_price:sber",
    "share_price:gazp",
    "bond_spread:corp",
    "overdue_share:mining",
    "housing_price_primary:belgorod",
}


def position(**changes) -> Position:
    values = {
        "id": "sber",
        "type": "common_stock",
        "notional": 10.0,
        "codes": {"secid": "SBER"},
    }
    values.update(changes)
    return Position.model_validate(values)


def only(position_: Position, paths_=None):
    batch = run([position_], paths_ or paths(), drivers=registry(), served=SERVED)
    return batch.results[0]


def test_an_equity_is_worth_its_driver_ratio():
    result = only(position())
    assert result.status == "ok"
    assert result.value_current == 10.0
    assert result.metrics["equity.pnl"].quantiles.p50 == pytest.approx(1.0)
    assert result.metrics["equity.loss"].quantiles.p50 == pytest.approx(-1.0)
    assert result.metrics["equity.loss"].quantiles.es95 is not None
    assert result.metrics["equity.return"].quantiles.p50 == pytest.approx(0.1)


def test_a_foreign_equity_is_valued_at_the_macro_rate():
    result = only(position(currency="USD"))
    # 10 million dollars at 80 roubles, the price up 10 % and the rouble rate up 25 %.
    assert result.value_current == pytest.approx(800.0)
    assert result.metrics["equity.pnl"].quantiles.p50 == pytest.approx(300.0)


def test_a_bond_loses_its_duration_times_the_spread_widening():
    result = only(
        position(
            id="corp",
            type="bond",
            notional=5.0,
            codes={"isin": "RU000A0JX0J2"},
            terms={"duration": 2.0},
        )
    )
    assert result.metrics["bond.pnl"].quantiles.p50 == pytest.approx(-0.1)


def test_a_loan_loses_its_lgd_times_the_overdue_share_change():
    result = only(
        position(
            id="mining",
            type="loan",
            notional=10.0,
            codes={"okved": "05"},
            terms={"lgd": 0.5},
        )
    )
    assert result.metrics["credit.pnl"].quantiles.p50 == pytest.approx(-0.15)


def test_a_deposit_reads_the_same_credit_model():
    result = only(
        position(
            id="deposit",
            type="deposit",
            notional=10.0,
            codes={"okved": "05"},
            terms={"lgd": 0.5},
        )
    )
    assert result.model_id == "credit.paths.v1"
    assert result.status == "ok"


def test_a_housing_position_follows_its_price():
    result = only(
        position(
            id="flat",
            type="real_estate",
            notional=5.0,
            codes={"okato": "14000000000"},
            terms={"market": "primary"},
        )
    )
    assert result.metrics["housing.pnl"].quantiles.p50 == pytest.approx(-0.5)


def test_the_quantiles_of_a_fan_are_ordered_and_the_tail_reaches_past_p95():
    paths_ = paths(draws=5)
    spread = dict(paths_.drivers)
    spread["share_price:sber"] = fan(100.0, 130.0, draws=5)
    result = only(position(), replace(paths_, drivers=spread))
    loss = result.metrics["equity.loss"].quantiles
    assert loss.p05 < loss.p50 < loss.p95
    assert loss.es95 >= loss.p95


def test_driver_flags_reach_the_result():
    paths_ = paths(flags={"share_price:sber": frozenset({"parameters_from_above"})})
    result = only(position(), paths_)
    assert result.flags == ["parameters_from_above:share_price:sber"]


def test_the_same_inputs_give_the_same_numbers():
    first = only(position())
    again = only(position())
    assert first == again

