"""The position models: the number each one returns on known paths."""

from dataclasses import replace

import pytest

from mst.engine import run
from mst.plugin_sdk import FixedPaths, Position
from tests.engine.fixed import fan, line, paths
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


def bond(duration=2.0):
    return position(
        id="corp",
        type="bond",
        notional=5.0,
        codes={"isin": "RU000A0JX0J2"},
        terms={"duration": duration},
    )


def bond_pnl(duration=2.0, **changes):
    return only(bond(duration), paths(**changes)).metrics["bond.pnl"].quantiles.p50


def test_a_bond_earns_its_yield_when_nothing_moves():
    # The key rate 15 % and the spread 5 p.p. give a yield of 20 % for the two months.
    assert bond_pnl(bond_spread=(5.0, 5.0)) == pytest.approx(5.0 * (1.2 ** (2 / 12) - 1))


def test_a_bond_loses_its_remaining_duration_times_the_spread_widening():
    # Two months later 2 − 2/12 years of duration are left to carry the 1 p.p. widening.
    expected = 5.0 * (1.2 ** (2 / 12) * (1 - (2 - 2 / 12) * 1 / 100) - 1)
    assert bond_pnl(bond_spread=(5.0, 6.0)) == pytest.approx(expected)


def test_a_rate_rise_costs_the_bond_as_much_as_a_spread_widening():
    rate = bond_pnl(bond_spread=(5.0, 5.0), key_rate=(15.0, 16.0))
    assert rate == pytest.approx(bond_pnl(bond_spread=(5.0, 6.0)))


def test_the_curve_is_read_at_the_duration_of_the_bond():
    # Duration 3 years now and 2 years a year later: 14 % on the curve now, 13 % then.
    flat = line(5.0, 5.0, horizon=12)
    curve = {
        "ru.key_rate": line(10.0, 10.0, horizon=12),
        "ru.ofz_1y": line(12.0, 12.0, horizon=12),
        "ru.ofz_5y": line(16.0, 16.0, horizon=12),
        "ru.ofz_10y": line(18.0, 18.0, horizon=12),
    }
    paths_ = FixedPaths(macros=curve, drivers={"bond_spread:corp": flat})
    result = only(bond(3.0), paths_)
    expected = 5.0 * (1.19 * (1 - 2 * (13 - 14) / 100) - 1)
    assert result.metrics["bond.pnl"].quantiles.p50 == pytest.approx(expected)


def test_a_bond_redeemed_before_the_horizon_carries_no_spread_risk():
    redeemed = bond_pnl(0.1, bond_spread=(5.0, 9.0))
    assert redeemed == pytest.approx(5.0 * (1.2 ** (2 / 12) - 1))


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

