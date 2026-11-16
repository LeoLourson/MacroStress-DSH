"""mst.metrics.valuation: the paths of a window in own units and the positions valued on them.

The fake equity position of ``mst.plugins.equity`` reads one share price: its paths are chosen
so that the engine's value change is the wanted number, which makes every valuation known.
"""

from datetime import date

import numpy as np
import pytest

from mst.data.store import Observation, SeriesStore
from mst.drivers.bundle import SatelliteBundle
from mst.drivers.registry import Registry
from mst.macro.panel import Panel
from mst.metrics.valuation import (
    conditional_paths,
    portfolio_exclusions,
    portfolio_run,
    realized_paths,
    value,
)
from mst.model_sdk.contracts import Check, DriverPaths, MacroSimulation
from mst.plugin_sdk import FixedPaths, Position
from mst.portfolio.spec import Portfolio

T0 = date(2026, 9, 1)
"""The calculation month of the fake episodes."""


CHECKED_ON = date(2027, 2, 1)
"""The date the facts of the window are known on."""


def driver_registry() -> Registry:
    """A registry with the one quantity and driver the fake equity position reads."""
    return Registry.model_validate(
        {
            "quantities": [
                {
                    "id": "share_price",
                    "label": "share price",
                    "unit": "RUB",
                    "transform": "log100",
                    "regressors": ["gl.brent"],
                    "lags": 0,
                }
            ],
            "objects": [
                {
                    "id": "sber",
                    "class": "company",
                    "label": "Sber",
                    "codes": {"secid": ["SBER"]},
                }
            ],
            "drivers": [
                {
                    "quantity": "share_price",
                    "object": "sber",
                    "series": "drv.share_price.sber",
                }
            ],
        }
    )


DRIVERS = driver_registry()


SERVED = ["share_price:sber"]


PORTFOLIO = Portfolio(
    name="tail",
    positions=[
        Position(id="sber", type="common_stock", notional=10.0, codes={"secid": "SBER"}),
    ],
)
"""One share of ten million roubles: the fake paths set its loss exactly."""


def bundle() -> SatelliteBundle:
    """A satellite bundle that serves the share price of Sber."""
    return SatelliteBundle(
        macro="fake",
        as_of=T0,
        registry=DRIVERS,
        parts={"share_price": {"driver": ["share_price:sber"]}},
        quantity_model={"share_price": "fake.sat"},
        models={},
        plugins={},
        checks={
            "share_price": {
                0: Check(satellite=1.0, baseline=2.0, forecasts=3, served=True, reason="")
            }
        },
        correlated=[],
        correlation=np.zeros((0, 0)),
        arrays={},
    )


def loss_paths(losses) -> FixedPaths:
    """The share's paths whose equity value change is minus the wanted portfolio losses."""
    loss = np.asarray(losses, dtype=float).reshape(-1, 1)
    level = 10.0
    path = np.concatenate([np.full_like(loss, level), level - loss], axis=1)
    return FixedPaths(drivers={"share_price:sber": path}, start=T0)


def macro_panel() -> Panel:
    """A two-variable panel of the calculation date, in the model scale."""
    y = np.array([[100 * np.log(80.0), 9.5], [100 * np.log(82.0), 10.0]])
    return Panel(
        ids=["gl.brent", "ru.key_rate"],
        n_global=1,
        months=[date(2026, 8, 1), T0],
        y=y,
        approximate=np.zeros(y.shape, dtype=bool),
    )


def fact_store(root, *, october: bool = True, november: bool = True) -> SeriesStore:
    """The share-price facts: known on the calculation date and on the check date."""
    store = SeriesStore(root)
    rows = [
        Observation(date(2026, 8, 1), 90.0, date(2026, 8, 31), False),
        Observation(date(2026, 9, 1), 100.0, date(2026, 9, 30), False),
        # A revision published after the calculation date: t = 0 must not read it.
        Observation(date(2026, 9, 1), 105.0, CHECKED_ON, False),
    ]
    if october:
        rows.append(Observation(date(2026, 10, 1), 110.0, date(2026, 11, 2), False))
    if november:
        rows.append(Observation(date(2026, 11, 1), 120.0, date(2026, 12, 5), False))
    store.write("drv.share_price.sber", rows)
    return store


def test_conditional_paths_come_back_in_their_own_units():
    simulation = MacroSimulation(
        ids=["gl.brent", "ru.key_rate"],
        months=[T0, date(2026, 10, 1)],
        paths=np.array([[[100 * np.log(80.0), 10.0], [100 * np.log(70.0), 11.0]]]),
    )
    driver_paths = DriverPaths(
        ids=["share_price:sber"],
        paths=np.array([[[100 * np.log(100.0)], [100 * np.log(120.0)]]]),
        flags={"share_price:sber": frozenset({"outside_training_range"})},
    )
    found = conditional_paths(simulation, driver_paths, bundle(), seed=11)
    assert found.macro("gl.brent")[0] == pytest.approx([80.0, 70.0])
    assert found.macro("ru.key_rate")[0] == pytest.approx([10.0, 11.0])
    assert found.driver("share_price:sber")[0] == pytest.approx([100.0, 120.0])
    assert found.flags("share_price:sber") == frozenset({"outside_training_range"})
    assert found.months[0] == T0
    assert found.seed == 11


def test_realized_paths_take_the_calculation_date_vintage_and_the_checked_facts(tmp_path):
    months = [date(2026, 10, 1), date(2026, 11, 1)]
    realized = np.array([[100 * np.log(70.0), 11.0], [100 * np.log(75.0), 12.0]])
    found, missing = realized_paths(
        fact_store(tmp_path),
        as_of=date(2026, 9, 30),
        known_on=CHECKED_ON,
        months=months,
        panel=macro_panel(),
        realized=realized,
        bundle=bundle(),
    )
    assert missing == []
    assert found is not None
    # t = 0 is what the calculation date knows: the revision published later is not read.
    assert found.macro("gl.brent")[0] == pytest.approx([82.0, 70.0, 75.0])
    assert found.macro("ru.key_rate")[0] == pytest.approx([10.0, 11.0, 12.0])
    # t = 0 is the latest driver value known on the calculation date, t = 1 .. H the facts.
    assert found.driver("share_price:sber")[0] == pytest.approx([100.0, 110.0, 120.0])
    assert found.months[0] == T0
    assert found.draws == 1


def test_the_gaps_of_the_middle_are_named_but_keep_the_path_whole(tmp_path):
    months = [date(2026, 10, 1), date(2026, 11, 1)]
    realized = np.array([[np.nan, 11.0], [100 * np.log(75.0), 12.0]])
    found, missing = realized_paths(
        fact_store(tmp_path, october=False),
        as_of=date(2026, 9, 30),
        known_on=CHECKED_ON,
        months=months,
        panel=macro_panel(),
        realized=realized,
        bundle=bundle(),
    )
    assert found is not None
    assert "gl.brent/2026-10" in missing
    assert "drv.share_price.sber/2026-10" in missing
    assert np.isnan(found.macro("gl.brent")[0, 1])
    assert np.isnan(found.driver("share_price:sber")[0, 1])


def test_a_missing_fact_at_an_end_stays_nan_and_is_named(tmp_path):
    # A gap at t = H does not void the window: the position that reads the gap cannot be
    # valued and the episode is dropped by the caller, with the missing fact named.
    months = [date(2026, 10, 1), date(2026, 11, 1)]
    realized = np.array([[100 * np.log(70.0), 11.0], [np.nan, 12.0]])
    found, missing = realized_paths(
        fact_store(tmp_path),
        as_of=date(2026, 9, 30),
        known_on=CHECKED_ON,
        months=months,
        panel=macro_panel(),
        realized=realized,
        bundle=bundle(),
    )
    assert found is not None
    assert "gl.brent/2026-11" in missing
    assert np.isnan(found.macro("gl.brent")[0, -1])

    found, missing = realized_paths(
        fact_store(tmp_path, november=False),
        as_of=date(2026, 9, 30),
        known_on=CHECKED_ON,
        months=months,
        panel=macro_panel(),
        realized=np.array([[100 * np.log(70.0), 11.0], [100 * np.log(75.0), 12.0]]),
        bundle=bundle(),
    )
    assert found is not None
    assert "drv.share_price.sber/2026-11" in missing
    assert np.isnan(found.driver("share_price:sber")[0, -1])


def test_a_served_driver_without_a_series_keeps_the_window(tmp_path):
    # A driver whose parameters come from above has no series of its own: a driver nobody
    # reads must not void the window, and the reason names it once instead of every month.
    registry = Registry.model_validate(
        {
            "quantities": [
                {
                    "id": "share_price",
                    "label": "share price",
                    "unit": "RUB",
                    "transform": "log100",
                    "regressors": ["gl.brent"],
                    "lags": 0,
                }
            ],
            "objects": [
                {"id": "sber", "class": "company", "label": "Sber", "codes": {"secid": ["SBER"]}},
                {"id": "gazp", "class": "company", "label": "Gazprom", "codes": {"secid": []}},
            ],
            "drivers": [
                {"quantity": "share_price", "object": "sber", "series": "drv.share_price.sber"},
                {"quantity": "share_price", "object": "gazp"},
            ],
        }
    )
    served = SatelliteBundle(
        macro="fake",
        as_of=T0,
        registry=registry,
        parts={"share_price": {"driver": ["share_price:sber", "share_price:gazp"]}},
        quantity_model={"share_price": "fake.sat"},
        models={},
        plugins={},
        checks={
            "share_price": {
                0: Check(satellite=1.0, baseline=2.0, forecasts=3, served=True, reason=""),
                1: Check(satellite=1.0, baseline=2.0, forecasts=3, served=True, reason=""),
            }
        },
        correlated=[],
        correlation=np.zeros((0, 0)),
        arrays={},
    )
    found, missing = realized_paths(
        fact_store(tmp_path),
        as_of=date(2026, 9, 30),
        known_on=CHECKED_ON,
        months=[date(2026, 10, 1)],
        panel=macro_panel(),
        realized=np.array([[100 * np.log(70.0), 11.0]]),
        bundle=served,
    )
    assert found is not None
    assert "share_price:gazp/no series" in missing
    assert np.all(np.isnan(found.driver("share_price:gazp")))
    assert np.isfinite(found.driver("share_price:sber")).all()


def test_a_panel_of_another_month_is_refused(tmp_path):
    with pytest.raises(ValueError, match="calculation month"):
        realized_paths(
            fact_store(tmp_path),
            as_of=date(2026, 10, 31),
            known_on=CHECKED_ON,
            months=[date(2026, 11, 1)],
            panel=macro_panel(),
            realized=np.array([[100 * np.log(70.0), 11.0]]),
            bundle=bundle(),
        )


def test_a_position_that_cannot_be_calculated_is_named_and_left_out():
    portfolio = Portfolio(
        positions=[
            Position(id="sber", type="common_stock", notional=10.0, codes={"secid": "SBER"}),
            Position(id="other", type="common_stock", notional=1.0, codes={"secid": "NOWHERE"}),
        ]
    )
    batch = portfolio_run(
        portfolio, loss_paths([1.0, 2.0]), drivers=DRIVERS, served=SERVED, as_of=date(2026, 9, 30)
    )
    assert portfolio_exclusions(batch) == [
        {"position": "other", "reason": "the run serves no share_price for secid NOWHERE"}
    ]
    assert [result.position_id for result in batch.calculated] == ["sber"]


def test_the_valued_positions_give_the_return_of_any_of_them_together():
    portfolio = Portfolio(
        positions=[
            Position(id="sber", type="common_stock", notional=10.0, codes={"secid": "SBER"}),
            Position(id="other", type="common_stock", notional=1.0, codes={"secid": "NOWHERE"}),
        ]
    )
    losses = np.array([1.0, -2.0, 3.5])
    found = value(portfolio, loss_paths(losses), bundle=bundle(), as_of=date(2026, 9, 30))

    assert set(found.changes) == {"sber"}
    assert found.changes["sber"] == pytest.approx(-losses)
    assert found.values == {"sber": pytest.approx(10.0)}
    assert found.excluded == {"other": "the run serves no share_price for secid NOWHERE"}
    # The return is the change over the current value, in percent.
    assert found.returns(["sber"]) == pytest.approx(-100.0 * losses / 10.0)
