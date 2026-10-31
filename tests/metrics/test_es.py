"""The Acerbi-Székely Z of the tail section: calibration, a compressed forecast, strictness.

The core statistics are checked on synthetic ``(loss, VaR, ES)`` terms. The episode-level part
builds a fake episode plan with the equity position model of ``mst.plugins.equity``: the share's
price path is chosen so that the engine's portfolio loss is the wanted number, which makes both
the predictive distribution and the realized loss of every episode known.
"""

from datetime import date

import numpy as np
import pytest

from mst.data.store import Observation, SeriesStore
from mst.drivers.bundle import SatelliteBundle
from mst.drivers.registry import Registry
from mst.macro.panel import Panel
from mst.metrics.es import (
    conditional_paths,
    contributions,
    portfolio_exclusions,
    portfolio_losses,
    portfolio_run,
    realized_paths,
    tail_section,
    trim_terms,
    value_at_risk,
    z_statistic,
)
from mst.model_sdk.contracts import Check, DriverPaths, MacroSimulation
from mst.plugin_sdk import FixedPaths, Position, expected_shortfall
from mst.portfolio.spec import Portfolio

DRAWS = 2000
"""The repetitions of one episode's predictive loss distribution."""
EPISODES = 200
SEEDS = tuple(range(20))
"""Enough seeds that the mean of Z is tight: one episode's contribution fires rarely."""
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


def tail_rows(
    rng, *, episodes: int = 60, draws: int = 2000, compress: float = 1.0
) -> list[dict]:
    """Episode rows whose realized loss is drawn from the predictive law itself.

    With ``compress`` below one the predictive tail is understated: its losses are squeezed
    towards the centre, as if the forecast had been compressed, while the facts keep their law.
    """
    rows = []
    for number in range(episodes):
        predictive = np.clip(rng.standard_normal(draws) * compress, -9.0, 9.0)
        realized = float(np.clip(rng.standard_normal(), -9.0, 9.0))
        rows.append(
            {
                "id": f"e{number}",
                "as_of": date(2026, 9, 30),
                "known_on": CHECKED_ON,
                "horizon": 1,
                "months": [date(2026, 10, 1)],
                "paths": loss_paths(predictive),
                "realized": loss_paths([realized]),
                "drivers": DRIVERS,
                "served": SERVED,
            }
        )
    return rows


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



def terms(rng, *, episodes: int = EPISODES, compress: float = 1.0):
    """(loss, VaR, ES) of every episode: the realized loss from the predictive law itself.

    With ``compress`` below one the predictive tail is understated — VaR and ES are both
    scaled down, as if the forecast had been squeezed towards the centre.
    """
    predictive = rng.standard_normal((episodes, DRAWS))
    losses = rng.standard_normal(episodes)
    var = np.array([value_at_risk(row) for row in predictive]) * compress
    es = np.array([expected_shortfall(row) for row in predictive]) * compress
    return list(zip(losses.tolist(), var.tolist(), es.tolist(), strict=True))


def test_a_calibrated_tail_averages_zero_over_seeds():
    # One episode's contribution is zero with probability 0.95 and ~20 with probability 0.05,
    # so a single Z wanders by ~0.3/sqrt(episodes); the mean over seeds reads the bias only.
    found = [z_statistic(terms(np.random.default_rng(seed))) for seed in SEEDS]
    assert abs(float(np.mean(found))) < 0.15
    assert all(abs(z) < 1.0 for z in found)


def test_a_tail_compressed_towards_the_centre_reads_positive():
    zs = [z_statistic(terms(np.random.default_rng(seed), compress=0.5)) for seed in SEEDS[:5]]
    assert all(z > 0.5 for z in zs)


def test_a_loss_exactly_at_var_contributes_nothing():
    # The comparison is strict: l_e > VaR_e, not >=.
    assert contributions([(5.0, 5.0, 1.0)]) == [0.0]
    assert z_statistic([(5.0, 5.0, 1.0)]) == pytest.approx(-1.0)
    assert z_statistic([(5.0, 4.0, 1.0)]) == pytest.approx(99.0)


def test_z_is_the_mean_of_the_contributions_less_one():
    found = terms(np.random.default_rng(7))
    assert z_statistic(found) == pytest.approx(np.mean(contributions(found)) - 1.0)


def test_trimming_the_heaviest_draws_lowers_var_and_es():
    # A 1 % cut of 200 draws drops exactly the two heaviest losses of the tail.
    predictive = [np.arange(200.0), np.arange(200.0) * 2.0]
    trimmed = trim_terms([7.0, 9.0], predictive, share=0.01)
    assert trimmed[0][0] == 7.0 and trimmed[1][0] == 9.0  # the realized losses are untouched
    for (loss, var, es), draws in zip(trimmed, predictive, strict=True):
        kept = draws[:-2]
        # ES of the kept sample at level 0.95, its weighted upper tail — not the mean of kept.
        assert (loss, var, es) == (
            loss,
            value_at_risk(kept),
            expected_shortfall(kept),
        )
        assert value_at_risk(kept) < value_at_risk(draws)
        assert expected_shortfall(kept) < expected_shortfall(draws)


def test_trimming_below_one_repetition_keeps_the_terms():
    # Fewer than 1/share draws: nothing is cut, the documented boundary — the trimmed
    # VaR and ES equal the untrimmed ones.
    draws = np.arange(99.0)
    found = trim_terms([3.0], [draws], share=0.01)
    assert found == [(3.0, value_at_risk(draws), expected_shortfall(draws))]
    with pytest.raises(ValueError, match="share"):
        trim_terms([1.0], [draws], share=0.0)


def test_value_at_risk_is_the_quantile_of_the_losses():
    grid = np.arange(101, dtype=float)
    assert value_at_risk(grid) == pytest.approx(95.0)
    assert value_at_risk(grid, level=0.9) == pytest.approx(90.0)
    with pytest.raises(ValueError, match="no repetition"):
        value_at_risk(np.empty(0))


def test_bad_terms_are_refused_by_name():
    with pytest.raises(ValueError, match="no episodes"):
        z_statistic([])
    with pytest.raises(ValueError, match="episode 1"):
        z_statistic([(1.0, 1.0, 1.0), (1.0, 1.0, 0.0)])
    with pytest.raises(ValueError, match="episode 0"):
        z_statistic([(np.nan, 1.0, 1.0)])
    with pytest.raises(ValueError, match="triple"):
        z_statistic([(1.0, 1.0), (2.0, 2.0)])


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


def test_portfolio_losses_are_minus_the_calculated_value_changes():
    losses = np.array([1.0, -2.0, 3.5])
    found = portfolio_losses(
        PORTFOLIO, loss_paths(losses), drivers=DRIVERS, served=SERVED, as_of=date(2026, 9, 30)
    )
    assert found == pytest.approx(losses)


def test_a_position_that_cannot_be_calculated_is_named_and_left_out():
    portfolio = Portfolio(
        positions=[
            Position(id="sber", type="common_stock", notional=10.0, codes={"secid": "SBER"}),
            Position(id="other", type="common_stock", notional=1.0, codes={"secid": "NOWHERE"}),
        ]
    )
    paths = loss_paths([1.0, 2.0])
    batch = portfolio_run(
        portfolio, paths, drivers=DRIVERS, served=SERVED, as_of=date(2026, 9, 30)
    )
    assert portfolio_exclusions(batch) == [
        {"position": "other", "reason": "the run serves no share_price for secid NOWHERE"}
    ]
    assert portfolio_losses(
        portfolio, paths, drivers=DRIVERS, served=SERVED, as_of=date(2026, 9, 30)
    ) == pytest.approx([1.0, 2.0])
    only_bad = Portfolio(
        positions=[Position(id="other", type="common_stock", notional=1.0, codes={"secid": "X"})]
    )
    assert portfolio_losses(
        only_bad, paths, drivers=DRIVERS, served=SERVED, as_of=date(2026, 9, 30)
    ).size == 0


def test_a_calibrated_fake_plan_averages_zero_over_seeds():
    # One episode's contribution is zero with probability 0.95 and ~20 with probability 0.05,
    # so a single Z wanders by ~0.3/sqrt(episodes); the mean over seeds reads the bias only.
    found = [
        tail_section(
            tail_rows(np.random.default_rng(seed), episodes=200), PORTFOLIO, seed=seed
        )["z"]
        for seed in range(20)
    ]
    assert abs(float(np.mean(found))) < 0.25
    assert all(abs(z) < 1.5 for z in found)


def test_a_compressed_forecast_reads_positive():
    found = tail_section(
        tail_rows(np.random.default_rng(1), episodes=200, compress=0.5), PORTFOLIO, seed=1
    )
    assert found["z"] > 0.5


def test_trimming_the_heaviest_paths_raises_z():
    rows = tail_rows(np.random.default_rng(3), episodes=200, draws=1000)
    found = tail_section(rows, PORTFOLIO, seed=5)
    assert found["trimmed_z"] > found["z"]


def test_missing_facts_and_uncalculated_positions_drop_episodes_by_name():
    good = tail_rows(np.random.default_rng(1), episodes=1)[0]
    no_fact = {
        **good,
        "id": "no-fact",
        "realized": None,
        "realized_missing": ["share_price:sber/2026-11"],
    }
    unserved = {**good, "id": "unserved", "served": ["bond_spread:corp"]}
    found = tail_section([good, no_fact, unserved], PORTFOLIO, seed=2)
    dropped = {item["id"]: item["reason"] for item in found["dropped"]}
    assert set(dropped) == {"no-fact", "unserved"}
    assert "share_price:sber/2026-11" in dropped["no-fact"]
    assert "sber" in dropped["unserved"] and "share_price" in dropped["unserved"]
    assert [episode["id"] for episode in found["episodes"]] == ["e0"]


def test_an_episode_without_paths_is_dropped_by_name():
    good = tail_rows(np.random.default_rng(1), episodes=1)[0]
    row = {
        **good,
        "paths": None,
        "realized": None,
        "realized_missing": ["no satellite bundle on the episode"],
    }
    found = tail_section([row], PORTFOLIO, seed=0)
    assert found["episodes"] == []
    assert found["z"] is None
    assert found["dropped"][0]["id"] == good["id"]
    assert "no paths for the episode" in found["dropped"][0]["reason"]
    assert "no satellite bundle" in found["dropped"][0]["reason"]


def test_a_section_with_no_judged_episode_is_stated_not_raised():
    good = tail_rows(np.random.default_rng(1), episodes=1)[0]
    unserved = {**good, "served": ["bond_spread:corp"]}
    found = tail_section([unserved], PORTFOLIO, seed=0)
    assert found["z"] is None
    assert found["interval"] is None
    assert found["wild"] is None
    assert found["trimmed_z"] is None
    assert found["episodes"] == []
    assert [item["id"] for item in found["dropped"]] == [good["id"]]


def test_the_horizon_is_marked():
    found = tail_section(tail_rows(np.random.default_rng(4), episodes=3), PORTFOLIO, seed=0)
    assert found["horizon"]["kind"] == "episode window"
    assert found["horizon"]["month"] == 1
    assert found["horizon"]["note"]


def test_the_same_seed_gives_the_same_numbers():
    rows = tail_rows(np.random.default_rng(6), episodes=20, draws=500)
    one = tail_section(rows, PORTFOLIO, seed=7)
    two = tail_section(rows, PORTFOLIO, seed=7)
    keys = ("z", "interval", "wild", "wild_disagrees", "trimmed_z", "trim", "meta", "horizon")
    assert {key: one[key] for key in keys} == {key: two[key] for key in keys}
    assert one["episodes"] == two["episodes"]

