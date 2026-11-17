"""mst.metrics.trust and mst.metrics.trial: the criteria of trust and their trial."""

from dataclasses import replace

import numpy as np

from mst.metrics.baselines import DOMAIN, MOVES
from mst.metrics.trial import DEFECTS, World, trial
from mst.metrics.trust import admission, decision, verdict


def grid(*, quantities=("a", "b", "c"), dates=24, months=6, seed=0, **change):
    """Grid rows of a calibrated forecast as good as the walk; ``change`` maps a quantity to a
    function that edits its row."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(dates):
        for t in range(1, months + 1):
            for q in quantities:
                level = rng.uniform(0.5, 1.5)
                row = {
                    "date": f"{u:03d}",
                    "half": "validation",
                    "month": t,
                    "quantity": q,
                    "level": "macro",
                    "model": level,
                    "walk": level * rng.uniform(0.98, 1.02),
                    "conditional_walk": None,
                    "base": None,
                    "pit": float(rng.uniform()),
                    DOMAIN: False,
                    MOVES: False,
                }
                if q in change:
                    change[q](row, rng)
                rows.append(row)
    return rows


def admit(rows):
    return admission(rows, unit="date", block=3, calibration=3)


def test_a_calibrated_forecast_as_good_as_the_walk_is_admitted():
    found = admit(grid())
    assert not found["failed"]
    assert set(found["criteria"]) == {"domain", "moves", "tails", "failing", "portfolio_tail"}
    assert not found["criteria"]["portfolio_tail"]["checked"]


def test_one_band_outside_the_domain_fails_the_domain():
    rows = grid()
    rows[5][DOMAIN] = True
    found = admit(rows)
    assert found["criteria"]["domain"]["failed"] == [f"macro/{rows[5]['quantity']}"]
    assert found["failed"]


def test_the_tails_name_the_quantity_and_the_side():
    def high(row, rng):
        row["pit"] = 0.99 if rng.uniform() < 0.3 else row["pit"]

    found = admit(grid(b=high))
    assert found["criteria"]["tails"]["failed"] == ["macro (above)"]


def test_a_quantity_much_worse_than_its_walk_fails_and_the_others_do_not():
    def worse(row, rng):
        row["model"] = 2.0 * row["walk"]

    found = admit(grid(c=worse))
    assert found["criteria"]["failing"]["failed"] == ["macro/c"]


def test_the_moves_rule_fails_a_quantity_broken_far_above_the_correct_share():
    def broken(row, rng):
        row[MOVES] = bool(rng.uniform() < 0.3)

    found = admit(grid(a=broken))
    assert found["criteria"]["moves"]["failed"] == ["macro/a"]


def test_the_episodes_read_neither_the_tails_nor_the_failing_quantity():
    rows = [{**row, "episode": row["date"]} for row in grid()]
    found = admission(rows, unit="episode", block=1, calibration=None)
    assert set(found["criteria"]) == {"domain", "moves"}


def episodes(ratio, *, units=16, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(units):
        for t in range(1, 7):
            base = rng.uniform(0.5, 1.5)
            rows.append(
                {
                    "episode": f"E{u}",
                    "half": "validation",
                    "month": t,
                    "quantity": "a",
                    "level": "macro",
                    "model": ratio * base * rng.uniform(0.95, 1.05),
                    "base": base,
                    "conditional_walk": base,
                }
            )
    return rows


def test_the_decision_reads_accuracy_harm_and_usefulness():
    found = decision(episodes(1.0), grid(), target="macro", grid_block=3)
    assert found["accuracy"]["passed"]
    assert found["harm"]["passed"]
    assert not found["usefulness"]["proven"]
    harmful = decision(episodes(1.5), grid(), target="macro", grid_block=3)
    assert not harmful["harm"]["passed"]
    useful = decision(episodes(0.7), grid(), target="macro", grid_block=3)
    assert useful["usefulness"]["proven"]

    def worse(row, rng):
        row["model"] = 1.3 * row["walk"]

    inaccurate = decision(
        [],
        grid(a=worse, b=worse, c=worse),
        target="macro",
        grid_block=3,
    )
    assert not inaccurate["accuracy"]["passed"]


def test_without_the_tail_of_the_portfolio_the_model_is_not_trusted():
    admitted = {"admission": admit(grid()), "dropped": {"refused": {}}}
    decided = decision(episodes(1.0), grid(), target="macro", grid_block=3)
    found = verdict({"grid": admitted}, decided)
    assert not found["trusted"]
    assert found["reasons"] == ["portfolio_tail: not checked"]


def test_the_correct_model_of_a_small_world_is_rarely_rejected():
    """The trial of the criteria: the correct model fails the grid's criteria in no more worlds
    than the error budget allows, with room for the sampling of twenty worlds."""
    small = replace(World(), quantities=4, dates=12, draws=200, repetitions=200)
    table = trial(small, ["correct"], worlds=20, seed=3)
    assert table["correct"]["any"] <= 0.25
    assert set(DEFECTS) >= {"correct", "narrow", "wide", "shifted", "thin", "failing"}


def test_an_understated_tail_of_the_portfolio_fails_the_admission():
    def tail(beyond_every):
        windows = [
            {"id": f"{u:03d}", "loss": 3.0 if u % beyond_every == 0 else 0.5, "var": 1.0, "es": 1.5}
            for u in range(24)
        ]
        return {"windows": windows}

    rows = grid()
    calibrated = admission(rows, unit="date", block=3, calibration=3, tail=tail(20))
    assert calibrated["criteria"]["portfolio_tail"]["checked"]
    assert calibrated["criteria"]["portfolio_tail"]["failed"] == []
    understated = admission(rows, unit="date", block=3, calibration=3, tail=tail(2))
    assert understated["criteria"]["portfolio_tail"]["beyond_var"] == 0.5
    assert understated["criteria"]["portfolio_tail"]["failed"] == ["z", "beyond_var"]
    assert understated["failed"]
