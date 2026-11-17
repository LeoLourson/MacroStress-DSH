"""mst.metrics.sanity: the per-quantity checks read before any score."""

import pytest

from mst.metrics.baselines import DOMAIN, MOVES
from mst.metrics.sanity import check


def row(
    quantity,
    month=1,
    *,
    unit="E1",
    level="macro",
    model=1.0,
    walk=1.0,
    outside=False,
    width=1.0,
    moves=10.0,
    shift=None,
    miss=None,
    base_width=1.0,
):
    return {
        "episode": unit,
        "month": month,
        "quantity": quantity,
        "level": level,
        "model": model,
        "walk": walk,
        "width": width,
        "move_range": moves,
        DOMAIN: outside,
        MOVES: False,
        "shift": shift,
        "miss": miss,
        "base_width": base_width,
    }


def test_the_width_is_read_against_the_range_of_the_moves():
    rows = [row("a", m, width=w, moves=10.0) for m, w in enumerate([5.0, 20.0, 8.0], 1)]
    assert check(rows)["quantities"][0]["width_to_moves"] == pytest.approx(0.8)


def test_the_direction_counts_only_the_moves_above_the_noise():
    rows = [
        row("a", 1, shift=0.5, miss=2.0),  # towards the fact
        row("a", 2, shift=-0.5, miss=2.0),  # away from it
        row("a", 3, shift=0.6, miss=1.0),  # towards
        row("a", 4, shift=0.01, miss=-3.0),  # below a tenth of the base band: noise
    ]
    direction = check(rows)["quantities"][0]["direction"]
    assert direction == {"moved": 3, "cells": 4, "toward": pytest.approx(2 / 3)}


def test_the_rolling_grid_has_no_direction():
    assert check([row("a")])["quantities"][0]["direction"] is None


def test_the_score_without_a_quantity_shows_what_it_carries():
    rows = [row("a", m, model=1.0, walk=1.0) for m in (1, 2)]
    rows += [row("b", m, model=4.0, walk=1.0) for m in (1, 2)]

    entries = {entry["quantity"]: entry for entry in check(rows)["quantities"]}

    assert entries["b"]["score"] == pytest.approx(4.0)
    assert entries["a"]["without"] == pytest.approx(4.0)
    assert entries["b"]["without"] == pytest.approx(1.0)


def test_the_levels_are_judged_apart():
    rows = [row("a", level="macro"), row("x", level="driver/share_price", model=2.0)]
    entries = check(rows)["quantities"]
    assert [(entry["level"], entry["quantity"]) for entry in entries] == [
        ("driver/share_price", "x"),
        ("macro", "a"),
    ]
    assert all(entry["without"] is None for entry in entries)
