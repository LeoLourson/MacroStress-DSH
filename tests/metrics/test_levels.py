"""mst.metrics.levels: the cells of every level and the rows the scores and the sanity read."""

import numpy as np
import pytest

from mst.drivers.registry import Quantity
from mst.macro.registry import load_registry
from mst.metrics import levels
from mst.metrics.baselines import DOMAIN, WIDTH, macro_domain, quantity_domain
from mst.metrics.cells import Cell
from mst.metrics.sanity import check
from mst.model_sdk.contracts import Intervention, LinearConstraint, LinearTerm


def rows_of(found):
    return levels.rows(found, "episode", {cell.unit: "tuning" for cell in found})


def test_the_macro_domain_bounds_rates_and_leaves_spreads_and_logs_open():
    registry = load_registry()
    assert macro_domain(registry.variable("ru.key_rate")) == (0.0, np.inf)
    # A spread has been below zero in the history, and 100 ln X is any real number.
    assert macro_domain(registry.variable("ru.spread_aa")) is None
    assert macro_domain(registry.variable("ru.usdrub")) is None
    cell = Cell(
        "e",
        1,
        "ru.key_rate",
        0.0,
        np.array([-10.0, -2.0, 0.0, 1.0]),
        history_range=100.0,
        domain=macro_domain(registry.variable("ru.key_rate")),
    )
    (row,) = rows_of([cell])
    assert row[DOMAIN] is True and row[WIDTH] is False
    assert row["half"] == "tuning" and row["level"] == "macro"


def test_the_driver_domain_maps_the_natural_endpoints():
    def quantity(**fields):
        return Quantity(id="q", label="q", unit="u", regressors=["ru.cpi"], lags=1, **fields)

    # A price of 0.05 is 100 ln 0.05 = -300 in the model scale: only a price of nought is
    # out of bounds, and it maps to minus infinity.
    assert quantity_domain(quantity(transform="log100")) is None
    assert quantity_domain(quantity(transform="log100", domain=[0.0, 1e6]))[0] == -np.inf
    assert quantity_domain(quantity(transform="log100", domain=[0.0, 1e6]))[1] == (
        pytest.approx(100 * np.log(1e6))
    )
    # A logit declared on [0, 1] is unbounded in the model scale, not silently "no gate".
    assert quantity_domain(quantity(transform="logit", domain=[0.0, 1.0])) is None
    low, high = quantity_domain(quantity(transform="logit", domain=[0.01, 0.99]))
    assert low == pytest.approx(np.log(0.01 / 0.99))
    assert high == pytest.approx(np.log(0.99 / 0.01))
    assert quantity_domain(quantity(transform="identity", domain=[0.0, 100.0])) == (0.0, 100.0)


def test_the_admission_rules_are_read_apart_for_every_quantity():
    def cell(month, draws, quantity="q"):
        return Cell(
            "e",
            month,
            quantity,
            0.0,
            np.asarray(draws, dtype=float),
            history_range=10.0,
            domain=(0.0, np.inf),
        )

    def admission(found):
        return check(rows_of(found))["admission"]

    domain_cells = [cell(month, [-2.0, -1.0, 0.0, 1.0]) for month in range(1, 5)]
    width_cells = [cell(month, [0.0, 0.0, 100.0, 100.0]) for month in range(1, 5)]
    fine = [cell(month, [1.0, 2.0, 3.0, 4.0]) for month in range(1, 93)]
    # Four cells break each rule on its own: the union is 8 %, but neither rule reaches 5 %.
    found = admission(domain_cells + width_cells + fine)
    assert found["cells"] == 100
    assert found["rules"][DOMAIN]["worst"] == pytest.approx(0.04)
    assert found["rules"][WIDTH]["worst"] == pytest.approx(0.04)
    assert found["failed"] is False
    # Six cells breaking the width rule fail that rule and the version.
    extra = [cell(month, [0.0, 0.0, 100.0, 100.0]) for month in range(1, 3)]
    failed = admission(domain_cells + width_cells + extra + fine)
    assert failed["rules"][WIDTH]["worst"] == pytest.approx(6 / 102)
    assert failed["rules"][WIDTH]["quantities"] == ["q"]
    assert failed["rules"][DOMAIN]["quantities"] == []
    assert failed["failed"] is True
    # The threshold is strict: exactly five percent does not pass ("fewer than 5 %").
    exact = [cell(month, [0.0, 0.0, 100.0, 100.0]) for month in range(1, 6)]
    enough = [cell(month, [1.0, 2.0, 3.0, 4.0]) for month in range(1, 96)]
    assert admission(exact + enough)["failed"] is True
    # The same broken cells beside a quiet second quantity still fail: the pool would hide them.
    other = [cell(month, [1.0, 2.0, 3.0, 4.0], quantity="r") for month in range(1, 200)]
    hidden = admission(width_cells[:3] + enough[:40] + other)
    assert hidden["rules"][WIDTH]["pooled"] < 0.05
    assert hidden["rules"][WIDTH]["quantities"] == ["q"]


def test_a_row_carries_every_comparison_and_the_move_of_the_scenario():
    draws = np.linspace(0.0, 4.0, 101)
    cell = Cell(
        "e",
        2,
        "q",
        3.0,
        draws,
        baseline=draws + 1.0,
        base=draws - 1.0,
        conditional=draws,
        history_range=10.0,
    )
    (row,) = rows_of([cell])
    assert row["model"] == pytest.approx(row["conditional_walk"])
    assert row["walk"] is not None and row["base"] is not None and row["naive"] is None
    # The scenario moved the median from 1 (the base run) to 2, towards the fact at 3.
    assert row["shift"] == pytest.approx(1.0)
    assert row["miss"] == pytest.approx(2.0)
    assert row["base_width"] == pytest.approx(3.6)


def test_a_constrained_or_non_finite_cell_is_never_scored():
    told = Cell("e", 1, "q", 1.0, np.ones(4), constrained=True)
    lost = Cell("e", 2, "q", 1.0, np.full(4, np.nan))
    assert rows_of([told, lost]) == []
    assert levels.non_finite([told, lost]) == {"q": 1}


def test_every_point_pins_its_cell_whatever_its_rounding():
    rng = np.random.default_rng(3)
    term = LinearTerm("x", 1, 1.0)
    for value, tolerance in zip(rng.uniform(0, 20, 500), rng.uniform(1e-4, 0.1, 500), strict=True):
        point = LinearConstraint(
            terms=(term,),
            low=value - tolerance,
            high=value + tolerance,
            point=True,
            tolerance=tolerance,
        )
        assert levels._pinned([point]) == {("x", 1)}
    wide = LinearConstraint(terms=(term,), low=0.0, high=1.0, tolerance=0.01)
    assert levels._pinned([wide]) == set()


def test_an_intervention_pins_the_cells_it_sets():
    item = Intervention(target="ru.key_rate", months=(1, 3), kind="set", values=np.ones((2, 2)))
    assert levels._installed([item]) == {("ru.key_rate", 1), ("ru.key_rate", 3)}
