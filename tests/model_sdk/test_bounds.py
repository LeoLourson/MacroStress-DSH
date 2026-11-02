"""The lower bounds of the macro variables: the shadow value and the observed one."""

from datetime import date

import numpy as np
import pytest

from mst.macro.registry import Registry, Variable, load_registry
from mst.model_sdk.bounds import clip_lower, lower_bounds, observed
from mst.model_sdk.contracts import MacroSimulation


def variable(id_, transform="identity", lower=None):
    return Variable.model_validate(
        {
            "id": id_,
            "label": "l",
            "unit": "%",
            "transform": transform,
            "source": "s",
            "lower": lower,
        }
    )


def test_the_floor_is_read_in_the_model_scale():
    known = Registry(
        variables=[
            variable("ru.key_rate", lower=0.0),
            variable("ru.usdrub", transform="log100", lower=100.0),
            variable("gl.vix", transform="log100", lower=0.0),
            variable("ru.b"),
        ]
    )
    bounds = lower_bounds(known, ["ru.key_rate", "ru.usdrub", "gl.vix", "ru.b", "ru.nope"])
    assert bounds[0] == 0.0
    # A log100 floor above zero maps into the model scale; one at zero bounds nothing.
    assert bounds[1] == pytest.approx(100.0 * np.log(100.0))
    assert bounds[2] == -np.inf
    assert bounds[3] == -np.inf  # no floor
    assert bounds[4] == -np.inf  # unknown id


def test_no_floor_anywhere_is_none():
    assert lower_bounds(Registry(variables=[variable("ru.b")]), ["ru.b", "ru.c"]) is None
    assert lower_bounds(Registry(variables=[variable("ru.key_rate", lower=0.0)]), ["ru.b"]) is None


def test_the_clip_reads_each_column_through_its_floor():
    paths = np.array([[[2.0, -1.0], [-3.0, 0.5]]])
    bounds = np.array([0.0, -np.inf])
    assert np.array_equal(clip_lower(paths, bounds), [[[2.0, -1.0], [0.0, 0.5]]])


def test_without_bounds_the_paths_are_their_own():
    paths = np.array([[[2.0, -1.0]]])
    assert clip_lower(paths, None) is paths


def test_the_shipped_registry_bounds_the_spread_whose_history_stays_above_zero():
    known = {variable.id: variable for variable in load_registry().variables}
    assert known["ru.spread_bbb"].lower == 0.0
    assert known["ru.spread_bbb"].negative is False
    # The spreads whose history went below zero stay unbounded.
    assert known["ru.spread_aa"].lower is None and known["ru.spread_aa"].negative
    assert known["ru.spread_a"].lower is None and known["ru.spread_a"].negative


def test_observed_reads_a_simulation_through_the_floors_of_the_registry():
    registry = Registry(variables=[variable("ru.key_rate", lower=0.0), variable("ru.b")])
    shadow = MacroSimulation(
        ids=["ru.key_rate", "ru.b"],
        months=[date(2020, 1, 1)],
        paths=np.array([[[-3.0, -3.0]]]),
        diagnostics={"kept": 1},
    )
    found = observed(shadow, registry)
    assert np.array_equal(found.paths, [[[0.0, -3.0]]])
    assert found.diagnostics == {"kept": 1}
    assert np.array_equal(shadow.paths, [[[-3.0, -3.0]]])  # the shadow stays what it was
