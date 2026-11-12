"""Satellite interventions are applied once, with one value per macro group."""

from dataclasses import replace

import numpy as np
import pytest

from mst.model_sdk import Check, DriverPaths, Intervention
from mst.model_sdk.contracts import UpperPaths
from mst.plugins.reflevel2.model import Fit, RefLevel2, RefLevel2Model
from tests.drivers.test_bundle import FakeSatellite, _bundle_of, macro_simulation, registry_of


def test_bundle_dispatches_grouped_set_and_shift_without_reapplying(tmp_path):
    one = FakeSatellite("one.a", quantities=("price",))
    two = FakeSatellite("two.b", quantities=("spread",))
    bundle = _bundle_of(tmp_path, [one, two], registry_of("price", "spread"))
    macro = replace(macro_simulation(months=3, draws=4), group=np.array([0, 0, 1, 1]))
    drivers = ["price:sector", "spread:sector"]
    plain = bundle.simulate(macro, drivers, seed=3, inner=2)
    interventions = [
        Intervention("price:sector", (1, 3), "set", np.array([[10, 11], [20, 21]])),
        Intervention("spread:sector", (2,), "shift", np.array([[3], [7]])),
    ]
    result = bundle.simulate(macro, drivers, interventions, seed=3, inner=2)
    np.testing.assert_array_equal(result.paths[:, 1, 0], [10, 10, 20, 20])
    np.testing.assert_array_equal(result.paths[:, 3, 0], [11, 11, 21, 21])
    np.testing.assert_array_equal(result.paths[:, 2, 0], plain.paths[:, 2, 0])
    np.testing.assert_array_equal(result.paths[:, 2, 1] - plain.paths[:, 2, 1], [3, 3, 7, 7])
    np.testing.assert_array_equal(result.paths[:, [0, 1, 3], 1], plain.paths[:, [0, 1, 3], 1])


def test_reference_model_reads_intervened_upper_paths_and_sets_own_cell():
    macro = replace(macro_simulation(months=2, draws=4), group=np.array([0, 0, 1, 1]))
    upper_paths = DriverPaths(ids=["price:sector"], paths=np.full((4, 3, 1), 3.0),
                              flags={"price:sector": frozenset()})
    upper = UpperPaths(macro=macro, levels={1: upper_paths})
    fit = Fit(months=macro.months, nodes=["sector"], driver=["exposure:sector"],
              rows=np.array([3]), beta=np.array([[[0.0, 2.0]]]), sigma=np.zeros((1, 1)),
              inputs={"sector": [("driver", "price:sector", 0.0)]},
              train_low=np.array([0.0]), train_high=np.array([20.0]), d=np.ones((3, 1)))
    model = RefLevel2(fits={"exposure": fit}, checks={"exposure": {0: Check(
        satellite=1.0, baseline=2.0, forecasts=3, served=True, reason="")}})
    intervention = Intervention("exposure:sector", (2,), "set", np.array([[11.0], [19.0]]))
    result = RefLevel2Model().simulate(model, upper, ["exposure:sector"], [intervention],
                                      shocks=np.zeros((4, 2, 1)), seed=1, stream=1)
    np.testing.assert_array_equal(result.paths[:, 1, 0], [6, 6, 6, 6])
    np.testing.assert_array_equal(result.paths[:, 2, 0], [11, 11, 19, 19])
    assert result.flags["exposure:sector"] == {"impact"}


def test_bundle_rejects_unknown_intervention_target(tmp_path):
    one = FakeSatellite("one.a", quantities=("price",))
    bundle = _bundle_of(tmp_path, [one], registry_of("price"))
    intervention = Intervention("unknown:x", (1,), "set", np.ones((3, 1)))
    with pytest.raises(ValueError, match="unknown:x"):
        bundle.simulate(macro_simulation(), ["price:sector"], [intervention], seed=1)


def test_bundle_rejects_noncontiguous_groups(tmp_path):
    one = FakeSatellite("one.a", quantities=("price",))
    bundle = _bundle_of(tmp_path, [one], registry_of("price"))
    macro = replace(macro_simulation(draws=4), group=np.array([0, 1, 0, 1]))
    with pytest.raises(ValueError, match="group"):
        bundle.simulate(macro, ["price:sector"], seed=1, inner=2)
