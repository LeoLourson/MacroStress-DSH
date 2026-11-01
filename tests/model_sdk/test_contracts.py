"""mst.model_sdk.contracts: the neutral structures the model plugins speak."""

import inspect
from datetime import date

import numpy as np
import pytest

from mst.model_sdk import (
    BUNDLE_META_REQUIRED,
    MACRO_ARRAYS_REQUIRED,
    MACRO_META_REQUIRED,
    MODEL_SDK_VERSION,
    PART_META_REQUIRED,
    AdditiveImpact,
    DriverPaths,
    LinearConstraint,
    LinearTerm,
    MacroModel,
    MacroSimulation,
    SatelliteModel,
    ScenarioPlan,
    model_compatible,
)


def test_a_plan_holds_constraints_and_causes():
    plan = ScenarioPlan(
        horizon=3,
        constraints=(
            LinearConstraint(
                terms=(LinearTerm(variable="ru.key_rate", month=2, weight=1.0),),
                low=18.0,
                high=21.0,
            ),
        ),
        causes=("ru.rate_hike",),
    )
    assert plan.constraints[0].terms[0].variable == "ru.key_rate"
    assert plan.causes == ("ru.rate_hike",)
    assert ScenarioPlan(horizon=1).constraints == ()


def test_an_impact_names_its_target_months_and_delta():
    impact = AdditiveImpact(target="share_price:sber", start=1, end=6, delta=-10.0)
    assert impact.target == "share_price:sber"
    assert (impact.start, impact.end, impact.delta) == (1, 6, -10.0)


def test_a_macro_simulation_carries_its_history():
    simulation = MacroSimulation(
        ids=["gl.brent"],
        months=[date(2026, 9, 1), date(2026, 10, 1)],
        paths=np.zeros((4, 2, 1)),
        history=np.zeros((4, 1, 1)),
    )
    assert simulation.paths.shape == (4, 2, 1)
    assert simulation.history is not None


def test_the_abstract_models_demand_the_three_steps():
    class Bare(MacroModel):
        pass

    with pytest.raises(TypeError):
        Bare()  # estimate, load and simulate are abstract

    class BareSatellite(SatelliteModel):
        pass

    with pytest.raises(TypeError):
        BareSatellite()


def test_describe_names_the_kind_and_the_served_quantities():
    class One(SatelliteModel):
        model_id = "gold.sat.v1"
        version = "0.1.0"
        label = "Gold satellite"
        quantities = ("share_price",)

        def estimate(self, context): ...

        def load(self, parts): ...

        def simulate(self, model, macro, drivers, impacts, *, correlation=None, seed, stream):
            return DriverPaths(ids=[], paths=np.empty((0, 0, 0)), flags={})

    class Two(MacroModel):
        model_id = "gold.macro.v1"
        version = "0.1.0"
        label = "Gold macro"
        variables = ("gl.brent",)

        def estimate(self, context): ...

        def load(self, meta, arrays): ...

        def simulate(self, model, plan, *, draws, seed): ...

    assert One().describe() == {
        "model_id": "gold.sat.v1",
        "kind": "satellite",
        "version": "0.1.0",
        "label": "Gold satellite",
        "quantities": ["share_price"],
        "description": "",
    }
    assert Two().describe()["kind"] == "macro"
    assert Two().constraint_law(None, ScenarioPlan(horizon=1), draws=1, seed=0) is None


def test_the_version_contracts_name_the_required_keys():
    assert MACRO_META_REQUIRED == ("model", "as_of", "ids", "months", "shocks")
    assert MACRO_ARRAYS_REQUIRED == ("panel_y", "panel_approximate")
    assert BUNDLE_META_REQUIRED == ("macro", "as_of", "registry", "quantities", "correlated")
    assert PART_META_REQUIRED == ("model", "months", "nodes", "driver", "checks", "rows")


def test_the_shocks_argument_is_a_required_keyword_of_a_satellite_simulate():
    # The platform draws the joint shocks of every bundle, whatever its number of models, so
    # a model cannot stay on its own draw.
    parameter = inspect.signature(SatelliteModel.simulate).parameters["shocks"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty


def test_the_plugin_ranges_are_checked_against_the_model_sdk_version():
    assert MODEL_SDK_VERSION == "2.0.0"
    assert model_compatible(">=2.0,<3.0")
    assert model_compatible(">=2.0")
    assert not model_compatible(">=1.0,<2.0")
    assert not model_compatible(">=1.1,<2.0")
