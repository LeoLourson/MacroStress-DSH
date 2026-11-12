from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest

from mst.model_sdk.contracts import ConstraintLaw, MacroSimulation, StructuralForm
from mst.model_sdk.dist import ContradictsModels
from mst.model_sdk.interventions import apply_default
from mst.scenario.build import build, dist_of
from mst.scenario.context import context_from_versions
from mst.scenario.contracts import Scenario
from tests.scenario.test_any_model import DefaultModel, Satellites


def macro(plugin):
    panel = SimpleNamespace(
        ids=["ru.key_rate"],
        y=np.array([[1.0], [np.nan], [2.0]]),
        months=[date(2026, 8, 1), date(2026, 9, 1), date(2026, 10, 1)],
    )
    return SimpleNamespace(panel=panel, ids=panel.ids, plugin=plugin, model=None)


def scenario(kind, amount, constraint=None):
    value = {
        "narrative": "test",
        "interventions": [
            {
                "quantity": "ru.key_rate",
                "months": ["2026-11"],
                "kind": kind,
                "values": [amount],
                "unit": "level",
                "reason": "test",
                "basis": "event",
            }
        ],
    }
    if constraint:
        value["observations"] = [
            {
                "kind": "constraint",
                "terms": [{"quantity": "ru.key_rate", "month": "2026-11", "weight": 1}],
                "unit": "level",
                "low": constraint[0],
                "high": constraint[1],
                "basis": "event",
            }
        ]
    return Scenario.model_validate(value)


class Recursive(DefaultModel):
    def structural_form(self, model):
        return StructuralForm(("ru.key_rate",), True)

    def simulate(self, model, plan, *, draws, seed):
        paths = np.ones((draws * plan.inner, plan.horizon + 1, 1))
        paths = apply_default(paths, ["ru.key_rate"], plan.interventions, inner=plan.inner)
        return MacroSimulation(
            ["ru.key_rate"],
            [date(2026, 10, 1)] * (plan.horizon + 1),
            paths,
            group=np.repeat(np.arange(draws), plan.inner),
        )


def test_recursive_model_applies_intervention_only_once():
    result = build(
        scenario("shift", 2), macro(Recursive()), Satellites(), draws=2, seed=4, horizon=1, inner=2
    )
    assert np.all(result.shadow[:, 1, 0] == 3)


def test_set_outside_constraint_is_contradiction():
    with pytest.raises(ContradictsModels):
        build(
            scenario("set", 8, (2, 4)),
            macro(DefaultModel()),
            Satellites(),
            draws=2,
            seed=4,
            horizon=1,
        )


def test_context_keeps_exact_history_and_last_finite_value():
    data = context_from_versions(macro(DefaultModel()))
    assert data.origin == "2026-10"
    assert data.last_model["ru.key_rate"] == 2
    np.testing.assert_equal(data.history["ru.key_rate"], [1, np.nan, 2])


def test_law_receives_original_bounds_and_intervention():
    class LawModel(DefaultModel):
        def constraint_law(self, model, plan, *, draws, seed):
            assert plan.constraints[0].low == 2
            assert plan.constraints[0].high == 4
            assert plan.interventions[0].values[0, 0] == 2
            return ConstraintLaw(
                np.full((draws, 1), 3.0), np.ones((draws, 1, 1)), np.zeros(draws), np.arange(draws)
            )

    result = dist_of(scenario("shift", 2, (2, 4)), macro(LawModel()), draws=2, seed=4, horizon=1)
    assert result is not None
    assert result.distance > 0


def random_scenario():
    """A shift imposed with probability one half, drawn once per group."""
    return Scenario.model_validate(
        {
            "narrative": "test",
            "interventions": [
                {
                    "quantity": "ru.key_rate",
                    "months": ["2026-11"],
                    "kind": "shift",
                    "values": [{"with_probability": 0.5, "value": 2.0}],
                    "unit": "level",
                    "reason": "test",
                    "basis": "assumption",
                }
            ],
        }
    )


def test_a_probabilistic_intervention_moves_only_its_active_groups():
    """The draw decides per group, so the paths split into the shifted and the untouched ones."""
    groups = 400
    result = build(
        random_scenario(), macro(DefaultModel()), Satellites(), draws=groups, seed=9, horizon=1
    )
    installed = result.macro[:, 1, 0]
    assert np.isin(installed, (1.0, 3.0)).all(), "a path is neither shifted nor untouched"
    assert float(np.mean(installed == 3.0)) == pytest.approx(0.5, abs=0.06)


def test_a_uniform_on_one_value_is_the_deterministic_shift_bit_for_bit():
    def uniform():
        return Scenario.model_validate(
            {
                "narrative": "test",
                "interventions": [
                    {
                        "quantity": "ru.key_rate",
                        "months": ["2026-11"],
                        "kind": "shift",
                        "values": [{"uniform": [2.0, 2.0]}],
                        "unit": "level",
                        "reason": "test",
                        "basis": "assumption",
                    }
                ],
            }
        )

    drawn = build(uniform(), macro(DefaultModel()), Satellites(), draws=8, seed=4, horizon=1)
    certain = build(
        scenario("shift", 2), macro(DefaultModel()), Satellites(), draws=8, seed=4, horizon=1
    )
    assert np.array_equal(drawn.macro, certain.macro)
    assert np.array_equal(drawn.shadow, certain.shadow)
