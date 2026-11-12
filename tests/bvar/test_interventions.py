"""The default BVAR form conditions pre-intervention paths; the core changes cells once."""

from dataclasses import replace

import numpy as np
import pytest

from mst.model_sdk.contracts import Intervention, LinearConstraint, LinearTerm, ScenarioPlan
from mst.model_sdk.dist import ContradictsModels
from mst.model_sdk.interventions import apply_default
from mst.plugins.bvar.forecast import constraint_law, simulate
from mst.plugins.bvar.model import BvarModel
from tests.bvar.test_bounds import bounded


def observation(month, low, high):
    return LinearConstraint((LinearTerm("ru.key_rate", month, 1.0),), low, high)


def test_default_form_is_not_recursive():
    assert BvarModel().structural_form(bounded(0.0)) is None


def test_shift_conditions_each_group_before_core_applies_it_once():
    intervention = Intervention("ru.key_rate", (1,), "shift", np.array([[2.0], [-3.0]]))
    plan = ScenarioPlan(3, (observation(1, 10.0, 10.0),), (intervention,), inner=2)
    result = simulate(bounded(0.0), plan, draws=2, seed=8, candidates=2)
    assert np.array_equal(result.group, [0, 0, 1, 1])
    assert np.allclose(result.paths[:, 1, 0], [8.0, 8.0, 13.0, 13.0])
    actual = apply_default(result.paths, result.ids, plan.interventions, inner=plan.inner)
    assert np.allclose(actual[:, 1, 0], 10.0)
    assert np.array_equal(actual[:, 2:], result.paths[:, 2:])


def test_set_removes_satisfied_observation_without_conditioning_noise():
    intervention = Intervention("ru.key_rate", (1,), "set", np.array([[20.0], [21.0]]))
    plan = ScenarioPlan(3, (observation(1, 19.0, 22.0),), (intervention,))
    result = simulate(bounded(0.0), plan, draws=2, seed=8, candidates=2)
    plain = simulate(bounded(0.0), ScenarioPlan(3), draws=2, seed=8, candidates=2)
    assert result.paths.tobytes() == plain.paths.tobytes()
    actual = apply_default(result.paths, result.ids, plan.interventions)
    assert np.array_equal(actual[:, 1, 0], [20.0, 21.0])


def test_false_set_observation_names_original_row_and_group():
    intervention = Intervention("ru.key_rate", (1,), "set", np.array([[20.0], [21.0]]))
    plan = ScenarioPlan(3, (observation(1, 19.0, 20.5),), (intervention,))
    with pytest.raises(ContradictsModels) as caught:
        simulate(bounded(0.0), plan, draws=2, seed=8, candidates=2)
    assert caught.value.constraints == (1,)
    assert caught.value.group == 1


def test_default_law_retains_original_rows_offsets_and_box():
    interventions = (
        Intervention("ru.key_rate", (1,), "set", np.array([[20.0], [21.0]])),
        Intervention("ru.key_rate", (2,), "shift", np.array([[2.0], [-3.0]])),
    )
    constraints = (
        observation(1, 19.0, 22.0),
        LinearConstraint(
            (LinearTerm("ru.key_rate", 1, 2.0), LinearTerm("ru.key_rate", 2, 1.0)),
            0.0,
            100.0,
        ),
    )
    law = constraint_law(
        bounded(0.0), ScenarioPlan(3, constraints, interventions), draws=2, seed=8, candidates=2
    )
    assert np.array_equal(law.group, [0, 0, 1, 1])
    assert np.array_equal(law.means, [[20.0, 46.0], [20.0, 46.0], [21.0, 43.0], [21.0, 43.0]])
    assert np.array_equal(law.covs, np.tile([[0.0, 0.0], [0.0, 2.0]], (4, 1, 1)))


def test_inactive_cell_retains_its_unconditioned_law():
    intervention = Intervention(
        "ru.key_rate", (1,), "set", np.array([[20.0], [21.0]]), np.array([[True], [False]])
    )
    law = constraint_law(
        bounded(0.0),
        ScenarioPlan(2, (observation(1, 0.0, 30.0),), (intervention,)),
        draws=2,
        seed=8,
        candidates=1,
    )
    assert np.array_equal(law.means[:, 0], [20.0, 4.0])
    assert np.array_equal(law.covs[:, 0, 0], [0.0, 1.0])


@pytest.mark.parametrize("method", [simulate, constraint_law])
def test_group_count_must_match_intervention_values(method):
    intervention = Intervention("ru.key_rate", (1,), "shift", np.zeros((1, 1)))
    with pytest.raises(ValueError, match="group"):
        method(
            bounded(0.0),
            ScenarioPlan(2, (observation(1, 0.0, 30.0),), (intervention,)),
            draws=2,
            seed=8,
            candidates=1,
        )


def test_actual_stored_lag_count_is_used_with_default_intervention():
    model = bounded(0.0)
    model = replace(
        model,
        lags_count=2,
        lags=np.array([[[[1.5]], [[-0.5]]]]),
        history=np.array([[[2.0], [4.0]]]),
    )
    intervention = Intervention("ru.key_rate", (1,), "shift", np.array([[2.0]]))
    law = constraint_law(
        model,
        ScenarioPlan(2, (observation(2, 0.0, 30.0),), (intervention,)),
        draws=1,
        seed=8,
        candidates=1,
    )
    assert np.array_equal(law.means, [[5.5]])
    assert np.array_equal(law.covs, [[[3.25]]])
