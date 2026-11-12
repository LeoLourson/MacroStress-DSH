import numpy as np
import pytest

from mst.model_sdk.contracts import Intervention, LinearConstraint, LinearTerm, ScenarioPlan
from mst.model_sdk.dist import ContradictsModels
from mst.model_sdk.interventions import apply_default
from mst.plugins.expert.forecast import constraint_law, simulate
from tests.expert.test_subset import _linear_model


@pytest.mark.parametrize("kind", ["set", "shift"])
def test_group_interventions_condition_original_functional(kind):
    model = _linear_model(frequency=0, residual=1.0)
    constraint = LinearConstraint(
        (LinearTerm("ru.cpi", 1, 1), LinearTerm("ru.cpi", 2, 1)), 204.0, 204.2
    )
    values = [[101.0], [102.0]] if kind == "set" else [[0.5], [-0.5]]
    intervention = Intervention("ru.cpi", (1,), kind, np.array(values))
    plan = ScenarioPlan(2, (constraint,), (intervention,), inner=4)
    result = simulate(model, plan, draws=2, seed=6)
    observed = apply_default(result.paths, model.ids, plan.interventions, inner=4)
    assert not np.array_equal(result.paths[:, 1, 0], observed[:, 1, 0])
    values = observed[:, 1, 0] + observed[:, 2, 0]
    assert np.all((values >= 204.0 - 1e-8) & (values <= 204.2 + 1e-8))
    assert np.array_equal(result.group, np.repeat(np.arange(2), 4))
    law = constraint_law(model, plan, draws=2, seed=6)
    means = law.means.reshape(2, model.candidates, 1)[:, 0, 0]
    expected = [203.0, 204.0] if kind == "set" else [203.5, 202.5]
    np.testing.assert_allclose(means, expected)
    np.testing.assert_allclose(law.covs[:, 0, 0], 2.0 if kind == "set" else 5.0)
    assert (constraint.low, constraint.high) == (204.0, 204.2)


def test_set_constant_row_preserves_law_and_rejects_bad_group():
    model = _linear_model(frequency=0, residual=1.0)
    constraint = LinearConstraint((LinearTerm("ru.cpi", 1, 1),), 100.5, 101.5)
    intervention = Intervention("ru.cpi", (1,), "set", np.array([[101.0], [102.0]]))
    plan = ScenarioPlan(1, (constraint,), (intervention,))
    law = constraint_law(model, plan, draws=2, seed=2)
    np.testing.assert_allclose(law.means[:, 0], np.repeat([101.0, 102.0], model.candidates))
    np.testing.assert_allclose(law.covs, 0.0)
    with pytest.raises(ContradictsModels):
        simulate(model, plan, draws=2, seed=2)


def test_inactive_intervention_keeps_original_law():
    model = _linear_model(frequency=0, residual=1.0)
    constraint = LinearConstraint((LinearTerm("ru.cpi", 1, 1),), 100.5, 101.5)
    intervention = Intervention("ru.cpi", (1,), "set", np.array([[103.0]]), np.array([[False]]))
    plan = ScenarioPlan(1, (constraint,), (intervention,))
    law = constraint_law(model, plan, draws=1, seed=2)
    np.testing.assert_allclose(law.means, 101.0)
    np.testing.assert_allclose(law.covs, 1.0)
    result = simulate(model, plan, draws=1, seed=2)
    assert 100.5 <= result.paths[0, 1, 0] <= 101.5
