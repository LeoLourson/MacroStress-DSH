from dataclasses import replace

import numpy as np
import pytest

from mst.model_sdk.contracts import Intervention, LinearConstraint, LinearTerm, ScenarioPlan
from mst.model_sdk.dist import ContradictsModels
from mst.model_sdk.interventions import apply_default
from mst.plugins.msvar.forecast import constraint_law, simulate
from tests.msvar.test_bounds import bounded
from tests.msvar.test_forecast import toy_posterior


def test_legacy_unnamed_streams_are_exact():
    model = toy_posterior(
        np.array([[0.7, 0.3], [0.5, 0.5]]),
        np.array([[0.0, 0.0], [4.0, -6.0]]),
        np.zeros((2, 2)),
    )
    row = LinearConstraint((LinearTerm("v0", 1, 1.0),), 3.8, 4.6)
    expected = [
        [[[0., 0.],
            [0.4918415519133836, 0.7598779662749252],
            [0.394611565564678, -1.0708457209369002]],
            [[0., 0.],
            [3.226838795367249, -4.193775594493767],
            [3.1982344165332846, -7.121323627868002]],
            [[0., 0.],
            [0.22903967837896577, 2.1220062195054705],
            [3.4599576920757387, -7.063645554279542]]],
        [[[0., 0.],
            [4.132101133143338, -4.8635360379197214],
            [0.28229915805186395, 1.1128426890886032]],
            [[0., 0.],
            [3.8325771646358837, -5.378483198707236],
            [4.322039007940468, -5.15947380348839]],
            [[0., 0.],
            [3.847256177891328, -6.397680746137081],
            [-0.2562351685792245, 0.428339234536836]]],
    ]
    for constraints, values in zip(((), (row,)), expected, strict=True):
        actual = simulate(model, ScenarioPlan(2, constraints), draws=3, seed=6, candidates=5)
        assert np.array_equal(actual.paths, np.array(values))


def test_group_set_observation_and_affine_law():
    model = bounded(0.0)
    rows = (
        LinearConstraint((LinearTerm("ru.key_rate", 1, 1.0),), 4.0, 8.0),
        LinearConstraint(
            (LinearTerm("ru.key_rate", 1, 1.0), LinearTerm("ru.key_rate", 2, 1.0)),
            9.0, 12.0,
        ),
    )
    intervention = Intervention("ru.key_rate", (1,), "set", np.array([[5.0], [6.0]]))
    plan = ScenarioPlan(2, rows, (intervention,), inner=3)
    result = simulate(model, plan, draws=2, seed=4, candidates=5)
    final = apply_default(result.paths, model.ids, plan.interventions, inner=3)
    assert np.all(final[:, 1, 0] == np.repeat([5.0, 6.0], 3))
    total = final[:, 1, 0] + final[:, 2, 0]
    assert np.all((total >= 9.0) & (total <= 12.0))
    assert np.any(result.paths[:, 1, 0] != final[:, 1, 0])
    law = constraint_law(model, plan, draws=2, seed=4, candidates=5)
    assert np.array_equal(law.means[:, 0], np.repeat([5.0, 6.0], 5))
    assert np.all(law.covs[:, 0, :] == 0.0)
    assert np.allclose(law.means[:, 1], np.repeat([9.0, 10.0], 5))
    assert np.allclose(law.covs[:, 1, 1], 2.0)
    bad = replace(
        plan,
        interventions=(Intervention("ru.key_rate", (1,), "set", np.array([[9.0], [6.0]])),),
    )
    with pytest.raises(ContradictsModels):
        simulate(model, bad, draws=2, seed=4, candidates=5)


def test_inactive_set_and_shift_preserve_group_specific_laws():
    model = bounded(0.0)
    row = LinearConstraint((LinearTerm("ru.key_rate", 1, 1.0),), 3.0, 8.0)
    setting = Intervention("ru.key_rate", (1,), "set", np.array([[5.0], [6.0]]),
                           np.array([[True], [False]]))
    plan = ScenarioPlan(1, (row,), (setting,))
    law = constraint_law(model, plan, draws=2, seed=3, candidates=3)
    assert np.all(law.means[:3, 0] == 5.0)
    assert np.all(law.covs[:3, 0, 0] == 0.0)
    assert np.all(law.means[3:, 0] == 4.0)
    assert np.all(law.covs[3:, 0, 0] == 1.0)
    paths = simulate(model, plan, draws=2, seed=3, candidates=3).paths
    final = apply_default(paths, model.ids, plan.interventions)
    assert np.all((final[:, 1, 0] >= 3.0) & (final[:, 1, 0] <= 8.0))
    shifted = replace(
        plan,
        interventions=(
            Intervention("ru.key_rate", (1,), "shift", np.array([[2.0], [-1.0]])),
        ),
    )
    law = constraint_law(model, shifted, draws=2, seed=3, candidates=3)
    assert np.all(law.means[:, 0] == np.repeat([6.0, 3.0], 3))
    assert np.all(law.covs[:, 0, 0] == 1.0)
