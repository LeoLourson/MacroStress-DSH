"""The generated macro model follows the SDK without plugin discovery."""

from datetime import date
from pathlib import Path

import numpy as np

from mst.model_sdk.contracts import Intervention, LinearConstraint, LinearTerm, ScenarioPlan
from mst.model_sdk.interventions import apply_default


def template_model():
    path = Path(__file__).parents[2] / "src/mst/templates/macro/model.py.tmpl"
    source = path.read_text(encoding="utf-8")
    for key, value in (("plugin_id", "example"), ("class", "Example"), ("label", "Example")):
        source = source.replace("{{" + key + "}}", value)
    namespace = {"__name__": "example_template"}
    exec(compile(source, str(path), "exec"), namespace)
    loaded = namespace["ExampleModel"](
        ids=["a", "b"], last_month=date(2026, 9, 1),
        last=np.array([1.0, 2.0]), drift=np.zeros(2), chol=np.eye(2),
    )
    return namespace["Example"](), loaded


def test_macro_template_keeps_contiguous_groups_and_default_form():
    plugin, model = template_model()
    simulation = plugin.simulate(model, ScenarioPlan(horizon=2, inner=3), draws=2, seed=4)
    assert simulation.paths.shape == (6, 3, 2)
    np.testing.assert_array_equal(simulation.group, [0, 0, 0, 1, 1, 1])
    assert plugin.structural_form(model) is None


def test_macro_template_conditions_after_set_substitution_without_double_applying():
    plugin, model = template_model()
    intervention = Intervention("a", (1,), "set", np.array([[7.0], [9.0]]))
    constraint = LinearConstraint((LinearTerm("a", 1, 1.0),), 6.0, 10.0)
    plan = ScenarioPlan(
        horizon=2, inner=3, constraints=(constraint,), interventions=(intervention,)
    )
    simulation = plugin.simulate(model, plan, draws=2, seed=4)
    assert not np.all(simulation.paths[:3, 1, 0] == 7.0)
    paths = apply_default(simulation.paths, simulation.ids, plan.interventions, inner=3)
    np.testing.assert_array_equal(paths[:, 1, 0], [7.0, 7.0, 7.0, 9.0, 9.0, 9.0])
