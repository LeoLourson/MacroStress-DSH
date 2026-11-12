from datetime import date
from types import SimpleNamespace

import numpy as np

from mst.model_sdk.contracts import DriverPaths, MacroSimulation
from mst.model_sdk.interventions import constraints_under_default
from mst.scenario.build import build
from mst.scenario.contracts import Scenario


class DefaultModel:
    def structural_form(self, model):
        return None

    def simulate(self, model, plan, *, draws, seed):
        for group in range(draws):
            constraints_under_default(
                plan.constraints, plan.interventions, ["ru.key_rate"], group=group
            )
        paths = np.ones((draws * plan.inner, plan.horizon + 1, 1))
        return MacroSimulation(
            ["ru.key_rate"],
            [date(2026, 10, 1)] * (plan.horizon + 1),
            paths,
            group=np.repeat(np.arange(draws), plan.inner),
        )

    def constraint_law(self, model, plan, *, draws, seed):
        return None

    def log_box_probability(self, model, plan, low, high, *, draws, seed):
        return None


class Satellites:
    parts = {}
    arrays = {}
    registry = SimpleNamespace(quantities=[])

    def served(self):
        return []

    def simulate(self, macro, drivers, interventions, *, seed, inner):
        self.macro = macro
        self.interventions = interventions
        return DriverPaths([], np.empty((*macro.paths.shape[:2], 0)), {})


def test_default_intervention_applied_once_before_satellites():
    panel = SimpleNamespace(
        ids=["ru.key_rate"],
        y=np.array([[1.0], [2.0]]),
        months=[date(2026, 9, 1), date(2026, 10, 1)],
    )
    macro = SimpleNamespace(panel=panel, ids=panel.ids, plugin=DefaultModel(), model=None)
    satellites = Satellites()
    scenario = Scenario.model_validate(
        {
            "narrative": "test",
            "interventions": [
                {
                    "quantity": "ru.key_rate",
                    "months": ["2026-11"],
                    "kind": "shift",
                    "values": [2],
                    "unit": "level",
                    "reason": "test",
                    "basis": "event",
                }
            ],
        }
    )
    result = build(scenario, macro, satellites, draws=2, seed=4, horizon=1, inner=2)
    assert np.all(result.shadow[:, 1, 0] == 3)
    assert np.all(satellites.macro.paths[:, 1, 0] == 3)
    assert result.dist is None
