"""Random interventions: the law of the event is drawn once per group of paths.

The three checks are the ones the plan names: the share of the groups that get a
``with_probability`` intervention agrees with its probability; an observation does not move
that share, because the law is drawn from the seed alone; and a ``uniform`` on a degenerate
interval gives bit for bit the same plan as the deterministic number it collapses to.
"""

import numpy as np
import pytest

from mst.scenario.contracts import Scenario
from mst.scenario.translate import QuantityData, TranslationData, plan_of

DATA = TranslationData(
    origin="2026-09",
    quantities={"ru.key_rate": QuantityData("ru.key_rate", "identity")},
    last_model={"ru.key_rate": 16.5},
    history={"ru.key_rate": np.array([16.0, 16.5])},
)


def scenario(values, *, observations=()):
    return Scenario.model_validate(
        {
            "narrative": "заседание",
            "interventions": [
                {
                    "quantity": "ru.key_rate",
                    "months": ["2026-11"],
                    "kind": "set",
                    "values": values,
                    "unit": "level",
                    "reason": "заседание",
                    "basis": "assumption",
                }
            ],
            "observations": list(observations),
        }
    )


def test_the_share_of_groups_with_a_probabilistic_intervention_agrees_with_its_probability():
    spec = scenario([{"with_probability": 0.3, "value": 12.0}])
    plan = plan_of(spec, DATA, 3, groups=4000, seed=7)
    intervention = plan.interventions[0]
    assert intervention.active is not None
    assert intervention.active.shape == (4000, 1)
    assert intervention.active.mean() == pytest.approx(0.3, abs=0.03)
    assert np.all(intervention.values == 12.0)


def test_an_observation_moves_no_drawn_value():
    """The law of the intervention enters R through the seed; observations only reweight."""
    observation = {
        "kind": "constraint",
        "terms": [{"quantity": "ru.key_rate", "month": "2026-12", "weight": 1.0}],
        "unit": "level",
        "low": 10.0,
        "high": 14.0,
        "basis": "assumption",
    }
    law = [{"uniform": [10.0, 13.0]}]
    without = plan_of(scenario(law), DATA, 3, groups=500, seed=11)
    with_observation = plan_of(
        scenario(law, observations=[observation]), DATA, 3, groups=500, seed=11
    )
    drawn, observed = without.interventions[0], with_observation.interventions[0]
    assert np.array_equal(drawn.values, observed.values)
    assert np.array_equal(drawn.active, observed.active)
    assert len(with_observation.constraints) == 1 and not without.constraints


def test_a_uniform_on_a_degenerate_interval_is_the_deterministic_intervention():
    """A zero-width interval draws the same value in every group, bit for bit."""
    drawn = plan_of(scenario([{"uniform": [11.5, 11.5]}]), DATA, 3, groups=64, seed=5)
    certain = plan_of(scenario([11.5]), DATA, 3, groups=64, seed=5)
    assert np.array_equal(drawn.interventions[0].values, certain.interventions[0].values)
    assert drawn.interventions[0].active is None and certain.interventions[0].active is None
    assert np.all(drawn.interventions[0].values == 11.5)


def test_a_shift_carries_the_drawn_displacement():
    """A drawn displacement is a change from the last value, translated once."""
    spec = Scenario.model_validate(
        {
            "narrative": "заседание",
            "interventions": [
                {
                    "quantity": "ru.key_rate",
                    "months": ["2026-11"],
                    "kind": "shift",
                    "values": [{"uniform": [-2.0, 2.0]}],
                    "unit": "change",
                    "reason": "заседание",
                    "basis": "assumption",
                }
            ],
        }
    )
    plan = plan_of(spec, DATA, 3, groups=200, seed=13)
    values = plan.interventions[0].values
    assert values.shape == (200, 1)
    assert np.all((values >= -2.0) & (values <= 2.0))
    assert values.std() > 0.5
