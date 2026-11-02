"""The shadow value and the observed one: the registry's floors in the BVAR simulation."""

from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from mst.macro.registry import load_registry
from mst.model_sdk.bounds import observed
from mst.model_sdk.contracts import LinearConstraint, LinearTerm, ScenarioPlan
from mst.plugins.bvar.estimate import Posterior
from mst.plugins.bvar.forecast import simulate

MONTHS = [date(2000 + m // 12, m % 12 + 1, 1) for m in range(13)]


def bounded(const: float) -> Posterior:
    """A one-variable posterior of ``ru.key_rate``: a random walk with drift ``const`` and
    unit noise, so the shadow path is the cumulated drift plus the shocks."""
    return Posterior(
        ids=["ru.key_rate"],
        n_global=0,
        months=MONTHS,
        lags_count=1,
        dummies=[],
        const=np.array([[const]]),
        lags=np.ones((1, 1, 1, 1)),
        dummy=np.zeros((1, 1, 0)),
        chol=np.eye(1)[None],
        logvar=np.zeros((1, 12, 1)),
        volvol=np.zeros((1, 1)),
        dof=np.full((1, 1), np.inf),
        history=np.full((1, 1, 1), 4.0),
        rotation=np.eye(1)[None],
        shocks=["ru.shock"],
    )


def plan(constraints=(), horizon=6) -> ScenarioPlan:
    return ScenarioPlan(
        horizon=horizon,
        constraints=tuple(
            LinearConstraint(
                terms=(LinearTerm(variable="ru.key_rate", month=month, weight=1.0),),
                low=low,
                high=low,
            )
            for month, low in constraints
        ),
    )


def test_no_reported_path_goes_below_the_floor():
    run = observed(simulate(bounded(-2.0), plan(), draws=200, seed=1), load_registry())
    assert np.all(run.paths >= 0.0)
    # The shadow keeps drifting down, so almost every path ends pinned at the floor.
    assert float(np.mean(run.paths[:, -1, 0] == 0.0)) > 0.9


def test_the_condition_holds_on_the_shadow_and_a_rise_starts_from_it():
    # An exact condition two points below the floor: the shadow meets it — month 1 comes
    # out at the floor — and month 2 rises from the shadow (mean four: minus two plus the
    # drift of six), not from the floor (mean six).
    shadow = simulate(bounded(6.0), plan([(1, -2.0)]), draws=400, seed=2)
    assert np.allclose(shadow.paths[:, 1, 0], -2.0, atol=1e-9, rtol=0)
    run = observed(shadow, load_registry())
    assert np.allclose(run.paths[:, 1, 0], 0.0, atol=1e-9, rtol=0)
    second = run.paths[:, 2, 0]
    error = 4 * second.std() / np.sqrt(len(second))
    assert second.mean() == pytest.approx(4.0, abs=error)
    assert not np.allclose(second.mean(), 6.0, atol=error)


def test_a_variable_the_registry_does_not_know_stays_unbounded():
    # The same posterior under a test id: no floor, the paths are the shadow ones.
    model = replace(bounded(-2.0), ids=["ru.b"])
    run = observed(simulate(model, plan(), draws=50, seed=3), load_registry())
    assert float(np.mean(run.paths[:, -1, 0] < 0.0)) > 0.9
