"""The shadow value and the observed one: the registry's floors in the MS-VAR simulation."""

from dataclasses import replace
from datetime import date

import numpy as np

from mst.model_sdk.contracts import ScenarioPlan
from mst.plugins.msvar.estimate import Posterior
from mst.plugins.msvar.forecast import simulate

MONTHS = [date(2000 + m // 12, m % 12 + 1, 1) for m in range(13)]


def bounded(const: float) -> Posterior:
    """A one-variable posterior of ``ru.key_rate``: two identical regimes — a random walk
    with drift ``const`` and unit noise, so the regime path does not matter."""
    return Posterior(
        ids=["ru.key_rate"],
        n_global=0,
        months=MONTHS,
        lags_count=1,
        n_regimes=2,
        const=np.array([[[const], [const]]]),
        lags=np.ones((1, 2, 1, 1, 1)),
        chol=np.stack([np.eye(1)] * 2)[None],
        logvar=np.zeros((1, 2, 1)),
        transition=np.full((1, 2, 2), 0.5),
        states=np.zeros((1, 12), dtype=int),
        history=np.full((1, 1, 1), 4.0),
        rotation=np.stack([np.eye(1)] * 2)[None],
        shocks=["ru.shock"],
        regime_ids=["regime.calm", "regime.stress"],
    )


def test_no_reported_path_goes_below_the_floor():
    run = simulate(bounded(-2.0), ScenarioPlan(horizon=6), draws=200, seed=1)
    assert np.all(run.paths >= 0.0)
    # The shadow keeps drifting down, so almost every path ends pinned at the floor.
    assert float(np.mean(run.paths[:, -1, 0] == 0.0)) > 0.9


def test_a_variable_the_registry_does_not_know_stays_unbounded():
    model = replace(bounded(-2.0), ids=["ru.b"])
    run = simulate(model, ScenarioPlan(horizon=6), draws=50, seed=3)
    assert float(np.mean(run.paths[:, -1, 0] < 0.0)) > 0.9
