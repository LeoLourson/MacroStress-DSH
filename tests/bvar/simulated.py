"""Panels simulated from a known VAR for the macro model tests."""

from datetime import date

import numpy as np

from mst.macro.panel import Panel, months_between, shift
from mst.plugins.bvar.spec import Spec

START = date(2000, 1, 1)

CONST = np.array([1.0, 0.5, -0.5])
LAG = np.array([[0.7, 0.0, 0.0], [0.3, 0.5, 0.1], [-0.2, 0.2, 0.6]])
CHOL = np.array([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [-0.3, 0.4, 1.0]])
LOGVAR = np.log(np.array([1.0, 0.5, 2.0]))
IDS = ["gl.a", "ru.b", "ru.c"]


def spec(draws=300, burn=0, thin=1, seed=7, **changes) -> Spec:
    values = {
        "start": START,
        "lags": 1,
        "tightness": 10.0,
        "volatility": "constant",
        "volvol_mean": 0.03,
        "volvol_df": 5.0,
        "stationary": IDS,
        "sign_lag": 2,
        "rotation_tries": 5000,
        "sampler": {"draws": draws, "burn": burn, "thin": thin, "seed": seed},
    }
    values.update(changes)
    return Spec.model_validate(values)


def simulate(count: int, rng, logvar: np.ndarray | None = None, dof=None):
    """``count`` months of the true VAR(1); ``logvar`` (count, n) gives time-varying variances
    of the orthogonal errors, ``dof`` Student tails."""
    n = len(CONST)
    logvar = np.tile(LOGVAR, (count, 1)) if logvar is None else logvar
    z = rng.standard_normal((count, n))
    if dof is not None:
        z *= np.sqrt(dof / 2 / rng.gamma(dof / 2, size=(count, n)))
    y = np.zeros((count, n))
    previous = np.linalg.solve(np.eye(n) - LAG, CONST)
    for t in range(count):
        y[t] = CONST + LAG @ previous + CHOL @ (np.exp(logvar[t] / 2) * z[t])
        previous = y[t]
    return y


def panel(y: np.ndarray) -> Panel:
    return Panel(
        ids=IDS,
        n_global=1,
        months=months_between(START, shift(START, len(y) - 1)),
        y=y,
        approximate=np.zeros(y.shape, dtype=bool),
    )
