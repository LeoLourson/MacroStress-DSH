"""Panels simulated from a known MS-VAR for the macro model tests."""

from datetime import date

import numpy as np

from mst.macro.panel import Panel, months_between, shift
from mst.plugins.msvar.spec import Spec

START = date(2000, 1, 1)
IDS = ["gl.a", "ru.b", "ru.c"]
N_GLOBAL = 1

#: The calm regime (0) and the stress regime (1): the stress one drifts down, carries its
#: shocks with four times the variance and forgets its past a little sooner.
CONST = np.array([[1.0, 0.5, -0.5], [-1.0, 0.5, -0.5]])
LAGS = np.array(
    [
        [[0.7, 0.0, 0.0], [0.3, 0.5, 0.1], [-0.2, 0.2, 0.6]],
        [[0.5, 0.0, 0.0], [0.2, 0.3, 0.1], [-0.1, 0.1, 0.3]],
    ]
)
CHOL = np.array([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [-0.3, 0.4, 1.0]])
LOGVAR = np.array([np.log([1.0, 0.5, 2.0]), np.log([4.0, 2.0, 8.0])])
#: Persistent calm, shorter stress: what the Russian panel looks like.
TRANSITION = np.array([[0.95, 0.05], [0.3, 0.7]])


def spec(draws=60, burn=60, thin=1, seed=11, **changes) -> Spec:
    values = {
        "start": START,
        "lags": 1,
        "tightness": 10.0,
        "stationary": IDS,
        "sign_lag": 2,
        "rotation_tries": 5000,
        "transition_stay": 4.0,
        "transition_switch": 0.5,
        "sampler": {"draws": draws, "burn": burn, "thin": thin, "seed": seed},
    }
    values.update(changes)
    return Spec.model_validate(values)


def regimes(count: int, rng) -> np.ndarray:
    """A regime path of the true kernel: persistent calm, shorter stress."""
    states = np.empty(count, dtype=int)
    state = 0
    for t in range(count):
        state = _draw(TRANSITION[state], rng)
        states[t] = state
    return states


def _draw(weights: np.ndarray, rng) -> int:
    cumulated = np.cumsum(weights)
    return int(np.searchsorted(cumulated, rng.random() * cumulated[-1]))


def simulate(count: int, rng, states: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``count`` months of the true MS-VAR, and the regime path they were drawn under.

    The path starts in the calm regime and follows ``TRANSITION``; the observation of month
    ``t`` uses the parameters of its regime.
    """
    n = len(IDS)
    if states is None:
        states = regimes(count, rng)
    y = np.zeros((count, n))
    previous = np.linalg.solve(np.eye(n) - LAGS[0], CONST[0])
    for t in range(count):
        s = states[t]
        z = rng.standard_normal(n)
        y[t] = CONST[s] + LAGS[s] @ previous + CHOL @ (np.exp(LOGVAR[s] / 2) * z)
        previous = y[t]
    return y, states


def panel(y: np.ndarray) -> Panel:
    return Panel(
        ids=IDS,
        n_global=N_GLOBAL,
        months=months_between(START, shift(START, len(y) - 1)),
        y=y,
        approximate=np.zeros(y.shape, dtype=bool),
    )
