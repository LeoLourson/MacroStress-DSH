"""Algebra of the Markov-switching vector autoregression shared by estimation and forecasting.

The regressors of month ``t`` are ``x_t = (1, y_{t-1}, ..., y_{t-p})``: a constant and ``p``
lags, no dummies — the regime is the crisis month, so an intervention dummy would fight it
for the same variation. A coefficient matrix of regime ``s`` is ``(n, 1 + n p)``. Variables
are ordered global first (``n_global`` of them): a global equation reads only the constant
and global lags, so Russian variables never move global ones, in every regime.

Forecasting walks a regime path: ``lags`` becomes ``(H, p, n, n)`` — the lag matrices of the
month's regime — and the impulse of month ``s`` travels to month ``t`` through the regimes
between them, which is what ``_responses`` of ``forecast`` tabulates.
"""

import numpy as np


def regressors(y: np.ndarray, p: int) -> np.ndarray:
    """``(T - p, 1 + n p)``: one row ``x_t`` per month ``t = p .. T-1``."""
    count = len(y) - p
    lagged = [y[p - lag : p - lag + count] for lag in range(1, p + 1)]
    return np.hstack([np.ones((count, 1)), *lagged])


def allowed(n: int, n_global: int, p: int) -> np.ndarray:
    """``(n, 1 + n p)`` True where an equation may have a non-zero coefficient."""
    mask = np.ones((n, 1 + n * p), dtype=bool)
    for lag in range(p):
        mask[:n_global, 1 + lag * n + n_global : 1 + (lag + 1) * n] = False
    return mask


def split(coefficients: np.ndarray, p: int):
    """``(const (n,), lags (p, n, n))`` of one regime's coefficient matrix."""
    n = coefficients.shape[0]
    lags = coefficients[:, 1 : 1 + n * p].reshape(n, p, n).transpose(1, 0, 2)
    return coefficients[:, 0], lags


def responses(lags: np.ndarray, horizon: int) -> np.ndarray:
    """``Psi (horizon, n, n)``: ``Psi[h]`` moves ``y_{t+h}`` per unit of ``u_t`` in one regime."""
    p, n, _ = lags.shape
    psi = np.zeros((horizon, n, n))
    psi[0] = np.eye(n)
    for h in range(1, horizon):
        for lag in range(1, min(h, p) + 1):
            psi[h] += lags[lag - 1] @ psi[h - lag]
    return psi


def iterate(history: np.ndarray, lags: np.ndarray, intercept: np.ndarray, shocks: np.ndarray):
    """The path ``y_1 .. y_H`` after the last ``p`` months ``history`` along a regime path.

    ``lags`` is ``(H, p, n, n)`` — the lag matrices of each month's regime — and
    ``intercept`` (H, n) and ``shocks`` (H, n) are the constant and the reduced-form error
    of each month. With one regime's matrices tiled over the months this is the ordinary
    VAR iteration.
    """
    p = lags.shape[1]
    path = np.vstack([history, np.zeros_like(shocks)])
    for t in range(len(shocks)):
        value = intercept[t] + shocks[t]
        for lag in range(1, p + 1):
            value = value + lags[t, lag - 1] @ path[p + t - lag]
        path[p + t] = value
    return path[p:]
