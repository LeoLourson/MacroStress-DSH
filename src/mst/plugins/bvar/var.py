"""Algebra of the vector autoregression of section 4.1 shared by estimation and forecasting.

The regressors of month ``t`` are ``x_t = (1, y_{t-1}, ..., y_{t-p}, d_t)``: a constant, ``p``
lags and the regime dummies. A coefficient matrix ``(n, 1 + n p + d)`` multiplies ``x_t``.
Variables are ordered global first (``n_global`` of them): a global equation reads only the
constant and global lags, so Russian variables never move global ones.
"""

import numpy as np


def regressors(y: np.ndarray, dummies: np.ndarray, p: int) -> np.ndarray:
    """``(T - p, 1 + n p + d)``: one row ``x_t`` per month ``t = p .. T-1``."""
    count = len(y) - p
    lagged = [y[p - lag : p - lag + count] for lag in range(1, p + 1)]
    return np.hstack([np.ones((count, 1)), *lagged, dummies[p:]])


def allowed(n: int, n_global: int, p: int, d: int) -> np.ndarray:
    """``(n, 1 + n p + d)`` True where an equation may have a non-zero coefficient."""
    mask = np.ones((n, 1 + n * p + d), dtype=bool)
    for lag in range(p):
        mask[:n_global, 1 + lag * n + n_global : 1 + (lag + 1) * n] = False
    mask[:n_global, 1 + n * p :] = False
    return mask


def split(coefficients: np.ndarray, p: int):
    """``(const (n,), lags (p, n, n), dummy (n, d))`` of a coefficient matrix."""
    n = coefficients.shape[0]
    lags = coefficients[:, 1 : 1 + n * p].reshape(n, p, n).transpose(1, 0, 2)
    return coefficients[:, 0], lags, coefficients[:, 1 + n * p :]


def responses(lags: np.ndarray, horizon: int) -> np.ndarray:
    """``Psi (horizon, n, n)``: ``Psi[h]`` moves ``y_{t+h}`` per unit of ``u_t``."""
    p, n, _ = lags.shape
    psi = np.zeros((horizon, n, n))
    psi[0] = np.eye(n)
    for h in range(1, horizon):
        for lag in range(1, min(h, p) + 1):
            psi[h] += lags[lag - 1] @ psi[h - lag]
    return psi


def iterate(history: np.ndarray, lags: np.ndarray, intercept: np.ndarray, shocks: np.ndarray):
    """The path ``y_1 .. y_H`` after the last ``p`` months ``history``.

    ``intercept`` (H, n) is the constant plus the dummy terms of each month; ``shocks`` (H, n)
    are the reduced-form errors ``u_t``.
    """
    p = len(lags)
    path = np.vstack([history, np.zeros_like(shocks)])
    for t in range(len(shocks)):
        value = intercept[t] + shocks[t]
        for lag in range(1, p + 1):
            value = value + lags[lag - 1] @ path[p + t - lag]
        path[p + t] = value
    return path[p:]
