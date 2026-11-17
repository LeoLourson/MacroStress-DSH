"""Equations on monthly changes and the VAR in levels they are equivalent to.

A trending series in levels gives a VAR whose sampled draws are explosive: the sum of its
own-lag coefficients sits near one and the noise of a short history pushes it over. On
``w_t = y_t - S y_{t-1}``, with ``S`` the diagonal of ones for the differenced variables, the
unit root is exact and the constant is a monthly drift. A version keeps the equivalent VAR in
levels with one lag more, so the forecast, the missing months and the identification read
the shape they always did.
"""

import numpy as np


def changes(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """``w_t = y_t - s y_{t-1}`` from the second month on: changes where ``s`` is one, levels
    elsewhere. Without differences ``y`` itself, every month kept."""
    if not s.any():
        return y
    return y[1:] - s * y[:-1]


def levels_lags(lags: np.ndarray, s: np.ndarray) -> np.ndarray:
    """``(p + 1, n, n)``: the lag matrices in levels of ``p`` lag matrices ``G`` on changes.

    ``y_t = S y_{t-1} + c + sum_k G_k (y_{t-k} - S y_{t-k-1})``, so ``A_1 = S + G_1``,
    ``A_k = G_k - G_{k-1} S`` and ``A_{p+1} = -G_p S``.
    """
    p, n, _ = lags.shape
    scale = np.diag(s)
    levels = np.zeros((p + 1, n, n))
    levels[0] = scale + lags[0]
    for k in range(1, p):
        levels[k] = lags[k] - lags[k - 1] @ scale
    levels[p] = -lags[p - 1] @ scale
    return levels
