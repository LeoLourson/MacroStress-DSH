"""Stability of a VAR draw: the spectral radius of its companion matrix.

A vector autoregression ``y_t = A_1 y_{t-1} + ... + A_p y_{t-p} + u_t`` is stable when every
root of its characteristic polynomial lies at or inside the unit circle, which is the same as
the spectral radius of its companion matrix being at or below one. A unit root is admissible
for nonstationary levels — variables that stay near their last value — while a radius above
one means a shock grows without bound along the path, and conditions on a path multiply the
effect. The macro models keep only the draws whose radius is at or below their ``rho_max``
(``mst.plugins.bvar.identify``, ``mst.plugins.msvar.identify``), the same mechanism as the
identification filter, and the diagnostics of a version read the radius per draw
(``mst.models diagnose``).
"""

import numpy as np


def companion_radius(lags: np.ndarray) -> np.ndarray:
    """The spectral radius of the companion matrix; ``lags`` is (..., p, n, n).

    The companion matrix stacks the lag matrices on its first block row — ``[A_1 ... A_p]`` —
    and identities below, so its eigenvalues are the roots of the characteristic polynomial
    ``z^p = A_1 z^{p-1} + ... + A_p``. Returns the largest modulus of them, one entry per
    leading index; a scalar panel gives a scalar radius.
    """
    matrix = np.asarray(lags, dtype=float)
    if matrix.ndim < 3 or matrix.shape[-2] != matrix.shape[-1]:
        raise ValueError(
            f"lags must be (..., p, n, n) with square lag matrices, got {matrix.shape}"
        )
    batch = matrix.shape[:-3]
    p, n = matrix.shape[-3], matrix.shape[-2]
    size = p * n
    companion = np.zeros((*batch, size, size))
    # The first block row is [A_1 ... A_p]: the row names the equation, the column the lag
    # and the variable of that lag.
    companion[..., :n, :] = matrix.reshape(*batch, size, n).swapaxes(-2, -1)
    if p > 1:
        companion[..., n:, : size - n] = np.eye(size - n)
    return np.max(np.abs(np.linalg.eigvals(companion)), axis=-1)
