"""The laws built from the three quantiles of an answer: ``pareto`` and ``skew_t``.

An answer says what lies between 5 and 95 % and nothing beyond, so the tails are the
plugin's assumption, one for the whole model: the tail index ``alpha`` of the spec. The
``pareto`` family is linear in probability between the three quantiles with Pareto tails
beyond, chosen so the density does not jump at the seams; it reproduces the answer exactly.
The ``skew_t`` family is Azzalini's skew t with ``nu = alpha`` degrees of freedom, whose
location, scale and skewness are fitted to the three quantiles by least squares; it is
smooth but passes through the quantiles only approximately, and a fit that misses by more
than 1 % of the width is reported.

Everything here is a pure function of arrays: no store, no registry, no file.
"""

import numpy as np
from functools import lru_cache
from scipy.integrate import cumulative_simpson
from scipy.optimize import least_squares
from scipy.special import ndtr, ndtri
from scipy.stats import t as student

#: The ends of the level grid: the quantile table covers this much of the law.
LEVEL_LOW, LEVEL_HIGH = 0.0005, 0.9995
#: The probabilities the ``skew_t`` fit reads.
FIT_P = (0.05, 0.5, 0.95)
#: The shape grid of the standardized skew-t quantiles and the z grid of its CDF.
SHAPE_GRID = np.linspace(-20.0, 20.0, 401)
Z_GRID = np.linspace(-36.0, 36.0, 12001)


def level_grid(count: int) -> np.ndarray:
    """``count`` levels, uniform in the normal score between 0.0005 and 0.9995."""
    return ndtr(np.linspace(ndtri(LEVEL_LOW), ndtri(LEVEL_HIGH), count))


def pareto_quantile(q5, q50, q95, p, alpha):
    """The quantile function of the ``pareto`` family, vectorised over cells and levels.

    Linear in probability between 5, 50 and 95 %; beyond, Pareto tails of index ``alpha``
    whose scale keeps the density continuous at the seams: with
    ``s- = alpha (q50 - q5) / 9`` and ``s+ = alpha (q95 - q50) / 9``,
    ``Q(p) = q5 - s- ((0.05/p)^(1/alpha) - 1)`` below 5 % and
    ``Q(p) = q95 + s+ ((0.05/(1-p))^(1/alpha) - 1)`` above 95 %. The cells ``q5, q50, q95``
    broadcast against one another; ``p`` is a level vector, and the result carries
    ``cells + (len(p),)``.
    """
    q5, q50, q95 = np.broadcast_arrays(
        np.asarray(q5, dtype=float), np.asarray(q50, dtype=float), np.asarray(q95, dtype=float)
    )
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1.0 - 1e-15)
    low, mid, high = q5[..., None], q50[..., None], q95[..., None]
    left_scale = alpha * (mid - low) / 9.0
    right_scale = alpha * (high - mid) / 9.0
    with np.errstate(over="ignore"):
        left = low - left_scale * ((0.05 / p) ** (1.0 / alpha) - 1.0)
        right = high + right_scale * ((0.05 / (1.0 - p)) ** (1.0 / alpha) - 1.0)
        lower = low + (p - 0.05) / 0.45 * (mid - low)
        upper = mid + (p - 0.5) / 0.45 * (high - mid)
    return np.where(
        p < 0.05, left, np.where(p <= 0.5, lower, np.where(p <= 0.95, upper, right))
    )


def _density(z: np.ndarray, shape: np.ndarray, nu: float) -> np.ndarray:
    """The skew-t density of Azzalini and Capitanio on ``z`` for every shape, ``(S, Z)``."""
    scale = np.sqrt((nu + 1.0) / (nu + z * z))
    argument = shape[:, None] * z[None, :] * scale[None, :]
    return 2.0 * student.pdf(z, nu)[None, :] * student.cdf(argument, nu + 1)


@lru_cache(maxsize=8)
def _standardized(nu: float, p: tuple[float, ...]) -> np.ndarray:
    """The standardized skew-t quantiles at ``p``, one row per shape of ``SHAPE_GRID``.

    The CDF is accumulated on ``Z_GRID`` by Simpson's rule and inverted by interpolation;
    the mass the grid truncates is normalised away, an error below a millionth of the
    width at the probabilities the fit and the table read. Cached by the degrees of
    freedom and the probabilities, so the fit and the table of one estimate share it.
    """
    density = _density(Z_GRID, SHAPE_GRID, nu)
    cdf = cumulative_simpson(density, x=Z_GRID, initial=0.0, axis=1)
    cdf = np.maximum.accumulate(cdf / cdf[:, -1:], axis=1)
    levels = np.array(p)
    rows = np.empty((len(SHAPE_GRID), len(levels)))
    for index in range(len(SHAPE_GRID)):
        rows[index] = np.interp(levels, cdf[index], Z_GRID)
    return rows


def skew_t_fit(q5, q50, q95, nu):
    """``(loc, scale, shape, miss)`` of every cell, fitted to its three quantiles.

    The fit is least squares on the three quantiles standardised by the width
    ``q95 - q5``; ``miss`` is the largest residual as a share of the width. A symmetric
    answer is a plain Student t and is reproduced exactly.
    """
    from scipy.interpolate import PchipInterpolator

    q5 = np.asarray(q5, dtype=float)
    q50 = np.asarray(q50, dtype=float)
    q95 = np.asarray(q95, dtype=float)
    quantiles = PchipInterpolator(SHAPE_GRID, _standardized(nu, FIT_P))
    count = q5.size
    loc = np.empty(count)
    scale = np.empty(count)
    shape = np.empty(count)
    miss = np.empty(count)
    for cell in range(count):
        width = q95.flat[cell] - q5.flat[cell]
        target = np.array([q5.flat[cell], q50.flat[cell], q95.flat[cell]])

        def residual(beta):
            values = beta[0] + np.exp(beta[1]) * quantiles(beta[2])
            return (values - target) / width

        start = np.array([q50.flat[cell], np.log(width / 4.0), 0.0])
        found = least_squares(
            residual, start, bounds=([-np.inf, -30.0, -20.0], [np.inf, 30.0, 20.0])
        )
        loc[cell], scale[cell], shape[cell] = found.x[0], np.exp(found.x[1]), found.x[2]
        miss[cell] = float(np.max(np.abs(residual(found.x))))
    return loc, scale, shape, miss


def skew_t_quantile(loc, scale, shape, nu, p):
    """The quantile function of the fitted skew t, by inverting the CDF on the level grid.

    ``loc``, ``scale`` and ``shape`` are per cell, one dimensional; ``p`` is the level
    vector. The result is ``(cells, len(p))``.
    """
    from scipy.interpolate import PchipInterpolator

    p = np.asarray(p, dtype=float)
    loc = np.atleast_1d(np.asarray(loc, dtype=float))
    scale = np.atleast_1d(np.asarray(scale, dtype=float))
    shape = np.atleast_1d(np.asarray(shape, dtype=float))
    standard = PchipInterpolator(SHAPE_GRID, _standardized(nu, tuple(p.tolist())))
    return loc[:, None] + scale[:, None] * standard(shape)


def table(q5, q50, q95, spec):
    """``(levels, values, misses)``: the quantile table of every cell on the level grid.

    A cell with ``q5 == q95`` is a constant, whatever the family. The ``pareto`` family
    reproduces the answer exactly, so its misses are zero; the ``skew_t`` family fits and
    reports the largest miss of every cell as a share of its width.
    """
    q5 = np.asarray(q5, dtype=float)
    q50 = np.asarray(q50, dtype=float)
    q95 = np.asarray(q95, dtype=float)
    levels = level_grid(spec.levels)
    count = q5.size
    values = np.empty((count, len(levels)))
    misses = np.zeros(count)
    constant = (q5 == q95).ravel()
    if constant.any():
        values[constant] = q50.ravel()[constant][:, None]
    moving = ~constant
    if not moving.any():
        return levels, values, misses
    cells = moving
    if spec.family == "pareto":
        values[cells] = pareto_quantile(
            q5.ravel()[cells], q50.ravel()[cells], q95.ravel()[cells], levels, spec.alpha
        )
        return levels, values, misses
    loc, scale, shape, miss = skew_t_fit(
        q5.ravel()[cells], q50.ravel()[cells], q95.ravel()[cells], spec.alpha
    )
    values[cells] = skew_t_quantile(loc, scale, shape, spec.alpha, levels)
    misses[cells] = miss
    return levels, values, misses
