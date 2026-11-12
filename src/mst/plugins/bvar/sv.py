"""Gibbs steps for stochastic volatility with Student tails (section 4.1).

An orthogonalised error is ``e_t = exp(h_t / 2) sqrt(omega_t) z_t`` with ``h_t = h_{t-1} +
s xi_t`` and ``omega_t ~ InvGamma(nu / 2, nu / 2)``. Given ``h`` and ``omega`` each step below
draws one block from its full conditional. A variable the spec names in ``jumps`` replaces the
Student mixing by a jump: ``omega_t = kappa`` with probability ``pi`` and 1 otherwise.

``log(e_t^2 / omega_t) = h_t + log z_t^2``; ``log z_t^2`` is approximated by the seven-normal
mixture of Kim, Shephard and Chib (1998, Review of Economic Studies 65(3), table 4), after
which ``h`` is Gaussian and drawn at once from its banded precision (Chan and Jeliazkov, 2009).
"""

import numpy as np
from scipy import stats
from scipy.linalg import cho_solve_banded, cholesky_banded, solve_banded
from scipy.special import gammaln

# Kim, Shephard and Chib (1998), table 4: weights, means (before the -1.2704 shift) and
# variances of the components approximating log chi^2_1.
MIX_WEIGHT = np.array([0.00730, 0.10556, 0.00002, 0.04395, 0.34001, 0.24566, 0.25750])
MIX_MEAN = np.array([-10.12999, -3.97281, -8.56686, 2.77786, 0.61942, 1.79518, -1.08819]) - 1.2704
MIX_VAR = np.array([5.79596, 2.61369, 5.17950, 0.16735, 0.64009, 0.34023, 1.26261])

# Uniform prior of nu over a grid of degrees of freedom; 50 is close to normal. The grid
# starts at the floor the spec names (``dof_min``, the shipped 5): variance is infinite at
# 2 and kurtosis at 4, and for rates that sit still and then step the floor absorbs what
# the volatility process should carry. The default keeps the grid the versions estimated
# before the field carried.
DOF_FLOOR = 3.0
DOF_TOP = 50.0


def dof_grid(dof_min: float = DOF_FLOOR) -> np.ndarray:
    """The uniform grid of degrees of freedom, from the floor the spec names."""
    return np.arange(float(dof_min), DOF_TOP + 1.0)


# log(e^2 + OFFSET) keeps an exactly zero error finite.
OFFSET = 1e-8


def draw_logvar(
    log_e2: np.ndarray, h: np.ndarray, s2: float, h0: tuple[float, float], rng
) -> np.ndarray:
    """Draws the path ``h`` (T,) given ``log(e^2 / omega)``, the random-walk variance ``s2``
    and the prior ``h_1 ~ N(h0[0], h0[1])``."""
    density = MIX_WEIGHT * np.exp(-0.5 * (log_e2[:, None] - h[:, None] - MIX_MEAN) ** 2 / MIX_VAR)
    density /= np.sqrt(MIX_VAR)
    cumulative = np.cumsum(density, axis=1)
    pick = rng.random(len(h))[:, None] * cumulative[:, -1:]
    component = np.minimum((cumulative < pick).sum(axis=1), len(MIX_WEIGHT) - 1)

    count = len(h)
    # Precision of h: the random walk H' diag(1/h0_var, 1/s2, ...) H plus the measurement.
    diagonal = np.full(count, 2.0 / s2)
    diagonal[-1] = 1.0 / s2
    diagonal[0] = 1.0 / h0[1] + (1.0 / s2 if count > 1 else 0.0)
    diagonal += 1.0 / MIX_VAR[component]
    upper = np.zeros((2, count))
    upper[0, 1:] = -1.0 / s2
    upper[1] = diagonal
    target = (log_e2 - MIX_MEAN[component]) / MIX_VAR[component]
    target[0] += h0[0] / h0[1]
    factor = cholesky_banded(upper)
    mean = cho_solve_banded((factor, False), target)
    return mean + solve_banded((0, 1), factor, rng.standard_normal(count))


def draw_volvol(h: np.ndarray, prior: tuple[float, float], rng) -> float:
    """``s^2`` from its inverse-gamma full conditional; ``prior`` is (shape, scale)."""
    steps = np.diff(h)
    shape = prior[0] + len(steps) / 2
    scale = prior[1] + steps @ steps / 2
    return scale / rng.gamma(shape)


JUMP_PRIOR = (1.0, 19.0)
"""Beta prior of the jump probability ``pi``: a jump in one month of twenty, weighing as
twenty months of data."""
KAPPA_SHAPE, KAPPA_MEAN = 2.0, 25.0
"""Inverse-gamma prior of the jump scale ``kappa``, truncated above one: shape 2, mean 25."""


def draw_jumps(scaled_e2: np.ndarray, pi: float, kappa: float, rng) -> np.ndarray:
    """``omega_t`` of a jump mixture: ``kappa`` in a month with a jump, 1 without, given
    ``e_t^2 / exp(h_t)`` (Chib, Nardari and Shephard, 2002, with a jump in the variance)."""
    with_jump = np.log(pi) - 0.5 * np.log(kappa) - scaled_e2 / (2 * kappa)
    without = np.log1p(-pi) - scaled_e2 / 2
    probability = 1 / (1 + np.exp(np.clip(without - with_jump, -700, 700)))
    return np.where(rng.random(len(scaled_e2)) < probability, kappa, 1.0)


def draw_jump_parameters(scaled_e2: np.ndarray, omega: np.ndarray, rng) -> tuple[float, float]:
    """``pi`` and ``kappa`` given the months with a jump (``omega > 1``)."""
    jump = omega > 1
    count = int(jump.sum())
    pi = float(rng.beta(JUMP_PRIOR[0] + count, JUMP_PRIOR[1] + len(omega) - count))
    shape = KAPPA_SHAPE + count / 2
    scale = KAPPA_MEAN * (KAPPA_SHAPE - 1) + scaled_e2[jump].sum() / 2
    # The truncation above one: a draw from the gamma's upper part by inversion.
    low = stats.gamma.cdf(1.0, shape, scale=1 / scale)
    kappa = 1 / stats.gamma.ppf(rng.uniform(0.0, low), shape, scale=1 / scale)
    return pi, float(max(kappa, 1.0 + 1e-9))


def draw_mixing(scaled_e2: np.ndarray, nu: float, rng) -> np.ndarray:
    """``omega_t`` given ``e_t^2 / exp(h_t)``."""
    return (nu + scaled_e2) / 2 / rng.gamma((nu + 1) / 2, size=len(scaled_e2))


def draw_dof(omega: np.ndarray, rng, grid: np.ndarray | None = None) -> float:
    """``nu`` on the grid given the mixing variables ``omega``; the default grid starts at
    ``DOF_FLOOR``."""
    values = dof_grid() if grid is None else np.asarray(grid, dtype=float)
    half = values[:, None] / 2
    loglik = (half * np.log(half) - gammaln(half) - (half + 1) * np.log(omega) - half / omega).sum(
        axis=1
    )
    weight = np.exp(loglik - loglik.max())
    return float(rng.choice(values, p=weight / weight.sum()))
