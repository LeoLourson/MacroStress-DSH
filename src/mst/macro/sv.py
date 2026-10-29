"""Gibbs steps for stochastic volatility with Student tails (section 4.1).

An orthogonalised error is ``e_t = exp(h_t / 2) sqrt(omega_t) z_t`` with ``h_t = h_{t-1} +
s xi_t`` and ``omega_t ~ InvGamma(nu / 2, nu / 2)``. Given ``h`` and ``omega`` each step below
draws one block from its full conditional.

``log(e_t^2 / omega_t) = h_t + log z_t^2``; ``log z_t^2`` is approximated by the seven-normal
mixture of Kim, Shephard and Chib (1998, Review of Economic Studies 65(3), table 4), after
which ``h`` is Gaussian and drawn at once from its banded precision (Chan and Jeliazkov, 2009).
"""

import numpy as np
from scipy.linalg import cho_solve_banded, cholesky_banded, solve_banded
from scipy.special import gammaln

# Kim, Shephard and Chib (1998), table 4: weights, means (before the -1.2704 shift) and
# variances of the components approximating log chi^2_1.
MIX_WEIGHT = np.array([0.00730, 0.10556, 0.00002, 0.04395, 0.34001, 0.24566, 0.25750])
MIX_MEAN = np.array([-10.12999, -3.97281, -8.56686, 2.77786, 0.61942, 1.79518, -1.08819]) - 1.2704
MIX_VAR = np.array([5.79596, 2.61369, 5.17950, 0.16735, 0.64009, 0.34023, 1.26261])

# Uniform prior of nu over these degrees of freedom; 50 is close to normal.
DOF_GRID = np.arange(3.0, 51.0)
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


def draw_mixing(scaled_e2: np.ndarray, nu: float, rng) -> np.ndarray:
    """``omega_t`` given ``e_t^2 / exp(h_t)``."""
    return (nu + scaled_e2) / 2 / rng.gamma((nu + 1) / 2, size=len(scaled_e2))


def draw_dof(omega: np.ndarray, rng) -> float:
    """``nu`` on ``DOF_GRID`` given the mixing variables ``omega``."""
    half = DOF_GRID[:, None] / 2
    loglik = (half * np.log(half) - gammaln(half) - (half + 1) * np.log(omega) - half / omega).sum(
        axis=1
    )
    weight = np.exp(loglik - loglik.max())
    return float(rng.choice(DOF_GRID, p=weight / weight.sum()))
