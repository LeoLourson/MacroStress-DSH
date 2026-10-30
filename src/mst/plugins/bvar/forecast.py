"""Conditional forecast (section 4.3) and severity (section 4.4) of a scenario.

A repetition takes one identified parameter draw and draws the future volatilities: ``h_t =
h_{t-1} + s xi_t`` from the last month of history and ``omega_t ~ InvGamma(nu / 2, nu / 2)``.
The impact matrix of month ``t`` is ``B_t = C diag(exp(h_t / 2) sqrt(omega_t)) Q``, and the
path is ``y = mu + Phi eps`` with standard normal ``eps``; with constant volatility ``B_t``
is the ``B`` of section 4.1. ``Phi`` is never formed: only the rows ``W Phi`` the constraints
need are, and the path is iterated from the shocks.

Driving shocks are drawn given the constraints ``a <= W y <= b``. With ``Z = W Phi_D`` the
image ``v = Z eps_D`` is normal; its exact rows are fixed, its interval rows are drawn from the
box-truncated normal by the Gibbs sampler of Geweke (1991), and ``eps_D`` given ``v`` is
``eta + Z' (Z Z')^{-1} (v - Z eta)`` with standard normal ``eta`` (Antolin-Diaz, Petrella and
Rubio-Ramirez, 2021, Journal of Monetary Economics 117). The scenario is infeasible when the
rows of ``Z`` are linearly dependent.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from mst.macro.panel import shift
from mst.model_sdk.conditional import (
    SWEEPS,
    InfeasibleScenario,
    box_gibbs,
    dependent_rows,
    draw_driving,
    truncated_normal,
)
from mst.model_sdk.contracts import MacroSimulation, Severity
from mst.scenario.contracts import Scenario

from .estimate import Posterior
from .var import iterate, responses

Simulation = MacroSimulation
"""The simulation of the BVAR, in the platform's shared shape."""

__all__ = [
    "InfeasibleScenario",
    "Simulation",
    "Severity",
    "box_gibbs",
    "dependent_rows",
    "draw_driving",
    "severity",
    "simulate",
    "truncated_normal",
]


@dataclass(frozen=True)
class _Plan:
    horizon: int
    terms: list[list[tuple[int, int, float]]]  # per constraint: (variable, month, weight)
    low: np.ndarray
    high: np.ndarray
    driving: np.ndarray  # column indices of the driving shocks


def _plan(model: Posterior, scenario: Scenario) -> _Plan:
    if model.rotation is None:
        raise ValueError("the posterior is not identified; run mst.macro.identify first")
    index = {id_: i for i, id_ in enumerate(model.ids)}
    terms, problems = [], []
    for constraint in scenario.constraints:
        row = []
        for term in constraint.terms:
            if term.variable not in index:
                problems.append(f"unknown variable {term.variable}")
            else:
                row.append((index[term.variable], term.month, term.weight))
        terms.append(row)
    unknown = [s for s in scenario.driving_shocks if s not in model.shocks]
    problems += [f"unknown shock {s}" for s in unknown]
    if problems:
        raise ValueError("; ".join(problems))
    driving = [model.shocks.index(s) for s in scenario.driving_shocks]
    return _Plan(
        horizon=scenario.horizon,
        terms=terms,
        low=np.array([c.low for c in scenario.constraints]),
        high=np.array([c.high for c in scenario.constraints]),
        driving=np.array(driving or range(len(model.ids)), dtype=int),
    )


@dataclass(frozen=True)
class _Repetition:
    lags: np.ndarray
    history: np.ndarray
    intercept: np.ndarray  # (H, n)
    impacts: np.ndarray  # (H, n, n) B_t
    mean: np.ndarray  # (H, n) mu
    psi: np.ndarray  # (H, n, n)


def _future(model: Posterior, horizon: int) -> np.ndarray:
    last = model.months[-1]
    return np.array(
        [[d.value(shift(last, t)) for d in model.dummies] for t in range(1, horizon + 1)]
    ).reshape(horizon, len(model.dummies))


def _repetition(model: Posterior, horizon: int, rng) -> _Repetition:
    draw = int(rng.integers(model.draws))
    n = len(model.ids)
    h = model.logvar[draw, -1] + np.cumsum(
        model.volvol[draw] * rng.standard_normal((horizon, n)), axis=0
    )
    dof = model.dof[draw]
    finite = np.isfinite(dof)
    omega = np.ones((horizon, n))
    omega[:, finite] = dof[finite] / 2 / rng.gamma(dof[finite] / 2, size=(horizon, finite.sum()))
    scale = np.exp(h / 2) * np.sqrt(omega)
    impacts = np.einsum("ij,tj,jk->tik", model.chol[draw], scale, model.rotation[draw])
    intercept = model.const[draw] + _future(model, horizon) @ model.dummy[draw].T
    lags, history = model.lags[draw], model.history[draw]
    return _Repetition(
        lags=lags,
        history=history,
        intercept=intercept,
        impacts=impacts,
        mean=iterate(history, lags, intercept, np.zeros((horizon, n))),
        psi=responses(lags, horizon),
    )


def _loadings(plan: _Plan, rep: _Repetition) -> np.ndarray:
    """``(m, H, n)``: ``W Phi`` arranged by the month and column of each shock."""
    n = rep.mean.shape[1]
    z = np.zeros((len(plan.terms), plan.horizon, n))
    for c, row in enumerate(plan.terms):
        for variable, month, weight in row:
            # y_{month} moves by Psi[month - s] B_s per unit of eps_s, s = 1 .. month.
            reach = rep.psi[month - 1 :: -1, variable, :]
            z[c, :month] += weight * np.einsum("sn,snj->sj", reach, rep.impacts[:month])
    return z


def _level(plan: _Plan, mean: np.ndarray) -> np.ndarray:
    return np.array([sum(w * mean[t - 1, i] for i, t, w in row) for row in plan.terms])


def simulate(
    model: Posterior, scenario: Scenario, *, draws: int, seed: int, sweeps: int = SWEEPS
) -> Simulation:
    """Steps 1-3 of section 7: ``draws`` paths of the macro variables under the scenario.

    Raises ``InfeasibleScenario`` naming the constraints the driving shocks cannot meet.
    """
    plan = _plan(model, scenario)
    n, horizon = len(model.ids), plan.horizon
    driving = np.zeros(n, dtype=bool)
    driving[plan.driving] = True
    paths = np.empty((draws, horizon + 1, n))
    history = np.empty((draws, model.lags_count, n))
    for r in range(draws):
        rng = np.random.default_rng([seed, r])
        rep = _repetition(model, horizon, rng)
        eps = rng.standard_normal((horizon, n))
        if plan.terms:
            z = _loadings(plan, rep)
            z_d = z[:, :, driving].reshape(len(z), -1)
            bad = dependent_rows(z_d, z.reshape(len(z), -1))
            if bad:
                raise InfeasibleScenario(bad)
            offset = _level(plan, rep.mean) + np.einsum(
                "csj,sj->c", z[:, :, ~driving], eps[:, ~driving]
            )
            drawn = draw_driving(z_d, plan.low - offset, plan.high - offset, rng, sweeps)
            eps[:, driving] = drawn.reshape(horizon, driving.sum())
        shocks = np.einsum("tij,tj->ti", rep.impacts, eps)
        future = iterate(rep.history, rep.lags, rep.intercept, shocks)
        paths[r] = np.vstack([rep.history[-1], future])
        history[r] = rep.history
    last = model.months[-1]
    return Simulation(
        ids=model.ids,
        months=[shift(last, t) for t in range(horizon + 1)],
        paths=paths,
        history=history,
    )


def severity(model: Posterior, scenario: Scenario, *, draws: int, seed: int) -> Severity:
    """Section 4.4 over ``draws`` unconditional repetitions.

    Given a repetition ``W y`` is normal with mean ``W mu`` and covariance ``W Phi Phi' W'``,
    so ``pi_c`` averages normal probabilities and the mean and covariance of ``W y`` are the
    exact moments of that mixture; ``exceeded`` draws one ``W y`` per repetition.
    """
    plan = _plan(model, scenario)
    if not plan.terms:
        raise ValueError("a scenario without constraints has no severity")
    midpoint = (plan.low + plan.high) / 2
    means, covs, probabilities = [], [], []
    rngs = [np.random.default_rng([seed, r]) for r in range(draws)]
    for rng in rngs:
        rep = _repetition(model, plan.horizon, rng)
        z = _loadings(plan, rep).reshape(len(plan.terms), -1)
        mean, cov = _level(plan, rep.mean), z @ z.T
        means.append(mean)
        covs.append(cov)
        probabilities.append(ndtr((midpoint - mean) / np.sqrt(np.diag(cov))))
    means, covs = np.array(means), np.array(covs)
    centre = means.mean(axis=0)
    spread = covs.mean(axis=0) + np.cov(means.T, bias=True).reshape(covs.shape[1:])
    precision = np.linalg.inv(spread)
    gap = midpoint - centre
    distance = float(np.sqrt(gap @ precision @ gap))
    samples = (
        np.array(
            [rng.multivariate_normal(m, c) for rng, m, c in zip(rngs, means, covs, strict=True)]
        )
        - centre
    )
    farther = np.einsum("ri,ij,rj->r", samples, precision, samples) >= distance**2
    return Severity(
        probabilities=np.mean(probabilities, axis=0),
        distance=distance,
        exceeded=float(farther.mean()),
    )
