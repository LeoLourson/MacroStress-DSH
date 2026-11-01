"""The conditional forecast and the constraint law of a scenario.

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

from mst.macro.panel import shift
from mst.model_sdk.conditional import (
    SWEEPS,
    InfeasibleScenario,
    box_gibbs,
    dependent_rows,
    draw_driving,
    truncated_normal,
)
from mst.model_sdk.contracts import (
    LATENT_CANDIDATES,
    ConstraintLaw,
    MacroSimulation,
    ScenarioPlan,
)

from .estimate import Posterior
from .var import iterate, responses

#: The name of the volatility path as one coordinate of ``omega``: the hidden coordinate the
#: BVAR declares, which a scenario names among its causes to keep it out of ``R``.
VOLATILITY_PATH = "volatility_path"

Simulation = MacroSimulation
"""The simulation of the BVAR, in the platform's shared shape."""

__all__ = [
    "InfeasibleScenario",
    "Simulation",
    "box_gibbs",
    "constraint_law",
    "dependent_rows",
    "draw_driving",
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


def _plan(model: Posterior, scenario: ScenarioPlan) -> _Plan:
    if model.rotation is None:
        raise ValueError("the posterior is not identified; run mst.plugins.bvar.identify first")
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
    known = model.shocks + [VOLATILITY_PATH]
    unknown = [s for s in scenario.causes if s not in known]
    problems += [f"unknown cause {s}: the model knows {', '.join(known)}" for s in unknown]
    if problems:
        raise ValueError("; ".join(problems))
    driving = [model.shocks.index(s) for s in scenario.causes if s in model.shocks]
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


def _repetition(model: Posterior, horizon: int, rng, draw: int | None = None) -> _Repetition:
    if draw is None:
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


def constraint_law(
    model: Posterior,
    scenario: ScenarioPlan,
    *,
    draws: int,
    seed: int,
    candidates: int = LATENT_CANDIDATES,
) -> ConstraintLaw | None:
    """The law of the constraint functionals given ``R``, one group of rows per draw of it.

    The streams are separate: ``[seed, r, 0]`` draws the parameter set, ``[seed, r, 1]`` the
    volatility path, ``[seed, r, 2]`` the free shocks, so two scenarios that differ only in
    their bounds share the rows. With no driving shocks the volatility path is a hidden
    coordinate and every group carries ``candidates`` of them, each row the gaussian law of
    ``W y`` under its path — a tail event then prices the volatility it needs instead of
    riding the moderate paths a single draw happens to pick. With driving shocks named the
    volatility path and the free shocks belong to ``R`` and one row reports the law given
    both. ``None`` when the scenario has no constraints.
    """
    plan = _plan(model, scenario)
    if not plan.terms:
        return None
    n = len(model.ids)
    # The law reads the causes themselves, without the fill ``_plan`` gives the imposition:
    # a shock the causes do not name belongs to ``R`` and is drawn into the mean.
    driving = np.zeros(n, dtype=bool)
    driving[[model.shocks.index(s) for s in scenario.causes if s in model.shocks]] = True
    # The volatility path belongs to ``R`` only when other causes are named without it:
    # then the repetition draws the path and one row reports the law given it. Unnamed
    # causes — or ``volatility_path`` itself among them — leave the path a hidden coordinate,
    # and every group carries its candidates.
    named = bool(scenario.causes) and VOLATILITY_PATH not in scenario.causes
    means, covs, log_weights, groups = [], [], [], []
    for r in range(draws):
        draw = int(np.random.default_rng([seed, r, 0]).integers(model.draws))
        free = (
            np.random.default_rng([seed, r, 2])
            .standard_normal((plan.horizon, int((~driving).sum())))
            .reshape(-1)
            if scenario.causes
            else None
        )
        if named:
            rep = _repetition(model, plan.horizon, np.random.default_rng([seed, r, 1]), draw)
            mean, cov = _law_row(plan, rep, driving, free)
            means.append(mean)
            covs.append(cov)
            log_weights.append(0.0)
            groups.append(r)
            continue
        rng_volatility = np.random.default_rng([seed, r, 1])
        for _ in range(candidates):
            rep = _repetition(model, plan.horizon, rng_volatility, draw)
            mean, cov = _law_row(plan, rep, driving, free)
            means.append(mean)
            covs.append(cov)
            log_weights.append(0.0)
            groups.append(r)
    m = len(plan.terms)
    return ConstraintLaw(
        means=np.asarray(means, dtype=float).reshape(len(means), m) if means
        else np.zeros((0, m)),
        covs=np.asarray(covs, dtype=float).reshape(len(covs), m, m) if covs
        else np.zeros((0, m, m)),
        log_weight=np.asarray(log_weights, dtype=float),
        group=np.asarray(groups, dtype=int),
    )


def _law_row(plan: _Plan, rep: _Repetition, driving: np.ndarray, eps_free):
    """The mean and covariance of ``W y`` under one repetition.

    ``eps_free`` are the drawn non-driving shocks — part of ``R`` when the scenario names
    causes; without them every shock is free and the covariance counts all of it.
    """
    m = len(plan.terms)
    if m == 0:
        return np.zeros(0), np.zeros((0, 0))
    z = _loadings(plan, rep)
    z_flat = z.reshape(m, -1)
    mean = _level(plan, rep.mean)
    if not driving.any():
        return mean, z_flat @ z_flat.T
    z_free = z[:, :, ~driving].reshape(len(plan.terms), -1)
    z_driving = z[:, :, driving].reshape(len(plan.terms), -1)
    if eps_free is not None:
        mean = mean + z_free @ eps_free
    return mean, z_driving @ z_driving.T


def simulate(
    model: Posterior, scenario: ScenarioPlan, *, draws: int, seed: int, sweeps: int = SWEEPS
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
            z_all = z.reshape(len(z), -1)
            z_d = z[:, :, driving].reshape(len(z), -1)
            bad = dependent_rows(z_d, z_all)
            if bad:
                raise InfeasibleScenario(bad)
            offset = _level(plan, rep.mean) + np.einsum(
                "csj,sj->c", z[:, :, ~driving], eps[:, ~driving]
            )
            drawn = draw_driving(
                z_d, plan.low - offset, plan.high - offset, rng, sweeps, every=z_all
            )
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
