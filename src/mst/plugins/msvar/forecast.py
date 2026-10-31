"""Conditional forecast, severity and moments of a scenario under the MS-VAR.

A repetition takes one identified parameter draw and its regime path over the horizon: the
path is drawn from the transition matrix out of the last month of history — unless the
scenario names a regime element among its driving shocks, which fixes the path to that
regime over the whole horizon, the event asserting the regime the way the scheme's amended
section 3 puts the regime part of the event beside its shocks. Given the path the model is
a linear-Gaussian VAR with time-varying coefficients, and the constraints are met with the
shared machinery of ``mst.model_sdk.conditional`` exactly as the BVAR meets them: the
loadings of each constraint functional on each month's structural shocks follow the regime
path, the image of the driving shocks is normal, its exact rows are fixed and its interval
rows drawn from the box-truncated normal, and the scenario is infeasible when its rows are
linearly dependent (Antolin-Diaz, Petrella and Rubio-Ramirez, 2021, Journal of Monetary
Economics 117).

The severity and the moments are measured on the unconditional run, where the regime path
is drawn from the kernel: each repetition carries its own parameter draw and its own
regime path, so the covariance the severity returns is the mixture's — the mean of the
covariances plus the covariance of the means, the spread between the regime paths counted
in. The platform's ``dist`` consumes the per-repetition moments as the components of the
mixture, one component per (parameter draw, regime path).
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from mst.macro.panel import shift
from mst.model_sdk.conditional import (
    SWEEPS,
    InfeasibleScenario,
    dependent_rows,
    draw_driving,
)
from mst.model_sdk.contracts import MacroSimulation, ScenarioPlan, Severity

from .estimate import Posterior
from .var import iterate

__all__ = ["InfeasibleScenario", "constraint_moments", "severity", "simulate"]


@dataclass(frozen=True)
class _Plan:
    horizon: int
    terms: list[list[tuple[int, int, float]]]  # per constraint: (variable, month, weight)
    low: np.ndarray
    high: np.ndarray
    driving: np.ndarray  # column indices of the driving economic shocks
    regime: int | None  # the regime element of the event, fixed over the horizon


def _plan(model: Posterior, plan: ScenarioPlan) -> _Plan:
    if model.rotation is None:
        raise ValueError("the posterior is not identified; run mst.plugins.msvar.identify first")
    index = {id_: i for i, id_ in enumerate(model.ids)}
    terms, problems = [], []
    for constraint in plan.constraints:
        row = []
        for term in constraint.terms:
            if term.variable not in index:
                problems.append(f"unknown variable {term.variable}")
            else:
                row.append((index[term.variable], term.month, term.weight))
        terms.append(row)
    regimes = [s for s in plan.driving_shocks if s in model.regime_ids]
    if len(set(regimes)) > 1:
        problems.append(f"conflicting regime elements {', '.join(regimes)}")
    known = model.shocks + model.regime_ids
    unknown = [s for s in plan.driving_shocks if s not in known]
    problems += [f"unknown shock {s}" for s in unknown]
    if problems:
        raise ValueError("; ".join(problems))
    driving = [model.shocks.index(s) for s in plan.driving_shocks if s in model.shocks]
    return _Plan(
        horizon=plan.horizon,
        terms=terms,
        low=np.array([c.low for c in plan.constraints]),
        high=np.array([c.high for c in plan.constraints]),
        driving=np.array(driving or range(len(model.shocks)), dtype=int),
        regime=model.regime_ids.index(regimes[0]) if regimes else None,
    )


@dataclass(frozen=True)
class _Repetition:
    history: np.ndarray  # (p, n)
    lags: np.ndarray  # (H, p, n, n) the month's regime
    intercept: np.ndarray  # (H, n) mu
    impacts: np.ndarray  # (H, n, n) B_t
    mean: np.ndarray  # (H, n)


def _regime_path(model: Posterior, draw: int, horizon: int, rng, regime: int | None):
    """The regime path of the horizon: from the kernel, or the event's fixed regime."""
    if regime is not None:
        return np.full(horizon, regime, dtype=int)
    path = np.empty(horizon, dtype=int)
    state = int(model.states[draw, -1])
    for t in range(horizon):
        state = _choice(model.transition[draw, state], rng)
        path[t] = state
    return path


def _choice(weights: np.ndarray, rng) -> int:
    """One draw from a distribution given by weights; they may not sum exactly to one."""
    cumulated = np.cumsum(weights)
    return min(int(np.searchsorted(cumulated, rng.random() * cumulated[-1])), len(weights) - 1)


def _repetition(model: Posterior, horizon: int, rng, regime: int | None) -> _Repetition:
    """One repetition: a parameter draw and its regime path, with the month's matrices."""
    draw = int(rng.integers(model.draws))
    path = _regime_path(model, draw, horizon, rng, regime)
    history = model.history[draw]
    lags = model.lags[draw][path]
    intercept = model.const[draw][path]
    impacts = np.einsum(
        "tij,tj,tjk->tik",
        model.chol[draw][path],
        np.exp(model.logvar[draw][path] / 2),
        model.rotation[draw][path],
    )
    return _Repetition(
        history=history,
        lags=lags,
        intercept=intercept,
        impacts=impacts,
        mean=iterate(history, lags, intercept, np.zeros((horizon, len(model.ids)))),
    )


def _slabs(rep: _Repetition, needed: set[int]) -> dict[int, np.ndarray]:
    """``D_t(s)`` for the needed months: how the reduced-form error of ``s <= t`` moves ``y_t``.

    Under a regime path the response of ``y_t`` to the error of month ``s`` depends on the
    regimes between them, so the responses are tabulated forward with a rolling buffer of
    the last ``p`` months: ``D_t(t) = I`` and ``D_t(s) = sum_l A_{l, s_t} D_{t-l}(s)`` —
    the same recursion the time-invariant ``Psi`` of a single-regime VAR hides.
    """
    n = rep.mean.shape[1]
    p = rep.lags.shape[1]
    recent: list[np.ndarray] = []  # the slabs of months t-1, ..., t-p
    found: dict[int, np.ndarray] = {}
    for t in range(1, max(needed) + 1):
        slab = np.zeros((t, n, n))
        slab[t - 1] = np.eye(n)
        for lag in range(1, min(p, t - 1) + 1):
            older = recent[-lag]  # the slab of month t - lag
            slab[: t - lag] += rep.lags[t - 1, lag - 1] @ older
        recent.append(slab)
        if len(recent) > p:
            recent.pop(0)
        if t in needed:
            found[t] = slab
    return found


def _loadings(plan: _Plan, rep: _Repetition) -> np.ndarray:
    """``(m, H, n)``: how a unit of each structural shock in each month moves each constraint."""
    n = rep.mean.shape[1]
    z = np.zeros((len(plan.terms), plan.horizon, n))
    needed = {month for row in plan.terms for _, month, _ in row}
    if not needed:
        return z
    slabs = _slabs(rep, needed)
    for c, row in enumerate(plan.terms):
        for variable, month, weight in row:
            # y_{month} moves by D(s) B_s per unit of eps_s, s = 1 .. month.
            z[c, :month] += weight * np.einsum(
                "sn,snk->sk", slabs[month][:, variable, :], rep.impacts[:month]
            )
    return z


def _level(plan: _Plan, mean: np.ndarray) -> np.ndarray:
    return np.array([sum(w * mean[t - 1, i] for i, t, w in row) for row in plan.terms])


def constraint_moments(model: Posterior, plan: ScenarioPlan, *, draws: int, seed: int):
    """The mean and covariance of the constraint functionals under every repetition.

    A repetition carries one parameter draw and one regime path of the unconditional run;
    given both, ``W y`` is normal, so the platform's mixture over the repetitions is the
    mixture over the regimes. Raises ``ValueError`` when the scenario has no constraints or
    names a variable or shock the model lacks.
    """
    means, covs, _ = _moments(model, plan, draws, seed)
    return means, covs


def _moments(model: Posterior, plan: ScenarioPlan, draws: int, seed: int):
    """The per-repetition moments and the repetitions' generators, in the order of the
    constraints.

    The generators are returned half-consumed — ``_repetition`` has drawn the parameter draw
    and the regime path from each one — because ``severity`` continues every repetition's
    stream to place one ``W y`` on it; everything downstream of the moments stays on the
    same stream this way.
    """
    chosen = _plan(model, plan)
    if not chosen.terms:
        raise ValueError("a scenario without constraints has no moments to measure")
    means, covs, rngs = [], [], []
    for r in range(draws):
        rng = np.random.default_rng([seed, r])
        rep = _repetition(model, chosen.horizon, rng, None)
        z = _loadings(chosen, rep).reshape(len(chosen.terms), -1)
        means.append(_level(chosen, rep.mean))
        covs.append(z @ z.T)
        rngs.append(rng)
    return np.array(means), np.array(covs), rngs


def simulate(
    model: Posterior, plan: ScenarioPlan, *, draws: int, seed: int, sweeps: int = SWEEPS
) -> MacroSimulation:
    """``draws`` paths of the macro variables under the scenario, one regime path each.

    Raises ``InfeasibleScenario`` naming the constraints the driving shocks cannot meet.
    """
    chosen = _plan(model, plan)
    n, horizon = len(model.ids), chosen.horizon
    driving = np.zeros(n, dtype=bool)
    driving[chosen.driving] = True
    paths = np.empty((draws, horizon + 1, n))
    history = np.empty((draws, model.lags_count, n))
    for r in range(draws):
        rng = np.random.default_rng([seed, r])
        rep = _repetition(model, horizon, rng, chosen.regime)
        eps = rng.standard_normal((horizon, n))
        if chosen.terms:
            z = _loadings(chosen, rep)
            z_all = z.reshape(len(z), -1)
            z_d = z[:, :, driving].reshape(len(z), -1)
            bad = dependent_rows(z_d, z_all)
            if bad:
                raise InfeasibleScenario(bad)
            offset = _level(chosen, rep.mean) + np.einsum(
                "csj,sj->c", z[:, :, ~driving], eps[:, ~driving]
            )
            drawn = draw_driving(
                z_d, chosen.low - offset, chosen.high - offset, rng, sweeps, every=z_all
            )
            eps[:, driving] = drawn.reshape(horizon, driving.sum())
        shocks = np.einsum("tij,tj->ti", rep.impacts, eps)
        future = iterate(rep.history, rep.lags, rep.intercept, shocks)
        paths[r] = np.vstack([rep.history[-1], future])
        history[r] = rep.history
    last = model.months[-1]
    return MacroSimulation(
        ids=model.ids,
        months=[shift(last, t) for t in range(horizon + 1)],
        paths=paths,
        history=history,
    )


def severity(model: Posterior, plan: ScenarioPlan, *, draws: int, seed: int) -> Severity:
    """The severity of the plan under the unconditional mixture of the model.

    The per-repetition moments come from ``constraint_moments``; given a repetition ``W y``
    is normal, so ``pi_c`` averages normal probabilities, and the mean and covariance of
    ``W y`` are the exact moments of the mixture — the mean of the covariances plus the
    covariance of the means, which counts the spread between the parameter draws and the
    regime paths both. ``exceeded`` draws one ``W y`` per repetition.
    """
    chosen = _plan(model, plan)
    if not chosen.terms:
        raise ValueError("a scenario without constraints has no severity")
    means, covs, rngs = _moments(model, plan, draws, seed)
    midpoint = (chosen.low + chosen.high) / 2
    probabilities = np.array(
        [
            ndtr((midpoint - mean) / np.sqrt(np.diag(cov)))
            for mean, cov in zip(means, covs, strict=True)
        ]
    )
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
        probabilities=probabilities.mean(axis=0),
        distance=distance,
        exceeded=float(farther.mean()),
    )
