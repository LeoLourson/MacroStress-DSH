"""The conditional forecast and the constraint law of a scenario under the MS-VAR.

A repetition takes one identified parameter draw and its regime path over the horizon: the
path is drawn from the transition matrix out of the last month of history — unless the
scenario names a regime element among its driving shocks, which fixes the path to that
regime over the whole horizon, the event asserting the regime the way the canon puts the
regime part of the event beside its shocks. Given the path the model is a linear-Gaussian
VAR with time-varying coefficients, and the constraints are met with the shared machinery
of ``mst.model_sdk.conditional`` exactly as the BVAR meets them: the loadings of each
constraint functional on each month's structural shocks follow the regime path, the image
of the driving shocks is normal, its exact rows are fixed and its interval rows drawn from
the box-truncated normal, and the scenario is infeasible when its rows are linearly
dependent (Antolin-Diaz, Petrella and Rubio-Ramirez, 2021, Journal of Monetary Economics
117).

The law of the constraint functionals — what the price of a scenario mixes — depends on
what the plan's driving shocks leave outside ``R``. With no driving shocks the regime path
is a hidden coordinate: each parameter draw carries ``LATENT_CANDIDATES`` candidate paths,
each row of the law holding the gaussian law of ``W y`` under that path, and the group's
rows mix into ``P(S | theta)``. With driving shocks named the path belongs to ``R`` — the
repetition draws it and the free shocks, and one row reports the law given both. A regime
event forces the path instead: one row carries the path's prior weight ``pi_theta(k,..,k)``,
so the price of fixing the regime is the mean of ``-log pi_theta(k,..,k)`` the canon asks
for. The streams are separate — ``[seed, r, 0]`` draws the parameters, ``[seed, r, 1]`` the
regime paths, ``[seed, r, 2]`` the free shocks — so a scenario and its softened variant
share everything the conditions do not touch.
"""

from dataclasses import dataclass

import numpy as np

from mst.macro.panel import shift
from mst.model_sdk.conditional import (
    SWEEPS,
    InfeasibleScenario,
    dependent_rows,
    draw_driving,
)
from mst.model_sdk.contracts import (
    LATENT_CANDIDATES,
    ConstraintLaw,
    MacroSimulation,
    ScenarioPlan,
)

from .estimate import Posterior
from .var import iterate

__all__ = ["InfeasibleScenario", "constraint_law", "simulate"]


#: The name of the regime path as one coordinate of ``omega``: the hidden coordinate the
#: MS-VAR declares, which a scenario names among its causes to explain the event through
#: the regime.
REGIME_PATH = "regime_path"


@dataclass(frozen=True)
class _Plan:
    horizon: int
    terms: list[list[tuple[int, int, float]]]  # per constraint: (variable, month, weight)
    low: np.ndarray
    high: np.ndarray
    driving: np.ndarray  # column indices of the economic shocks named as causes
    regime: int | None  # the regime element of the event, fixed over the horizon
    regime_cause: bool  # the regime path is named as a cause

    @property
    def regime_in_r(self) -> bool:
        """Whether the regime path belongs to ``R``: a cause is missing, so the law of the
        path stays as it was."""
        return not self.regime_cause


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
    known = model.shocks + [REGIME_PATH]
    unknown = [s for s in plan.causes if s not in known]
    problems += [f"unknown cause {s}: the model knows {', '.join(known)}" for s in unknown]
    regime = None
    if plan.regime_event is not None:
        if plan.regime_event not in model.regime_ids:
            problems.append(f"unknown regime element {plan.regime_event}")
        else:
            regime = model.regime_ids.index(plan.regime_event)
        if plan.causes and REGIME_PATH not in plan.causes:
            problems.append(
                f"the regime event {plan.regime_event} needs {REGIME_PATH} among its causes, "
                "or no causes at all: the event fixes the regime, so its law cannot stay "
                "as it was (lemma 2 of the canon)"
            )
    if problems:
        raise ValueError("; ".join(problems))
    driving = [model.shocks.index(s) for s in plan.causes if s in model.shocks]
    return _Plan(
        horizon=plan.horizon,
        terms=terms,
        low=np.array([c.low for c in plan.constraints]),
        high=np.array([c.high for c in plan.constraints]),
        driving=np.array(driving or range(len(model.shocks)), dtype=int),
        regime=regime,
        regime_cause=REGIME_PATH in plan.causes,
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
    return _repetition_of(model, draw, path)


def _repetition_of(model: Posterior, draw: int, path: np.ndarray) -> _Repetition:
    """The repetition of a chosen parameter draw and regime path."""
    horizon = len(path)
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


def _path_probability(model: Posterior, draw: int, regime: int, horizon: int) -> float:
    """``pi_theta(k, .., k)``: the prior probability of the regime path staying in ``regime``."""
    state = int(model.states[draw, -1])
    probability = 1.0
    for _ in range(horizon):
        probability *= model.transition[draw, state, regime]
        state = regime
    return probability


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


def constraint_law(
    model: Posterior,
    plan: ScenarioPlan,
    *,
    draws: int,
    seed: int,
    candidates: int = LATENT_CANDIDATES,
) -> ConstraintLaw | None:
    """The law of the constraint functionals given ``R``, one group of rows per draw of it.

    The streams are separate: ``[seed, r, 0]`` draws the parameter set, ``[seed, r, 1]`` the
    regime paths, ``[seed, r, 2]`` the free shocks, so two scenarios that differ only in
    their bounds share the rows. With no driving shocks the regime path is a hidden
    coordinate and every group carries ``candidates`` of them, each row the gaussian law of
    ``W y`` under its path. With driving shocks named the path and the free shocks belong to
    ``R`` and one row reports the law given both. A regime event forces the path and its row
    carries the prior weight ``pi_theta(k, .., k)``, whatever driving shocks are named: the
    price of the event is then the mean of ``-log pi_theta(k, .., k)`` the canon asks for.
    ``None`` when there is nothing to price — no constraints and no regime event.
    """
    chosen = _plan(model, plan)
    if not chosen.terms and chosen.regime is None:
        return None
    n = len(model.ids)
    driving = np.zeros(n, dtype=bool)
    driving[chosen.driving] = True
    # The regime path belongs to ``R`` only when other causes are named without it: then the
    # repetition draws the path and one row reports the law given it. Unnamed causes — or
    # ``regime_path`` itself among them — leave the path a hidden coordinate, and every
    # group carries its candidates.
    named = bool(plan.causes) and chosen.regime_in_r
    means, covs, log_weights, groups = [], [], [], []
    for r in range(draws):
        rng_theta = np.random.default_rng([seed, r, 0])
        draw = int(rng_theta.integers(model.draws))
        if chosen.regime is not None:
            path = np.full(chosen.horizon, chosen.regime, dtype=int)
            rep = _repetition_of(model, draw, path)
            eps_free = (
                np.random.default_rng([seed, r, 2])
                .standard_normal((chosen.horizon, int((~driving).sum())))
                .reshape(-1)
                if driving.any()
                else None
            )
            mean, cov = _law_row(chosen, rep, driving, eps_free)
            means.append(mean)
            covs.append(cov)
            log_weights.append(np.log(_path_probability(model, draw, chosen.regime, chosen.horizon)))
            groups.append(r)
            continue
        rng_path = np.random.default_rng([seed, r, 1])
        if named:
            path = _regime_path(model, draw, chosen.horizon, rng_path, None)
            rep = _repetition_of(model, draw, path)
            eps_free = (
                np.random.default_rng([seed, r, 2])
                .standard_normal((plan.horizon, int((~driving).sum())))
                .reshape(-1)
            )
            mean, cov = _law_row(chosen, rep, driving, eps_free)
            means.append(mean)
            covs.append(cov)
            log_weights.append(0.0)
            groups.append(r)
            continue
        for _ in range(candidates):
            path = _regime_path(model, draw, chosen.horizon, rng_path, None)
            rep = _repetition_of(model, draw, path)
            mean, cov = _law_row(chosen, rep, driving, None)
            means.append(mean)
            covs.append(cov)
            log_weights.append(0.0)
            groups.append(r)
    m = len(chosen.terms)
    return ConstraintLaw(
        means=np.asarray(means, dtype=float).reshape(len(means), m) if means
        else np.zeros((0, m)),
        covs=np.asarray(covs, dtype=float).reshape(len(covs), m, m) if covs
        else np.zeros((0, m, m)),
        log_weight=np.asarray(log_weights, dtype=float),
        group=np.asarray(groups, dtype=int),
    )


def _law_row(chosen: _Plan, rep: _Repetition, driving: np.ndarray, eps_free):
    """The mean and covariance of ``W y`` under one repetition.

    ``eps_free`` are the drawn non-driving shocks — part of ``R`` when the plan names causes;
    without them every shock is free and the covariance counts all of it.
    """
    m = len(chosen.terms)
    if m == 0:
        return np.zeros(0), np.zeros((0, 0))
    z = _loadings(chosen, rep)
    z_flat = z.reshape(m, -1)
    mean = _level(chosen, rep.mean)
    if not driving.any():
        return mean, z_flat @ z_flat.T
    z_free = z[:, :, ~driving].reshape(len(chosen.terms), -1)
    z_driving = z[:, :, driving].reshape(len(chosen.terms), -1)
    if eps_free is not None:
        mean = mean + z_free @ eps_free
    return mean, z_driving @ z_driving.T


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
