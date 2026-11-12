"""Conditional forecasts and constraint laws for the MS-VAR."""

from dataclasses import dataclass, replace

import numpy as np

from mst.macro.panel import shift
from mst.model_sdk.conditional import (
    SWEEPS,
    InfeasibleScenario,
    dependent_rows,
    draw_driving,
    log_condition_weight,
    pick_latent,
)
from mst.model_sdk.contracts import (
    DRIVING_LIMIT,
    LATENT_CANDIDATES,
    ConstraintLaw,
    MacroSimulation,
    ScenarioPlan,
)
from mst.model_sdk.dist import ContradictsModels
from mst.model_sdk.interventions import (
    affine_constraints_under_default,
    constraints_under_default,
)

from .estimate import Posterior
from .var import iterate

__all__ = ["InfeasibleScenario", "constraint_law", "simulate"]


@dataclass(frozen=True)
class _Plan:
    horizon: int
    terms: list[list[tuple[int, int, float]]]  # per constraint: (variable, month, weight)
    low: np.ndarray
    high: np.ndarray


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
    if problems:
        raise ValueError("; ".join(problems))
    return _Plan(
        horizon=plan.horizon,
        terms=terms,
        low=np.array([c.low for c in plan.constraints]),
        high=np.array([c.high for c in plan.constraints]),
    )


@dataclass(frozen=True)
class _Repetition:
    history: np.ndarray  # (p, n)
    lags: np.ndarray  # (H, p, n, n) the month's regime
    intercept: np.ndarray  # (H, n) mu
    impacts: np.ndarray  # (H, n, n) B_t
    mean: np.ndarray  # (H, n)


def _regime_path(model: Posterior, draw: int, horizon: int, rng):
    """Draw a regime path from the transition kernel."""
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
    """The Gaussian mixture of the original constraint functionals, per group."""
    if any(item.values.shape[0] != draws for item in plan.interventions):
        raise ValueError("intervention groups must match draws")
    if not plan.constraints:
        return None
    means, covs, groups = [], [], []
    for r in range(draws):
        rows, offset = affine_constraints_under_default(
            plan.constraints, plan.interventions, model.ids, group=r
        )
        chosen = _plan(model, replace(plan, constraints=rows))
        draw = int(np.random.default_rng([seed, r, 0]).integers(model.draws))
        rng_path = np.random.default_rng([seed, r, 1])
        for _ in range(candidates):
            path = _regime_path(model, draw, chosen.horizon, rng_path)
            rep = _repetition_of(model, draw, path)
            z = _loadings(chosen, rep).reshape(len(rows), -1)
            means.append(_level(chosen, rep.mean) + offset)
            covs.append(z @ z.T)
            groups.append(r)
    m = len(plan.constraints)
    return ConstraintLaw(
        means=np.asarray(means, dtype=float).reshape(-1, m),
        covs=np.asarray(covs, dtype=float).reshape(-1, m, m),
        log_weight=np.zeros(len(groups)),
        group=np.asarray(groups, dtype=int),
    )


#: The GHK draws behind the weight of one hidden-coordinate candidate.
CANDIDATE_SAMPLES = 64
"""How many GHK draws estimate the box probability of one candidate of a hidden coordinate:
the weight only has to rank the candidates, not price them, and the price carries its own
error of its own."""


def _hidden(
    model: Posterior,
    chosen: _Plan,
    draw: int,
    horizon: int,
    driving: np.ndarray,
    rng_path,
    candidates: int,
    cache: dict,
):
    """Draw and score regime-path candidates."""
    # An exact row (low == high) holds under every candidate through the truncation, but
    # weighs by its density: a far value is likelier under some regime paths than others.
    # The interval rows weigh by their box given the exact ones.
    log_probabilities = []
    paths = []
    for _ in range(candidates):
        path = _regime_path(model, draw, horizon, rng_path)
        key = (draw, path.tobytes())
        if key not in cache:
            rep_of_path = _repetition_of(model, draw, path)
            cache[key] = (_loadings(chosen, rep_of_path), _level(chosen, rep_of_path.mean))
        z, level = cache[key]
        paths.append(path)
        if not chosen.terms:
            log_probabilities.append(0.0)
            continue
        rows = len(chosen.terms)
        z_driving = z[:, :, driving].reshape(rows, -1)
        cov = z_driving @ z_driving.T
        mean = level
        log_probabilities.append(
            log_condition_weight(
                mean, cov, chosen.low, chosen.high, rng_path, CANDIDATE_SAMPLES
            )
        )
    if np.all(np.isneginf(log_probabilities)):
        raise ContradictsModels()
    picked, ess = pick_latent(log_probabilities, rng_path)
    return _repetition_of(model, draw, paths[picked]), ess


def simulate(
    model: Posterior,
    plan: ScenarioPlan,
    *,
    draws: int,
    seed: int,
    sweeps: int = SWEEPS,
    candidates: int = LATENT_CANDIDATES,
    inner: int | None = None,
) -> MacroSimulation:
    """Return pre-intervention paths conditioned under default substitution."""
    inner = plan.inner if inner is None else inner
    n, horizon = len(model.ids), plan.horizon
    driving = np.ones(n, dtype=bool)
    if any(item.values.shape[0] != draws for item in plan.interventions):
        raise ValueError("intervention groups must match draws")
    paths = np.empty((draws * inner, horizon + 1, n))
    history = np.empty((draws * inner, model.lags_count, n))
    group = np.repeat(np.arange(draws), inner)
    ess: list[float] = []
    # The largest driving shock of each month, in standard deviations: what the conditions
    # required by the observations, month by month.
    driving_sd: list[np.ndarray] = []
    for r in range(draws):
        constraints = constraints_under_default(
            plan.constraints, plan.interventions, model.ids, group=r
        )
        chosen = _plan(model, replace(plan, constraints=constraints))
        cache: dict = {}
        draw = int(np.random.default_rng([seed, r, 0]).integers(model.draws))
        rep, size = _hidden(
            model, chosen, draw, horizon, driving,
            np.random.default_rng([seed, r, 1]), candidates, cache,
        )
        if size is not None:
            ess.append(size)
        z = _loadings(chosen, rep) if chosen.terms and driving.any() else None
        if z is not None:
            z_all = z.reshape(len(z), -1)
            z_d = z[:, :, driving].reshape(len(z), -1)
            bad = dependent_rows(z_d, z_all)
            if bad:
                raise InfeasibleScenario(bad)
        rng_driving = np.random.default_rng([seed, r, 3])
        for j in range(inner):
            eps = np.zeros((horizon, n))
            if z is not None:
                offset = _level(chosen, rep.mean) + np.einsum(
                    "csj,sj->c", z[:, :, ~driving], eps[:, ~driving]
                )
                drawn = draw_driving(
                    z_d,
                    chosen.low - offset,
                    chosen.high - offset,
                    rng_driving,
                    sweeps,
                    every=z_all,
                )
                eps[:, driving] = drawn.reshape(horizon, driving.sum())
            else:
                eps[:, driving] = rng_driving.standard_normal(
                    (horizon, int(driving.sum()))
                )
            shocks = np.einsum("tij,tj->ti", rep.impacts, eps)
            future = iterate(rep.history, rep.lags, rep.intercept, shocks)
            paths[r * inner + j] = np.vstack([rep.history[-1], future])
            history[r * inner + j] = rep.history
            if chosen.terms and driving.any():
                driving_sd.append(np.abs(eps[:, driving]).max(axis=1))
    last = model.months[-1]
    return MacroSimulation(
        ids=model.ids,
        months=[shift(last, t) for t in range(horizon + 1)],
        paths=paths,
        history=history,
        diagnostics=_ess_summary(ess, candidates) | _driving_summary(driving_sd),
        group=group,
    )


def _ess_summary(ess: list[float], candidates: int) -> dict:
    """The spread of the candidate weights, as the report reads it."""
    if not ess:
        return {}
    values = np.asarray(ess, dtype=float)
    return {
        "latent_candidates": candidates,
        "latent_ess": {
            "min": float(values.min()),
            "median": float(np.median(values)),
            "max": float(values.max()),
        },
    }


def _driving_summary(driving_sd: list[np.ndarray]) -> dict:
    """The size of the shocks the conditions need, in standard deviations.

    A cut of five points through one shock needs a large shock — the reason such a scenario
    moves so much. The number stays visible, per month of the horizon and at its largest.
    """
    if not driving_sd:
        return {}
    sizes = np.asarray(driving_sd)
    return {
        "driving_sd": {
            "median_by_month": [float(value) for value in np.median(sizes, axis=0)],
            "max": float(sizes.max()),
            "share_above_limit": float(np.mean(sizes.max(axis=1) > DRIVING_LIMIT)),
        }
    }
