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

from dataclasses import dataclass, replace

import numpy as np

from mst.macro.panel import shift
from mst.model_sdk.conditional import (
    SWEEPS,
    InfeasibleScenario,
    box_gibbs,
    dependent_rows,
    draw_driving,
    log_condition_weight,
    pick_latent,
    truncated_normal,
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
from .var import iterate, responses

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
    if problems:
        raise ValueError("; ".join(problems))
    return _Plan(
        horizon=scenario.horizon,
        terms=terms,
        low=np.array([c.low for c in scenario.constraints]),
        high=np.array([c.high for c in scenario.constraints]),
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
    jumps = model.jump_prob[draw] > 0
    if jumps.any():
        # A jumping variable has no Student tails; its month jumps with probability pi.
        hit = rng.random((horizon, int(jumps.sum()))) < model.jump_prob[draw, jumps]
        omega[:, jumps] = np.where(hit, model.jump_scale[draw, jumps], 1.0)
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


def _overrides(
    scenario: ScenarioPlan, ids: list[str], group: int
) -> dict[int, tuple[tuple[int, str, float], ...]]:
    """The realized intervention cells of one group, by month: ``(column, kind, value)``."""
    index = {id_: i for i, id_ in enumerate(ids)}
    found: dict[int, list[tuple[int, str, float]]] = {}
    for item in scenario.interventions:
        column = index[item.target]
        for position, month in enumerate(item.months):
            if item.active is not None and not item.active[group, position]:
                continue
            found.setdefault(month, []).append(
                (column, item.kind, float(item.values[group, position]))
            )
    return {month: tuple(cells) for month, cells in found.items()}


def _substituted_impacts(
    impacts: np.ndarray, overrides: dict[int, tuple[tuple[int, str, float], ...]]
) -> np.ndarray:
    """``B_t`` with the row of every set quantity zeroed: the equation is replaced."""
    result = impacts.copy()
    for month, cells in overrides.items():
        for column, kind, _ in cells:
            if kind == "set":
                result[month - 1, column, :] = 0.0
    return result


def _recursive_loadings(
    plan: _Plan,
    rep: _Repetition,
    overrides: dict[int, tuple[tuple[int, str, float], ...]],
) -> np.ndarray:
    """``(m, H, n)``: the response of each functional to each shock under the substitution.

    The adjoint of the VAR recursion, zeroed at the set coordinates of their months: a set
    quantity's path responds to no shock, and the later months read the imposed value as a
    constant. The default ``_loadings`` is the same adjoint without the zeroing.
    """
    n = rep.mean.shape[1]
    z = np.zeros((len(plan.terms), plan.horizon, n))
    set_at: dict[int, set[int]] = {}
    for month, cells in overrides.items():
        for column, kind, _ in cells:
            if kind == "set":
                set_at.setdefault(month, set()).add(column)
    lags = rep.lags
    p = len(lags)
    for c, row in enumerate(plan.terms):
        for variable, month, weight in row:
            adjoint = np.zeros((month + 1, n))
            adjoint[month, variable] = (
                0.0 if variable in set_at.get(month, ()) else weight
            )
            for t in range(month - 1, 0, -1):
                value = np.zeros(n)
                for lag in range(1, min(p, month - t) + 1):
                    value += lags[lag - 1].T @ adjoint[t + lag]
                for column in set_at.get(t, ()):
                    value[column] = 0.0
                adjoint[t] = value
            for s in range(1, month + 1):
                z[c, s - 1] += adjoint[s] @ rep.impacts[s - 1]
    return z


def _recursive_path(
    history: np.ndarray,
    lags: np.ndarray,
    intercept: np.ndarray,
    shocks: np.ndarray,
    overrides: dict[int, tuple[tuple[int, str, float], ...]],
) -> np.ndarray:
    """The VAR recursion with the interventions substituted into their month.

    A set replaces its equation's value and a shift adds to it; the other equations of the
    month use their own reduced-form errors, and later months read the value through the
    lags. Month zero stays the last history month.
    """
    p = len(lags)
    path = np.vstack([history, np.zeros_like(shocks)])
    for t in range(len(shocks)):
        value = intercept[t] + shocks[t]
        for lag in range(1, p + 1):
            value = value + lags[lag - 1] @ path[p + t - lag]
        for column, kind, amount in overrides.get(t + 1, ()):
            if kind == "set":
                value[column] = amount
            else:
                value[column] += amount
        path[p + t] = value
    return path[p:]


def constraint_law(
    model: Posterior,
    scenario: ScenarioPlan,
    *,
    draws: int,
    seed: int,
    candidates: int = LATENT_CANDIDATES,
) -> ConstraintLaw | None:
    """The law of the original observation rows given parameters and interventions.

    Each group retains its parameter draw and intervention values. Volatility candidates
    remain hidden, and every shock contributes to each candidate's Gaussian covariance.
    Parameter and volatility streams retain their original ``[seed, r, 0/1]`` seeds.
    """
    _check_groups(scenario, draws)
    if getattr(model, "structural", None) is not None:
        return _recursive_law(
            model, scenario, draws=draws, seed=seed, candidates=candidates
        )
    original = _plan(model, scenario)
    if not original.terms:
        return None
    means, covs, log_weights, groups = [], [], [], []
    for r in range(draws):
        rows, offsets = affine_constraints_under_default(
            scenario.constraints, scenario.interventions, model.ids, group=r
        )
        plan = _plan(model, replace(scenario, constraints=rows))
        draw = int(np.random.default_rng([seed, r, 0]).integers(model.draws))
        rng_volatility = np.random.default_rng([seed, r, 1])
        for _ in range(candidates):
            rep = _repetition(model, plan.horizon, rng_volatility, draw)
            mean, cov = _law_row(plan, rep)
            means.append(mean + offsets if scenario.interventions else mean)
            covs.append(cov)
            log_weights.append(0.0)
            groups.append(r)
    m = len(original.terms)
    return ConstraintLaw(
        means=np.asarray(means, dtype=float).reshape(len(means), m) if means
        else np.zeros((0, m)),
        covs=np.asarray(covs, dtype=float).reshape(len(covs), m, m) if covs
        else np.zeros((0, m, m)),
        log_weight=np.asarray(log_weights, dtype=float),
        group=np.asarray(groups, dtype=int),
    )


def _check_groups(scenario: ScenarioPlan, draws: int) -> None:
    if any(item.values.shape[0] != draws for item in scenario.interventions):
        raise ValueError("intervention groups must match draws")


def _law_row(plan: _Plan, rep: _Repetition):
    """The mean and covariance of ``W y`` under one volatility candidate."""
    m = len(plan.terms)
    if m == 0:
        return np.zeros(0), np.zeros((0, 0))
    z = _loadings(plan, rep)
    mean = _level(plan, rep.mean)
    z_flat = z.reshape(m, -1)
    return mean, z_flat @ z_flat.T


def _recursive_law(
    model: Posterior,
    scenario: ScenarioPlan,
    *,
    draws: int,
    seed: int,
    candidates: int,
):
    """The law of the original observation rows under the model's structural form.

    The mean is the deterministic recursion with the interventions substituted; the
    covariance counts every shock, with a set quantity's equation absent from its month.
    The rows and bounds stay the original ones: the caller prices the event it wrote.
    """
    plan = _plan(model, scenario)
    if not plan.terms:
        return None
    n = len(model.ids)
    means, covs, log_weights, groups = [], [], [], []
    for r in range(draws):
        overrides = _overrides(scenario, model.ids, r)
        draw = int(np.random.default_rng([seed, r, 0]).integers(model.draws))
        rng_volatility = np.random.default_rng([seed, r, 1])
        zeros = np.zeros((plan.horizon, n))
        for _ in range(candidates):
            rep = _repetition(model, plan.horizon, rng_volatility, draw)
            deterministic = _recursive_path(
                rep.history, rep.lags, rep.intercept, zeros, overrides
            )
            z = _recursive_loadings(plan, rep, overrides).reshape(len(plan.terms), -1)
            means.append(_level(plan, deterministic))
            covs.append(z @ z.T)
            log_weights.append(0.0)
            groups.append(r)
    m = len(plan.terms)
    return ConstraintLaw(
        means=np.asarray(means, dtype=float).reshape(len(means), m),
        covs=np.asarray(covs, dtype=float).reshape(len(covs), m, m),
        log_weight=np.asarray(log_weights, dtype=float),
        group=np.asarray(groups, dtype=int),
    )


#: How many GHK draws estimate the box probability of one volatility candidate.
CANDIDATE_SAMPLES = 64
"""How many GHK draws estimate the box probability of one candidate of the volatility path:
the weight only has to rank the candidates, not price them."""


def _constraints_of(
    scenario: ScenarioPlan, ids: list[str], group: int
) -> tuple:
    """The observations under the interventions of one group, in their original numbering.

    A set term is removed and the bounds are shifted by its value; a fully determined row is
    dropped when it holds and refused when it does not, with its original 1-based number.
    """
    rows, offsets = affine_constraints_under_default(
        scenario.constraints, scenario.interventions, ids, group=group
    )
    result = []
    for number, (row, offset) in enumerate(zip(rows, offsets, strict=True), start=1):
        chosen = replace(row, low=row.low - offset, high=row.high - offset)
        if chosen.terms:
            result.append(chosen)
        elif not chosen.low <= 0.0 <= chosen.high:
            raise ContradictsModels((number,), group=group)
    return tuple(result)


def _summaries(
    ess: list[float], driving_sd: list[np.ndarray], candidates: int
) -> dict:
    """The candidate spread and the size of the shocks the conditions need."""
    diagnostics = {}
    if ess:
        values = np.asarray(ess, dtype=float)
        diagnostics = {
            "latent_candidates": candidates,
            "latent_ess": {
                "min": float(values.min()),
                "median": float(np.median(values)),
                "max": float(values.max()),
            },
        }
    if driving_sd:
        sizes = np.asarray(driving_sd)
        diagnostics["driving_sd"] = {
            "median_by_month": [float(value) for value in np.median(sizes, axis=0)],
            "max": float(sizes.max()),
            "share_above_limit": float(np.mean(sizes.max(axis=1) > DRIVING_LIMIT)),
        }
    return diagnostics


def _simulate_recursive(
    model: Posterior,
    scenario: ScenarioPlan,
    *,
    draws: int,
    seed: int,
    sweeps: int,
    candidates: int,
    inner: int | None,
) -> Simulation:
    """The model's own form: the interventions are substituted into the VAR recursion.

    A set replaces its equation's value in its month and its reduced-form shock is not drawn
    (the row of ``B_t`` is zeroed); a shift adds to the equation. Later months read the value
    through the lags, so the effect persists instead of reverting to the base path. The
    observations are met by the remaining shocks. The default form's numbers are untouched.
    """
    inner = scenario.inner if inner is None else inner
    n, horizon = len(model.ids), scenario.horizon
    driving = np.ones(n, dtype=bool)
    paths = np.empty((draws * inner, horizon + 1, n))
    history = np.empty((draws * inner, model.lags_count, n))
    group = np.repeat(np.arange(draws), inner)
    ess: list[float] = []
    driving_sd: list[np.ndarray] = []
    zeros = np.zeros((horizon, n))
    for r in range(draws):
        overrides = _overrides(scenario, model.ids, r)
        plan = _plan(
            model,
            replace(scenario, constraints=_constraints_of(scenario, model.ids, r)),
        )
        draw = int(np.random.default_rng([seed, r, 0]).integers(model.draws))
        rng_volatility = np.random.default_rng([seed, r, 1])
        reps, dets, log_probabilities = [], [], []
        for _ in range(candidates):
            rep = _repetition(model, horizon, rng_volatility, draw)
            deterministic = _recursive_path(
                rep.history, rep.lags, rep.intercept, zeros, overrides
            )
            reps.append(rep)
            dets.append(deterministic)
            if not plan.terms:
                log_probabilities.append(0.0)
                continue
            z = _recursive_loadings(plan, rep, overrides)
            mean = _level(plan, deterministic)
            rows = len(plan.terms)
            z_driving = z[:, :, driving].reshape(rows, -1)
            cov = z_driving @ z_driving.T
            log_probabilities.append(
                log_condition_weight(
                    mean, cov, plan.low, plan.high, rng_volatility, CANDIDATE_SAMPLES
                )
            )
        if np.all(np.isneginf(log_probabilities)):
            raise ContradictsModels()
        picked, size = pick_latent(log_probabilities, rng_volatility)
        ess.append(size)
        rep, deterministic = reps[picked], dets[picked]
        impacts = _substituted_impacts(rep.impacts, overrides)
        z = None
        if plan.terms:
            z = _recursive_loadings(plan, rep, overrides)
            z_all = z.reshape(len(z), -1)
            z_d = z[:, :, driving].reshape(len(z), -1)
            bad = dependent_rows(z_d, z_all)
            if bad:
                raise InfeasibleScenario(bad)
        rng_driving = np.random.default_rng([seed, r, 3])
        for j in range(inner):
            eps = np.zeros((horizon, n))
            if z is not None:
                offset = _level(plan, deterministic) + np.einsum(
                    "csj,sj->c", z[:, :, ~driving], eps[:, ~driving]
                )
                drawn = draw_driving(
                    z_d,
                    plan.low - offset,
                    plan.high - offset,
                    rng_driving,
                    sweeps,
                    every=z_all,
                )
                eps[:, driving] = drawn.reshape(horizon, int(driving.sum()))
            else:
                eps[:, driving] = rng_driving.standard_normal(
                    (horizon, int(driving.sum()))
                )
            shocks = np.einsum("tij,tj->ti", impacts, eps)
            future = _recursive_path(
                rep.history, rep.lags, rep.intercept, shocks, overrides
            )
            paths[r * inner + j] = np.vstack([rep.history[-1], future])
            history[r * inner + j] = rep.history
            if plan.terms:
                driving_sd.append(np.abs(eps[:, driving]).max(axis=1))
    last = model.months[-1]
    return Simulation(
        ids=model.ids,
        months=[shift(last, t) for t in range(horizon + 1)],
        paths=paths,
        history=history,
        diagnostics=_summaries(ess, driving_sd, candidates),
        group=group,
    )


def simulate(
    model: Posterior,
    scenario: ScenarioPlan,
    *,
    draws: int,
    seed: int,
    sweeps: int = SWEEPS,
    candidates: int = LATENT_CANDIDATES,
    inner: int | None = None,
) -> Simulation:
    """``draws`` paths of the macro variables under the scenario.

    The streams are separate: ``[seed, r, 0]`` draws parameters, ``[seed, r, 1]``
    volatility candidates and ``[seed, r, 3]`` shocks. Candidate selection is unchanged
    from the unnamed-cause forecast. Every shock can meet observations; only parameters
    and intervention values are preserved per group. The core applies interventions
    after these pre-intervention paths are returned.
    """
    _check_groups(scenario, draws)
    if getattr(model, "structural", None) is not None:
        return _simulate_recursive(
            model,
            scenario,
            draws=draws,
            seed=seed,
            sweeps=sweeps,
            candidates=candidates,
            inner=inner,
        )
    inner = scenario.inner if inner is None else inner
    n, horizon = len(model.ids), scenario.horizon
    driving = np.ones(n, dtype=bool)
    paths = np.empty((draws * inner, horizon + 1, n))
    history = np.empty((draws * inner, model.lags_count, n))
    group = np.repeat(np.arange(draws), inner)
    ess: list[float] = []
    # The largest driving shock of each month, in standard deviations: what the observations
    # ask of the shocks, month by month. Every shock is driving.
    driving_sd: list[np.ndarray] = []
    for r in range(draws):
        constraints = constraints_under_default(
            scenario.constraints, scenario.interventions, model.ids, group=r
        )
        plan = _plan(model, replace(scenario, constraints=constraints))
        draw = int(np.random.default_rng([seed, r, 0]).integers(model.draws))
        rng_volatility = np.random.default_rng([seed, r, 1])
        reps, log_probabilities = [], []
        for _ in range(candidates):
            rep = _repetition(model, horizon, rng_volatility, draw)
            reps.append(rep)
            if not plan.terms:
                log_probabilities.append(0.0)
                continue
            z = _loadings(plan, rep)
            mean = _level(plan, rep.mean)
            # An exact row (low == high) holds under every candidate through the
            # truncation, but weighs by its density: a far value is likelier under a
            # volatile path. The interval rows weigh by their box given the exact ones.
            rows = len(plan.terms)
            z_driving = z[:, :, driving].reshape(rows, -1)
            cov = z_driving @ z_driving.T
            log_probabilities.append(
                log_condition_weight(
                    mean, cov, plan.low, plan.high, rng_volatility, CANDIDATE_SAMPLES
                )
            )
        if np.all(np.isneginf(log_probabilities)):
            raise ContradictsModels()
        picked, size = pick_latent(log_probabilities, rng_volatility)
        ess.append(size)
        rep = reps[picked]
        z = None
        if plan.terms:
            z = _loadings(plan, rep)
            z_all = z.reshape(len(z), -1)
            z_d = z[:, :, driving].reshape(len(z), -1)
            bad = dependent_rows(z_d, z_all)
            if bad:
                raise InfeasibleScenario(bad)
        rng_driving = np.random.default_rng([seed, r, 3])
        for j in range(inner):
            eps = np.zeros((horizon, n))
            if z is not None:
                offset = _level(plan, rep.mean) + np.einsum(
                    "csj,sj->c", z[:, :, ~driving], eps[:, ~driving]
                )
                drawn = draw_driving(
                    z_d,
                    plan.low - offset,
                    plan.high - offset,
                    rng_driving,
                    sweeps,
                    every=z_all,
                )
                eps[:, driving] = drawn.reshape(horizon, int(driving.sum()))
            else:
                # Nothing to meet: every shock is drawn freely from its own stream.
                eps[:, driving] = rng_driving.standard_normal(
                    (horizon, int(driving.sum()))
                )
            shocks = np.einsum("tij,tj->ti", rep.impacts, eps)
            future = iterate(rep.history, rep.lags, rep.intercept, shocks)
            paths[r * inner + j] = np.vstack([rep.history[-1], future])
            history[r * inner + j] = rep.history
            if plan.terms:
                driving_sd.append(np.abs(eps[:, driving]).max(axis=1))
    last = model.months[-1]
    return Simulation(
        ids=model.ids,
        months=[shift(last, t) for t in range(horizon + 1)],
        paths=paths,
        history=history,
        diagnostics=_summaries(ess, driving_sd, candidates),
        group=group,
    )
