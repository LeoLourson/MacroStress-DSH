"""Conditional forecasts and constraint laws for the expert model."""

from dataclasses import dataclass, replace

import numpy as np

from mst.macro.panel import shift
from mst.model_sdk.conditional import (
    SWEEPS,
    InfeasibleScenario,
    dependent_rows,
    draw_driving,
    log_box_probabilities,
    pick_latent,
)
from mst.model_sdk.contracts import ConstraintLaw, MacroSimulation, ScenarioPlan
from mst.model_sdk.interventions import affine_constraints_under_default, constraints_under_default

from .estimate import HORIZON, Expert
from .stream import contributions, draw

#: The GHK draws behind the weight of one candidate, as many as ``msvar`` uses.
CANDIDATE_SAMPLES = 64


@dataclass(frozen=True)
class _Plan:
    """The plan as the expert model reads it: terms and bounds."""

    horizon: int
    terms: list[list[tuple[int, int, float]]]
    low: np.ndarray
    high: np.ndarray
    driving: tuple[int, ...]
    """The indices of all modeled shocks."""
    inner: int


def _split(model: Expert, plan: ScenarioPlan) -> _Plan:
    """Translate the constraint terms into model coordinates."""
    if plan.horizon > HORIZON:
        raise ValueError(f"the expert model answers horizons up to {HORIZON} months")
    index = {id_: k for k, id_ in enumerate(model.ids)}
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
        driving=tuple(range(len(model.tables))),
        inner=plan.inner,
    )


def _weights(chosen: _Plan, count: int) -> np.ndarray:
    """``(m, horizon + 1, n)``: how the constraint functionals read the path."""
    found = np.zeros((len(chosen.terms), chosen.horizon + 1, count))
    for c, row in enumerate(chosen.terms):
        for variable, month, weight in row:
            found[c, month, variable] += weight
    return found


def _functional(weights: np.ndarray, path: np.ndarray) -> np.ndarray:
    """The functionals of ``path``, ``(..., m)`` for ``path`` ``(..., horizon + 1, n)``."""
    return np.einsum("mtk,...tk->...m", weights, path)


def _residual_loadings(chosen: _Plan, chol_u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(loadings, covariance)`` of the residual's monthly increments on the functionals.

    ``u_t = sum_{s <= t} L eps_s`` with ``L`` the Cholesky factor of the residual
    covariance, so the loading of ``eps_{s, i}`` on a functional is the sum of its weights
    on the variable from month ``s`` on, taken through ``L``. The loadings are
    ``(m, horizon * n)`` over the stacked increments; the covariance is their gram matrix.
    """
    horizon = chosen.horizon
    count = chol_u.shape[0]
    z = np.zeros((len(chosen.terms), horizon, count))
    for c, row in enumerate(chosen.terms):
        for variable, month, weight in row:
            z[c, :month, variable] += weight
    loaded = np.einsum("csi,ij->csj", z, chol_u)
    covariance = np.einsum("asj,bsj->ab", loaded, loaded)
    return loaded.reshape(len(chosen.terms), horizon * count), covariance


def _contributions_of(model: Expert, chosen: _Plan, rng, rows: int, shocks) -> np.ndarray:
    """``(rows, horizon + 1, n)``: the sum of the named shocks' contributions."""
    total = np.zeros((rows, chosen.horizon + 1, len(model.ids)))
    for j in shocks:
        inputs = draw(model.tables[j], rng, chosen.horizon, rows)
        total += contributions(model.tables[j], inputs, chosen.horizon)
    return total


def constraint_law(
    model: Expert,
    plan: ScenarioPlan,
    *,
    draws: int,
    seed: int,
) -> ConstraintLaw | None:
    """Law of the original functionals after the default intervention."""
    if any(item.values.shape[0] != draws for item in plan.interventions):
        raise ValueError("intervention groups do not match draws")
    if not plan.constraints:
        _split(model, plan)
        return None
    means, covs, groups = [], [], []
    for r in range(draws):
        rows, offsets = affine_constraints_under_default(
            plan.constraints, plan.interventions, model.ids, group=r
        )
        chosen = _split(model, replace(plan, constraints=rows))
        weights = _weights(chosen, len(model.ids))
        _, covariance = _residual_loadings(chosen, model.chol_u)
        candidates = _contributions_of(
            model, chosen, np.random.default_rng([seed, r, 1]), model.candidates,
            chosen.driving,
        )
        values = _functional(weights, model.base_path[: chosen.horizon + 1][None] + candidates)
        means.append(values + offsets)
        covs.append(np.broadcast_to(covariance, (len(values), len(rows), len(rows))))
        groups.append(np.full(len(values), r))
    means = np.concatenate(means)
    return ConstraintLaw(means=means, covs=np.concatenate(covs),
                         log_weight=np.zeros(len(means)), group=np.concatenate(groups))


def simulate(
    model: Expert,
    plan: ScenarioPlan,
    *,
    draws: int,
    seed: int,
    sweeps: int = SWEEPS,
) -> MacroSimulation:
    """Draw pre-intervention paths conditioned within each preserved group."""
    if any(item.values.shape[0] != draws for item in plan.interventions):
        raise ValueError("intervention groups do not match draws")
    chosen = _split(model, plan)
    horizon, n = chosen.horizon, len(model.ids)
    base = model.base_path[: horizon + 1]
    paths = np.empty((draws * chosen.inner, horizon + 1, n))
    group = np.repeat(np.arange(draws), chosen.inner)
    ess: list[float] = []
    for r in range(draws):
        constraints = constraints_under_default(
            plan.constraints, plan.interventions, model.ids, group=r
        )
        chosen = _split(model, replace(plan, constraints=constraints))
        weights = _weights(chosen, n)
        z_flat = covariance = None
        if chosen.terms:
            z_flat, covariance = _residual_loadings(chosen, model.chol_u)
            bad = dependent_rows(z_flat, z_flat)
            if bad:
                raise InfeasibleScenario(bad)
        free = np.zeros((horizon + 1, n))
        candidates = _contributions_of(
            model, chosen, np.random.default_rng([seed, r, 1]), model.candidates,
            chosen.driving,
        )
        rng_pick = np.random.default_rng([seed, r, 2])
        rng_drive = np.random.default_rng([seed, r, 3])
        if chosen.terms:
            rows = _functional(weights, base[None] + free[None] + candidates)
            log_probabilities = log_box_probabilities(
                rows,
                np.broadcast_to(
                    covariance, (len(rows), len(chosen.terms), len(chosen.terms))
                ).copy(),
                chosen.low,
                chosen.high,
                rng_pick,
                CANDIDATE_SAMPLES,
            )
        else:
            log_probabilities = np.zeros(len(candidates))
        for j in range(chosen.inner):
            picked, size = pick_latent(log_probabilities, rng_pick)
            ess.append(size)
            centre = base + free + candidates[picked]
            eta = np.zeros((horizon, n))
            if chosen.terms:
                offset = _functional(weights, centre[None])[0]
                drawn = draw_driving(
                    z_flat,
                    chosen.low - offset,
                    chosen.high - offset,
                    rng_drive,
                    sweeps,
                )
                eta = drawn.reshape(horizon, n)
            else:
                eta = rng_drive.standard_normal((horizon, n))
            walk = np.vstack(
                [np.zeros((1, n)), np.cumsum(eta @ model.chol_u.T, axis=0)]
            )
            paths[r * chosen.inner + j] = centre + walk
    return MacroSimulation(
        ids=model.ids,
        months=[shift(model.months[-1], t) for t in range(horizon + 1)],
        paths=paths,
        history=None,
        diagnostics=_ess_summary(ess, model.candidates),
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

