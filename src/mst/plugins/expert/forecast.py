"""The conditional forecast and the constraint law of the expert model.

Given everything the shocks make — the occurrences, the sizes, the copula levels — the path
is gaussian: ``y = b + c + u`` with ``u`` a gaussian random walk, the structure of
``msvar.markov`` with the shock inputs in the place of the regime path. When the residual
is among the causes the platform's gaussian machinery prices the scenario: every candidate
of the shocks the causes name is one row of the law, its mean carrying the candidate's
contribution and its covariance the random walk of the residual, and the imposition picks
a candidate by the probability its law gives the box, then draws the residual from the
box-truncated normal (Antolin-Diaz, Petrella and Rubio-Ramirez, 2021, Journal of Monetary
Economics 117).

When the causes name shocks without the residual, the functionals given ``R`` are a
deterministic nonlinear function of the named shocks' inputs, which no mixture of gaussian
rows can carry: the price comes from subset simulation (``subset.py``) and the imposition
from the samples it leaves inside the box.
"""

from dataclasses import dataclass

import numpy as np

from mst.model_sdk.conditional import (
    SWEEPS,
    InfeasibleScenario,
    dependent_rows,
    draw_driving,
    log_box_probabilities,
    pick_latent,
)
from mst.model_sdk.contracts import (
    ConstraintLaw,
    MacroSimulation,
    ScenarioPlan,
)
from mst.macro.panel import shift

from .estimate import HORIZON, Expert
from .model import RESIDUAL
from .stream import contributions, draw

#: The GHK draws behind the weight of one candidate, as many as ``msvar`` uses.
CANDIDATE_SAMPLES = 64


@dataclass(frozen=True)
class _Plan:
    """The plan as the expert model reads it: terms, bounds and the named causes."""

    horizon: int
    terms: list[list[tuple[int, int, float]]]
    low: np.ndarray
    high: np.ndarray
    driving: tuple[int, ...]
    """The indices of the shocks in ``D``; the residual joins them through ``residual``."""
    residual: bool
    inner: int


def _split(model: Expert, plan: ScenarioPlan) -> _Plan:
    """``D`` from the causes: the shocks they name, and whether the residual joins them.

    With no causes at all, ``D`` holds every shock and the residual, as the README says. A
    cause the model does not know is refused with the names it knows.
    """
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
    shocks = model.shocks[:-1]
    unknown = [cause for cause in plan.causes if cause not in shocks and cause != RESIDUAL]
    problems += [
        f"unknown cause {cause}: the model knows {', '.join(model.shocks)}" for cause in unknown
    ]
    if problems:
        raise ValueError("; ".join(problems))
    if plan.causes:
        driving = tuple(shocks.index(cause) for cause in plan.causes if cause != RESIDUAL)
        residual = RESIDUAL in plan.causes
    else:
        driving = tuple(range(len(shocks)))
        residual = True
    return _Plan(
        horizon=plan.horizon,
        terms=terms,
        low=np.array([c.low for c in plan.constraints]),
        high=np.array([c.high for c in plan.constraints]),
        driving=driving,
        residual=residual,
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
    """The law of the constraint functionals given ``R``, when the residual is among the
    causes.

    One group of rows per draw of ``R``: the inputs of the shocks outside ``D`` are drawn
    once and held, and every candidate of the shocks in ``D`` is one row whose mean carries
    its contribution and whose covariance is the random walk of the residual. ``None`` when
    there is nothing to price or the causes name shocks without the residual, whose price
    subset simulation carries (``log_box_probability``).
    """
    chosen = _split(model, plan)
    if not chosen.terms or not chosen.residual:
        return None
    weights = _weights(chosen, len(model.ids))
    base = model.base_path[: chosen.horizon + 1]
    z_flat, covariance = _residual_loadings(chosen, model.chol_u)
    outside = tuple(j for j in range(len(model.tables)) if j not in chosen.driving)
    means = []
    groups = []
    for r in range(draws):
        free = _contributions_of(
            model, chosen, np.random.default_rng([seed, r, 0]), 1, outside
        )[0]
        candidates = _contributions_of(
            model, chosen, np.random.default_rng([seed, r, 1]), model.candidates,
            chosen.driving,
        )
        rows = _functional(weights, base[None] + free[None] + candidates)
        means.append(rows)
        groups.append(np.full(len(rows), r))
    m = len(chosen.terms)
    means = np.concatenate(means)
    covs = np.broadcast_to(covariance, (len(means), m, m)).copy()
    return ConstraintLaw(
        means=means,
        covs=covs,
        log_weight=np.zeros(len(means)),
        group=np.concatenate(groups),
    )


def simulate(
    model: Expert,
    plan: ScenarioPlan,
    *,
    draws: int,
    seed: int,
    sweeps: int = SWEEPS,
) -> MacroSimulation:
    """``draws`` paths of the macro variables under the scenario.

    With the residual among the causes every group draws ``candidates`` of the shocks the
    causes name, picks one by the probability its gaussian law gives the box and draws the
    residual from the box-truncated normal; with ``inner`` above one the same group serves
    that many repetitions, which the conditions on means reweight. Otherwise the samples
    subset simulation leaves inside the box are the imposition.
    """
    chosen = _split(model, plan)
    if not chosen.residual:
        raise NotImplementedError(
            "the causes name shocks without the residual: the imposition arrives with "
            "subset simulation"
        )
    horizon, n = chosen.horizon, len(model.ids)
    weights = _weights(chosen, n)
    base = model.base_path[: horizon + 1]
    paths = np.empty((draws * chosen.inner, horizon + 1, n))
    group = np.repeat(np.arange(draws), chosen.inner)
    ess: list[float] = []
    z_flat = covariance = None
    if chosen.terms:
        z_flat, covariance = _residual_loadings(chosen, model.chol_u)
        bad = dependent_rows(z_flat, z_flat)
        if bad:
            raise InfeasibleScenario(bad)
    outside = tuple(j for j in range(len(model.tables)) if j not in chosen.driving)
    for r in range(draws):
        free = _contributions_of(
            model, chosen, np.random.default_rng([seed, r, 0]), 1, outside
        )[0]
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
