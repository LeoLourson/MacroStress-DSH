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

from mst.macro.panel import shift
from mst.model_sdk.checkpoint import report
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
from mst.model_sdk.dist import ContradictsModels

from .estimate import HORIZON, Expert
from .model import RESIDUAL
from .stream import contributions, draw
from .subset import subset

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
    tolerances: tuple[float, ...]
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
        tolerances=tuple(c.tolerance for c in plan.constraints),
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
        if not chosen.terms:
            return _simulate_free(model, chosen, draws, seed)
        return _simulate_subset(model, chosen, draws, seed)
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


#: How many values of R the subset price measures: the cost of one subset run bounds the
#: count, and the standard error of the price reads the spread over these values.
SUBSET_GROUPS = 32


def box_probabilities(
    model: Expert,
    plan: ScenarioPlan,
    low: np.ndarray,
    high: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> np.ndarray | None:
    """``(groups,)`` log box probabilities given R, by subset simulation.

    The causes name shocks without the residual, so given R — the parameters, the other
    shocks' inputs and the residual's path, drawn once per group — the functionals are a
    deterministic nonlinear function of the named shocks' standard normal inputs. The
    streams match ``simulate``: the same seed reproduces the same groups, the same chains
    and the same samples inside the box.
    """
    chosen = _split(model, plan)
    if not chosen.terms or chosen.residual:
        return None
    groups = min(draws, SUBSET_GROUPS)
    logs = []
    for r in range(groups):
        centre, function = _subset_group(model, chosen, r, seed, low, high)
        result = subset(
            function,
            _dimension_of(model, chosen),
            samples=model.subset_samples,
            rng=np.random.default_rng([seed, r, 1]),
        )
        logs.append(result.log_probability)
        _progress("expert: subset price", r, groups)
    return np.asarray(logs, dtype=float)


def _progress(label: str, done: int, total: int) -> None:
    """A line on stderr at every tenth of the groups: a subset run takes a while each."""
    if (done + 1) * 10 // total > done * 10 // total:
        report(f"{label} {(done + 1) * 100 // total}%")


def _dimension_of(model: Expert, chosen: _Plan) -> int:
    """The width of the named shocks' standard normal inputs over the horizon."""
    return sum(
        2 + model.tables[j].variables + 1 for j in chosen.driving
    ) * chosen.horizon


def _subset_group(model: Expert, chosen: _Plan, r: int, seed: int, low, high):
    """``(centre, function)``: the R part of one group and its violation function.

    The R part holds the base path, the contributions of the shocks the causes do not
    name and the residual's random walk, drawn once from the group's own stream. The
    violation of the box, each constraint scaled by its width, is the performance
    function subset simulation walks on.
    """
    horizon, n = chosen.horizon, len(model.ids)
    rng = np.random.default_rng([seed, r, 0])
    outside = tuple(j for j in range(len(model.tables)) if j not in chosen.driving)
    centre = model.base_path[: horizon + 1] + _contributions_of(
        model, chosen, rng, 1, outside
    )[0]
    eta = rng.standard_normal((horizon, n))
    centre = centre + np.vstack(
        [np.zeros((1, n)), np.cumsum(eta @ model.chol_u.T, axis=0)]
    )
    weights = _weights(chosen, n)
    columns = list(_columns_of(model, chosen))
    scales = _scales(chosen, low, high)

    def function(x: np.ndarray) -> np.ndarray:
        total = np.zeros((len(x), horizon + 1, n))
        for offset, width, j in columns:
            inputs = x[
                :, offset * horizon : (offset + width) * horizon
            ].reshape(-1, horizon, width)
            total += contributions(model.tables[j], inputs, horizon)
        values = _functional(weights, centre[None] + total)
        below = np.where(np.isfinite(low), (low - values) / scales, -np.inf)
        above = np.where(np.isfinite(high), (values - high) / scales, -np.inf)
        return np.maximum(np.maximum(below, above), 0.0).max(axis=-1)

    return centre, function


def _columns_of(model: Expert, chosen: _Plan):
    """``(offset, width, shock)`` of every named shock's inputs in the flat vector."""
    offset = 0
    for j in chosen.driving:
        width = 2 + model.tables[j].variables + 1
        yield offset, width, j
        offset += width


def _scales(chosen: _Plan, low, high) -> np.ndarray:
    """The scale of every constraint's violation: its width, its tolerance, or one.

    A ray of the one-sided price has an infinite side, and a point-fixed constraint
    carries its tolerance; either gives the violation a scale the level selection of
    subset simulation can read.
    """
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    width = np.where(np.isfinite(low) & np.isfinite(high), high - low, np.inf)
    tolerance = np.asarray(chosen.tolerances or (0.0,), dtype=float)
    return np.where(width > 0, width, np.where(tolerance > 0, tolerance, 1.0))


def _simulate_free(model: Expert, chosen: _Plan, draws: int, seed: int) -> MacroSimulation:
    """The paths with nothing to impose: every coordinate drawn from its own stream.

    The one-sided price reads the medians of the functionals from this simulation, with
    the causes of the plan kept: their law stays as it was, only the box is gone.
    """
    horizon, n = chosen.horizon, len(model.ids)
    rng = np.random.default_rng([seed, 0, 3])
    inputs = rng.standard_normal((draws * chosen.inner, horizon, n))
    walk = np.concatenate(
        [
            np.zeros((draws * chosen.inner, 1, n)),
            np.cumsum(inputs @ model.chol_u.T, axis=1),
        ],
        axis=1,
    )
    every = tuple(range(len(model.tables)))
    total = np.zeros((draws * chosen.inner, horizon + 1, n))
    for j in every:
        drawn = draw(
            model.tables[j],
            np.random.default_rng([seed, 1, 3, j]),
            horizon,
            draws * chosen.inner,
        )
        total += contributions(model.tables[j], drawn, horizon)
    paths = model.base_path[: horizon + 1][None] + total + walk
    return MacroSimulation(
        ids=model.ids,
        months=[shift(model.months[-1], t) for t in range(horizon + 1)],
        paths=paths,
        history=None,
        diagnostics={},
        group=np.repeat(np.arange(draws), chosen.inner),
    )


def _simulate_subset(model: Expert, chosen: _Plan, draws: int, seed: int) -> MacroSimulation:
    """The imposition when the causes name shocks without the residual.

    Every group runs subset simulation on its own value of R and resamples the samples
    inside the box to the plan's repetitions; a group whose box no sample reaches
    contradicts the models.
    """
    horizon, n = chosen.horizon, len(model.ids)
    paths = np.empty((draws * chosen.inner, horizon + 1, n))
    group = np.repeat(np.arange(draws), chosen.inner)
    dimension = _dimension_of(model, chosen)
    for r in range(draws):
        centre, function = _subset_group(
            model, chosen, r, seed, chosen.low, chosen.high
        )
        result = subset(
            function,
            dimension,
            samples=model.subset_samples,
            rng=np.random.default_rng([seed, r, 1]),
        )
        if not len(result.inside):
            raise ContradictsModels()
        rng_pick = np.random.default_rng([seed, r, 2])
        for j in range(chosen.inner):
            sample = result.inside[int(rng_pick.integers(len(result.inside)))]
            total = np.zeros((horizon + 1, n))
            flat = 0
            for shock in chosen.driving:
                columns = 2 + model.tables[shock].variables + 1
                inputs = sample[flat : flat + columns * horizon].reshape(
                    1, horizon, columns
                )
                flat += columns * horizon
                total += contributions(model.tables[shock], inputs, horizon)[0]
            paths[r * chosen.inner + j] = centre + total
        _progress("expert: subset imposition", r, draws)
    return MacroSimulation(
        ids=model.ids,
        months=[shift(model.months[-1], t) for t in range(horizon + 1)],
        paths=paths,
        history=None,
        diagnostics={"subset_samples": model.subset_samples},
        group=group,
    )
