"""Joint Student steps, full conditional interventions and Gaussian box conditioning."""

from dataclasses import dataclass

import numpy as np

from mst.macro.panel import shift
from mst.model_sdk import (
    DRIVING_LIMIT,
    LATENT_CANDIDATES,
    RANK_TOLERANCE,
    SWEEPS,
    ConstraintLaw,
    InfeasibleScenario,
    MacroSimulation,
    ScenarioPlan,
    draw_driving,
)
from mst.model_sdk.conditional import log_condition_weight, pick_latent

CANDIDATE_SAMPLES = 64


def transfer(scale: np.ndarray, interventions: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Gain from set changes and residual loadings, with all set coordinates first.

    Columns retain their registry positions. A set column has no free innovation, and
    no global/domestic restriction is imposed on the conditional projection.
    """
    n = len(scale)
    fixed = list(interventions)
    free = [i for i in range(n) if i not in fixed]
    gain = np.zeros((n, len(fixed)))
    factor = np.zeros_like(scale)
    if fixed:
        gain = np.linalg.solve(scale[np.ix_(fixed, fixed)], scale[fixed]).T
        gain[fixed] = np.eye(len(fixed))
        residual = scale[np.ix_(free, free)] - gain[free] @ scale[np.ix_(fixed, free)]
    else:
        residual = scale
    if free:
        factor[np.ix_(free, free)] = np.linalg.cholesky((residual + residual.T) / 2)
    return gain, factor


def scale_candidates(
    nu: float, horizon: int, count: int, rng, phi: float = 0.0, sigma: float = 0.0
) -> np.ndarray:
    """Common monthly scales: iid ``nu / chi_square(nu)`` or a log AR(1) path.

    With ``sigma = 0`` this is the Student scale of the law. With a positive volatility the
    log scale follows ``s_t = phi s_{t-1} + sigma e_t`` from its stationary distribution, and
    the level is set so that ``E[lambda] = nu / (nu - 2)``: the persistence reshapes the path,
    it does not change the unconditional monthly covariance.
    """
    if sigma <= 0:
        return nu / rng.chisquare(nu, size=(count, horizon))
    if not 0 <= phi < 1:
        raise ValueError("scale persistence must lie in [0, 1)")
    stationary = sigma**2 / (1 - phi**2)
    level = np.log(nu / (nu - 2)) - stationary / 2
    state = np.empty((count, horizon + 1))
    state[:, 0] = rng.normal(0, np.sqrt(stationary), count)
    noise = rng.normal(0, sigma, (count, horizon))
    for t in range(1, horizon + 1):
        state[:, t] = phi * state[:, t - 1] + noise[:, t - 1]
    return np.exp(level + state[:, 1:])


def _validate(model, plan: ScenarioPlan, draws: int) -> None:
    if type(draws) is not int or draws < 1:
        raise ValueError("draws must be a positive integer")
    ids = set(model.ids)
    for item in plan.interventions:
        if item.values.shape[0] != draws:
            raise ValueError("intervention groups must match draws")
        if item.target not in ids:
            raise ValueError(f"unknown intervention variable {item.target}")
    _validate_observations(plan, ids)


def _validate_observations(plan: ScenarioPlan, ids: set[str]) -> None:
    for row in plan.constraints:
        if row.low > row.high or np.isnan([row.low, row.high]).any():
            raise ValueError("observation bounds must be ordered and not NaN")
        for term in row.terms:
            if term.variable not in ids:
                raise ValueError(f"unknown observation variable {term.variable}")
            if not 0 <= term.month <= plan.horizon:
                raise ValueError(f"observation month {term.month} exceeds horizon")
            if not np.isfinite(term.weight):
                raise ValueError("observation weights must be finite")


def _cells(model, plan: ScenarioPlan, group: int) -> dict:
    index = {id_: i for i, id_ in enumerate(model.ids)}
    result: dict[int, list] = {}
    for item in plan.interventions:
        for j, month in enumerate(item.months):
            if item.active is None or item.active[group, j]:
                result.setdefault(month, []).append(
                    (index[item.target], item.kind, float(item.values[group, j]))
                )
    return result


@dataclass(frozen=True)
class _Affine:
    mean: np.ndarray  # (H+1,n)
    paths: np.ndarray  # (H+1,n,(H+1)*n), unscaled innovations
    centre: np.ndarray  # (m,)
    rows: np.ndarray  # (m,(H+1)*n)


def _step(model, cells) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The transition, the intercept and the innovation factor of one month.

    The law is ``y_t = A y_{t-1} + c + sqrt(lambda_t) B z_t`` with a diagonal ``A``. A cell
    set to a value fixes that month's realised coordinate, so the free innovations are
    conditioned on the set change: the free mean gains the projection of the set innovation,
    and the transition loses the same projection of the set row.
    """
    n = len(model.ids)
    persistence = getattr(model, "persistence", None)
    intercept = getattr(model, "intercept", None)
    if persistence is None or intercept is None:
        persistence = np.ones(n)
        intercept = model.drift
    persistence = np.asarray(persistence, dtype=float)
    sets = sorted((i, value) for i, kind, value in cells if kind == "set")
    fixed = tuple(i for i, _ in sets)
    gain, factor = transfer(model.scale, fixed)
    # Keep the arithmetic explicit: the free rows lose the projection of the set rows.
    transition = np.diag(persistence).copy()
    intercept = np.asarray(intercept, dtype=float).copy()
    if fixed:
        transition -= gain @ np.diag(persistence)[list(fixed), :]
        intercept += gain @ (np.array([value for _, value in sets]) - intercept[list(fixed)])
        # Exact assignments avoid round-off in cells that must be deterministic.
        transition[list(fixed)] = 0.0
        intercept[list(fixed)] = [value for _, value in sets]
    for i, kind, value in cells:
        if kind == "shift":
            intercept[i] += value
    return transition, intercept, factor


def _affine(model, plan: ScenarioPlan, group: int) -> _Affine:
    """The entire substituted recursion, including uncertain month-zero coordinates."""
    n, horizon = len(model.ids), plan.horizon
    mean = np.zeros((horizon + 1, n))
    paths = np.zeros((horizon + 1, n, (horizon + 1) * n))
    mean[0] = model.start_mean
    paths[0, :, :n] = np.diag(model.start_sd)
    cells = _cells(model, plan, group)
    for t in range(1, horizon + 1):
        transition, intercept, factor = _step(model, cells.get(t, ()))
        mean[t] = transition @ mean[t - 1] + intercept
        paths[t] = transition @ paths[t - 1]
        paths[t, :, t * n : (t + 1) * n] = factor
    centre = np.zeros(len(plan.constraints))
    rows = np.zeros((len(plan.constraints), paths.shape[-1]))
    index = {id_: i for i, id_ in enumerate(model.ids)}
    for c, row in enumerate(plan.constraints):
        for term in row.terms:
            i, t, w = index[term.variable], term.month, term.weight
            centre[c] += w * mean[t, i]
            rows[c] += w * paths[t, i]
    return _Affine(mean, paths, centre, rows)


def _multipliers(scales: np.ndarray, n: int) -> np.ndarray:
    return np.repeat(np.concatenate(([1.0], np.sqrt(scales))), n)


def _independent(affine: _Affine, low, high) -> np.ndarray:
    """Drop true deterministic rows and reject other dependencies by original row number."""
    kept = []
    for c, row in enumerate(affine.rows):
        norm = np.linalg.norm(row)
        if norm == 0:
            if not low[c] <= affine.centre[c] <= high[c]:
                raise InfeasibleScenario([c])
            continue
        residual = row
        if kept:
            basis = affine.rows[kept].T
            residual = row - basis @ np.linalg.lstsq(basis, row, rcond=None)[0]
        if np.linalg.norm(residual) <= RANK_TOLERANCE * norm:
            raise InfeasibleScenario([c])
        kept.append(c)
    return np.array(kept, dtype=int)


def _scale_law(model) -> tuple[float, float]:
    """The persistence and volatility of the common scale; zero for the base law."""
    return (
        float(getattr(model, "scale_persistence", 0.0)),
        float(getattr(model, "scale_volatility", 0.0)),
    )


def constraint_law(
    model, plan: ScenarioPlan, *, draws: int, seed: int, candidates: int = LATENT_CANDIDATES
) -> ConstraintLaw | None:
    """Unweighted Gaussian mixture of the original observation rows after interventions."""
    _validate(model, plan, draws)
    if not plan.constraints:
        return None
    means, covs = [], []
    n = len(model.ids)
    for group in range(draws):
        affine = _affine(model, plan, group)
        scales = scale_candidates(
            model.nu,
            plan.horizon,
            candidates,
            np.random.default_rng([seed, group, 1]),
            *_scale_law(model),
        )
        for scale in scales:
            rows = affine.rows * _multipliers(scale, n)
            means.append(affine.centre)
            covs.append(rows @ rows.T)
    return ConstraintLaw(
        np.array(means),
        np.array(covs),
        np.zeros(draws * candidates),
        np.repeat(np.arange(draws), candidates),
    )


def _candidate_rows(model, plan, affine, kept, low, high, seed, group, candidates):
    rng = np.random.default_rng([seed, group, 1])
    scales = scale_candidates(
        model.nu, plan.horizon, candidates, rng, *_scale_law(model)
    )
    rows, weights = [], []
    for scale in scales:
        z = affine.rows[kept] * _multipliers(scale, len(model.ids))
        rows.append(z)
        weights.append(
            log_condition_weight(
                affine.centre[kept], z @ z.T, low[kept], high[kept], rng, CANDIDATE_SAMPLES
            )
        )
    return scales, rows, weights


def _group_paths(model, plan, affine, *, group, seed, candidates, sweeps):
    n = len(model.ids)
    low = np.array([row.low for row in plan.constraints])
    high = np.array([row.high for row in plan.constraints])
    kept = _independent(affine, low, high)
    rng = np.random.default_rng([seed, group, 3])
    picked_rng = np.random.default_rng([seed, group, 2])
    if len(kept):
        scales, rows, weights = _candidate_rows(
            model, plan, affine, kept, low, high, seed, group, candidates
        )
    else:
        scales = scale_candidates(
            model.nu,
            plan.horizon,
            plan.inner,
            np.random.default_rng([seed, group, 1]),
            *_scale_law(model),
        )
    result, sizes, chosen, ess = [], [], [], []
    for j in range(plan.inner):
        if len(kept):
            pick, effective = pick_latent(weights, picked_rng)
            eps = draw_driving(
                rows[pick],
                low[kept] - affine.centre[kept],
                high[kept] - affine.centre[kept],
                rng,
                sweeps,
            )
            ess.append(effective)
        else:
            pick = j
            eps = rng.standard_normal(affine.paths.shape[-1])
        scale = scales[pick]
        path = affine.mean + np.einsum("tij,j->ti", affine.paths, eps * _multipliers(scale, n))
        result.append(path)
        sizes.append(np.abs(eps.reshape(-1, n)[1:]).max(axis=1))
        chosen.append(scale)
    return np.array(result), sizes, chosen, ess


def _diagnostics(scales, sizes, ess, candidates):
    result = {"scale_path_mean": np.mean(scales, axis=0).tolist()}
    if ess:
        result["latent_candidates"] = candidates
        result["latent_ess"] = {
            "min": float(np.min(ess)),
            "median": float(np.median(ess)),
            "max": float(np.max(ess)),
        }
        result["driving_sd"] = {
            "median_by_month": np.median(sizes, axis=0).tolist(),
            "max": float(np.max(sizes)),
            "share_above_limit": float(np.mean(np.max(sizes, axis=1) > DRIVING_LIMIT)),
        }
    return result


def simulate(
    model,
    plan: ScenarioPlan,
    *,
    draws: int,
    seed: int,
    candidates: int = LATENT_CANDIDATES,
    sweeps: int = SWEEPS,
) -> MacroSimulation:
    """Draw shadow paths, preserving only parameters and interventions inside each group."""
    _validate(model, plan, draws)
    paths, scales, sizes, ess = [], [], [], []
    for group in range(draws):
        affine = _affine(model, plan, group)
        output, size, scale, effective = _group_paths(
            model, plan, affine, group=group, seed=seed, candidates=candidates, sweeps=sweeps
        )
        paths.extend(output)
        scales.extend(scale)
        sizes.extend(size)
        ess.extend(effective)
    paths = np.array(paths)
    return MacroSimulation(
        ids=model.ids,
        months=[shift(model.months[-1], t) for t in range(plan.horizon + 1)],
        paths=paths,
        history=paths[:, :1].copy(),
        group=np.repeat(np.arange(draws), plan.inner),
        diagnostics=_diagnostics(scales, sizes, ess, candidates),
    )
