"""Missing-data Student ECME and the housing interval approximation."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

from mst.macro.panel import Panel

from .spec import Spec

HOUSING = "ru.housing_price"


def changes(panel: Panel, spec: Spec) -> np.ndarray:
    result = np.diff(panel.y, axis=0)
    for item in spec.breaks:
        if item.month in panel.months[1:]:
            row = panel.months.index(item.month) - 1
            for variable in item.variables:
                if variable in panel.ids:
                    result[row, panel.ids.index(variable)] = np.nan
    return result


def drift(data: np.ndarray, ids: list[str], spec: Spec) -> np.ndarray:
    result = np.zeros(len(ids))
    for i, variable in enumerate(ids):
        if variable in spec.drifts and variable != HOUSING:
            known = np.isfinite(data[:, i])
            if known.any():
                result[i] = data[known, i].mean()
    return result


def nearest_correlation(matrix: np.ndarray, floor: float) -> np.ndarray:
    """Frobenius projection onto unit diagonal and eigenvalues >= floor.

    Dykstra's correction makes alternating projections solve the nearest-matrix
    problem, rather than merely find a feasible correlation matrix.
    """
    current = (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T) / 2
    correction = np.zeros_like(current)
    for _ in range(10000):
        previous = current.copy()
        residual = current - correction
        values, vectors = np.linalg.eigh(residual)
        projected = (vectors * np.maximum(values, floor)) @ vectors.T
        correction = projected - residual
        current = projected.copy()
        np.fill_diagonal(current, 1)
        if (
            np.linalg.norm(current - previous) < 1e-12
            and np.linalg.eigvalsh(current)[0] >= floor - 1e-12
        ):
            # Remove the final floating-point feasibility error without changing the diagonal.
            minimum = np.linalg.eigvalsh(current)[0]
            if minimum < floor:
                alpha = (floor - minimum) / (1 - minimum) + np.finfo(float).eps
                current = (1 - alpha) * current + alpha * np.eye(len(current))
            return (current + current.T) / 2
    raise ValueError("nearest correlation projection did not converge")


def _positive_scale(matrix: np.ndarray, floor: float) -> np.ndarray:
    sd = np.sqrt(np.diag(matrix))
    if not np.isfinite(sd).all() or np.any(sd <= 0):
        raise ValueError("scale requires a positive variance for every variable")
    correlation = matrix / np.outer(sd, sd)
    return nearest_correlation(correlation, floor) * np.outer(sd, sd)


def _initial_scale(data: np.ndarray, spec: Spec) -> np.ndarray:
    n = data.shape[1]
    scale = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            known = np.isfinite(data[:, i]) & np.isfinite(data[:, j])
            value = np.cov(data[known, i], data[known, j], ddof=0)[0, 1] if known.sum() > 1 else 0
            scale[i, j] = scale[j, i] = value
    return _positive_scale(scale, spec.eig_floor)


def _groups(data: np.ndarray, mu: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    masks, inverse = np.unique(np.isfinite(data), axis=0, return_inverse=True)
    groups = []
    for number, mask in enumerate(masks):
        if mask.any():
            rows = np.flatnonzero(inverse == number)
            observed = np.flatnonzero(mask)
            groups.append((rows, observed, data[np.ix_(rows, observed)] - mu[observed]))
    return groups


def _distances(
    scale: np.ndarray, observed: np.ndarray, residual: np.ndarray
) -> tuple[np.ndarray, float]:
    block = scale[np.ix_(observed, observed)]
    factor = np.linalg.cholesky(block)
    whitened = np.linalg.solve(factor, residual.T)
    return np.sum(whitened**2, axis=0), float(2 * np.log(np.diag(factor)).sum())


def _likelihood_terms(groups: list, scale: np.ndarray) -> list[tuple[int, np.ndarray, float]]:
    terms = []
    for _, observed, residual in groups:
        delta, logdet = _distances(scale, observed, residual)
        terms.append((len(observed), delta, logdet))
    return terms


def _likelihood(terms: list, nu: float) -> float:
    result = 0.0
    for p, delta, logdet in terms:
        constant = gammaln((nu + p) / 2) - gammaln(nu / 2) - p / 2 * np.log(nu * np.pi) - logdet / 2
        result += float(np.sum(constant - (nu + p) / 2 * np.log1p(delta / nu)))
    return result


def _scale_step(groups: list, scale: np.ndarray, nu: float) -> np.ndarray:
    total = np.zeros_like(scale)
    count = 0
    for rows, observed, residual in groups:
        delta, _ = _distances(scale, observed, residual)
        weights = (nu + len(observed)) / (nu + delta)
        projection = np.linalg.solve(scale[np.ix_(observed, observed)], scale[observed, :]).T
        complete = residual @ projection.T
        # E[w (d-mu)(d-mu)' | observed] has the Gaussian Schur covariance,
        # not w times the conditional Student covariance.
        schur = scale - projection @ scale[observed, :]
        schur[observed, :] = 0
        schur[:, observed] = 0
        total += (complete.T * weights) @ complete + len(rows) * schur
        count += len(rows)
    result = total / count
    return (result + result.T) / 2


def _fit_nu(terms: list, spec: Spec) -> float:
    low, high = spec.nu_bounds
    fit = minimize_scalar(
        lambda nu: -_likelihood(terms, nu),
        bounds=(low, high),
        method="bounded",
        options={"xatol": 1e-9},
    )
    if not fit.success:
        raise ValueError(f"nu optimisation failed: {fit.message}")
    return max([low, float(fit.x), high], key=lambda nu: _likelihood(terms, nu))


def _weights(groups: list, scale: np.ndarray, nu: float, count: int) -> np.ndarray:
    result = np.full(count, np.nan)
    for rows, observed, residual in groups:
        delta, _ = _distances(scale, observed, residual)
        result[rows] = (nu + len(observed)) / (nu + delta)
    return result


def ecme(
    data: np.ndarray, mu: np.ndarray, spec: Spec
) -> tuple[np.ndarray, float, np.ndarray, list[float]]:
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[1] == 0 or not np.isfinite(data).any(axis=0).all():
        raise ValueError("ECME requires observed changes for every variable")
    groups = _groups(data, mu)
    scale = _initial_scale(data, spec)
    nu = float(np.clip(10, *spec.nu_bounds))
    trace = [_likelihood(_likelihood_terms(groups, scale), nu)]
    for _ in range(spec.em_iterations):
        candidate = _scale_step(groups, scale, nu)
        terms = _likelihood_terms(groups, candidate)
        candidate_nu = _fit_nu(terms, spec)
        likelihood = _likelihood(terms, candidate_nu)
        delta = likelihood - trace[-1]
        if delta < -1e-8:
            raise ValueError(f"ECME likelihood decreased: delta={delta:.12g}")
        trace.append(likelihood)
        scale, nu = candidate, candidate_nu
        if abs(delta) < spec.em_tol:
            return scale, nu, _weights(groups, scale, nu, len(data)), trace
    raise ValueError(
        f"ECME did not converge after {spec.em_iterations} iterations: "
        f"likelihood delta={delta:.12g}"
    )


def _intervals(panel: Panel, index: int, spec: Spec) -> list[tuple[int, int, int]]:
    known = np.flatnonzero(np.isfinite(panel.y[:, index]))
    breaks = [item.month for item in spec.breaks if panel.ids[index] in item.variables]
    intervals = []
    for first, last in zip(known[:-1], known[1:], strict=True):
        if not any(panel.months[first] < month <= panel.months[last] for month in breaks):
            a, b = panel.months[first], panel.months[last]
            duration = (b.year - a.year) * 12 + b.month - a.month
            intervals.append((int(first), int(last), duration))
    return intervals


def _housing_row(
    panel: Panel, mu: np.ndarray, scale: np.ndarray, nu: float, spec: Spec
) -> np.ndarray:
    h = panel.ids.index(HOUSING)
    intervals = _intervals(panel, h, spec)
    residuals = [(a, b, k, panel.y[b, h] - panel.y[a, h] - k * mu[h]) for a, b, k in intervals]
    variance = np.mean([residual**2 / k for _, _, k, residual in residuals])
    row = np.zeros(len(panel.ids))
    row[h] = (nu - 2) / nu * variance
    for i in range(len(panel.ids)):
        if i != h:
            breaks = [item.month for item in spec.breaks if panel.ids[i] in item.variables]
            shared = [
                residual * (panel.y[b, i] - panel.y[a, i] - k * mu[i]) / k
                for a, b, k, residual in residuals
                if np.isfinite(panel.y[[a, b], i]).all()
                and not any(panel.months[a] < month <= panel.months[b] for month in breaks)
            ]
            row[i] = (nu - 2) / nu * np.mean(shared) if shared else 0
    return row


def housing(panel: Panel, mu: np.ndarray, scale: np.ndarray, nu: float, spec: Spec) -> np.ndarray:
    """Insert housing into a full-size scale matrix, preserving its diagonal."""
    result = scale.copy()
    h = panel.ids.index(HOUSING)
    row = _housing_row(panel, mu, scale, nu, spec)
    result[h, :] = row
    result[:, h] = row
    return _positive_scale(result, spec.eig_floor)


def _counts(panel: Panel, data: np.ndarray, spec: Spec) -> dict[str, int]:
    counts = {variable: int(np.isfinite(data[:, i]).sum()) for i, variable in enumerate(panel.ids)}
    if HOUSING in panel.ids:
        counts[HOUSING] = len(_intervals(panel, panel.ids.index(HOUSING), spec))
    short = [
        f"{variable} ({count})" for variable, count in counts.items() if count < spec.min_changes
    ]
    if short:
        raise ValueError(f"fewer than {spec.min_changes} changes or intervals: {', '.join(short)}")
    return counts


def _start(
    panel: Panel, mu: np.ndarray, scale: np.ndarray, nu: float
) -> tuple[np.ndarray, np.ndarray]:
    mean, sd = np.zeros(len(panel.ids)), np.zeros(len(panel.ids))
    for i in range(len(panel.ids)):
        last = np.flatnonzero(np.isfinite(panel.y[:, i]))[-1]
        duration = len(panel.y) - 1 - last
        mean[i] = panel.y[last, i] + duration * mu[i]
        sd[i] = np.sqrt(duration * nu / (nu - 2) * scale[i, i])
    return mean, sd


def estimate(panel: Panel, spec: Spec) -> tuple[dict, dict, float]:
    spec.validate_ids(panel.ids)
    data = changes(panel, spec)
    counts = _counts(panel, data, spec)
    mu = drift(data, panel.ids, spec)
    ordinary = [i for i, variable in enumerate(panel.ids) if variable != HOUSING]
    if not ordinary:
        raise ValueError("ECME requires at least one non-housing variable")
    fitted, nu, weights, trace = ecme(data[:, ordinary], mu[ordinary], spec)
    scale = np.zeros((len(panel.ids), len(panel.ids)))
    scale[np.ix_(ordinary, ordinary)] = fitted
    distance = 0.0
    if HOUSING in panel.ids:
        h = panel.ids.index(HOUSING)
        intervals = _intervals(panel, h, spec)
        if HOUSING in spec.drifts:
            mu[h] = sum(panel.y[b, h] - panel.y[a, h] for a, b, _ in intervals) / sum(
                k for _, _, k in intervals
            )
        row = _housing_row(panel, mu, scale, nu, spec)
        scale[h, :] = row
        scale[:, h] = row
        original = scale / np.sqrt(np.outer(np.diag(scale), np.diag(scale)))
        scale = _positive_scale(scale, spec.eig_floor)
        corrected = scale / np.sqrt(np.outer(np.diag(scale), np.diag(scale)))
        distance = float(np.linalg.norm(corrected - original))
    start_mean, start_sd = _start(panel, mu, scale, nu)
    chosen = np.flatnonzero(np.isfinite(weights))
    chosen = chosen[np.argsort(weights[chosen], kind="stable")[:10]]
    diagnostics = {
        "nu": nu,
        "iterations": len(trace) - 1,
        "last_likelihood_delta": trace[-1] - trace[-2],
        "trace": trace,
        "counts": counts,
        "higham_distance": distance,
        "smallest_weights": [
            {"month": panel.months[i + 1].isoformat(), "weight": float(weights[i])} for i in chosen
        ],
    }
    arrays = {
        "drift": mu,
        "scale": scale,
        "weights": weights,
        "start_mean": start_mean,
        "start_sd": start_sd,
    }
    return diagnostics, arrays, nu
