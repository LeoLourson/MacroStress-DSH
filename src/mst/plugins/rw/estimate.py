"""Missing-data Student ECME and the housing interval approximation."""

from collections.abc import Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from scipy.stats import norm, rankdata

from mst.macro.panel import Panel
from mst.model_sdk import ShortHistory

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


def shrinkage(data: np.ndarray) -> float:
    """The weight of the diagonal target in the scale: the Schäfer–Strimmer intensity on the
    normal scores of the changes.

    Over the pairs of variables, the sum of the sampling variances of their correlations over
    the sum of the squared correlations, clipped to ``[0, 1]``; every pair reads the months
    where both are known. A long history estimates its correlations well and the weight is
    small; a history with about as many months as variables gets a large one. The variance of
    a correlation is a fourth moment, which Student changes with ``nu`` below four do not
    have: on the raw changes it would grow with the crisis months instead of the history's
    shortness, so every variable enters by its normal scores. A variable without spread has no
    correlation to estimate and adds nothing.
    """
    x = np.asarray(data, dtype=float)
    known = np.isfinite(x)
    variance = 0.0
    squares = 0.0
    z = np.full(x.shape, np.nan)
    for j in range(x.shape[1]):
        rows = known[:, j]
        if rows.sum() > 1 and np.ptp(x[rows, j]) > 0:
            z[rows, j] = norm.ppf((rankdata(x[rows, j]) - 0.5) / rows.sum())
    for i in range(x.shape[1]):
        for j in range(i):
            both = known[:, i] & known[:, j]
            n = int(both.sum())
            products = z[both, i] * z[both, j]
            if n < 3 or not np.isfinite(products).all():
                continue
            correlation = n / (n - 1) * products.mean()
            variance += n / (n - 1) ** 3 * float(((products - products.mean()) ** 2).sum())
            squares += correlation**2
    return 1.0 if squares == 0 else float(min(1.0, variance / squares))


def ecme(
    data: np.ndarray, mu: np.ndarray, spec: Spec, *, shrink: float = 0.0
) -> tuple[np.ndarray, float, np.ndarray, list[float]]:
    """The Student scale and ``nu`` of the changes, with the month weights and the trace.

    ``shrink`` is the weight ``lambda`` of the diagonal target ``D`` in every scale step:
    ``Sigma = (1 - lambda) S + lambda D``, the posterior mode under an inverse-Wishart prior
    worth ``n lambda / (1 - lambda)`` months, so the ECME stays monotone in the penalised
    likelihood and the trace holds that objective. ``D`` is the scale of the ECME with a
    diagonal ``Sigma``, which every variable's own changes identify however short the panel.
    Zero is the plain ECME; one is the diagonal scale.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[1] == 0 or not np.isfinite(data).any(axis=0).all():
        raise ValueError("ECME requires observed changes for every variable")
    if not 0 <= shrink <= 1:
        raise ValueError(f"the shrinkage weight must lie in [0, 1], not {shrink}")
    groups = _groups(data, mu)
    initial = _initial_scale(data, spec)
    if shrink == 0:
        scale, nu, trace = _ecme_steps(groups, initial, spec, lambda step: step)
    else:
        target, nu, trace = _ecme_steps(
            groups, np.diag(np.diag(initial)), spec, lambda step: np.diag(np.diag(step))
        )
        scale = target
        if shrink < 1:
            months = sum(len(rows) for rows, _, _ in groups)
            weight = months * shrink / (1 - shrink)
            scale, nu, trace = _ecme_steps(
                groups,
                (1 - shrink) * initial + shrink * target,
                spec,
                lambda step: (1 - shrink) * step + shrink * target,
                lambda candidate: _wishart_penalty(candidate, target, weight),
            )
    return scale, nu, _weights(groups, scale, nu, len(data)), trace


def _wishart_penalty(scale: np.ndarray, target: np.ndarray, weight: float) -> float:
    """The log density, up to a constant, of the inverse-Wishart prior worth ``weight``
    months and centred on ``target``: ``-(weight / 2) (log det Sigma + tr(D Sigma^-1))``."""
    _, logdet = np.linalg.slogdet(scale)
    return -weight / 2 * (logdet + float(np.trace(np.linalg.solve(scale, target))))


def _ecme_steps(groups: list, scale: np.ndarray, spec: Spec, update, penalty=None):
    """ECME from ``scale``: the scale step mapped by ``update``, then ``nu`` by the observed
    likelihood. The objective is that likelihood plus ``penalty`` of the scale; a step that
    lowers it is an error, as is no convergence within ``spec.em_iterations``."""
    penalty = penalty or (lambda _: 0.0)
    nu = float(np.clip(10, *spec.nu_bounds))
    trace = [_likelihood(_likelihood_terms(groups, scale), nu) + penalty(scale)]
    for _ in range(spec.em_iterations):
        candidate = update(_scale_step(groups, scale, nu))
        terms = _likelihood_terms(groups, candidate)
        candidate_nu = _fit_nu(terms, spec)
        objective = _likelihood(terms, candidate_nu) + penalty(candidate)
        delta = objective - trace[-1]
        if delta < -1e-8:
            raise ValueError(f"ECME likelihood decreased: delta={delta:.12g}")
        trace.append(objective)
        scale, nu = candidate, candidate_nu
        if abs(delta) < spec.em_tol:
            return scale, nu, trace
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


def mean_reversion(panel: Panel, spec: Spec) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Diagonal AR(1) persistence and neutral level of the variables the spec names.

    The fit is an OLS regression of the level of the series on its previous month with a
    constant. The neutral is the panel mean of the level, not the estimated intercept over
    ``1 - a``: the intercept of a trending series implies a long-run mean far from the data,
    and a clipped coefficient above one has no intercept at all. A variable not named in the
    spec keeps ``a = 1`` and its drift. Two-step estimate: the scale and ``nu`` come from
    ECME on the changes first, this fit is independent of them.
    """
    count = len(panel.ids)
    persistence = np.ones(count)
    neutral = np.zeros(count)
    report = []
    for i, variable in enumerate(panel.ids):
        series = panel.y[:, i]
        neutral[i] = float(np.nanmean(series))
        if variable not in spec.reversion:
            report.append({"id": variable, "persistence": 1.0, "neutral": None, "source": "walk"})
            continue
        previous, current = series[:-1], series[1:]
        known = np.isfinite(previous) & np.isfinite(current)
        if known.sum() < max(spec.min_changes, 3):
            report.append(
                {"id": variable, "persistence": 1.0, "neutral": neutral[i], "source": "short"}
            )
            continue
        design = np.column_stack([np.ones(int(known.sum())), previous[known]])
        coefficient = np.linalg.lstsq(design, current[known], rcond=None)[0]
        persistence[i] = float(np.clip(coefficient[1], 0.0, 1.0))
        report.append(
            {
                "id": variable,
                "persistence": persistence[i],
                "neutral": neutral[i],
                "source": "ar1",
            }
        )
    return persistence, neutral, report


def variance_ratio(phi: float, horizon: int) -> float:
    """``Var`` of a sum of ``horizon`` stationary AR(1) changes over ``horizon`` single ones."""
    return 1.0 + 2.0 * sum((1 - k / horizon) * phi**k for k in range(1, horizon))


def _window_sums(column: np.ndarray, horizon: int) -> np.ndarray:
    """Sums of ``horizon`` consecutive changes; a window with a missing change is skipped."""
    sums = np.array([column[t : t + horizon].sum() for t in range(len(column) - horizon + 1)])
    return sums[np.isfinite(sums)]


def momentum(panel: Panel, data: np.ndarray, spec: Spec) -> tuple[np.ndarray, list[dict]]:
    """The AR(1) coefficient of the monthly changes of every variable, and how it was chosen.

    ``phi`` solves ``variance_ratio(phi, h) = Var(h-month change) / (h Var(monthly change))``
    over the change months from ``scale_start`` on, clipped to ``momentum_bound``. Matching
    the ratio rather than the first autocorrelation fits the horizon a stress test reads: the
    monthly means of a walk have a positive first autocorrelation, while the year may revert.
    A variable in ``reversion``, housing and a variable with fewer than two windows keep 0.
    """
    from scipy.optimize import brentq

    count, horizon, bound = len(panel.ids), spec.momentum_horizon, spec.momentum_bound
    window = np.array([month >= spec.scale_start for month in panel.months[1:]])
    phi, report = np.zeros(count), []
    for i, variable in enumerate(panel.ids):
        row = {"id": variable, "momentum": 0.0, "variance_ratio": None}
        column = data[window, i]
        sums = _window_sums(column, horizon)
        if not spec.momentum or variable in spec.reversion or variable == HOUSING:
            report.append(row | {"source": "off"})
            continue
        if len(sums) < 2 * horizon:
            report.append(row | {"source": "short"})
            continue
        monthly = np.nanvar(column, ddof=1)
        target = float(np.var(sums, ddof=1) / (horizon * monthly))
        if target <= variance_ratio(-bound, horizon):
            phi[i] = -bound
        elif target >= variance_ratio(bound, horizon):
            phi[i] = bound
        else:
            phi[i] = brentq(lambda x: variance_ratio(x, horizon) - target, -bound, bound)
        report.append(row | {"momentum": float(phi[i]), "variance_ratio": target, "source": "fit"})
    return phi, report


def scale_path_parameters(weights: np.ndarray) -> dict:
    """Log AR(1) of the implied common scale, from the ECME weights.

    The weight of a month is proportional to ``1 / lambda_t``: the posterior mean of the
    inverse scale under the Student likelihood. A regression of its logarithm on the previous
    month of the same series gives the persistence and the vol-of-vol of the scale path. The
    implied scale is a posterior mean, so its variance is shrunk and the vol-of-vol is
    approximate. Missing months are skipped; a pair that is not two adjacent months is not
    used, because the panel has no change for the missing month.
    """
    known = np.flatnonzero(np.isfinite(weights))
    if len(known) < 4:
        return {"persistence": 0.0, "volatility": 0.0, "months": int(len(known))}
    implied = np.log(1.0 / weights[known])
    previous, current = implied[:-1], implied[1:]
    adjacent = np.diff(known) == 1
    previous, current = previous[adjacent], current[adjacent]
    if len(previous) < 3:
        return {"persistence": 0.0, "volatility": 0.0, "months": int(len(known))}
    design = np.column_stack([np.ones(len(previous)), previous])
    coefficient = np.linalg.lstsq(design, current, rcond=None)[0]
    residual = current - design @ coefficient
    persistence = float(np.clip(coefficient[1], 0.0, 0.98))
    volatility = float(np.std(residual, ddof=2))
    return {
        "persistence": persistence,
        "volatility": volatility,
        "months": int(len(known)),
        "intercept": float(coefficient[0]),
    }


def sample_covariance(
    panel: Panel, data: np.ndarray, spec: Spec
) -> tuple[np.ndarray, list[int], float]:
    """The unweighted sample covariance of the changes, as a positive-definite matrix, with
    its correlations shrunk to zero by the weight ``shrinkage`` gives the window.

    The ECME weight of a crisis month is small, so the t-scale it estimates is below the
    sample second moment of a variable whose large changes cluster. This matrix is the
    second step of the scale estimate: the level and the correlations the innovation
    covariance is matched to, over the months from ``scale_start`` on. A window with about as
    many months as variables estimates its correlations badly, and the scenario carries them
    to every quantity it does not fix, so they are shrunk; the variances stay the sample's.
    Housing has no monthly changes and is left to the interval estimate.
    """
    ordinary = [i for i, variable in enumerate(panel.ids) if variable != HOUSING]
    months = np.array([month >= spec.scale_start for month in panel.months[1:]])
    rows = data[months][:, ordinary]
    matrix = _pairwise_covariance(rows)
    weight = shrinkage(rows)
    shrunk = (1 - weight) * matrix + weight * np.diag(np.diag(matrix))
    return _positive_scale(shrunk, spec.eig_floor), ordinary, weight


def _pairwise_covariance(data: np.ndarray) -> np.ndarray:
    n = data.shape[1]
    result = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            known = np.isfinite(data[:, i]) & np.isfinite(data[:, j])
            value = (
                float(np.cov(data[known, i], data[known, j], ddof=1)[0, 1])
                if known.sum() > 1
                else 0.0
            )
            result[i, j] = result[j, i] = value
    return result


def _counts(panel: Panel, data: np.ndarray, spec: Spec) -> dict[str, int]:
    counts = {variable: int(np.isfinite(data[:, i]).sum()) for i, variable in enumerate(panel.ids)}
    if HOUSING in panel.ids:
        counts[HOUSING] = len(_intervals(panel, panel.ids.index(HOUSING), spec))
    short = {
        variable: f"{count} changes or intervals, the model needs {spec.min_changes}"
        for variable, count in counts.items()
        if count < spec.min_changes
    }
    if short:
        raise ShortHistory(short)
    return counts


def _start(
    panel: Panel,
    scale: np.ndarray,
    nu: float,
    persistence: np.ndarray,
    intercept: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """The month-zero law from the last known value, under the AR(1) recursion.

    A known value has no spread; an unknown one accumulates ``duration`` steps of the law,
    whose mean reverts toward the neutral and whose variance grows as ``1 - a ** (2 duration)``.
    """
    mean, sd = np.zeros(len(panel.ids)), np.zeros(len(panel.ids))
    for i in range(len(panel.ids)):
        last = np.flatnonzero(np.isfinite(panel.y[:, i]))[-1]
        duration = len(panel.y) - 1 - last
        a = float(persistence[i])
        if a < 1:
            neutral = intercept[i] / (1 - a)
            mean[i] = neutral + a**duration * (panel.y[last, i] - neutral)
            factor = (1 - a ** (2 * duration)) / (1 - a**2)
        else:
            mean[i] = panel.y[last, i] + duration * intercept[i]
            factor = float(duration)
        sd[i] = np.sqrt(max(factor, 0.0) * nu / (nu - 2) * scale[i, i])
    return mean, sd


def _tail(panel: Panel) -> np.ndarray:
    """The last months of the panel, from the latest one at which every variable was known.

    The forecast draws the months between, which a satellite reads as the recent past of a
    series published with a lag (``forecast.recent``)."""
    first = len(panel.y) - 1
    while True:
        moved = first
        for i, variable in enumerate(panel.ids):
            known = np.flatnonzero(np.isfinite(panel.y[: first + 1, i]))
            if not len(known):
                raise ValueError(f"{variable} has no value by {panel.months[first]:%Y-%m}")
            first = min(first, int(known[-1]))
        if first == moved:
            return panel.y[first:].copy()


def estimate(
    panel: Panel, spec: Spec, *, left_out: Sequence[str] = ()
) -> tuple[dict, dict, float]:
    """The version of the panel; ``left_out`` are variables the spec may name but the panel
    lacks, because the platform left them out as too short."""
    spec.validate_ids([*panel.ids, *left_out])
    data = changes(panel, spec)
    counts = _counts(panel, data, spec)
    mu = drift(data, panel.ids, spec)
    ordinary = [i for i, variable in enumerate(panel.ids) if variable != HOUSING]
    if not ordinary:
        raise ValueError("ECME requires at least one non-housing variable")
    shrunk = {"ecme": shrinkage(data[:, ordinary])}
    fitted, nu, weights, trace = ecme(data[:, ordinary], mu[ordinary], spec, shrink=shrunk["ecme"])
    scale = np.zeros((len(panel.ids), len(panel.ids)))
    scale[np.ix_(ordinary, ordinary)] = fitted
    ecme_variance = {
        panel.ids[i]: float(np.diag(fitted)[k] * nu / (nu - 2))
        for k, i in enumerate(ordinary)
    }
    sample_variance: dict[str, float] = {}
    if spec.sample_scale:
        # The Student law has monthly covariance nu / (nu - 2) * Sigma, so this rescaled
        # sample covariance makes that covariance the sample covariance of the changes.
        sample, ordinary, shrunk["sample"] = sample_covariance(panel, data, spec)
        scale[np.ix_(ordinary, ordinary)] = (nu - 2) / nu * sample
        sample_variance = {
            panel.ids[i]: float(np.diag(sample)[k]) for k, i in enumerate(ordinary)
        }
    phi, persistence_of_changes = momentum(panel, data, spec)
    if phi.any():
        # A diagonal AR(1) of changes has monthly covariance Sigma_e / (1 - phi phi'),
        # elementwise; this innovation scale keeps the monthly covariance estimated above.
        block = np.ix_(ordinary, ordinary)
        scale[block] = _positive_scale(
            scale[block] * (1 - np.outer(phi[ordinary], phi[ordinary])), spec.eig_floor
        )
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
    persistence, neutral, reversion = mean_reversion(panel, spec)
    dynamics = (
        scale_path_parameters(weights)
        if spec.stochastic_scale
        else {"persistence": 0.0, "volatility": 0.0, "months": int(np.isfinite(weights).sum())}
    )
    intercept = mu.copy()
    for i, variable in enumerate(panel.ids):
        if variable in spec.reversion:
            intercept[i] = (1.0 - persistence[i]) * neutral[i]
    start_mean, start_sd = _start(panel, scale, nu, persistence, intercept)
    chosen = np.flatnonzero(np.isfinite(weights))
    chosen = chosen[np.argsort(weights[chosen], kind="stable")[:10]]
    diagnostics = {
        "nu": nu,
        "iterations": len(trace) - 1,
        "last_likelihood_delta": trace[-1] - trace[-2],
        "trace": trace,
        "counts": counts,
        "shrinkage": shrunk,
        "higham_distance": distance,
        "smallest_weights": [
            {"month": panel.months[i + 1].isoformat(), "weight": float(weights[i])} for i in chosen
        ],
        "mean_reversion": reversion,
        "momentum": persistence_of_changes,
        "scale_path": dynamics,
        "scale_correction": {
            "mode": "sample" if spec.sample_scale else "ecme",
            "ecme_sd": {
                variable: float(np.sqrt(max(value, 0.0)))
                for variable, value in ecme_variance.items()
            },
            "sample_sd": {
                variable: float(np.sqrt(max(value, 0.0)))
                for variable, value in sample_variance.items()
            },
        },
    }
    arrays = {
        "drift": mu,
        "scale": scale,
        "weights": weights,
        "start_mean": start_mean,
        "start_sd": start_sd,
        "persistence": persistence,
        "intercept": intercept,
        "momentum": phi,
        # The month before t = 0: the last observed change the momentum starts from.
        "previous": panel.y[-2] if len(panel.y) > 1 else np.full(len(panel.ids), np.nan),
        "tail": _tail(panel),
    }
    return diagnostics, arrays, nu
