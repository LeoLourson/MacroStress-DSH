from datetime import date

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mst.macro.panel import Panel, months_between
from mst.plugins.rw.estimate import (
    _groups,
    _scale_step,
    changes,
    drift,
    ecme,
    estimate,
    housing,
    mean_reversion,
    nearest_correlation,
    sample_covariance,
    scale_path_parameters,
)
from mst.plugins.rw.spec import Break, Spec


def panel(y, ids):
    y = np.asarray(y, dtype=float)
    return Panel(
        ids,
        0,
        months_between(date(2018, 1, 1), date(2018 + (len(y) - 1) // 12, (len(y) - 1) % 12 + 1, 1)),
        y,
        np.zeros(y.shape, dtype=bool),
    )


def test_changes_respect_gaps_breaks_and_do_not_mutate_panel():
    p = panel([[0, 2], [1, 3], [20, np.nan], [22, 8]], ["x", "y"])
    spec = Spec(drifts=["x"], breaks=[Break(month=p.months[2], variables=["x"])])
    before = p.y.copy()
    d = changes(p, spec)
    assert_allclose(d, [[1, 1], [np.nan, np.nan], [2, np.nan]], equal_nan=True)
    assert_allclose(p.y, before, equal_nan=True)
    assert_allclose(drift(d, p.ids, spec), [1.5, 0])


def test_nearest_correlation_is_projection_not_eigenvalue_clipping():
    a = np.array([[1, 1.2, 0.9], [1.2, 1, -0.4], [0.9, -0.4, 1]])
    c = nearest_correlation(a, 1e-5)
    assert_allclose(np.diag(c), 1, atol=1e-12)
    assert_allclose(c, c.T, atol=1e-12)
    assert np.linalg.eigvalsh(c).min() >= 1e-5 - 1e-10
    # A feasible comparison from a common shrinkage correction bounds the optimum.
    w, v = np.linalg.eigh(a)
    clipped = (v * np.maximum(w, 1e-5)) @ v.T
    clipped /= np.sqrt(np.outer(np.diag(clipped), np.diag(clipped)))
    clipped = (1 - 1e-5) * clipped + 1e-5 * np.eye(3)
    assert np.linalg.norm(c - a) < np.linalg.norm(clipped - a) - 1e-4
    assert_allclose(nearest_correlation(np.eye(3), 1e-5), np.eye(3))


def student_sample(seed=401, size=5000, nu=5):
    rng = np.random.default_rng(seed)
    scale = np.array([[1.5, 0.7], [0.7, 2.0]])
    d = rng.multivariate_normal([0, 0], scale, size) / np.sqrt(
        rng.chisquare(nu, size)[:, None] / nu
    )
    d[rng.random(d.shape) < 0.2] = np.nan
    return d, scale


def test_ecme_student_recovery_missingness_and_monotonicity():
    d, true_scale = student_sample()
    spec = Spec(drifts=[], breaks=[], em_tol=1e-7)
    scale, nu, weights, trace = ecme(d, np.zeros(2), spec)
    # Sampling tolerances are wide enough for the fixed sample and reject Gaussian scale.
    assert_allclose(scale, true_scale, rtol=0.12, atol=0.08)
    assert 4 < nu < 6
    assert np.diff(trace).min() >= -1e-8
    assert abs(trace[-1] - trace[-2]) < spec.em_tol
    assert weights.shape == (len(d),)
    assert np.isnan(weights[~np.isfinite(d).any(axis=1)]).all()
    assert np.isfinite(weights[np.isfinite(d).any(axis=1)]).all()


def test_ecme_gaussian_nu_reaches_upper_bound():
    rng = np.random.default_rng(92)
    d = rng.normal(size=(4000, 2))
    _, nu, _, trace = ecme(d, np.zeros(2), Spec(drifts=[], breaks=[]))
    assert nu > 59.99
    assert np.diff(trace).min() >= -1e-8


def test_ecme_stops_on_nonconvergence_and_reports_likelihood_delta():
    d, _ = student_sample(size=100)
    with pytest.raises(ValueError, match="converge.*delta"):
        ecme(d, np.zeros(2), Spec(drifts=[], breaks=[], em_iterations=1, em_tol=1e-15))


def test_missing_covariance_uses_gaussian_schur_not_student_covariance():
    scale = np.array([[2.0, 0.5], [0.5, 3.0]])
    data = np.array([[8.0, np.nan]])
    result = _scale_step(_groups(data, np.zeros(2)), scale, 5.0)
    weight = (5 + 1) / (5 + 8**2 / 2)
    complete = np.array([8.0, 2.0])
    schur = np.array([[0.0, 0.0], [0.0, 3 - 0.5**2 / 2]])
    assert_allclose(result, weight * np.outer(complete, complete) + schur)


def test_estimate_housing_intervals_drift_start_and_json_diagnostics():
    import json

    rng = np.random.default_rng(714)
    d = rng.normal(size=(119, 2))
    y = np.vstack([np.zeros(2), np.cumsum(d, axis=0)])
    h = np.full(120, np.nan)
    observed = [0, 3, 9, 18, 24, 36, 48, 60, 72, 84, 96, 108, 117]
    h[observed] = np.cumsum(np.arange(len(observed)) * 0.7 + 0.1)
    p = panel(np.column_stack([y, h]), ["x", "y", "ru.housing_price"])
    spec = Spec(drifts=["x", "ru.housing_price"], breaks=[], em_tol=1e-7)
    diag, arrays, nu = estimate(p, spec)
    expected = (h[117] - h[0]) / 117
    assert_allclose(arrays["drift"], [d[:, 0].mean(), 0, expected])
    assert arrays["scale"].shape == (3, 3)
    assert np.linalg.eigvalsh(arrays["scale"]).min() > 0
    assert arrays["weights"].shape == (119,)
    assert_allclose(arrays["start_mean"][2], h[117] + 2 * expected)
    assert_allclose(arrays["start_sd"][2] ** 2, 2 * nu / (nu - 2) * arrays["scale"][2, 2])
    assert diag["counts"]["ru.housing_price"] == len(observed) - 1
    json.dumps(diag, allow_nan=False)


def test_housing_always_uses_intervals_even_with_monthly_observations():
    rng = np.random.default_rng(51)
    d = rng.normal(size=(80, 2))
    p = panel(np.vstack([np.zeros(2), np.cumsum(d, axis=0)]), ["x", "ru.housing_price"])
    diag, arrays, nu = estimate(p, Spec(drifts=[], breaks=[]))
    assert_allclose(arrays["scale"][1, 1], (nu - 2) / nu * np.mean(d[:, 1] ** 2))
    assert diag["counts"]["ru.housing_price"] == 80
    base_scale = arrays["scale"].copy()
    base_scale[1, :] = 0
    base_scale[:, 1] = 0
    assert_allclose(
        housing(p, arrays["drift"], base_scale, nu, Spec(drifts=[], breaks=[])), arrays["scale"]
    )


def test_housing_break_excludes_whole_crossing_interval():
    rng = np.random.default_rng(31)
    y = np.r_[0, rng.normal(size=30).cumsum()]
    h = np.full(31, np.nan)
    h[[0, 3, 7, 12, 20, 30]] = [0, 4, 1000, 1004, 1012, 1022]
    p = panel(np.column_stack([y, h]), ["x", "ru.housing_price"])
    spec = Spec(
        drifts=["ru.housing_price"],
        breaks=[Break(month=p.months[5], variables=["ru.housing_price"])],
    )
    diag, arrays, _ = estimate(p, spec)
    assert_allclose(arrays["drift"][1], 1)
    assert diag["counts"]["ru.housing_price"] == 4


@pytest.mark.parametrize(
    "ids,y,expected",
    [
        (["x", "y"], [[1, 2], [2, np.nan], [3, np.nan], [4, 3]], "y"),
        (["x", "ru.housing_price"], [[1, 2], [2, np.nan], [3, np.nan], [4, 3]], "ru.housing_price"),
    ],
)
def test_insufficient_data_names_variable(ids, y, expected):
    with pytest.raises(ValueError, match=expected):
        estimate(panel(y, ids), Spec(drifts=[], breaks=[]))


def test_mean_reversion_recovers_the_ar1_coefficient_and_the_neutral():
    rng = np.random.default_rng(5)
    count, truth, neutral = 500, 0.9, 5.0
    series = np.empty(count)
    series[0] = neutral
    for t in range(1, count):
        series[t] = neutral + truth * (series[t - 1] - neutral) + rng.normal(0, 0.5)
    p = panel(series[:, None], ["ru.key_rate"])
    spec = Spec(drifts=[], breaks=[], reversion=["ru.key_rate"], sample_scale=False)
    persistence, level, report = mean_reversion(p, spec)
    assert truth - 0.05 < persistence[0] < truth + 0.05
    assert_allclose(level[0], series.mean(), atol=0.05)
    assert report[0] == {
        "id": "ru.key_rate",
        "persistence": persistence[0],
        "neutral": level[0],
        "source": "ar1",
    }


def test_a_variable_not_named_keeps_the_random_walk():
    p = panel([[0, 0], [1, 1], [3, 2], [6, 3]], ["x", "y"])
    persistence, _, report = mean_reversion(p, Spec(drifts=[], breaks=[]))
    assert_allclose(persistence, [1.0, 1.0])
    assert [row["source"] for row in report] == ["walk", "walk"]


def test_sample_scale_is_the_sample_covariance_in_the_student_scale():
    rng = np.random.default_rng(17)
    scale = np.array([[1.0, 0.4], [0.4, 2.0]])
    d = rng.multivariate_normal([0, 0], scale, 800)
    p = panel(np.vstack([np.zeros(2), np.cumsum(d, axis=0)]), ["x", "y"])
    spec = Spec(drifts=[], breaks=[], sample_scale=True)
    _, arrays, nu = estimate(p, spec)
    expected = (nu - 2) / nu * np.cov(d.T, ddof=1)
    assert_allclose(arrays["scale"], expected, rtol=1e-6, atol=1e-10)


def test_sample_covariance_uses_the_window():
    p = panel(np.zeros((14, 2)), ["x", "y"])
    months = p.months
    data = np.zeros((13, 2))
    data[:2] = 100.0
    data[2:, 0] = np.arange(11) + 1.0
    data[2:, 1] = 2.0 * (np.arange(11) + 1.0)
    early = sample_covariance(p, data, Spec(drifts=[], breaks=[], scale_start=months[0]))[0]
    late = sample_covariance(p, data, Spec(drifts=[], breaks=[], scale_start=months[4]))[0]
    assert late[0, 0] < early[0, 0] / 100
    assert late[1, 1] > late[0, 0]


def test_scale_path_parameters_recovers_a_log_ar1():
    rng = np.random.default_rng(31)
    truth, volatility = 0.7, 0.4
    state = np.zeros(600)
    state[0] = rng.normal(0, volatility / np.sqrt(1 - truth**2))
    for t in range(1, len(state)):
        state[t] = truth * state[t - 1] + rng.normal(0, volatility)
    weights = np.exp(-state)
    found = scale_path_parameters(weights)
    assert abs(found["persistence"] - truth) < 0.06
    assert abs(found["volatility"] - volatility) < 0.05
