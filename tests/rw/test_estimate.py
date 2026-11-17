from datetime import date

import numpy as np
import pytest
from scipy.stats import t as student
from numpy.testing import assert_allclose, assert_array_equal

from mst.macro.panel import Panel, months_between
from mst.plugins.rw.estimate import (
    _groups,
    _scale_step,
    band_nu,
    changes,
    drift,
    ecme,
    estimate,
    housing,
    mean_reversion,
    nearest_correlation,
    sample_covariance,
    scale_path_parameters,
    scale_window,
    shrinkage,
)
from mst.plugins.rw.forecast import mixing
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


def short_sample(seed: int, months: int, variables: int = 21, missing: float = 0.15):
    """Student changes of many variables over few months, with gaps: the 2014 panel's shape."""
    rng = np.random.default_rng(seed)
    loadings = rng.normal(size=(variables, variables))
    covariance = loadings @ loadings.T / variables + 0.1 * np.eye(variables)
    d = rng.multivariate_normal(np.zeros(variables), covariance, months)
    d /= np.sqrt(rng.chisquare(5, months) / 5)[:, None]
    d[rng.random(d.shape) < missing] = np.nan
    return d


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_shrunk_ecme_converges_when_the_months_barely_outnumber_the_variables(seed):
    """21 variables over 30 months with gaps: the plain ECME breaks (a singular block or a
    falling likelihood), the one shrunk to the diagonal converges and its objective rises."""
    d = short_sample(seed, 30)
    spec = Spec(drifts=[], breaks=[])
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        ecme(d, np.zeros(d.shape[1]), spec)
    weight = shrinkage(d)
    scale, nu, weights, trace = ecme(d, np.zeros(d.shape[1]), spec, shrink=weight)
    assert 0.3 < weight < 1
    assert np.diff(trace).min() >= -1e-8
    assert np.linalg.eigvalsh(scale)[0] > 0
    assert spec.nu_bounds[0] <= nu <= spec.nu_bounds[1]
    assert np.isfinite(weights[np.isfinite(d).any(axis=1)]).all()


def test_shrinkage_is_small_on_a_long_history_and_large_on_a_short_one():
    assert shrinkage(short_sample(3, 2000, missing=0.0)) < 0.05
    assert shrinkage(short_sample(3, 30, missing=0.0)) > 0.3


def test_full_shrinkage_is_the_diagonal_scale():
    d = short_sample(4, 40, variables=3, missing=0.0)
    scale, _, _, _ = ecme(d, np.zeros(3), Spec(drifts=[], breaks=[]), shrink=1.0)
    assert_allclose(scale, np.diag(np.diag(scale)))


def test_ecme_gaussian_nu_reaches_upper_bound():
    rng = np.random.default_rng(92)
    d = rng.normal(size=(4000, 2))
    _, nu, _, trace = ecme(d, np.zeros(2), Spec(drifts=[], breaks=[]))
    assert nu > 59.99
    assert np.diff(trace).min() >= -1e-8


def test_band_nu_puts_ninety_percent_of_the_changes_inside_p5_p95():
    rng = np.random.default_rng(93)
    spec = Spec(drifts=[], breaks=[])
    d = rng.standard_t(5, size=(20000, 2)) * [1.0, 30.0]
    nu = band_nu(d, spec)
    z = (d - d.mean(axis=0)) / d.std(axis=0, ddof=1)
    half = student.ppf(0.95, nu) * np.sqrt((nu - 2) / nu)
    assert nu == pytest.approx(5, rel=0.25)
    assert np.mean(np.abs(z) > half) == pytest.approx(0.10, abs=0.002)
    assert band_nu(rng.normal(size=(4000, 2)), spec) == 60.0
    rare = rng.normal(size=(600, 3))
    rare[::25] *= 10
    assert band_nu(rare, spec) == 2.5


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
        housing(p, arrays["drift"], base_scale, np.full(2, nu[0] / (nu[0] - 2)), Spec()),
        arrays["scale"],
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
    "ids,y,expected,count",
    [
        (["x", "y"], [[1, 2], [2, np.nan], [4, np.nan], [3, 3]], "y", 0),
        (
            ["x", "ru.housing_price"],
            [[1, 2], [2, np.nan], [4, np.nan], [3, 3]],
            "ru.housing_price",
            1,
        ),
    ],
)
def test_insufficient_data_names_variable(ids, y, expected, count):
    from mst.model_sdk import ShortHistory

    with pytest.raises(ShortHistory) as raised:
        estimate(panel(y, ids), Spec(drifts=[], breaks=[]))
    assert list(raised.value.variables) == [expected]
    assert raised.value.variables[expected] == f"{count} changes or intervals, the model needs 3"


def test_changes_that_never_vary_are_a_short_history():
    """A rate that has not moved since the panel began gives the scale no variance: the
    variable is short on that date, not the whole date refused."""
    from mst.model_sdk import ShortHistory

    y = [[1.0, 5.0], [2.0, 5.0], [4.0, 5.0], [3.0, 5.0], [5.0, 5.0]]
    with pytest.raises(ShortHistory) as raised:
        estimate(panel(y, ["x", "ru.key_rate"]), Spec(drifts=[], breaks=[]))
    assert raised.value.variables == {"ru.key_rate": "4 changes, all equal: no variance to scale"}


def test_a_variable_the_platform_left_out_may_stay_in_the_spec():
    y = np.cumsum(np.random.default_rng(7).standard_normal((30, 2)), axis=0)
    spec = Spec(drifts=["ru.housing_price"], breaks=[])
    with pytest.raises(ValueError, match="unknown variables: ru.housing_price"):
        estimate(panel(y, ["x", "y"]), spec)
    diagnostics, _, _ = estimate(panel(y, ["x", "y"]), spec, left_out=["ru.housing_price"])
    assert set(diagnostics["counts"]) == {"x", "y"}


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
    spec = Spec(drifts=[], breaks=[], regimes=[], sample_scale=True, momentum=False)
    meta, arrays, nu = estimate(p, spec)
    # The correlation of the sample is shrunk to zero by the weight its history earns; the
    # variances stay the sample's.
    weight = meta["shrinkage"]["sample"]
    assert weight == pytest.approx(shrinkage(d))
    sample = np.cov(d.T, ddof=1)
    expected = (nu - 2) / nu * ((1 - weight) * sample + weight * np.diag(np.diag(sample)))
    assert_allclose(arrays["scale"], expected, rtol=1e-6, atol=1e-10)


def test_a_short_sample_shrinks_its_correlations_and_keeps_its_variances():
    d = short_sample(5, 30, missing=0.0)
    p = panel(np.vstack([np.zeros(d.shape[1]), np.cumsum(d, axis=0)]), [f"v{k}" for k in range(21)])
    matrix, _, weight = sample_covariance(p, d, Spec(drifts=[], breaks=[]))
    raw = np.cov(d.T, ddof=1)
    assert weight > 0.3
    assert_allclose(np.diag(matrix), np.diag(raw), rtol=1e-6)
    off = ~np.eye(21, dtype=bool)
    assert np.abs(matrix[off]).sum() < (1 - weight + 0.01) * np.abs(raw[off]).sum()


def test_sample_covariance_takes_the_correlation_of_the_regime_and_the_variance_of_the_panel():
    p = panel(np.zeros((14, 2)), ["x", "y"])
    data = np.zeros((13, 2))
    data[:3, 0] = [30.0, -30.0, 30.0]
    data[:3, 1] = -data[:3, 0]
    data[3:, 0] = [1.0, -1.0, 2.0, -2.0, 1.0, -1.0, 2.0, -2.0, 1.0, -1.0]
    data[3:, 1] = data[3:, 0] + np.tile([0.1, -0.1], 5)

    def matrix(regimes):
        spec = Spec(drifts=[], breaks=[], regimes=regimes, regime_months=2)
        return sample_covariance(p, data, spec)[0]

    whole, regime = matrix([]), matrix([p.months[4]])
    assert whole[0, 1] <= 0 < regime[0, 1]
    assert_allclose(np.diag(regime), np.diag(np.cov(data.T, ddof=1)), rtol=1e-6)
    assert_allclose(np.diag(whole), np.diag(regime), rtol=1e-6)


def test_the_window_starts_with_the_regime_unless_the_regime_is_young():
    p = panel(np.zeros((66, 1)), ["x"])  # 2018-01 .. 2023-06
    regimes = [date(2014, 11, 1), date(2020, 3, 1), date(2022, 3, 1)]
    assert scale_window(p, Spec(regimes=regimes, regime_months=12)) == date(2022, 3, 1)
    assert scale_window(p, Spec(regimes=regimes, regime_months=24)) == date(2021, 7, 1)
    assert scale_window(p, Spec(regimes=[], regime_months=24)) == date(2018, 1, 1)
    assert scale_window(p, Spec(regimes=[date(2030, 1, 1)], regime_months=12)) == date(2018, 1, 1)
    assert scale_window(p, Spec(regimes=regimes, regime_months=120)) == date(2018, 1, 1)


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
    assert found["state"] == pytest.approx(state[-1] - state.mean(), abs=1e-9)
    weights[-2:] = np.nan
    decayed = scale_path_parameters(weights)["state"]
    assert decayed == pytest.approx(
        (state[-3] - state[:-2].mean()) * found["persistence"] ** 2, rel=0.2
    )


def ar_changes(rng, phi, count, sd=1.0):
    d = np.zeros(count)
    for t in range(1, count):
        d[t] = phi * d[t - 1] + rng.normal(0, sd)
    return d


def test_momentum_matches_the_variance_of_twelve_month_changes():
    from mst.plugins.rw.estimate import momentum, variance_ratio

    rng = np.random.default_rng(41)
    d = np.column_stack([ar_changes(rng, 0.4, 3000), ar_changes(rng, -0.3, 3000)])
    p = panel(np.vstack([np.zeros(2), np.cumsum(d, axis=0)]), ["x", "y"])
    spec = Spec(drifts=[], breaks=[], regimes=[])
    phi, report = momentum(p, changes(p, spec), spec)
    assert_allclose(phi, [0.4, -0.3], atol=0.08)
    assert_allclose(variance_ratio(phi[0], 12), report[0]["variance_ratio"], rtol=1e-6)
    assert report[1]["id"] == "y"


def test_momentum_keeps_the_sample_monthly_covariance():
    rng = np.random.default_rng(43)
    d = np.column_stack([ar_changes(rng, 0.4, 2500), ar_changes(rng, 0.2, 2500)])
    d[:, 1] += 0.5 * d[:, 0]
    p = panel(np.vstack([np.zeros(2), np.cumsum(d, axis=0)]), ["x", "y"])
    spec = Spec(drifts=[], breaks=[], regimes=[])
    meta, arrays, nu = estimate(p, spec)
    phi = arrays["momentum"]
    monthly = nu / (nu - 2) * arrays["scale"] / (1 - np.outer(phi, phi))
    weight = meta["shrinkage"]["sample"]
    sample = np.cov(d.T, ddof=1)
    shrunk = (1 - weight) * sample + weight * np.diag(np.diag(sample))
    assert_allclose(monthly, shrunk, rtol=1e-6)
    assert_allclose(arrays["previous"], p.y[-2])


def test_momentum_is_bounded_and_off_for_mean_reverting_variables():
    from mst.plugins.rw.estimate import momentum

    rng = np.random.default_rng(47)
    d = np.column_stack([ar_changes(rng, 0.9, 2000), ar_changes(rng, 0.4, 2000)])
    p = panel(np.vstack([np.zeros(2), np.cumsum(d, axis=0)]), ["x", "y"])
    spec = Spec(drifts=[], breaks=[], regimes=[], reversion=["y"])
    phi, _ = momentum(p, changes(p, spec), spec)
    assert phi[0] == spec.momentum_bound
    assert phi[1] == 0.0
    off = Spec(drifts=[], breaks=[], regimes=[], momentum=False)
    assert_allclose(momentum(p, changes(p, off), off)[0], [0.0, 0.0])


def test_blocks_have_their_own_tails_and_keep_the_sample_covariance():
    rng = np.random.default_rng(23)
    count = 600
    heavy = rng.standard_t(3, size=count)
    light = 0.5 * heavy * np.sqrt(1 / 3) + rng.normal(size=count)
    d = np.column_stack([heavy, light])
    p = panel(np.vstack([np.zeros(2), np.cumsum(d, axis=0)]), ["x", "y"])
    spec = Spec(drifts=[], breaks=[], regimes=[], momentum=False, scale_blocks=[["y"]])
    meta, arrays, nu = estimate(p, spec)
    assert_array_equal(arrays["blocks"], [0, 1])
    assert nu[0] < 5 < 10 < nu[1]
    assert [block["ids"] for block in meta["scale_blocks"]] == [["x"], ["y"]]
    weight = meta["shrinkage"]["sample"]
    sample = np.cov(d.T, ddof=1)
    expected = (1 - weight) * sample + weight * np.diag(np.diag(sample))
    implied = arrays["scale"] * mixing(nu, arrays["blocks"], 0.0, 0.0)
    assert_allclose(implied, expected, rtol=1e-6)
