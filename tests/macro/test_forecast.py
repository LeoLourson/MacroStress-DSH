"""mst.macro.forecast: conditional paths and severity of a scenario."""

from dataclasses import replace

import numpy as np
import pytest
from scipy import stats

from mst.macro.estimate import estimate
from mst.macro.forecast import (
    InfeasibleScenario,
    dependent_rows,
    draw_driving,
    severity,
    simulate,
    truncated_normal,
)
from mst.macro.identify import identify
from mst.macro.registry import Shock
from mst.scenario.contracts import Scenario

from . import simulated as sim

pytestmark = pytest.mark.slow

SHOCKS = [
    Shock(id="gl.s", label="g", rises=["gl.a"]),
    Shock(id="ru.t", label="t", rises=["ru.b", "ru.c"]),
]


@pytest.fixture(scope="module", params=["constant", "stochastic"])
def model(request):
    y = sim.simulate(200, np.random.default_rng(0))
    posterior = estimate(
        sim.panel(y), sim.spec(volatility=request.param, draws=40, burn=40, seed=2)
    )
    return identify(posterior, SHOCKS, sign_lag=2, tries=5000, seed=3)


def scenario(constraints, shocks=(), horizon=6):
    return Scenario.model_validate(
        {
            "narrative": "test",
            "horizon": horizon,
            "constraints": constraints,
            "driving_shocks": list(shocks),
        }
    )


def value(variable, month, low, high=None):
    return {
        "terms": [{"variable": variable, "month": month, "weight": 1.0}],
        "low": low,
        "high": low if high is None else high,
    }


def test_exact_constraints_hold_to_machine_precision_in_every_path(model):
    change = {
        "terms": [
            {"variable": "ru.b", "month": 4, "weight": 1.0},
            {"variable": "ru.b", "month": 1, "weight": -1.0},
        ],
        "low": 2.0,
        "high": 2.0,
    }
    run = simulate(model, scenario([value("gl.a", 3, 9.0), change]), draws=50, seed=1)

    assert run.paths.shape == (50, 7, 3)
    assert np.allclose(run.paths[:, 3, 0], 9.0, atol=1e-9, rtol=0)
    assert np.allclose(run.paths[:, 4, 1] - run.paths[:, 1, 1], 2.0, atol=1e-9, rtol=0)
    assert run.paths[:, 6, 2].std() > 0  # unconstrained paths still vary


def test_interval_constraints_keep_paths_inside_their_bounds(model):
    chosen = scenario(
        [value("ru.c", 2, 5.0, 6.0), value("ru.b", 5, -1.0, 4.0), value("ru.b", 2, 1.0)]
    )
    run = simulate(model, chosen, draws=60, seed=2)
    assert np.all((run.paths[:, 2, 2] >= 5.0) & (run.paths[:, 2, 2] <= 6.0))
    assert np.all((run.paths[:, 5, 1] >= -1.0) & (run.paths[:, 5, 1] <= 4.0))
    assert np.allclose(run.paths[:, 2, 1], 1.0, atol=1e-9, rtol=0)


def test_same_seed_repeats_the_paths_bit_for_bit(model):
    chosen = scenario([value("ru.c", 2, 5.0, 6.0)], shocks=["ru.t"])
    first = simulate(model, chosen, draws=20, seed=5).paths
    assert np.array_equal(first, simulate(model, chosen, draws=20, seed=5).paths)


def test_a_russian_shock_cannot_move_a_global_variable(model):
    chosen = scenario([value("ru.b", 1, 1.0), value("gl.a", 2, 1.0)], shocks=["ru.t"])
    with pytest.raises(InfeasibleScenario, match="constraints 2 ") as caught:
        simulate(model, chosen, draws=5, seed=0)
    assert caught.value.constraints == [1]


def test_a_loading_at_rounding_level_counts_as_none():
    # Driving shocks move the first row by 1e-13 where all shocks move it by about 1.
    z_driving = np.array([[1e-13, 0.0], [0.0, 1.0]])
    z_all = np.array([[1e-13, 0.0, 1.0], [0.0, 1.0, 0.5]])
    assert dependent_rows(z_driving, z_all) == [0]
    assert dependent_rows(np.array([[1.0, 2.0], [2.0, 4.0]]), np.eye(2) * 3) == [1]


def test_one_shock_cannot_fix_two_variables_in_its_impact_month(model):
    chosen = scenario([value("ru.b", 1, 1.0), value("ru.c", 1, 1.0)], shocks=["ru.t"], horizon=1)
    with pytest.raises(InfeasibleScenario):
        simulate(model, chosen, draws=5, seed=0)


def test_unknown_names_are_refused(model):
    with pytest.raises(ValueError, match="unknown shock ru.nope"):
        simulate(model, scenario([value("ru.b", 1, 1.0)], shocks=["ru.nope"]), draws=1, seed=0)


def test_truncated_normal_far_in_the_tail_stays_in_its_interval():
    rng = np.random.default_rng(0)
    draws = [truncated_normal(40.0, 41.0, rng) for _ in range(200)]
    assert all(40.0 <= d <= 41.0 for d in draws) and np.mean(draws) < 40.1


def test_truncated_normal_over_a_wide_interval_is_finite():
    rng = np.random.default_rng(0)
    draws = [truncated_normal(-1e3, 0.5, rng) for _ in range(2000)]
    assert np.all(np.isfinite(draws)) and max(draws) <= 0.5
    assert stats.kstest(draws, stats.truncnorm(-1e3, 0.5).cdf).pvalue > 0.01


def test_gibbs_matches_rejection_sampling_on_two_variables():
    z = np.array([[1.0, 0.5, 0.0], [0.8, 0.6, 0.2]])
    low, high = np.array([0.5, -0.4]), np.array([1.5, 0.6])
    rng = np.random.default_rng(1)
    gibbs = np.array([draw_driving(z, low, high, rng, sweeps=20) for _ in range(4000)])
    raw = rng.standard_normal((400000, 3))
    image = raw @ z.T
    inside = raw[np.all((image >= low) & (image <= high), axis=1)][:4000]
    assert len(inside) == 4000
    for k in range(3):
        assert stats.ks_2samp(gibbs[:, k], inside[:, k]).pvalue > 0.01
    for found, truth in ((gibbs @ z.T, inside @ z.T), (gibbs, inside)):
        assert np.allclose(np.corrcoef(found.T), np.corrcoef(truth.T), atol=0.05)


def test_severity_matches_the_analytic_value_for_a_gaussian_model(model):
    # One draw, constant volatility: W y is exactly normal.
    one = replace(
        model,
        **{
            name: getattr(model, name)[:1]
            for name in ("const", "lags", "dummy", "chol", "logvar", "history", "rotation")
        },
        volvol=np.zeros((1, 3)),
        dof=np.full((1, 3), np.inf),
    )
    lag, const = one.lags[0, 0], one.const[0]
    b = one.chol[0] * np.exp(one.logvar[0, -1] / 2) @ one.rotation[0]
    sigma = b @ b.T
    y0 = one.history[0, -1]
    mean = [const + lag @ y0, const + lag @ (const + lag @ y0)]
    cov11, cov22 = sigma, sigma + lag @ sigma @ lag.T
    cov12 = sigma @ lag.T  # cov(y_1, y_2)
    w_mean = np.array([mean[0][1], mean[1][2]])
    w_cov = np.array([[cov11[1, 1], cov12[1, 2]], [cov12[1, 2], cov22[2, 2]]])
    targets = np.array([w_mean[0] + 2.0, w_mean[1] - 1.0])
    gap = targets - w_mean

    found = severity(
        one,
        scenario([value("ru.b", 1, targets[0]), value("ru.c", 2, targets[1])]),
        draws=4000,
        seed=0,
    )

    assert found.distance == pytest.approx(np.sqrt(gap @ np.linalg.solve(w_cov, gap)), rel=1e-9)
    assert found.probabilities == pytest.approx(stats.norm.cdf(gap / np.sqrt(np.diag(w_cov))))
    assert found.exceeded == pytest.approx(stats.chi2.sf(found.distance**2, 2), abs=0.03)


def test_a_rarer_move_is_more_severe(model):
    mild = severity(model, scenario([value("ru.b", 2, 0.0, 0.0)]), draws=200, seed=0)
    wild = severity(model, scenario([value("ru.b", 2, 50.0, 50.0)]), draws=200, seed=0)
    assert wild.distance > mild.distance and wild.exceeded <= mild.exceeded
    assert wild.probabilities[0] > 0.99


def test_severity_counts_the_spread_between_parameter_draws(model):
    two = replace(
        model,
        **{
            name: getattr(model, name)[:2]
            for name in ("lags", "dummy", "chol", "logvar", "history", "rotation", "volvol", "dof")
        },
        const=model.const[:2] + np.array([[0.0, 0.0, 0.0], [0.0, 6.0, 0.0]]),
    )
    chosen = scenario([value("ru.b", 3, 0.0), value("ru.c", 3, 0.0)])
    found = severity(two, chosen, draws=400, seed=0)
    paths = simulate(two, scenario([], horizon=3), draws=6000, seed=1).paths[:, 3, 1:]
    gap = -paths.mean(axis=0)
    empirical = np.sqrt(gap @ np.linalg.solve(np.cov(paths.T), gap))
    assert found.distance == pytest.approx(empirical, rel=0.05)
