"""Conditional paths and the constraint law of a scenario under the BVAR."""

from dataclasses import replace

import numpy as np
import pytest

from mst.model_sdk.contracts import LATENT_CANDIDATES, LinearConstraint, LinearTerm, ScenarioPlan
from mst.model_sdk.dist import price_of_law
from mst.plugins.bvar.estimate import estimate
from mst.plugins.bvar.forecast import InfeasibleScenario, constraint_law, simulate
from mst.plugins.bvar.identify import identify
from mst.plugins.bvar.signs import Shock
from tests.bvar import simulated as sim

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


def scenario(constraints, horizon=6):
    return ScenarioPlan(
        horizon=horizon,
        constraints=tuple(
            LinearConstraint(
                terms=tuple(
                    LinearTerm(
                        variable=term["variable"], month=term["month"], weight=term["weight"]
                    )
                    for term in row["terms"]
                ),
                low=row["low"],
                high=row["high"],
            )
            for row in constraints
        ),
    )


def value(variable, month, low, high=None):
    return {
        "terms": [{"variable": variable, "month": month, "weight": 1.0}],
        "low": low,
        "high": low if high is None else high,
    }


def test_exact_constraints_hold_to_machine_precision_in_every_path(model):
    run = simulate(
        model,
        scenario(
            [
                value("gl.a", 3, 9.0),
                {
                    "terms": [
                        {"variable": "ru.b", "month": 4, "weight": 1.0},
                        {"variable": "ru.b", "month": 1, "weight": -1.0},
                    ],
                    "low": 2.0,
                    "high": 2.0,
                },
            ]
        ),
        draws=50,
        seed=1,
    )

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


def test_a_repeated_constraint_is_redundant_and_still_holds(model):
    # Two copies of one constraint are consistent: the redundant row is dropped, not refused.
    chosen = scenario([value("ru.c", 2, 5.0), value("ru.c", 2, 5.0)])
    run = simulate(model, chosen, draws=30, seed=3)
    assert np.allclose(run.paths[:, 2, 2], 5.0, atol=1e-9, rtol=0)


def test_a_scaled_copy_of_a_row_is_dependent_and_reduced(model):
    # A copy with another weight loads the same direction of every shock: the second row
    # adds no new direction, so the dependence check refuses it.
    chosen = scenario(
        [
            value("ru.b", 2, 0.0, 1.0),
            {"terms": [{"variable": "ru.b", "month": 2, "weight": 2.0}],
             "low": 0.0, "high": 2.0},
        ]
    )
    with pytest.raises(InfeasibleScenario):
        simulate(model, chosen, draws=5, seed=0)


def test_same_seed_repeats_the_paths_bit_for_bit(model):
    chosen = scenario([value("ru.c", 2, 5.0, 6.0)])
    first = simulate(model, chosen, draws=20, seed=5).paths
    assert np.array_equal(first, simulate(model, chosen, draws=20, seed=5).paths)


def test_all_shocks_are_conditioned(model):
    # No cause names the driving shocks: an observation on a global variable and one on a
    # Russian variable are met together, in the same month, by moving every shock. The old
    # block-exogeneity restriction of a named shock no longer applies.
    chosen = scenario([value("gl.a", 2, 1.0), value("ru.b", 2, 1.0)])
    run = simulate(model, chosen, draws=10, seed=0)
    assert np.allclose(run.paths[:, 2, 0], 1.0, atol=1e-9, rtol=0)
    assert np.allclose(run.paths[:, 2, 1], 1.0, atol=1e-9, rtol=0)
    assert "driving_sd" in run.diagnostics


def test_two_variables_are_met_in_the_same_month(model):
    # Two observations in one impact month used to need two separate shocks because a single
    # named cause could not fix both; every shock now contributes, so both rows hold.
    chosen = scenario([value("ru.b", 1, 1.0), value("ru.c", 1, 0.5)], horizon=1)
    run = simulate(model, chosen, draws=10, seed=0)
    assert np.allclose(run.paths[:, 1, 1], 1.0, atol=1e-9, rtol=0)
    assert np.allclose(run.paths[:, 1, 2], 0.5, atol=1e-9, rtol=0)


def test_unknown_variables_are_refused(model):
    with pytest.raises(ValueError, match="unknown variable ru.nope"):
        simulate(model, scenario([value("ru.nope", 1, 1.0)]), draws=1, seed=0)


def test_the_law_of_one_gaussian_draw_matches_the_analytic_value(model):
    # One parameter draw, constant volatility, one candidate: the law is exactly normal,
    # with the covariance of the base measure, which counts every shock. The third shock is
    # zeroed, so the two remaining shocks carry all the variation and the law is the
    # analytic gaussian of the two constraints; the price is -log of their box probability.
    rotation = model.rotation[:1].copy()
    rotation[:, :, 2] = 0.0
    one = replace(
        model,
        **{
            name: getattr(model, name)[:1]
            for name in ("const", "lags", "dummy", "chol", "logvar", "history")
        },
        rotation=rotation,
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
    low = np.array([w_mean[0] + 1.5, w_mean[1] - 1.0])
    high = np.array([w_mean[0] + 2.5, w_mean[1] + 0.0])

    found = constraint_law(
        one,
        scenario(
            [
                {"terms": [{"variable": "ru.b", "month": 1, "weight": 1.0}],
                 "low": low[0], "high": high[0]},
                {"terms": [{"variable": "ru.c", "month": 2, "weight": 1.0}],
                 "low": low[1], "high": high[1]},
            ],
        ),
        draws=4,
        seed=0,
        candidates=1,
    )
    assert found.means.shape == (4, 2) and found.covs.shape == (4, 2, 2)
    for k in range(4):
        assert np.allclose(found.means[k], w_mean)
        assert np.allclose(found.covs[k], w_cov)
    distance, _, _ = price_of_law(
        found, low, high, rng=np.random.default_rng(1), samples=4000
    )
    from scipy.stats import multivariate_normal

    exact = multivariate_normal(w_mean, w_cov).cdf(high, lower_limit=low)
    assert distance == pytest.approx(-np.log(exact), rel=0.02)


def test_the_law_carries_volatility_candidates(model):
    # The volatility path is a hidden coordinate: every parameter draw owns a group of
    # LATENT_CANDIDATES rows. Under stochastic volatility the candidates differ in their
    # covariance; under the constant variant every candidate is the same draw repeated.
    found = constraint_law(
        model, scenario([value("ru.b", 2, 0.0, 1.0)]), draws=3, seed=0
    )
    assert len(found.group) == 3 * LATENT_CANDIDATES
    assert set(found.group.tolist()) == {0, 1, 2}
    per_group = [found.covs[found.group == g].ravel() for g in range(3)]
    if model.volvol[0].max() > 0:
        assert any(rows.min() < rows.max() for rows in per_group)
    else:
        assert all(np.allclose(rows, rows[0]) for rows in per_group)


def test_the_law_covariance_counts_every_shock(model):
    # The price law is the base law of the functionals given R, so every shock enters its
    # covariance. Zeroing one shock's column shrinks the covariance; a law that counted
    # only the named shocks would not change.
    one = replace(
        model,
        **{
            name: getattr(model, name)[:1]
            for name in (
                "const", "lags", "dummy", "chol", "logvar", "history", "rotation",
                "volvol", "dof",
            )
        },
    )
    box = scenario([value("ru.b", 2, 0.0, 1.0)])
    full = constraint_law(one, box, draws=1, seed=0, candidates=1)
    rotation = one.rotation.copy()
    rotation[:, :, 2] = 0.0
    clipped = constraint_law(
        replace(one, rotation=rotation), box, draws=1, seed=0, candidates=1
    )
    assert clipped.covs[0, 0, 0] < full.covs[0, 0, 0]


def test_a_tail_event_prices_the_volatility_candidates(model):
    # A tail event needs volatility: the candidate rows of a group mix the volatile paths
    # in, so a box in the tail costs less than the first candidate alone would say — the
    # mixture reaches further than one unconditional draw.
    one = replace(
        model,
        **{name: getattr(model, name)[:1] for name in ("const", "lags", "dummy", "chol",
                                                        "logvar", "history", "rotation",
                                                        "volvol", "dof")},
    )
    if one.volvol[0].max() <= 0:
        pytest.skip("the constant variant has no volatility to price")
    reference = constraint_law(one, scenario([value("ru.b", 2, 0.0, 1.0)]), draws=1, seed=0)
    spread = float(np.sqrt(reference.covs[0, 0, 0]))
    low = float(reference.means[0, 0]) + 3.0 * spread
    chosen = scenario([value("ru.b", 2, low, low + 0.5)])
    law = constraint_law(one, chosen, draws=8, seed=0)
    first_only = replace(
        law,
        means=law.means[::LATENT_CANDIDATES],
        covs=law.covs[::LATENT_CANDIDATES],
        log_weight=law.log_weight[::LATENT_CANDIDATES],
        group=law.group[::LATENT_CANDIDATES],
    )
    rng = np.random.default_rng(3)
    distance, _, _ = price_of_law(
        law, np.array([low]), np.array([low + 0.5]), rng=rng, samples=4000
    )
    single, _, _ = price_of_law(
        first_only, np.array([low]), np.array([low + 0.5]), rng=rng, samples=4000
    )
    assert distance < single


def test_an_exact_tail_event_picks_the_volatility_it_needs(model):
    # An exact condition holds under every candidate, but a far value is likelier under a
    # volatile path: the candidates weigh by its density, so they are no longer equal.
    if model.volvol[0].max() <= 0:
        pytest.skip("the constant variant has no volatility to weigh")
    reference = constraint_law(model, scenario([value("ru.b", 2, 0.0, 1.0)]), draws=1, seed=0)
    far = float(reference.means[0, 0]) + 4.0 * float(np.sqrt(reference.covs[0, 0, 0]))
    run = simulate(model, scenario([value("ru.b", 2, far)]), draws=4, seed=1)
    assert run.diagnostics["latent_ess"]["max"] < LATENT_CANDIDATES


def test_the_diagnostics_count_the_paths_that_need_huge_driving_shocks(model):
    # A point far outside the law of a functional needs a shock of many standard deviations
    # on every path, and the admission of the run reads the share; a box around the mean
    # asks for a moderate draw and is not flagged.
    reference = constraint_law(
        model, scenario([value("ru.b", 2, 0.0, 1.0)]), draws=1, seed=0, candidates=1
    )
    mean = float(reference.means[0, 0])
    spread = float(np.sqrt(reference.covs[0, 0, 0]))
    far = mean + 50.0 * spread
    run = simulate(model, scenario([value("ru.b", 2, far)]), draws=8, seed=1)
    assert run.diagnostics["driving_sd"]["share_above_limit"] == 1.0
    near = simulate(
        model,
        scenario([value("ru.b", 2, mean - 3.0 * spread, mean + 3.0 * spread)]),
        draws=8,
        seed=1,
    )
    assert near.diagnostics["driving_sd"]["share_above_limit"] == 0.0


def test_a_jump_widens_the_impact_of_its_variable(model):
    # A jump in every month with 100 times the variance multiplies the scale of the jumping
    # component by ten and leaves the others alone.
    from mst.plugins.bvar.forecast import _repetition

    if np.isfinite(model.dof).any():
        pytest.skip("the Student draws of the stochastic variant make the scales random")
    probability = np.zeros_like(model.volvol)
    scale = np.ones_like(model.volvol)
    probability[:, 1], scale[:, 1] = 1.0, 100.0
    jumping = replace(model, jump_prob=probability, jump_scale=scale)
    calm = _repetition(model, 3, np.random.default_rng(5), draw=0)
    wide = _repetition(jumping, 3, np.random.default_rng(5), draw=0)
    factor = np.array([1.0, 10.0, 1.0])
    calm_shocks = np.linalg.solve(model.chol[0], calm.impacts[0])
    expected = np.einsum("ij,j,jk->ik", model.chol[0], factor, calm_shocks)
    assert np.allclose(wide.impacts[0], expected)
