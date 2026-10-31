"""Conditional paths, the regime element of the event and severity under the MS-VAR."""

from dataclasses import replace

import numpy as np
import pytest
from scipy import stats

from mst.model_sdk.contracts import LinearConstraint, LinearTerm, ScenarioPlan
from mst.plugins.msvar import estimate, identify
from mst.plugins.msvar.forecast import (
    InfeasibleScenario,
    constraint_moments,
    severity,
    simulate,
)
from mst.plugins.msvar.signs import Signs
from mst.plugins.msvar.var import iterate
from tests.msvar import simulated as sim

pytestmark = pytest.mark.slow

SIGNS = Signs.model_validate(
    {
        "shocks": [
            {"id": "gl.s", "label": "g", "rises": ["gl.a"]},
            {"id": "ru.t", "label": "t", "rises": ["ru.b", "ru.c"]},
        ],
        "regimes": [
            {"id": "regime.calm", "label": "спокойный"},
            {"id": "regime.stress", "label": "стрессовый"},
        ],
    }
)


@pytest.fixture(scope="module")
def model():
    y, _ = sim.simulate(240, np.random.default_rng(0))
    posterior = estimate(sim.panel(y), sim.spec(), 2)
    return identify(posterior, SIGNS, sign_lag=2, tries=5000, seed=3)


def plan(constraints=(), shocks=(), horizon=6) -> ScenarioPlan:
    return ScenarioPlan(
        horizon=horizon,
        constraints=tuple(
            LinearConstraint(
                terms=tuple(
                    LinearTerm(variable=variable, month=month, weight=weight)
                    for variable, month, weight in row
                ),
                low=low,
                high=low if high is None else high,
            )
            for row, low, high in constraints
        ),
        driving_shocks=tuple(shocks),
    )


def value(variable, month, low, high=None):
    return ([(variable, month, 1.0)], low, high)


def one_draw(model, *, transition=None, states=None):
    """The model of a single parameter draw, with the transition and state pinned."""
    return replace(
        model,
        const=model.const[:1],
        lags=model.lags[:1],
        chol=model.chol[:1],
        logvar=model.logvar[:1],
        transition=(
            model.transition[:1]
            if transition is None
            else np.tile(transition, (1, 1, 1)).reshape(1, *np.shape(transition))
        ),
        states=model.states[:1] if states is None else states,
        history=model.history[:1],
        rotation=model.rotation[:1],
    )


def regime_mean(model, regime: int, horizon: int) -> np.ndarray:
    """The deterministic path of one regime from the draw's history."""
    draw = 0
    n = len(model.ids)
    lags = np.tile(model.lags[draw, regime][None], (horizon, 1, 1, 1))
    intercept = np.tile(model.const[draw, regime][None], (horizon, 1))
    return iterate(model.history[draw], lags, intercept, np.zeros((horizon, n)))


def test_exact_constraints_hold_to_machine_precision_in_every_path(model):
    change = ([("ru.b", 4, 1.0), ("ru.b", 1, -1.0)], 2.0, 2.0)
    run = simulate(model, plan([value("gl.a", 3, 9.0), change]), draws=50, seed=1)

    assert run.paths.shape == (50, 7, 3)
    assert np.allclose(run.paths[:, 3, 0], 9.0, atol=1e-9, rtol=0)
    assert np.allclose(run.paths[:, 4, 1] - run.paths[:, 1, 1], 2.0, atol=1e-9, rtol=0)
    assert run.paths[:, 6, 2].std() > 0  # unconstrained paths still vary


def test_interval_constraints_keep_paths_inside_their_bounds(model):
    chosen = plan(
        [
            value("ru.c", 2, 5.0, 6.0),
            value("ru.b", 5, -1.0, 4.0),
            value("ru.b", 2, 1.0),
        ]
    )
    run = simulate(model, chosen, draws=60, seed=2)
    assert np.all((run.paths[:, 2, 2] >= 5.0) & (run.paths[:, 2, 2] <= 6.0))
    assert np.all((run.paths[:, 5, 1] >= -1.0) & (run.paths[:, 5, 1] <= 4.0))
    assert np.allclose(run.paths[:, 2, 1], 1.0, atol=1e-9, rtol=0)


def test_a_repeated_constraint_is_redundant_and_still_holds(model):
    chosen = plan([value("ru.c", 2, 5.0), value("ru.c", 2, 5.0)])
    run = simulate(model, chosen, draws=30, seed=3)
    assert np.allclose(run.paths[:, 2, 2], 5.0, atol=1e-9, rtol=0)


def test_same_seed_repeats_the_paths_bit_for_bit(model):
    chosen = plan([value("ru.c", 2, 5.0, 6.0)], shocks=["ru.t"])
    first = simulate(model, chosen, draws=20, seed=5).paths
    assert np.array_equal(first, simulate(model, chosen, draws=20, seed=5).paths)


def test_a_russian_shock_cannot_move_a_global_variable(model):
    chosen = plan([value("ru.b", 1, 1.0), value("gl.a", 2, 1.0)], shocks=["ru.t"])
    with pytest.raises(InfeasibleScenario, match="constraints 2 ") as caught:
        simulate(model, chosen, draws=5, seed=0)
    assert caught.value.constraints == [1]


def test_one_shock_cannot_fix_two_variables_in_its_impact_month(model):
    chosen = plan([value("ru.b", 1, 1.0), value("ru.c", 1, 1.0)], shocks=["ru.t"], horizon=1)
    with pytest.raises(InfeasibleScenario):
        simulate(model, chosen, draws=5, seed=0)


def test_unknown_names_are_refused(model):
    with pytest.raises(ValueError, match="unknown shock ru.nope"):
        simulate(model, plan([value("ru.b", 1, 1.0)], shocks=["ru.nope"]), draws=1, seed=0)


def test_conflicting_regime_elements_are_refused(model):
    chosen = plan([value("ru.b", 1, 1.0)], shocks=["regime.stress", "regime.calm"])
    with pytest.raises(ValueError, match="conflicting regime elements"):
        simulate(model, chosen, draws=1, seed=0)


def test_an_unknown_regime_element_is_refused(model):
    with pytest.raises(ValueError, match="unknown shock regime.nope"):
        simulate(model, plan([value("ru.b", 1, 1.0)], shocks=["regime.nope"]), draws=1, seed=0)


def test_the_regime_event_holds_the_paths_in_the_named_regime(model):
    pinned = one_draw(model, transition=np.eye(2), states=np.zeros((1, model.states.shape[1]), int))
    horizon = 4
    run = simulate(
        pinned,
        plan(shocks=["regime.stress"], horizon=horizon),
        draws=400,
        seed=2,
    )
    expected = regime_mean(pinned, 1, horizon)
    # The paths follow the stress regime's own law; four standard errors of the mean.
    tolerance = 4 * np.exp(pinned.logvar[0, 1] / 2) * np.sqrt(horizon / 400)
    assert np.allclose(run.paths[:, 1:].mean(axis=0), expected, atol=tolerance)
    assert not np.allclose(run.paths[:, 1:].mean(axis=0), regime_mean(pinned, 0, horizon))


def test_the_unconditional_run_mixes_the_regimes(model):
    half = np.full((2, 2), 0.5)
    pinned = one_draw(model, transition=half, states=np.zeros((1, model.states.shape[1]), int))
    horizon = 5
    run = simulate(pinned, plan(horizon=horizon), draws=600, seed=3)
    # With independent regimes the mixture mean follows the average dynamics.
    average = regime_mean(
        replace(
            pinned,
            lags=pinned.lags[:1].mean(axis=1, keepdims=True),
            const=pinned.const[:1].mean(axis=1, keepdims=True),
        ),
        0,
        horizon,
    )
    error = 4 * run.paths[:, 1:].std(axis=0) / np.sqrt(len(run.paths))
    assert np.allclose(run.paths[:, 1:].mean(axis=0), average, atol=error)
    assert not np.allclose(
        run.paths[:, 1:].mean(axis=0), regime_mean(pinned, 0, horizon)
    )  # the mixture is neither regime alone


def test_the_moments_count_the_spread_between_regimes(model):
    half = np.full((2, 2), 0.5)
    pinned = one_draw(model, transition=half, states=np.zeros((1, model.states.shape[1]), int))
    means, covs = constraint_moments(pinned, plan([value("ru.b", 3, 0.0, 1.0)]), draws=400, seed=0)
    # Every repetition carries its own regime path, so the means of the two regimes both
    # appear and the variance of the means is positive.
    assert means.min() < means.max()
    assert means.var() > 0
    assert covs.shape == (400, 1, 1) and (covs > 0).all()


def test_severity_matches_the_analytic_value_for_a_gaussian_model(model):
    # One draw, the regime pinned to calm: W y is exactly normal.
    pinned = one_draw(model, transition=np.eye(2), states=np.zeros((1, model.states.shape[1]), int))
    lag, const = pinned.lags[0, 0, 0], pinned.const[0, 0]
    b = pinned.chol[0, 0] * np.exp(pinned.logvar[0, 0] / 2) @ pinned.rotation[0, 0]
    sigma = b @ b.T
    y0 = pinned.history[0, -1]
    mean = [const + lag @ y0, const + lag @ (const + lag @ y0)]
    cov11, cov22 = sigma, sigma + lag @ sigma @ lag.T
    cov12 = sigma @ lag.T  # cov(y_1, y_2)
    w_mean = np.array([mean[0][1], mean[1][2]])
    w_cov = np.array([[cov11[1, 1], cov12[1, 2]], [cov12[1, 2], cov22[2, 2]]])
    targets = np.array([w_mean[0] + 2.0, w_mean[1] - 1.0])
    gap = targets - w_mean

    found = severity(
        pinned,
        plan([value("ru.b", 1, targets[0]), value("ru.c", 2, targets[1])]),
        draws=4000,
        seed=0,
    )

    assert found.distance == pytest.approx(np.sqrt(gap @ np.linalg.solve(w_cov, gap)), rel=1e-9)
    assert found.probabilities == pytest.approx(stats.norm.cdf(gap / np.sqrt(np.diag(w_cov))))
    assert found.exceeded == pytest.approx(stats.chi2.sf(found.distance**2, 2), abs=0.03)


def test_a_rarer_move_is_more_severe(model):
    mild = severity(model, plan([value("ru.b", 2, 0.0, 0.0)]), draws=200, seed=0)
    wild = severity(model, plan([value("ru.b", 2, 50.0, 50.0)]), draws=200, seed=0)
    assert wild.distance > mild.distance and wild.exceeded <= mild.exceeded
    assert wild.probabilities[0] > 0.99


def test_severity_counts_the_spread_between_regimes(model):
    half = np.full((2, 2), 0.5)
    pinned = one_draw(model, transition=half, states=np.zeros((1, model.states.shape[1]), int))
    chosen = plan([value("ru.b", 3, 0.0), value("ru.c", 3, 0.0)])
    found = severity(pinned, chosen, draws=400, seed=0)
    paths = simulate(pinned, plan(horizon=3), draws=6000, seed=1).paths[:, 3, 1:]
    gap = -paths.mean(axis=0)
    empirical = np.sqrt(gap @ np.linalg.solve(np.cov(paths.T), gap))
    assert found.distance == pytest.approx(empirical, rel=0.05)
