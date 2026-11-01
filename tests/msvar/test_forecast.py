"""Conditional paths, the regime element of the event and the law under the MS-VAR."""

from dataclasses import replace

import numpy as np
import pytest

from mst.model_sdk.contracts import LATENT_CANDIDATES, LinearConstraint, LinearTerm, ScenarioPlan
from mst.model_sdk.dist import price_of_law
from mst.plugins.msvar import estimate, identify
from mst.plugins.msvar.forecast import (
    InfeasibleScenario,
    constraint_law,
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


def plan(constraints=(), causes=(), horizon=6, regime_event=None) -> ScenarioPlan:
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
        causes=tuple(causes),
        regime_event=regime_event,
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
    chosen = plan([value("ru.c", 2, 5.0, 6.0)], causes=["ru.t"])
    first = simulate(model, chosen, draws=20, seed=5).paths
    assert np.array_equal(first, simulate(model, chosen, draws=20, seed=5).paths)


def test_a_russian_shock_cannot_move_a_global_variable(model):
    chosen = plan([value("ru.b", 1, 1.0), value("gl.a", 2, 1.0)], causes=["ru.t"])
    with pytest.raises(InfeasibleScenario, match="constraints 2 ") as caught:
        simulate(model, chosen, draws=5, seed=0)
    assert caught.value.constraints == [1]


def test_one_shock_cannot_fix_two_variables_in_its_impact_month(model):
    chosen = plan([value("ru.b", 1, 1.0), value("ru.c", 1, 1.0)], causes=["ru.t"], horizon=1)
    with pytest.raises(InfeasibleScenario):
        simulate(model, chosen, draws=5, seed=0)


def test_unknown_causes_are_refused(model):
    with pytest.raises(ValueError, match="unknown cause ru.nope"):
        simulate(model, plan([value("ru.b", 1, 1.0)], causes=["ru.nope"]), draws=1, seed=0)


def test_an_unknown_regime_event_is_refused(model):
    with pytest.raises(ValueError, match="unknown regime element regime.nope"):
        simulate(
            model,
            plan([value("ru.b", 1, 1.0)], regime_event="regime.nope"),
            draws=1,
            seed=0,
        )


def test_a_regime_event_without_the_regime_among_its_causes_is_refused(model):
    # The event fixes the regime, so its law cannot stay as it was: the regime path has to
    # be named as a cause, or no cause at all (lemma 2 of the canon).
    with pytest.raises(ValueError, match="needs regime_path among its causes"):
        simulate(
            model,
            plan([value("ru.b", 1, 1.0)], causes=["ru.t"], regime_event="regime.stress"),
            draws=1,
            seed=0,
        )
    # With the regime path among the causes, or with no causes at all, the plan is sound.
    simulate(
        model,
        plan([value("ru.b", 1, 1.0)], causes=["regime_path"], regime_event="regime.stress"),
        draws=1,
        seed=0,
    )
    simulate(
        model,
        plan([value("ru.b", 1, 1.0)], regime_event="regime.stress"),
        draws=1,
        seed=0,
    )


def test_the_regime_event_holds_the_paths_in_the_named_regime(model):
    pinned = one_draw(model, transition=np.eye(2), states=np.zeros((1, model.states.shape[1]), int))
    horizon = 4
    run = simulate(
        pinned,
        plan(horizon=horizon, regime_event="regime.stress"),
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


def test_without_causes_the_law_carries_regime_candidates(model):
    # The regime path is a hidden coordinate: every parameter draw owns a group of
    # LATENT_CANDIDATES rows, and the candidate regime paths spread the means of the group.
    half = np.full((2, 2), 0.5)
    pinned = one_draw(model, transition=half, states=np.zeros((1, model.states.shape[1]), int))
    found = constraint_law(pinned, plan([value("ru.b", 3, 0.0, 1.0)]), draws=3, seed=0)
    assert len(found.group) == 3 * LATENT_CANDIDATES
    assert set(found.group.tolist()) == {0, 1, 2}
    for g in range(3):
        rows = found.means[found.group == g]
        assert rows.min() < rows.max()


def test_with_named_shocks_the_law_has_one_row_per_group(model):
    half = np.full((2, 2), 0.5)
    pinned = one_draw(model, transition=half, states=np.zeros((1, model.states.shape[1]), int))
    found = constraint_law(
        pinned, plan([value("ru.b", 3, 0.0, 1.0)], causes=["ru.t"]), draws=5, seed=0
    )
    assert len(found.group) == 5
    assert set(found.group.tolist()) == set(range(5))
    assert np.all(found.log_weight == 0.0)


def test_the_regime_event_costs_the_log_of_the_path_prior(model):
    # A regime event with no conditions on y: every group is one forced regime path whose
    # row carries the prior weight pi_theta(k, .., k), and the price is the mean of
    # -log pi_theta(k, .., k) — the canon's price of fixing the regime.
    pinned = one_draw(model, transition=np.eye(2), states=np.zeros((1, model.states.shape[1]), int))
    horizon = 4
    found = constraint_law(
        pinned, plan(horizon=horizon, regime_event="regime.stress"), draws=1, seed=0
    )
    assert found.means.shape == (1, 0) and found.covs.shape == (1, 0, 0)
    prior = 1.0
    state = int(pinned.states[0, -1])
    for _ in range(horizon):
        prior *= pinned.transition[0, state, 1]
        state = 1
    assert found.log_weight[0] == pytest.approx(np.log(prior))
    distance, log_probability, _ = price_of_law(
        found, np.zeros(0), np.zeros(0), rng=np.random.default_rng(1)
    )
    assert distance == pytest.approx(-np.log(prior))
    assert log_probability == pytest.approx(np.log(prior))


def test_a_regime_event_with_conditions_prices_the_box_at_the_regime(model):
    # The forced regime row carries the path prior and the box probability under that
    # regime: P(S | R) = pi_theta(k,..,k) * P(box | theta, k..k).
    pinned = one_draw(model, transition=np.eye(2), states=np.zeros((1, model.states.shape[1]), int))
    chosen = plan([value("ru.b", 2, 5.0, 6.0)], regime_event="regime.stress")
    found = constraint_law(pinned, chosen, draws=1, seed=0)
    assert found.means.shape == (1, 1)
    distance, _, _ = price_of_law(
        found, np.array([5.0]), np.array([6.0]), rng=np.random.default_rng(2), samples=4000
    )
    # The closed form: -log of pi times the exact probability of the box under the one
    # gaussian row the group carries.
    from scipy.stats import norm

    box = norm.cdf(6.0, found.means[0, 0], np.sqrt(found.covs[0, 0, 0])) - norm.cdf(
        5.0, found.means[0, 0], np.sqrt(found.covs[0, 0, 0])
    )
    assert distance == pytest.approx(-(found.log_weight[0] + np.log(box)), abs=0.05)
