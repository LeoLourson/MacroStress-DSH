"""Conditional paths, the regime element of the event and the law under the MS-VAR."""

from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from mst.model_sdk.contracts import LATENT_CANDIDATES, LinearConstraint, LinearTerm, ScenarioPlan
from mst.model_sdk.dist import price_of_law
from mst.plugins.msvar import estimate, identify
from mst.plugins.msvar.forecast import (
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


def plan(constraints=(), horizon=6) -> ScenarioPlan:
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
    chosen = plan([value("ru.c", 2, 5.0, 6.0)])
    first = simulate(model, chosen, draws=20, seed=5).paths
    assert np.array_equal(first, simulate(model, chosen, draws=20, seed=5).paths)














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


def test_full_noise_law_carries_regime_candidates(model):
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










def toy_posterior(transition, const, logvar):
    """A two-regime posterior of one draw with identity rotations, zero lags and a fixed
    last state: every month is then independent given its regime, and the closed forms of
    the imposition are normal probabilities."""
    from mst.plugins.msvar.estimate import Posterior

    n = const.shape[1]
    draws = 1
    return Posterior(
        ids=[f"v{k}" for k in range(n)],
        n_global=0,
        months=[date(2000 + m // 12, m % 12 + 1, 1) for m in range(24)],
        lags_count=1,
        n_regimes=2,
        const=const[None],
        lags=np.zeros((draws, 2, 1, n, n)),
        chol=np.stack([np.eye(n), np.eye(n)])[None],
        logvar=logvar[None],
        transition=transition[None],
        states=np.zeros((draws, 23), dtype=int),
        history=np.zeros((draws, 1, n)),
        rotation=np.stack([np.eye(n), np.eye(n)])[None],
        shocks=[f"s{k}" for k in range(n)],
        regime_ids=["regime.calm", "regime.stress"],
    )


class TestHiddenRegime:
    """The regime path outside ``R``: the imposition conditions it on the event."""

    def test_a_tail_event_takes_the_stress_regime(self):
        # The stress regime lifts the first variable to 4 with unit noise; the calm regime
        # leaves it at 0. A box around 4.3 that only stress reaches: the imposed paths sit
        # in the stress regime almost surely, where the kernel alone would leave them in
        # the calm one — the share the plan's toy names (0.1 against 0.9999).
        from scipy.stats import norm

        transition = np.array([[0.7, 0.3], [0.5, 0.5]])
        const = np.array([[0.0, 0.0], [4.0, 0.0]])
        logvar = np.zeros((2, 2))
        model = toy_posterior(transition, const, logvar)
        low, high = 4.2, 4.4
        chosen = plan([(([("v0", 1, 1.0)], low, high))])
        run = simulate(model, chosen, draws=600, seed=2)

        # The closed form: P(S | theta) = sum over the first month's regime, every later
        # month unconstrained; the imposed share of stress is the conditional share.
        calm = norm.cdf(high) - norm.cdf(low)
        stress = norm.cdf(high - 4.0) - norm.cdf(low - 4.0)
        probability = 0.3 * stress + 0.7 * calm
        imposed = 0.3 * stress / probability
        # A path is read as stress when its first month sits near the stress mean.
        share = float(np.mean(run.paths[:, 1, 0] > 2.0))
        expected = imposed * norm.sf(2.0 - 4.0) + (1 - imposed) * norm.sf(2.0)
        assert share == pytest.approx(expected, abs=0.06)
        # The price: -log P(S | theta) at the single parameter draw.
        law = constraint_law(model, chosen, draws=1, seed=3)
        distance, _, _ = price_of_law(
            law, np.array([low]), np.array([high]), rng=np.random.default_rng(4), samples=8000
        )
        assert distance == pytest.approx(-np.log(probability), rel=0.05)

    def test_the_price_falls_as_the_candidates_grow(self):
        # The mixture over the regime candidates approaches the closed form from above:
        # the logarithm of an average is bent down by Jensen, and more candidates straighten
        # it — the plan's open question about the bias at a finite M.
        from scipy.stats import norm

        transition = np.array([[0.7, 0.3], [0.5, 0.5]])
        const = np.array([[0.0], [4.0]])
        logvar = np.zeros((2, 1))
        model = toy_posterior(transition, const, logvar)
        low, high = 3.8, 4.6
        chosen = plan([(([("v0", 1, 1.0)], low, high))])
        stress = norm.cdf(high - 4.0) - norm.cdf(low - 4.0)
        calm = norm.cdf(high) - norm.cdf(low)
        closed = -np.log(0.3 * stress + 0.7 * calm)

        prices = {}
        for candidates in (8, 64):
            law = constraint_law(model, chosen, draws=30, seed=3, candidates=candidates)
            distance, _, _ = price_of_law(
                law,
                np.array([low]),
                np.array([high]),
                rng=np.random.default_rng(4),
                samples=4000,
            )
            prices[candidates] = distance
        assert prices[64] == pytest.approx(closed, abs=0.15)
        assert prices[8] >= prices[64] - 0.05  # fewer candidates, more bias upward


    def test_the_parameter_draws_survive_the_conditions(self):
        # The parameters come from their own stream: a scenario and the same scenario
        # without any condition share them, seen in the last month of history every
        # repetition starts from.
        model = model_of_known_draws()
        full = plan([value("gl.a", 2, 1.0, 9.0)])
        plain = plan(horizon=6)
        seeded = simulate(model, full, draws=40, seed=8)
        empty = simulate(model, plain, draws=40, seed=8)
        assert np.array_equal(seeded.paths[:, 0, :], empty.paths[:, 0, :])

    def test_the_conditions_report_their_effective_candidate_sizes(self):
        transition = np.array([[0.98, 0.02], [0.5, 0.5]])
        const = np.array([[0.0], [4.0]])
        logvar = np.zeros((2, 1))
        model = toy_posterior(transition, const, logvar)
        chosen = plan([(([("v0", 1, 1.0)], 4.2, 4.4))])
        run = simulate(model, chosen, draws=30, seed=10)
        sizes = run.diagnostics["latent_ess"]
        assert run.diagnostics["latent_candidates"] == LATENT_CANDIDATES
        assert 1.0 <= sizes["min"] <= sizes["median"] <= sizes["max"] <= LATENT_CANDIDATES






def model_of_known_draws():
    """A posterior of one draw with distinct histories per draw: the t = 0 month reveals
    which parameter set a repetition took."""
    from mst.plugins.msvar.estimate import Posterior

    n = 3
    return Posterior(
        ids=["gl.a", "ru.b", "ru.c"],
        n_global=1,
        months=[date(2000 + m // 12, m % 12 + 1, 1) for m in range(24)],
        lags_count=1,
        n_regimes=2,
        const=sim.CONST[None],
        lags=np.zeros((1, 2, 1, n, n)),
        chol=np.stack([np.eye(n)] * 2)[None],
        logvar=sim.LOGVAR[None],
        transition=sim.TRANSITION[None],
        states=np.zeros((1, 23), dtype=int),
        history=np.tile(np.array([[1.0, 2.0, 3.0]]), (1, 1, 1)),
        rotation=np.stack([np.eye(n)] * 2)[None],
        shocks=["gl.s", "ru.t"],
        regime_ids=["regime.calm", "regime.stress"],
    )


def test_an_exact_tail_event_picks_the_regime_paths_it_needs(model):
    # An exact condition holds under every candidate, but a far value is likelier under some
    # regime paths: the candidates weigh by its density, so they are no longer equal.
    half = np.full((2, 2), 0.5)
    pinned = one_draw(model, transition=half, states=np.zeros((1, model.states.shape[1]), int))
    law = constraint_law(pinned, plan([value("ru.b", 3, 0.0, 1.0)]), draws=1, seed=0)
    far = float(law.means[:, 0].max()) + 4.0 * float(np.sqrt(law.covs[:, 0, 0].max()))
    run = simulate(pinned, plan([value("ru.b", 3, far)]), draws=4, seed=1)
    assert run.diagnostics["latent_ess"]["max"] < LATENT_CANDIDATES
