"""The conditional forecast and the price when the residual is among the causes."""

import numpy as np
import pytest
from scipy.stats import norm

from mst.model_sdk.contracts import LinearConstraint, LinearTerm, ScenarioPlan
from mst.model_sdk.dist import price_of_law
from mst.plugins.expert.forecast import constraint_law, simulate
from tests.expert import fakes


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    """A loaded expert model of the fixture's answers, with every frequency zero."""
    from mst.plugins.expert.estimate import estimate

    ctx = fakes.answers_context(
        tmp_path_factory.mktemp("series"),
        changes={
            key: (lambda payload: {**payload, "frequency": 0.0}) for key in fakes.answers()
        },
    )
    found = estimate(ctx, base="rw", spec=fakes.quick_spec())
    from mst.plugins.expert.estimate import load

    return load(found.meta, found.arrays)


def plan(constraints=(), causes=(), horizon=6, inner=1) -> ScenarioPlan:
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
        inner=inner,
    )


def test_zero_frequencies_match_the_random_walk_benchmark(model):
    found = simulate(model, plan(), draws=800, seed=5)
    assert found.paths.shape == (800, 7, 23)
    base = model.base_path[:7]
    # The mean and the spread per month: y_t = x0 + t mu + N(0, t Sigma) cumulated.
    draws = found.paths.shape[0]
    tolerance = 4 * np.array(
        [np.linalg.norm(model.chol_u, axis=0) * np.sqrt(t) for t in range(7)]
    ) / np.sqrt(draws)
    assert np.all(np.abs(found.paths.mean(axis=0) - base) <= tolerance + 1e-9)
    for t in (3, 6):
        for k in (0, 5, 22):
            observed = found.paths[:, t, k].std()
            expected = np.sqrt(t) * np.linalg.norm(model.chol_u[k])
            assert observed == pytest.approx(expected, rel=0.15), (t, k)
    assert found.group is not None and set(found.group) == set(range(draws))


def test_every_simulated_path_meets_the_constraints(model):
    level = model.base_path[3, model.ids.index("ru.key_rate")]
    found = simulate(
        model,
        plan(constraints=[([("ru.key_rate", 3, 1.0)], level - 0.05, level + 0.05)]),
        draws=200,
        seed=7,
    )
    values = found.paths[:, 3, found.ids.index("ru.key_rate")]
    assert np.all(values >= level - 0.05 - 1e-8) and np.all(values <= level + 0.05 + 1e-8)
    assert found.diagnostics["latent_candidates"] > 1


def test_an_impossible_constraint_is_refused(model):
    from mst.model_sdk.conditional import InfeasibleScenario

    with pytest.raises(InfeasibleScenario):
        simulate(
            model,
            plan(
                constraints=[
                    ([("ru.key_rate", 1, 1.0)], 1.0, 2.0),
                    ([("ru.key_rate", 1, 1.0)], 1.5, 2.5),
                    ([("ru.key_rate", 1, 2.0)], 1.0, 2.0),
                ]
            ),
            draws=2,
            seed=1,
        )


def test_the_price_of_one_interval_is_the_gaussian_probability(model):
    chosen = plan(constraints=[([("ru.key_rate", 4, 1.0)], 8.42, 8.48)])
    law = constraint_law(model, chosen, draws=40, seed=11)
    assert law is not None and len(law.means) == 40 * model.candidates
    rng = np.random.default_rng(0)
    distance, _, error = price_of_law(
        law, np.array([8.42]), np.array([8.48]), rng=rng
    )
    mean = law.means[:, 0].mean()
    variance = law.covs[0, 0, 0]
    exact = -np.log(
        norm.cdf((8.48 - mean) / np.sqrt(variance))
        - norm.cdf((8.42 - mean) / np.sqrt(variance))
    )
    assert distance == pytest.approx(exact, abs=3 * max(error, 1e-12))


def test_an_expectation_condition_moves_the_mean_through_the_overlay(model):
    from mst.scenario.overlay import solve_lambda

    found = simulate(model, plan(inner=32, horizon=6), draws=20, seed=3)
    values = found.paths[:, 6, found.ids.index("ru.usdrub")][:, None]
    target = values.mean() + 0.5
    tilt = solve_lambda(values, found.group, np.array([target]), np.array([np.inf]))
    assert tilt.means[0] == pytest.approx(target, rel=0.05)
    assert tilt.price > 0


def test_a_named_large_shock_makes_a_large_move_cheaper(tmp_path_factory):
    from mst.plugins.expert.estimate import estimate, load

    def answer(scale, frequency):
        def change(payload):
            responses = {
                variable: {
                    horizon: {
                        node: (
                            {
                                "q5": scale * cell["q5"],
                                "q50": scale * cell["q50"],
                                "q95": scale * cell["q95"],
                            }
                            if variable == "gl.brent"
                            else dict(cell)
                        )
                        for node, cell in by_node.items()
                    }
                    for horizon, by_node in by_horizon.items()
                }
                for variable, by_horizon in payload["responses"].items()
            }
            return {**payload, "frequency": frequency, "rho": 0.9, "responses": responses}

        return change

    def prepared(frequency):
        found = estimate(
            fakes.answers_context(
                tmp_path_factory.mktemp("series"),
                changes={"gl.oil": answer(120.0, frequency)},
            ),
            base="rw",
            spec=fakes.quick_spec(),
        )
        return load(found.meta, found.arrays)

    big = prepared(3.0)
    quiet = prepared(0.0)
    column = big.ids.index("gl.brent")
    # The same large move of every version's own base: what the answer's 95 % size
    # makes at the third month. Naming the shock that explains the move is cheaper
    # than leaving it to the residual, whatever the base the move is measured from.
    move = big.tables[0].responses[column, 1, 2].mean()

    def price_of(model, causes):
        target = model.base_path[3, column] + move
        bounds = (np.array([target - 0.3]), np.array([target + 0.3]))
        chosen = plan(
            constraints=[([("gl.brent", 3, 1.0)], target - 0.3, target + 0.3)],
            causes=causes,
        )
        law = constraint_law(model, chosen, draws=16, seed=13)
        return price_of_law(law, *bounds, rng=np.random.default_rng(0))[0]

    assert price_of(big, ("gl.oil", "residual")) < price_of(quiet, ("residual",))


def test_the_same_seed_gives_the_same_paths_and_price(model):
    chosen = plan(constraints=[([("gl.brent", 2, 1.0)], 480.0, 520.0)])
    first = simulate(model, chosen, draws=50, seed=21)
    second = simulate(model, chosen, draws=50, seed=21)
    assert np.array_equal(first.paths, second.paths)
    law = constraint_law(model, chosen, draws=20, seed=21)
    again = constraint_law(model, chosen, draws=20, seed=21)
    assert np.array_equal(law.means, again.means)


def test_causes_without_the_residual_go_to_subset_simulation(model):
    from mst.model_sdk.dist import ContradictsModels
    from mst.plugins.expert.forecast import box_probabilities

    chosen = plan(causes=("ru.monetary",))
    assert constraint_law(model, chosen, draws=2, seed=1) is None, "no law carries them"
    assert (
        box_probabilities(model, chosen, np.array([0.0]), np.array([1.0]), draws=2, seed=1)
        is None
    ), "a plan with nothing to price has no box probabilities either"
    # A box the quiet shock cannot reach contradicts the models: the imposition of these
    # plans is the subset machinery, which test_subset checks on a speaking model.
    level = model.base_path[2, model.ids.index("ru.key_rate")]
    far = plan(
        constraints=[([("ru.key_rate", 2, 1.0)], level + 1.0, level + 1.1)],
        causes=("ru.monetary",),
    )
    with pytest.raises(ContradictsModels):
        simulate(model, far, draws=2, seed=1)
