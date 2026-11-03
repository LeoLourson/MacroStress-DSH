"""Subset simulation: a small probability level by level, and the price it carries."""

import numpy as np
import pytest
from scipy.special import ndtri

from mst.model_sdk.contracts import LinearConstraint, LinearTerm, ScenarioPlan
from mst.plugins.expert.forecast import box_probabilities, simulate
from mst.plugins.expert.laws import level_grid, pareto_quantile
from mst.plugins.expert.stream import ShockTables
from mst.plugins.expert.subset import subset
from tests.expert import fakes


def test_a_gaussian_tail_probability_of_one_in_a_million():
    # The box is x >= 4.75 standard deviations: P = 1e-6, six levels of p0 = 0.1.
    threshold = float(ndtri(1.0 - 1e-6))
    result = subset(
        lambda x: threshold - x[:, 0],
        1,
        samples=2000,
        rng=np.random.default_rng(1),
    )
    probability = float(np.exp(result.log_probability))
    error = result.coefficient_of_variation * probability
    assert abs(probability - 1e-6) < 3 * error
    assert len(result.inside) > 0 and np.all(result.inside >= threshold)


def test_the_reported_spread_follows_the_spread_over_independent_runs():
    # Four levels of p0 = 0.1: every level adds its spread, so the reported coefficient of
    # variation stays within a factor of two of what independent runs show.
    threshold = float(ndtri(1.0 - 1e-4))
    found = [
        subset(lambda x: threshold - x[:, 0], 1, samples=500, rng=np.random.default_rng(k))
        for k in range(100)
    ]
    probabilities = np.exp([result.log_probability for result in found])
    empirical = probabilities.std() / probabilities.mean()
    reported = np.mean([result.coefficient_of_variation for result in found])
    assert empirical / 2.0 < reported < empirical * 2.0, (reported, empirical)


def test_the_samples_inside_the_box_are_inside():
    result = subset(
        lambda x: np.abs(x).max(axis=1) - 2.5, 2, samples=1000, rng=np.random.default_rng(2)
    )
    assert np.all(np.abs(result.inside) <= 2.5 + 1e-12)


def _linear_model(frequency=1e6, sizes=(-1.0, 0.0, 1.0), slope=1.0, residual=0.0):
    """A one-variable expert model: the shock always occurs and moves it linearly.

    ``residual`` is the standard deviation of the residual's monthly step; zero makes the
    R part of every group deterministic, so a closed form can be written down.
    """
    from mst.plugins.expert.estimate import Expert

    grid = level_grid(129)
    # The response of the single variable at every (horizon node, size node): a constant
    # law whose value is proportional to the asked size, so the size line is straight.
    values = slope * np.asarray(sizes)
    responses = np.broadcast_to(
        values[None, None, :, None], (1, 4, 3, len(grid))
    ).copy()
    medians = np.broadcast_to(values[None, None, :], (1, 4, 3)).copy()
    tables = ShockTables(
        frequency=frequency,
        rho=0.5,
        sizes=np.array(sizes),
        size_table=pareto_quantile(*sizes, grid, 4.0),
        responses=responses,
        medians=medians,
        levels=grid,
        alpha=4.0,
    )
    import datetime as dt

    return Expert(
        ids=["ru.cpi"],
        months=[dt.date(2026, 10, 1)],
        shocks=["gl.oil", "residual"],
        x0=np.array([100.0]),
        base_path=np.array([[100.0], [101.0], [102.0], [103.0], [104.0]]),
        chol_u=np.array([[residual]]),
        tables=(tables,),
        levels=grid,
        candidates=16,
        subset_samples=500,
    )


def test_the_price_of_a_linear_named_shock_matches_the_closed_form():
    model = _linear_model()
    low, high = np.array([100.6]), np.array([100.8])
    plan = ScenarioPlan(
        horizon=1,
        constraints=(
            LinearConstraint(
                terms=(LinearTerm(variable="ru.cpi", month=1, weight=1.0),),
                low=float(low[0]),
                high=float(high[0]),
            ),
        ),
        causes=("gl.oil",),
    )
    logs = box_probabilities(model, plan, low, high, draws=8, seed=3)
    assert logs is not None and len(logs) == 8

    # The closed form: with no residual, y_1 = 101 + Q(Phi(z)) exactly, Q the pareto
    # law of the size, so the box is an interval of the size law's probability —
    # the measure of the levels whose size quantile lands in the box.
    base = 101.0
    grid = model.levels
    size_table = model.tables[0].size_table
    inside = (size_table >= low[0] - base) & (size_table <= high[0] - base)
    probability = float(np.trapezoid(inside.astype(float), grid))
    closed = -np.log(probability)
    found = float(-np.asarray(logs).mean())
    assert found == pytest.approx(closed, rel=0.25), (found, closed)


def test_the_imposition_resamples_the_samples_inside_the_box():
    model = _linear_model()
    plan = ScenarioPlan(
        horizon=1,
        constraints=(
            LinearConstraint(
                terms=(LinearTerm(variable="ru.cpi", month=1, weight=1.0),),
                low=100.6,
                high=100.8,
            ),
        ),
        causes=("gl.oil",),
        inner=2,
    )
    found = simulate(model, plan, draws=4, seed=5)
    assert found.paths.shape == (8, 2, 1)
    values = found.paths[:, 1, 0]
    assert np.all((values >= 100.6 - 1e-9) & (values <= 100.8 + 1e-9))
    assert set(found.group) == set(range(4))


def test_an_empty_box_contradicts_the_models():
    from mst.model_sdk.dist import ContradictsModels

    model = _linear_model()
    plan = ScenarioPlan(
        horizon=1,
        constraints=(
            LinearConstraint(
                terms=(LinearTerm(variable="ru.cpi", month=1, weight=1.0),),
                low=500.0,
                high=501.0,
            ),
        ),
        causes=("gl.oil",),
    )
    with pytest.raises(ContradictsModels):
        simulate(model, plan, draws=2, seed=7)
    logs = box_probabilities(
        model, plan, np.array([500.0]), np.array([501.0]), draws=2, seed=7
    )
    assert np.all(np.isneginf(logs))


def test_the_expert_model_prices_a_named_shock_without_the_residual(tmp_path_factory):
    from mst.plugins.expert.estimate import estimate, load

    found = estimate(
        fakes.answers_context(tmp_path_factory.mktemp("series")),
        base="rw",
        spec=fakes.quick_spec(),
    )
    model = load(found.meta, found.arrays)
    column = model.ids.index("gl.brent")
    # A box wide enough that the named shock reaches it at most values of R: the
    # residual of R is held, so a narrow box would be unreachable for most groups.
    move = float(model.tables[0].responses[column, 0, 2].mean())
    target = model.base_path[2, column] + move
    plan = ScenarioPlan(
        horizon=2,
        constraints=(
            LinearConstraint(
                terms=(LinearTerm(variable="gl.brent", month=2, weight=1.0),),
                low=target - 0.3,
                high=target + 0.3,
            ),
        ),
        causes=("gl.oil",),
    )
    logs = box_probabilities(
        model,
        plan,
        np.array([target - 0.3]),
        np.array([target + 0.3]),
        draws=4,
        seed=11,
    )
    assert logs is not None
    assert np.any(np.isfinite(logs)), "the shock reaches the box at some value of R"
    assert float(-np.where(np.isfinite(logs), logs, 0.0).mean()) > 0
    # The same seed gives the same numbers.
    again = box_probabilities(
        model,
        plan,
        np.array([target - 0.3]),
        np.array([target + 0.3]),
        draws=4,
        seed=11,
    )
    assert np.array_equal(logs, again)


def test_two_constraints_are_priced_and_imposed_together():
    model = _linear_model()
    bounds = ((1, 100.6, 101.4), (2, 101.0, 104.0))
    plan = ScenarioPlan(
        horizon=2,
        constraints=tuple(
            LinearConstraint(
                terms=(LinearTerm(variable="ru.cpi", month=month, weight=1.0),),
                low=low,
                high=high,
            )
            for month, low, high in bounds
        ),
        causes=("gl.oil",),
    )
    low = np.array([low for _, low, _ in bounds])
    high = np.array([high for _, _, high in bounds])
    logs = box_probabilities(model, plan, low, high, draws=2, seed=3)
    assert logs is not None and np.all(np.isfinite(logs)) and np.all(logs < 0.0)
    found = simulate(model, plan, draws=2, seed=3)
    for month, low, high in bounds:
        values = found.paths[:, month, 0]
        assert np.all((values >= low - 1e-9) & (values <= high + 1e-9))


def test_the_imposition_reports_its_progress(capsys):
    model = _linear_model()
    plan = ScenarioPlan(
        horizon=1,
        constraints=(
            LinearConstraint(
                terms=(LinearTerm(variable="ru.cpi", month=1, weight=1.0),),
                low=100.6,
                high=100.8,
            ),
        ),
        causes=("gl.oil",),
    )
    simulate(model, plan, draws=20, seed=5)
    lines = [line for line in capsys.readouterr().err.splitlines() if "subset" in line]
    assert len(lines) == 10 and lines[-1].endswith("100%")
