"""The shock stream: occurrences, sizes, copula levels and the response profiles."""

import numpy as np
from scipy.special import ndtr, ndtri

from mst.plugins.expert.laws import level_grid, pareto_quantile, table
from mst.plugins.expert.spec import Spec
from mst.plugins.expert.stream import (
    ShockTables,
    contributions,
    draw,
    levels,
    occurrences,
)
from tests.expert import fakes

ALPHA = 4.0
COLUMNS = 2 + 2 + 1  # occurrence, size, two variables, the chi-square scale


def spec(**changes):
    values = {
        "family": "pareto",
        "alpha": ALPHA,
        "levels": 129,
        "stream_draws": 2000,
        "candidates": 8,
        "subset_samples": 500,
        "history_start": fakes.START,
        "seed": 3,
    }
    values.update(changes)
    return Spec.model_validate(values)


def tables(specification=None, frequency=1.2, rho=0.5, sizes=(-4.0, 0.5, 4.0), cells=None):
    """The tables of one two-variable shock from crafted cell quantiles."""
    specification = specification or spec()
    grid = level_grid(specification.levels)
    size_table = pareto_quantile(*sizes, grid, specification.alpha)
    if cells is None:
        # variable 0: positive medians; variable 1: negative medians.
        cells = [
            [(0.0, 1.0, 2.0) if variable == 0 else (-2.0, -1.0, -0.2)] * 4 * 3
            for variable in (0, 1)
        ]
    q5 = np.array([[cell[0] for cell in variable] for variable in cells])
    q50 = np.array([[cell[1] for cell in variable] for variable in cells])
    q95 = np.array([[cell[2] for cell in variable] for variable in cells])
    _, values, _ = table(q5.ravel(), q50.ravel(), q95.ravel(), specification)
    responses = values.reshape(2, 4, 3, len(grid))
    medians = q50.reshape(2, 4, 3)
    return ShockTables(
        frequency=frequency,
        rho=rho,
        sizes=np.array(sizes, dtype=float),
        size_table=size_table,
        responses=responses,
        medians=medians,
        levels=grid,
        alpha=specification.alpha,
    )


def forced(size_level=0.5, copula=0.0, occur=True, months=3):
    """Inputs whose first month occurs at the pinned size level, the rest stay quiet.

    The occurrence is ``ndtr(z0) < rate`` with a small rate, so a very negative ``z0``
    occurs and a very positive one stays quiet.
    """
    inputs = np.zeros((1, months, COLUMNS))
    inputs[..., 0] = 8.0
    inputs[:, 0, 0] = -8.0 if occur else 8.0
    inputs[:, 0, 1] = ndtri(size_level)
    inputs[:, 0, 2:4] = copula
    inputs[..., 4] = 0.0
    return inputs


def test_the_share_of_months_with_an_occurrence_matches_the_rate():
    model = tables(frequency=2.4)
    inputs = draw(model, np.random.default_rng(1), 1, 40000)
    rate = 1.0 - np.exp(-2.4 / 12.0)
    share = occurrences(model, inputs).mean()
    error = 3 * np.sqrt(rate * (1 - rate) / inputs.shape[0])
    assert abs(share - rate) < error


def test_zero_frequency_gives_zero_contributions():
    model = tables(frequency=0.0)
    inputs = draw(model, np.random.default_rng(2), 6, 500)
    found = contributions(model, inputs, 6)
    assert found.shape == (500, 7, 2)
    assert not found.any()


def test_at_unit_rho_every_variable_takes_one_level():
    inputs = np.random.default_rng(3).standard_normal((50, 4, COLUMNS))
    found = levels(inputs, 1.0, ALPHA)
    assert np.allclose(found, found[..., :1]), "all variables share the level"


def test_a_negative_median_moves_away_from_zero_when_the_level_rises():
    model = tables()
    high = contributions(model, forced(copula=2.0), 3)
    low = contributions(model, forced(copula=-2.0), 3)
    # Variable 1 has negative medians: a higher level pushes it further below zero.
    assert np.all(high[:, 1:, 1] < low[:, 1:, 1])
    assert np.all(low[:, 1:, 1] < 0.0)


def test_a_size_beyond_the_last_node_keeps_its_response():
    model = tables(sizes=(0.5, 2.0, 5.0))
    at_the_node = contributions(model, forced(size_level=0.95, months=1), 1)
    beyond = contributions(model, forced(size_level=0.9995, months=1), 1)
    assert np.allclose(at_the_node[:, 1], beyond[:, 1])


def test_a_size_of_the_opposite_sign_gives_no_response():
    model = tables(sizes=(0.5, 2.0, 5.0))
    found = contributions(model, forced(size_level=0.001, months=1), 1)
    assert np.allclose(found[:, 1], 0.0)


def test_the_first_month_between_nodes_is_the_mean_of_its_neighbours():
    model = tables()
    found = contributions(model, forced(), 3)
    assert np.allclose(found[:, 2], 0.5 * (found[:, 1] + found[:, 3]))


def test_the_same_seed_gives_the_same_contributions_bit_for_bit():
    model = tables()
    first = contributions(model, draw(model, np.random.default_rng(7), 6, 200), 6)
    second = contributions(model, draw(model, np.random.default_rng(7), 6, 200), 6)
    assert np.array_equal(first, second)


def test_the_levels_stay_inside_the_table_and_the_size_matches_its_law():
    model = tables()
    inputs = draw(model, np.random.default_rng(11), 3, 2000)
    found = levels(inputs, 0.5, ALPHA)
    assert found.min() >= 0.0 and found.max() <= 1.0
    share = occurrences(model, inputs).mean()
    rate = 1.0 - np.exp(-1.2 / 12.0)
    assert abs(share - rate) < 4 * np.sqrt(rate * (1 - rate) / inputs[..., 0].size)
    assert np.isfinite(contributions(model, inputs, 3)).all()
    assert 0.0 < ndtr(0.0) < 1.0
