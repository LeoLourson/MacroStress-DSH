"""Subset simulation: a small probability level by level, and the price it carries."""

import numpy as np
from scipy.special import ndtri

from mst.plugins.expert.laws import level_grid, pareto_quantile
from mst.plugins.expert.stream import ShockTables
from mst.plugins.expert.subset import subset


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


