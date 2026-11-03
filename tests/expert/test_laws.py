"""The laws built from three quantiles: pareto exact, skew t fitted, tables monotone."""

import numpy as np
from scipy.special import ndtri
from scipy.stats import t as student

from mst.plugins.expert.laws import (
    FIT_P,
    level_grid,
    pareto_quantile,
    skew_t_fit,
    skew_t_quantile,
    table,
)
from mst.plugins.expert.spec import Spec
from tests.expert import fakes

ALPHA = 4.0


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


def test_pareto_passes_through_the_three_quantiles_exactly():
    found = pareto_quantile(-4.0, 0.5, 4.0, np.array(FIT_P), ALPHA)
    assert np.allclose(found, [-4.0, 0.5, 4.0], atol=1e-12)


def test_pareto_density_does_not_jump_at_the_seams():
    # The derivative of the quantile function is the inverse density; the one-sided
    # slopes at 5 and 95 % must agree, which is what the / 9 scaling of the tails buys.
    p = np.array([0.0499, 0.05, 0.0501, 0.9499, 0.95, 0.9501])
    q = pareto_quantile(-4.0, 0.5, 4.0, p, ALPHA)
    left = (q[1] - q[0]) / (p[1] - p[0])
    right = (q[2] - q[1]) / (p[2] - p[1])
    upper_left = (q[4] - q[3]) / (p[4] - p[3])
    upper_right = (q[5] - q[4]) / (p[5] - p[4])
    assert abs(left - right) < 0.05 and abs(upper_left - upper_right) < 0.05
    assert abs(right - (0.5 - -4.0) / 0.45) < 1e-9
    assert abs(upper_left - (4.0 - 0.5) / 0.45) < 1e-9


def test_pareto_tail_follows_the_index():
    p = np.array([0.99, 0.995])
    q = pareto_quantile(-4.0, 0.5, 4.0, p, ALPHA)
    growth = (q[1] - 4.0) / (q[0] - 4.0)
    expected = (10.0 ** (1.0 / ALPHA) - 1.0) / (5.0 ** (1.0 / ALPHA) - 1.0)
    assert np.isclose(growth, expected)


def test_skew_t_reproduces_a_symmetric_answer_exactly():
    loc, scale, shape = 3.0, 2.0, 0.0
    target = loc + scale * student.ppf(np.array(FIT_P), ALPHA)
    found_loc, found_scale, found_shape, miss = skew_t_fit(*target, ALPHA)
    assert miss < 1e-6
    assert np.isclose(found_loc, loc) and np.isclose(found_shape, shape, atol=1e-4)
    assert np.isclose(found_scale, scale, rtol=1e-3)


def test_skew_t_fits_a_skewed_answer_within_one_percent():
    truth = skew_t_quantile(1.0, 2.0, 3.0, ALPHA, np.array(FIT_P))[0]
    found_loc, found_scale, found_shape, miss = skew_t_fit(*truth, ALPHA)
    assert miss < 0.01
    assert np.isclose(found_loc, 1.0, atol=0.05) and np.isclose(found_shape, 3.0, atol=0.1)


def test_skew_t_reports_a_miss_on_an_extreme_skew():
    # A point mass beside a far tail: no smooth skew t passes through these.
    _, _, _, miss = skew_t_fit(-1.0, -0.99, 1.0, ALPHA)
    assert miss > 0.01


def test_a_constant_cell_gives_the_constant_at_every_level():
    levels, values, misses = table(
        np.array([-1.0, 2.0]), np.array([0.0, 2.0]), np.array([1.0, 2.0]), spec()
    )
    assert np.all(values[1] == 2.0)
    assert misses[1] == 0.0


def test_tables_are_monotone_in_the_level():
    cells_q5 = np.array([-4.0, -1.0, 0.0])
    cells_q50 = np.array([0.5, -0.5, 0.0])
    cells_q95 = np.array([4.0, 1.0, 3.0])
    for family in ("pareto", "skew_t"):
        levels, values, misses = table(cells_q5, cells_q50, cells_q95, spec(family=family))
        assert len(levels) == 129 and levels[0] == level_grid(129)[0]
        assert np.all(np.diff(values, axis=1) >= -1e-12), family
        if family == "pareto":
            assert np.allclose(misses, 0.0)
            # The answer is reproduced exactly at its own quantiles.
            assert np.allclose(
                np.interp(FIT_P, levels, values[0]), [-4.0, 0.5, 4.0], atol=1e-9
            )


def test_the_level_grid_is_uniform_in_the_normal_score():
    levels = level_grid(513)
    scores = np.diff(ndtri(levels))
    assert np.allclose(scores, scores[0])
    assert np.isclose(levels[0], 0.0005) and np.isclose(levels[-1], 0.9995)
