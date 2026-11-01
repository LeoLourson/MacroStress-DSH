"""The conditions on means and probabilities: the exponential tilt of the causes.

The checks follow the plan: a normal shift against its closed form — the multiplier, the
price and the tilted mean; a probability condition against a numerical solution; the price
of a probability condition approaching the price of the support as ``p`` grows to one; an
inactive condition leaving the sample as it is; and the law of ``R`` surviving the weights.
"""

import numpy as np
import pytest

from mst.scenario.overlay import resample, solve_lambda


def normal_sample(draws: int = 4000, groups: int = 20, seed: int = 1):
    """``(values, group)``: standard normal draws in equal groups."""
    values = np.random.default_rng(seed).standard_normal((draws, 1))
    return values, np.repeat(np.arange(groups), draws // groups)


def test_a_normal_shift_matches_the_closed_form():
    # The tilt e^{lam x} of a standard normal has mean lam and price lam^2 / 2.
    values, group = normal_sample()
    found = solve_lambda(values, group, np.array([1.0]), np.array([np.inf]))
    assert found.lam[0] == pytest.approx(1.0, abs=0.03)
    assert found.price == pytest.approx(0.5, abs=0.03)
    assert found.means[0] == pytest.approx(1.0, abs=0.03)
    assert found.active == (True,)
    # An upper bound alone tilts the other way with the same price.
    found = solve_lambda(values, group, np.array([-np.inf]), np.array([-1.0]))
    assert found.lam[0] == pytest.approx(-1.0, abs=0.03)
    assert found.price == pytest.approx(0.5, abs=0.03)


def test_an_inactive_condition_costs_nothing_and_changes_nothing():
    values, group = normal_sample()
    plain = values.mean()
    found = solve_lambda(
        values, group, np.array([plain - 0.5]), np.array([plain + 0.5])
    )
    assert found.lam[0] == 0.0
    assert found.price == 0.0
    assert np.all(found.weights == found.weights[0])
    assert found.effective_size == pytest.approx(len(values))
    assert found.active == (False,)


def test_a_probability_condition_matches_the_numerical_solution():
    # Q(B) >= p with B an interval: the tilt weights the paths inside B by e^{lam}, and the
    # share becomes a function of lam the test solves by bisection.
    rng = np.random.default_rng(2)
    values = rng.standard_normal(4000)[:, None]
    group = np.repeat(np.arange(20), 200)
    inside = (values[:, 0] > 0.2) & (values[:, 0] < 1.4)
    values = inside.astype(float)[:, None]
    target = 0.8

    def share_of(lam: float) -> float:
        weights = np.exp(lam * values[:, 0])
        total = 0.0
        for g in range(20):
            rows = group == g
            total += weights[rows][inside[rows]].sum() / weights[rows].sum()
        return total / 20

    low, high = 0.0, 20.0
    for _ in range(60):
        middle = (low + high) / 2
        if share_of(middle) < target:
            low = middle
        else:
            high = middle
    numerical = (low + high) / 2

    found = solve_lambda(values, group, np.array([target]), np.array([np.inf]))
    assert found.lam[0] == pytest.approx(numerical, abs=0.05)
    assert found.means[0] == pytest.approx(target, abs=0.01)
    # The indicator tilt concentrates the weight on the paths inside B without ever
    # falling below their count: the effective size sits between it and the whole sample.
    assert inside.sum() <= found.effective_size <= len(values)


def test_the_price_of_a_probability_condition_reaches_the_price_of_the_support():
    # As p grows to one the tilt concentrates on the paths inside B, and its price
    # approaches the price of imposing B as a support condition: -mean_g log(share_g).
    rng = np.random.default_rng(3)
    base = rng.standard_normal(4000)
    group = np.repeat(np.arange(20), 200)
    inside = (base > 0.5) & (base < 2.5)
    values = inside.astype(float)[:, None]
    support = np.mean(
        [
            -np.log(inside[group == g].mean())
            for g in range(20)
        ]
    )
    prices = []
    for target in (0.95, 0.995, 0.9995):
        found = solve_lambda(values, group, np.array([target]), np.array([np.inf]))
        prices.append(found.price)
    assert prices[0] < prices[1] < prices[2]
    assert prices[2] == pytest.approx(support, rel=0.02)
    # A condition the plain sample already meets costs nothing.
    easy = solve_lambda(values, group, np.array([inside.mean() - 0.01]), np.array([np.inf]))
    assert easy.price == 0.0


def test_the_law_of_r_survives_the_weights():
    # The resampling runs inside the groups: every group keeps its size and receives only
    # its own paths, so the law of R — one draw per group, equal weights — is untouched.
    rng = np.random.default_rng(4)
    values = rng.standard_normal((40, 1)) + np.repeat(np.arange(4), 10)[:, None]
    group = np.repeat(np.arange(4), 10)
    found = solve_lambda(values, group, np.array([2.0]), np.array([np.inf]))
    picked = resample(found.weights, group, np.random.default_rng(5))
    assert (np.bincount(group[picked]) == 10).all()
    for g in range(4):
        assert set(picked[group == g]) <= set(np.flatnonzero(group == g))
    # Weight one everywhere returns the paths in their order.
    identity = resample(np.full(40, 1.0), group, np.random.default_rng(5))
    assert np.array_equal(identity, np.arange(40))


def test_an_unreachable_condition_is_named_infeasible():
    values, group = normal_sample(draws=400, groups=4, seed=6)
    beyond = solve_lambda(
        values, group, np.array([values.max() + 5.0]), np.array([np.inf])
    )
    assert beyond.feasible is False
    assert beyond.price == float("inf")
