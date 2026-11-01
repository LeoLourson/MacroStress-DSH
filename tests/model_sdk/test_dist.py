"""The price of a scenario ``dist(s) = E_P[-log P(S | R)]`` on gaussian laws.

The checks follow the plan: the GHK sampler against the univariate cdf and against the exact
probability of independent components, with a finite answer in the far tail; the kernel
against closed forms — the average over ``theta`` at ``R = theta``, the conditional law at a
named cause, the Jensen inequality ``dist >= -log P(S)``, the price of a finer ``R'`` that
determines ``R``; the box moved by an impact; and the ``-k log(2 tau)`` growth of a point
condition as ``tau`` shrinks.
"""

import numpy as np
import pytest
from scipy.stats import norm

from mst.model_sdk.conditional import log_box_probabilities, log_box_probability
from mst.model_sdk.contracts import ConstraintLaw
from mst.model_sdk.dist import price_of_law


def law(means, covs, log_weight=None, group=None) -> ConstraintLaw:
    """A law of ``K`` groups, one row each, unless the weights and groups say otherwise."""
    means = np.asarray(means, dtype=float)
    if means.size == 0:
        means = means.reshape(len(means) if hasattr(means, "__len__") else 0, 0)
        rows = len(means)
        m = 0
    else:
        means = means.reshape(len(means), -1)
        rows, m = means.shape
    covs = np.asarray(covs, dtype=float).reshape(rows, m, m)
    if log_weight is None:
        log_weight = np.zeros(rows)
    if group is None:
        group = np.arange(rows)
    return ConstraintLaw(
        means=means,
        covs=covs,
        log_weight=np.asarray(log_weight, dtype=float),
        group=np.asarray(group, dtype=int),
    )


def price(found: ConstraintLaw, low, high, *, samples=4000, seed=1):
    return price_of_law(
        found, np.asarray(low, dtype=float), np.asarray(high, dtype=float),
        rng=np.random.default_rng(seed), samples=samples,
    )


class TestGHK:
    def test_one_dimension_matches_the_univariate_cdf(self):
        rng = np.random.default_rng(5)
        for mean, variance, low, high in (
            (1.0, 4.0, 0.5, 2.5),
            (-3.0, 0.25, -3.2, -2.6),
            (0.0, 1.0, -1.0, 1.0),
        ):
            found = log_box_probability([mean], [[variance]], [low], [high], rng, 7)
            exact = np.log(
                norm.cdf((high - mean) ** 0.5 * 0 + (high - mean) / np.sqrt(variance))
                - norm.cdf((low - mean) / np.sqrt(variance))
            )
            assert found == pytest.approx(float(exact), abs=1e-12)

    def test_independent_components_match_the_product_of_marginals(self):
        rng = np.random.default_rng(6)
        mean, cov = [0.5, -1.0], np.diag([1.0, 4.0])
        low, high = np.array([-1.0, -3.0]), np.array([2.0, 0.0])
        exact = (norm.cdf(2.0, 0.5, 1) - norm.cdf(-1.0, 0.5, 1)) * (
            norm.cdf(0.0, -1.0, 2) - norm.cdf(-3.0, -1.0, 2)
        )
        found = np.exp(log_box_probability(mean, cov, low, high, rng, 4000))
        assert found == pytest.approx(exact, rel=0.01)

    def test_the_far_tail_stays_finite(self):
        rng = np.random.default_rng(7)
        found = log_box_probability([0.0], [[1.0]], [8.0], [9.0], rng, 2000)
        assert np.isfinite(found)
        assert found == pytest.approx(np.log(norm.cdf(9.0) - norm.cdf(8.0)), rel=0.05)

    def test_a_batch_matches_the_single_rows(self):
        rng = np.random.default_rng(8)
        means = np.array([[0.0, 0.0], [0.5, -1.0], [3.0, 3.0]])
        covs = np.array(
            [
                [[1.0, 0.6], [0.6, 2.0]],
                np.diag([1.0, 4.0]),
                [[2.0, -0.3], [-0.3, 1.0]],
            ]
        )
        low, high = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
        batch = log_box_probabilities(means, covs, low, high, rng, 3000)
        for k in range(len(means)):
            one = log_box_probability(means[k], covs[k], low, high, rng, 3000)
            assert batch[k] == pytest.approx(float(one), abs=0.02)

    def test_a_coordinate_the_covariance_does_not_move(self):
        rng = np.random.default_rng(9)
        # Inside the box the coordinate costs nothing; outside it the box has no probability.
        inside = log_box_probability(
            [1.0, 5.0], [[1.0, 0.0], [0.0, 0.0]], [0.0, 4.0], [2.0, 6.0], rng, 8
        )
        assert np.isfinite(inside)
        outside = log_box_probability(
            [1.0, 5.0], [[1.0, 0.0], [0.0, 0.0]], [0.0, 6.0], [2.0, 7.0], rng, 8
        )
        assert outside == float("-inf")


class TestPriceOfLaw:
    def test_the_price_at_r_theta_is_the_average_over_theta(self):
        # Two parameter sets with different means; the closed form is the mean of
        # -log P_theta(S) over the two, not -log of the mean.
        means = [[0.0], [4.0]]
        covs = [[[1.0]], [[1.0]]]
        low, high = np.array([1.5]), np.array([2.5])
        distance, log_probability, _ = price(law(means, covs), low, high)
        per_theta = [
            -np.log(norm.cdf(2.5, mu, 1) - norm.cdf(1.5, mu, 1)) for mu in (0.0, 4.0)
        ]
        assert distance == pytest.approx(float(np.mean(per_theta)), abs=0.02)
        assert distance >= -log_probability == pytest.approx(
            -np.log(np.mean(np.exp(-np.array(per_theta)))), abs=0.02
        )

    def test_a_named_cause_prices_the_conditional_law(self):
        # W y = c(theta) + b(theta) eps_D + a(theta) eps_free with eps_D the one named cause:
        # given R = (theta, eps_free) the functional is normal around c + a eps_free with
        # variance b^2. The price is the mean of -log of that probability over both draws.
        rng = np.random.default_rng(11)
        c, a, b = 1.0, 0.5, 0.3
        draws = 400
        free = rng.standard_normal(draws)
        theta = rng.integers(2, size=draws)
        means = [[c + a * f + 2.0 * t] for f, t in zip(free, theta, strict=True)]
        covs = [[[b * b]] for _ in range(draws)]
        low, high = np.array([2.0]), np.array([2.6])
        distance, _, _ = price(law(means, covs), low, high, samples=2000)
        closed = np.mean(
            [
                -np.log(norm.cdf(2.6, m[0], b) - norm.cdf(2.0, m[0], b))
                for m in means
            ]
        )
        assert distance == pytest.approx(float(closed), abs=0.05)

    def test_the_price_is_at_least_minus_log_p_of_the_event(self):
        # Jensen: E[-log P(S|R)] >= -log E[P(S|R)], and the kernel reports both.
        rng = np.random.default_rng(12)
        rows = 300
        means = rng.normal(0, 2, size=(rows, 1))
        covs = [[[1.0]] for _ in range(rows)]
        low, high = np.array([1.0]), np.array([2.0])
        distance, log_probability, _ = price(law(means, covs), low, high)
        assert distance >= -log_probability + 1e-6

    def test_a_finer_r_that_determines_r_costs_no_less(self):
        # The same rows grouped by theta (the coarse R) and by (theta, eps_free) (the finer
        # R' that determines R): conditioning on more coordinates can only raise the price.
        rng = np.random.default_rng(13)
        thetas = np.repeat([0.0, 4.0], 40)
        free = rng.normal(0, 0.5, size=len(thetas))
        means = np.array([[mu + f] for mu, f in zip(thetas, free, strict=True)])
        covs = [[[0.6]] for _ in range(len(thetas))]
        low, high = np.array([1.4]), np.array([2.4])
        coarse = law(means, covs, group=np.repeat(np.arange(2), 40))
        fine = law(means, covs)  # every row its own group
        distance_coarse, _, _ = price(coarse, low, high)
        distance_fine, _, _ = price(fine, low, high)
        assert distance_fine >= distance_coarse - 1e-6

    def test_a_group_of_one_weighted_row_carries_the_weight(self):
        # The regime event: one row per group, weight pi, no constraint functionals at all —
        # the price is the mean of -log pi, the canon's E_P[-log pi_theta(k,..,k)].
        log_pi = np.log(np.array([0.1, 0.2, 0.05, 0.3]))
        found = law(np.zeros((4, 0)), np.zeros((4, 0, 0)), log_weight=log_pi)
        distance, log_probability, _ = price(found, np.zeros(0), np.zeros(0))
        assert distance == pytest.approx(float(-log_pi.mean()))
        assert log_probability == pytest.approx(float(np.log(np.exp(log_pi).mean())))

    def test_a_group_without_probability_contradicts_the_models(self):
        # One group whose box is unreachable — the functional is fixed at 10, the box is
        # [0, 1] — so the distance is infinite and log P(S) is -inf.
        means = [[0.0], [10.0]]
        covs = [[[1.0]], [[0.0]]]
        distance, log_probability, _ = price(
            law(means, covs), np.array([0.0]), np.array([1.0])
        )
        assert distance == float("inf")
        assert log_probability == float("-inf")

    def test_a_box_moved_by_an_impact_changes_the_price(self):
        # An impact on the constrained variable shifts the box the plan carries; the same law
        # priced against the shifted box is the price of the scenario with the impact.
        means = [[2.0]] * 50
        covs = [[[1.0]] * 1] * 50
        plain, _, _ = price(law(means, covs), np.array([0.5]), np.array([1.5]))
        shifted, _, _ = price(law(means, covs), np.array([-0.5]), np.array([0.5]))
        assert shifted > plain + 0.1  # the box moved away from the means

    def test_a_point_condition_costs_minus_log_of_two_tau_per_condition(self):
        # As tau -> 0 the price of k point conditions grows as -k log(2 tau): adding
        # k log(2 tau) back leaves the same constant, one or two conditions alike.
        full_means = np.array([[1.0, -1.0]])
        full_covs = np.array([np.diag([1.0, 4.0])])
        for conditions in (1, 2):
            means = full_means[:, :conditions]
            covs = full_covs[:, :conditions, :conditions]
            prices = []
            for exponent in (6, 7, 8):
                tau = 2.0 ** (-exponent)
                low = np.full(conditions, -tau)
                high = np.full(conditions, tau)
                distance, _, _ = price(
                    law(means, covs), low, high, samples=8000, seed=3
                )
                prices.append(distance + conditions * np.log(2 * tau))
            spread = max(prices) - min(prices)
            assert spread < 0.1 * conditions, (conditions, prices)

    def test_the_standard_error_reads_the_spread_over_groups(self):
        means = [[0.0], [4.0]]
        covs = [[[1.0]], [[1.0]]]
        _, _, standard_error = price(
            law(means, covs), np.array([1.5]), np.array([2.5])
        )
        per_theta = np.array(
            [-np.log(norm.cdf(2.5, mu, 1) - norm.cdf(1.5, mu, 1)) for mu in (0.0, 4.0)]
        )
        assert standard_error == pytest.approx(
            float(per_theta.std(ddof=1) / np.sqrt(2)), abs=0.05
        )

    def test_a_law_without_groups_is_refused(self):
        with pytest.raises(ValueError, match="has no price"):
            price(law([], []), np.array([0.0]), np.array([1.0]))
