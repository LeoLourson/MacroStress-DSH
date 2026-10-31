"""The platform distance ``dist(P_s, P) = -log P(C)`` on gaussian boxes and mixtures."""

import numpy as np
import pytest
from scipy.stats import norm

from mst.model_sdk.contracts import LinearConstraint, LinearTerm
from mst.model_sdk.dist import (
    box_probability_mc,
    box_probability_mixture,
    constraint_matrix,
    dist_from_probability,
    dist_mc,
    dist_mixture,
    satisfied,
)


def one_term(variable="q", month=1, weight=1.0, low=0.0, high=1.0):
    return LinearConstraint(
        terms=(LinearTerm(variable=variable, month=month, weight=weight),),
        low=low,
        high=high,
    )


def test_one_constraint_one_component_matches_the_univariate_analytic():
    # w'y ~ N(mu, sigma): P(low <= w'y <= high) = Phi((high-mu)/sigma) - Phi((low-mu)/sigma).
    means = np.array([[2.0]])
    covs = np.array([[[4.0]]])  # sigma = 2
    constraint = one_term(low=1.0, high=3.0)
    analytic = float(norm.cdf((3.0 - 2.0) / 2.0) - norm.cdf((1.0 - 2.0) / 2.0))
    assert box_probability_mixture(means, covs, (constraint,)) == pytest.approx(analytic)
    assert dist_mixture(means, covs, (constraint,)) == pytest.approx(-np.log(analytic))


def test_correlated_constraints_are_not_the_product_of_the_marginals():
    # Two heavily correlated functionals: the joint box probability differs from the product.
    covs = np.array([[[1.0, 0.8], [0.8, 1.0]]])
    means = np.array([[0.0, 0.0]])
    constraints = (
        one_term(variable="a", low=-0.5, high=0.5),
        one_term(variable="b", low=-0.5, high=0.5),
    )
    joint = box_probability_mixture(means, covs, constraints)
    marginal = norm.cdf(0.5) - norm.cdf(-0.5)
    assert joint != pytest.approx(marginal**2, rel=1e-3)
    # The same through the path-level indicator: the joint share of satisfied() is not the
    # product of the per-constraint shares.
    rng = np.random.default_rng(7)
    paths = rng.multivariate_normal(np.zeros(2), covs[0], size=40000)
    base_paths = paths[:, None, :]  # (R, 1 month? no: (R, H+1, n)) -- month axis is dummy
    ids = ["a", "b"]
    constraints = (
        LinearConstraint(
            terms=(LinearTerm(variable="a", month=0, weight=1.0),), low=-0.5, high=0.5
        ),
        LinearConstraint(
            terms=(LinearTerm(variable="b", month=0, weight=1.0),), low=-0.5, high=0.5
        ),
    )
    joint_mc = float(np.mean(satisfied(base_paths, ids, constraints)))
    a_share = float(np.mean((paths[:, 0] >= -0.5) & (paths[:, 0] <= 0.5)))
    b_share = float(np.mean((paths[:, 1] >= -0.5) & (paths[:, 1] <= 0.5)))
    assert joint_mc != pytest.approx(a_share * b_share, rel=1e-3)
    assert box_probability_mc(base_paths, ids, constraints) == pytest.approx(joint_mc)


def test_mc_and_mixture_agree_within_the_mc_error():
    rng = np.random.default_rng(11)
    draws = rng.multivariate_normal(np.zeros(2), [[1.0, 0.5], [0.5, 1.0]], size=40000)
    base_paths = draws[:, None, :]
    ids = ["a", "b"]
    constraints = (
        LinearConstraint(
            terms=(LinearTerm(variable="a", month=0, weight=1.0),), low=-1.0, high=0.5
        ),
        LinearConstraint(
            terms=(LinearTerm(variable="b", month=0, weight=1.0),), low=-2.0, high=1.0
        ),
    )
    means = np.zeros((1, 2))
    covs = np.array([[[1.0, 0.5], [0.5, 1.0]]])
    exact_pc = box_probability_mixture(means, covs, constraints)
    mc_pc = box_probability_mc(base_paths, ids, constraints)
    assert mc_pc == pytest.approx(exact_pc, abs=0.01)
    assert dist_mc(base_paths, ids, constraints) == pytest.approx(
        dist_mixture(means, covs, constraints), abs=0.05
    )


def test_satisfied_demands_every_constraint_at_once():
    rng = np.random.default_rng(3)
    base_paths = rng.standard_normal((10, 2, 2))
    first = LinearConstraint(
        terms=(LinearTerm(variable="a", month=1, weight=1.0),), low=-1.0, high=1.0
    )
    second = LinearConstraint(
        terms=(LinearTerm(variable="b", month=1, weight=1.0),), low=-0.01, high=0.01
    )
    flag = satisfied(base_paths, ["a", "b"], (first, second))
    manual = (
        (base_paths[:, 1, 0] >= -1.0)
        & (base_paths[:, 1, 0] <= 1.0)
        & (np.abs(base_paths[:, 1, 1]) <= 0.01)
    )
    assert np.array_equal(flag, manual)


def test_the_weights_of_a_constraint_are_summed_over_terms():
    constraints = (
        LinearConstraint(
            terms=(
                LinearTerm(variable="a", month=1, weight=0.5),
                LinearTerm(variable="a", month=1, weight=0.25),
            ),
            low=0.0,
            high=10.0,
        ),
    )
    rows = constraint_matrix(constraints, ["a", "b"])
    assert rows[0][0] == pytest.approx(0.75)
    assert rows[0][1] == 0.0


def test_a_constraint_across_months_keeps_the_mc_estimate():
    across = LinearConstraint(
        terms=(
            LinearTerm(variable="a", month=1, weight=1.0),
            LinearTerm(variable="b", month=2, weight=1.0),
        ),
        low=0.0,
        high=5.0,
    )
    with pytest.raises(ValueError, match="one month per constraint"):
        constraint_matrix((across,), ["a", "b"])


def test_dist_from_probability():
    assert dist_from_probability(1.0) == 0.0
    assert dist_from_probability(0.0) == float("inf")
    assert dist_from_probability(0.5) == pytest.approx(np.log(2.0))


def test_an_empty_constraint_list_has_no_dist():
    with pytest.raises(ValueError, match="no dist"):
        box_probability_mc(np.zeros((4, 2, 2)), ["a", "b"], ())


def test_a_zero_joint_probability_is_an_infinite_dist():
    # The box lies entirely outside the component's support in the limit of a tiny sd.
    means = np.array([[10.0]])
    covs = np.array([[[1e-12]]])
    assert dist_mixture(means, covs, (one_term(low=0.0, high=1.0),)) == float("inf")
