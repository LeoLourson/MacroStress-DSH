"""The imposition for a model given by a sample of paths.

The checks follow the canon's section 8: a path meets the event when it meets every
condition at once, the survivors carry the run with equal weights, the price of one support
condition is ``-log(|S| / n)``, and a sample nothing survives prices at infinity.
"""

import numpy as np
import pytest

from mst.model_sdk.contracts import LinearConstraint, LinearTerm
from mst.model_sdk.overlay import resample, reweight


def paths_of(seed: int = 1, n: int = 400) -> np.ndarray:
    """``(n, 3, 1)``: normal draws on two horizon months, one variable."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 3, 1))


def condition(low: float, high: float, month: int = 2, variable: str = "x") -> LinearConstraint:
    return LinearConstraint(
        terms=(LinearTerm(variable=variable, month=month, weight=1.0),),
        low=low,
        high=high,
    )


def test_the_price_of_one_support_condition_is_the_log_of_the_share():
    paths = paths_of()
    found = reweight(paths, ["x"], (condition(-1.0, 1.0),))
    share = float(np.mean((paths[:, 2, 0] >= -1.0) & (paths[:, 2, 0] <= 1.0)))
    assert found.survivors == int(share * len(paths))
    assert found.price == pytest.approx(-np.log(share), abs=1e-12)
    assert found.effective_size == found.survivors
    assert np.all(found.weights[paths[:, 2, 0] > 1.0] == 0.0)


def test_two_conditions_count_the_joint_share_not_the_product():
    paths = paths_of()
    both = reweight(
        paths, ["x"], (condition(-1.0, 1.0), condition(-0.5, 0.5, month=1))
    )
    inside = (
        (paths[:, 2, 0] >= -1.0)
        & (paths[:, 2, 0] <= 1.0)
        & (paths[:, 1, 0] >= -0.5)
        & (paths[:, 1, 0] <= 0.5)
    )
    assert both.survivors == int(inside.sum())


def test_a_sample_nothing_survives_prices_at_infinity():
    paths = paths_of()
    found = reweight(paths, ["x"], (condition(50.0, 51.0),))
    assert found.survivors == 0
    assert found.price == float("inf")
    assert found.effective_size == 0.0


def test_the_resampling_keeps_the_sample_and_its_law():
    paths = paths_of()
    found = reweight(paths, ["x"], (condition(-1.0, 1.0),))
    picked = resample(found.weights, np.random.default_rng(2))
    assert len(picked) == len(paths)
    # Every picked path meets the condition: the weights pick only the survivors.
    assert np.all((paths[picked, 2, 0] >= -1.0) & (paths[picked, 2, 0] <= 1.0))
    # Weight one everywhere returns the paths in their order.
    identity = resample(np.full(len(paths), 1.0), np.random.default_rng(3))
    assert np.array_equal(identity, np.arange(len(paths)))


def test_a_condition_naming_a_missing_variable_is_refused():
    paths = paths_of()
    with pytest.raises(ValueError, match="the paths do not carry"):
        reweight(paths, ["y"], (condition(0.0, 1.0),))
