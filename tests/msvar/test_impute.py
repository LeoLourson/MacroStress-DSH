"""The missing-month draw of the MS-VAR against a dense reference of the same equations."""

import numpy as np

from mst.plugins.msvar.estimate import impute


class Quiet:
    """A generator that adds no noise: the draw is the conditional mean itself."""

    def standard_normal(self, size):
        return np.zeros(size)


def dense_mean(y, missing, coefficients, p, variance, a=None):
    """The conditional mean of the unknown cells, built with dense matrices.

    Every month ``t = p .. T-1`` contributes one equation per variable:
    ``y_t - sum_l A_l y_{t-l} = const`` with the error precision of that month.
    """
    months, n = y.shape
    count = months - p
    const = coefficients[:, 0]
    lags = coefficients[:, 1:].reshape(n, p, n)  # lags[:, k] multiplies y_{t-1-k}
    equation = np.zeros((count * n, months * n))
    for t in range(count):
        for lag in range(p + 1):
            month = t + p - lag
            block = np.eye(n) if lag == 0 else -lags[:, lag - 1, :]
            equation[t * n : (t + 1) * n, month * n : (month + 1) * n] = block
    root = np.eye(n) if a is None else a  # e_t = A u_t, so the precision of u_t is A' D^-1 A
    weight = np.kron(np.eye(count), root.T @ np.diag(1 / variance) @ root)
    precision = equation.T @ weight @ equation
    linear = equation.T @ (weight @ np.tile(const, count))
    flat = np.flatnonzero(missing.ravel())
    known = np.where(missing.ravel(), 0.0, y.ravel())
    return np.linalg.solve(precision[np.ix_(flat, flat)], linear[flat] - precision[flat] @ known)


def test_the_unknown_months_come_from_the_var_equations():
    # A ragged tail: the last months of two variables are missing, as in the workspace panel.
    rng = np.random.default_rng(7)
    n, p, months = 2, 2, 24
    y = rng.standard_normal((months, n))
    coefficients = rng.standard_normal((n, 1 + p * n))
    variance = np.array([0.5, 2.0])
    missing = np.zeros((months, n), dtype=bool)
    missing[-1, 0] = True
    missing[-2, 1] = True
    expected = dense_mean(y, missing, coefficients, p, variance)
    holed = y.copy()
    holed[missing] = np.nan
    drawn = impute(
        holed,
        missing,
        states=np.zeros(months - p, dtype=int),
        coefficients=[coefficients],
        a=[np.eye(n)],
        logvar=[np.log(variance)],
        p=p,
        rng=Quiet(),
    )
    assert np.isfinite(drawn).all()
    assert np.allclose(drawn[missing], expected, atol=1e-9)


def test_the_error_precision_is_that_of_the_regime_covariance():
    # e_t = A u_t with A lower triangular: A' D^-1 A and A D^-1 A' differ once A is not diagonal.
    rng = np.random.default_rng(11)
    n, p, months = 3, 1, 20
    y = rng.standard_normal((months, n))
    coefficients = rng.standard_normal((n, 1 + p * n))
    variance = np.array([0.5, 2.0, 1.0])
    a = np.array([[1.0, 0.0, 0.0], [0.8, 1.0, 0.0], [-0.5, 0.3, 1.0]])
    missing = np.zeros((months, n), dtype=bool)
    missing[5, 1] = True
    missing[-1, 0] = True
    expected = dense_mean(y, missing, coefficients, p, variance, a)
    holed = y.copy()
    holed[missing] = np.nan
    drawn = impute(
        holed,
        missing,
        states=np.zeros(months - p, dtype=int),
        coefficients=[coefficients],
        a=[a],
        logvar=[np.log(variance)],
        p=p,
        rng=Quiet(),
    )
    assert np.allclose(drawn[missing], expected, atol=1e-9)
