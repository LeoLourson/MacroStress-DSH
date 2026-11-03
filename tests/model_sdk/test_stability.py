"""The spectral radius of the companion matrix on matrices with known roots."""

import numpy as np
import pytest

from mst.model_sdk.stability import companion_radius


def test_a_first_order_var_radius_is_its_own_lag():
    assert companion_radius(np.array([[[0.5]]])) == pytest.approx(0.5)
    assert companion_radius(np.array([[[1.0]]])) == pytest.approx(1.0)  # a unit root
    assert companion_radius(np.array([[[1.05]]])) == pytest.approx(1.05)  # explosive


def test_a_diagonal_lag_has_the_largest_own_root():
    lags = np.array([[0.3, 0.0], [0.0, 0.8]])[None]  # p = 1, n = 2
    assert companion_radius(lags) == pytest.approx(0.8)


def test_a_second_order_scalar_var_has_both_roots():
    # z^2 - 1.1 z + 0.3 = (z - 0.6)(z - 0.5).
    lags = np.array([[[1.1]], [[-0.3]]])  # p = 2, n = 1
    assert companion_radius(lags) == pytest.approx(0.6)


def test_complex_roots_enter_by_their_modulus():
    # z^2 - 0.8 z + 0.8 has the roots 0.4 +- 0.8i, both of modulus sqrt(0.8).
    lags = np.array([[[0.8]], [[-0.8]]])
    assert companion_radius(lags) == pytest.approx(np.sqrt(0.8))


def test_a_third_order_var_places_the_identities():
    # Roots of z^3 = 0.5 z^2 + 0.1 z + 0.2: the largest is near 0.69, all inside the circle.
    lags = np.array([[[0.5]], [[0.1]], [[0.2]]])
    radius = companion_radius(lags)
    roots = np.roots([1.0, -0.5, -0.1, -0.2])
    assert radius == pytest.approx(np.abs(roots).max())


def test_leading_indices_are_draws_and_regimes():
    stable = np.zeros((1, 2, 2))
    explosive = 1.02 * np.eye(2)[None]
    batch = np.stack([stable, explosive, stable])  # (3, p, n, n)
    found = companion_radius(batch)
    assert found.shape == (3,)
    assert found[0] == pytest.approx(0.0)
    assert found[1] == pytest.approx(1.02)
    # A batch over regimes keeps its shape: (draws, regimes).
    per_regime = np.stack([batch, batch], axis=1)  # (3, 2, p, n, n)
    assert companion_radius(per_regime).shape == (3, 2)


def test_an_explosive_draw_stands_out_in_a_batch():
    draws = np.stack([0.5 * np.eye(2), 1.2 * np.eye(2), 0.9 * np.eye(2)])[:, None]
    radius = companion_radius(draws)
    assert np.all(radius <= np.array([0.5, 1.2, 0.9]) + 1e-12)
    assert np.argmax(radius) == 1


def test_a_wrong_shape_is_refused():
    with pytest.raises(ValueError, match=r"\(\.\.\., p, n, n\)"):
        companion_radius(np.eye(2))
    with pytest.raises(ValueError, match="square lag matrices"):
        companion_radius(np.ones((1, 2, 3)))
