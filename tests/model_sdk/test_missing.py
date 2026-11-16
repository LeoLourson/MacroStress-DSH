"""mst.model_sdk.missing: the unknown months drawn from their conditional law."""

import numpy as np
import pytest
from scipy import sparse

from mst.model_sdk.missing import draw_missing


def test_a_month_between_two_known_ones_of_a_walk_is_their_bridge():
    # y_t - y_{t-1} = e_t with unit variance; y_1 is unknown between y_0 = 0 and y_2 = 4.
    equation = sparse.csr_matrix(np.array([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]]))
    precisions = np.ones((2, 1, 1))
    intercept = np.zeros((2, 1))
    y = np.array([[0.0], [np.nan], [4.0]])
    missing = np.isnan(y)
    rng = np.random.default_rng(3)

    draws = np.array(
        [draw_missing(y, missing, equation, precisions, intercept, rng)[1, 0] for _ in range(20000)]
    )

    assert draws.mean() == pytest.approx(2.0, abs=0.02)
    assert draws.var() == pytest.approx(0.5, rel=0.03)
    completed = draw_missing(y, missing, equation, precisions, intercept, rng)
    assert completed[0, 0] == 0.0 and completed[2, 0] == 4.0
