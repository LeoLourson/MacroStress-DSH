"""The unknown months of a history, drawn given the parameters of a linear Gaussian model.

A VAR with known parameters makes the whole history Gaussian: every equation month says
``K vec(y) - c ~ N(0, W^-1)`` with a sparse ``K`` (the identity and the lag blocks) and a
block-diagonal precision ``W``. The unknown entries ``m`` are then normal given the known ones
``o``, with precision ``P_mm`` and mean ``P_mm^-1 (K' W c - P_mo y_o)``, ``P = K' W K``.

Only the rows of ``P`` at the unknown entries are formed: ``K_m' W K``, where ``K_m`` are the
columns of ``K`` at them. A history of 177 months and 23 variables has about 4000 entries and
a hundred unknown ones, so the full ``P`` would cost forty times the rows a sampler step needs.
"""

import numpy as np
from scipy import sparse
from scipy.linalg import cho_solve, solve_triangular


def draw_missing(
    y: np.ndarray,
    missing: np.ndarray,
    equation: sparse.spmatrix,
    precisions: np.ndarray,
    intercept: np.ndarray,
    rng,
) -> np.ndarray:
    """``y`` with its ``missing`` entries drawn from their conditional normal law.

    ``equation`` is ``K`` of shape ``(E * n, T * n)`` over the flattened ``y`` (month-major),
    ``precisions`` the ``(E, n, n)`` error precisions of the equation months and ``intercept``
    the ``(E, n)`` constants. Returns a completed copy; the known entries stay as they are.
    """
    flat = np.flatnonzero(missing.ravel())
    known = np.nan_to_num(y.ravel())
    known[flat] = 0.0
    weight = sparse.block_diag(list(precisions), format="csr")
    of_missing = sparse.csc_matrix(equation)[:, flat]
    weighted = (of_missing.T @ weight).tocsr()
    rows = (weighted @ equation).tocsr()
    rhs = weighted @ np.asarray(intercept).ravel() - rows @ known
    root = np.linalg.cholesky(rows[:, flat].toarray())
    mean = cho_solve((root, True), rhs)
    completed = known.copy()
    completed[flat] = mean + solve_triangular(root.T, rng.standard_normal(len(flat)), lower=False)
    return completed.reshape(y.shape)
