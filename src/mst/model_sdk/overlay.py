"""The imposition for a model given by a sample of paths.

The canon's section 8 closes with the model whose ``Omega`` is the sample itself:
``Omega = {1, .., n}`` with ``P`` uniform and ``F(i) = x_i``, so ``R`` is constant and the
theorem hands every path the weight ``1_S(i) e^{lambda' f(i)}``. This module carries the
support half of that: ``reweight`` marks the paths that meet the conditions, weighs the
sample by the marks and prices the imposition at ``-log(|S| / n)`` — one condition on the
support costs exactly the logarithm of the surviving share. The moment half — the tilt by
``lambda`` — is the platform's overlay (``mst.scenario.overlay``): a model plugin never
sees the scenario's moment conditions.

``resample`` turns the weights into a plain sample again, so the consumers of the paths
read equal weights; the template of a macro model (``templates/macro/model.py.tmpl``)
inherits exactly this behaviour when the scenario names no cause.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Reweight:
    """The weighed sample of a path-sample model."""

    weights: np.ndarray
    """``(n,)`` the weight of every path: one inside the event, zero outside."""
    price: float
    """``-log(|S| / n)``: the price of the support conditions; infinite when nothing survives."""
    survivors: int
    """How many paths meet every condition at once."""
    effective_size: float
    """``(sum w)^2 / sum w^2``: the survivors carry the whole weight."""


def reweight(paths: np.ndarray, ids, constraints) -> Reweight:
    """The weights and the price of the support conditions on a sample of paths.

    ``paths`` is ``(n, H + 1, k)`` in the model scale; ``constraints`` are linear conditions
    over the paths' variables and months, the same shape ``mst.model_sdk.contracts`` carries.
    A path meets the event when it meets every condition at once — the joint share, not the
    product of the per-condition shares.
    """
    index = {id_: k for k, id_ in enumerate(ids)}
    inside = np.ones(paths.shape[0], dtype=bool)
    for constraint in constraints:
        value = np.zeros(paths.shape[0])
        for term in constraint.terms:
            if term.variable not in index:
                raise ValueError(
                    f"the condition names {term.variable}, which the paths do not carry"
                )
            value += term.weight * paths[:, term.month, index[term.variable]]
        inside &= (constraint.low <= value) & (value <= constraint.high)
    survivors = int(inside.sum())
    if survivors == 0:
        return Reweight(
            weights=inside.astype(float),
            price=float("inf"),
            survivors=0,
            effective_size=0.0,
        )
    weights = inside.astype(float) / survivors
    return Reweight(
        weights=weights,
        price=float(-np.log(survivors / paths.shape[0])),
        survivors=survivors,
        effective_size=float(survivors),
    )


def resample(weights: np.ndarray, rng) -> np.ndarray:
    """``(n,)`` indices: a systematic resampling of the paths by their weights.

    The sample keeps its size and its law; weight one everywhere returns the paths in their
    order.
    """
    weights = np.asarray(weights, dtype=float)
    n = len(weights)
    share = weights / weights.sum()
    positions = (rng.random() + np.arange(n)) / n
    return np.searchsorted(np.cumsum(share), positions - 1e-12)
