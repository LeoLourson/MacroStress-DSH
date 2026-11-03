"""Sign restrictions of the MS-VAR within each regime (Rubio-Ramirez, Waggoner and Zha, 2010,
Review of Economic Studies 77(2)).

For a parameter draw and a regime the impact matrix is ``B_s = C_s diag(exp(h_s / 2)) Q``;
any orthogonal ``Q`` fits the data equally well, in every regime. ``Q`` is block diagonal,
global block first, so Russian shocks never move global variables. Within a block, ``Q`` is
drawn from the uniform (Haar) distribution until the column of each named shock moves its
variables the required way in the impact month and ``sign_lag`` months after it — under the
regime's own dynamics. A draw is kept when every regime admits such a ``Q``; a column may
change sign, which leaves the distribution uniform. Only the named columns are drawn by
rejection, the others complete them uniformly.

Columns are ordered by block, named shocks first in registry order, the rest unnamed — the
same order in every regime, so a shock id names one column of every ``B_s``.
"""

from dataclasses import replace

import numpy as np
from scipy.linalg import block_diag

from mst.macro.registry import block_of
from mst.model_sdk.stability import companion_radius

from .estimate import Posterior
from .signs import Signs
from .var import responses

BATCH = 1000


def stable(posterior: Posterior, rho_max: float) -> Posterior:
    """The posterior restricted to the draws whose every regime keeps its companion radius
    at or below ``rho_max``.

    A unit root is admissible — most of these variables are nonstationary — but growth
    without bound is not: a path simulated from an explosive regime drifts ever further,
    and a condition on a path multiplies the effect. The mechanism is the identification
    filter's — the posterior restricted to a region — and it runs before the rotation
    search. Raises ``ValueError`` naming the model and the limit when no draw survives.
    """
    radius = companion_radius(posterior.lags)  # (draws, regimes)
    pick = np.flatnonzero((radius <= rho_max).all(axis=1))
    if not len(pick):
        raise ValueError(
            "no draw of msvar.markov keeps the companion radius of every regime "
            f"at or below rho_max {rho_max}"
        )
    return replace(
        posterior,
        const=posterior.const[pick],
        lags=posterior.lags[pick],
        chol=posterior.chol[pick],
        logvar=posterior.logvar[pick],
        transition=posterior.transition[pick],
        states=posterior.states[pick],
        history=posterior.history[pick],
        diagnostics={**posterior.diagnostics, "stable": len(pick)},
    )


def impact(posterior: Posterior, draw: int, regime: int) -> np.ndarray:
    """``C_s diag(exp(h_s / 2))`` of a parameter draw in a regime."""
    return posterior.chol[draw, regime] * np.exp(posterior.logvar[draw, regime] / 2)[None, :]


def haar_columns(size: int, count: int, batch: int, rng) -> np.ndarray:
    """``(batch, size, count)``: the first ``count`` columns of uniform orthogonal matrices."""
    q, r = np.linalg.qr(rng.standard_normal((batch, size, count)))
    return q * np.sign(np.diagonal(r, axis1=1, axis2=2))[:, None, :]


def complete(columns: np.ndarray, rng) -> np.ndarray:
    """An orthogonal matrix whose first columns are ``columns``, the rest uniform."""
    size, count = columns.shape
    q, r = np.linalg.qr(np.hstack([columns, rng.standard_normal((size, size - count))]))
    q = q * np.sign(np.diag(r))[None, :]
    q[:, :count] = columns
    return q


def rotate(response: np.ndarray, signs: np.ndarray, tries: int, rng) -> np.ndarray | None:
    """An admissible ``Q`` of one block, or None.

    ``response`` (months, n, size) moves each variable per unit of each orthogonal shock of
    the block; ``signs`` (named, n) holds +1 / -1 for a restricted variable and 0 otherwise.
    """

    size = response.shape[2]
    named = len(signs)
    for start in range(0, tries, BATCH):
        q = haar_columns(size, named, min(BATCH, tries - start), rng)
        moved = np.einsum("hvn,bnk->bhvk", response, q)
        ok = np.ones(len(q), dtype=bool)
        flips = np.ones((len(q), named))
        for j, sign in enumerate(signs):
            restricted = sign != 0
            value = moved[:, :, restricted, j] * sign[restricted]
            up, down = (value > 0).all(axis=(1, 2)), (value < 0).all(axis=(1, 2))
            ok &= up | down
            flips[:, j] = np.where(up, 1.0, -1.0)
        if ok.any():
            first = int(np.argmax(ok))
            return complete(q[first] * flips[first][None, :], rng)
    return None


def identify(
    posterior: Posterior,
    signs: Signs,
    *,
    sign_lag: int,
    tries: int,
    seed: int,
    rho_max: float | None = None,
) -> Posterior:
    """The posterior restricted to draws whose every regime admits an admissible ``Q``.

    Sets the whole ``J`` set of the version: the economic shocks — the columns of every
    ``B_s`` — and the regime elements the signs table declares. Raises ``ValueError`` when
    no draw has an admissible rotation in some regime.

    With ``rho_max`` set, the stability filter of ``stable`` runs first, in every regime,
    so the rotations are searched on the stable draws alone.
    """
    estimated = posterior.draws
    if rho_max is not None:
        posterior = stable(posterior, rho_max)
    shocks = signs.shocks
    ids, n, g = posterior.ids, len(posterior.ids), posterior.n_global
    index = {id_: i for i, id_ in enumerate(ids)}
    blocks = {"gl": range(0, g), "ru": range(g, n)}
    named = {block: [s for s in shocks if block_of(s.id) == block] for block in blocks}
    restrictions = {}
    for block, chosen in named.items():
        restrictions[block] = np.zeros((len(chosen), n))
        for j, shock in enumerate(chosen):
            restrictions[block][j, [index[v] for v in shock.rises]] = 1.0
            restrictions[block][j, [index[v] for v in shock.falls]] = -1.0
        if len(chosen) > len(blocks[block]):
            raise ValueError(f"more named {block} shocks than {block} variables")
    labels = []
    for block, columns in blocks.items():
        labels += [s.id for s in named[block]]
        labels += [f"{block}.unnamed_{k + 1}" for k in range(len(columns) - len(named[block]))]

    rng = np.random.default_rng(seed)
    kept, rotations = [], []
    for draw in range(posterior.draws):
        of_regime = []
        for s in range(posterior.n_regimes):
            psi = responses(posterior.lags[draw, s], sign_lag + 1) @ impact(posterior, draw, s)
            parts = [
                rotate(psi[:, :, list(columns)], restrictions[block], tries, rng)
                for block, columns in blocks.items()
            ]
            if any(part is None for part in parts):
                of_regime = []
                break
            of_regime.append(block_diag(*parts))
        if of_regime:
            kept.append(draw)
            rotations.append(np.stack(of_regime))
    if not kept:
        raise ValueError(
            f"no draw admits the sign restrictions in every regime within {tries} rotations"
        )
    pick = np.array(kept)
    return replace(
        posterior,
        const=posterior.const[pick],
        lags=posterior.lags[pick],
        chol=posterior.chol[pick],
        logvar=posterior.logvar[pick],
        transition=posterior.transition[pick],
        states=posterior.states[pick],
        history=posterior.history[pick],
        rotation=np.array(rotations),
        shocks=labels,
        regime_ids=signs.regime_ids,
        diagnostics={
            **posterior.diagnostics,
            "identified": len(kept),
            "estimated": estimated,
        },
    )
