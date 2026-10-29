"""Sign restrictions of section 4.2 (Rubio-Ramirez, Waggoner and Zha, 2010).

For a parameter draw the impact matrix at the calculation month is ``B = C diag(exp(h_0 / 2))
Q``; any orthogonal ``Q`` fits the data equally well. ``Q`` is block diagonal, global block
first, so Russian shocks never move global variables. Within a block, ``Q`` is drawn from the
uniform (Haar) distribution until the column of each named shock moves its variables in the
required direction in the impact month and ``sign_lag`` months after it; a column may change
sign, which leaves the distribution uniform. Only the named columns are drawn by rejection,
the others complete them uniformly. A draw without an admissible ``Q`` after ``tries``
rotations is dropped.

Columns are ordered by block, named shocks first in registry order; the rest are unnamed.
"""

from dataclasses import replace

import numpy as np
from scipy.linalg import block_diag

from .estimate import Posterior
from .registry import Shock, block_of
from .var import responses

BATCH = 1000


def impact(posterior: Posterior, draw: int, month: int = -1) -> np.ndarray:
    """``C diag(exp(h / 2))`` of a draw at a month of history (the last by default)."""
    return posterior.chol[draw] * np.exp(posterior.logvar[draw, month] / 2)[None, :]


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

    ``response`` (months, n, size) moves each variable per unit of each orthogonal shock of the
    block; ``signs`` (named, n) holds +1 / -1 for a restricted variable and 0 otherwise.
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


def identify(posterior: Posterior, shocks: list[Shock], *, sign_lag: int, tries: int, seed: int):
    """The posterior restricted to draws with an admissible ``Q``, with ``rotation`` and
    ``shocks`` set. Raises ``ValueError`` if no draw has one."""
    ids, n, g = posterior.ids, len(posterior.ids), posterior.n_global
    index = {id_: i for i, id_ in enumerate(ids)}
    blocks = {"gl": range(0, g), "ru": range(g, n)}
    named = {block: [s for s in shocks if block_of(s.id) == block] for block in blocks}
    signs = {}
    for block, chosen in named.items():
        signs[block] = np.zeros((len(chosen), n))
        for j, shock in enumerate(chosen):
            signs[block][j, [index[v] for v in shock.rises]] = 1.0
            signs[block][j, [index[v] for v in shock.falls]] = -1.0
        if len(chosen) > len(blocks[block]):
            raise ValueError(f"more named {block} shocks than {block} variables")
    labels = []
    for block, columns in blocks.items():
        labels += [s.id for s in named[block]]
        labels += [f"{block}.unnamed_{k + 1}" for k in range(len(columns) - len(named[block]))]

    rng = np.random.default_rng(seed)
    kept, rotations = [], []
    for draw in range(posterior.draws):
        psi = responses(posterior.lags[draw], sign_lag + 1) @ impact(posterior, draw)
        parts = [rotate(psi[:, :, list(c)], signs[b], tries, rng) for b, c in blocks.items()]
        if all(part is not None for part in parts):
            kept.append(draw)
            rotations.append(block_diag(*parts))
    if not kept:
        raise ValueError(f"no draw admits the sign restrictions within {tries} rotations")
    pick = np.array(kept)
    return replace(
        posterior,
        const=posterior.const[pick],
        lags=posterior.lags[pick],
        dummy=posterior.dummy[pick],
        chol=posterior.chol[pick],
        logvar=posterior.logvar[pick],
        volvol=posterior.volvol[pick],
        dof=posterior.dof[pick],
        history=posterior.history[pick],
        rotation=np.array(rotations),
        shocks=labels,
        diagnostics={
            **posterior.diagnostics,
            "identified": len(kept),
            "estimated": posterior.draws,
        },
    )
