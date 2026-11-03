"""The diagnostics of a saved macro version: ``python -m mst.models diagnose``.

The command reads a version exactly as it was saved — the meta, the arrays, the spec — and
measures what ``plans/macro-models.md`` measures:

- the spectral radius of every draw's companion matrix, per regime, with its quantiles and
  the shares of draws above 1, 1.005, 1.01 and 1.02 — the stability gate ``G1`` and the
  numbers behind it;
- the draws that survived the estimation, stability and identification filters;
- the regime shares of history per year and the share of the last month (the MS-VAR);
- the degrees of freedom by variable, where the model draws them;
- the weight the Minnesota prior carries against the data, per equation: ``P_0 / (X'X +
  P_0)`` on the diagonal of the lag block, the audit's measure of a prior that weighs less
  than its nominal value.

Nothing here estimates anything, and a version changed on disk is refused by the reader, as
every version is.
"""

from datetime import date
from pathlib import Path

import numpy as np

from mst.macro import version as macro_version
from mst.macro.panel import Panel
from mst.model_sdk.stability import companion_radius

#: The radii the shares of draws are reported above, the plan's fixed grid.
ABOVE = (1.0, 1.005, 1.01, 1.02)


def diagnose(models: Path, name: str) -> dict:
    """Everything the diagnose command says about the version ``name``; JSON-ready."""
    meta, arrays = macro_version.read(models, name)
    if "lags" not in arrays:
        raise ValueError(f"macro version {name} carries no lags to diagnose")
    found: dict = {
        "version": name,
        "model": meta.get("model"),
        "as_of": meta.get("as_of"),
        "draws": {
            key: meta.get("diagnostics", {})[key]
            for key in ("estimated", "stable", "identified")
            if key in meta.get("diagnostics", {})
        },
        "rho": _rho(arrays["lags"], meta.get("regimes")),
        "prior_weight": _prior_weight(meta, arrays),
    }
    if "states" in arrays:
        found |= _regime_parts(arrays, meta)
    if "dof" in arrays:
        found["dof"] = _dof(arrays["dof"], meta["ids"])
    return found


def _summary(radius: np.ndarray) -> dict:
    """The quantiles of a radius sample and the shares of draws above the fixed grid."""
    return {
        "median": float(np.median(radius)),
        "p95": float(np.quantile(radius, 0.95)),
        "p99": float(np.quantile(radius, 0.99)),
        "max": float(np.max(radius)),
        "above": {f"{limit:g}": float(np.mean(radius > limit)) for limit in ABOVE},
    }


def _rho(lags: np.ndarray, regimes: list[str] | None) -> dict:
    """The radius summary per regime; a model without regimes has one, called ``model``."""
    matrix = np.asarray(lags)
    if matrix.ndim == 5:
        names = list(regimes or (f"regime {k + 1}" for k in range(matrix.shape[1])))
        radii = companion_radius(matrix)  # (draws, regimes)
        return {name: _summary(radii[:, j]) for j, name in enumerate(names)}
    if matrix.ndim == 4:
        return {"model": _summary(companion_radius(matrix))}
    raise ValueError(
        f"lags must be (draws, p, n, n) or (draws, regimes, p, n, n), got {matrix.shape}"
    )


def _regime_parts(arrays: dict, meta: dict) -> dict:
    """The regime shares of history per year and of the last month, from the saved path."""
    states = arrays["states"]  # (draws, T - p)
    months = [date.fromisoformat(m) for m in meta["months"]]
    names = list(meta.get("regimes") or (f"regime {k + 1}" for k in range(states.max() + 1)))
    equation = months[len(months) - states.shape[1] :]
    by_year = {}
    for year in sorted({month.year for month in equation}):
        columns = [t for t, month in enumerate(equation) if month.year == year]
        by_year[str(year)] = {
            name: float(np.mean(states[:, columns] == j)) for j, name in enumerate(names)
        }
    last = {name: float(np.mean(states[:, -1] == j)) for j, name in enumerate(names)}
    # The declaration order is the volatility order, so the last regime is the stress one.
    return {
        "regime_share_by_year": by_year,
        "regime_share_last_month": last,
        "stress_share_last_month": last[names[-1]],
    }


def _dof(dof: np.ndarray, ids: list[str]) -> dict[str, float | None]:
    """The median degrees of freedom by variable; ``None`` where every draw is Gaussian."""
    values = np.asarray(dof, dtype=float)
    found = {}
    for k, id_ in enumerate(ids):
        finite = values[:, k][np.isfinite(values[:, k])]
        found[id_] = float(np.median(finite)) if finite.size else None
    return found


def _prior_weight(meta: dict, arrays: dict) -> dict[str, dict[str, float]]:
    """``P_0 / (X'X + P_0)`` on the diagonal of the lag block, per equation.

    One at zero is a prior that does nothing, one at one a prior that swallows the data. The
    prior precision ``P_0`` is the one the estimating model builds — imported from its own
    package — taken relative to the error variance of the equation, the scale ``X'X`` stands
    on, so the number does not depend on the units of any variable: it is the audit's. The
    dummy rows of the prior count in ``P_0`` through the diagonal of their own ``X'X``. ``X``
    is the interpolated panel, the deterministic stand-in for the augmented one the sampler
    sees. ``mean`` runs over the coefficients the equation is allowed to hold (block
    exogeneity leaves the global equations without the Russian lags); ``own_first_lag`` is
    the coefficient the audit quotes.
    """
    model = meta.get("model")
    if model == "bvar.sv":
        from mst.plugins.bvar.estimate import fill, minnesota
        from mst.plugins.bvar.spec import Spec
        from mst.plugins.bvar.var import allowed

        dummies = len(Spec.model_validate(meta["spec"]).dummies)
    elif model == "msvar.markov":
        from mst.plugins.msvar.estimate import fill, minnesota
        from mst.plugins.msvar.spec import Spec
        from mst.plugins.msvar.var import allowed

        dummies = None
    else:
        return {}
    spec = Spec.model_validate(meta["spec"])
    panel = Panel(
        ids=meta["ids"],
        n_global=meta["n_global"],
        months=[date.fromisoformat(m) for m in meta["months"]],
        y=arrays["panel_y"],
        approximate=arrays["panel_approximate"],
    )
    prior = minnesota(panel, spec)
    n, p = len(panel.ids), spec.lags
    mask = (
        allowed(n, panel.n_global, p)
        if dummies is None
        else allowed(n, panel.n_global, p, dummies)
    )
    y = fill(panel.y)
    count = len(y) - p
    gram = np.empty(n * p)
    for lag in range(1, p + 1):
        for j in range(n):
            # The column of lag ``lag`` and variable ``j`` reads y_{t-lag, j}, t = p .. T-1.
            gram[(lag - 1) * n + j] = np.sum(y[p - lag : p - lag + count, j] ** 2)
    block = slice(1, 1 + n * p)
    precision = prior.scale[:, None] ** 2 / prior.var[:, block]
    if prior.observation_x is not None:
        precision = precision + np.sum(prior.observation_x[:, block] ** 2, axis=0)[None, :]
    weight = precision / (gram[None, :] + precision)
    chosen = mask[:, block]
    return {
        id_: {
            "mean": float(np.mean(weight[k][chosen[k]])),
            "own_first_lag": float(weight[k, k]),
        }
        for k, id_ in enumerate(panel.ids)
    }
