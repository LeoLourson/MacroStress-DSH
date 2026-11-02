"""``diagnose-satellites``: what a saved bundle implies, read back without estimating again.

The repair plan of the satellites asks for numbers before it changes anything: the share of
the previous month a driver keeps, the elasticity the long-run level carries to every macro
input, the distance of that level from the last observation and the pull to it over twelve
months, the reasons the out-of-sample check rejected drivers with and the quantiles of the
served drivers' RMSE ratio against the autoregression without macro. All of it is in the meta
and the arrays of the bundle; this module only reads and rearranges.

A quantity estimated in differences has no long-run level, so its distance and pull are
``None``; the loadings stay, because a permanent shift of an input still moves the level by
the same sum of coefficients over ``1 - rho``.
"""

from pathlib import Path

import numpy as np

from mst import versions
from mst.drivers.bundle import HIER, long_run_loadings

MONTHS = 12
"""The pull of the plan: how far twelve months take a driver towards its long-run level."""

INTERVAL = (5.0, 50.0, 95.0)
"""The posterior quantiles every number of this report carries."""


def diagnose_satellites(models: Path, version: str) -> dict:
    """The diagnosis of the satellite bundle ``version`` of ``models``, as JSON-ready data."""
    meta, arrays = versions.read(Path(models) / "satellites", version, "satellite version")
    return {
        "version": version,
        "macro": meta["macro"],
        "as_of": meta["as_of"],
        "quantities": {
            quantity: _quantity(quantity, part, arrays)
            for quantity, part in sorted(meta["quantities"].items())
        },
    }


def _quantity(quantity: str, part: dict, arrays: dict) -> dict:
    """One quantity: the diagnosis of every named object and the aggregates over them."""
    gamma = arrays.get(f"{quantity}.gamma")
    differenced = quantity in part.get("spec", {}).get("differences", [])
    linear = part.get("model") == HIER and gamma is not None
    today = _today(part, arrays, quantity) if linear and not differenced else None
    objects = {}
    rhos, ratios = [], []
    reasons: dict[str, int] = {}
    served = 0
    for node, driver in enumerate(part["driver"]):
        if driver is None:
            continue
        row = part["checks"].get(str(node))
        reason = row.get("reason", "") if row is not None else "no check of this driver"
        reasons[reason] = reasons.get(reason, 0) + 1
        if row is not None and row.get("served"):
            served += 1
        ratio = None
        if row is not None and row.get("satellite") and row.get("baseline"):
            ratio = row["satellite"] / row["baseline"]
            if row.get("served"):
                ratios.append(ratio)
        entry = {
            "rows": part["rows"][node],
            "served": bool(row.get("served")) if row is not None else False,
            "reason": reason,
            "rmse_ratio": ratio,
        }
        if linear:
            entry |= _linear(part, gamma, node, today, arrays, quantity, differenced)
            if part["rows"][node] > 0:
                rhos.append(float(np.median(gamma[:, node, 1])))
        objects[driver] = entry
    return {
        "model": part.get("model"),
        "differenced": differenced,
        "objects": dict(sorted(objects.items())),
        "rho": _spread(rhos),
        "checks": dict(sorted(reasons.items())),
        "served": served,
        "rmse_ratio": (
            {"p25": float(np.percentile(ratios, 25)), "p50": float(np.percentile(ratios, 50)),
             "p75": float(np.percentile(ratios, 75))}
            if ratios
            else None
        ),
    }


def _linear(part, gamma, node, today, arrays, quantity, differenced) -> dict:
    """The numbers of the linear form: rho, drift, loadings and, in levels, the pull."""
    draws = gamma[:, node]
    out = {
        "rho": _interval(draws[:, 1]),
        "drift": _interval(draws[:, 0]),
        "long_run": {
            name: _interval(loading)
            for name, loading in long_run_loadings(part["quantity"], gamma, node).items()
        },
    }
    if differenced or part["rows"][node] == 0 or today is None:
        out["d_star_minus_d0"] = None
        out["pull_12m"] = None
        return out
    d = arrays[f"{quantity}.d"][:, node]
    known = np.flatnonzero(np.isfinite(d))
    if not len(known):
        out["d_star_minus_d0"] = None
        out["pull_12m"] = None
        return out
    rho = draws[:, 1]
    loadings = long_run_loadings(part["quantity"], gamma, node)
    total = sum(values * float(x) for values, x in zip(loadings.values(), today, strict=True))
    gap = draws[:, 0] / (1 - rho) + total - d[known[-1]]
    out["d_star_minus_d0"] = _interval(gap)
    out["pull_12m"] = _interval(gap * (1 - rho**MONTHS))
    return out


def _today(part, arrays, quantity) -> np.ndarray | None:
    """The last month of the panel whose every input is known; the pull is measured at it."""
    x = arrays.get(f"{quantity}.x")
    if x is None or not len(x):
        return None
    known = np.flatnonzero(np.all(np.isfinite(x), axis=1))
    return x[known[-1]] if len(known) else None


def _interval(draws: np.ndarray) -> dict:
    return dict(zip(("p05", "p50", "p95"), (float(v) for v in np.percentile(draws, INTERVAL)),
                    strict=True))


def _spread(values: list[float]) -> dict | None:
    """The quantiles of the per-node medians of a quantity; None without a node with data."""
    if not values:
        return None
    return dict(
        zip(("p05", "p50", "p95"), (float(v) for v in np.percentile(values, INTERVAL)),
            strict=True)
    )
