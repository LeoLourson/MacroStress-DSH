"""The reference model of the second level: a regression on the quantities above.

``estimate`` reads, for every assigned quantity, its drivers' own series from the store and
builds the design ``g(x^{<l}_t) = [1, x_t]`` over the declared inputs: a macro input reads
the panel, an input naming a quantity reads ``upper_history`` at the object itself or the
nearest object above it that carries the input — the object tree is resolved once, here, so
the simulation reads paths and nothing else. Every driver with enough observations is fitted
by the conjugate normal-inverse-gamma regression — the exact posterior of ``(beta, sigma)`` —
and the part records the training bounds of its node. ``simulate`` draws one posterior index
per repetition and walks the horizon with ``x_t`` from ``UpperPaths`` and the platform's
shocks as residuals.

The model declares no quantity: a quantity reaches it through an explicit assignment
(``--model QUANTITY=reflevel2.regression``), which the level tests and the catalog
``tests/drivers/level2_catalog`` show. Around two hundred lines, no knobs of its own: it is
the reference the canon's section 8 names, not a production satellite.
"""

from dataclasses import dataclass
from datetime import date
from zlib import crc32

import numpy as np

from mst.drivers.registry import to_model
from mst.model_sdk.contracts import (
    AdditiveImpact,
    Check,
    DriverPaths,
    SatelliteEstimate,
    SatelliteEstimateContext,
    SatelliteModel,
    SatellitePart,
    UpperHistory,
    UpperPaths,
)

POSTERIOR_DRAWS = 64
"""How many posterior draws of ``(beta, sigma)`` a fit carries for the simulation to pick."""
MINIMUM_ROWS = 12
"""A driver with fewer known months is not served: the posterior would be all prior."""

NOT_ENOUGH = f"fewer than {MINIMUM_ROWS} known months"
"""Why a driver without enough observations is not served."""
EXPLAINS_NOTHING = "the levels above explain nothing"
"""Why a served-looking driver is refused: the regression beats no constant."""

#: One column of the design: a macro variable or the driver of a quantity of a level above,
#: with the threshold of a feature (zero for a plain regressor).
Input = tuple[str, str, float]


@dataclass(frozen=True)
class Fit:
    """The posterior of one quantity's regression, per node with data."""

    months: list[date]
    nodes: list[str]
    driver: list[str | None]
    rows: np.ndarray
    beta: np.ndarray
    """``(S, N, k)`` posterior draws of the coefficients of every node."""
    sigma: np.ndarray
    """``(S, N)`` posterior draws of the residual scale of every node."""
    inputs: dict[str, list[Input]]
    """The resolved design columns of every node: ``(kind, id, threshold)``."""
    train_low: np.ndarray
    train_high: np.ndarray
    d: np.ndarray
    """``(T, N)`` the observed values on the panel months, NaN where unknown."""


@dataclass(frozen=True)
class RefLevel2:
    """The loaded fits and checks of the assigned quantities."""

    fits: dict[str, Fit]
    checks: dict[str, dict[int, Check]]


def _object_chain(registry, node: str) -> list[str]:
    """The object itself, then its ancestors, nearest first."""
    chain = [node]
    current = registry.object(node)
    while current.parent is not None:
        chain.append(current.parent)
        current = registry.object(current.parent)
    return chain


def _resolve_inputs(registry, quantity, upper: dict[str, UpperHistory], node: str):
    """The design columns of one node: ``(kind, id, threshold)`` per input, or None.

    A macro input resolves to the panel column, a feature to the same column with its
    threshold; an input naming a quantity resolves to the driver of that quantity at the
    object itself or the nearest object above it that carries one — the way the hierarchical
    model reads the level above. The object tree is walked once, here.
    """
    quantities = {q.id for q in registry.quantities}
    resolved: list[Input] = []
    for name in quantity.regressors + [f.variable for f in quantity.features]:
        threshold = next(
            (float(f.above) for f in quantity.features if f.variable == name), 0.0
        )
        if name in quantities:
            history = upper.get(name)
            driver = None
            if history is not None:
                for candidate in _object_chain(registry, node):
                    if f"{name}:{candidate}" in history.drivers:
                        driver = f"{name}:{candidate}"
                        break
            if driver is None:
                return None
            resolved.append(("driver", driver, threshold))
        else:
            resolved.append(("macro", name, threshold))
    return resolved


def _design_history(
    panel, upper: dict[str, UpperHistory], inputs: list[Input]
) -> np.ndarray | None:
    """``(T, k)`` the design of one node on the panel months, NaN where an input is unknown."""
    index = {id_: k for k, id_ in enumerate(panel.ids)}
    parts = [np.ones(len(panel.months))]
    for kind, name, threshold in inputs:
        if kind == "macro":
            if name not in index:
                return None
            column = panel.y[:, index[name]]
        else:
            quantity, node = name.split(":", 1)
            history = upper.get(quantity)
            if history is None or name not in history.drivers:
                return None
            column = history.values[:, history.drivers.index(name)]
        parts.append(np.maximum(0.0, column - threshold) if threshold else column)
    return np.column_stack(parts)


def _design_paths(upper: UpperPaths, inputs: list[Input]) -> np.ndarray:
    """``(R, H + 1, k)`` the design of one node on the horizon, per repetition."""
    macro = upper.macro
    index = {id_: k for k, id_ in enumerate(macro.ids)}
    parts = [np.ones(macro.paths.shape[:2])]
    for kind, name, threshold in inputs:
        if kind == "macro":
            column = macro.paths[:, :, index[name]]
        else:
            column = None
            for level_paths in upper.levels.values():
                if name in level_paths.ids:
                    column = level_paths.paths[:, :, level_paths.ids.index(name)]
                    break
            if column is None:
                raise ValueError(f"the paths carry no driver {name}")
        parts.append(np.maximum(0.0, column - threshold) if threshold else column)
    return np.stack(parts, axis=-1)


def _posterior(x: np.ndarray, d: np.ndarray, *, draws: int, seed: int):
    """The conjugate posterior of a regression ``d = x beta + e``: draws of ``(beta, sigma)``
    and the mean fit, or ``None`` without enough known months."""
    known = np.isfinite(x).all(axis=1) & np.isfinite(d)
    x, d = x[known], d[known]
    if len(d) < MINIMUM_ROWS:
        return None
    precision = np.linalg.inv(x.T @ x)
    beta_hat = precision @ (x.T @ d)
    residual = d - x @ beta_hat
    dof = len(d) - x.shape[1]
    scale = float(residual @ residual / dof)
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(scale * dof / rng.chisquare(dof, size=draws))
    root = np.linalg.cholesky(precision)
    beta = np.empty((draws, x.shape[1]))
    for s in range(draws):
        beta[s] = beta_hat + sigma[s] * (root @ rng.standard_normal(x.shape[1]))
    return beta, sigma, beta_hat, scale


class RefLevel2Model(SatelliteModel):
    """The regression of a quantity of the second level on the levels above it."""

    model_id = "reflevel2.regression"
    version = "0.1.0"
    label = "Эталонная регрессия второго уровня"
    quantities: tuple[str, ...] = ()

    def estimate(self, context: SatelliteEstimateContext) -> SatelliteEstimate:
        parts: dict[str, SatellitePart] = {}
        residuals: dict[str, np.ndarray] = {}
        for quantity_id in context.quantities:
            quantity = context.registry.quantity(quantity_id)
            nodes = [
                driver.object
                for driver in context.registry.drivers
                if driver.quantity == quantity_id
            ]
            months = list(context.panel.months)
            d = np.full((len(months), len(nodes)), np.nan)
            approximate = np.zeros(d.shape, dtype=bool)
            beta = np.zeros((POSTERIOR_DRAWS, len(nodes), 1 + len(quantity.inputs)))
            sigma = np.ones((POSTERIOR_DRAWS, len(nodes)))
            rows = np.zeros(len(nodes), dtype=int)
            checks: dict[int, dict] = {}
            inputs: dict[str, list[Input]] = {}
            fitted: list[np.ndarray | None] = [None] * len(nodes)
            for k, node in enumerate(nodes):
                driver = next(
                    (
                        item
                        for item in context.registry.drivers
                        if item.quantity == quantity_id and item.object == node
                    ),
                    None,
                )
                if driver is not None:
                    at = {
                        row.month: row
                        for row in context.store.as_of(driver.series, context.as_of)
                    }
                    for t, month in enumerate(months):
                        if month in at:
                            d[t, k] = to_model(quantity.transform, at[month].value)
                            approximate[t, k] = at[month].approximate
                resolved = _resolve_inputs(
                    context.registry, quantity, context.upper_history, node
                )
                inputs[node] = resolved or []
                design = (
                    _design_history(context.panel, context.upper_history, resolved)
                    if resolved
                    else None
                )
                found = (
                    _posterior(
                        design,
                        d[:, k],
                        draws=POSTERIOR_DRAWS,
                        seed=crc32(f"{quantity_id}:{node}".encode()) % 2**31,
                    )
                    if design is not None
                    else None
                )
                if found is None:
                    checks[k] = {
                        "satellite": None,
                        "baseline": None,
                        "forecasts": 0,
                        "served": False,
                        "reason": NOT_ENOUGH,
                    }
                    continue
                beta[:, k, : design.shape[1]] = found[0]
                sigma[:, k] = found[1]
                rows[k] = int(np.isfinite(d[:, k]).sum())
                known = np.isfinite(d[:, k])
                fitted[k] = d[known, k] - design[known] @ found[2]
                rmse = float(np.sqrt(found[3]))
                spread = float(np.std(d[known, k]))
                served = rmse < spread
                checks[k] = {
                    "satellite": rmse,
                    "baseline": spread,
                    "forecasts": int(rows[k]),
                    "served": served,
                    "reason": "" if served else EXPLAINS_NOTHING,
                }
            parts[quantity_id] = SatellitePart(
                meta={
                    "quantity": quantity.model_dump(mode="json", by_alias=True),
                    "months": [m.isoformat() for m in months],
                    "nodes": nodes,
                    "driver": [
                        f"{quantity_id}:{node}" if rows[k] else None
                        for k, node in enumerate(nodes)
                    ],
                    "rows": rows.tolist(),
                    "inputs": {node: [list(column) for column in own] for node, own in inputs.items()},
                    "train_low": [
                        float(np.nanmin(d[:, k])) if rows[k] else None
                        for k in range(len(nodes))
                    ],
                    "train_high": [
                        float(np.nanmax(d[:, k])) if rows[k] else None
                        for k in range(len(nodes))
                    ],
                    "checks": {str(k): row for k, row in checks.items()},
                },
                arrays={
                    "beta": beta,
                    "sigma": sigma,
                    "d": d,
                    "approximate": approximate,
                },
            )
            for k, node in enumerate(nodes):
                if rows[k] and fitted[k] is not None:
                    residuals[f"{quantity_id}:{node}"] = fitted[k]
        return SatelliteEstimate(parts=parts, residuals=residuals)

    def load(self, parts: dict[str, SatellitePart]) -> RefLevel2:
        fits: dict[str, Fit] = {}
        checks: dict[str, dict[int, Check]] = {}
        for quantity, part in parts.items():
            meta = part.meta
            fits[quantity] = Fit(
                months=[date.fromisoformat(m) for m in meta["months"]],
                nodes=meta["nodes"],
                driver=meta["driver"],
                rows=np.array(meta["rows"], dtype=int),
                beta=np.asarray(part.arrays["beta"], dtype=float),
                sigma=np.asarray(part.arrays["sigma"], dtype=float),
                inputs={
                    node: [(kind, name, float(threshold)) for kind, name, threshold in own]
                    for node, own in meta["inputs"].items()
                },
                train_low=np.array(
                    [np.nan if v is None else v for v in meta["train_low"]], dtype=float
                ),
                train_high=np.array(
                    [np.nan if v is None else v for v in meta["train_high"]], dtype=float
                ),
                d=np.asarray(part.arrays["d"], dtype=float),
            )
            checks[quantity] = {
                int(node): Check(**row) for node, row in meta["checks"].items()
            }
        return RefLevel2(fits=fits, checks=checks)

    def simulate(
        self,
        model: RefLevel2,
        upper: UpperPaths,
        drivers: list[str],
        impacts: list[AdditiveImpact],
        *,
        correlation=None,
        shocks: np.ndarray,
        seed: int,
        stream: int,
    ) -> DriverPaths:
        draws, horizon = upper.macro.paths.shape[0], upper.macro.paths.shape[1] - 1
        paths = np.empty((draws, horizon + 1, len(drivers)))
        flags: dict[str, frozenset[str]] = {}
        plan: list[tuple[str, Fit, int, Check]] = []
        for driver in drivers:
            quantity, node = driver.split(":", 1)
            if quantity not in model.fits:
                raise KeyError(driver)
            fit_ = model.fits[quantity]
            if node not in fit_.nodes:
                raise KeyError(driver)
            column = fit_.nodes.index(node)
            check = model.checks[quantity].get(column)
            if check is None or not check.served:
                raise ValueError(
                    f"{driver} is not served: {check.reason if check else 'no check of it'}"
                )
            plan.append((driver, fit_, column, check))
        own = [driver for driver, fit_, column, _ in plan if fit_.rows[column] > 0]
        if shocks.ndim != 3 or shocks.shape[0] != draws or shocks.shape[2] != len(own):
            raise ValueError(
                f"the platform passed shocks of shape {shocks.shape}, this model expects "
                f"({draws}, steps, {len(own)}) for {len(own)} correlated drivers"
            )
        for k, (driver, fit_, column, _) in enumerate(plan):
            node = fit_.nodes[column]
            design = _design_paths(upper, fit_.inputs[node])
            known = np.flatnonzero(np.isfinite(fit_.d[:, column]))
            start = float(fit_.d[known[-1], column]) if len(known) else 0.0
            for r in range(draws):
                rng = np.random.default_rng([seed, stream, r])
                s = int(rng.integers(fit_.beta.shape[0]))
                beta = fit_.beta[s, column]
                sigma = fit_.sigma[s, column]
                paths[r, 0, k] = start
                noise = (
                    sigma * shocks[r, :horizon, own.index(driver)]
                    if driver in own
                    else sigma * rng.standard_normal(horizon)
                )
                for t in range(1, horizon + 1):
                    paths[r, t, k] = beta @ design[r, t] + noise[t - 1]
            flags[driver] = frozenset()
        for impact in impacts:
            if impact.target not in flags:
                raise ValueError(
                    f"an impact on {impact.target}, which this model does not simulate"
                )
            k = drivers.index(impact.target)
            paths[:, impact.start : impact.end + 1, k] += impact.delta
            flags[impact.target] = flags[impact.target] | {"impact"}
        return DriverPaths(ids=list(drivers), paths=paths, flags=flags)


MODEL = RefLevel2Model()
