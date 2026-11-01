"""The hierarchical satellite as a model plugin: estimate, load and simulate.

The Gibbs sampler and the out-of-sample check live in ``estimate`` and ``check``; this module
adapts them to the model SDK. ``estimate`` prepares the history of every assigned quantity
(``mst.drivers.history``), fits and checks it and returns the part of the version. A part
carries the quantity's own declaration, so ``load`` rebuilds the fits without any registry at
hand; the residual correlation is measured by the platform from the residuals returned here.
The part also records the lowest and highest value every node was observed at inside its own
estimation window (``train_low``, ``train_high``), so the platform checks simulated paths
against the range the node was trained on rather than against the whole history column.
"""

from dataclasses import dataclass, field
from datetime import date

import numpy as np

from mst.drivers.history import History, history
from mst.model_sdk import Check
from mst.model_sdk.contracts import (
    AdditiveImpact,
    DriverPaths,
    SatelliteEstimate,
    SatelliteEstimateContext,
    SatelliteModel,
    SatellitePart,
)
from mst.plugins.hier.check import check
from mst.plugins.hier.estimate import Fit, fit, residuals
from mst.plugins.hier.simulate import simulate as simulate_hier
from mst.plugins.hier.spec import Spec, load_spec

#: The quantities of the shipped registry this model serves by default; the platform may
#: assign it any other quantity, and a quantity several models declare needs an explicit
#: ``--model`` to name the one that estimates it.
QUANTITIES = (
    "share_price",
    "bond_spread",
    "overdue_share",
    "housing_price_primary",
    "housing_price_secondary",
)


@dataclass(frozen=True)
class HierSatellites:
    """The loaded model of a version: the fits of its quantities and their checks."""

    fits: dict[str, Fit] = field(default_factory=dict)
    checks: dict[str, dict[int, object]] = field(default_factory=dict)


class HierModel(SatelliteModel):
    """The hierarchical satellite model of section 6.2, over the objects of any quantity.

    ``spec`` is the estimation specification; without one the model reads ``spec.toml`` next to
    it. The platform's plugin uses the file; a test (or a variant of the model) may hand in its
    own.
    """

    model_id = "hier.gibbs"
    version = "0.1.0"
    label = "Иерархическая сателлитная модель"
    quantities = QUANTITIES
    description = (
        "Передача от макропеременных к величине объекта: регрессия каждого драйвера на своё "
        "прошлое и x_t, параметры объекта разыгрываются вокруг параметров объекта уровнем "
        "выше (Гиббс), корреляция остатков — по версии."
    )

    def __init__(self, spec: Spec | None = None):
        self.spec = spec

    def estimate(self, context: SatelliteEstimateContext) -> SatelliteEstimate:
        spec = self.spec or load_spec()
        parts: dict[str, SatellitePart] = {}
        residual_series: dict[str, np.ndarray] = {}
        for quantity in context.quantities:
            hist = history(context.registry, quantity, context.panel, context.store, context.as_of)
            fit_ = fit(hist, spec)
            checks = check(hist, spec)
            parts[quantity] = SatellitePart(
                meta={
                    "quantity": hist.quantity.model_dump(mode="json", by_alias=True),
                    "months": [m.isoformat() for m in hist.months],
                    "nodes": hist.nodes,
                    "parent": hist.parent.tolist(),
                    "classes": hist.classes,
                    "driver": hist.driver,
                    "rows": fit_.rows.tolist(),
                    "train_low": _bounds_list(fit_.train_low),
                    "train_high": _bounds_list(fit_.train_high),
                    "checks": {
                        str(node): _check_dict(item) for node, item in checks.items()
                    },
                    "spec": spec.model_dump(mode="json"),
                },
                arrays={
                    "gamma": fit_.gamma,
                    "sigma": fit_.sigma,
                    "d": hist.d,
                    "approximate": hist.approximate,
                    "x": hist.x,
                },
            )
            for node, driver in enumerate(hist.driver):
                if driver is not None and fit_.rows[node] > 0:
                    residual_series[driver] = residuals(fit_, node)
        return SatelliteEstimate(parts=parts, residuals=residual_series)

    def load(self, parts: dict[str, SatellitePart]) -> HierSatellites:
        from mst.drivers.registry import Quantity

        fits: dict[str, Fit] = {}
        checks: dict[str, dict[int, Check]] = {}
        for quantity, part in parts.items():
            q = Quantity.model_validate(part.meta["quantity"])
            hist = History(
                quantity=q,
                months=[date.fromisoformat(m) for m in part.meta["months"]],
                nodes=part.meta["nodes"],
                parent=np.array(part.meta["parent"], dtype=int),
                classes=part.meta["classes"],
                driver=part.meta["driver"],
                d=part.arrays["d"],
                approximate=part.arrays["approximate"],
                x=part.arrays["x"],
            )
            fits[quantity] = Fit(
                history=hist,
                gamma=part.arrays["gamma"],
                sigma=part.arrays["sigma"],
                rows=np.array(part.meta["rows"], dtype=int),
                train_low=_bounds(_required_bounds(part.meta, quantity, "train_low")),
                train_high=_bounds(_required_bounds(part.meta, quantity, "train_high")),
            )
            checks[quantity] = {
                int(node): Check(**row) for node, row in part.meta["checks"].items()
            }
        return HierSatellites(fits=fits, checks=checks)

    def simulate(
        self,
        model,
        upper,
        drivers: list[str],
        impacts: list[AdditiveImpact],
        *,
        correlation=None,
        shocks: np.ndarray,
        seed: int,
        stream: int,
    ) -> DriverPaths:
        return simulate_hier(
            model,
            upper,
            drivers,
            impacts,
            correlation=correlation,
            shocks=shocks,
            seed=seed,
            stream=stream,
        )


def _bounds_list(values: np.ndarray | None) -> list[float | None]:
    """The recorded training bounds of every node, ``None`` where a node has no range."""
    if values is None:
        return []
    return [None if not np.isfinite(value) else float(value) for value in values]


def _bounds(recorded: list) -> np.ndarray:
    """Bounds read back from a part; a node without a range stays ``NaN``."""
    return np.array([np.nan if value is None else value for value in recorded], dtype=float)


def _required_bounds(meta: dict, quantity: str, key: str) -> list:
    """The training bounds a part must record; a part without them is refused."""
    if key not in meta:
        raise KeyError(
            f"the part of {quantity} records no {key}: the outside-range check cannot run "
            "on it; re-estimate the bundle"
        )
    return meta[key]


def _check_dict(item) -> dict:
    return {
        "satellite": item.satellite,
        "baseline": item.baseline,
        "forecasts": item.forecasts,
        "served": item.served,
        "reason": item.reason,
    }
