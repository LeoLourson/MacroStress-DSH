"""The satellite bundle: one version, one model per quantity, loaded back and simulated."""

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pytest

from mst.drivers import bundle as satellite_bundle
from mst.drivers.registry import Registry
from mst.macro.panel import shift
from mst.model_sdk import AdditiveImpact, DriverPaths, SatelliteModel
from mst.model_sdk.contracts import MacroSimulation, SatelliteEstimate, SatellitePart
from tests.drivers import simulated

AS_OF = date(2030, 1, 1)
START = date(2028, 1, 1)


def registry_of(*quantities: str) -> Registry:
    """A driver registry with one object per named quantity."""
    return Registry.model_validate(
        {
            "quantities": [
                {
                    "id": quantity,
                    "label": quantity,
                    "unit": "ед.",
                    "transform": "identity",
                    "regressors": ["gl.brent"],
                    "lags": 1,
                }
                for quantity in quantities
            ],
            "objects": [{"id": "sector", "class": "sector", "label": "Сектор"}],
            "drivers": [{"quantity": quantity, "object": "sector"} for quantity in quantities],
        }
    )


@dataclass
class FakeSatellite(SatelliteModel):
    """A satellite whose estimate and paths are its model id, letter by letter."""

    model_id: str
    version: str = "0.1.0"
    label: str = ""
    quantities: tuple = ()
    calls: list = field(default_factory=list)

    def estimate(self, context):
        parts, residuals = {}, {}
        for quantity in context.quantities:
            declared = next(q for q in context.registry.quantities if q.id == quantity)
            parts[quantity] = SatellitePart(
                meta={
                    "quantity": declared.model_dump(mode="json", by_alias=True),
                    "months": [m.isoformat() for m in context.panel.months],
                    "nodes": ["sector"],
                    "parent": [-1],
                    "classes": ["sector"],
                    "driver": [f"{quantity}:sector"],
                    "rows": [len(context.panel.months)],
                    "checks": {
                        "0": {
                            "satellite": 1.0,
                            "baseline": 2.0,
                            "forecasts": 3,
                            "served": True,
                            "reason": "",
                        },
                    },
                    "letter": self.model_id.split(".")[-1],
                },
                arrays={
                    "d": np.zeros((len(context.panel.months), 1)),
                    "extra": np.array([1.0 if self.model_id == "one.a" else 2.0]),
                },
            )
            residuals[f"{quantity}:sector"] = np.arange(10, dtype=float)
        return SatelliteEstimate(parts=parts, residuals=residuals)

    def load(self, parts):
        return {quantity: part.meta["letter"] for quantity, part in parts.items()}

    def simulate(
        self, model, macro, drivers, impacts, *, correlation=None, seed: int, stream: int
    ) -> DriverPaths:
        self.calls.append(
            (tuple(drivers), stream, tuple(sorted(item.target for item in impacts)))
        )
        paths = np.stack(
            [
                np.full(macro.paths.shape[:2], float(ord(model[driver.split(":")[0]]) + stream))
                for driver in drivers
            ],
            axis=-1,
        )
        flags: dict[str, frozenset[str]] = {driver: frozenset() for driver in drivers}
        for impact in impacts:
            k = drivers.index(impact.target)
            paths[:, impact.start : impact.end + 1, k] += impact.delta
            flags[impact.target] = frozenset({"impact"})
        return DriverPaths(ids=list(drivers), paths=paths, flags=flags)


def _context(registry, quantities):
    from mst.model_sdk.contracts import SatelliteEstimateContext

    panel = simulated.panel(24)
    return SatelliteEstimateContext(
        store=None,
        registry=registry,
        panel=panel,
        as_of=AS_OF,
        macro="macro-sha",
        quantities=quantities,
    )


def macro_simulation(months: int = 2, draws: int = 3) -> MacroSimulation:
    grid = [shift(START, k) for k in range(months + 1)]
    return MacroSimulation(
        ids=["gl.brent", "ru.key_rate"],
        months=grid,
        paths=np.arange(draws * (months + 1) * 2, dtype=float).reshape(
            draws, months + 1, 2
        ),
        history=np.zeros((draws, 2, 2)),
    )


def _bundle_of(tmp_path, models, registry):
    """The bundle of the models' estimates over their own quantities of ``registry``."""
    estimates = {}
    for model in models:
        own = tuple(
            quantity.id
            for quantity in registry.quantities
            if quantity.id in model.quantities
        )
        estimates[model.model_id] = model.estimate(_context(registry, own))
    meta, arrays = satellite_bundle.assemble(registry, "macro-sha", AS_OF, estimates)
    name = satellite_bundle.save(meta, arrays, tmp_path)
    installed = {model.model_id: model for model in models}
    return satellite_bundle.load(tmp_path, name, installed)


def test_a_bundle_holds_one_model_per_quantity_and_loads_back(tmp_path):
    registry = registry_of("price", "spread")
    one = FakeSatellite("one.a", quantities=("price",))
    two = FakeSatellite("two.b", quantities=("spread",))

    bundle = _bundle_of(tmp_path, [one, two], registry)

    assert bundle.quantity_model == {"price": "one.a", "spread": "two.b"}
    assert set(bundle.served()) == {"price:sector", "spread:sector"}
    assert bundle.models == {"one.a": {"price": "a"}, "two.b": {"spread": "b"}}
    assert bundle.correlated == ["price:sector", "spread:sector"]
    assert bundle.correlation.shape == (2, 2)
    assert np.allclose(np.diag(bundle.correlation), 1.0)


def test_simulate_dispatches_each_quantity_to_its_own_model(tmp_path):
    registry = registry_of("price", "spread")
    one = FakeSatellite("one.a", quantities=("price",))
    two = FakeSatellite("two.b", quantities=("spread",))
    bundle = _bundle_of(tmp_path, [one, two], registry)

    impacts = [AdditiveImpact(target="spread:sector", start=1, end=2, delta=5.0)]
    found = bundle.simulate(
        macro_simulation(), ["spread:sector", "price:sector"], impacts, seed=3
    )

    # Each model simulated its own drivers, with its own stream and its own impacts;
    # the streams follow the alphabetical order of the model ids.
    assert one.calls == [(("price:sector",), 0, ())]
    assert two.calls == [(("spread:sector",), 1, ("spread:sector",))]
    assert found.ids == ["spread:sector", "price:sector"]
    assert found.paths[0, 0, 0] == ord("b") + 1
    assert found.paths[0, 0, 1] == ord("a")
    assert np.all(found.paths[:, 1:3, 0] == ord("b") + 1 + 5)
    # The fake's paths sit far outside its all-zero history, so the range flag is set too.
    assert found.flags["spread:sector"] == frozenset({"impact", "outside_training_range"})


def test_a_driver_the_check_rejected_is_not_simulated(tmp_path):
    registry = registry_of("price")
    one = FakeSatellite("one.a", quantities=("price",))
    estimate = one.estimate(_context(registry, ("price",)))
    estimate.parts["price"].meta["checks"]["0"]["served"] = False
    estimate.parts["price"].meta["checks"]["0"]["reason"] = "not accurate"
    meta, arrays = satellite_bundle.assemble(registry, "macro-sha", AS_OF, {one.model_id: estimate})
    name = satellite_bundle.save(meta, arrays, tmp_path)
    bundle = satellite_bundle.load(tmp_path, name, {one.model_id: one})

    with pytest.raises(ValueError, match="price:sector is not served: not accurate"):
        bundle.simulate(macro_simulation(), ["price:sector"], seed=1)


def test_a_part_without_the_required_keys_is_refused(tmp_path):
    registry = registry_of("price")
    one = FakeSatellite("one.a", quantities=("price",))
    estimate = one.estimate(_context(registry, ("price",)))
    meta, arrays = satellite_bundle.assemble(registry, "macro-sha", AS_OF, {one.model_id: estimate})
    del meta["quantities"]["price"]["checks"]
    with pytest.raises(ValueError, match="the part of price must record"):
        satellite_bundle.save(meta, arrays, tmp_path)


def test_the_assignment_picks_the_model_of_every_quantity():
    from mst import models as models_pkg
    from mst import plugins

    installed = plugins.registry()
    assert "hier.gibbs" in installed.satellites

    chosen, unserved = models_pkg.assignment(
        ["share_price", "bond_spread"], {"bond_spread": "hier.gibbs"}
    )
    assert chosen == {"share_price": "hier.gibbs", "bond_spread": "hier.gibbs"}
    assert unserved == []

    with pytest.raises(ValueError, match="no installed satellite model"):
        models_pkg.assignment(["share_price"], {"share_price": "gone.v1"})


def test_a_quantity_no_model_serves_is_recorded_not_estimated():
    from mst import models as models_pkg

    chosen, unserved = models_pkg.assignment(["overdue_share", "nowhere"])
    assert chosen == {"overdue_share": "hier.gibbs"}
    assert unserved == ["nowhere"]
