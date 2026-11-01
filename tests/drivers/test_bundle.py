"""The satellite bundle: one version, one model per quantity, loaded back and simulated."""

from dataclasses import dataclass, field, replace
from datetime import date

import numpy as np
import pytest

from mst.drivers import bundle as satellite_bundle
from mst.drivers.registry import Registry
from mst.macro.panel import shift
from mst.model_sdk import PART_META_REQUIRED, AdditiveImpact, DriverPaths, SatelliteModel
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
    shocks_seen: list = field(default_factory=list)

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
        self,
        model,
        macro,
        drivers,
        impacts,
        *,
        correlation=None,
        shocks: np.ndarray,
        seed: int,
        stream: int,
    ) -> DriverPaths:
        self.calls.append(
            (tuple(drivers), stream, tuple(sorted(item.target for item in impacts)))
        )
        self.shocks_seen.append(shocks)
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


class MinimalSatellite(FakeSatellite):
    """A model whose part carries exactly the keys the published contract requires."""

    def estimate(self, context):
        parts, residuals = {}, {}
        for quantity in context.quantities:
            parts[quantity] = SatellitePart(
                meta={
                    "months": [m.isoformat() for m in context.panel.months],
                    "nodes": ["sector"],
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
                },
                arrays={"d": np.zeros((len(context.panel.months), 1))},
            )
            residuals[f"{quantity}:sector"] = np.arange(10, dtype=float)
        return SatelliteEstimate(parts=parts, residuals=residuals)

    def load(self, parts):
        return {quantity: part.meta for quantity, part in parts.items()}


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


def test_a_part_that_names_no_model_is_refused(tmp_path):
    from mst import versions

    registry = registry_of("price")
    one = FakeSatellite("one.a", quantities=("price",))
    estimate = one.estimate(_context(registry, ("price",)))
    meta, arrays = satellite_bundle.assemble(
        registry, "macro-sha", AS_OF, {"one.a": estimate}
    )
    anonymous = {
        **meta,
        "quantities": {
            quantity: {
                key: value for key, value in part.items() if key not in ("model",)
            }
            for quantity, part in meta["quantities"].items()
        },
    }
    name = versions.save(tmp_path / "satellites", anonymous, arrays)

    with pytest.raises(KeyError, match="the part of price names no model"):
        satellite_bundle.load(tmp_path, name, {"one.a": one})


def test_simulate_dispatches_each_quantity_to_its_own_model(tmp_path):
    registry = registry_of("price", "spread")
    one = FakeSatellite("one.a", quantities=("price",))
    two = FakeSatellite("two.b", quantities=("spread",))
    bundle = _bundle_of(tmp_path, [one, two], registry)

    impacts = [AdditiveImpact(target="spread:sector", start=1, end=2, delta=5.0)]
    found = bundle.simulate(
        macro_simulation(), ["spread:sector", "price:sector"], impacts, seed=3
    )

    # Each model simulated its own drivers, with its own stream and its own impacts; the
    # streams follow the alphabetical order of the model ids and start at 1, above the
    # platform's own stream 0.
    assert one.calls == [(("price:sector",), 1, ())]
    assert two.calls == [(("spread:sector",), 2, ("spread:sector",))]
    # The platform draws the joint shocks of every model, one column per model here.
    rows = 2 + len(simulated.panel(24).months) - 1
    assert one.shocks_seen[0].shape == (3, rows, 1)
    assert two.shocks_seen[0].shape == (3, rows, 1)
    assert found.ids == ["spread:sector", "price:sector"]
    assert found.paths[0, 0, 0] == ord("b") + 2
    assert found.paths[0, 0, 1] == ord("a") + 1
    assert np.all(found.paths[:, 1:3, 0] == ord("b") + 2 + 5)
    # The fake records no training bounds, so no range check runs and no flag is set.
    assert found.flags["spread:sector"] == frozenset({"impact"})


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


def test_a_minimal_part_round_trips_through_assemble_and_save(tmp_path):
    # Exactly the published keys, no model's own extras: the bundle must work all the same.
    registry = registry_of("price")
    model = MinimalSatellite("one.a", quantities=("price",))
    estimate = model.estimate(_context(registry, ("price",)))
    assert set(estimate.parts["price"].meta) == set(PART_META_REQUIRED) - {"model"}

    meta, arrays = satellite_bundle.assemble(
        registry, "macro-sha", AS_OF, {model.model_id: estimate}
    )
    name = satellite_bundle.save(meta, arrays, tmp_path)
    bundle = satellite_bundle.load(tmp_path, name, {model.model_id: model})

    assert bundle.quantity_model == {"price": "one.a"}
    assert bundle.parts["price"]["model"] == "one.a"
    assert bundle.served() == ["price:sector"]


def test_a_part_without_rows_is_refused(tmp_path):
    registry = registry_of("price")
    model = MinimalSatellite("one.a", quantities=("price",))
    estimate = model.estimate(_context(registry, ("price",)))
    del estimate.parts["price"].meta["rows"]
    with pytest.raises(ValueError, match="the part of price must record .*: no rows"):
        satellite_bundle.assemble(registry, "macro-sha", AS_OF, {model.model_id: estimate})

    estimate = model.estimate(_context(registry, ("price",)))
    meta, arrays = satellite_bundle.assemble(
        registry, "macro-sha", AS_OF, {model.model_id: estimate}
    )
    del meta["quantities"]["price"]["rows"]
    with pytest.raises(ValueError, match="the part of price must record .*: no rows"):
        satellite_bundle.save(meta, arrays, tmp_path)


def test_the_model_that_estimated_a_part_cannot_be_spoofed(tmp_path):
    registry = registry_of("price")
    model = MinimalSatellite("one.a", quantities=("price",))

    spoofed = model.estimate(_context(registry, ("price",)))
    spoofed.parts["price"].meta["model"] = "someone.else"
    with pytest.raises(ValueError, match="names model 'someone.else', but 'one.a'"):
        satellite_bundle.assemble(
            registry, "macro-sha", AS_OF, {model.model_id: spoofed}
        )

    # A part that repeats its own id is accepted, and the platform records that same id.
    honest = model.estimate(_context(registry, ("price",)))
    honest.parts["price"].meta["model"] = "one.a"
    meta, arrays = satellite_bundle.assemble(
        registry, "macro-sha", AS_OF, {model.model_id: honest}
    )
    assert meta["quantities"]["price"]["model"] == "one.a"


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


class TrainingSatellite(FakeSatellite):
    """A satellite with a narrow recorded training window over a wide history column.

    The column runs from -500 to 500, while the node is recorded as trained on ``low .. high``:
    the platform must check paths against the recorded window, not against the whole column.
    """

    def __init__(self, *, low=1.0, high=2.0, value=100.0):
        super().__init__("range.a", quantities=("price",))
        self.low, self.high, self.value = low, high, value

    def estimate(self, context):
        estimate = super().estimate(context)
        part = estimate.parts["price"]
        months = len(context.panel.months)
        part.arrays["d"] = np.concatenate(
            [[-500.0], np.full(months - 2, 50.0), [500.0]]
        )[:, None]
        part.meta["train_low"] = [self.low]
        part.meta["train_high"] = [self.high]
        return estimate

    def simulate(
        self, model, macro, drivers, impacts, *, correlation=None, shocks=None, seed, stream
    ):
        paths = np.full(
            (macro.paths.shape[0], macro.paths.shape[1], len(drivers)), self.value
        )
        return DriverPaths(
            ids=list(drivers),
            paths=paths,
            flags={driver: frozenset() for driver in drivers},
        )


def _range_flags(tmp_path, model) -> frozenset[str]:
    """The flags of the only driver of a bundle whose model simulates a constant path."""
    registry = registry_of("price")
    bundle = _bundle_of(tmp_path, [model], registry)
    found = bundle.simulate(macro_simulation(months=4), ["price:sector"], seed=5)
    return found.flags["price:sector"]


def test_a_path_above_the_recorded_window_is_flagged(tmp_path):
    # Above the window the node was estimated on, still inside the whole history column.
    found = _range_flags(tmp_path, TrainingSatellite(value=100.0))
    assert found == frozenset({"outside_training_range"})


def test_a_path_above_the_whole_column_is_flagged_too(tmp_path):
    found = _range_flags(tmp_path, TrainingSatellite(value=1000.0))
    assert found == frozenset({"outside_training_range"})


def test_a_path_inside_the_recorded_window_is_not_flagged(tmp_path):
    found = _range_flags(tmp_path, TrainingSatellite(value=1.5))
    assert found == frozenset()


def two_model_registry() -> Registry:
    """Two quantities of one tree: ``qa`` has two objects, ``qb`` one."""
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
                for quantity in ("qa", "qb")
            ],
            "objects": [
                {"id": "top", "class": "sector", "label": "Верх"},
                {"id": "x", "class": "sector", "label": "X", "parent": "top"},
                {"id": "y", "class": "sector", "label": "Y", "parent": "top"},
            ],
            "drivers": [
                {"quantity": "qa", "object": "x"},
                {"quantity": "qa", "object": "y"},
                {"quantity": "qb", "object": "top"},
            ],
        }
    )


def correlated_residuals(months: int, matrix: np.ndarray, seed: int = 3) -> list[np.ndarray]:
    """``months``-long residual series whose sample correlation is exactly ``matrix``."""
    rng = np.random.default_rng(seed)
    centred = rng.standard_normal((months, matrix.shape[0]))
    centred -= centred.mean(axis=0)
    basis, _ = np.linalg.qr(centred)
    values = basis * np.sqrt(months) @ np.linalg.cholesky(matrix).T
    return [values[:, k] for k in range(matrix.shape[1])]


class ShockSatellite(SatelliteModel):
    """A satellite whose paths are the platform's shocks, month by month.

    ``received`` records what the platform passed, so a test can check the shape and the
    columns every model got.
    """

    version = "0.1.0"
    label = ""

    def __init__(self, model_id, quantities, drivers, residuals):
        self.model_id = model_id
        self.quantities = quantities
        self.drivers = drivers
        self.residuals = residuals
        self.received = []

    def estimate(self, context):
        parts = {}
        for quantity in context.quantities:
            declared = next(q for q in context.registry.quantities if q.id == quantity)
            own = [driver for driver in self.drivers if driver.split(":")[0] == quantity]
            months = len(context.panel.months)
            parts[quantity] = SatellitePart(
                meta={
                    "quantity": declared.model_dump(mode="json", by_alias=True),
                    "months": [m.isoformat() for m in context.panel.months],
                    "nodes": [driver.split(":", 1)[1] for driver in own],
                    "parent": [-1] * len(own),
                    "classes": ["sector"] * len(own),
                    "driver": own,
                    "rows": [months] * len(own),
                    "checks": {
                        str(node): {
                            "satellite": 1.0,
                            "baseline": 2.0,
                            "forecasts": 3,
                            "served": True,
                            "reason": "",
                        }
                        for node in range(len(own))
                    },
                },
                arrays={"d": np.zeros((months, len(own)))},
            )
        return SatelliteEstimate(parts=parts, residuals=dict(self.residuals))

    def load(self, parts):
        return {quantity: part.meta["driver"] for quantity, part in parts.items()}

    def simulate(
        self, model, macro, drivers, impacts, *, correlation=None, shocks=None, seed, stream
    ) -> DriverPaths:
        self.received.append(shocks)
        draws, horizon = macro.paths.shape[0], macro.paths.shape[1] - 1
        paths = np.zeros((draws, horizon + 1, len(drivers)))
        paths[:, 1:, :] = shocks[:, :horizon, :]
        return DriverPaths(
            ids=list(drivers),
            paths=paths,
            flags={driver: frozenset() for driver in drivers},
        )


def test_the_platform_draws_the_cross_model_shocks_once(tmp_path):
    # Three drivers of two models with a known correlation matrix: the platform must draw the
    # joint shocks from the full matrix and hand every model its own slice.
    matrix = np.array(
        [
            [1.0, 0.6, 0.3],
            [0.6, 1.0, 0.2],
            [0.3, 0.2, 1.0],
        ]
    )
    residuals = dict(zip(["qa:x", "qa:y", "qb:top"], correlated_residuals(24, matrix), strict=True))
    one = ShockSatellite("one.a", ("qa",), ("qa:x", "qa:y"), residuals)
    two = ShockSatellite("two.b", ("qb",), ("qb:top",), residuals)
    registry = two_model_registry()
    bundle = _bundle_of(tmp_path, [one, two], registry)
    assert bundle.correlated == ["qa:x", "qa:y", "qb:top"]
    assert np.allclose(bundle.correlation, matrix)

    found = bundle.simulate(
        macro_simulation(months=6, draws=10_000),
        ["qa:x", "qa:y", "qb:top"],
        seed=7,
    )

    # The platform gives each model the number of rows the longest part needs and only the
    # columns of that model's own correlated drivers.
    rows = 6 + 24 - 1
    assert one.received[0].shape == (10_000, rows, 2)
    assert two.received[0].shape == (10_000, rows, 1)

    observed = np.corrcoef([found.paths[:, 1:, k].ravel() for k in range(3)])
    # The measured correlation reaches the joint paths, within and across the models; a model
    # that drew its own residuals would leave the cross-model entries at zero.
    assert np.allclose(observed, matrix, atol=0.05)
    assert observed[0, 1] == pytest.approx(0.6, abs=0.05)
    assert observed[0, 2] == pytest.approx(0.3, abs=0.05)
    assert observed[1, 2] == pytest.approx(0.2, abs=0.05)


def test_a_one_model_bundle_gets_the_platform_shocks_too(tmp_path):
    residuals = dict(zip(["qa:x"], correlated_residuals(24, np.eye(1)), strict=True))
    model = ShockSatellite("one.a", ("qa",), ("qa:x",), residuals)
    bundle = _bundle_of(tmp_path, [model], two_model_registry())
    macro = macro_simulation(months=4, draws=5)

    found = bundle.simulate(macro, ["qa:x"], seed=7)

    # The platform draws the joint shocks whatever the number of models, so the measured
    # correlation reaches the paths of a one-model bundle the same way.
    assert model.received[0] is not None
    assert model.received[0].shape == (5, 4 + 24 - 1, 1)
    assert np.array_equal(found.paths[:, 1:, 0], model.received[0][:, :4, 0])


def test_a_correlation_matrix_that_is_not_positive_definite_is_refused(tmp_path):
    residuals = dict(zip(["qa:x", "qb:top"], correlated_residuals(24, np.eye(2)), strict=True))
    one = ShockSatellite("one.a", ("qa",), ("qa:x",), residuals)
    two = ShockSatellite("two.b", ("qb",), ("qb:top",), residuals)
    bundle = _bundle_of(tmp_path, [one, two], two_model_registry())
    # assemble clips the eigenvalues, so a matrix edited outside it can still be indefinite;
    # the platform must name the problem instead of failing inside the Cholesky.
    broken = replace(bundle, correlation=np.full((2, 2), 0.9))
    with pytest.raises(ValueError, match="not positive definite"):
        broken.simulate(macro_simulation(), ["qa:x", "qb:top"], seed=1)


def test_a_quantity_no_model_serves_is_recorded_not_estimated():
    from mst import models as models_pkg

    chosen, unserved = models_pkg.assignment(["overdue_share", "nowhere"])
    assert chosen == {"overdue_share": "hier.gibbs"}
    assert unserved == ["nowhere"]
