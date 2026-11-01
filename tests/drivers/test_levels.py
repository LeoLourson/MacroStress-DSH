"""The recursion over the levels: a chain of three, and the reference model of the second.

The checks follow the plan: a registry whose quantity reads its own level or a level above it
is refused; a simulated chain of three levels with known coefficients is estimated and the
second-level regression recovers them, with the conditional variance of ``x^2`` given ``x^1``
at the residual scale; an impact on the first level reaches the second through ``G^2`` while
an impact on the second leaves the first untouched; the levels draw independent shocks and a
level's models share the correlation measured for it; and a bundle of one level reproduces
the paths of its model called directly.
"""

import json
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from mst.data.store import Observation, SeriesStore
from mst.drivers import bundle as satellite_bundle
from mst.drivers.registry import Registry, merge
from mst.macro import version as macro_version
from mst.macro.panel import shift
from mst.model_sdk.contracts import (
    DriverPaths,
    MacroEstimate,
    MacroEstimateContext,
    MacroModel,
    MacroSimulation,
    ScenarioPlan,
    UpperPaths,
)
from tests.drivers import simulated as drivers_simulated

CATALOG = Path(__file__).with_name("level2_catalog.json")
"""How the objects and drivers of the level-2 quantity arrive: a catalog file, not the core
registry — a level above the first is a deliberate extension, never a default."""

MONTHS = 120
AS_OF = date(2020, 1, 1)
LEVEL2_BETA = np.array([0.4, 1.25])
"""The known intercept and slope of the exposure on the price of the level above."""
LEVEL2_SIGMA = 0.5
"""The known residual scale of the exposure."""


def level_registry() -> Registry:
    """The registry of the chain: ``price`` of level 1, ``exposure`` of level 2 on it."""
    raw = {
        "quantities": [
            {
                "id": "price",
                "label": "Цена",
                "unit": "руб.",
                "transform": "identity",
                "regressors": ["gl.brent"],
                "lags": 1,
            },
            {
                "id": "exposure",
                "label": "Экспозиция",
                "unit": "руб.",
                "transform": "identity",
                "level": 2,
                "regressors": ["price"],
                "lags": 0,
            },
        ],
        "objects": [
            {"id": "sector", "class": "sector", "label": "Сектор"},
            {"id": "alpha", "class": "company", "label": "Alpha", "parent": "sector"},
            {"id": "beta", "class": "company", "label": "Beta", "parent": "sector"},
        ],
        "drivers": [
            {"quantity": "price", "object": "sector", "series": "drv.price.sector"},
            {"quantity": "price", "object": "alpha", "series": "drv.price.alpha"},
            {"quantity": "price", "object": "beta", "series": "drv.price.beta"},
        ],
    }
    return Registry.model_validate(merge(raw, json.loads(CATALOG.read_text(encoding="utf-8"))))


@dataclass
class Chain:
    """The simulated chain: the panel, the series store and the driver registry."""

    registry: Registry
    panel: object
    store: SeriesStore
    price: dict[str, np.ndarray]
    """The true price paths of every object, as the store was written from."""


def chain(root, seed: int = 5) -> Chain:
    """The panel, the price series of the known hierarchical model and the exposure on it."""
    registry = level_registry()
    panel = drivers_simulated.panel(MONTHS)
    store = SeriesStore(root)
    rng = np.random.default_rng(seed)
    x = panel.y[:, 0]
    price = {"sector": drivers_simulated.path(drivers_simulated.SECTOR, x, rng)}
    for name, gamma in drivers_simulated.COMPANIES.items():
        if name in ("alpha", "beta"):
            price[name] = drivers_simulated.path(gamma, x, rng)
    for id_ in ("gl.brent", "ru.key_rate"):
        column = panel.ids.index(id_)
        store.write(
            id_,
            [
                Observation(month, float(value), month, False)
                for month, value in zip(panel.months, panel.y[:, column], strict=True)
            ],
        )
    for name, values in price.items():
        store.write(
            f"drv.price.{name}",
            [
                Observation(month, float(value), month, False)
                for month, value in zip(panel.months, values, strict=True)
            ],
        )
    # The exposure of every level-2 object rides the price of the sector above it: the
    # nearest object at or above sber and gazp that carries a price driver is the sector.
    sector = price["sector"]
    for node in ("sber", "gazp"):
        values = LEVEL2_BETA[0] + LEVEL2_BETA[1] * sector + LEVEL2_SIGMA * rng.standard_normal(
            len(sector)
        )
        store.write(
            f"drv.exposure.{node}",
            [
                Observation(month, float(value), month, False)
                for month, value in zip(panel.months, values, strict=True)
            ],
        )
    return Chain(registry=registry, panel=panel, store=store, price=price)


class WalkMacro(MacroModel):
    """A macro model of the test: a random walk that carries the panel's last months as the
    history the first level reads."""

    model_id = "levels.macro"
    version = "0.1.0"
    label = "Случайное блуждание уровней"
    variables = ("gl.brent", "ru.key_rate",)

    def estimate(self, context: MacroEstimateContext) -> MacroEstimate:
        months, columns = [], []
        for id_ in self.variables:
            rows = list(context.store.as_of(id_, context.as_of))
            months = [row.month for row in rows]
            columns.append([row.value for row in rows])
        y = np.array(columns, dtype=float).T
        return MacroEstimate(
            meta={
                "as_of": context.as_of.isoformat(),
                "ids": list(self.variables),
                "n_global": 1,
                "months": [m.isoformat() for m in months],
                "shocks": ["levels.demand"],
                "diagnostics": {},
            },
            arrays={
                "panel_y": y,
                "panel_approximate": np.zeros(y.shape, dtype=bool),
            },
        )

    def load(self, meta, arrays):
        return meta | {"panel_y": arrays["panel_y"]}

    def simulate(
        self, model, plan: ScenarioPlan, *, draws: int, seed: int, inner: int | None = None
    ):
        inner = plan.inner if inner is None else inner
        rng = np.random.default_rng([seed, plan.horizon])
        last = model["panel_y"][-1]
        p = 3
        steps = rng.normal(
            0, 0.3, size=(draws * inner, plan.horizon, len(model["ids"]))
        )
        paths = np.concatenate(
            [np.tile(last, (draws * inner, 1, 1)), last + np.cumsum(steps, axis=1)],
            axis=1,
        )
        last = date.fromisoformat(model["months"][-1])
        months = [shift(last, k) for k in range(plan.horizon + 1)]
        group = (
            np.arange(draws) if inner == 1 else np.repeat(np.arange(draws), inner)
        )
        return MacroSimulation(
            ids=list(model["ids"]),
            months=months,
            paths=paths,
            history=np.repeat(model["panel_y"][None, -p:], draws * inner, axis=0),
            group=group,
        )


@pytest.fixture(scope="module")
def estimated(tmp_path_factory):
    """The chain estimated as one bundle: the hierarchical price and the reference exposure."""
    from mst import models as models_pkg

    root = tmp_path_factory.mktemp("models")
    data = chain(tmp_path_factory.mktemp("series"))
    macro = WalkMacro()
    estimate = macro.estimate(
        MacroEstimateContext(store=data.store, registry=None, as_of=AS_OF)
    )
    name = macro_version.save("levels.macro", estimate.meta, estimate.arrays, root)
    found, _ = models_pkg.estimate_satellites(
        name,
        store=data.store.root,
        models=root,
        registry=data.registry,
        overrides={
            "price": "hier.gibbs",
            "exposure": "reflevel2.regression",
        },
    )
    installed = models_pkg.satellites()
    bundle = satellite_bundle.load(root, found, installed)
    return data, macro, name, bundle, root


def test_a_quantity_reading_its_own_or_an_upper_level_is_refused():
    def registry_of(**quantity):
        base = {
            "id": "exposure",
            "label": "Экспозиция",
            "unit": "руб.",
            "transform": "identity",
            "level": 2,
            "regressors": ["price"],
            "lags": 0,
        }
        base.update(quantity)
        return Registry.model_validate(
            {
                "quantities": [
                    {
                        "id": "price",
                        "label": "Цена",
                        "unit": "руб.",
                        "transform": "identity",
                        "regressors": ["gl.brent"],
                        "lags": 1,
                    },
                    base,
                ],
            }
        )

    with pytest.raises(ValueError, match="reads its own level"):
        registry_of(regressors=["exposure"])
    with pytest.raises(ValueError, match="a level above it"):
        base = {
            "id": "third",
            "label": "Третий",
            "unit": "руб.",
            "transform": "identity",
            "level": 3,
            "regressors": ["price"],
            "lags": 0,
        }
        Registry.model_validate(
            {
                "quantities": [
                    {
                        "id": "price",
                        "label": "Цена",
                        "unit": "руб.",
                        "transform": "identity",
                        "regressors": ["gl.brent"],
                        "lags": 1,
                    },
                    {
                        "id": "exposure",
                        "label": "Экспозиция",
                        "unit": "руб.",
                        "transform": "identity",
                        "level": 2,
                        "regressors": ["price"],
                        "lags": 0,
                    },
                    {**base, "regressors": ["exposure", "price"], "level": 2},
                ],
            }
        )
    # A quantity of the third level may read both levels below it.
    Registry.model_validate(
        {
            "quantities": [
                {
                    "id": "price",
                    "label": "Цена",
                    "unit": "руб.",
                    "transform": "identity",
                    "regressors": ["gl.brent"],
                    "lags": 1,
                },
                {
                    "id": "exposure",
                    "label": "Экспозиция",
                    "unit": "руб.",
                    "transform": "identity",
                    "level": 2,
                    "regressors": ["price"],
                    "lags": 0,
                },
                base,
            ],
        }
    )


def test_the_chain_estimation_recovers_the_known_coefficients(estimated):
    data, macro, name, bundle, root = estimated
    fit = bundle.models["reflevel2.regression"].fits["exposure"]
    column = fit.nodes.index("sber")
    # The posterior of the reference regression centres on the known coefficients; the
    # conditional variance of x^2 given x^1 is the known residual scale squared.
    beta = fit.beta[:, column, :2]
    # The posterior mean is the least-squares fit of one sample: three standard errors of
    # the intercept (the residual scale over a hundred and twenty months) and a tight slope.
    assert beta.mean(axis=0)[0] == pytest.approx(LEVEL2_BETA[0], abs=0.2)
    assert beta.mean(axis=0)[1] == pytest.approx(LEVEL2_BETA[1], abs=0.05)
    sigma = float(np.mean(fit.sigma[:, column]))
    assert sigma == pytest.approx(LEVEL2_SIGMA, rel=0.15)
    assert "exposure:sber" in bundle.served()


def test_the_conditional_variance_of_the_second_level_is_the_residual_scale(estimated):
    data, macro, name, bundle, root = estimated
    macro_version_loaded = macro_version.load(root, name, {"levels.macro": macro})
    simulation = macro.simulate(
        macro_version_loaded.model, ScenarioPlan(horizon=6), draws=400, seed=3
    )
    upper = UpperPaths(
        macro=simulation,
        levels={
            1: DriverPaths(
                ids=["price:sector"],
                paths=np.repeat(
                    data.price["sector"][None, -1:], 400, axis=0
                ).repeat(7, axis=1)[:, :, None],
                flags={"price:sector": frozenset()},
            )
        },
    )
    found = bundle.plugins["reflevel2.regression"].simulate(
        bundle.models["reflevel2.regression"],
        upper,
        ["exposure:sber"],
        [],
        correlation=([r"[price:sector]"], np.eye(1)),
        shocks=np.random.default_rng(4).standard_normal((400, 6, 1)),
        seed=5,
        stream=1,
    )
    # Given the paths of the first level, the second varies at its residual scale: the
    # regression passes the level above through and adds nothing but its own noise.
    spread = found.paths[:, 1:, 0].std(axis=0).mean()
    assert spread == pytest.approx(LEVEL2_SIGMA, rel=0.15)


def test_an_impact_on_the_first_level_reaches_the_second_and_not_back(estimated):
    data, macro, name, bundle, root = estimated
    macro_version_loaded = macro_version.load(root, name, {"levels.macro": macro})
    simulation = macro.simulate(
        macro_version_loaded.model, ScenarioPlan(horizon=4), draws=5, seed=7
    )
    drivers = ["price:sector", "exposure:sber"]
    plain = bundle.simulate(simulation, drivers, (), seed=7)
    from mst.model_sdk.contracts import AdditiveImpact

    lower = bundle.simulate(
        simulation,
        drivers,
        [AdditiveImpact(target="price:sector", start=1, end=4, delta=3.0)],
        seed=7,
    )
    upper = bundle.simulate(
        simulation,
        drivers,
        [AdditiveImpact(target="exposure:sber", start=1, end=4, delta=3.0)],
        seed=7,
    )
    # o^1 moves the price, and through G^2 the exposure reads the moved price; o^2 moves the
    # exposure alone and leaves the price untouched.
    price_column, exposure_column = 0, 1
    assert np.allclose(lower.paths[:, 1:, price_column] - plain.paths[:, 1:, price_column], 3.0)
    exposure_gap = lower.paths[:, 1:, exposure_column] - plain.paths[:, 1:, exposure_column]
    assert abs(exposure_gap.mean()) > 1.0  # the slope carries most of the price move
    assert np.allclose(upper.paths[:, :, price_column], plain.paths[:, :, price_column])
    moved = upper.paths[:, 1:, exposure_column] - plain.paths[:, 1:, exposure_column]
    assert np.allclose(moved, 3.0)


def test_levels_draw_independent_shocks_a_level_shares_its_measured_correlation(estimated):
    data, macro, name, bundle, root = estimated
    macro_version_loaded = macro_version.load(root, name, {"levels.macro": macro})
    simulation = macro.simulate(
        macro_version_loaded.model, ScenarioPlan(horizon=8), draws=4000, seed=11
    )
    # The levels are linked through G^2 — the exposure paths follow the price paths — but
    # their shocks are independent: the platform draws one stream per level. Inside a level
    # the measured correlation reaches the joint shocks through the Cholesky factor.
    first = bundle._platform_shocks(
        simulation,
        ["price:sector", "price:alpha"],
        {"hier.gibbs": ["price:sector", "price:alpha"]},
        1,
        8,
        11,
    )["hier.gibbs"]
    second = bundle._platform_shocks(
        simulation,
        ["exposure:sber", "exposure:gazp"],
        {"reflevel2.regression": ["exposure:sber", "exposure:gazp"]},
        2,
        8,
        11,
    )["reflevel2.regression"]
    across = np.corrcoef(first.ravel(), second.ravel())[0, 1]
    assert abs(across) < 0.03  # two independent streams of thirty-two thousand draws

    correlated = [
        driver for driver in ("exposure:sber", "exposure:gazp") if driver in bundle.correlated
    ]
    positions = {driver: k for k, driver in enumerate(bundle.correlated)}
    matrix = np.eye(len(bundle.correlated))
    for one in correlated:
        for other in correlated:
            if one != other:
                matrix[positions[one], positions[other]] = 0.6
    drawn = replace(bundle, correlation=matrix)
    shared = drawn._platform_shocks(
        simulation,
        ["exposure:sber", "exposure:gazp"],
        {"reflevel2.regression": ["exposure:sber", "exposure:gazp"]},
        2,
        8,
        11,
    )["reflevel2.regression"]
    if len(correlated) == 2:
        within = np.corrcoef(shared[:, :, 0].ravel(), shared[:, :, 1].ravel())[0, 1]
        assert within == pytest.approx(0.6, abs=0.03)


def test_a_bundle_of_one_level_matches_its_model_called_directly(estimated):
    data, macro, name, bundle, root = estimated
    macro_version_loaded = macro_version.load(root, name, {"levels.macro": macro})
    simulation = macro.simulate(
        macro_version_loaded.model, ScenarioPlan(horizon=4), draws=6, seed=13
    )
    drivers = ["price:sector", "price:alpha"]
    found = bundle.simulate(simulation, drivers, (), seed=13)
    model = bundle.models["hier.gibbs"]
    platform = bundle._platform_shocks(
        simulation, drivers, {"hier.gibbs": drivers}, 1, 4, 13
    )
    direct = bundle.plugins["hier.gibbs"].simulate(
        model,
        UpperPaths(macro=simulation, levels={}),
        drivers,
        [],
        correlation=(bundle.correlated, bundle.correlation),
        shocks=platform["hier.gibbs"],
        seed=13,
        stream=1,
    )
    assert np.array_equal(found.paths, direct.paths)


def bundle_name_of(root):
    """The hash of the satellite bundle the fixture estimated — the newest one."""
    from mst.drivers import bundle as satellite_bundle

    return satellite_bundle.catalogue(root)[0]["version"]


class TestDriverConditions:
    """The conditions on the quantities of the levels above, carried by the overlay."""

    def _spec(self, data, root, name, **changes):
        from mst.scenario.spec import Spec

        values = {
            "narrative": "экспозиция",
            "horizon": 4,
            "macro": name,
            "satellites": bundle_name_of(root),
            "approved_by": "И. Иванов",
        }
        values.update(changes)
        return Spec.model_validate(values)

    def test_a_condition_on_the_second_level_keeps_only_the_survivors(self, estimated):
        from mst.scenario.build import build

        data, macro, name, bundle, root = estimated
        macro = macro_version.load(root, name, {"levels.macro": macro})
        # The exposure of sber in month 1 is asked into a band the regression reaches for
        # some draws of the causes and not for others: the survivors carry the run, the
        # macro paths and the driver paths resampled together.
        plain = build(
            self._spec(data, root, name, inner=4), macro, bundle, draws=6, seed=3
        )
        column = plain.drivers.paths[:, 1, plain.drivers.ids.index("exposure:sber")]
        low, high = float(np.percentile(column, 25)), float(np.percentile(column, 75))
        constrained = build(
            self._spec(
                data,
                root,
                name,
                inner=4,
                constraints=[
                    {
                        "terms": [
                            {"variable": "exposure:sber", "month": 1, "weight": 1.0}
                        ],
                        "low": low,
                        "high": high,
                        "basis": "assumption",
                    }
                ],
            ),
            macro,
            bundle,
            draws=6,
            seed=3,
        )
        moved = constrained.drivers.paths[
            :, 1, constrained.drivers.ids.index("exposure:sber")
        ]
        assert np.all((moved >= low - 1e-9) & (moved <= high + 1e-9))
        # The price carries the level above its own share and the level split names it.
        assert constrained.dist is not None
        assert constrained.dist.by_level and 2 in constrained.dist.by_level
        assert constrained.dist.by_level[2] > 0

    def test_a_group_without_a_survivor_contradicts_the_models(self, estimated):
        from mst.model_sdk.dist import ContradictsModels
        from mst.scenario.build import build

        data, macro, name, bundle, root = estimated
        macro = macro_version.load(root, name, {"levels.macro": macro})
        with pytest.raises(ContradictsModels, match="contradicts the models"):
            build(
                self._spec(
                    data,
                    root,
                    name,
                    inner=4,
                    constraints=[
                        {
                            "terms": [
                                {"variable": "exposure:sber", "month": 1, "weight": 1.0}
                            ],
                            "low": 1e6,
                            "high": 1e7,
                            "basis": "assumption",
                        }
                    ],
                ),
                macro,
                bundle,
                draws=4,
                seed=3,
            )

    def test_the_price_of_driver_conditions_is_their_joint_share_once(self, estimated):
        from mst.scenario.build import build

        data, macro, name, bundle, root = estimated
        macro = macro_version.load(root, name, {"levels.macro": macro})
        plain = build(self._spec(data, root, name, inner=8), macro, bundle, draws=6, seed=3)
        column = plain.drivers.paths[:, 1, plain.drivers.ids.index("exposure:sber")]
        condition = {
            "terms": [{"variable": "exposure:sber", "month": 1, "weight": 1.0}],
            "low": float(np.percentile(column, 10)),
            "high": float(np.percentile(column, 90)),
            "basis": "assumption",
        }

        def priced(*conditions):
            spec = self._spec(data, root, name, inner=8, constraints=list(conditions))
            return build(spec, macro, bundle, draws=6, seed=3).dist

        single, twice = priced(condition), priced(condition, condition)
        # The price enters the total, and a repeated condition removes no further path.
        assert single.distance > 0
        assert single.distance == pytest.approx(single.by_level[2])
        assert twice.distance == pytest.approx(single.distance)

    def test_a_driver_expectation_tilts_the_survivors_and_is_priced(self, estimated):
        from mst.scenario.build import build

        data, macro, name, bundle, root = estimated
        macro = macro_version.load(root, name, {"levels.macro": macro})
        plain = build(self._spec(data, root, name, inner=16), macro, bundle, draws=8, seed=3)
        k = plain.drivers.ids.index("exposure:sber")
        column = plain.drivers.paths[:, 1, k]
        term = {"variable": "exposure:sber", "month": 1, "weight": 1.0}
        target = float(column.mean() + 0.5 * column.std())
        found = build(
            self._spec(
                data,
                root,
                name,
                inner=16,
                constraints=[
                    {
                        "terms": [term],
                        "low": float(np.percentile(column, 5)),
                        "high": float(np.percentile(column, 95)),
                        "basis": "assumption",
                    }
                ],
                expectations=[{"terms": [term], "low": target, "basis": "assumption"}],
            ),
            macro,
            bundle,
            draws=8,
            seed=3,
        )
        moved = found.drivers.paths[:, 1, k]
        assert moved.mean() == pytest.approx(target, abs=0.5 * column.std())
        assert column.min() <= moved.min()
        assert found.dist.distance > 0 and found.dist.by_level[2] == pytest.approx(
            found.dist.distance
        )

    def test_a_driver_condition_needs_inner_draws(self, estimated):
        from mst.scenario.build import build

        data, macro, name, bundle, root = estimated
        macro = macro_version.load(root, name, {"levels.macro": macro})
        with pytest.raises(ValueError, match="inner >= 2"):
            build(
                self._spec(
                    data,
                    root,
                    name,
                    constraints=[
                        {
                            "terms": [
                                {"variable": "exposure:sber", "month": 1, "weight": 1.0}
                            ],
                            "low": 0.0,
                            "high": 1.0,
                            "basis": "assumption",
                        }
                    ],
                ),
                macro,
                bundle,
                draws=4,
                seed=3,
            )
