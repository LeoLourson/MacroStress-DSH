"""Fakes for the metrics tests: a macro model and a satellite model fast enough to run."""

from dataclasses import dataclass
from datetime import date

import numpy as np

from mst.data.store import Observation, SeriesStore
from mst.macro.panel import months_between, shift
from mst.model_sdk.contracts import (
    DriverPaths,
    MacroEstimate,
    MacroEstimateContext,
    MacroModel,
    MacroSimulation,
    SatelliteEstimate,
    SatelliteEstimateContext,
    SatelliteModel,
    SatellitePart,
    ScenarioPlan,
)

START = date(2020, 1, 1)
VARIABLES = ("gl.brent", "ru.key_rate")
SHOCKS = ("fake.demand", "fake.rate")


def series_store(root, months: int = 60, seed: int = 1) -> SeriesStore:
    """A store with a known random walk for every variable, one value a month."""
    store = SeriesStore(root)
    rng = np.random.default_rng(seed)
    grid = months_between(START, shift(START, months - 1))
    values = 100.0 + np.cumsum(rng.standard_normal((months, len(VARIABLES))) * 0.5, axis=0)
    for k, id_ in enumerate(VARIABLES):
        store.write(
            id_,
            [
                Observation(month, float(value), month, False)
                for month, value in zip(grid, values[:, k], strict=True)
            ],
        )
    return store


@dataclass
class _Loaded:
    """What FakeMacro.load returns: the panel and the ids."""

    ids: list[str]
    months: list[date]
    last: np.ndarray
    sigma: np.ndarray


class FakeMacro(MacroModel):
    """A macro model whose estimate is a panel read from the store, and whose paths are
    random walks around the last values — fast, and honest about what it ignores."""

    model_id = "fake.macro"
    version = "0.1.0"
    label = "Fake macro"
    variables = VARIABLES

    def __init__(self):
        self.calls = []

    def estimate(self, context: MacroEstimateContext) -> MacroEstimate:
        self.calls.append(context.as_of)
        months, columns = [], []
        for id_ in VARIABLES:
            rows = context.store.as_of(id_, context.as_of)
            months = [row.month for row in rows]
            columns.append([row.value for row in rows])
        y = np.array(columns, dtype=float).T
        return MacroEstimate(
            meta={
                "as_of": context.as_of.isoformat(),
                "ids": list(VARIABLES),
                "n_global": 1,
                "months": [m.isoformat() for m in months],
                "shocks": list(SHOCKS),
                "diagnostics": {},
            },
            arrays={
                "panel_y": y,
                "panel_approximate": np.zeros(y.shape, dtype=bool),
            },
        )

    def load(self, meta, arrays):
        return _Loaded(
            ids=list(meta["ids"]),
            months=[date.fromisoformat(m) for m in meta["months"]],
            last=arrays["panel_y"][-1],
            sigma=arrays["panel_y"].std(axis=0) / 4,
        )

    def simulate(self, model, plan: ScenarioPlan, *, draws: int, seed: int):
        horizon = plan.horizon
        rng = np.random.default_rng([seed, horizon])
        steps = rng.normal(0, 1, size=(draws, horizon, len(model.ids))) * model.sigma
        paths = np.concatenate(
            [np.tile(model.last, (draws, 1, 1)), model.last + np.cumsum(steps, axis=1)],
            axis=1,
        )
        months = [shift(model.months[-1], k) for k in range(horizon + 1)]
        return MacroSimulation(
            ids=model.ids, months=months, paths=paths, history=np.zeros((draws, 1, 2))
        )


@dataclass
class FakeSatellite(SatelliteModel):
    """A satellite whose drivers follow their only macro input, with a check that serves."""

    model_id = "fake.sat"
    version = "0.1.0"
    label = "Fake satellite"
    quantities = ("share_price",)

    def estimate(self, context: SatelliteEstimateContext) -> SatelliteEstimate:
        part = SatellitePart(
            meta={
                "quantity": context.registry.quantity("share_price").model_dump(
                    mode="json", by_alias=True
                ),
                "months": [m.isoformat() for m in context.panel.months],
                "nodes": ["sber"],
                "parent": [-1],
                "classes": ["company"],
                "driver": ["share_price:sber"],
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
            arrays={
                "d": np.linspace(5.0, 6.0, len(context.panel.months))[:, None],
                "gamma": np.zeros((1, 1, 2)),
                "sigma": np.zeros((1, 1)),
                "approximate": np.zeros((len(context.panel.months), 1), dtype=bool),
                "x": np.zeros((len(context.panel.months), 1)),
            },
        )
        return SatelliteEstimate(
            parts={"share_price": part},
            residuals={"share_price:sber": np.linspace(-1, 1, len(context.panel.months))},
        )

    def load(self, parts):
        return {"share_price": parts["share_price"].arrays["d"]}

    def simulate(
        self, model, macro, drivers, impacts, *, correlation=None, seed: int, stream: int
    ) -> DriverPaths:
        horizon = macro.paths.shape[1] - 1
        draws = macro.paths.shape[0]
        history = model["share_price"]
        rng = np.random.default_rng([seed, stream])
        paths = np.empty((draws, horizon + 1, len(drivers)))
        paths[:, 0, :] = history[-1]
        for k in range(len(drivers)):
            noise = rng.normal(0, 0.1, size=(draws, horizon))
            paths[:, 1:, k] = history[-1] + np.cumsum(noise, axis=1) + (
                macro.paths[:, 1:, 0] - macro.paths[:, 1, 0:1]
            ) * 0.1
        return DriverPaths(ids=list(drivers), paths=paths, flags={d: frozenset() for d in drivers})
