"""Drivers simulated from a known hierarchical satellite model, on a simulated macro panel."""

from datetime import date

import numpy as np

from mst.data.store import Observation, SeriesStore
from mst.drivers.registry import Registry
from mst.macro.panel import Panel, months_between

START = date(2010, 1, 1)
SECTOR = np.array([2.0, 0.5, 1.5, -0.8])
"""alpha, rho, beta on x_t, beta on x_{t-1} of the sector."""
COMPANIES = {
    "alpha": SECTOR + np.array([0.5, -0.05, 0.3, 0.1]),
    "beta": SECTOR + np.array([-0.4, 0.05, -0.2, -0.1]),
    "gamma": SECTOR + np.array([0.0, 0.1, 0.4, 0.0]),
}
SIGMA = 1.0


def registry(*, without_history=("delta",)) -> Registry:
    objects = [{"id": "sector", "class": "sector", "label": "Сектор"}]
    objects += [
        {"id": name, "class": "company", "label": name, "parent": "sector"}
        for name in [*COMPANIES, *without_history]
    ]
    drivers = [{"quantity": "price", "object": "sector", "series": "drv.price.sector"}]
    drivers += [
        {"quantity": "price", "object": name, "series": f"drv.price.{name}"} for name in COMPANIES
    ]
    drivers += [{"quantity": "price", "object": name} for name in without_history]
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
                }
            ],
            "objects": objects,
            "drivers": drivers,
        }
    )


def panel(months: int, seed: int = 1) -> Panel:
    rng = np.random.default_rng(seed)
    grid = months_between(START, date(START.year + (months - 1) // 12, (months - 1) % 12 + 1, 1))
    x = np.cumsum(rng.standard_normal(len(grid)))
    y = np.column_stack([x, np.zeros(len(grid))])
    return Panel(
        ids=["gl.brent", "ru.key_rate"],
        n_global=1,
        months=grid,
        y=y,
        approximate=np.zeros(y.shape, dtype=bool),
    )


def path(gamma: np.ndarray, x: np.ndarray, rng, sigma: float = SIGMA) -> np.ndarray:
    d = np.empty(len(x))
    d[0] = gamma[0] / (1 - gamma[1])
    for t in range(1, len(x)):
        d[t] = gamma @ [1.0, d[t - 1], x[t], x[t - 1]] + sigma * rng.standard_normal()
    return d


def store(root, data: Panel, seed: int = 2) -> SeriesStore:
    rng = np.random.default_rng(seed)
    result = SeriesStore(root)
    x = data.y[:, 0]
    for name, gamma in {"sector": SECTOR, **COMPANIES}.items():
        values = path(gamma, x, rng)
        result.write(
            f"drv.price.{name}",
            [Observation(m, float(v), m, False) for m, v in zip(data.months, values, strict=True)],
        )
    return result
