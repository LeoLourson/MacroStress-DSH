"""Paths a test can compute: every driver starts at a given value and ends at another."""

from datetime import date

import numpy as np

from mst.plugin_sdk import FixedPaths

START = date(2026, 9, 1)


def line(first: float, last: float, *, draws: int = 3, horizon: int = 2) -> np.ndarray:
    """A straight path from ``first`` at ``t = 0`` to ``last`` at ``t = H`` for every draw."""
    return np.tile(np.linspace(first, last, horizon + 1), (draws, 1))


def fan(first: float, last: float, *, draws: int = 5, horizon: int = 2) -> np.ndarray:
    """A path from ``first`` at ``t = 0`` to a different ``last`` in every draw."""
    ends = np.linspace(first, last, draws)
    months = np.linspace(0.0, 1.0, horizon + 1)
    return first + np.outer(ends - first, months)


def paths(
    *,
    draws: int = 3,
    horizon: int = 2,
    share_price=(100.0, 110.0),
    second_share=(50.0, 40.0),
    bond_spread=(5.0, 6.0),
    overdue_share=(0.02, 0.05),
    housing_price=(100_000.0, 90_000.0),
    usdrub=(80.0, 100.0),
    flags: dict | None = None,
) -> FixedPaths:
    """The paths every engine test reads: the drivers of ``tests.engine.simulated``."""
    drivers = {
        "share_price:sber": line(*share_price, draws=draws, horizon=horizon),
        "share_price:gazp": line(*second_share, draws=draws, horizon=horizon),
        "bond_spread:corp": line(*bond_spread, draws=draws, horizon=horizon),
        "overdue_share:mining": line(*overdue_share, draws=draws, horizon=horizon),
        "housing_price_primary:belgorod": line(*housing_price, draws=draws, horizon=horizon),
    }
    return FixedPaths(
        macros={"ru.usdrub": line(*usdrub, draws=draws, horizon=horizon)},
        drivers=drivers,
        driver_flags=flags or {},
        start=START,
        seed=7,
    )
