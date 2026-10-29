"""What a position model sees of a scenario: paths of macro variables and drivers.

A run simulates ``R`` repetitions of the months ``t = 0 .. H`` once (formalization, section 7)
and hands every model the same ``Paths``. Values are in their own units: a rate in percent, a
rate of exchange in roubles, a share price in roubles per share, a share as a fraction; the
transforms ``g`` stay inside the macro model and the satellites (plan, section 2, rule 3).
Month ``t = 0`` is the calculation month, the last month of history.

A model declares the macro variables and drivers it reads in ``DataRequirements`` and does not
know how they are obtained. A driver is named by quantity and object, or by quantity and the
position's codes in order of preference (``{"inn": ..., "okved": ...}``); the platform takes the
first code for which the driver registry has a served object covering it, the deepest such
object. A driver the run cannot serve is missing: the position is not calculated, and its
result names the driver (rule 4). The flags of a driver (parameters taken from the level above,
an approximate data version, an expert overlay) reach the result through ``flags`` (rule 5).
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Protocol

import numpy as np


class Paths(Protocol):
    """The run's paths, as a position model reads them."""

    draws: int
    """R, the repetitions."""
    horizon: int
    """H, the months after the calculation month."""
    months: list[date]
    """The first day of months t = 0 .. H."""
    variables: frozenset[str]
    """The macro variables the paths hold."""
    seed: int
    """The seed every repetition of the run was drawn with."""

    def macro(self, variable: str) -> np.ndarray:
        """(R, H + 1) values of a macro variable in its unit."""

    def driver(self, driver: str) -> np.ndarray:
        """(R, H + 1) values of a driver ``<quantity>:<object>`` in its unit."""

    def flags(self, driver: str) -> frozenset[str]:
        """The flags of a driver: ``parameters_from_above``, ``approximate_data``, ``overlay``."""


def relative(path: np.ndarray) -> np.ndarray:
    """``X_t / X_0`` for months ``t = 0 .. H``: the ratio a model applies to a value today."""
    path = np.asarray(path, dtype=float)
    return path / path[:, :1]


def change(path: np.ndarray) -> np.ndarray:
    """``X_t - X_0`` for months ``t = 0 .. H``: the change of a rate or a share."""
    path = np.asarray(path, dtype=float)
    return path - path[:, :1]


@dataclass(frozen=True)
class FixedPaths:
    """``Paths`` from given arrays, for a model's tests and hand-checked examples."""

    macros: dict[str, np.ndarray]
    drivers: dict[str, np.ndarray] = field(default_factory=dict)
    driver_flags: dict[str, frozenset[str]] = field(default_factory=dict)
    start: date = date(2026, 9, 1)
    seed: int = 0

    def __post_init__(self):
        arrays = {**self.macros, **self.drivers}
        if not arrays:
            raise ValueError("the paths carry nothing: give macro variables or drivers")
        shapes = {np.asarray(path).shape for path in arrays.values()}
        if len(shapes) != 1:
            raise ValueError(f"the paths differ in shape: {sorted(shapes)}")
        if len(shapes.pop()) != 2:
            raise ValueError("a path is (R, H + 1)")

    @property
    def draws(self) -> int:
        return self._any().shape[0]

    @property
    def horizon(self) -> int:
        return self._any().shape[1] - 1

    @property
    def months(self) -> list[date]:
        index = self.start.year * 12 + self.start.month - 1
        return [date((index + t) // 12, (index + t) % 12 + 1, 1) for t in range(self.horizon + 1)]

    @property
    def variables(self) -> frozenset[str]:
        return frozenset(self.macros)

    def macro(self, variable: str) -> np.ndarray:
        return self.macros[variable]

    def driver(self, driver: str) -> np.ndarray:
        return self.drivers[driver]

    def flags(self, driver: str) -> frozenset[str]:
        return self.driver_flags.get(driver, frozenset())

    def _any(self) -> np.ndarray:
        """Any one path: every path of the set has the same shape."""
        return next(iter({**self.macros, **self.drivers}.values()))
