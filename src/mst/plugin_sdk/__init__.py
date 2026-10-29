"""The plugin SDK: what a position model declares, reads from a run and returns.

A model imports only this package and the contracts; it never reads the workspace, the
network or the macro and satellite models themselves. Its package also exports
``plugin = Plugin(...)``, which the registry discovers; a demo case is what ``check`` and
``demo-run`` calculate. What a model needs and does not find here is a request to extend the
SDK, not a reason to reach into ``mst.engine``.
"""

from . import fx
from .model import (
    CalculationContext,
    CalculationModel,
    DataRequirements,
    Diagnostic,
    DriverRequirement,
    Position,
    PreparedInputs,
    RunOptions,
)
from .paths import FixedPaths, Paths, change, relative
from .plugin import DemoCase, Plugin
from .results import (
    CalculationResult,
    MetricValue,
    Quantiles,
    distribution,
    envelope,
    expected_shortfall,
    money,
    quantiles,
    scalar,
)
from .version import SDK_VERSION

__all__ = [
    "CalculationContext",
    "CalculationModel",
    "CalculationResult",
    "DataRequirements",
    "DemoCase",
    "Diagnostic",
    "DriverRequirement",
    "FixedPaths",
    "MetricValue",
    "Paths",
    "Plugin",
    "Position",
    "PreparedInputs",
    "Quantiles",
    "RunOptions",
    "SDK_VERSION",
    "change",
    "distribution",
    "envelope",
    "expected_shortfall",
    "fx",
    "money",
    "quantiles",
    "relative",
    "scalar",
]
