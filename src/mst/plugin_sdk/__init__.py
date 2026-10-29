"""The plugin SDK: what a position model declares, reads from a run and returns.

A model imports only this package and the contracts; it never reads the workspace, the
network or the macro and satellite models themselves. What a model needs and does not find
here is a request to extend the SDK, not a reason to reach into ``mst.engine``.
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

__all__ = [
    "CalculationContext",
    "CalculationModel",
    "CalculationResult",
    "DataRequirements",
    "Diagnostic",
    "DriverRequirement",
    "FixedPaths",
    "MetricValue",
    "Paths",
    "Position",
    "PreparedInputs",
    "Quantiles",
    "RunOptions",
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
