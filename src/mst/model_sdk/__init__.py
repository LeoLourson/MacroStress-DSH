"""The SDK of the model plugins: the macro model ``G_0`` and the satellite models ``G_l``.

A model plugin is developed outside the platform and connected to it: it declares
``plugin = Plugin(...)`` with ``macro=...`` or ``satellites=...``, is discovered like every
other plugin (``docs/howto/plugins.md``) and speaks only the contracts of this package. The macro
model estimates and simulates the paths of the macro variables under a scenario's constraints;
a satellite model estimates and simulates the quantities of its objects on those paths. What a
version holds is decided by the model that estimated it — the platform stores metadata and
arrays and reads back only the keys named in ``contracts.py``.
"""

from .conditional import (
    RANK_TOLERANCE,
    SWEEPS,
    InfeasibleScenario,
    box_gibbs,
    dependent_rows,
    draw_driving,
    truncated_normal,
)
from .contracts import (
    BUNDLE_META_REQUIRED,
    MACRO_ARRAYS_REQUIRED,
    MACRO_META_REQUIRED,
    PART_META_REQUIRED,
    AdditiveImpact,
    Check,
    Dist,
    DriverPaths,
    LinearConstraint,
    LinearTerm,
    MacroEstimate,
    MacroEstimateContext,
    MacroModel,
    MacroSimulation,
    SatelliteEstimate,
    SatelliteEstimateContext,
    SatelliteModel,
    SatellitePart,
    ScenarioPlan,
    Severity,
)
from .testing import Finding, check_shocks
from .version import MODEL_SDK_VERSION, model_compatible

__all__ = [
    "BUNDLE_META_REQUIRED",
    "MACRO_ARRAYS_REQUIRED",
    "MACRO_META_REQUIRED",
    "MODEL_SDK_VERSION",
    "PART_META_REQUIRED",
    "AdditiveImpact",
    "Check",
    "Dist",
    "DriverPaths",
    "Finding",
    "InfeasibleScenario",
    "LinearConstraint",
    "LinearTerm",
    "MacroEstimate",
    "MacroEstimateContext",
    "MacroModel",
    "MacroSimulation",
    "RANK_TOLERANCE",
    "SWEEPS",
    "SatelliteEstimate",
    "SatelliteEstimateContext",
    "SatelliteModel",
    "SatellitePart",
    "ScenarioPlan",
    "Severity",
    "box_gibbs",
    "check_shocks",
    "dependent_rows",
    "draw_driving",
    "model_compatible",
    "truncated_normal",
]
