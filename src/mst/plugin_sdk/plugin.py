"""The declaration a plugin package exports, and the demo case its check calculates.

A plugin is a package under ``mst.plugins`` with a module-level ``plugin = Plugin(...)``. The
platform discovers it at start-up; a duplicate identifier, an incompatible SDK range or an
inconsistent declaration blocks start-up with a message that names the plugin.

A demo case is self-contained: a position, the paths it is calculated on and the driver the
model's requirement resolves to. It needs no workspace, no network and no model versions, so
``mst-plugin check`` and ``demo-run`` work on a fresh package.
"""

from dataclasses import dataclass, field

from .model import CalculationModel, Position
from .paths import Paths


@dataclass(frozen=True)
class DemoCase:
    """A position calculated on known paths, for the conformance check and ``demo-run``."""

    name: str
    position: Position
    paths: Paths
    #: The model's driver requirement quantity -> the driver id the demo paths carry.
    drivers: dict[str, str] = field(default_factory=dict)
    #: Whether ``check`` requires the position to be calculated rather than rejected.
    completes: bool = True


@dataclass(frozen=True)
class Plugin:
    """Everything a plugin adds: its identity and its calculation models."""

    plugin_id: str
    version: str
    #: SDK versions the plugin works with, e.g. ``">=1.0,<2.0"``.
    sdk: str
    label: str
    description: str = ""
    models: tuple[CalculationModel, ...] = ()
    demo: tuple[DemoCase, ...] = ()

    def namespaced(self, identifier: str) -> bool:
        return identifier.startswith(self.plugin_id + ".")
