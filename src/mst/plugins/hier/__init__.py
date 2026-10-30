"""The hierarchical satellite model plugin: the reference ``G_l`` of the platform.

The Gibbs sampler over the object tree, the out-of-sample check and the driver simulation are
here; the platform knows this package only through its ``plugin`` declaration and the model
SDK. Another satellite model is another package: it serves the quantities it declares, the
platform refuses a quantity claimed by two models, and a version records which model
estimated which quantity.
"""

from .model import QUANTITIES, HierModel, HierSatellites
from .plugin import MODEL, plugin

__all__ = ["MODEL", "HierModel", "HierSatellites", "QUANTITIES", "plugin"]
