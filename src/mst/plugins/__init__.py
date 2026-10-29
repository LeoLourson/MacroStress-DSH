"""The position models: a share, a bond, a loan or deposit, a real-estate position.

Each is a package with a ``MODEL`` instance; a model declares the instrument types it serves
and reads only ``mst.plugin_sdk``. ``mst.engine.model_catalog`` lists them, and step 10 of the
plan turns this module into discovery of ``mst.plugins.<id>`` packages. The names are ordered so
the catalog is stable: asset types of the same name must not depend on import order.
"""

from .bond import MODEL as BOND
from .credit import MODEL as CREDIT
from .equity import MODEL as EQUITY
from .housing import MODEL as HOUSING

MODELS = (EQUITY, BOND, CREDIT, HOUSING)

__all__ = ["MODELS"]
