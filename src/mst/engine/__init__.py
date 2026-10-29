"""The type-agnostic calculation: models by instrument type, inputs, results and status.

The models come from the ``mst.plugins`` registry; nothing here names an instrument type or a
plugin package.
"""

from .model_catalog import Catalog, catalog
from .run import Batch, run
from .status import worst

__all__ = ["Batch", "Catalog", "catalog", "run", "worst"]
