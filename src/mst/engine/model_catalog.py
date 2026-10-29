"""The catalog of registered calculation models: the model of an instrument type.

The packages of ``mst.plugins`` register their models here; nothing in ``mst.engine`` branches on
an instrument type, it asks the catalog. Step 10 of the plan replaces this module's explicit
import with discovery of ``mst.plugins.<id>`` packages, keeping the catalog's answer the same.
"""

from ..plugin_sdk.model import CalculationModel, Position
from ..plugins import MODELS


class Catalog:
    """The models by id, and the model that serves an instrument type."""

    def __init__(self, models: tuple[CalculationModel, ...] = MODELS) -> None:
        self.models = {model.model_id: model for model in models}
        if len(self.models) != len(models):
            raise ValueError("a model is registered twice under one id")

    def model(self, model_id: str) -> CalculationModel:
        """The model ``model_id``."""
        try:
            return self.models[model_id]
        except KeyError:
            raise KeyError(f"unknown model {model_id!r}") from None

    def model_for(self, position: Position) -> CalculationModel:
        """The model that serves the position's type; two of them is an ambiguity, not a choice."""
        found = [model for model in self.models.values() if position.type in model.instrument_types]
        if not found:
            raise KeyError(f"no model serves the instrument type {position.type!r}")
        if len(found) > 1:
            names = ", ".join(sorted(model.model_id for model in found))
            raise ValueError(f"the instrument type {position.type!r} is served by {names}")
        return found[0]

    def catalogue(self) -> list[dict]:
        """Every registered model, by id."""
        return [self.models[model_id].describe() for model_id in sorted(self.models)]


catalog = Catalog()
