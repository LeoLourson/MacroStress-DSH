"""The catalog of registered calculation models: the model of an instrument type.

The packages of ``mst.plugins`` declare their models in ``mst.plugins``; the catalog indexes the
registry's models and answers by model id or by the position's instrument type. Nothing in
``mst.engine`` branches on an instrument type, it asks the catalog.
"""

from ..plugin_sdk.model import CalculationModel, Position
from ..plugins import registry


class Catalog:
    """The models by id, and the model that serves an instrument type."""

    def __init__(self, models: tuple[CalculationModel, ...] | None = None) -> None:
        found = tuple(models) if models is not None else tuple(registry().models.values())
        self.models = {model.model_id: model for model in found}
        if len(self.models) != len(found):
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


catalog = Catalog()
