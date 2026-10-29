"""A model, a demo case and a plugin declaration for the conformance tests."""

from datetime import date

import numpy as np

from mst.plugin_sdk import (
    CalculationContext,
    CalculationModel,
    DataRequirements,
    DemoCase,
    Diagnostic,
    DriverRequirement,
    FixedPaths,
    Plugin,
    Position,
    PreparedInputs,
    RunOptions,
    distribution,
    envelope,
    money,
)


class DemoModel(CalculationModel):
    """A model with one driver, so a demo case can be calculated without the workspace."""

    model_id = "gold.paths.v1"
    version = "0.1.0"
    label = "Gold"
    instrument_types = ("gold",)
    terms_schema = None
    metrics = ("gold.loss",)
    required_metrics = ("gold.loss",)
    description = "A demo model: the position follows the path of its price."

    def requirements(self, position: Position, options: RunOptions) -> DataRequirements:
        return DataRequirements(drivers=[DriverRequirement(key="price", quantity="gold_price")])

    def validate(self, prepared: PreparedInputs) -> list[Diagnostic]:
        return []

    def calculate(self, prepared: PreparedInputs, context: CalculationContext):
        path = np.asarray(context.paths.driver(prepared.driver("price")), dtype=float)
        change = prepared.value_current * (path[:, -1] / path[:, 0] - 1)
        return envelope(
            prepared,
            context,
            self,
            value_change=change,
            metrics={"gold.loss": distribution(money("RUB"), -change, tail=True)},
        )


DEMO = FixedPaths(
    drivers={"gold_price:demo": np.array([[100.0, 90.0], [100.0, 110.0]])},
    start=date(2026, 9, 1),
)

CASE = DemoCase(
    name="demo",
    position=Position(id="demo", type="gold", notional=10.0),
    paths=DEMO,
    drivers={"gold_price": "gold_price:demo"},
)


def plugin(**changes) -> Plugin:
    """The demo plugin, with the named fields replaced."""
    values = {
        "plugin_id": "gold",
        "version": "0.1.0",
        "sdk": ">=1.0,<2.0",
        "label": "Gold",
        "description": "",
        "models": (DemoModel(),),
        "demo": (CASE,),
    }
    values.update(changes)
    return Plugin(**values)
