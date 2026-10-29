"""``bond.paths.v1``: a coupon bond on the run's paths of its spread driver.

The spread ``s`` of the bond's object is the driver ``bond_spread`` of the issue or its rating
group, in percentage points to the OFZ curve. The price of the bond moves by the modified
duration ``D``: ``V = notional · (1 − D · (s_H − s_0) / 100) · FX_H / FX_0``. A widening of the
spread is a loss; the coupon is outside the model, and the result states that convention.

The bond's terms carry the duration in years. It is the analyst's input: no default stands in
for it.
"""

import numpy as np

from mst.contracts import Contract, Positive
from mst.plugin_sdk import (
    CalculationContext,
    CalculationModel,
    DataRequirements,
    Diagnostic,
    DriverRequirement,
    Position,
    PreparedInputs,
    RunOptions,
    distribution,
    envelope,
    fx,
    money,
    scalar,
)

MODEL_ID = "bond.paths.v1"
QUANTITY = "bond_spread"


class BondTerms(Contract):
    #: Modified duration, years: how much the price moves per percentage point of the spread.
    duration: Positive


class BondOnPaths(CalculationModel):
    model_id = MODEL_ID
    version = "1.0.0"
    label = "Облигация на путях спреда"
    instrument_types = ("bond",)
    terms_schema = BondTerms
    metrics = ("bond.value_current", "bond.pnl", "bond.loss", "bond.return")
    required_metrics = ("bond.value_current", "bond.pnl", "bond.loss")
    description = (
        "Стоимость V = номинал·(1 − D·Δs/100)·FX_H/FX_0 по драйверу спреда к кривой ОФЗ; "
        "купон вне модели, дюрация — допущение аналитика."
    )

    def requirements(self, position: Position, options: RunOptions) -> DataRequirements:
        return DataRequirements(
            drivers=[
                DriverRequirement(key="spread", quantity=QUANTITY, purpose="путь спреда облигации")
            ],
            macro=fx.variables(position.currency, options.base_currency),
        )

    def validate(self, prepared: PreparedInputs) -> list[Diagnostic]:
        return []

    def calculate(self, prepared: PreparedInputs, context: CalculationContext):
        driver = prepared.driver("spread")
        spread = np.asarray(context.paths.driver(driver), dtype=float)
        duration = BondTerms.model_validate(prepared.position.terms).duration
        ratio = 1 - duration * (spread[:, -1] - spread[:, 0]) / 100
        rate = fx.ratio(context.paths, prepared.position.currency, context.base_currency)
        value0 = prepared.value_current
        change = value0 * (ratio * rate - 1)
        unit = money(context.base_currency)
        return envelope(
            prepared,
            context,
            self,
            value_change=change,
            metrics={
                "bond.value_current": scalar(unit, value0),
                "bond.pnl": distribution(unit, change),
                "bond.loss": distribution(unit, -change, tail=True),
                "bond.return": distribution("fraction", ratio * rate - 1),
            },
        )


MODEL = BondOnPaths()

