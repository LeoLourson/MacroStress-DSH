"""``equity.paths.v1``: an ordinary share on the run's paths of its price driver.

The price follows the driver ``share_price`` of the share's object (a ticker's satellite carries
the share's sensitivity to the market, the rouble, the rates, formalization section 6):
``P_H = P_0 · D_H / D_0``, with ``D`` the driver's path in roubles per share. A position in
another currency is valued at the calculation month's spot rate moved by the macro rate of its
pair: ``FX_H = FX_0 · X_H / X_0``. ``V = notional · (P_H / P_0 · FX_H / FX_0 − 1)``.

The share's terms are empty: everything the model reads is the driver and the currency pair. A
share whose ticker no served object covers is not calculated, and the result names the code.
"""

import numpy as np

from mst.contracts import Contract
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

MODEL_ID = "equity.paths.v1"
QUANTITY = "share_price"


class ShareTerms(Contract):
    """A share states no terms: its codes name the object and its price is its driver."""


class ShareOnPaths(CalculationModel):
    model_id = MODEL_ID
    version = "1.0.0"
    label = "Акция на путях драйверов"
    instrument_types = ("common_stock",)
    terms_schema = ShareTerms
    metrics = ("equity.value_current", "equity.pnl", "equity.loss", "equity.return")
    required_metrics = ("equity.value_current", "equity.pnl", "equity.loss")
    description = (
        "Цена P_H = P_0·D_H/D_0 по драйверу цены акции; курс — по пути макропеременной пары. "
        "Стоимость в базовой валюте — номинал на изменение цены и курса."
    )

    def requirements(self, position: Position, options: RunOptions) -> DataRequirements:
        return DataRequirements(
            drivers=[DriverRequirement(key="price", quantity=QUANTITY, purpose="путь цены акции")],
            macro=fx.variables(position.currency, options.base_currency),
        )

    def validate(self, prepared: PreparedInputs) -> list[Diagnostic]:
        return []

    def calculate(self, prepared: PreparedInputs, context: CalculationContext):
        driver = prepared.driver("price")
        price = np.asarray(context.paths.driver(driver), dtype=float)
        if np.any(price <= 0):
            raise ValueError(f"{driver} carries a price at or below zero")
        ratio = price[:, -1] / price[:, 0]
        rate = fx.ratio(context.paths, prepared.position.currency, context.base_currency)
        value0 = prepared.value_current
        change = value0 * (ratio * rate - 1)
        unit = money(context.base_currency)
        flags = [f"{flag}:{driver}" for flag in sorted(context.paths.flags(driver))]
        return envelope(
            prepared,
            context,
            self,
            value_change=change,
            flags=flags,
            metrics={
                "equity.value_current": scalar(unit, value0),
                "equity.pnl": distribution(unit, change),
                "equity.loss": distribution(unit, -change, tail=True),
                "equity.return": distribution("fraction", ratio * rate - 1),
            },
        )


MODEL = ShareOnPaths()

