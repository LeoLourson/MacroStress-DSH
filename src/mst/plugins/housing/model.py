"""``housing.paths.v1``: a real-estate position on the run's paths of a housing price.

The position is a claim on a square metre in a region: the driver ``housing_price_primary`` or
``housing_price_secondary`` of the region (``okato``) gives ``P_H = P_0 · D_H / D_0``, and
``V = notional · (D_H / D_0 · FX_H / FX_0 − 1)``. The region is matched by the position's codes,
so a position without ``okato`` is not served, and the result names the code.

The market is a term, not a default: a position states whether it is the primary or the
secondary market, and the model reads the driver of that market only.
"""

from typing import Literal

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

MODEL_ID = "housing.paths.v1"
QUANTITIES = {"primary": "housing_price_primary", "secondary": "housing_price_secondary"}


class HousingTerms(Contract):
    market: Literal["primary", "secondary"]


class HousingOnPaths(CalculationModel):
    model_id = MODEL_ID
    version = "1.0.0"
    label = "Недвижимость на путях цены квадратного метра"
    instrument_types = ("real_estate",)
    terms_schema = HousingTerms
    metrics = ("housing.value_current", "housing.pnl", "housing.loss", "housing.return")
    required_metrics = ("housing.value_current", "housing.pnl", "housing.loss")
    description = (
        "Цена P_H = P_0·D_H/D_0 по драйверу цены квадратного метра региона (ОКАТО) на "
        "первичном или вторичном рынке; стоимость — номинал на изменение цены и курса."
    )

    def requirements(self, position: Position, options: RunOptions) -> DataRequirements:
        market = HousingTerms.model_validate(position.terms).market
        return DataRequirements(
            drivers=[
                DriverRequirement(
                    key="price", quantity=QUANTITIES[market], purpose="путь цены квадратного метра"
                )
            ],
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
                "housing.value_current": scalar(unit, value0),
                "housing.pnl": distribution(unit, change),
                "housing.loss": distribution(unit, -change, tail=True),
                "housing.return": distribution("fraction", ratio * rate - 1),
            },
        )


MODEL = HousingOnPaths()

