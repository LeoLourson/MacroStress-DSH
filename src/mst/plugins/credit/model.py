"""``credit.paths.v1``: a loan or a deposit on the paths of its overdue-share driver.

The driver ``overdue_share`` of the borrower's activity (OKVED) is the share of overdue debt in
the loans of that activity. A position loses ``LGD`` of its notional for every point the share
rises: ``V = notional · (1 − LGD · (o_H − o_0)) · FX_H / FX_0``. ``LGD`` is the analyst's input,
required and never defaulted. A deposit is the same claim on the bank's distress: insurance and
recovery of a defaulted debt are outside the model, and the result states that convention.
"""

import numpy as np

from mst.contracts import Contract, Probability
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

MODEL_ID = "credit.paths.v1"
QUANTITY = "overdue_share"


class CreditTerms(Contract):
    #: Loss given default: the share of the notional lost when the overdue share rises by one.
    lgd: Probability


class CreditOnOverdue(CalculationModel):
    model_id = MODEL_ID
    version = "1.0.0"
    label = "Кредит или депозит на путях доли просроченной задолженности"
    instrument_types = ("loan", "deposit")
    terms_schema = CreditTerms
    metrics = ("credit.value_current", "credit.pnl", "credit.loss", "credit.return")
    required_metrics = ("credit.value_current", "credit.pnl", "credit.loss")
    description = (
        "Стоимость V = номинал·(1 − LGD·Δo)·FX_H/FX_0 по драйверу доли просроченной "
        "задолженности отрасли заёмщика; LGD — допущение аналитика."
    )

    def requirements(self, position: Position, options: RunOptions) -> DataRequirements:
        return DataRequirements(
            drivers=[
                DriverRequirement(
                    key="overdue", quantity=QUANTITY, purpose="путь доли просроченной задолженности"
                )
            ],
            macro=fx.variables(position.currency, options.base_currency),
        )

    def validate(self, prepared: PreparedInputs) -> list[Diagnostic]:
        return []

    def calculate(self, prepared: PreparedInputs, context: CalculationContext):
        driver = prepared.driver("overdue")
        share = np.asarray(context.paths.driver(driver), dtype=float)
        lgd = CreditTerms.model_validate(prepared.position.terms).lgd
        ratio = 1 - lgd * (share[:, -1] - share[:, 0])
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
                "credit.value_current": scalar(unit, value0),
                "credit.pnl": distribution(unit, change),
                "credit.loss": distribution(unit, -change, tail=True),
                "credit.return": distribution("fraction", ratio * rate - 1),
            },
        )


MODEL = CreditOnOverdue()

