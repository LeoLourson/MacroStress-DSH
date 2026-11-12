"""``bond.paths.v1``: a coupon bond on the run's paths of its yield.

The yield of the bond is the risk-free curve at its duration plus its spread ``s``, the driver
``bond_spread`` of the issue or its rating group (p.p. to the OFZ curve). The curve is read
from the macro paths: the key rate is its short end, ``ru.ofz_1y``, ``ru.ofz_5y`` and
``ru.ofz_10y`` its tenors of 1, 5 and 10 years, linear between them and flat beyond; a tenor
the macro model does not carry is left out. ``D`` is the modified duration now; ``h`` months
later ``D_h = max(D − h/12, 0)`` is left. With ``y_0 = r_0(D) + s_0`` and
``y_h = r_h(D_h) + s_h`` the value in the currency of the bond is

    V_h = notional · (1 + y_0/100)^(h/12) · (1 − D_h · (y_h − y_0) / 100),

the yield earned over the horizon (coupons and the pull to par, reinvested at ``y_0``) and the
price move of the bond that is left. A bond whose duration runs out before ``h`` is redeemed:
it carries no rate risk after that. The move is first order: no convexity.

The bond's terms carry the modified duration in years now. It is the analyst's input: no
default stands in for it. The exchange gives the Macaulay duration in days; the modified one
is ``days / 365 / (1 + y / f)`` for ``f`` coupons a year.
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
SHORT = "ru.key_rate"
CURVE = ((0.0, SHORT), (1.0, "ru.ofz_1y"), (5.0, "ru.ofz_5y"), (10.0, "ru.ofz_10y"))
"""Tenors in years and the macro variables of the risk-free curve, %."""


class BondTerms(Contract):
    #: Modified duration now, years: the share of the price lost per percentage point of yield.
    duration: Positive


def curve_rate(paths, month: int, tenor: float) -> np.ndarray:
    """The risk-free rate at ``tenor`` years in ``month`` of every repetition, %."""
    nodes = [(years, name) for years, name in CURVE if name in paths.variables]
    years = np.array([item[0] for item in nodes])
    rates = np.column_stack([np.asarray(paths.macro(name))[:, month] for _, name in nodes])
    weights = np.array([np.interp(tenor, years, unit) for unit in np.eye(len(nodes))])
    return rates @ weights


class BondOnPaths(CalculationModel):
    model_id = MODEL_ID
    version = "2.0.0"
    label = "Облигация на путях доходности"
    instrument_types = ("bond",)
    terms_schema = BondTerms
    metrics = ("bond.value_current", "bond.pnl", "bond.loss", "bond.return")
    required_metrics = ("bond.value_current", "bond.pnl", "bond.loss")
    description = (
        "Стоимость V = номинал·(1+y₀)^(h/12)·(1 − D_h·Δy/100)·FX_H/FX_0: доходность — кривая "
        "ОФЗ на дюрации плюс спред, D_h = max(D − h/12, 0); дюрация — допущение аналитика."
    )

    def requirements(self, position: Position, options: RunOptions) -> DataRequirements:
        return DataRequirements(
            drivers=[
                DriverRequirement(key="spread", quantity=QUANTITY, purpose="путь спреда облигации")
            ],
            macro=[SHORT, *fx.variables(position.currency, options.base_currency)],
        )

    def validate(self, prepared: PreparedInputs) -> list[Diagnostic]:
        return []

    def calculate(self, prepared: PreparedInputs, context: CalculationContext):
        driver = prepared.driver("spread")
        spread = np.asarray(context.paths.driver(driver), dtype=float)
        duration = BondTerms.model_validate(prepared.position.terms).duration
        months = spread.shape[1] - 1
        left = max(duration - months / 12, 0.0)
        start = curve_rate(context.paths, 0, duration) + spread[:, 0]
        end = curve_rate(context.paths, months, left) + spread[:, -1]
        ratio = (1 + start / 100) ** (months / 12) * (1 - left * (end - start) / 100)
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
