"""What a model returns and the helpers that fill it, so a model states only what it calculated.

Money is in millions of its currency, a return and a share as fractions, a log return as a
number. Every distribution carries P5, P50 and P95 together — no quantile without the others —
and a loss carries ES95 as well, because the report reads both.
"""

import numpy as np

from ..contracts import (
    Contract,
    Currency,
    Identifier,
    NonNegative,
    Number,
    ResultStatus,
)
from .model import Diagnostic


def money(currency: str) -> str:
    """The unit of money in millions of ``currency``."""
    return f"{currency}_million"


class Quantiles(Contract):
    """P5, P50 and P95 of a distribution, and ES95 of a loss (higher is worse)."""

    p05: Number
    p50: Number
    p95: Number
    es95: Number | None = None


class MetricValue(Contract):
    """One metric of a position: its unit, its central value and its distribution."""

    unit: Identifier
    value: float | None = None
    quantiles: Quantiles | None = None


class CalculationResult(Contract):
    """What a model returns, and what a run reports for a position it did not calculate."""

    position_id: Identifier
    name: str = ""
    instrument_type: Identifier
    model_id: Identifier
    model_version: Identifier
    status: ResultStatus
    base_currency: Currency
    #: The position's value at ``t = 0`` in millions of the base currency.
    value_current: NonNegative
    #: ``V_q^(r)``, the change of the position's value, per repetition, in base millions.
    value_change: list[Number] = []
    metrics: dict[Identifier, MetricValue] = {}
    flags: list[Identifier] = []
    diagnostics: list[Diagnostic] = []


def expected_shortfall(draws, level: float = 0.95) -> float:
    """``ES = 1/(1-a) * integral_a^1 VaR_u du`` (spec 16).

    On a discrete sample the integral becomes a weighted sum that gives the boundary
    observation exactly the mass the tail still needs, rather than dropping or double
    counting it.
    """
    values = np.sort(np.asarray(draws, dtype=float))
    n = values.size
    if n == 0:
        return 0.0
    tail = 1.0 - level
    if tail <= 0:
        return float(values[-1])
    upper = np.arange(1, n + 1) / n
    lower = np.arange(0, n) / n
    weight = np.clip(upper - np.maximum(lower, level), 0.0, None)
    return float(np.sum(values * weight) / tail)


def quantiles(draws, *, tail: bool = False) -> Quantiles:
    """P5/P50/P95 of the realisations; ``tail`` adds ES95 for a loss."""
    values = np.asarray(draws, dtype=float).reshape(-1)
    p05, p50, p95 = (float(value) for value in np.percentile(values, [5, 50, 95]))
    return Quantiles(
        p05=p05,
        p50=p50,
        p95=p95,
        es95=max(expected_shortfall(values), p95) if tail else None,
    )


def distribution(unit: str, draws, *, value: float | None = None, tail: bool = False):
    """A metric with its distribution."""
    return MetricValue(unit=unit, value=value, quantiles=quantiles(draws, tail=tail))


def scalar(unit: str, value: float) -> MetricValue:
    """A metric that holds one number."""
    return MetricValue(unit=unit, value=float(value))


def envelope(
    prepared,
    context,
    model,
    *,
    value_change,
    metrics: dict[str, MetricValue],
    status: str = "ok",
    flags=(),
    diagnostics=(),
) -> CalculationResult:
    """A completed result of ``model`` for the prepared position under ``context``.

    The flags of the resolved drivers travel with the result, so the platform, not the model,
    guarantees that a disclosure (parameters from the level above, approximate data, an overlay)
    reaches the result.
    """
    disclosures = [
        f"{flag}:{driver}"
        for driver in sorted(set(prepared.drivers.values()))
        for flag in sorted(context.paths.flags(driver))
    ]
    return CalculationResult(
        position_id=prepared.position.id,
        name=prepared.position.name,
        instrument_type=prepared.position.type,
        model_id=model.model_id,
        model_version=model.version,
        status=status,
        base_currency=prepared.base_currency,
        value_current=prepared.value_current,
        value_change=np.asarray(value_change, dtype=float).tolist(),
        metrics=metrics,
        flags=[*flags, *disclosures],
        diagnostics=list(diagnostics),
    )
