"""The report of a run: the loss at P5, P50 and P95, and the severity of its scenario.

Losses are summed inside each repetition and only then turned into quantiles: the sum of the
positions' own P95 is descriptive statistics and is never reported as the portfolio's P95. A
position that was not calculated does not quietly disappear: it is excluded with its reason, and
the report states the share of the value the numbers cover. ``delta`` is the distance of the
scenario's paths from the unconditional forecast (``mst.macro.forecast.Severity``); a scenario
without constraints has no severity, and the report says so instead of inventing a zero.
"""

from typing import Literal

import numpy as np

from ..engine.status import worst
from ..plugin_sdk import expected_shortfall

DEFAULT_COVERAGE_LIMIT = 0.05
"""Uncovered value above this share turns the portfolio's status into ``incomplete``."""

PortfolioStatus = Literal["ok", "needs_review", "incomplete"]


def severity(found) -> dict | None:
    """The severity of a scenario as the report states it, or None without constraints."""
    if found is None:
        return None
    return {
        "distance": found.distance,
        "exceeded": found.exceeded,
        "probabilities": [float(value) for value in found.probabilities],
    }


def build(
    portfolio,
    results,
    *,
    severity_found=None,
    coverage_limit: float = DEFAULT_COVERAGE_LIMIT,
) -> dict:
    """The report of ``results`` against the portfolio that produced them."""
    included = [result for result in results if result.status != "invalid"]
    excluded = [
        {
            "position": result.position_id,
            "status": result.status,
            "reason": "; ".join(result.flags) or "the position was not calculated",
        }
        for result in results
        if result.status == "invalid"
    ]
    # A notional is positive, so a zero value is a position the platform could not price: it
    # cannot be added to a portfolio value in another currency, and the report names it instead.
    unknown = [result.position_id for result in results if result.value_current == 0]
    total = sum(result.value_current for result in results)
    covered = sum(result.value_current for result in included)
    share = covered / total if total > 0 else None
    flags = [f"exposure_unknown:{position}" for position in unknown]
    if excluded:
        flags.append("portfolio_exclusions")
        if share is None or 1.0 - share > coverage_limit:
            flags.append("portfolio_incomplete")
    incomplete = bool(unknown) or (
        bool(excluded) and (share is None or 1.0 - share > coverage_limit)
    )
    position_status = worst(*(result.status for result in included))
    status = (
        "incomplete"
        if incomplete
        else "needs_review"
        if excluded or position_status == "needs_review"
        else "ok"
    )
    losses = (
        -np.sum([np.asarray(result.value_change, dtype=float) for result in included], axis=0)
        if included
        else np.zeros(0)
    )
    return {
        "portfolio": {
            "name": portfolio.name,
            "base_currency": portfolio.base_currency,
            "positions": len(portfolio.positions),
        },
        "status": status,
        "flags": flags,
        "loss": _loss(losses),
        "severity": severity(severity_found),
        "coverage": {
            "share": share,
            "limit": coverage_limit,
            "unknown": unknown,
            "excluded": excluded,
        },
        "positions": [
            {
                "position": result.position_id,
                "name": result.name,
                "type": result.instrument_type,
                "model": result.model_id,
                "status": result.status,
                "value_current": result.value_current,
                "flags": list(result.flags),
            }
            for result in results
        ],
    }


def _loss(losses: np.ndarray) -> dict | None:
    """P5/P50/P95 of the portfolio's loss and ES95; None when nothing was calculated."""
    if losses.size == 0:
        return None
    p05, p50, p95 = (float(value) for value in np.percentile(losses, [5, 50, 95]))
    return {"p05": p05, "p50": p50, "p95": p95, "es95": max(expected_shortfall(losses), p95)}
