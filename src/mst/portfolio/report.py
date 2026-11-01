"""The report of a run: the loss at P5, P50 and P95, and the price of its scenario.

Losses are summed inside each repetition and only then turned into quantiles: the sum of the
positions' own P95 is descriptive statistics and is never reported as the portfolio's P95. A
position that was not calculated does not quietly disappear: it is excluded with its reason, and
the report states the share of the value the numbers cover.

The price of the scenario — ``dist(s) = E_P[-log P(S | R)]`` of the canon
(``mst.model_sdk.dist``) — stands beside the loss: the mean of ``-log P(S | R)`` over the law
of ``R``, with its standard error and the causes the scenario names. An infinite distance is
passed as is and named: at some value of ``R`` the event has no probability, the case lemma 2
of the canon names, and the scenario contradicts the models.
"""

from typing import Literal

import numpy as np

from ..engine.status import worst
from ..metrics.es import es_block

DEFAULT_COVERAGE_LIMIT = 0.05
"""Uncovered value above this share turns the portfolio's status into ``incomplete``."""

PortfolioStatus = Literal["ok", "needs_review", "incomplete"]


def dist(found) -> dict | None:
    """The price of the scenario as the report states it, or None when there is none.

    An infinite distance — the event has no probability at some value of ``R`` — is passed
    as is, with the message the estimate attached, beside the standard error and the causes
    the scenario names.
    """
    if found is None:
        return None
    return {
        "distance": float(found.distance),
        "log_probability": float(found.log_probability),
        "standard_error": float(found.standard_error),
        "causes": list(found.causes),
        "tolerances": list(found.tolerances),
        "by_level": {str(level): float(price) for level, price in sorted(found.by_level.items())},
        "message": found.message,
    }


def build(
    portfolio,
    results,
    *,
    dist_found=None,
    seed: int = 0,
    horizon: str | None = None,
    coverage_limit: float = DEFAULT_COVERAGE_LIMIT,
) -> dict:
    """The report of ``results`` against the portfolio that produced them."""
    included = [result for result in results if result.status != "invalid"]
    excluded = [
        {
            "position": result.position_id,
            "status": result.status,
            "reason": "; ".join(
                item.message for item in result.diagnostics if item.severity == "error"
            )
            or "; ".join(result.flags)
            or "the position was not calculated",
        }
        for result in results
        if result.status == "invalid"
    ]
    # A notional is positive, so a zero value is a position the platform could not price: it
    # cannot be added to a portfolio value in another currency, and the report names it instead.
    unknown = [result.position_id for result in results if result.value_current == 0]
    total = sum(result.value_current for result in results)
    covered = sum(result.value_current for result in included)
    # The share is of the measurable value; with an unmeasurable position it is unknown, not 1.
    share = None if unknown or total <= 0 else covered / total
    flags = [f"exposure_unknown:{position}" for position in unknown]
    if excluded:
        flags.append("portfolio_exclusions")
        if share is None or 1.0 - share > coverage_limit:
            flags.append("portfolio_incomplete")
    incomplete = bool(unknown) or (
        bool(excluded) and (share is None or 1.0 - share > coverage_limit)
    )
    position_status = worst(*(result.status for result in included))
    status: PortfolioStatus = (
        "incomplete"
        if incomplete
        else "needs_review"
        if excluded or position_status == "needs_review"
        else "ok"
    )
    losses = losses_of(results)
    return {
        "portfolio": {
            "name": portfolio.name,
            "base_currency": portfolio.base_currency,
            "positions": len(portfolio.positions),
        },
        "status": status,
        "flags": flags,
        "loss": _loss(losses, seed=seed, horizon=horizon),
        "dist": dist(dist_found),
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


def losses_of(results) -> np.ndarray:
    """The portfolio loss of every repetition: minus the sum of the calculated positions."""
    included = [result for result in results if result.status != "invalid"]
    if not included:
        return np.zeros(0)
    return -np.sum(
        [np.asarray(result.value_change, dtype=float) for result in included], axis=0
    )


def loss_block(results, *, seed: int = 0, horizon: str | None = None) -> dict | None:
    """P5/P50/P95 and the ES block of one run's results — what a contribution reports."""
    return _loss(losses_of(results), seed=seed, horizon=horizon)


def _loss(losses: np.ndarray, *, seed: int = 0, horizon: str | None = None) -> dict | None:
    """P5/P50/P95 of the portfolio's loss and ES95 with its interval and trimming.

    The expected shortfall carries what the canon asks beside it — a bootstrap interval, the
    sensitivity to dropping the heaviest one percent of the paths and the horizon it belongs
    to — through the shared block of ``mst.metrics.es``; the horizon mark names the month the
    loss is measured at when the caller knows it.
    """
    if losses.size == 0:
        return None
    p05, p50, p95 = (float(value) for value in np.percentile(losses, [5, 50, 95]))
    tail = es_block(losses, seed=seed)
    return {
        "p05": p05,
        "p50": p50,
        "p95": p95,
        "es95": max(tail["es95"], p95),
        "es95_interval": tail["interval"],
        "trimmed_es95": tail["trimmed_es95"],
        "trim": tail["trim"],
        "horizon_month": horizon,
    }
