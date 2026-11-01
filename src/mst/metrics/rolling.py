"""The rolling check over the dates of history: the uniformity at ``s_0`` and the price.

The episode backtest judges the conditional forecast on a handful of named events; two things
it cannot see, and this module measures both on a grid of dates the plan names
(``[rolling] step_months``):

**The uniformity at ``s_0``.**  At the empty scenario the forecast has the law ``P`` itself,
and on every date of history the fact of a month is distributed by that law — so the
probability integral transform of the facts is uniform, whatever the date. The report
collects the transform over every date, variable and horizon month where a fact is known and
gives the shares of its four buckets.

**The price.**  A family of events fixed before the check (``[[price_events]]``) is priced on
every date: the model estimated on the data known then gives ``pi = P(S)``, the facts tell
whether the event happened in the window that follows, and the mean logarithmic score of
``pi`` stands beside the score of the historical frequency — the share of the grid's earlier
windows where the event happened. The report also gives the table of the mean ``pi`` against
the realized frequency in groups by ``pi``, the block-bootstrap interval of the scores (the
windows overlap, so the draws are not independent), and the difference
``dist + log P(S)`` — the part of the price the parameter uncertainty carries.

The models of the grid are the same cached episode versions the backtest uses
(``backtest._episode_macro``): an estimation per date, whatever the events are.
"""

from dataclasses import dataclass
from datetime import date

import numpy as np

from .episodes import Episodes

BUCKETS = (0.05, 0.5, 0.95)
"""The canon's calibration buckets: below a, between a and the median, median to b, above b."""
BLOCK = 12
"""The block length of the bootstrap, in grid dates: the windows overlap, so a block of
consecutive dates carries the dependence."""
REPETITIONS = 1000


@dataclass(frozen=True)
class Grid:
    """What the rolling check needs: the plan, the models and the facts."""

    dates: list[date]
    """The estimation dates of the grid, ascending."""
    versions: dict
    """The cached macro version of every date, as ``_episode_macro`` loads them."""
    forecasts: dict
    """The ``s_0`` simulation of every date, in the model scale."""
    facts: dict
    """The facts after every date in the model scale, by date: ``(month, variable) -> value``."""


def rolling_report(
    *,
    plugin,
    store,
    registry,
    root,
    plan: Episodes,
    known_on: date,
    macro_of,
) -> dict:
    """The rolling block of the report: the uniformity at ``s_0`` and the price.

    ``macro_of`` is the cached episode estimator (``backtest._episode_macro``): the rolling
    check never estimates a model itself. The report is empty when the plan names no rolling
    grid.
    """
    if plan.rolling is None:
        return {}
    grid = _grid(plugin, store, registry, root, plan, known_on, macro_of)
    report: dict = {"step_months": plan.rolling.step_months, "dates": len(grid.dates)}
    report["base_uniformity"] = _uniformity(grid, plan)
    if plan.rolling.price_events:
        report["price"] = _price(grid, plan, known_on)
    return report


def _grid(plugin, store, registry, root, plan, known_on, macro_of) -> Grid:
    """The dates of the grid with their model versions, ``s_0`` simulations and facts."""
    from mst.macro.panel import shift
    from mst.model_sdk.contracts import ScenarioPlan

    from .backtest import _realized, calculation_date  # the episode machinery, reused

    # The grid runs over the months the store knows; a model needs two years of history behind
    # its date, whatever the events are.
    months = sorted(
        {row.month for variable in registry.variables for row in store.as_of(variable.id, known_on)}
    )
    dates = months[24 :: plan.rolling.step_months]
    reach = max([plan.after_months, *(event.horizon for event in plan.rolling.price_events)])
    versions, forecasts, facts = {}, {}, {}
    for when in dates:
        as_of = calculation_date(when)
        version = macro_of(plugin, store, registry, as_of, root)
        versions[when] = version
        forecasts[when] = version.plugin.simulate(
            version.model,
            ScenarioPlan(horizon=plan.after_months),
            draws=plan.draws,
            seed=plan.seed,
        )
        # The facts in the scale of the model: a seasonal variable loses the calendar profile
        # the panel lost, on the date's own estimate.
        panel_months = list(version.panel.months)
        ahead = [shift(panel_months[-1], k) for k in range(1, reach + 1)]
        realized = _realized(
            store, registry, version.panel, panel_months, ahead, as_of, known_on, plan
        )
        facts[when] = {
            (month, id_): float(realized[t, i])
            for t, month in enumerate(ahead)
            for i, id_ in enumerate(version.panel.ids)
            if np.isfinite(realized[t, i])
        }
    return Grid(dates=dates, versions=versions, forecasts=forecasts, facts=facts)


def _uniformity(grid: Grid, plan: Episodes) -> dict:
    """The probability integral transform of the facts under the ``s_0`` forecasts.

    For every date, variable and horizon month with a fact, the transform is the share of
    the forecast's paths below the fact; a correct model makes it uniform over the grid, and
    the report gives the canon's four bucket shares with the count behind them.
    """
    transforms: list[float] = []
    for when in grid.dates:
        simulation = grid.forecasts[when]
        for k, id_ in enumerate(simulation.ids):
            horizon = simulation.paths.shape[1] - 1
            for t in range(1, horizon + 1):
                month = simulation.months[t]
                value = _fact_of(grid, when, month, id_)
                if value is None or not np.isfinite(value):
                    continue
                column = simulation.paths[:, t, k]
                transforms.append(float(np.mean(column <= value)))
    if not transforms:
        return {"cells": 0, "shares": None}
    values = np.asarray(transforms)
    edges = np.concatenate([[0.0], BUCKETS, [1.0]])
    shares = [
        float(np.mean((values >= edges[k]) & (values < edges[k + 1])))
        for k in range(len(edges) - 1)
    ]
    shares[-1] += float(np.mean(values == 1.0))
    return {
        "cells": len(values),
        "shares": shares,
        "expected": [0.05, 0.45, 0.45, 0.05],
        "note": (
            "the empty scenario's forecast has the law P itself, so the transform of the "
            "facts is uniform over every date of the grid"
        ),
    }


def _fact_of(grid: Grid, when: date, month: date, id_: str):
    """The realized value of a variable in a horizon month, or None when it is unknown."""
    return grid.facts[when].get((month, id_))


def _price(grid: Grid, plan: Episodes, known_on: date) -> dict:
    """The price check: the logarithmic score of ``P(S)`` against the historical frequency.

    Every event of the family is priced on every date of the grid, its outcome is read from
    the facts of the window that follows, and the score of the model's probability stands
    beside the score of the frequency — the share of the earlier windows where the event
    happened. The windows overlap, so the intervals come from a block bootstrap, and the
    difference ``dist + log P(S)`` reports the part of the price the parameter uncertainty
    carries.
    """
    from mst.model_sdk.contracts import LinearConstraint, LinearTerm, ScenarioPlan
    from mst.model_sdk.dist import DIST_STREAM, price_of_law

    rows = []
    for event in plan.rolling.price_events:
        terms = tuple(
            LinearTerm(
                variable=term["variable"], month=term["month"], weight=term["weight"]
            )
            for term in event.terms
        )
        constraint = LinearConstraint(terms=terms, low=event.low, high=event.high)
        for when in grid.dates:
            version = grid.versions[when]
            if not set(term.variable for term in terms) <= set(version.ids):
                continue
            law = version.plugin.constraint_law(
                version.model,
                ScenarioPlan(horizon=event.horizon, constraints=(constraint,)),
                draws=max(plan.draws // 4, 100),
                seed=plan.seed,
            )
            if law is None:
                continue
            distance, log_probability, _ = price_of_law(
                law,
                np.array([constraint.low]),
                np.array([constraint.high]),
                rng=np.random.default_rng([plan.seed, DIST_STREAM]),
            )
            happened = _happened(grid, when, terms, constraint)
            if happened is None:
                continue
            rows.append(
                {
                    "event": event.id,
                    "date": when.isoformat(),
                    "pi": float(np.exp(log_probability)),
                    "outcome": bool(happened),
                    "parameter_part": float(distance + log_probability),
                }
            )
    if not rows:
        return {"events": 0}
    return {
        "events": len({row["event"] for row in rows}),
        "rows": rows,
        "scores": _scores(rows),
        "groups": _groups(rows),
        "note": (
            "the logarithmic score of P(S) beside the score of the historical frequency; "
            "the intervals are block bootstraps, the windows overlap"
        ),
    }


def _happened(grid: Grid, when: date, terms, constraint):
    """Whether the event's box held in the window that follows the date; None without facts."""
    from mst.macro.panel import shift

    last = grid.versions[when].panel.months[-1]
    total = 0.0
    for term in terms:
        value = _fact_of(grid, when, shift(last, term.month), term.variable)
        if value is None or not np.isfinite(value):
            return None
        total += term.weight * value
    return constraint.low <= total <= constraint.high


def _scores(rows: list[dict]) -> dict:
    """The mean logarithmic scores of the model and the frequency, with block intervals.

    The frequency of an event on a date is the share of the earlier windows of the same
    event where it happened — a running rate, one number per date — and the frequency score
    clips it away from zero and one, where the logarithm would not exist.
    """
    from .block import block_interval

    model, frequency, outcomes, dates = [], [], [], []
    by_event: dict[str, list[dict]] = {}
    for row in rows:
        by_event.setdefault(row["event"], []).append(row)
    for found in by_event.values():
        seen = 0
        hits = 0
        for row in found:
            rate = (hits + 0.5) / (seen + 1.0)
            frequency.append(
                -row["outcome"] * np.log(rate) - (1 - row["outcome"]) * np.log(1 - rate)
            )
            model.append(
                -row["outcome"] * np.log(max(row["pi"], 1e-12))
                - (1 - row["outcome"]) * np.log(max(1 - row["pi"], 1e-12))
            )
            outcomes.append(row["outcome"])
            dates.append(row["date"])
            seen += 1
            hits += row["outcome"]
    return {
        "model": float(np.mean(model)),
        "frequency": float(np.mean(frequency)),
        "model_interval": block_interval(model, dates),
        "frequency_interval": block_interval(frequency, dates),
        "windows": len(model),
    }


def _groups(rows: list[dict]) -> list[dict]:
    """The mean ``pi`` against the realized frequency, in groups by ``pi``."""
    edges = (0.0, 0.02, 0.05, 0.2, 1.0)
    found = []
    for k in range(len(edges) - 1):
        inside = [
            row for row in rows if edges[k] <= row["pi"] < edges[k + 1]
        ]
        if not inside:
            continue
        found.append(
            {
                "pi": f"{edges[k]:.2f}-{edges[k + 1]:.2f}",
                "mean_pi": float(np.mean([row["pi"] for row in inside])),
                "frequency": float(np.mean([row["outcome"] for row in inside])),
                "windows": len(inside),
            }
        )
    return found
