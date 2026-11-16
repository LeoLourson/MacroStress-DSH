"""The sanity of a check, read before any score: is each quantity's forecast a sane number?

A score averaged over quantities hides the quantity that broke: a band twice as wide as the
whole history of the series, a forecast pushed away from the fact by the scenario, or an
aggregate that rests on one quantity. This module reads the scored rows of either check — the
episodes or the rolling grid, of any level — and answers per quantity:

- **admission**: the share of cells whose ``[p5, p95]`` band leaves the admissible domain, and
  the share whose band is wider than the whole range of the history known on the date. Each
  quantity must stay under ``ADMISSION_SHARE`` on each rule; one quantity over it fails the
  check, whatever the pooled share is;
- **width**: the median ratio of the band width to that range — a number a reader can judge
  without knowing the unit of the series;
- **direction**: where the scenario moved the median from the base run, the share of cells it
  moved towards the fact. Only a move larger than ``MOVED`` of the base band counts; below it
  the move is Monte-Carlo noise. Under one half the scenario pushes the forecast the wrong way;
- **weight**: the score against the level's plain benchmark, and the score of the level without
  the quantity beside it. When one quantity moves the aggregate, the reader sees which one.

The plain benchmark of a level is the first of ``BENCHMARKS`` its rows carry: the random walk
of a series, the historical simulation of a return (``benchmark_of``).

A row carries ``quantity``, ``level``, ``month``, the CRPS columns (``model`` and the
benchmarks') and the band columns: ``width`` (``p95 - p5``), ``history_range``, a boolean per
admission rule and, in the episodes, ``shift`` (median less the base run's median), ``miss``
(fact less the base run's median) and ``base_width`` (the base run's band).
"""

from collections.abc import Mapping, Sequence

import numpy as np

from .baselines import DOMAIN, WIDTH
from .quality import relative

ADMISSION_SHARE = 0.05
"""A quantity is admitted when fewer than this share of its cells break each admission rule."""
RULES = (DOMAIN, WIDTH)
MOVED = 0.1
"""The smallest move of the median, as a share of the base band, the direction reads."""
BENCHMARKS = ("walk", "historical")
"""The plain benchmarks, in the order a level's is chosen: a series is judged against the
random walk, a return against the historical simulation."""

Rows = Sequence[Mapping[str, object]]


def benchmark_of(rows: Rows) -> str | None:
    """The plain benchmark the rows carry: the first of ``BENCHMARKS`` with a value."""
    for name in BENCHMARKS:
        if any(row.get(name) is not None for row in rows):
            return name
    return None


def check(rows: Rows) -> dict:
    """The table of the quantities and the admission it gives."""
    table = quantities(rows)
    return {"quantities": table, "admission": admission(rows, table)}


def quantities(rows: Rows) -> list[dict]:
    """One entry per level and quantity, in that order."""
    groups: dict[tuple[str, str], list] = {}
    for row in rows:
        groups.setdefault((str(row["level"]), str(row["quantity"])), []).append(row)
    table = []
    for (level, quantity), own in sorted(groups.items()):
        others = [row for row in rows if row["level"] == level and row["quantity"] != quantity]
        benchmark = benchmark_of([row for row in rows if row["level"] == level])
        shares = {rule: _share(own, rule) for rule in RULES}
        scored = benchmark is not None
        table.append(
            {
                "level": level,
                "quantity": quantity,
                "cells": len(own),
                "shares": shares,
                "failed": [
                    rule
                    for rule in RULES
                    if shares[rule] is not None and shares[rule] >= ADMISSION_SHARE
                ],
                "width_to_range": _width_to_range(own),
                "direction": _direction(own),
                "benchmark": benchmark,
                "score": relative(own, "model", benchmark)["score"] if scored else None,
                "without": (
                    relative(others, "model", benchmark)["score"] if scored and others else None
                ),
            }
        )
    return table


def admission(rows: Rows, table: Sequence[Mapping[str, object]]) -> dict:
    """Per rule: the worst quantity's share, the quantities that fail and the pooled share.

    The check fails when any quantity fails any rule. The worst share is what the decision rule
    compares between two versions; the pooled share is reported for reference only.
    """
    rules = {}
    for rule in RULES:
        failing = [entry for entry in table if rule in entry["failed"]]
        known = [entry["shares"][rule] for entry in table if entry["shares"][rule] is not None]
        rules[rule] = {
            "worst": max(known, default=0.0),
            "pooled": _share(rows, rule),
            "quantities": [entry["quantity"] for entry in failing],
        }
    return {
        "threshold": ADMISSION_SHARE,
        "cells": len(rows),
        "rules": rules,
        "failed": any(rule["quantities"] for rule in rules.values()),
    }


def worst_shares(found: Mapping[str, object]) -> dict[str, float]:
    """The share per rule the decision rule reads: the worst quantity's."""
    return {rule: float(entry["worst"]) for rule, entry in found["rules"].items()}


def _share(rows: Rows, rule: str) -> float | None:
    """The share of the cells breaking ``rule`` among the cells it applies to; a row whose
    verdict is None — the width rule without a long enough history — is outside it."""
    applied = [bool(row[rule]) for row in rows if row[rule] is not None]
    return float(np.mean(applied)) if applied else None


def _width_to_range(rows: Rows) -> float | None:
    """The median band width over the history range, on the cells whose range is positive."""
    ratios = [
        float(row["width"]) / float(row["history_range"])
        for row in rows
        if np.isfinite(row["history_range"]) and row["history_range"] > 0
    ]
    return float(np.median(ratios)) if ratios else None


def _direction(rows: Rows) -> dict | None:
    """The share of the moved cells the scenario moved towards the fact; None without a base."""
    chosen = [row for row in rows if row.get("shift") is not None]
    if not chosen:
        return None
    moved = [
        row for row in chosen if abs(row["shift"]) > MOVED * row["base_width"] and row["miss"] != 0
    ]
    toward = sum(1 for row in moved if np.sign(row["shift"]) == np.sign(row["miss"]))
    return {
        "moved": len(moved),
        "cells": len(chosen),
        "toward": toward / len(moved) if moved else None,
    }
