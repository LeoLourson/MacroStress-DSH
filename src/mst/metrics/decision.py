"""Whether a candidate version replaces the incumbent (``docs/concept/success-criteria.md``).

The candidate replaces the incumbent when the candidate is trusted (``mst.metrics.trust``) and
either the incumbent is not, or the candidate is better than the incumbent on some check beyond
its error and worse on none. Better and worse read the CRPS of the two versions on the same
cells of the target level, pairwise, on the validation half when the check has one: the upper
bound of the log ratio below nought is better, the lower bound above nought worse, the bounds
at the criterion's share of the error budget. A difference inside the bounds is not proven,
and the incumbent stays.

Two versions judged on different cells are an error naming the cells: a comparison on the
intersection would hide what one of them could not forecast.
"""

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.stats import norm

from .quality import paired
from .trust import DECISION_SHARE

Rows = Sequence[Mapping[str, object]]


def compare(
    checks: Mapping[str, tuple[Rows, Rows, str, int]],
    *,
    trusted: bool,
    incumbent_trusted: bool,
    level: str,
    seed: int,
) -> dict:
    """The replacement rule. ``checks`` maps a check's name to the candidate's rows, the
    incumbent's rows, the key of its window and the block of its bootstrap."""
    z = norm.isf(DECISION_SHARE)
    found = {}
    for name, (mine, theirs, unit, block) in checks.items():
        mine, theirs = _pick(mine, level), _pick(theirs, level)
        if not mine:
            continue
        _same_cells(mine, theirs, unit)
        ratio = paired(mine, theirs, "model", unit=unit, block=block, seed=seed)
        if ratio["point"] is None or ratio["log_se"] is None:
            found[name] = {"ratio": ratio["point"], "better": False, "worse": False}
            continue
        point, se = float(np.log(ratio["point"])), ratio["log_se"]
        found[name] = {
            "ratio": ratio["point"],
            "low": float(np.exp(point - z * se)),
            "high": float(np.exp(point + z * se)),
            "better": point + z * se < 0,
            "worse": point - z * se > 0,
            "cells": len(mine),
        }
    better = any(entry["better"] for entry in found.values())
    worse = any(entry["worse"] for entry in found.values())
    replaces = trusted and (not incumbent_trusted or (better and not worse))
    if not trusted:
        reason = "the candidate is not trusted"
    elif not incumbent_trusted:
        reason = "the candidate is trusted and the incumbent is not"
    elif replaces:
        reason = "better on a check beyond its error and worse on none"
    else:
        reason = "no difference beyond the error, or worse on a check: the incumbent stays"
    return {
        "level": level,
        "checks": found,
        "trusted": trusted,
        "incumbent_trusted": incumbent_trusted,
        "replaces": replaces,
        "reason": reason,
    }


def _pick(rows: Rows, level: str) -> list:
    chosen = [row for row in rows if row["level"] == level]
    validation = [row for row in chosen if row["half"] == "validation"]
    return validation or chosen


def _same_cells(mine: Rows, theirs: Rows, unit: str) -> None:
    def key(row):
        return (str(row[unit]), int(row["month"]), str(row["quantity"]))

    difference = {key(row) for row in mine} ^ {key(row) for row in theirs}
    if difference:
        sample = ", ".join(f"{u}/{q}/t{t}" for u, t, q in sorted(difference)[:10])
        more = "..." if len(difference) > 10 else ""
        raise ValueError(f"the two reports do not share {len(difference)} cells: {sample}{more}")
