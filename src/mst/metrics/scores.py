"""The judgement of the scored rows, the same for the episodes and the rolling grid.

A check gives cells (``mst.metrics.levels``) and turns them into rows; ``judge`` reads the rows
the same way whatever the check:

- the sanity of every quantity (``mst.metrics.sanity``), first;
- per level, the relative CRPS of every pair of forecasts the rows carry (``PAIRS``), with the
  interval over resampled windows — single windows for the episodes, moving blocks as long as
  a window for the overlapping dates of the grid — the calibration of the facts and the
  coverage of the band;
- the same per half: the models are fitted on the tuning windows and judged on the validation
  ones, and the full table stays over every window;
- per quantity, the ratios, the calibration and the sharpness.

A relative score is the mean CRPS of the first forecast over the second on the same cells,
pooled in logarithms with equal weights over the quantities of a level and the months
(``mst.metrics.quality``); below one the first forecast wins.
"""

from collections.abc import Mapping, Sequence

import numpy as np

from .cells import BUCKET_EDGES, Cell, crps, finite_draws
from .quality import interval, relative
from .sanity import check

PAIRS = (
    ("model", "conditional_walk"),
    ("model", "walk"),
    ("base", "walk"),
    ("model", "base"),
    ("model", "naive"),
)
"""The forecast against the walk that knows the scenario's path, against the plain walk, the
base run against the walk, the forecast against the base run (what the scenario adds) and, for
drivers, against the naive forecast of a year earlier. A pair the rows do not carry is left out:
the empty scenario has no base run apart from itself and fixes no path."""
HORIZONS = (1, 3, 6, 12)
"""The horizon months the calibration is reported at, beside the pool of them all."""
REPETITIONS = 1000
"""The repetitions of every bootstrap of a score; read at call time, so a test may lower it."""
MC_STREAM = 13
"""The second word of the Monte-Carlo seed, apart from the stream of the window bootstrap."""

Rows = Sequence[Mapping[str, object]]


def judge(
    found: list[Cell], rows: list[dict], *, unit: str, block: int, seed: int, repetitions: int
) -> dict:
    """The sanity, the scores per level, the halves and the quantities of one check.

    ``unit`` names the window key of the rows; a row names its half.
    """
    halves = {"all": rows}
    for half in ("tuning", "validation"):
        chosen = [row for row in rows if row["half"] == half]
        if chosen:
            halves[half] = chosen
    levels = summary(rows, unit=unit, block=block, seed=seed, repetitions=repetitions)
    return {
        "sanity": check(rows, benchmark="walk"),
        "levels": levels,
        "halves": {
            name: {
                "units": sorted({str(row[unit]) for row in chosen}),
                "cells": len(chosen),
                "sanity": check(chosen, benchmark="walk")["admission"],
                "levels": summary(
                    chosen, unit=unit, block=block, seed=seed, repetitions=repetitions
                ),
            }
            for name, chosen in halves.items()
        },
        "quantities": quantities(found, rows, levels),
    }


def summary(rows: Rows, *, unit: str, block: int, seed: int, repetitions: int) -> dict:
    """Per level: the pairs with their intervals, the calibration and the coverage."""
    found = {}
    for level in sorted({str(row["level"]) for row in rows}):
        own = [row for row in rows if row["level"] == level]
        pairs = {}
        for first, second in PAIRS:
            score = relative(own, first, second)
            if score["score"] is None:
                continue
            pairs[f"{first}/{second}"] = {
                **score,
                "interval": interval(
                    own,
                    first,
                    second,
                    unit=unit,
                    block=block,
                    repetitions=repetitions,
                    seed=seed,
                ),
            }
        found[level] = {
            "units": len({row[unit] for row in own}),
            "cells": len(own),
            "pairs": pairs,
            "calibration": calibration(own),
            "calibration_by_horizon": {
                t: calibration([row for row in own if row["month"] == t])
                for t in HORIZONS
                if any(row["month"] == t for row in own)
            },
            "coverage": float(np.mean([row["inside"] for row in own])),
        }
    return found


def calibration(rows: Rows) -> list[float]:
    """The shares of the facts below p5, p5..p50, p50..p95 and above p95 of the forecast."""
    transforms = np.array([row["pit"] for row in rows])
    buckets = np.bincount(np.searchsorted(BUCKET_EDGES, transforms, side="right"), minlength=4)
    return (buckets / len(rows)).tolist()


def quantities(found: list[Cell], rows: Rows, levels: Mapping[str, dict]) -> dict:
    """Per quantity: the cells, the ratios, the calibration and the sharpness.

    A constrained cell is counted and never scored. Sharpness is the median width of the band,
    meaningful only beside the calibration.
    """
    table = {}
    for quantity in sorted({cell.quantity for cell in found}):
        own_cells = [cell for cell in found if cell.quantity == quantity]
        own = [row for row in rows if row["quantity"] == quantity]
        level = own_cells[0].level
        entry = {
            "level": level,
            "cells": len(own_cells),
            "constrained": sum(1 for cell in own_cells if cell.constrained),
            "scored": len(own),
            "unverified_tail": any(cell.unverified for cell in own_cells),
            "ratios": {
                name: pair["by_quantity"].get(quantity)
                for name, pair in levels.get(level, {}).get("pairs", {}).items()
            },
            "calibration": calibration(own) if own else None,
            "coverage": float(np.mean([row["inside"] for row in own])) if own else None,
            "sharpness": float(np.median([row["width"] for row in own])) if own else None,
        }
        table[quantity] = entry
    return table


def monte_carlo(
    found: list[Cell], rows: Rows, *, unit: str, level: str, seed: int, repetitions: int
) -> dict:
    """The Monte-Carlo interval of the forecast against the walk at ``level``.

    The cells of one calculation share the draws, so a repetition resamples one index vector
    per number of draws and applies it to every cell; the CRPS of the forecast and the relative
    score are recomputed. This measures the error of the finite number of paths, apart from the
    error of the sample of windows.
    """
    keyed = {
        (row[unit], row["month"], row["quantity"]): row
        for row in rows
        if row["level"] == level and row["walk"] is not None
    }
    pairs = [
        (cell, keyed[key])
        for cell in found
        if (key := (cell.unit, cell.month, cell.quantity)) in keyed
    ]
    point = relative([row for _, row in pairs], "model", "walk")["score"]
    if point is None:
        return {"point": None, "interval": None}
    rng = np.random.default_rng([seed, MC_STREAM])
    sizes = sorted({len(cell.draws) for cell, _ in pairs})
    logs = []
    for _ in range(repetitions):
        index = {size: rng.integers(0, size, size) for size in sizes}
        resampled = [
            {**row, "model": crps(finite_draws(cell.draws[index[len(cell.draws)]]), cell.fact)}
            for cell, row in pairs
        ]
        score = relative(resampled, "model", "walk")["score"]
        if score is not None:
            logs.append(np.log(score))
    low, high = np.exp(np.percentile(logs, [2.5, 97.5]))
    return {"point": point, "interval": (float(low), float(high))}
