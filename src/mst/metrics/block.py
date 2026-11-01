"""The block bootstrap: an interval of a mean when neighbours are dependent.

The rolling windows of the price check overlap, so their scores are not independent draws
and the ordinary bootstrap understates the interval. A block bootstrap resamples whole
blocks of consecutive dates instead of single rows: the dependence inside a block survives,
and the interval reads the spread of the mean over the blocks. ``wild.py`` keeps the wild
cluster bootstrap of the episode scores; this module keeps the block one — two tools, two
names.
"""

import numpy as np

REPETITIONS = 1000


def block_interval(values, keys, *, block: int = 12, repetitions: int = REPETITIONS, seed: int = 0):
    """``(low, high)`` of the mean of ``values``, resampling blocks of consecutive ``keys``.

    ``keys`` name the row's date or episode: rows sharing a key stay together, and the
    blocks are runs of consecutive keys in the order they first appear. A single block
    covers everything and the interval is the plain mean's degenerate one.
    """
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return None
    order, seen = [], {}
    for key in keys:
        if key not in seen:
            seen[key] = len(order)
            order.append(key)
    groups = {}
    for position, key in enumerate(keys):
        groups.setdefault(seen[key], []).append(values[position])
    ordered = [np.asarray(groups[index]) for index in sorted(groups)]
    if len(ordered) <= 1:
        mean = float(values.mean())
        return (mean, mean)
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions)
    for repetition in range(repetitions):
        picked = [ordered[int(rng.integers(0, len(ordered)))] for _ in range(len(ordered))]
        drawn = np.concatenate(picked)
        # The resampled sample keeps the length of the original: the last block is cut.
        draws[repetition] = drawn[: len(values)].mean()
    return (
        float(np.nanpercentile(draws, 2.5)),
        float(np.nanpercentile(draws, 97.5)),
    )
