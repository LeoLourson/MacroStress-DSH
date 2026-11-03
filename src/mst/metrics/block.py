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


def block_weights(count: int, block: int, rng) -> np.ndarray:
    """How many times each of ``count`` units is drawn, in moving blocks of consecutive units.

    The blocks start at random and join until they cover ``count`` units; the last is cut. A
    block of one is the ordinary bootstrap, and a block no shorter than the sample keeps it whole.
    """
    size = min(max(int(block), 1), count)
    starts = rng.integers(0, count - size + 1, size=int(np.ceil(count / size)))
    picked = (starts[:, None] + np.arange(size)).ravel()[:count]
    return np.bincount(picked, minlength=count).astype(float)


def block_interval(values, keys, *, block: int = 12, repetitions: int = REPETITIONS, seed: int = 0):
    """``(low, high)`` of the mean of ``values``, resampling blocks of consecutive ``keys``.

    ``keys`` name the row's date or episode: rows sharing a key stay together, and a block
    is a run of ``block`` consecutive keys in the order they first appear. A single key
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
        counts = block_weights(len(ordered), block, rng).astype(int)
        drawn = np.concatenate([ordered[k] for k in np.repeat(np.arange(len(ordered)), counts)])
        # The resampled sample keeps the length of the original: the last block is cut.
        draws[repetition] = drawn[: len(values)].mean()
    return (
        float(np.nanpercentile(draws, 2.5)),
        float(np.nanpercentile(draws, 97.5)),
    )
