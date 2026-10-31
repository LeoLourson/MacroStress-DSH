"""The equal-weight aggregate of section 5, its utility split and the Monte-Carlo interval.

``equal_weight`` pools scored backtest cells the way section 5 prescribes: first the mean
across episodes inside a quantity at a month, then the mean with equal weights — one per
quantity, not per cell — across the quantities of a level at that month, then the mean across
the horizons of the level. The columns are the model, its baseline, base-run and naive
comparisons and the paired differences against the baseline and the base run; every level
aggregate carries counters of the rows behind it.

``utility_by_episode`` splits the paired improvement over the base run by the source of the
scenario markup, ``manual`` against ``llm`` (section 5): the share of episodes whose mean
paired difference is negative, and the mean of those per-episode means.

``mc_interval`` bootstrap-resamples the draws of the cells — one shared index vector per
repetition, so the cells of one calculation keep their shared realisations — and returns the
2.5/97.5 percentile interval of the equal-weight aggregate of the resampled CRPS.
"""

from collections.abc import Mapping, Sequence

import numpy as np

COLUMNS = ("model", "baseline", "base", "naive", "diff_vs_baseline", "diff_vs_base")
"""The aggregate columns: four scored comparisons and two paired differences."""
SOURCES = ("manual", "llm")
"""The two kinds of scenario markup that ``utility_by_episode`` compares."""
MC_STREAM = 13
"""The fixed second word of the ``mc_interval`` seed key: change it here, not at call sites."""
BOOTSTRAP_STREAM = 17
"""The fixed second word of the ``aggregate_bootstrap`` seed key, apart from ``MC_STREAM``."""


def _row_values(row: Mapping[str, object]) -> dict[str, float | None]:
    """The six aggregate columns of one scored row, its paired differences included."""
    model = row.get("normalized")
    baseline = row.get("baseline_normalized")
    base = row.get("base_normalized")
    naive = row.get("naive_normalized")
    return {
        "model": None if model is None else float(model),
        "baseline": None if baseline is None else float(baseline),
        "base": None if base is None else float(base),
        "naive": None if naive is None else float(naive),
        "diff_vs_baseline": (
            None if model is None or baseline is None else float(model) - float(baseline)
        ),
        "diff_vs_base": None if model is None or base is None else float(model) - float(base),
    }


def _pool(members: Sequence[Mapping[str, float | None]]) -> dict[str, float | None]:
    """The mean of each column over the members that carry a value; None without any."""
    pooled: dict[str, float | None] = {}
    for column in COLUMNS:
        values = [member[column] for member in members if member.get(column) is not None]
        pooled[column] = sum(values) / len(values) if values else None
    return pooled


def _counters(values: Sequence[Mapping[str, float | None]]) -> dict[str, dict[str, int]]:
    """Row counters per column: ``cells``/``scored`` count rows, ``normalized`` the valued."""
    total = len(values)
    return {
        column: {
            "cells": total,
            "scored": total,
            "normalized": sum(1 for value in values if value[column] is not None),
        }
        for column in COLUMNS
    }


def equal_weight(rows: Sequence[Mapping[str, object]], *, target_level: str) -> dict:
    """The equal-weight aggregate of the scored cells, per level and horizon.

    ``rows`` are scored cell dictionaries with the keys ``episode``, ``month`` (1..H),
    ``quantity``, ``level`` and the normalised scores ``normalized``, ``baseline_normalized``,
    ``base_normalized``, ``naive_normalized`` (``None`` where a score does not exist); the
    paired differences ``diff_vs_baseline`` and ``diff_vs_base`` are formed here as
    ``normalized`` minus the comparison when both parts exist.

    The pooling follows section 5: the mean across episodes inside a quantity at a month
    (step 1), the mean with equal weights — one per quantity, not per cell — across the
    quantities of the level that have a value at that month (step 2), the mean across the
    months of the level (step 3). A column with no value anywhere in a group is ``None``.

    Returns ``{"levels": {level: {"by_horizon": {month: {column: value}}, "overall":
    {column: value}, "counters": {column: {"cells": rows of the level, "scored": rows of the
    level, "normalized": rows with a value of the column — for a difference, rows with both
    parts}}}}, "target": target_level}. Rows arrive already scored, so ``cells`` and
    ``scored`` agree; the counters count rows, keeping levels with different quantities
    comparable. Raises ``ValueError`` on empty ``rows`` or a ``target_level`` without cells.
    """
    if not rows:
        raise ValueError("rows is empty: there are no scored cells to aggregate")
    grouped: dict[str, dict[str, dict[int, list[dict[str, float | None]]]]] = {}
    for row in rows:
        values = _row_values(row)
        level = str(row["level"])
        quantity = str(row["quantity"])
        month = int(row["month"])
        grouped.setdefault(level, {}).setdefault(quantity, {}).setdefault(month, []).append(
            values
        )
    levels: dict[str, dict] = {}
    for level, quantities in grouped.items():
        months = sorted({month for per_month in quantities.values() for month in per_month})
        level_rows = [
            values
            for per_month in quantities.values()
            for month_values in per_month.values()
            for values in month_values
        ]
        by_horizon: dict[int, dict[str, float | None]] = {}
        for month in months:
            members = [
                _pool(quantities[quantity][month])
                for quantity in sorted(quantities)
                if month in quantities[quantity]
            ]
            by_horizon[month] = _pool(members)
        levels[level] = {
            "by_horizon": by_horizon,
            "overall": _pool(list(by_horizon.values())),
            "counters": _counters(level_rows),
        }
    if target_level not in levels:
        known = ", ".join(sorted(levels))
        raise ValueError(f"target_level {target_level!r} has no cells; levels present: {known}")
    return {"levels": levels, "target": target_level}


def utility_by_episode(
    rows: Sequence[Mapping[str, object]],
    sources: Mapping[str, str],
    *,
    target_level: str = "macro",
) -> dict:
    """The usefulness of the scenario by the source of its markup: ``manual`` against ``llm``.

    Section 5: the scenario is useful when the paired CRPS difference against the base run is
    negative; a markup by an LLM is trusted only in out-of-sample calculations, so the split
    is reported separately. Every episode in ``rows`` needs a source in ``sources`` — a
    ``ValueError`` names the missing ids — and a source outside ``manual``/``llm`` is also a
    ``ValueError``. Per level and source: ``share_negative`` is the share of episodes whose
    mean ``diff_vs_base`` is negative, ``mean_diff`` the mean of those per-episode means,
    ``episodes`` how many were judged, ``dropped`` how many episodes of the source have rows
    at the level but no paired difference there (they cannot be judged and are skipped). A
    level with no judged episodes carries ``None`` instead of the numbers. Returns
    ``{"by_source": {source: {"by_level": {level: {...}}, "target": {...}}}}`` where
    ``target`` repeats the ``target_level`` entry.
    """
    episode_ids = {str(row["episode"]) for row in rows}
    missing = sorted(episode_ids - set(sources))
    if missing:
        raise ValueError(f"episodes without a source in sources: {', '.join(missing)}")
    unknown = sorted({sources[episode] for episode in episode_ids} - set(SOURCES))
    if unknown:
        raise ValueError(f"sources outside {SOURCES}: {', '.join(unknown)}")
    seen: dict[tuple[str, str], set[str]] = {}
    diffs: dict[str, dict[str, dict[str, list[float]]]] = {}
    for row in rows:
        level = str(row["level"])
        episode = str(row["episode"])
        source = sources[episode]
        seen.setdefault((level, source), set()).add(episode)
        value = _row_values(row)["diff_vs_base"]
        if value is None:
            continue
        diffs.setdefault(level, {}).setdefault(source, {}).setdefault(episode, []).append(value)

    def entry(level: str, source: str) -> dict[str, float | int | None]:
        present = seen.get((level, source), set())
        per_episode = diffs.get(level, {}).get(source, {})
        means = [sum(values) / len(values) for values in per_episode.values()]
        negatives = sum(1 for mean in means if mean < 0)
        return {
            "share_negative": negatives / len(means) if means else None,
            "mean_diff": sum(means) / len(means) if means else None,
            "episodes": len(means),
            "dropped": len(present) - len(per_episode),
        }

    by_source: dict[str, dict] = {}
    for source in sorted({sources[episode] for episode in episode_ids}):
        by_level = {
            level: entry(level, source)
            for level in sorted({lv for lv, seen_source in seen if seen_source == source})
        }
        by_source[source] = {"by_level": by_level, "target": entry(target_level, source)}
    return {"by_source": by_source}


def episode_contributions(
    rows: Sequence[Mapping[str, object]], column: str, *, level: str
) -> dict[str, float]:
    """Per-episode additive contributions that sum to the level's equal-weight aggregate.

    The aggregate is ``sum_e c_e``: a cell enters with the weight ``1 / (M * Q_m * n_g)``,
    where ``M`` is the number of months the level has cells in, ``Q_m`` the quantities present
    at month ``m`` and ``n_g`` the episodes present in the quantity-month group ``g``. The wild
    cluster bootstrap of ``mst.metrics.wild`` reweights these contributions; they are not the
    per-episode means, because the equal weight is over quantities and horizons, not over cells.
    An empty mapping means the column has no value at the level.
    """
    grid = _level_grid(rows, [column], level=level)
    values = grid["values"][column]
    known = np.isfinite(values)
    if not known.any():
        return {}
    months = known.any(axis=(0, 2))
    quantities = known.any(axis=0)
    counts = known.sum(axis=0)
    weights = np.zeros_like(values)
    for month in np.flatnonzero(months):
        present = int(quantities[month].sum())
        for quantity in np.flatnonzero(quantities[month]):
            if counts[month, quantity]:
                weights[:, month, quantity] = 1.0 / (
                    int(months.sum()) * present * counts[month, quantity]
                )
    contributions = np.where(known, values * weights, 0.0).sum(axis=(1, 2))
    return {
        episode: float(value)
        for episode, value in zip(grid["episodes"], contributions, strict=True)
    }


def aggregate_bootstrap(
    rows: Sequence[Mapping[str, object]],
    columns: Sequence[str],
    *,
    level: str,
    repetitions: int = 1000,
    seed: int,
) -> dict[str, dict[str, float | None]]:
    """The equal-weight aggregate of one level with the sampling error of its episodes.

    ``rows`` are scored cell dictionaries (``episode``, ``month``, ``quantity``, ``level`` and
    the numeric ``columns``). The statistic is the one ``equal_weight`` reports for ``level``:
    the mean across episodes inside a quantity at a month, the mean with equal weights over
    the quantities present at that month, the mean across the months of the level. The
    episodes are resampled with replacement — the cells of one episode are dependent — and the
    statistic is recomputed on every repetition, so the interval belongs to the reported
    aggregate, not to a cell-weighted mean. Returns, per column, ``{"point", "se", "low",
    "high"}``: the point value, the standard deviation of the repetition values (the SE the
    two-step decision reads) and the 2.5/97.5 percentile interval. A column with no value at
    the level has all four ``None``. The stream is ``np.random.default_rng([seed,
    BOOTSTRAP_STREAM])``.
    """
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")
    grid = _level_grid(rows, columns, level=level)
    count = len(grid["episodes"])
    ones = np.ones(count)
    rng = np.random.default_rng([seed, BOOTSTRAP_STREAM])
    found: dict[str, dict[str, float | None]] = {}
    for column in columns:
        values = grid["values"][column]
        point = _pool_grid(values, ones)
        if point is None:
            found[column] = {"point": None, "se": None, "low": None, "high": None}
            continue
        samples = np.full(repetitions, np.nan)
        for repetition in range(repetitions):
            drawn = rng.integers(0, count, size=count)
            weights = np.bincount(drawn, minlength=count).astype(float)
            sample = _pool_grid(values, weights)
            if sample is not None:
                samples[repetition] = sample
        known = samples[np.isfinite(samples)]
        if len(known) < 2:
            found[column] = {"point": point, "se": None, "low": point, "high": point}
            continue
        low, high = (float(value) for value in np.percentile(known, [2.5, 97.5]))
        found[column] = {
            "point": point,
            "se": float(np.std(known, ddof=1)),
            "low": low,
            "high": high,
        }
    return found


def _level_grid(
    rows: Sequence[Mapping[str, object]], columns: Sequence[str], *, level: str
) -> dict:
    """The cells of one level as ``(episodes, months, quantities)`` arrays per column."""
    chosen = [row for row in rows if str(row["level"]) == level]
    if not chosen:
        known = ", ".join(sorted({str(row["level"]) for row in rows})) or "none"
        raise ValueError(f"level {level!r} has no rows; levels present: {known}")
    episodes = sorted({str(row["episode"]) for row in chosen})
    months = sorted({int(row["month"]) for row in chosen})
    quantities = sorted({str(row["quantity"]) for row in chosen})
    e_index = {episode: i for i, episode in enumerate(episodes)}
    m_index = {month: i for i, month in enumerate(months)}
    q_index = {quantity: i for i, quantity in enumerate(quantities)}
    shape = (len(episodes), len(months), len(quantities))
    values = {column: np.full(shape, np.nan) for column in columns}
    for row in chosen:
        i = e_index[str(row["episode"])]
        j = m_index[int(row["month"])]
        k = q_index[str(row["quantity"])]
        derived = _row_values(row)
        for column in columns:
            value = derived.get(column) if column in COLUMNS else row.get(column)
            if value is None:
                continue
            value = float(value)
            if np.isfinite(value):
                values[column][i, j, k] = value
    return {"episodes": episodes, "values": values}


def _pool_grid(values: np.ndarray, weights: np.ndarray) -> float | None:
    """One level aggregate of a grid: quantities equally at a month, then months equally."""
    known = np.isfinite(values)
    if not known.any():
        return None
    sized = np.where(known, weights[:, None, None], 0.0)
    sizes = sized.sum(axis=0)
    sums = (np.where(known, values, 0.0) * sized).sum(axis=0)
    quantity_mean = np.full(sums.shape, np.nan)
    np.divide(sums, sizes, out=quantity_mean, where=sizes > 0)
    present = np.isfinite(quantity_mean)
    month_sizes = present.sum(axis=1)
    month_totals = np.where(present, quantity_mean, 0.0).sum(axis=1)
    month_mean = np.full(month_totals.shape, np.nan)
    np.divide(month_totals, month_sizes, out=month_mean, where=month_sizes > 0)
    keep = np.isfinite(month_mean)
    return float(month_mean[keep].mean()) if keep.any() else None


def mc_interval(
    cells: Sequence[Mapping[str, object]], *, repetitions: int = 1000, seed: int
) -> tuple[float, float]:
    """The 95% interval of the equal-weight CRPS aggregate over resampled draws.

    ``cells`` carry the keys ``quantity``, ``month``, ``level``, ``draws`` — an ``(R,)``
    array — and ``fact``; every cell must share one ``R``, because each repetition draws a
    single index vector and applies it to all cells: the bootstrap runs over the realisations,
    so the cells of one calculation keep their shared draws. The ``month`` and ``level`` keys
    are accepted but not used: the statistic pools the CRPS within a quantity over all its
    cells, then averages the quantities with equal weights. Each resampled cell is scored as
    ``E|X - y| - E|X - X'| / 2`` from the sorted draws; the interval is the 2.5/97.5
    percentile of the per-repetition aggregates. The stream is
    ``np.random.default_rng([seed, MC_STREAM])``. Empty ``cells`` return ``(nan, nan)``.
    """
    if not cells:
        return float("nan"), float("nan")
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")
    sizes = {len(np.asarray(cell["draws"])) for cell in cells}
    if len(sizes) != 1:
        raise ValueError(f"cells must share one draws length R; got {sorted(sizes)}")
    (count,) = tuple(sizes)
    if count == 0:
        raise ValueError("cells carry empty draws")
    draws = np.stack([np.asarray(cell["draws"], dtype=float) for cell in cells])
    facts = np.array([float(cell["fact"]) for cell in cells])
    groups: dict[str, list[int]] = {}
    for index, cell in enumerate(cells):
        groups.setdefault(str(cell["quantity"]), []).append(index)
    quantity_slices = [np.array(indices, dtype=int) for indices in groups.values()]
    rng = np.random.default_rng([seed, MC_STREAM])
    stats = np.empty(repetitions)
    for repetition in range(repetitions):
        indices = rng.integers(0, count, size=count)
        scores = _crps_rows(draws[:, indices], facts)
        stats[repetition] = np.mean([scores[slice_].mean() for slice_ in quantity_slices])
    low, high = np.percentile(stats, [2.5, 97.5])
    return float(low), float(high)


def _crps_rows(sample: np.ndarray, facts: np.ndarray) -> np.ndarray:
    """The CRPS of each row of ``sample`` against its fact: ``E|X - y| - E|X - X'| / 2``.

    The pairwise term comes from the sorted rows and their prefix sums, so the cost per row
    is ``R log R``, the same estimator as ``cells.crps``.
    """
    x = np.sort(sample, axis=1)
    count = x.shape[1]
    first = np.abs(x - facts[:, None]).mean(axis=1)
    ranks = np.arange(count, dtype=float)
    prefix = np.concatenate([np.zeros((len(x), 1)), np.cumsum(x, axis=1)[:, :-1]], axis=1)
    pairwise = 2.0 * (ranks * x - prefix).sum(axis=1) / count**2
    return first - pairwise / 2
