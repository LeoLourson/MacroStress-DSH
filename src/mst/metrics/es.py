"""The expected shortfall of a run and the Acerbi-Székely Z: an ES backtest of the portfolio.

For every window ``e`` the backtest holds the realized portfolio loss ``l_e``, the predictive
``VaR_e`` — the P95 of the losses over the run's repetitions — and the predictive ``ES_e``, the
mean loss in that tail (``mst.plugin_sdk.expected_shortfall``). The statistic is

    Z = 1 / (0.05 * n) * sum_e l_e * 1{l_e > VaR_e} / ES_e - 1,

and its expectation is zero when the realized losses follow the predictive tails, so ``Z > 0``
reads the tails as understated. The comparison is strict: a loss exactly at ``VaR_e`` contributes
nothing. Beside it ``tail`` gives the trimmed ``Z``: ``VaR_e`` and ``ES_e`` recomputed with the
heaviest repetitions of the predictive law dropped (``trim_terms``). A non-finite term or a
non-positive ES raises rather than quietly entering the average; ``tail`` names such a window
instead.

The losses are the portfolio cells of a check (``mst.metrics.levels``) at the last month of
every window: minus the return of the portfolio, in percent of its current value. ``Z`` is a
ratio per window, so the unit of the loss does not change it.
"""

from collections.abc import Sequence

import numpy as np

from ..plugin_sdk import expected_shortfall
from .block import block_weights

TAIL = 0.05
"""The predictive tail share behind VaR and ES; the normalizer of Z and of every contribution."""

BOOTSTRAP_STREAM = 19
"""The stream of the window bootstrap of the contributions."""

#: The stream of the ES bootstrap of a single run, apart from every window stream.
RUN_BOOTSTRAP_STREAM = 23

#: The stream of the paired bootstrap of a comparison with the base, apart from every other.
PAIRED_STREAM = 29


def es_block(losses: np.ndarray, *, repetitions: int = 1000, seed: int = 0) -> dict:
    """The expected shortfall of one run's losses with what the canon asks beside it.

    The canon's section 7 asks an ES to carry an interval, the sensitivity to dropping the
    heaviest one percent of the paths and the horizon it belongs to; the interval is a
    bootstrap over the repetitions of the run, and the trimming cuts the heaviest paths
    before the shortfall is taken again. ``losses`` is one array — the portfolio loss of
    every repetition — and the horizon mark is the caller's to add.
    """
    values = np.asarray(losses, dtype=float)
    if values.size == 0:
        raise ValueError("the losses carry no repetition")
    es = float(expected_shortfall(values))
    rng = np.random.default_rng([seed, RUN_BOOTSTRAP_STREAM])
    draws = np.empty(repetitions)
    for repetition in range(repetitions):
        picked = values[rng.integers(0, values.size, values.size)]
        draws[repetition] = expected_shortfall(picked)
    cut = max(int(np.floor(values.size * 0.01)), 0)
    trimmed = float(expected_shortfall(np.sort(values)[:-cut])) if cut else es
    return {
        "es95": es,
        "interval": (
            float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)),
        ),
        "trimmed_es95": trimmed,
        "trim": 0.01,
    }


def value_at_risk(losses: np.ndarray, level: float = 0.95) -> float:
    """The ``level`` quantile of the losses: P95 by default, the episode's predictive ``VaR``."""
    values = np.asarray(losses, dtype=float)
    if values.size == 0:
        raise ValueError("the losses carry no repetition")
    return float(np.percentile(values, 100.0 * level))


def paired_difference(a, b, *, seed: int, replications: int = 1000) -> dict:
    """The differences ``a - b`` of the median, ``p95 - p5``, ``p95`` and ES95, with intervals.

    One set of indices is drawn per replication and applied to both samples: draw ``i`` of
    ``a`` and draw ``i`` of ``b`` come from the same random numbers, so whatever link the two
    samples have — the same random numbers of two runs — survives the resampling. A
    difference per draw is never reported; only the difference of a statistic carries, with
    its 95 % bootstrap interval. ``centre`` is the difference of medians, ``width`` the
    difference of ``p95 - p5``: the shift of the centre of the law apart from the change of
    its width, as the canon's comparison with ``s_0`` asks (section 6).
    """
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    if left.ndim != 1 or left.shape != right.shape or left.size == 0:
        raise ValueError(
            f"the comparison needs two samples of one shape, not {left.shape} and {right.shape}"
        )
    rng = np.random.default_rng([seed, PAIRED_STREAM])
    drawn = {name: np.empty(replications) for name in ("centre", "width", "p95", "es95")}
    for replication in range(replications):
        picked = rng.integers(0, left.size, left.size)
        first, second = left[picked], right[picked]
        drawn["centre"][replication] = np.median(first) - np.median(second)
        drawn["width"][replication] = _width(first) - _width(second)
        drawn["p95"][replication] = np.percentile(first, 95) - np.percentile(second, 95)
        drawn["es95"][replication] = expected_shortfall(first) - expected_shortfall(second)

    def block(name: str, value: float) -> dict:
        low, high = np.percentile(drawn[name], [2.5, 97.5])
        return {"value": float(value), "interval": [float(low), float(high)]}

    return {
        "centre": block("centre", np.median(left) - np.median(right)),
        "width": block("width", _width(left) - _width(right)),
        "p95": block("p95", np.percentile(left, 95) - np.percentile(right, 95)),
        "es95": block("es95", expected_shortfall(left) - expected_shortfall(right)),
    }


def _width(values: np.ndarray) -> float:
    """``p95 - p5``: the width the comparison with ``s_0`` reports beside the centre."""
    return float(np.percentile(values, 95) - np.percentile(values, 5))


def contributions(terms: Sequence[tuple[float, float, float]]) -> list[float]:
    """Every episode's contribution ``c_e = l_e * 1{l_e > VaR_e} / (0.05 * ES_e)`` to ``Z``.

    ``Z`` is their mean less one; the bootstrap of ``tail`` resamples these contributions, so
    the report gives an interval without valuing the windows again. A loss
    exactly at ``VaR_e`` contributes zero: the comparison is strict.
    """
    losses, vars_, ess = _columns(terms)
    return (np.where(losses > vars_, losses, 0.0) / (TAIL * ess)).tolist()


def z_statistic(terms: Sequence[tuple[float, float, float]]) -> float:
    """The Acerbi-Székely ``Z`` over the episodes' ``(loss, VaR, ES)`` triples.

    ``Z > 0`` says the realized losses exceed what the predictive tails promised, ``Z < 0`` the
    opposite; a calibrated tail keeps it near zero. Empty ``terms`` raise: the caller decides
    what a report without a single backtested episode shows.
    """
    found = contributions(terms)
    if not found:
        raise ValueError("no episodes to backtest: none were given or all were dropped")
    return float(np.mean(found)) - 1.0


def trim_terms(
    losses: Sequence[float], predictive: Sequence[np.ndarray], *, share: float = 0.01
) -> list[tuple[float, float, float]]:
    """The ``terms`` of the trimmed ``Z``: the heaviest ``share`` of the repetitions dropped.

    For every episode the top ``share`` of the predictive losses by magnitude is cut — the
    sensitivity the report shows beside ``Z`` — and ``VaR_e`` and ``ES_e`` are recomputed on
    what remains; the realized loss ``l_e`` is untouched. A sample so short that nothing is cut
    (fewer than ``1 / share`` repetitions) keeps its untrimmed ``VaR_e`` and ``ES_e``, which is
    the documented boundary: trimming below one repetition is not defined, and the trimmed ``Z``
    then equals the plain one for that episode.
    """
    if not 0 < share < 1:
        raise ValueError(f"the trimming share is a share of the tail, not {share}")
    rows = []
    for loss, draws in zip(losses, predictive, strict=True):
        values = np.asarray(draws, dtype=float)
        cut = int(np.floor(values.size * share))
        kept = np.sort(values)[:-cut] if cut else values
        rows.append((float(loss), value_at_risk(kept), expected_shortfall(kept)))
    return rows


def _columns(terms) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The losses, the VaRs and the ESs of the episodes, validated."""
    found = np.asarray(list(terms), dtype=float)
    if found.size == 0:
        return np.empty(0), np.empty(0), np.empty(0)
    if found.ndim != 2 or found.shape[1] != 3:
        raise ValueError("every episode is a (loss, VaR, ES) triple")
    losses, vars_, ess = found[:, 0], found[:, 1], found[:, 2]
    good = np.isfinite(losses) & np.isfinite(vars_) & np.isfinite(ess) & (ess > 0)
    if not np.all(good):
        raise ValueError(
            f"episode {int(np.flatnonzero(~good)[0])}: the loss and the VaR must be finite "
            "and the ES positive; drop such an episode by name instead"
        )
    return losses, vars_, ess


def tail(
    windows: Sequence[tuple[str, float, np.ndarray]],
    *,
    block: int,
    seed: int,
    repetitions: int,
    trim: float = 0.01,
) -> dict:
    """``Z`` over the windows' ``(id, realized loss, predictive losses)``, in the windows' order.

    ``interval`` is the percentile 95 % interval of ``Z`` over windows resampled in moving
    blocks of ``block`` — single windows for the episodes, as many dates as a window spans for
    the overlapping grid. ``trimmed_z`` recomputes ``VaR`` and ``ES`` with the heaviest ``trim``
    share of the predictive repetitions dropped. A window whose predictive ES is not a positive
    loss is named in ``dropped``; with no window left ``z`` and its interval are ``None``.
    """
    kept, dropped, terms, predictive = [], [], [], []
    for id_, loss, draws in windows:
        values = np.asarray(draws, dtype=float)
        es = expected_shortfall(values) if values.size else float("nan")
        if not (np.isfinite(loss) and np.all(np.isfinite(values)) and np.isfinite(es) and es > 0):
            dropped.append(
                {"id": id_, "reason": f"the predictive ES is {es:g}, not a positive loss"}
            )
            continue
        kept.append({"id": id_, "loss": float(loss), "var": value_at_risk(values), "es": es})
        terms.append((float(loss), value_at_risk(values), es))
        predictive.append(values)
    if not terms:
        return {"z": None, "interval": None, "trimmed_z": None, "windows": kept, "dropped": dropped}
    found = np.asarray(contributions(terms))
    rng = np.random.default_rng([seed, BOOTSTRAP_STREAM])
    samples = np.empty(repetitions)
    for repetition in range(repetitions):
        weights = block_weights(len(found), block, rng)
        samples[repetition] = float(np.sum(weights * found) / np.sum(weights)) - 1.0
    trimmed = trim_terms([term[0] for term in terms], predictive, share=trim)
    positive = all(np.isfinite(es) and es > 0.0 for _, _, es in trimmed)
    return {
        "z": z_statistic(terms),
        "interval": tuple(float(bound) for bound in np.percentile(samples, [2.5, 97.5])),
        "trimmed_z": z_statistic(trimmed) if positive else None,
        "trim": trim,
        "windows": kept,
        "dropped": dropped,
    }
