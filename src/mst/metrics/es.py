"""The Acerbi-Székely Z of section 5: an ES backtest of the portfolio's losses.

For every episode ``e`` the backtest holds the realized portfolio loss ``l_e``, the predictive
``VaR_e`` — the P95 of the losses over the run's repetitions — and the predictive ``ES_e``, the
mean loss in that tail (``mst.plugin_sdk.expected_shortfall``). The statistic is

    Z = 1 / (0.05 * n) * sum_e l_e * 1{l_e > VaR_e} / ES_e - 1,

and its expectation is zero when the realized losses follow the predictive tails, so ``Z > 0``
reads the tails as understated. The comparison is strict: a loss exactly at ``VaR_e`` contributes
nothing. ``Z`` is reported in the tail section only — the admission verdict of section 5 does not
read it. Beside it the report shows the trimmed ``Z``: ``VaR_e`` and ``ES_e`` recomputed with the
heaviest repetitions of the predictive law dropped (``trim_terms``). An episode without a realized
loss or without a positive predictive ES is dropped by the caller and counted by name; here a
non-finite term or a non-positive ES raises rather than quietly entering the average.

The predictive losses are the engine's values of the portfolio's positions on the episode's
conditional paths: the macro variables and the drivers come back in their own units
(``conditional_paths``), and the loss of a repetition is minus the sum of the positions'
``value_change`` in the portfolio's base currency (``portfolio_losses``). The realized loss is the
same valuation on the facts of the window (``realized_paths``, assumption 8 of the plan): the value
the calculation date knows at ``t = 0`` and the facts known on the check date at ``t = H``. An
episode whose facts are missing at either end of the window, whose realized loss is not finite, or
whose positions the engine could not calculate is dropped by name, never silently.
"""

from collections.abc import Sequence
from datetime import date

import numpy as np

from ..data.store import SeriesStore, shift
from ..drivers.bundle import SatelliteBundle
from ..drivers.registry import from_model
from ..engine.run import Batch
from ..engine.run import run as run_positions
from ..macro.panel import Panel
from ..macro.registry import load_registry
from ..plugin_sdk import FixedPaths, expected_shortfall
from ..portfolio.spec import Portfolio
from .cells import bootstrap
from .wild import disagrees, wild_interval

TAIL = 0.05
"""The predictive tail share behind VaR and ES; the normalizer of Z and of every contribution."""

BOOTSTRAP_STREAM = 19
"""The stream of the episode bootstrap of the contributions, apart from the wild bootstrap."""

DEFINITION = (
    "Z = 1/(0.05 n) sum_e l_e * 1{l_e > VaR_e} / ES_e - 1 over the episodes with a realized "
    "portfolio loss; VaR_e and ES_e are P95 and ES95 of the portfolio losses on the conditional "
    "paths, and l_e is the portfolio loss between the facts at t = 0 and t = H of the episode "
    "window"
)
"""What the tail section measures, as the report's ``meta`` states it."""


def value_at_risk(losses: np.ndarray, level: float = 0.95) -> float:
    """The ``level`` quantile of the losses: P95 by default, the episode's predictive ``VaR``."""
    values = np.asarray(losses, dtype=float)
    if values.size == 0:
        raise ValueError("the losses carry no repetition")
    return float(np.percentile(values, 100.0 * level))


def contributions(terms: Sequence[tuple[float, float, float]]) -> list[float]:
    """Every episode's contribution ``c_e = l_e * 1{l_e > VaR_e} / (0.05 * ES_e)`` to ``Z``.

    ``Z`` is their mean less one; the wild cluster bootstrap of the report resamples these
    contributions, so the report can give an interval without recomputing the episodes. A loss
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


def conditional_paths(
    simulation, driver_paths, bundle: SatelliteBundle, *, seed: int = 0
) -> FixedPaths:
    """The episode's conditional forecast in the own units the position models read.

    ``simulation`` is the macro forecast in the model scale and ``driver_paths`` the drivers the
    bundle simulated on it, also in the model scale. Every macro variable is mapped back through
    its registry transform (``variable.from_model``) and every driver through the transform of its
    quantity, exactly as ``mst.scenario.build`` reports a run; the drivers keep the flags their
    simulation set, and ``seed`` is the run's seed, recorded on the paths.
    """
    if len(simulation.months) != simulation.paths.shape[1]:
        raise ValueError(
            f"the simulation names {len(simulation.months)} months for "
            f"{simulation.paths.shape[1]} months of paths"
        )
    if driver_paths.paths.shape[:2] != simulation.paths.shape[:2]:
        raise ValueError(
            f"the drivers are {driver_paths.paths.shape[:2]}, the macro paths "
            f"{simulation.paths.shape[:2]}"
        )
    registry = load_registry()
    macros = {}
    for k, id_ in enumerate(simulation.ids):
        try:
            variable = registry.variable(id_)
        except KeyError:
            raise ValueError(
                f"the paths carry {id_}, which the macro registry does not know"
            ) from None
        macros[id_] = variable.from_model(simulation.paths[..., k])
    drivers = {}
    flags = {}
    for k, driver in enumerate(driver_paths.ids):
        try:
            quantity = bundle.registry.quantity(bundle.quantity_of(driver))
        except KeyError:
            raise ValueError(f"the satellite version knows no quantity of {driver}") from None
        drivers[driver] = from_model(quantity.transform, driver_paths.paths[..., k])
        flags[driver] = driver_paths.flags.get(driver, frozenset())
    return FixedPaths(
        macros=macros,
        drivers=drivers,
        driver_flags=flags,
        start=simulation.months[0],
        seed=seed,
    )


def realized_paths(
    store: SeriesStore,
    *,
    as_of: date,
    known_on: date,
    months: Sequence[date],
    panel: Panel,
    realized: np.ndarray,
    bundle: SatelliteBundle,
) -> tuple[FixedPaths | None, list[str]]:
    """The facts of the episode window in the own units the position models read.

    Month ``t = 0`` is the calculation month — the last month of ``panel``, checked against
    ``as_of`` — and holds what the calculation date knows: the panel's last row for a macro
    variable and the latest value of the driver's series known on ``as_of``. Months ``t = 1 .. H``
    are ``months`` exactly and hold the facts known on ``known_on``; the macro values come from
    ``realized`` ``(H, n)`` in the model scale, in the order of ``panel.ids``, and are mapped back
    through their registry transforms. Drivers in their own units are read from ``store``.

    ``missing`` names every ``<series or id>/<month>`` gap of the window, including gaps in the
    middle of it and at its ends, and every served driver without a series of its own. A gap
    stays NaN in the path rather than voiding the window: a position that reads that value
    cannot be valued, and the caller drops the episode by name — a driver nobody reads must not
    drop it. ``None`` — there is nothing to build: no macro variable and no driver at all.
    """
    start = _window_start(as_of, panel)
    window = list(months)
    expected = [shift(start, k) for k in range(1, len(window) + 1)]
    if window != expected:
        raise ValueError(
            f"the window {window[:1]}{'...' if window else ''} is not the months t = 1 .. H "
            f"after {start:%Y-%m}"
        )
    facts = np.asarray(realized, dtype=float)
    if facts.shape != (len(window), len(panel.ids)):
        raise ValueError(
            f"the realized values are {facts.shape}, the window {len(window)} months of "
            f"{len(panel.ids)} macro variables"
        )
    registry = load_registry()
    missing: list[str] = []
    macros = {}
    for i, id_ in enumerate(panel.ids):
        try:
            variable = registry.variable(id_)
        except KeyError:
            raise ValueError(
                f"the panel carries {id_}, which the macro registry does not know"
            ) from None
        column = np.concatenate([[panel.y[-1, i]], facts[:, i]])
        _record_missing(missing, id_, [start, *window], column)
        macros[id_] = variable.from_model(column)[None, :]
    series_of = {driver.id: driver.series for driver in bundle.registry.drivers}
    drivers = {}
    for driver in bundle.served():
        series = series_of.get(driver)
        column = np.full(len(window) + 1, np.nan)
        if series is None:
            missing.append(f"{driver}/no series")
        else:
            known_today = store.as_of(series, as_of)
            if known_today:
                column[0] = float(known_today[-1].value)
            known_facts = {row.month: float(row.value) for row in store.as_of(series, known_on)}
            for t, month in enumerate(window, start=1):
                if month in known_facts:
                    column[t] = known_facts[month]
            _record_missing(missing, series, [start, *window], column)
        drivers[driver] = column[None, :]
    if not macros and not drivers:
        return None, missing
    return (
        FixedPaths(
            macros=macros,
            drivers=drivers,
            driver_flags={driver: frozenset() for driver in drivers},
            start=start,
        ),
        missing,
    )


def portfolio_run(portfolio: Portfolio, paths, *, drivers, served, as_of) -> Batch:
    """The engine's results of the portfolio's positions on ``paths``, every reason kept.

    ``drivers`` is the driver registry of the episode's satellite version and ``served`` the
    drivers it serves, exactly as ``mst.engine.run`` reads them. A position the engine could not
    calculate stays in the batch as ``invalid`` with its diagnostics; ``portfolio_exclusions``
    names it.
    """
    return run_positions(
        portfolio.positions,
        paths,
        drivers=drivers,
        served=served,
        base_currency=portfolio.base_currency,
        as_of=as_of,
    )


def portfolio_exclusions(batch: Batch) -> list[dict]:
    """The positions of the batch that were not calculated, each with its reason."""
    found = []
    for result in batch.results:
        if result.status != "invalid":
            continue
        reason = (
            "; ".join(item.message for item in result.diagnostics if item.severity == "error")
            or "; ".join(result.flags)
            or "the position was not calculated"
        )
        found.append({"position": result.position_id, "reason": reason})
    return found


def portfolio_losses(portfolio: Portfolio, paths, *, drivers, served, as_of) -> np.ndarray:
    """``(R,)`` losses of the portfolio on ``paths``, in the portfolio's base currency.

    The loss of a repetition is minus the sum of the calculated positions' ``value_change``.
    A position the engine could not calculate never enters the sum, and it is not lost silently:
    ``portfolio_run`` keeps it with its reason and ``portfolio_exclusions`` names it. An empty
    array — nothing was calculated — is the documented boundary; the caller drops the episode.
    """
    return _losses(portfolio_run(portfolio, paths, drivers=drivers, served=served, as_of=as_of))


def tail_section(
    rows: Sequence[dict],
    portfolio: Portfolio,
    *,
    seed: int,
    repetitions: int = 1000,
    trim: float = 0.01,
) -> dict:
    """The Acerbi-Székely tail section over the episodes of ``rows``.

    Every row carries ``id``, ``as_of``, ``known_on``, ``horizon``, ``months``, the conditional
    ``paths`` and the factual ``realized`` paths (``None`` when the facts are missing), the driver
    registry ``drivers`` and the served drivers ``served``. The predictive losses are the
    portfolio's on the conditional paths; the realized loss is its single value on the factual
    paths. ``VaR_e`` is P95 of the predictive losses and ``ES_e`` their ES95; ``Z`` and the
    contributions are the statistics of ``z_statistic`` and ``contributions``.

    ``interval`` is the percentile 95 % bootstrap over the episodes' contributions, drawn from a
    stream of its own (``[seed, BOOTSTRAP_STREAM]``); ``wild`` is the wild cluster bootstrap with
    Webb weights over the same contributions, and ``wild_disagrees`` says whether the two read
    the significance of a departure from zero differently. Both intervals are shifted by minus one
    to be intervals of ``Z``: a calibrated tail covers zero. ``trimmed_z`` recomputes ``VaR`` and
    ``ES`` with the heaviest ``trim`` share of the predictive repetitions dropped; it is ``None``
    when that leaves no positive tail to normalize by.

    An episode is dropped by name when the facts at an end of the window are missing, the realized
    loss is not finite, a position on either set of paths was not calculated, or the predictive
    ES is not a positive loss. With nothing to backtest, ``z`` and every interval are ``None`` and
    every row is in ``dropped`` — an empty section is a statement, not an error.
    """
    if repetitions < 1:
        raise ValueError(f"the bootstrap needs repetitions, not {repetitions}")
    if not 0 < trim < 1:
        raise ValueError(f"the trimming share is a share of the tails, not {trim}")
    found_rows = list(rows)
    ids = [str(row["id"]) for row in found_rows]
    repeated = sorted({id_ for id_ in ids if ids.count(id_) > 1})
    if repeated:
        raise ValueError(f"two episode rows share the id {', '.join(repeated)}")
    episodes: list[dict] = []
    dropped: list[dict] = []
    terms: list[tuple[float, float, float]] = []
    predictive: list[np.ndarray] = []
    for row in found_rows:
        id_ = str(row["id"])
        draws, why = _valued(portfolio, row, row["paths"], where="the conditional forecast")
        if why is not None:
            dropped.append({"id": id_, "reason": why})
            continue
        loss, why = _realized_loss(portfolio, row)
        if why is not None:
            dropped.append({"id": id_, "reason": why})
            continue
        if not np.all(np.isfinite(draws)):
            dropped.append({"id": id_, "reason": "the predictive losses are not finite"})
            continue
        var = value_at_risk(draws)
        es = expected_shortfall(draws)
        if not (np.isfinite(var) and np.isfinite(es) and es > 0.0):
            dropped.append(
                {"id": id_, "reason": f"the predictive ES is {es:g}, not a positive loss"}
            )
            continue
        episodes.append({"id": id_, "loss": loss, "var": var, "es": es})
        terms.append((loss, var, es))
        predictive.append(draws)
    if terms:
        z = z_statistic(terms)
        per_episode = {
            episode["id"]: [value]
            for episode, value in zip(episodes, contributions(terms), strict=True)
        }
        low, high = bootstrap(per_episode, repetitions=repetitions, seed=[seed, BOOTSTRAP_STREAM])
        interval = (low - 1.0, high - 1.0)
        low, high = wild_interval(per_episode, repetitions=repetitions, seed=seed)
        wild = (low - 1.0, high - 1.0)
        wild_disagrees = disagrees(interval, wild)
        trimmed = trim_terms([term[0] for term in terms], predictive, share=trim)
        positive = all(np.isfinite(es) and es > 0.0 for _, _, es in trimmed)
        trimmed_z = z_statistic(trimmed) if positive else None
    else:
        z = interval = wild = wild_disagrees = trimmed_z = None
    return {
        "z": z,
        "interval": interval,
        "wild": wild,
        "wild_disagrees": wild_disagrees,
        "trimmed_z": trimmed_z,
        "trim": trim,
        "episodes": episodes,
        "dropped": dropped,
        "horizon": _horizon(found_rows),
        "meta": {"definition": DEFINITION, "repetitions": repetitions, "seed": seed},
    }


def _valued(
    portfolio: Portfolio, row: dict, paths, *, where: str
) -> tuple[np.ndarray | None, str | None]:
    """The portfolio's losses on one set of paths of the row, or the reason it has none."""
    if paths is None:
        reason = f"{where}: no paths for the episode"
        missing = tuple(row.get("realized_missing") or ())
        if missing:
            reason += f" ({', '.join(str(name) for name in missing[:4])})"
        return None, reason
    try:
        batch = portfolio_run(
            portfolio,
            paths,
            drivers=row["drivers"],
            served=row["served"],
            as_of=row["as_of"],
        )
    except Exception as error:  # noqa: BLE001 - one episode must not stop the whole section
        return None, f"{where}: the position models failed ({type(error).__name__}: {error})"
    excluded = portfolio_exclusions(batch)
    if excluded:
        reason = "; ".join(f"{item['position']}: {item['reason']}" for item in excluded)
        return None, f"{where}: {reason}"
    losses = _losses(batch)
    if not losses.size:
        return None, f"{where}: no position was calculated"
    return losses, None


def _realized_loss(portfolio: Portfolio, row: dict) -> tuple[float | None, str | None]:
    """The single realized loss of the row, or the reason the facts do not give one."""
    paths = row.get("realized")
    if paths is None:
        return None, _no_realized(row)
    if paths.draws != 1:
        raise ValueError(f"the realized paths must hold one path, not {paths.draws}")
    losses, why = _valued(portfolio, row, paths, where="the realized path")
    if why is not None:
        return None, why
    loss = float(losses[0])
    if not np.isfinite(loss):
        return None, _no_realized(row, "the realized loss is not finite")
    return loss, None


def _no_realized(row: dict, reason: str = "no realized path") -> str:
    """The reason of a missing realized loss, with the facts it lacked when the row names them."""
    missing = tuple(row.get("realized_missing") or ())
    if not missing:
        return reason
    shown = ", ".join(str(name) for name in missing[:8])
    return f"{reason} ({shown}{', ...' if len(missing) > 8 else ''})"


def _losses(batch: Batch) -> np.ndarray:
    """Minus the sum of the calculated positions' value changes; empty when none was calculated."""
    included = batch.calculated
    if not included:
        return np.empty(0)
    return -np.sum([np.asarray(result.value_change, dtype=float) for result in included], axis=0)


def _horizon(rows: Sequence[dict]) -> dict:
    """The horizon mark: the episode window and the month its realized loss is read at."""
    horizons = sorted({int(row["horizon"]) for row in rows if row.get("horizon") is not None})
    month = horizons[-1] if horizons else None
    note = (
        "the realized loss is the portfolio valued on the facts at t = 0 and t = H of the "
        "episode window; the predictive VaR and ES are P95 and ES95 of the portfolio losses on "
        "the conditional paths"
    )
    if len(horizons) > 1:
        note += f"; the episodes span horizons {horizons[0]}..{horizons[-1]}"
    return {"kind": "episode window", "month": month, "note": note}


def _window_start(as_of: date, panel: Panel) -> date:
    """The calculation month of the window: the last month of the panel, which ``as_of`` names."""
    if not panel.months:
        raise ValueError("the panel has no month: the calculation month of the window is unknown")
    start = panel.months[-1]
    if start != as_of.replace(day=1):
        raise ValueError(
            f"the panel ends at {start:%Y-%m}, the calculation date {as_of:%Y-%m-%d} is another "
            "month: t = 0 of the window must be the calculation month"
        )
    return start


def _record_missing(missing: list[str], name: str, months: Sequence[date], values) -> None:
    """Append every ``<name>/<month>`` of the window whose value is not finite."""
    for month, value in zip(months, values, strict=True):
        if not np.isfinite(value):
            missing.append(f"{name}/{month:%Y-%m}")


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
