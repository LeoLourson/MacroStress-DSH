"""The rolling grid: the forecast of every level on dates nobody selected, and the price.

The episodes judge the conditional forecast on a handful of named events, chosen with
hindsight. What they cannot see is measured here, on a grid of dates the plan names
(``[rolling] step_months``). Every date opens a window (``mst.metrics.windows``) with the empty
scenario: the models of every level are estimated on the data known that day and forecast the
months that follow, and the cells of every level (``mst.metrics.levels``) are judged as the
episodes are judged (``mst.metrics.scores.judge``). The dates are not selected, so quiet months
count and the calibration has the meaning a selected sample denies it: under the empty
scenario the forecast has the model's own law, and the probability integral transform of the
facts is uniform whatever the date. The windows of neighbouring dates overlap, so the intervals
come from moving blocks as long as a window.

**The price.**  A family of events fixed before the check (``[[price_events]]``) is priced on
every date: the model estimated on the data known then gives ``pi = P(S)``, the facts tell
whether the event happened in the window that follows, and the mean logarithmic score of
``pi`` stands beside the score of the historical frequency — the share of the grid's earlier
windows where the event happened. The report also gives the table of the mean ``pi`` against
the realized frequency in groups by ``pi``, the block-bootstrap interval of the scores, and the
difference ``dist + log P(S)`` — the part of the price the parameter uncertainty carries.

The versions of the grid are the cached versions of ``mst.metrics.windows``: an estimation per
date and model, whatever the events are.
"""

import json
from datetime import date
from pathlib import Path

import numpy as np

from . import levels, scores
from .decision import verdict
from .episodes import Episodes, calculation_date, load_episodes
from .tables import print_decision, print_dropped, print_judgement
from .windows import MACRO_ONLY, Scope, Window, extend, facts, open_window, simulate

UNIT = "date"
"""The key of a grid date in the rows."""
HISTORY_MONTHS = 24
"""The grid starts this many months into the history, whatever the model asks for itself."""


def rolling_report(
    *,
    plugin,
    store,
    registry,
    root: Path,
    plan: Episodes,
    known_on: date,
    inputs: Path | None = None,
    scope: Scope = MACRO_ONLY,
    only: list[date] | None = None,
) -> dict:
    """The rolling block of a report: what fell out, the judgement of every level, the price.

    ``scope`` names the levels judged beyond the macro variables: with a driver catalog every
    date also estimates the satellite bundle, with a portfolio the positions are valued too.
    ``only`` narrows the grid to these dates. The report is empty when the plan names no rolling
    grid.
    """
    if plan.rolling is None:
        return {}
    dates = grid_dates(store, registry, plan, known_on)
    if only is not None:
        dates = [when for when in dates if when in set(only)]
    windows, skipped, refused = [], {}, {}
    for when in dates:
        try:
            window = open_window(
                plugin,
                store,
                registry,
                plan,
                id_=when.isoformat(),
                start=when,
                half=half_of(when, plan),
                root=root,
                known_on=known_on,
                inputs=inputs,
            )
            extend(window, store, root, scope)
        except ValueError as error:
            short = _short_of_history(store, registry, calculation_date(when))
            (skipped if short else refused)[when.isoformat()] = str(error)
            continue
        simulate(window, registry, plan)
        windows.append(window)
    found = [cell for window in windows for cell in levels.cells(window, registry, plan)]
    rows = levels.rows(found, UNIT, {window.id: window.half for window in windows})
    block = block_dates(plan)
    report = {
        "step_months": plan.rolling.step_months,
        "validation_from": plan.rolling.validation_from,
        "block_dates": block,
        "dates": len(windows),
        "dropped": levels.fell_out(windows, found, refused=refused, skipped=skipped),
        **scores.judge(
            found, rows, unit=UNIT, block=block, seed=plan.seed, repetitions=scores.REPETITIONS
        ),
        "rows": rows,
    }
    if plan.rolling.price_events:
        report["price"] = _price(windows, registry, plan)
    return report


def grid_dates(store, registry, plan: Episodes, known_on: date) -> list[date]:
    """The dates of the grid: the months the store knows, from two years into its history."""
    months = sorted(
        {row.month for variable in registry.variables for row in store.as_of(variable.id, known_on)}
    )
    return months[HISTORY_MONTHS :: plan.rolling.step_months]


def block_dates(plan: Episodes) -> int:
    """The block of the bootstrap, in grid dates: as many as one window spans."""
    return max(int(np.ceil(plan.after_months / plan.rolling.step_months)), 1)


def half_of(when: date, plan: Episodes) -> str:
    """The half of the grid a date belongs to; every date is tuning without a split."""
    start = plan.rolling.validation_from
    return "validation" if start is not None and when >= start else "tuning"


def _short_of_history(store, registry, as_of: date) -> bool:
    """Whether the data known on a date cannot feed any model: a panel that cannot be built, or
    a variable with fewer than three values. The data decide, not the wording of an error; a
    variable the store never holds is no shortage of history."""
    from mst.macro.panel import build_panel

    held = [v for v in registry.variables if store.as_of(v.id, date.max)]
    months = [row.month for v in held for row in store.as_of(v.id, as_of)]
    if not months:
        return True
    seasonal = [v.id for v in held if v.seasonal]
    try:
        panel = build_panel(
            store,
            registry.model_copy(update={"variables": held}),
            start=min(months),
            as_of=as_of,
            seasonal=seasonal,
        )
    except ValueError:
        return True
    return bool((np.isfinite(panel.y).sum(axis=0) < 3).any())


def passed(report: dict) -> bool:
    """Whether the grid judged some cells, the model refused no date and admission passed."""
    return bool(
        report["rows"]
        and not report["dropped"]["refused"]
        and not report["sanity"]["admission"]["failed"]
    )


def rolling_decision(report: dict, against: Path, *, level: str = "macro", seed: int) -> dict:
    """The four steps of the decision rule against the incumbent's rolling details.

    The two grids must have the same step. A candidate that refused a date has failed whatever
    its score: scoring only the dates a model survives would hide its failures.
    """
    other = json.loads(Path(against).read_text(encoding="utf-8"))
    other = other.get("rolling", other)
    if other.get("step_months") != report["step_months"]:
        raise ValueError(f"--against {against}: the grid has another step")
    found = verdict(
        report["rows"],
        report["sanity"]["admission"],
        other,
        unit=UNIT,
        level=level,
        block=report["block_dates"],
        seed=seed,
    )
    if report["dropped"]["refused"]:
        found["verdict"] = "rejected"
        found["reason"] = f"the model refused {len(report['dropped']['refused'])} date(s)"
    return found


def run(
    *,
    store: Path,
    models: Path,
    episodes: Path,
    as_of: date,
    macro_id: str | None = None,
    dates: list[date] | None = None,
    half: str | None = None,
    details: Path | None = None,
    against: Path | None = None,
    inputs: Path | None = None,
    drivers: Path | None = None,
    portfolio: str | None = None,
    target_level: str = "macro",
) -> int:
    """The rolling check from the command line; 1 when the model refused a date, no date could
    be judged or a quantity failed admission."""
    from mst import models as models_pkg
    from mst.data.store import SeriesStore
    from mst.macro.registry import load_registry

    from .backtest import jsonable

    plan = load_episodes(episodes)
    if plan.rolling is None:
        raise ValueError("the episode plan names no [rolling] grid")
    data = SeriesStore(store)
    registry = load_registry()
    only = dates
    if half is not None:
        only = [
            when
            for when in (dates or grid_dates(data, registry, plan, as_of))
            if half_of(when, plan) == half
        ]
    report = rolling_report(
        plugin=models_pkg.selected(macro_id),
        store=data,
        registry=registry,
        root=Path(models) / "backtest",
        plan=plan,
        known_on=as_of,
        inputs=inputs,
        scope=Scope.of(store, drivers, portfolio),
        only=only,
    )
    if against is not None and report["rows"]:
        report["decision"] = rolling_decision(report, against, level=target_level, seed=plan.seed)
    print_rolling(report)
    if details is not None:
        details.write_text(
            json.dumps(report, ensure_ascii=False, indent=1, default=jsonable), encoding="utf-8"
        )
    return 0 if passed(report) else 1


def print_rolling(report: dict) -> None:
    """The grid as Markdown: what fell out first, the judgement, the price and the decision."""
    print("\n## Rolling grid: what fell out\n")
    print(
        f"Step {report['step_months']} month(s); validation from "
        f"{report['validation_from'] or 'nowhere'}; block of {report['block_dates']} dates in "
        "the intervals."
    )
    print_dropped(report["dropped"], judged=report["dates"], unit="dates")
    print_judgement(report, title="Rolling grid", reference_calibration=False)
    price = report.get("price")
    if price and price.get("events"):
        found = price["scores"]
        print("\n## Rolling grid: price of the events\n")
        print(
            f"Mean logarithmic score over {found['windows']} windows: model "
            f"{found['model']:.3f}, historical frequency {found['frequency']:.3f}."
        )
    if report.get("decision"):
        print_decision(report["decision"])


def _price(windows: list[Window], registry, plan: Episodes) -> dict:
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
    from mst.scenario.build import _price_of_logs

    rows = []
    for event in plan.rolling.price_events:
        terms = tuple(
            LinearTerm(variable=term["variable"], month=term["month"], weight=term["weight"])
            for term in event.terms
        )
        constraint = LinearConstraint(terms=terms, low=event.low, high=event.high)
        for window in windows:
            version = window.macro
            if not set(term.variable for term in terms) <= set(version.ids):
                continue
            the_plan = ScenarioPlan(horizon=event.horizon, constraints=(constraint,))
            law = version.plugin.constraint_law(
                version.model,
                the_plan,
                draws=max(plan.draws // 4, 100),
                seed=plan.seed,
            )
            if law is None:
                # A model that measures its own box probabilities still has a price.
                logs = version.plugin.log_box_probability(
                    version.model,
                    the_plan,
                    np.array([constraint.low]),
                    np.array([constraint.high]),
                    draws=max(plan.draws // 4, 100),
                    seed=plan.seed,
                )
                if logs is None:
                    continue
                # A value of R without probability gives P(S) = 0, as price_of_law does:
                # the miss stays in the score.
                distance, log_probability, _ = _price_of_logs(logs)
            else:
                distance, log_probability, _ = price_of_law(
                    law,
                    np.array([constraint.low]),
                    np.array([constraint.high]),
                    rng=np.random.default_rng([plan.seed, DIST_STREAM]),
                )
            happened = _happened(window, registry, plan, terms, constraint)
            if happened is None:
                continue
            rows.append(
                {
                    "event": event.id,
                    "date": window.id,
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


def _happened(window: Window, registry, plan: Episodes, terms, constraint):
    """Whether the event's box held in the months after the window's panel; None without facts.

    The event's months count from the last month of the panel, as the model's forecast does,
    and may run past the window's horizon.
    """
    from mst.macro.panel import shift

    last = window.panel.months[-1]
    reach = max(term.month for term in terms)
    months = [shift(last, k) for k in range(1, reach + 1)]
    realized = facts(
        window.store, registry, window.panel, months, window.as_of, window.known_on, plan
    )
    total = 0.0
    for term in terms:
        value = realized[term.month - 1, window.panel.ids.index(term.variable)]
        if not np.isfinite(value):
            return None
        total += term.weight * value
    return bool(constraint.low <= total <= constraint.high)


def _scores(rows: list[dict]) -> dict:
    """The mean logarithmic scores of the model and the frequency, with block intervals.

    The frequency of an event on a date is the share of the earlier windows of the same
    event where it happened — a running rate, one number per date — and the frequency score
    clips it away from zero and one, where the logarithm would not exist.
    """
    from .block import block_interval

    model, frequency, dates = [], [], []
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
        inside = [row for row in rows if edges[k] <= row["pi"] < edges[k + 1]]
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
