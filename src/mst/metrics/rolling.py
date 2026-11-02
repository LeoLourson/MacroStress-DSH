"""The rolling check over the dates of history: the forecast, the uniformity and the price.

The episode backtest judges the conditional forecast on a handful of named events, chosen with
hindsight. What it cannot see is measured here, on a grid of dates the plan names
(``[rolling] step_months``):

**The quality of the forecast.**  On every date the unconditional forecast of the model
estimated on the data known then is scored against the facts and against the random walk from
the same history: the relative CRPS of ``mst.metrics.quality``, per half of the grid, with the
calibration per horizon. The dates are not selected, so quiet months count and the calibration
shares have the meaning a selected sample denies them; the windows of neighbouring dates overlap,
so the intervals come from moving blocks as long as a window. ``--against`` compares two
versions on the same cells by the four steps of the decision rule.

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

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

from mst.model_sdk.bounds import lower_bounds, observed

from .baselines import DOMAIN, WIDTH, band_problems, macro_domain, random_walk
from .cells import crps, finite_draws, history_range, pit
from .decision import decide
from .episodes import Episodes, load_episodes
from .quality import interval, paired, relative

BUCKETS = (0.05, 0.5, 0.95)
"""The canon's calibration buckets: below a, between a and the median, median to b, above b."""
BLOCK = 12
"""The block length of the bootstrap of the price, in grid dates: the windows overlap, so a block
of consecutive dates carries the dependence."""
REPETITIONS = 1000
HISTORY_MONTHS = 24
"""The grid starts this many months into the history, whatever the model asks for itself."""
HORIZONS = (1, 3, 6, 12)
"""The horizon months the calibration is reported at, beside the pool of them all."""


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
    panels: dict = field(default_factory=dict)
    """The panel of every date, the history the walk of that date reads."""
    skipped: dict = field(default_factory=dict)
    """The dates whose data are too short for any model, with the reason."""
    refused: dict = field(default_factory=dict)
    """The dates with enough data where the model gave no version, with the reason."""


def rolling_report(
    *,
    plugin,
    store,
    registry,
    root,
    plan: Episodes,
    known_on: date,
    macro_of,
    only: list[date] | None = None,
) -> dict:
    """The rolling block of the report: the forecast, the uniformity at ``s_0`` and the price.

    ``macro_of`` is the cached episode estimator (``backtest._episode_macro``): the rolling
    check never estimates a model itself. The report is empty when the plan names no rolling
    grid. ``only`` narrows the grid to these dates.
    """
    if plan.rolling is None:
        return {}
    grid = _grid(plugin, store, registry, root, plan, known_on, macro_of, only=only)
    report: dict = {
        "step_months": plan.rolling.step_months,
        "validation_from": plan.rolling.validation_from,
        "dates": len(grid.dates),
        "skipped": {when.isoformat(): reason for when, reason in grid.skipped.items()},
        "refused": {when.isoformat(): reason for when, reason in grid.refused.items()},
    }
    report["base_uniformity"] = _uniformity(grid, plan)
    report["quality"] = _quality(grid, plan, registry)
    if plan.rolling.price_events:
        report["price"] = _price(grid, plan, known_on)
    return report


def grid_dates(store, registry, plan: Episodes, known_on: date) -> list[date]:
    """The dates of the grid: the months the store knows, from two years into its history."""
    months = sorted(
        {row.month for variable in registry.variables for row in store.as_of(variable.id, known_on)}
    )
    return months[HISTORY_MONTHS :: plan.rolling.step_months]


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


def half_of(when: date, plan: Episodes) -> str:
    """The half of the grid a date belongs to; every date is tuning without a split."""
    start = plan.rolling.validation_from
    return "validation" if start is not None and when >= start else "tuning"


def _grid(plugin, store, registry, root, plan, known_on, macro_of, *, only=None) -> Grid:
    """The dates of the grid with their model versions, ``s_0`` simulations and facts.

    A date with too little history for any model is named in ``skipped``. A date with enough
    data where the model gives no version is named in ``refused``: scoring only the dates a
    model survives would hide its failures. The versions are kept only for the price: on a long grid they are heavy.
    """
    from mst.macro.panel import shift
    from mst.model_sdk.contracts import ScenarioPlan

    from .backtest import _realized, calculation_date  # the episode machinery, reused

    dates = grid_dates(store, registry, plan, known_on)
    if only is not None:
        dates = [when for when in dates if when in set(only)]
    reach = max([plan.after_months, *(event.horizon for event in plan.rolling.price_events)])
    versions, forecasts, facts, panels, skipped, refused = {}, {}, {}, {}, {}, {}
    kept = []
    for when in dates:
        as_of = calculation_date(when)
        try:
            version = macro_of(plugin, store, registry, as_of, root)
        except ValueError as error:
            short = _short_of_history(store, registry, as_of)
            (skipped if short else refused)[when] = str(error)
            continue
        kept.append(when)
        if plan.rolling.price_events:
            versions[when] = version
        panels[when] = version.panel
        forecasts[when] = observed(
            version.plugin.simulate(
                version.model,
                ScenarioPlan(horizon=plan.after_months),
                draws=plan.draws,
                seed=plan.seed,
            ),
            registry,
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
    return Grid(
        dates=kept,
        versions=versions,
        forecasts=forecasts,
        facts=facts,
        panels=panels,
        skipped=skipped,
        refused=refused,
    )


def _quality(grid: Grid, plan: Episodes, registry) -> dict:
    """The forecast against the facts and the walk, over the whole grid and per half."""
    rows = _quality_rows(grid, plan, registry)
    block = max(int(np.ceil(plan.after_months / plan.rolling.step_months)), 1)
    found = {"block_dates": block, "rows": rows, "halves": {}}
    for name in ("all", "tuning", "validation"):
        chosen = rows if name == "all" else [row for row in rows if row["half"] == name]
        if chosen:
            found["halves"][name] = _summary(chosen, block, plan.seed)
    return found


def _quality_rows(grid: Grid, plan: Episodes, registry) -> list[dict]:
    """One row per date, horizon month and variable with a fact: the two CRPS and the rules."""
    rows = []
    for when in grid.dates:
        simulation = grid.forecasts[when]
        panel = grid.panels[when]
        for k, id_ in enumerate(simulation.ids):
            history = panel.y[:, panel.ids.index(id_)]
            floors = lower_bounds(registry, [id_])
            horizon = simulation.paths.shape[1] - 1
            walk = random_walk(
                history,
                horizon,
                draws=plan.draws,
                seed=plan.seed,
                lower=None if floors is None else float(floors[0]),
            )
            domain = macro_domain(registry.variable(id_))
            spread = history_range(history)
            for t in range(1, horizon + 1):
                value = _fact_of(grid, when, simulation.months[t], id_)
                draws = finite_draws(simulation.paths[:, t, k])
                if value is None or not np.isfinite(value) or not len(draws):
                    continue
                known = finite_draws(walk[:, t - 1])
                band = np.percentile(draws, [5, 95])
                broken = band_problems(draws, value, domain=domain, history_range=spread)
                rows.append(
                    {
                        "date": when.isoformat(),
                        "half": half_of(when, plan),
                        "month": t,
                        "quantity": id_,
                        "model": crps(draws, value),
                        "walk": crps(known, value) if len(known) else None,
                        "pit": pit(draws, value),
                        "inside": bool(band[0] <= value <= band[1]),
                        DOMAIN: DOMAIN in broken,
                        WIDTH: WIDTH in broken,
                    }
                )
    return rows


def _summary(rows: list[dict], block: int, seed: int) -> dict:
    """The relative CRPS with its interval, the calibration and the admission of some rows."""
    score = relative(rows, "model", "walk")
    return {
        "dates": len({row["date"] for row in rows}),
        "cells": len(rows),
        "vs_walk": score["score"],
        "interval": interval(rows, "model", "walk", unit="date", block=block, seed=seed),
        "by_horizon": score["by_horizon"],
        "by_quantity": score["by_quantity"],
        "calibration": _calibration(rows),
        "calibration_by_horizon": {
            t: _calibration([row for row in rows if row["month"] == t])
            for t in HORIZONS
            if any(row["month"] == t for row in rows)
        },
        "coverage_90": float(np.mean([row["inside"] for row in rows])),
        "admission": {
            rule: float(np.mean([row[rule] for row in rows])) for rule in (DOMAIN, WIDTH)
        },
    }


def _calibration(rows: list[dict]) -> list[float]:
    """The four bucket shares of the transform of the facts."""
    values = np.array([row["pit"] for row in rows])
    edges = np.array([0.0, *BUCKETS, np.inf])
    return [
        float(np.mean((values >= edges[k]) & (values < edges[k + 1])))
        for k in range(len(edges) - 1)
    ]


def rolling_decision(report: dict, against: Path, *, seed: int) -> dict:
    """The four steps of the decision rule on the logarithm of the relative CRPS.

    The reports must share their cells, and the comparison runs on the validation half when
    there is one: the benchmark, at zero, is the walk; the incumbent is the other report. A
    candidate that refused a date both reports should cover has failed whatever its score.
    """
    other = json.loads(Path(against).read_text(encoding="utf-8"))
    other = other.get("rolling", other)
    if other.get("step_months") != report["step_months"]:
        raise ValueError(f"--against {against}: the grid has another step")
    rows, others = report["quality"]["rows"], other["quality"]["rows"]
    key = lambda row: (row["date"], row["month"], row["quantity"])  # noqa: E731
    difference = {key(row) for row in rows} ^ {key(row) for row in others}
    if difference:
        sample = ", ".join(f"{d}/{q}/t{t}" for d, t, q in sorted(difference)[:10])
        raise ValueError(f"the two reports do not share {len(difference)} cells: {sample}")
    half = "validation" if any(row["half"] == "validation" for row in rows) else "tuning"
    mine = [row for row in rows if row["half"] == half]
    theirs = [row for row in others if row["half"] == half]
    block = report["quality"]["block_dates"]
    versus_walk = interval(mine, "model", "walk", unit="date", block=block, seed=seed)
    versus_other = paired(mine, theirs, "model", unit="date", block=block, seed=seed)
    incumbent = relative(theirs, "model", "walk")["score"]
    verdict = decide(
        candidate_crps=float(np.log(versus_walk["point"])),
        benchmark_crps=0.0,
        incumbent_crps=None if incumbent is None else float(np.log(incumbent)),
        se_vs_benchmark=versus_walk["log_se"] or 0.0,
        se_vs_incumbent=versus_other["log_se"] or 0.0,
        candidate_admission_shares=report["quality"]["halves"][half]["admission"],
        incumbent_admission_shares=other["quality"]["halves"][half]["admission"],
    )
    verdict.update(half=half, vs_walk=versus_walk, vs_incumbent=versus_other)
    if report["refused"]:
        verdict["verdict"] = "rejected"
        verdict["reason"] = f"the model refused {len(report['refused'])} date(s) of the grid"
    return verdict


def run(
    *,
    store: Path,
    models: Path,
    as_of: date,
    macro_id: str | None = None,
    dates: list[date] | None = None,
    half: str | None = None,
    details: Path | None = None,
    against: Path | None = None,
) -> int:
    """The rolling check from the command line; returns 1 when the model refused a date or no date
    could be judged."""
    from mst import models as models_pkg
    from mst.data.store import SeriesStore
    from mst.macro.registry import load_registry

    from .backtest import _episode_macro, _jsonable

    plan = load_episodes()
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
        macro_of=_episode_macro,
        only=only,
    )
    if against is not None and report["quality"]["rows"]:
        report["decision"] = rolling_decision(report, against, seed=plan.seed)
    print_rolling(report)
    if details is not None:
        details.write_text(
            json.dumps(report, ensure_ascii=False, indent=1, default=_jsonable),
            encoding="utf-8",
        )
    return 0 if report["quality"]["rows"] and not report["refused"] else 1


def print_rolling(report: dict) -> None:
    """The forecast quality over the grid as Markdown tables."""
    from .backtest import _interval_text, _number

    quality = report["quality"]
    print("\n## Forecast quality over the grid\n")
    print("Relative CRPS: below one the model beats the walk.\n")
    print(
        f"{report['dates']} dates, step {report['step_months']} month(s); "
        f"block of {quality['block_dates']} dates in the intervals."
    )
    print(
        "\n| half | dates | cells | model / walk | 95 % interval | coverage of p5..p95 "
        "| shares p5/p50/p95 (0.05 0.45 0.45 0.05) | domain | width |"
    )
    print("|---|---:|---:|---:|---|---:|---|---:|---:|")
    for name, block in quality["halves"].items():
        found = block["interval"]
        bounds = None if found["low"] is None else (found["low"], found["high"])
        shares = " ".join(f"{share:.2f}" for share in block["calibration"])
        print(
            f"| {name} | {block['dates']} | {block['cells']} | {_number(block['vs_walk'])} "
            f"| {_interval_text(bounds)} | {block['coverage_90']:.3f} | {shares} "
            f"| {block['admission'][DOMAIN]:.3f} | {block['admission'][WIDTH]:.3f} |"
        )
    top = quality["halves"].get("all")
    if top:
        print("\n| horizon | model / walk | shares p5/p50/p95 |")
        print("|---:|---:|---|")
        for t in HORIZONS:
            if t in top["calibration_by_horizon"]:
                shares = " ".join(f"{share:.2f}" for share in top["calibration_by_horizon"][t])
                print(f"| {t} | {_number(top['by_horizon'].get(t))} | {shares} |")
        print("\n| quantity | model / walk |")
        print("|---|---:|")
        for quantity, value in sorted(top["by_quantity"].items(), key=lambda item: item[1]):
            print(f"| {quantity} | {_number(value)} |")
    if report["skipped"]:
        print(f"\nDates left out, too little history: {len(report['skipped'])}")
        for when, reason in report["skipped"].items():
            print(f"- {when}: {reason}")
    if report["refused"]:
        print(
            f"\n**The model refused {len(report['refused'])} of "
            f"{len(report['refused']) + report['dates']} dates with enough data.** "
            "The scores above cover only the dates it survived."
        )
        for when, reason in report["refused"].items():
            print(f"- {when}: {reason}")
    verdict = report.get("decision")
    if verdict:
        print(
            f"\nDecision on the {verdict['half']} half: **{verdict['verdict']}** "
            f"(step {verdict['step']}) — {verdict['note']}; against the walk "
            f"{_number(verdict['vs_walk']['point'])}, against the incumbent "
            f"{_number(verdict['vs_incumbent']['point'])} "
            f"[{_number(verdict['vs_incumbent']['low'])}, "
            f"{_number(verdict['vs_incumbent']['high'])}]."
        )


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
            LinearTerm(variable=term["variable"], month=term["month"], weight=term["weight"])
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
