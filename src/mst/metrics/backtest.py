"""The episodes: what a model does when the event is known, judged as the grid is judged.

For an episode of the plan the calculation date is the day before it starts. The episode opens
a window there (``mst.metrics.windows``): the models of every level are estimated on the data
known that day, and the scenario is what actually happened — the realized paths of the
episode's observed variables as point constraints month by month, and each decision of the
episode as an intervention that sets its realized level. The cells the scenario fixes are
reported and never scored. Every other cell of every level (``mst.metrics.levels``) is judged
by the same judgement as the rolling grid (``mst.metrics.scores.judge``): what fell out first,
then the sanity of every quantity, then the relative CRPS of the forecast against the walk that
knows the scenario's path, the plain walk, its own base run under the empty scenario and, for
drivers, the naive forecast of a year earlier.

The plain walk does not know the future path the model is given, so a win over it says
little; the conditional walk is the like-for-like benchmark, and the forecast against the base
run is what the scenario adds — reported per episode and split by who formalised the scenario.
The quality of the forecast itself is judged on the rolling grid (``mst.metrics.rolling``), on
dates nobody selected, which ``run`` adds to the report unless told not to. Beside the scores
an episode reports the probability its conditions have under the base run: a scenario resting
on a rare tail is marked. ``--portfolio`` adds the Acerbi-Székely tail section.
"""

import contextlib
import io
import json
from datetime import date
from pathlib import Path

import numpy as np

from mst.data.store import SeriesStore
from mst.macro.registry import REGISTRY_PATH as MACRO_REGISTRY_PATH
from mst.macro.registry import Registry, load_registry
from mst.model_sdk.contracts import Intervention, LinearTerm, ScenarioPlan
from mst.scenario.build import point_constraint

from . import levels, scores
from .baselines import CONSTRAINT, CONSTRAINT_THRESHOLD, constraint_probabilities
from .decision import verdict
from .episodes import Episodes, load_episodes
from .quality import relative
from .tables import number, print_decision, print_dropped, print_judgement
from .windows import Window, attach_satellites, file_digest, open_window, provenance, simulate

UNIT = "episode"
"""The key of an episode in the rows."""
MISSING = ()
"""What the report cannot compute; empty when every metric is computed."""


def run(
    *,
    store: Path,
    models: Path,
    episodes: Path,
    as_of: date,
    macro_id: str | None = None,
    drivers: Path | None = None,
    satellites: bool = False,
    only: list[str] = (),
    details: Path | None = None,
    against: Path | None = None,
    portfolio: str | None = None,
    mc_bootstrap: bool = False,
    target_level: str = "macro",
    rolling: bool = True,
    inputs: Path | None = None,
    save: bool = False,
) -> int:
    """The check of every episode the plan names; prints Markdown tables, returns an exit code.

    ``episodes`` is the plan file, ``bench/model-backtest/episodes.toml`` in the repository.
    ``as_of`` is the date the facts are read on; each episode's models are estimated on the day
    before it starts, whatever ``as_of`` is. With ``satellites`` the drivers of the catalog
    ``drivers`` are judged too, on the bundle estimated on each window's macro version, on the
    episodes and on the grid. ``against`` names the ``--details`` file of the incumbent version,
    checked on the same plan. ``portfolio`` names a stored portfolio for the tail section;
    ``mc_bootstrap`` adds the Monte-Carlo interval of the forecast against the walk;
    ``target_level`` is the level the decision and the usefulness read. ``rolling`` off leaves
    out the grid, which ``python -m mst.metrics rolling`` also runs. ``inputs`` is the folder of
    the answers a model with declared inputs reads. ``save`` keeps the report without its cells
    and its tables in ``results/`` beside the plan; a part of the plan named by ``only`` is not
    saved.
    """
    from mst import bench
    from mst import models as models_pkg

    if save and only:
        raise ValueError("--save keeps the result of the whole plan: leave out --only")
    if satellites and drivers is None:
        raise ValueError("--satellites needs --drivers, the driver catalog file")
    if portfolio is not None and not satellites:
        raise ValueError(
            "--portfolio needs --satellites: the tail values the positions on the driver paths"
        )
    plan = load_episodes(episodes)
    plugin = models_pkg.selected(macro_id)
    registry = load_registry()
    data = SeriesStore(store)
    root = Path(models) / "backtest"
    catalog = drivers if satellites else None
    refused: dict[str, str] = {}
    windows: list[Window] = []
    for check in plan.episodes:
        if only and check.id not in only:
            continue
        try:
            window = _episode(plugin, data, registry, plan, check, root, as_of, inputs, catalog)
        except ValueError as error:
            refused[check.id] = str(error)
            continue
        simulate(window, registry, plan)
        windows.append(window)
    report = _report(
        windows,
        plan,
        registry,
        refused=refused,
        target_level=target_level,
        mc_bootstrap=mc_bootstrap,
    )
    report["meta"] = _meta(
        windows, plan, plugin=plugin, known_on=as_of, store=store, models=models, root=root
    )
    if against is not None:
        report["decision"] = _decision_block(report, against, plan, target_level)
    if portfolio is not None:
        report["tail"] = _tail_section(windows, plan, portfolio, store=store, known_on=as_of)
    if plan.rolling is not None and rolling:
        from .rolling import rolling_report

        report["rolling"] = rolling_report(
            plugin=plugin,
            store=data,
            registry=registry,
            root=root,
            plan=plan,
            known_on=as_of,
            inputs=inputs,
            drivers=catalog,
        )
    tables = io.StringIO()
    with contextlib.redirect_stdout(tables):
        print_report(report)
    print(tables.getvalue(), end="")
    if details is not None:
        details.write_text(
            json.dumps(report, ensure_ascii=False, indent=1, default=jsonable), encoding="utf-8"
        )
    if save:
        plain = json.loads(json.dumps(compact(report), default=jsonable))
        bench.save_result(
            bench.results(plan.path.parent), plugin.model_id, plain, tables.getvalue()
        )
    return 1 if refused or report["admission_failed"] else 0


def _episode(
    plugin,
    data: SeriesStore,
    registry: Registry,
    plan: Episodes,
    check,
    root: Path,
    known_on: date,
    inputs: Path | None,
    catalog: Path | None,
) -> Window:
    """The window of an episode with its scenario built from the facts.

    A variable the scenario fixes must be one the model forecasts: a variable the model left out
    as too short, or one its panel does not know, voids the episode with the reason.
    """
    window = open_window(
        plugin,
        data,
        registry,
        plan,
        id_=check.id,
        start=check.start,
        half=check.half,
        root=root,
        known_on=known_on,
        inputs=inputs,
    )
    lost = [name for name in check.constrained if name in window.left_out]
    if lost:
        raise ValueError(
            "; ".join(
                f"the model left out {name}, which the episode fixes: {window.left_out[name]}"
                for name in lost
            )
        )
    ids = window.panel.ids
    unknown = [name for name in check.constrained if name not in ids]
    if unknown:
        raise ValueError(f"the macro panel knows no variables {', '.join(unknown)}")
    constraints = []
    for id_ in check.observations:
        for k, value in enumerate(window.realized[:, ids.index(id_)], start=1):
            if not np.isnan(value):
                constraints.append(
                    point_constraint(
                        (LinearTerm(variable=id_, month=k, weight=1.0),), float(value), window.macro
                    )
                )
    window.scenario = ScenarioPlan(
        horizon=window.horizon,
        constraints=tuple(constraints),
        interventions=_interventions(check.interventions, window.realized, ids, plan.draws),
    )
    window.observations = list(check.observations)
    window.interventions = list(check.interventions)
    window.source = check.source
    if catalog is not None:
        try:
            attach_satellites(window, data, root, catalog)
        except ValueError as error:
            raise ValueError(f"satellites: {error}") from error
    return window


def _interventions(
    names: list[str], realized: np.ndarray, ids: list[str], draws: int
) -> tuple[Intervention, ...]:
    """The decisions of the episode as sets of the realized level, one row per draw.

    An episode asserts that the decision was taken, so the realized value is the value the
    model must take in those months; every group of paths carries the same row, because the
    episode fixes the level rather than a law of it. A month without a fact is left out, as
    the observations leave it out.
    """
    found = []
    for name in names:
        column = ids.index(name)
        months, values = [], []
        for month, value in enumerate(realized[:, column], start=1):
            if not np.isnan(value):
                months.append(month)
                values.append(float(value))
        if not months:
            continue
        found.append(
            Intervention(
                target=name,
                months=tuple(months),
                kind="set",
                values=np.tile(np.asarray(values, dtype=float), (draws, 1)),
            )
        )
    return tuple(found)


def _report(
    windows: list[Window],
    plan: Episodes,
    registry: Registry,
    *,
    refused: dict[str, str],
    target_level: str,
    mc_bootstrap: bool,
) -> dict:
    """What fell out, the judgement of every level, the gate of the conditions and the
    usefulness of the scenario."""
    found = [cell for window in windows for cell in levels.cells(window, registry, plan)]
    rows = levels.rows(found, UNIT, {window.id: window.half for window in windows})
    if rows and target_level not in {row["level"] for row in rows}:
        known = ", ".join(sorted({row["level"] for row in rows}))
        raise ValueError(f"target_level {target_level!r} has no cells; levels present: {known}")
    shares = {
        window.id: constraint_probabilities(
            window.base_shadow.paths, window.base_shadow.ids, window.scenario.constraints
        )
        if window.base_shadow is not None
        else {}
        for window in windows
    }
    judgement = scores.judge(
        found, rows, unit=UNIT, block=1, seed=plan.seed, repetitions=scores.REPETITIONS
    )
    gate = _constraint_gate(shares)
    report = {
        "dropped": {
            "refused": dict(refused),
            "left_out": {window.id: window.left_out for window in windows if window.left_out},
            "non_finite": levels.non_finite(found),
        },
        **judgement,
        "constraint_gate": gate,
        "admission_failed": judgement["sanity"]["admission"]["failed"] or gate["failed"],
        "target_level": target_level,
        "episodes": [
            {
                "id": window.id,
                "as_of": window.as_of.isoformat(),
                "horizon": window.horizon,
                "half": window.half,
                "source": window.source,
                "observations": window.observations,
                "interventions": window.interventions,
                "macro_version": window.macro.name,
                "satellite_version": window.bundle_name,
                "left_out": window.left_out,
                "constraint_probabilities": shares[window.id],
            }
            for window in windows
        ],
        "utility": _utility(rows, {window.id: window.source for window in windows}, target_level),
        "rows": rows,
    }
    if mc_bootstrap:
        report["monte_carlo"] = scores.monte_carlo(
            found,
            rows,
            unit=UNIT,
            level=target_level,
            seed=plan.seed,
            repetitions=scores.REPETITIONS,
        )
    return report


def _utility(rows: list[dict], sources: dict[str, str], target_level: str) -> dict:
    """What the scenario adds: the forecast against the base run, per episode and per source.

    An episode counts as helped when its ratio is below one. The sources are reported apart:
    a language model may have known the outcome of a historical event from its training.
    """
    chosen = [row for row in rows if row["level"] == target_level and row["base"] is not None]
    by_episode: dict[str, list[dict]] = {}
    for row in chosen:
        by_episode.setdefault(row[UNIT], []).append(row)
    ratios = {
        episode: relative(own, "model", "base")["score"] for episode, own in by_episode.items()
    }
    found = {}
    for source in sorted({sources[episode] for episode in ratios}):
        named = sorted(episode for episode in ratios if sources[episode] == source)
        judged = [ratios[episode] for episode in named if ratios[episode] is not None]
        found[source] = {
            "episodes": len(named),
            "judged": len(judged),
            "share_helped": float(np.mean([ratio < 1 for ratio in judged])) if judged else None,
            "score": relative(
                [row for row in chosen if sources[row[UNIT]] == source], "model", "base"
            )["score"],
        }
    return {"target": target_level, "by_episode": ratios, "by_source": found}


def _constraint_gate(shares: dict[str, dict[int, float | None]]) -> dict:
    """``pi_c > 1 %`` for every informative constraint, per episode.

    A constraint that fixes a point — the facts of an episode, expanded to ``value ± tau`` —
    would have its probability measure the tolerance rather than the scenario, so it is not
    applicable and is left out of the gate. An intervention is a decision, not an observation,
    and is not in the gate at all.
    """
    applicable = 0
    not_applicable = 0
    failed = []
    for episode, found in shares.items():
        for position, share in found.items():
            if share is None:
                not_applicable += 1
                continue
            applicable += 1
            if share <= CONSTRAINT_THRESHOLD:
                failed.append(f"{episode}#{position}")
    return {
        "rule": CONSTRAINT,
        "threshold": CONSTRAINT_THRESHOLD,
        "applicable": applicable,
        "not_applicable": not_applicable,
        "failed_constraints": failed,
        "failed": bool(failed),
    }


def _decision_block(report: dict, against: Path, plan: Episodes, target_level: str) -> dict:
    """The two-standard-error verdict against the incumbent's details, on the same plan."""
    if not against.exists():
        raise ValueError(f"--against {against}: the file does not exist")
    incumbent = json.loads(against.read_text(encoding="utf-8"))
    if incumbent.get("meta", {}).get("episodes_toml") != file_digest(plan.path):
        raise ValueError(
            "--against was produced from a different episode plan; "
            "the two versions must be compared on the same cells"
        )
    return verdict(
        report["rows"],
        report["sanity"]["admission"],
        incumbent,
        unit=UNIT,
        level=target_level,
        block=1,
        seed=plan.seed,
    )


def _tail_section(
    windows: list[Window], plan: Episodes, portfolio_id: str, *, store: Path, known_on: date
) -> dict:
    """The Acerbi-Székely section: the portfolio tail backtest over the episodes.

    The conditional forecast of every episode valued through the engine gives the predictive
    loss distribution; the realized loss comes from the portfolio value on the facts at both
    ends of the window. An episode without the facts of its window or with a position the engine
    could not calculate is dropped with its reason. Admission does not read ``Z``.
    """
    from mst.portfolio import tool as portfolio_tool

    from .es import conditional_paths, realized_paths, tail_section

    # The portfolios live beside the store, in the workspace the run was pointed at.
    portfolios_dir = Path(store).parent / "portfolios"
    try:
        portfolio = portfolio_tool.load(portfolios_dir, portfolio_id)
    except FileNotFoundError:
        raise ValueError(
            f"--portfolio {portfolio_id}: no portfolio {portfolio_id!r} in {portfolios_dir}"
        ) from None
    rows = []
    for window in windows:
        realized, missing = realized_paths(
            window.store,
            as_of=window.as_of,
            known_on=known_on,
            months=window.months,
            panel=window.panel,
            realized=window.realized,
            bundle=window.bundle,
        )
        rows.append(
            {
                "id": window.id,
                "as_of": window.as_of,
                "known_on": known_on,
                "horizon": window.horizon,
                "months": window.months,
                "paths": conditional_paths(
                    window.macro_paths, window.driver_paths, window.bundle, seed=plan.seed
                ),
                "realized": realized,
                "realized_missing": missing,
                "drivers": window.bundle.registry,
                "served": window.bundle.served(),
            }
        )
    return tail_section(rows, portfolio, seed=plan.seed)


def _meta(
    windows: list[Window],
    plan: Episodes,
    *,
    plugin,
    known_on: date,
    store: Path,
    models: Path,
    root: Path,
) -> dict:
    """What a rerun needs to reproduce the report: versions, data vintage, seed and commit."""
    satellites = set()
    for window in windows:
        if window.bundle is not None:
            for model_id, model in window.bundle.plugins.items():
                satellites.add((model_id, model.version))
    return {
        "generated": date.today().isoformat(),
        "known_on": known_on.isoformat(),
        "draws": plan.draws,
        "seed": plan.seed,
        "after_months": plan.after_months,
        "episodes_toml": file_digest(plan.path),
        "macro_registry": file_digest(MACRO_REGISTRY_PATH),
        "store": str(store),
        "models": str(models),
        "backtest_root": str(root),
        **provenance(),
        "macro_model": {"model_id": plugin.model_id, "version": plugin.version},
        "satellite_models": [
            {"model_id": model_id, "version": version} for model_id, version in sorted(satellites)
        ],
        "macro_versions": sorted({window.macro.name for window in windows}),
        "satellite_versions": sorted(
            {window.bundle_name for window in windows if window.bundle_name}
        ),
        "missing": list(MISSING),
    }


def print_report(report: dict) -> None:
    """The report as Markdown: what fell out and the sanity first, the scores after them."""
    print("## Episodes: what fell out\n")
    print_dropped(report["dropped"], judged=len(report["episodes"]), unit="episodes")
    gate = report["constraint_gate"]
    print(
        f"\n`{gate['rule']}`: {gate['applicable']} applicable constraint(s), "
        f"{gate['not_applicable']} not applicable (exact point constraints), "
        f"threshold {gate['threshold']:.3f} — {'failed' if gate['failed'] else 'pass'}."
    )
    print_judgement(report, title="Episodes", reference_calibration=True)
    _print_utility(report)
    _print_episodes(report)
    if report.get("monte_carlo"):
        found = report["monte_carlo"]
        low, high = found["interval"] or (None, None)
        print(
            f"\nMonte-Carlo interval of model/walk at `{report['target_level']}` (the draws "
            f"resampled, not the episodes): {number(low)} .. {number(high)}."
        )
    if report.get("decision"):
        print_decision(report["decision"])
    if report.get("tail"):
        _print_tail(report)
    if report.get("rolling"):
        from .rolling import print_rolling

        print_rolling(report["rolling"])
    meta = report["meta"]
    print("\n## Reproducibility\n")
    print(
        f"- data known on {meta['known_on']}; draws {meta['draws']}; seed {meta['seed']}; "
        f"git commit {meta['git_commit'] or 'unknown'}"
    )
    print(f"- episodes plan {meta['episodes_toml']}; macro registry {meta['macro_registry']}")
    model = meta["macro_model"]
    print(f"- macro {model['model_id']} {model['version']} at {meta['backtest_root']}")
    for model in meta["satellite_models"]:
        print(f"- satellites {model['model_id']} {model['version']}")


def _print_utility(report: dict) -> None:
    """What the scenario adds: the forecast against the base run, by source."""
    utility = report["utility"]
    print(f"\n## Episodes: usefulness of the scenario (model/base at `{utility['target']}`)\n")
    print("| source | episodes | judged | share helped | model / base |")
    print("|---|---:|---:|---:|---:|")
    for source, entry in utility["by_source"].items():
        print(
            f"| {source} | {entry['episodes']} | {entry['judged']} "
            f"| {number(entry['share_helped'])} | {number(entry['score'])} |"
        )
    if "llm" not in utility["by_source"]:
        print(
            "\nAll scenarios are marked manual. An llm scenario counts only in calculations "
            "forward of the date it was formalised."
        )


def _print_episodes(report: dict) -> None:
    print("\n## Episodes: list\n")
    print(
        "| episode | as of | horizon | observations | interventions | model/base "
        "| min constraint probability |"
    )
    print("|---|---|---:|---|---|---:|---|")
    ratios = report["utility"]["by_episode"]
    for row in report["episodes"]:
        shares = row["constraint_probabilities"]
        known = [share for share in shares.values() if share is not None]
        exact = sum(1 for share in shares.values() if share is None)
        worst = f"{min(known):.2f}" if known else "n/a"
        if exact:
            worst += f" ({exact} n/a)"
        print(
            f"| {row['id']} | {row['as_of']} | {row['horizon']} "
            f"| {', '.join(row['observations']) or '-'} | {', '.join(row['interventions']) or '-'} "
            f"| {number(ratios.get(row['id']))} | {worst} |"
        )


def _print_tail(report: dict) -> None:
    """The Acerbi-Székely ``Z``, its intervals, the trimming and the episodes left out."""
    tail = report["tail"]
    horizon = tail.get("horizon", {})
    print("\n## Tail (Acerbi-Székely)\n")
    print(f"- horizon: {horizon.get('kind', 'episode window')}, month {horizon.get('month', '-')}")
    interval = tail.get("interval") or (None, None)
    wild = tail.get("wild") or (None, None)
    print(
        f"- Z {number(tail.get('z'))}; episode interval {number(interval[0])} .. "
        f"{number(interval[1])}; wild {number(wild[0])} .. {number(wild[1])}; "
        f"wild disagrees {tail.get('wild_disagrees')}"
    )
    print(
        f"- trimmed Z with {tail.get('trim')} of the heaviest draws dropped: "
        f"{number(tail.get('trimmed_z'))}"
    )
    print(
        f"- episodes judged {len(tail.get('episodes', []))}, dropped {len(tail.get('dropped', []))}"
    )
    for dropped in tail.get("dropped", []):
        print(f"- dropped {dropped['id']}: {dropped['reason']}")


def compact(report: dict) -> dict:
    """The report without its cells, small enough to keep in the repository.

    The cells stay in the ``--details`` file: a decision against this version reads them there.
    """
    plain = {key: value for key, value in report.items() if key != "rows"}
    if plain.get("rolling"):
        rolling = {key: value for key, value in plain["rolling"].items() if key != "rows"}
        if "price" in rolling:
            rolling["price"] = {
                key: value for key, value in rolling["price"].items() if key != "rows"
            }
        plain["rolling"] = rolling
    return plain


def jsonable(value):
    if isinstance(value, np.floating | np.integer | np.bool_):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set | frozenset):
        return sorted(value)
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(type(value).__name__)
