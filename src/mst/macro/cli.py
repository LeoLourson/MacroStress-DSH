"""``python -m mst.macro``: estimate the macro model and publish a model version.

::

  python -m mst.macro estimate --as-of 2026-10-29 --store DIR --models DIR
  python -m mst.macro backtest --as-of 2026-10-29 --store DIR --models DIR --corpus FILE

``estimate`` builds the panel known on ``--as-of`` from the series store, draws the posterior
of ``src/mst/macro/spec.toml``, identifies the shocks of the registry and writes the version to
``<models>/macro/<sha256>/``. It prints the hash and the diagnostics. Nothing is fetched.

``backtest`` checks the model on the episodes of ``src/mst/macro/backtest.toml`` with the
realized values known on ``--as-of``, once with the sign table of the registry and once with
each ``--signs`` file (a TOML file of ``[[shocks]]`` in the registry format). Episode start
months come from the corpus manifest at ``--corpus``. It prints the coverage of ``[P5, P95]`` by
episode and by variable; ``--details`` also writes every month's quantiles and realized value.
"""

import argparse
import json
import tomllib
from datetime import date
from pathlib import Path

from ..corpus import manifest
from ..data.store import SeriesStore
from .backtest import coverage, explosive_share, load_backtest, models_for, prepare
from .estimate import estimate
from .forecast import InfeasibleScenario, simulate
from .identify import identify
from .panel import build_panel
from .registry import Registry, load_registry
from .spec import load_spec
from .version import save


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mst.macro")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("estimate", help="estimate and publish a model version")
    run.add_argument("--as-of", type=date.fromisoformat, required=True)
    run.add_argument("--store", type=Path, required=True)
    run.add_argument("--models", type=Path, required=True)
    check = commands.add_parser("backtest", help="check the model on historical episodes")
    check.add_argument("--as-of", type=date.fromisoformat, required=True)
    check.add_argument("--signs", type=Path, action="append", default=[])
    check.add_argument("--only", nargs="+", default=[])
    check.add_argument("--details", type=Path)
    check.add_argument("--store", type=Path, required=True)
    check.add_argument("--models", type=Path, required=True)
    check.add_argument("--corpus", type=Path, required=True)
    args = parser.parse_args(argv)

    spec, registry = load_spec(), load_registry()
    if args.command == "backtest":
        return backtest(args, spec, registry)
    panel = build_panel(
        SeriesStore(args.store),
        registry,
        start=spec.start,
        as_of=args.as_of,
        seasonal=spec.seasonal,
    )
    posterior = estimate(panel, spec)
    model = identify(
        posterior,
        registry.shocks,
        sign_lag=spec.sign_lag,
        tries=spec.rotation_tries,
        seed=spec.sampler.seed,
    )
    name = save(model, spec, panel, args.as_of, args.models)
    print(name)
    print(json.dumps(model.diagnostics, ensure_ascii=False))
    return 0


def backtest(args, spec, registry: Registry) -> int:
    plan = load_backtest()
    tables = {"registry": registry}
    for path in args.signs:
        with path.open("rb") as file:
            shocks = tomllib.load(file)["shocks"]
        tables[path.stem] = Registry.model_validate(
            {**registry.model_dump(mode="json"), "shocks": shocks}
        )
    starts = {e.id: e.start_month for e in manifest.load(args.corpus).episodes}
    unknown = [id_ for id_ in args.only if id_ not in {c.id for c in plan.episodes}]
    if unknown:
        raise SystemExit(f"not in backtest.toml: {', '.join(unknown)}")
    store = SeriesStore(args.store)
    rows, problems, details = [], [], {}
    hits = {name: {} for name in tables}
    counts = {name: {} for name in tables}
    for check in plan.episodes:
        if args.only and check.id not in args.only:
            continue
        try:
            case, panel, history = prepare(
                check, starts[check.id], plan, store, registry, spec, args.as_of
            )
            models = models_for(
                panel, history, case.as_of, {k: t.shocks for k, t in tables.items()}, args.models
            )
        except ValueError as error:
            # Too short a history on the calculation date: the episode is reported, not run.
            problems.append(f"{check.id}: {error}")
            rows.append([check.id, "-", "-", "", "", *["not estimated"] * len(tables)])
            continue
        cells = []
        for name, table in tables.items():
            table.check(case.scenario)
            version, model = models[name]
            try:
                simulation = simulate(model, case.scenario, draws=plan.draws, seed=plan.seed)
            except InfeasibleScenario as error:
                problems.append(f"{check.id} / {name}: {error}")
                cells.append("infeasible")
                continue
            result = coverage(case, simulation)
            for id_ in result.counts:
                hits[name][id_] = hits[name].get(id_, 0) + result.hits[id_]
                counts[name][id_] = counts[name].get(id_, 0) + result.counts[id_]
            cells.append(
                f"{_share(sum(result.hits.values()), sum(result.counts.values()))}; "
                f"{model.draws}/{model.diagnostics['estimated']} draws, "
                f"{explosive_share(model):.0%} explosive"
            )
            details.setdefault(check.id, {})[name] = {
                "version": version,
                "as_of": case.as_of.isoformat(),
                "months": [m.isoformat() for m in case.months],
                "ids": simulation.ids,
                "constrained": case.constrained,
                "realized": _rows(case.realized),
                "p5": _rows(result.quantiles[0]),
                "p50": _rows(result.quantiles[1]),
                "p95": _rows(result.quantiles[2]),
            }
        rows.append(
            [
                check.id,
                case.as_of.isoformat(),
                str(len(case.months)),
                ", ".join(check.constrained),
                ", ".join(check.driving),
                *cells,
            ]
        )
    names = list(tables)
    print("## Episodes\n")
    _table(["episode", "as of", "months", "constrained", "driving", *names], rows)
    print("\n## Variables\n")
    ids = [v.id for v in registry.variables if any(v.id in counts[n] for n in names)]
    _table(
        ["variable", *names],
        [
            [id_, *(_share(hits[n].get(id_, 0), counts[n].get(id_, 0)) for n in names)]
            for id_ in ids
        ],
    )
    print("\n## All\n")
    _table(
        ["", *names],
        [["all", *(_share(sum(hits[n].values()), sum(counts[n].values())) for n in names)]],
    )
    for problem in problems:
        print(problem)
    if args.details:
        args.details.write_text(json.dumps(details, ensure_ascii=False), encoding="utf-8")
    return 0


def _share(hits: int, count: int) -> str:
    return f"{hits / count:.2f} ({count})" if count else "-"


def _rows(array) -> list[list[float | None]]:
    return [[None if x != x else round(float(x), 4) for x in row] for row in array]


def _table(header: list[str], rows: list[list[str]]) -> None:
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in rows:
        print("| " + " | ".join(row) + " |")
