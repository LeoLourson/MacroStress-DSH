"""``python -m mst.metrics``: the episode backtest of the models.

::

  python -m mst.metrics backtest [MODEL_ID] --as-of DATE --store DIR --models DIR --episodes FILE
      [--drivers FILE] [--only ID ...] [--details FILE] [--against REPORT.json]
      [--portfolio ID] [--mc-bootstrap] [--no-rolling] [--save] [--workers N]
  python -m mst.metrics rolling [MODEL_ID] --as-of DATE --store DIR --models DIR --episodes FILE
      [--dates YYYY-MM-DD ...] [--half tuning|validation] [--details FILE] [--against REPORT.json]
      [--drivers FILE] [--portfolio ID] [--target-level LEVEL] [--workers N]

``rolling`` judges the forecast itself: on a grid of dates nobody selected, the unconditional
forecast of the model estimated on that date against the random walk, in relative CRPS, with
the calibration and a block-bootstrap interval. ``--dates`` narrows the grid to these
dates; ``--against`` names the incumbent report for the decision.

``backtest`` estimates the macro model (the named one, or the only one installed) on the day
before every episode of the plan ``--episodes`` starts, builds the scenario from what actually
happened and judges the forecast on the cells the scenario does not fix. The report reads what
fell out first — the episodes not judged and the variables a model left out as too short, with
the reasons — then the sanity of every quantity (admission per quantity, the band against the
history range, the direction the scenario moved the forecast, the score of the level without
the quantity), and only then the relative CRPS: the forecast against the walk that knows the
scenario's path, against the plain walk with drift, against its own base run and, for targets,
against the naive forecast of a year earlier, with 95 % episode-bootstrap intervals, per level,
the usefulness of the scenario split by the manual/llm mark, and the tuning/validation halves
apart. ``--against`` names the incumbent's ``--details`` JSON for the four-step 2-SE decision;
``--portfolio`` names a stored portfolio for the Acerbi-Székely tail section;
``--mc-bootstrap`` adds the Monte-Carlo interval of the forecast against the walk, separated
from the episode-sampling interval. With ``--drivers`` the served drivers are scored too, on
the bundle estimated on each episode's macro version. It exits with 1 when an episode could
not be checked or a quantity failed admission. ``--save`` keeps the report without its cells
and the tables in ``results/<model>.json`` and ``results/<model>.md`` beside the plan, over
the last result of the model. Nothing is fetched.

Both commands first estimate the versions every date needs and the cache lacks, in ``--workers``
processes (half the logical processors by default, ``mst.metrics.prepare``); the report does not
depend on their number.
"""

import argparse
import sys
from datetime import date
from pathlib import Path

PLAN = "the episode plan, bench/model-backtest/episodes.toml in the repository"


def main(argv: list[str] | None = None) -> int:
    from .backtest import run
    from .rolling import run as run_rolling

    if hasattr(sys.stdout, "reconfigure"):
        # Help and Russian labels must also survive a Windows pipe.
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="mst.metrics", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("backtest", help="the episode check of the models")
    check.add_argument("model_id", nargs="?", help="the macro model id; the only one installed")
    check.add_argument("--as-of", type=date.fromisoformat, required=True)
    check.add_argument("--store", type=Path, required=True)
    check.add_argument("--models", type=Path, required=True)
    check.add_argument("--episodes", type=Path, required=True, help=PLAN)
    check.add_argument("--drivers", type=Path, help="the driver catalog: judge the drivers too")
    check.add_argument("--only", nargs="+", default=[], help="episode ids; all by default")
    check.add_argument("--details", type=Path, help="write the full report as JSON here")
    check.add_argument(
        "--against",
        type=Path,
        help="the incumbent's --details JSON, same episode plan: the two-standard-error decision",
    )
    check.add_argument(
        "--portfolio",
        help="a stored portfolio id: judge its positions and its tail (needs --drivers)",
    )
    check.add_argument(
        "--mc-bootstrap",
        action="store_true",
        help="bootstrap the draws for the Monte-Carlo interval of the forecast against the walk",
    )
    check.add_argument(
        "--target-level",
        default="macro",
        help="the level whose aggregate the decision and the utility read (macro by default)",
    )
    check.add_argument(
        "--no-rolling", action="store_true", help="leave out the rolling check of the plan"
    )
    check.add_argument(
        "--inputs",
        type=Path,
        help="the folder of the answers a model with declared inputs reads (workspace/elicitation)",
    )
    check.add_argument(
        "--save",
        action="store_true",
        help="keep the report without its cells in results/ beside the plan, over the last one",
    )
    grid = commands.add_parser("rolling", help="the forecast quality over a grid of dates")
    grid.add_argument("model_id", nargs="?", help="the macro model id; the only one installed")
    grid.add_argument("--as-of", type=date.fromisoformat, required=True)
    grid.add_argument("--store", type=Path, required=True)
    grid.add_argument("--models", type=Path, required=True)
    grid.add_argument("--episodes", type=Path, required=True, help=PLAN)
    grid.add_argument(
        "--dates", nargs="+", type=date.fromisoformat, help="only these dates of the grid"
    )
    grid.add_argument("--half", choices=("tuning", "validation"), help="only this half of the grid")
    grid.add_argument("--details", type=Path, help="write the full report as JSON here")
    grid.add_argument("--against", type=Path, help="the incumbent's rolling --details JSON")
    grid.add_argument(
        "--inputs",
        type=Path,
        help="the folder of the answers a model with declared inputs reads (workspace/elicitation)",
    )
    grid.add_argument("--drivers", type=Path, help="the driver catalog: judge the drivers too")
    grid.add_argument(
        "--portfolio",
        help="a stored portfolio id: judge its positions and its tail (needs --drivers)",
    )
    grid.add_argument(
        "--target-level",
        default="macro",
        help="the level the decision reads (macro by default)",
    )
    tried = commands.add_parser(
        "trial", help="the false rejections and the power of the criteria on synthetic worlds"
    )
    tried.add_argument("--worlds", type=int, default=200, help="worlds per model (200)")
    tried.add_argument("--seed", type=int, default=0)
    tried.add_argument(
        "--defects", nargs="*", default=None, help="the models to try; every defect by default"
    )
    for command in (check, grid, tried):
        command.add_argument(
            "--workers",
            type=int,
            default=None,
            help="processes estimating the versions of the dates; half the processors by default",
        )
    args = parser.parse_args(argv)
    from .prepare import default_workers

    workers = default_workers() if args.workers is None else args.workers
    if workers < 1:
        parser.error("--workers must be at least 1")

    if args.command == "trial":
        from .trial import CRITERIA, DEFECTS, World, trial

        defects = args.defects or list(DEFECTS)
        table = trial(World(), defects, worlds=args.worlds, seed=args.seed, workers=workers)
        print(f"| model | {' | '.join(CRITERIA)} | any |")
        print(f"|---|{'---:|' * (len(CRITERIA) + 1)}")
        for defect, shares in table.items():
            cells = " | ".join(f"{shares[name]:.3f}" for name in (*CRITERIA, "any"))
            print(f"| {defect} | {cells} |")
        return 0
    if args.command == "rolling":
        return run_rolling(
            store=args.store,
            models=args.models,
            episodes=args.episodes,
            as_of=args.as_of,
            macro_id=args.model_id,
            dates=args.dates,
            half=args.half,
            details=args.details,
            against=args.against,
            inputs=args.inputs,
            drivers=args.drivers,
            portfolio=args.portfolio,
            target_level=args.target_level,
            workers=workers,
        )
    return run(
        store=args.store,
        models=args.models,
        episodes=args.episodes,
        as_of=args.as_of,
        macro_id=args.model_id,
        drivers=args.drivers,
        only=args.only,
        details=args.details,
        against=args.against,
        portfolio=args.portfolio,
        mc_bootstrap=args.mc_bootstrap,
        target_level=args.target_level,
        rolling=not args.no_rolling,
        inputs=args.inputs,
        save=args.save,
        workers=workers,
    )


if __name__ == "__main__":
    sys.exit(main())
