"""``python -m mst.metrics``: the episode backtest of the models.

::

  python -m mst.metrics backtest [MODEL_ID] --as-of DATE --store DIR --models DIR --corpus FILE
      [--drivers FILE] [--satellites] [--only ID ...] [--details FILE] [--against REPORT.json]
      [--portfolio ID] [--mc-bootstrap]

``backtest`` estimates the macro model (the named one, or the only one installed) on the day
before every episode of ``episodes.toml`` starts, builds the scenario from what actually
happened, scores the conditional forecast against the fact — CRPS normalised by the typical
movement known at that episode's calculation date, calibration, sharpness, coverage, the
admission rules pooled per rule — and reports it next to the random walk with drift, the
unconditional base run and, for targets, the naive forecast of a year earlier. Every score comes
with the 95 % episode-bootstrap interval, re-checked by the wild cluster bootstrap with Webb
weights; the equal-weight aggregate of section 5 is reported per level and horizon, with the
utility of the scenario split by the manual/llm mark, and the tuning/validation halves apart.
``--against`` names the incumbent report's JSON for the four-step 2-SE decision;
``--portfolio`` names a stored portfolio for the Acerbi-Székely tail section;
``--mc-bootstrap`` adds the Monte-Carlo interval of the aggregate, separated from the
episode-sampling intervals. With ``--satellites`` the served drivers are scored too, on the
bundle estimated on each episode's macro version. It exits with 1 when an episode could not be
checked or an admission rule failed. Nothing is fetched.
"""

import argparse
import sys
from datetime import date
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    from .backtest import run

    parser = argparse.ArgumentParser(prog="mst.metrics", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("backtest", help="the episode check of the models")
    check.add_argument("model_id", nargs="?", help="the macro model id; the only one installed")
    check.add_argument("--as-of", type=date.fromisoformat, required=True)
    check.add_argument("--store", type=Path, required=True)
    check.add_argument("--models", type=Path, required=True)
    check.add_argument("--corpus", type=Path, required=True)
    check.add_argument("--drivers", type=Path, help="the driver catalog, with --satellites")
    check.add_argument("--satellites", action="store_true", help="score the drivers too")
    check.add_argument("--only", nargs="+", default=[], help="episode ids; all by default")
    check.add_argument("--details", type=Path, help="write the full report as JSON here")
    check.add_argument(
        "--against",
        type=Path,
        help="the incumbent report's JSON, same episode plan: the 2-SE decision of section 5",
    )
    check.add_argument(
        "--portfolio", help="a stored portfolio id: the Acerbi-Székely tail section"
    )
    check.add_argument(
        "--mc-bootstrap",
        action="store_true",
        help="bootstrap the draws for the Monte-Carlo interval of the aggregate",
    )
    check.add_argument(
        "--target-level",
        default="macro",
        help="the level whose aggregate the decision and the utility read (macro by default)",
    )
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        # Russian labels survive a Windows pipe.
        sys.stdout.reconfigure(encoding="utf-8")
    return run(
        store=args.store,
        models=args.models,
        corpus=args.corpus,
        as_of=args.as_of,
        macro_id=args.model_id,
        drivers=args.drivers,
        satellites=args.satellites,
        only=args.only,
        details=args.details,
        against=args.against,
        portfolio=args.portfolio,
        mc_bootstrap=args.mc_bootstrap,
        target_level=args.target_level,
    )


if __name__ == "__main__":
    sys.exit(main())
