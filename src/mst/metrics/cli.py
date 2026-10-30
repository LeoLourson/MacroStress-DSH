"""``python -m mst.metrics``: the episode backtest of the models.

::

  python -m mst.metrics backtest [MODEL_ID] --as-of DATE --store DIR --models DIR --corpus FILE
      [--drivers FILE] [--satellites] [--only ID ...] [--details FILE]

``backtest`` estimates the macro model (the named one, or the only one installed) on the day
before every episode of ``episodes.toml`` starts, builds the scenario from what actually
happened, scores the conditional forecast against the fact — CRPS normalised by the typical
movement, calibration, sharpness, the admission rules — and compares it with the base run and
the random walk with drift. With ``--satellites`` the served drivers are scored too, on the
bundle estimated on each episode's macro version. It exits with 1 when an episode could not
be checked or the admission failed. Nothing is fetched.
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
    )


if __name__ == "__main__":
    sys.exit(main())
