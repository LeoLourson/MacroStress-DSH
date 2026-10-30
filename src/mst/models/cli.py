"""``python -m mst.models``: the model jobs of the ``model`` tool.

::

  python -m mst.models estimate-macro [MODEL_ID] --as-of DATE --store DIR --models DIR
  python -m mst.models backtest [MODEL_ID] --as-of DATE --store DIR --models DIR --corpus FILE
  python -m mst.models list --models DIR
  python -m mst.models describe --version SHA256 --models DIR
  python -m mst.models catalog

``estimate-macro`` estimates the named macro model (or the only one installed) on the panel
known on ``--as-of`` and writes the version to ``<models>/macro/<sha256>/``; it prints the
hash. ``backtest`` runs the episode check of the model that implements one. ``list`` and
``describe`` read the versions of the workspace; ``catalog`` lists the installed model
plugins with the quantities their satellites serve.
"""

import argparse
import sys
from datetime import date
from pathlib import Path

from . import catalog, estimate_macro, selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mst.models", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    for name, help_ in (
        ("estimate-macro", "estimate a macro model and publish its version"),
        ("backtest", "the episode check of a macro model that implements one"),
    ):
        run = commands.add_parser(name, help=help_)
        run.add_argument("model_id", nargs="?", help="the macro model id; the only one installed")
        run.add_argument("--as-of", type=date.fromisoformat, required=True)
        run.add_argument("--store", type=Path, required=True)
        run.add_argument("--models", type=Path, required=True)
        run.add_argument("--only", nargs="+", default=[], help="episode ids; all by default")
        run.add_argument("--details", type=Path, help="write every month's quantiles here")
    backtest = commands.choices["backtest"]
    backtest.add_argument("--corpus", type=Path, required=True)

    listed = commands.add_parser("list", help="the model versions of the workspace")
    listed.add_argument("--models", type=Path, required=True)
    described = commands.add_parser("describe", help="one version, as its meta records it")
    described.add_argument("--version", required=True)
    described.add_argument("--models", type=Path, required=True)
    commands.add_parser("catalog", help="the installed model plugins")

    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        # Russian labels survive a Windows pipe.
        sys.stdout.reconfigure(encoding="utf-8")

    if args.command == "estimate-macro":
        print(estimate_macro(args.model_id, store=args.store, models=args.models, as_of=args.as_of))
        return 0
    if args.command == "backtest":
        model = selected(args.model_id)
        runner = getattr(model, "backtest", None)
        if not callable(runner):
            print(f"{model.model_id} implements no episode check", file=sys.stderr)
            return 2
        return runner(
            store=args.store,
            models=args.models,
            corpus=args.corpus,
            as_of=args.as_of,
            only=args.only,
            details=args.details,
        )
    if args.command == "list":
        from mst.macro import version as macro_version

        for row in macro_version.catalogue(args.models):
            print(row["version"], row["model"], row["as_of"])
        return 0
    if args.command == "describe":
        import json

        from mst.macro import version as macro_version

        print(json.dumps(macro_version.describe(args.models, args.version), ensure_ascii=False))
        return 0
    import json

    print(json.dumps(catalog(), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
