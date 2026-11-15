"""``python -m mst.bench``: the commands of the check sets."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from . import ROOT, results
from . import __doc__ as PACKAGE_DOC


def _print(text: str) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(text)


def _scenarios(_args) -> int:
    from .scenarios import scenario_summary

    summary = scenario_summary()
    file = results(ROOT / "scenario-from-corpus") / "summary.md"
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(summary, encoding="utf-8", newline="\n")
    _print(summary)
    return 0


def _elicitation(args) -> int:
    from . import elicitation

    if args.describe is not None:
        parts = json.loads(Path(args.describe).read_text(encoding="utf-8"))
        _print(json.dumps(elicitation.describe(parts), ensure_ascii=False, indent=2))
        return 0
    if args.root is None:
        raise SystemExit("--root names the experiment's own folder")
    if args.root.resolve() == args.source.resolve():
        raise SystemExit("--root must differ from the source workspace")
    experiment = elicitation.Experiment(
        root=args.root.resolve(),
        source=args.source,
        gateway_model=args.model,
        model_id=args.model_id,
        as_of=args.as_of,
        key=args.key,
    )
    _print(elicitation.compact(asyncio.run(elicitation.run(experiment))))
    return 0


def _forecast(args) -> int:
    from . import forecast

    plan = forecast.load_plan(args.plan)
    card = forecast.run(
        plan,
        workspace=args.workspace,
        macro=args.macro,
        satellites=args.satellites,
        rolling=not args.no_rolling,
    )
    files = forecast.save(card, args.plan.parent)
    _print(forecast.markdown(card))
    _print("\n".join(str(file) for file in files))
    return 0 if card["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mst.bench", description=PACKAGE_DOC.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("scenarios", help="the verdict shares of scenario-from-corpus")

    found = commands.add_parser("forecast", help="the forecast scorecard of the models")
    found.add_argument("--plan", type=Path, default=ROOT / "forecast" / "plan.toml")
    found.add_argument("--workspace", type=Path, default=Path("workspace"))
    found.add_argument("--macro", required=True, help="the macro version hash or its prefix")
    found.add_argument("--satellites", required=True, help="the satellite version or prefix")
    found.add_argument(
        "--no-rolling", action="store_true", help="skip the out-of-sample dates (minutes)"
    )

    asked = commands.add_parser(
        "elicitation", help="the context experiment of the expert interview (gateway, opt-in)"
    )
    asked.add_argument("--root", type=Path, help="the experiment's own folder")
    asked.add_argument("--source", type=Path, default=Path("workspace"))
    asked.add_argument("--model", default="gpt-6-astra", help="the gateway's model id")
    asked.add_argument("--model-id", default="expert.elicited.rw")
    asked.add_argument("--as-of", default="2026-11-05")
    asked.add_argument("--key", default="gl.oil")
    asked.add_argument(
        "--describe", type=Path, help="describe the saved answers of one key instead"
    )

    args = parser.parse_args(argv)
    handlers = {"scenarios": _scenarios, "forecast": _forecast, "elicitation": _elicitation}
    return handlers[args.command](args)
