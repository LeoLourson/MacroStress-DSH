"""``python -m mst.drivers``: discover the drivers, estimate their satellites and check them.

::

  python -m mst.drivers update [QUANTITY ...] --store DIR --drivers FILE
  python -m mst.drivers estimate --macro SHA256 --store DIR --models DIR --drivers FILE
  python -m mst.drivers check --macro SHA256 --store DIR --models DIR --drivers FILE

``update`` runs the catalog of every quantity of ``src/mst/drivers/registry.toml`` (or of the
named ones), fetches the series of the objects found from January 2012 on and writes the drivers
with history to ``--drivers``. It sends requests only to the sources the catalogs name and exits
with 1 if anything failed; what was fetched stays.

``estimate`` reads the drivers of the registry and of ``--drivers`` from the series store as
known on the calculation date of the macro version, estimates their satellites on its panel
(``src/mst/drivers/spec.toml``), checks each one out of sample and writes the version to
``<models>/satellites/<sha256>/``. ``check`` does the same and prints the table without writing
a version. Both print the check of every driver as a Markdown table. Nothing is fetched.
"""

import argparse
import sys
from datetime import date
from pathlib import Path

from ..data.cli import START, client
from ..data.store import SeriesStore
from ..macro.registry import load_registry as load_macro_registry
from .registry import load_registry
from .spec import load_spec
from .update import update
from .version import build, macro_panel, save


def _number(value: float | None) -> str:
    return "-" if value is None else f"{value:.3g}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mst.drivers")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("update", help="discover drivers and fetch their series")
    fetch.add_argument("quantities", nargs="*", help="quantity ids; all by default")
    fetch.add_argument("--store", type=Path, required=True)
    fetch.add_argument("--drivers", type=Path, required=True)
    for name, help_ in (
        ("estimate", "estimate, check and publish the satellites"),
        ("check", "estimate and check the satellites, publish nothing"),
    ):
        run = commands.add_parser(name, help=help_)
        run.add_argument("--macro", required=True, help="the macro model version")
        run.add_argument("--store", type=Path, required=True)
        run.add_argument("--models", type=Path, required=True)
        run.add_argument("--drivers", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "update":
        return _update(args)

    registry, spec = load_registry(catalog=args.drivers), load_spec()
    registry.check(load_macro_registry())
    panel, as_of = macro_panel(args.models, args.macro)
    sat = build(registry, spec, panel, SeriesStore(args.store), as_of, args.macro)
    if args.command == "estimate":
        print(f"version {save(sat, args.models)}")
    print(f"macro {args.macro}, as of {as_of}, t = 0 at {panel.months[-1]:%Y-%m}")
    print()
    print("| driver | months | RMSE satellite | RMSE without macro | forecasts | served |")
    print("|---|---|---|---|---|---|")
    for quantity, checks in sat.checks.items():
        fit_ = sat.fits[quantity]
        for node, item in sorted(checks.items()):
            served = "yes" if item.served else f"no: {item.reason}"
            print(
                f"| {fit_.history.driver[node]} | {fit_.rows[node]} | {_number(item.satellite)} "
                f"| {_number(item.baseline)} | {item.forecasts} | {served} |"
            )
    print()
    print(f"served {len(sat.served())} of {sum(len(c) for c in sat.checks.values())}")
    return 0


def _update(args) -> int:
    registry = load_registry(catalog=args.drivers)
    unknown = sorted(set(args.quantities) - {q.id for q in registry.quantities})
    if unknown:
        print(f"unknown quantities: {', '.join(unknown)}", file=sys.stderr)
        return 2
    with client() as http:
        outcome = update(
            registry,
            SeriesStore(args.store),
            http,
            args.drivers,
            fetched_on=date.today(),
            start=START,
            only=args.quantities or None,
        )
    for quantity, count in outcome.drivers.items():
        print(f"{quantity}: {count} drivers")
    print(f"rows added: {sum(outcome.added.values())} in {len(outcome.added)} series")
    for name, reason in outcome.skipped.items():
        print(f"{name}: skipped, {reason}")
    for name, reason in outcome.failed.items():
        print(f"{name}: failed, {reason}", file=sys.stderr)
    return 1 if outcome.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
