"""``python -m mst.drivers update``: discover the drivers and fetch their series.

``update`` runs the catalog of every quantity of ``src/mst/drivers/registry.toml`` (or of the
named ones), fetches the series of the objects found from January 2012 on and writes the drivers
with history to ``--drivers``. It sends requests only to the sources the catalogs name and exits
with 1 if anything failed; what was fetched stays. The satellites of a macro version are
estimated by ``python -m mst.models estimate-satellites`` — the models live in plugins.
"""

import argparse
import sys
from datetime import date
from pathlib import Path

from ..data.cli import START, client
from ..data.store import SeriesStore
from .registry import load_registry
from .update import update


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mst.drivers")
    parser.add_argument("command", choices=["update"], help="discover drivers and fetch series")
    parser.add_argument("quantities", nargs="*", help="quantity ids; all by default")
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--drivers", type=Path, required=True)
    args = parser.parse_args(argv)
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
