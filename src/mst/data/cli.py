"""``python -m mst.data update``: fetch the monthly series into the workspace.

::

  python -m mst.data update [SERIES ...] --workspace DIR

``update`` fetches every series of ``src/mst/data/sources.toml`` (or only the named ones) from
January 2012 on and adds the new versions to ``<workspace>/series/<id>.csv``. It sends requests
only to the sources named there and exits with 1 if any series failed.
"""

import argparse
import ssl
import sys
from datetime import date
from pathlib import Path

import httpx

from ..store import Workspace
from .sources import SOURCES_PATH, load_sources
from .store import SeriesStore
from .update import update

START = date(2012, 1, 1)
"""T0 of docs/scenario-implementation-plan.md, section 4, decision 1."""

ROOTS_PATH = SOURCES_PATH.with_name("mintsifry-roots.pem")
"""Roots and sub CAs of the Russian Trusted Root CA (Mintsifry), which signs rosstat.gov.ru."""


def rosstat_context() -> ssl.SSLContext:
    """The default verification plus the Mintsifry roots, for rosstat.gov.ru only."""
    context = ssl.create_default_context()
    context.load_verify_locations(cafile=str(ROOTS_PATH))
    return context


def client() -> httpx.Client:
    """A verifying client: every host is checked, rosstat.gov.ru against the extra roots too."""
    return httpx.Client(
        timeout=60,
        follow_redirects=True,
        mounts={"all://*rosstat.gov.ru": httpx.HTTPTransport(verify=rosstat_context())},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mst.data")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("update", help="fetch series into the workspace")
    run.add_argument("series", nargs="*", help="series ids; all by default")
    run.add_argument("--workspace", required=True, type=Path)
    args = parser.parse_args(argv)

    sources = load_sources()
    unknown = sorted(set(args.series) - set(sources))
    if unknown:
        parser.error(f"no source for {', '.join(unknown)}")
    chosen = {series: sources[series] for series in args.series} if args.series else sources

    store = SeriesStore(Workspace(args.workspace).series)
    with client() as http:
        outcome = update(store, chosen, http, fetched_on=date.today(), start=START)
    for series, added in outcome.added.items():
        print(f"{series}: +{added}")
    for series, reason in outcome.failed.items():
        print(f"{series}: failed, {reason}", file=sys.stderr)
    return 1 if outcome.failed else 0
