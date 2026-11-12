"""``python -m mst.scenario reverse``: the reverse problem of the canon, section 6.

::

  python -m mst.scenario reverse ID --workspace DIR --portfolio ID --draws N --seed S
      (--kappa K | --level L) [--macro HASH] [--satellites HASH]

Either the worst loss among the family's scenarios priced at ``kappa`` or less
(``--kappa``), or the cheapest price among those whose expected shortfall reaches
``level`` (``--level``). The family is the named scenario with every support condition
narrowed and widened around its midpoint by one factor; the interventions stay put, and the
versions reach the search as arguments: ``--macro`` and ``--satellites`` name them, and the
newest finished pair of the workspace is taken when they are omitted. The scenario file names
no versions. A family whose interventions move with the factor is refused, because a free
intervention costs nothing and a bound on the price would bound nothing. Nothing is fetched.
"""

import argparse
import json
import sys
from pathlib import Path


def versions_of(ws, macro_name: str | None = None, satellites_name: str | None = None):
    """The pair the search runs on: the explicit names or the newest finished pair.

    The scenario file no longer names the versions, so the command line picks the pair the
    workspace holds through the same selection ``validate`` uses: the newest macro version and
    the newest satellites estimated on it, unless the caller names them.
    """
    from mst.scenario import tool as scenario_tool

    return scenario_tool.versions_of(ws.models, macro=macro_name, satellites=satellites_name)


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "migrate":
        from mst.scenario.migrate import migrate

        parser = argparse.ArgumentParser(prog="mst.scenario migrate")
        parser.add_argument("directory", type=Path)
        roots = parser.add_mutually_exclusive_group(required=True)
        roots.add_argument("--workspace", type=Path)
        roots.add_argument("--models", type=Path)
        args = parser.parse_args(arguments[1:])
        try:
            count = migrate(args.directory, args.models or args.workspace / "models")
        except (ValueError, OSError, KeyError) as error:
            print(str(error), file=sys.stderr)
            return 1
        print(f"Migrated {count} scenario(s).")
        return 0
    from mst import store
    from mst.portfolio import tool as portfolio_tool
    from mst.scenario import tool as scenario_tool
    from mst.scenario.reverse import inf_dist, sup_rho, width_family

    parser = argparse.ArgumentParser(prog="mst.scenario", description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["reverse"])
    parser.add_argument("scenario", help="the scenario id whose family is searched")
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--portfolio", required=True, help="the portfolio whose loss is rho")
    parser.add_argument("--draws", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--macro", help="the macro version; the newest finished one when omitted")
    parser.add_argument(
        "--satellites", help="the satellites; the newest estimated on that macro when omitted"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--kappa", type=float, help="the price bound: maximise rho under it")
    group.add_argument("--level", type=float, help="the loss level: minimise the price over it")
    args = parser.parse_args(argv)

    ws = store.Workspace(args.workspace)
    spec = scenario_tool.load(ws.scenarios, args.scenario)
    macro, satellites = versions_of(ws, args.macro, args.satellites)
    portfolio = portfolio_tool.load(ws.portfolios, args.portfolio)
    family = width_family(spec, macro, satellites)
    if args.kappa is not None:
        found = sup_rho(
            family,
            args.kappa,
            macro=macro,
            satellites=satellites,
            positions=portfolio.positions,
            draws=args.draws,
            seed=args.seed,
        )
    else:
        found = inf_dist(
            family,
            args.level,
            macro=macro,
            satellites=satellites,
            positions=portfolio.positions,
            draws=args.draws,
            seed=args.seed,
        )
    print(json.dumps(found, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
