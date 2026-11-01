"""``python -m mst.scenario reverse``: the reverse problem of the canon, section 6.

::

  python -m mst.scenario reverse ID --workspace DIR --portfolio ID --draws N --seed S
      (--kappa K | --level L) [--points ...]

Either the worst loss among the family's scenarios priced at ``kappa`` or less
(``--kappa``), or the cheapest price among those whose expected shortfall reaches
``level`` (``--level``). The family is the named scenario with every condition narrowed and
widened around its midpoint by one factor; the versions, the impacts and the causes stay
put — a family whose impacts move with the factor is refused, because a free impact costs
nothing and a bound on the price would bound nothing. Nothing is fetched.
"""

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    from mst import plugins, store
    from mst.drivers import bundle as satellite_bundle
    from mst.macro import version as macro_version
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
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--kappa", type=float, help="the price bound: maximise rho under it")
    group.add_argument("--level", type=float, help="the loss level: minimise the price over it")
    args = parser.parse_args(argv)

    ws = store.Workspace(args.workspace)
    spec = scenario_tool.load(ws.scenarios, args.scenario)
    installed = plugins.registry()
    macro = macro_version.load(ws.models, spec.macro, installed.macro)
    satellites = satellite_bundle.load(ws.models, spec.satellites, installed.satellites)
    portfolio = portfolio_tool.load(ws.portfolios, args.portfolio)
    family = width_family(spec, macro)
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
