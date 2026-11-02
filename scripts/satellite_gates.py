"""The gates of the satellite repair plan, read from the artefacts of a saved base run.

::

  python scripts/satellite_gates.py --workspace DIR --run ID [--month 12]
      [--imoex ru.imoex] [--usdrub ru.usdrub] [--shares DRIVER ...]

The run must be the base scenario of the bundle the gates judge. The script prints S1-S3 and
S6 of the plan, the share of paths that leave the training range of their driver, and, for
every named driver, the band at ``--month`` twice: the full band of the run and the band with
every macro path fixed at its median, so the difference is what the macro paths add over the
residuals. S3 is recomputed from the paths in the model scale of every driver — the scale
error of the old ``_checks`` made the recorded shares of runs before the canon alignment
wrong, so the recorded share is printed beside it, not trusted.

Nothing here estimates anything: the macro simulation of the run is drawn again with the same
seed, so the residuals-only band shares the random numbers of the run and the decomposition
is exact.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

S1_LIMIT = 10.0
S2_LIMIT = 35.0
S3_LIMIT = 0.05
S6_LIMIT = 2.5
"""The gates of ``plans/satellite-models.md``: fixed in advance, never tuned here."""


def gates(
    ws,
    run: str,
    *,
    month: int = 12,
    imoex: str = "ru.imoex",
    usdrub: str = "ru.usdrub",
    shares: list[str] | None = None,
) -> dict:
    """The gates of the base run ``run``; every number is read or recomputed, none estimated."""
    from mst import plugins
    from mst.drivers import bundle as satellite_bundle
    from mst.macro import version as macro_version
    from mst.runs import folder as run_folder
    from mst.runs import meta as run_meta

    meta = run_meta(ws, run)
    if month > meta["horizon"]:
        raise ValueError(
            f"run {run} covers a horizon of {meta['horizon']} months, the gates read month "
            f"{month}: pass --month within the horizon"
        )
    installed = plugins.registry()
    macro = macro_version.load(ws.models, meta["macro"], installed.macro)
    satellites = satellite_bundle.load(ws.models, meta["satellites"], installed.satellites)
    with np.load(run_folder(ws, run) / "paths.npz") as stored:
        macro_paths, driver_paths = stored["macro"], stored["drivers"]

    served = [
        driver
        for driver in meta["drivers"]
        if driver in set(satellites.served())
        and satellites.quantity_of(driver) == "share_price"
    ]
    named = shares or served
    unknown = [driver for driver in named if driver not in meta["drivers"]]
    if unknown:
        raise ValueError(f"the run carries no paths of {', '.join(unknown)}")

    index = _index_return(meta, macro_paths, imoex, month)
    returns = {
        driver: _log_return(
            driver_paths[:, :, meta["drivers"].index(driver)],
            satellites.registry.quantity("share_price").transform,
            month,
        )
        for driver in named
    }
    s1 = float(np.median(list(returns.values())) - index)
    return {
        "run": run,
        "bundle": meta["satellites"],
        "macro": meta["macro"],
        "month": month,
        "shares": list(named),
        "s1": {"value": s1, "limit": S1_LIMIT, "holds": abs(s1) <= S1_LIMIT},
        "s2": {
            driver: {
                "value": value - index,
                "limit": S2_LIMIT,
                "holds": abs(value - index) <= S2_LIMIT,
            }
            for driver, value in returns.items()
        },
        "s3": _s3(meta, satellites, driver_paths),
        "s6": _s6(satellites, named, usdrub),
        "training_range": _training_range(meta, satellites, driver_paths),
        "band": _bands(ws, meta, macro, satellites, driver_paths, named, month),
    }


def _log_return(paths: np.ndarray, transform: str, month: int) -> float:
    """The median log return of a driver from month 0 to ``month``, in log points."""
    from mst.drivers.registry import path_to_model

    column = path_to_model(transform, paths)
    return float(np.median(column[:, month] - column[:, 0]))


def _index_return(meta: dict, macro_paths: np.ndarray, imoex: str, month: int) -> float:
    """The median log return of the index over the same months, in log points."""
    from mst.drivers.registry import path_to_model
    from mst.macro.registry import load_registry

    if imoex not in meta["variables"]:
        raise ValueError(
            f"the run carries no paths of {imoex}: the variables are "
            f"{', '.join(meta['variables'])}"
        )
    variable = load_registry().variable(imoex)
    column = path_to_model(variable.transform, macro_paths[:, :, meta["variables"].index(imoex)])
    return float(np.median(column[:, month] - column[:, 0]))


def _s3(meta: dict, satellites, driver_paths: np.ndarray) -> dict:
    """The share of driver cells whose band is wider than history, recomputed from the paths.

    The recorded share of the run is printed beside it: a run made before the scale error of
    ``_checks`` was fixed records wrong shares, and the plan judges the recomputed one.
    """
    from mst.metrics.baselines import WIDTH, quantity_domain
    from mst.runs import _fan_problems

    shares = []
    for driver in meta["drivers"]:
        quantity, node = satellites.node(driver)
        declared = satellites.registry.quantity(quantity)
        own = satellites.arrays.get(f"{quantity}.d")
        if own is None:
            continue
        problems = _fan_problems(
            _model_scale(driver_paths[:, :, meta["drivers"].index(driver)], declared),
            own[:, node],
            quantity_domain(declared),
        )
        if problems:
            shares.append(problems.get(WIDTH, 0.0))
    recorded = [
        row.get(WIDTH, 0.0)
        for name, row in meta["checks"]["paths"].items()
        if name.startswith("driver/")
    ]
    share = float(np.mean(shares)) if shares else 0.0
    return {
        "share": share,
        "recorded": float(np.mean(recorded)) if recorded else 0.0,
        "limit": S3_LIMIT,
        "holds": share < S3_LIMIT,
    }


def _model_scale(paths: np.ndarray, declared) -> np.ndarray:
    """A driver's own-unit paths back in the model scale the history and domain are in."""
    from mst.drivers.registry import path_to_model

    return path_to_model(declared.transform, paths)


def _s6(satellites, named: list[str], usdrub: str) -> dict:
    """The largest absolute long-run elasticity to the dollar among the named shares."""
    from mst.drivers.bundle import long_run_loadings

    part = satellites.parts["share_price"]
    gamma = satellites.arrays["share_price.gamma"]
    if usdrub not in part["quantity"]["regressors"]:
        return {
            "largest": None,
            "driver": None,
            "limit": S6_LIMIT,
            "holds": None,
            "note": f"share_price reads no {usdrub}",
        }
    largest, who = 0.0, None
    for driver in named:
        node = part["driver"].index(driver)
        loading = float(np.median(long_run_loadings(part["quantity"], gamma, node)[usdrub]))
        if abs(loading) > largest:
            largest, who = abs(loading), driver
    return {"largest": largest, "driver": who, "limit": S6_LIMIT, "holds": largest < S6_LIMIT}


def _training_range(meta: dict, satellites, driver_paths: np.ndarray) -> dict:
    """The share of paths that leave the trained range of their driver, by quantity."""
    out: dict[str, dict] = {}
    for driver in meta["drivers"]:
        quantity, node = satellites.node(driver)
        part = satellites.parts[quantity]
        low = (part.get("train_low") or [None] * len(part["driver"]))[node]
        high = (part.get("train_high") or [None] * len(part["driver"]))[node]
        if low is None or high is None:
            continue
        declared = satellites.registry.quantity(quantity)
        column = _model_scale(driver_paths[:, :, meta["drivers"].index(driver)], declared)
        outside = float(((column[:, 1:] < low) | (column[:, 1:] > high)).any(axis=1).mean())
        out.setdefault(quantity, {"median": None, "drivers": {}})["drivers"][driver] = outside
    for row in out.values():
        values = list(row["drivers"].values())
        row["median"] = float(np.median(values)) if values else None
    return out


def _bands(ws, meta, macro, satellites, driver_paths, named, month) -> dict:
    """The band of every named driver at ``month``: the full one and the residuals-only one."""
    from mst.model_sdk.contracts import MacroSimulation
    from mst.scenario import build as scenario_build
    from mst.scenario import tool as scenario_tool

    full = {}
    for driver in named:
        declared = satellites.registry.quantity("share_price")
        column = _model_scale(driver_paths[:, :, meta["drivers"].index(driver)], declared)
        full[driver] = float(
            np.percentile(column[:, month], 95) - np.percentile(column[:, month], 5)
        )
    # The same macro model, the same seed, every path fixed at its median: the noise the
    # satellites draw is the noise of the run, so only x and the posterior differ.
    spec = scenario_tool.load(ws.scenarios, meta["scenario"])
    base = spec.model_copy(
        update={
            "constraints": [],
            "impacts": [],
            "expectations": [],
            "probabilities": [],
            "causes": [],
            "regime_event": None,
        }
    )
    simulation = macro.plugin.simulate(
        macro.model,
        scenario_build.plan_with_impacts(base, macro),
        draws=meta["draws"],
        seed=meta["seed"],
    )
    fixed = MacroSimulation(
        ids=simulation.ids,
        months=simulation.months,
        paths=np.tile(np.median(simulation.paths, axis=0), (simulation.paths.shape[0], 1, 1)),
        history=(
            np.tile(np.median(simulation.history, axis=0), (simulation.history.shape[0], 1, 1))
            if simulation.history is not None
            else None
        ),
    )
    drawn = satellites.simulate(fixed, list(meta["drivers"]), seed=meta["seed"])
    residuals = {}
    for driver in named:
        column = drawn.paths[:, month, drawn.ids.index(driver)]
        residuals[driver] = float(np.percentile(column, 95) - np.percentile(column, 5))
    return {
        driver: {"full": full[driver], "residuals_only": residuals[driver]} for driver in named
    }


def _printed(report: dict) -> str:
    """The gates as Markdown: one row per gate and one per named driver."""
    s3, s6 = report["s3"], report["s6"]
    lines = [
        f"# Satellite gates of run {report['run']}",
        "",
        f"Bundle `{report['bundle']}` on macro `{report['macro']}`, month {report['month']}.",
        "",
        "| gate | value | limit | holds |",
        "|---|---|---|---|",
        f"| S1 shares minus the index | {report['s1']['value']:+.1f} | "
        f"{report['s1']['limit']:.0f} | {_yes(report['s1']['holds'])} |",
        f"| S3 driver cells wider than history | {100 * s3['share']:.1f}% "
        f"(recorded {100 * s3['recorded']:.1f}%) | {100 * s3['limit']:.0f}% | "
        f"{_yes(s3['holds'])} |",
        f"| S6 largest elasticity to the dollar | {_number(s6['largest'])} "
        f"({s6['driver'] or '-'}) | {s6['limit']:.1f} | {_yes(s6['holds'])} |",
        "",
        "| share | return minus the index (S2) | band full | band residuals only |",
        "|---|---|---|---|",
    ]
    for driver, row in report["s2"].items():
        band = report["band"][driver]
        lines.append(
            f"| {driver} | {row['value']:+.1f} | {band['full']:.1f} | "
            f"{band['residuals_only']:.1f} |"
        )
    lines += ["", "Paths leaving the training range, by quantity (median over drivers):"]
    for quantity, row in report["training_range"].items():
        lines.append(f"- {quantity}: {100 * row['median']:.1f}%")
    return "\n".join(lines)


def _number(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _yes(holds: bool | None) -> str:
    return "yes" if holds else "no" if holds is not None else "n/a"


def main(argv: list[str] | None = None) -> int:
    """Read the artefacts of a base run and print its gates."""
    parser = argparse.ArgumentParser(
        prog="satellite_gates", description=__doc__.split("\n\n")[0]
    )
    parser.add_argument("--workspace", type=Path, required=True, help="the workspace directory")
    parser.add_argument("--run", required=True, help="the base run to read")
    parser.add_argument("--month", type=int, default=12, help="the month of the returns")
    parser.add_argument("--imoex", default="ru.imoex", help="the index the shares are judged by")
    parser.add_argument("--usdrub", default="ru.usdrub", help="the dollar rate of S6")
    parser.add_argument(
        "--shares",
        nargs="*",
        help="the named shares of S2, S6 and the bands; all served shares by default",
    )
    args = parser.parse_args(argv)
    from mst import store

    report = gates(
        store.Workspace(args.workspace),
        args.run,
        month=args.month,
        imoex=args.imoex,
        usdrub=args.usdrub,
        shares=args.shares,
    )
    print(_printed(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
