"""A run on the paths of a scenario: the files of one run in the workspace.

``runs/<id>/`` holds ``request.json`` (what to run, written before the job starts), ``meta.json``
(the scenario, the two versions, the seed and the horizon), ``paths.npz`` (the macro variables and
drivers of every repetition) and the results of every position: ``results.json`` with the metrics
and the reason of a position that was not calculated, and ``changes.npz`` with the value change
of each repetition. The conditional forecast of the target indicators — the quantiles of every
target driver by month, in its own unit — is written to ``forecast.json``: a run answers both
the distribution of the indicators and the portfolio on them. A run of a portfolio also writes
``summary.json`` — its loss at P5, P50 and P95, the severity ``delta`` of the scenario and the
platform's ``dist(P_s, P)`` beside it; the run's ``meta.json`` records the definition of
``dist`` (the plan's assumption) and the way its probability was estimated.

A run is a job: the tool writes the request and starts ``python -m mst.runs <id>``; the job builds
the paths from the scenario and writes the rest. Repeating a request with the same seed gives the
same numbers. The status of a run is the report's for a portfolio run — ``ok``, ``needs_review``
or ``incomplete`` — and the batch's otherwise: ``ok``, ``needs_review`` or ``invalid``.
"""

import argparse
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pydantic import Field, model_validator

from . import store
from .contracts import Contract, Currency, Identifier, Seed
from .drivers import bundle as satellite_bundle
from .engine import run as calculate_run
from .macro import version as macro_version
from .plugin_sdk import FixedPaths, Position
from .portfolio import report as portfolio_report
from .portfolio import tool as portfolio_tool
from .scenario import build as scenario_build
from .scenario import tool as scenario_tool
from .scenario.spec import Spec

ID = re.compile(r"^[0-9]{8}T[0-9]{6}-[0-9a-f]{4}$")
"""A run's id is also the name of its folder: the job's timestamp and four hex digits."""


class RunRequest(Contract):
    """What to run: a scenario, how many repetitions, the seed and what is calculated on them."""

    scenario: Identifier
    draws: int = Field(strict=True, ge=1)
    seed: Seed
    #: The base currency of ad-hoc ``positions``; a portfolio brings its own.
    base_currency: Currency = "RUB"
    positions: list[Position] | None = None
    portfolio: Identifier | None = None
    targets: list[Identifier] | None = None
    """The target indicators of the conditional forecast, by driver id; every driver by default."""
    contributions: bool = False
    """Whether to compute the contribution of every condition and impact: the same run
    without the element, on the same random numbers, its loss beside the full one's."""

    @model_validator(mode="after")
    def one_source(self):
        if (self.positions is None) == (self.portfolio is None):
            raise ValueError("a run names positions or a portfolio, not both and not neither")
        if self.positions is not None and not self.positions:
            raise ValueError("a run needs at least one position")
        if self.base_currency != "RUB":
            raise ValueError("the base currency is RUB: the models price against the rouble")
        ids = [position.id for position in self.positions or []]
        if len(set(ids)) != len(ids):
            raise ValueError("two positions share one id")
        if self.targets is not None:
            if not self.targets:
                raise ValueError("targets name the drivers of the forecast, or be omitted")
            _named_once(self.targets)
        return self


def _named_once(targets: list[str]) -> None:
    """Refuse a target named twice: the rows of the forecast are keyed by the driver."""
    seen = set()
    for target in targets:
        if target in seen:
            raise ValueError(f"target {target} is named twice")
        seen.add(target)


def new_id(now: datetime | None = None) -> str:
    """A new run id: the UTC time and four hex digits, the format of a job id."""
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{os.urandom(2).hex()}"


def folder(ws: store.Workspace, run_id: str) -> Path:
    """The folder of run ``run_id``; an id that is not a run id is refused."""
    if not ID.fullmatch(run_id or ""):
        raise ValueError(f"{run_id!r} is not a run id")
    return ws.runs / run_id


def begin(ws: store.Workspace, run_id: str, request: dict) -> dict:
    """Check ``request``, create the run's folder and write ``request.json``.

    The folder is created exclusively: a run id that already exists is not overwritten.
    """
    checked = RunRequest.model_validate(request).model_dump(mode="json")
    directory = folder(ws, run_id)
    directory.mkdir(parents=True, exist_ok=False)
    store.write_json(directory / "request.json", checked)
    return checked


def execute(ws: store.Workspace, run_id: str) -> dict:
    """Compute the run of its request and write the paths, the results and the report.

    The run's ``meta.json`` says ``running`` from the first moment, so a job that dies leaves the
    reason next to the request instead of an empty folder.
    """
    from mst import plugins

    directory = folder(ws, run_id)
    request = RunRequest.model_validate(store.read_json(directory / "request.json"))
    meta = {
        "id": run_id,
        "scenario": request.scenario,
        "draws": request.draws,
        "seed": request.seed,
        "status": "running",
        "started": _now(),
    }
    store.write_json(directory / "meta.json", meta)
    try:
        spec = scenario_tool.load(ws.scenarios, request.scenario)
        installed = plugins.registry()
        macro = macro_version.load(ws.models, spec.macro, installed.macro)
        satellites = satellite_bundle.load(ws.models, spec.satellites, installed.satellites)
        if satellites.macro != spec.macro:
            raise ValueError(
                f"the satellites {spec.satellites} were estimated on the macro version "
                f"{satellites.macro}, the scenario names {spec.macro}"
            )
        paths = scenario_build.build(
            spec, macro, satellites, draws=request.draws, seed=request.seed
        )
        # The empty scenario on the same seed and streams: what the forecast is compared
        # against and what the constraint gate reads.
        empty = scenario_build.build(
            _without_everything(spec),
            macro,
            satellites,
            draws=request.draws,
            seed=request.seed,
        )
        portfolio = (
            portfolio_tool.load(ws.portfolios, request.portfolio) if request.portfolio else None
        )
        positions = portfolio.positions if portfolio is not None else request.positions
        base_currency = portfolio.base_currency if portfolio is not None else request.base_currency

        def valued(found) -> object:
            return calculate_run(
                positions,
                readable(found),
                drivers=satellites.registry,
                served=satellites.served(),
                base_currency=base_currency,
                as_of=macro.as_of,
            )

        batch = valued(paths)
        contributions = None
        if request.contributions:
            full = portfolio_report.loss_block(
                batch.results, seed=request.seed, horizon=paths.months[-1].isoformat()
            )
            contributions = []
            for name, variant in _variants(spec):
                variant_paths = scenario_build.build(
                    variant, macro, satellites, draws=request.draws, seed=request.seed
                )
                found = portfolio_report.loss_block(
                    valued(variant_paths).results,
                    seed=request.seed,
                    horizon=variant_paths.months[-1].isoformat(),
                )
                contributions.append(
                    {
                        "element": name,
                        "loss": found,
                        "difference": None
                        if found is None or full is None
                        else {
                            key: found[key] - full[key]
                            for key in ("p05", "p50", "p95", "es95")
                            if found.get(key) is not None and full.get(key) is not None
                        },
                    }
                )
        written = _write(
            directory,
            run_id,
            request,
            spec,
            macro.meta,
            paths,
            batch,
            portfolio,
            base_currency,
            meta["started"],
            forecast(paths, request.targets),
            s0_forecast=forecast(empty, request.targets),
            checks=_checks(spec, macro, paths, empty, satellites),
            contributions=contributions,
            workspace=ws,
        )
    except BaseException as error:
        store.write_json(
            directory / "meta.json",
            meta
            | {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "finished": _now(),
            },
        )
        raise
    return {"id": run_id, "status": written["status"], "positions": written["positions"]}


def meta(ws: store.Workspace, run_id: str) -> dict:
    """The meta of a run; a run whose job has not written it yet has only its request."""
    stored = store.read_json(folder(ws, run_id) / "meta.json")
    if stored is None:
        raise FileNotFoundError(f"run {run_id!r} has not started")
    return stored


def result(ws: store.Workspace, run_id: str) -> dict:
    """The meta, the forecast, the results and the summary of a run."""
    stored = meta(ws, run_id)
    positions = store.read_json(folder(ws, run_id) / "results.json")
    return {
        "run": stored,
        "positions": [] if positions is None else positions["positions"],
        "forecast": store.read_json(folder(ws, run_id) / "forecast.json"),
        "summary": store.read_json(folder(ws, run_id) / "summary.json"),
    }


def forecast(paths, targets: list[str] | None = None) -> dict:
    """The conditional forecast of the target indicators: quantiles by month, in own units.

    The targets name drivers of the paths; without them every driver is a target. A named
    driver the paths do not carry is an error, not a silent omission.
    """
    drivers = list(paths.drivers.ids)
    chosen = list(targets) if targets is not None else drivers
    _named_once(chosen)
    missing = [driver for driver in chosen if driver not in drivers]
    if missing:
        raise ValueError(
            f"the paths carry no drivers {', '.join(missing)}; they carry "
            f"{', '.join(drivers) or 'none'}"
        )
    months = [month.isoformat() for month in paths.months]
    rows = {}
    for driver in chosen:
        k = drivers.index(driver)
        column = paths.drivers.paths[:, 1:, k]
        low, mid, high = np.percentile(column, [5, 50, 95], axis=0)
        rows[driver] = {
            "p05": [round(float(x), 6) for x in low],
            "p50": [round(float(x), 6) for x in mid],
            "p95": [round(float(x), 6) for x in high],
            "flags": sorted(paths.drivers.flags[driver]),
        }
    return {"targets": chosen, "months": months[1:], "drivers": rows}


def catalogue(ws: store.Workspace) -> dict:
    """Every run folder, newest first, with the meta it has written so far."""
    if not ws.runs.is_dir():
        return {"runs": []}
    found = []
    for directory in sorted(ws.runs.iterdir(), reverse=True):
        if not directory.is_dir() or not ID.fullmatch(directory.name):
            continue
        stored = store.read_json(directory / "meta.json") or {}
        found.append(
            {
                "id": directory.name,
                "status": stored.get("status", "requested"),
                "scenario": stored.get("scenario"),
                "as_of": stored.get("as_of"),
                "horizon": stored.get("horizon"),
                "draws": stored.get("draws"),
                "seed": stored.get("seed"),
                "positions": stored.get("positions"),
                "finished": stored.get("finished"),
            }
        )
    return {"runs": found}


def readable(paths) -> FixedPaths:
    """The scenario's paths as a position model reads them (``mst.plugin_sdk.Paths``)."""
    return FixedPaths(
        macros={name: paths.macro[..., index] for index, name in enumerate(paths.ids)},
        drivers={
            name: paths.drivers.paths[..., index] for index, name in enumerate(paths.drivers.ids)
        },
        driver_flags={name: paths.drivers.flags[name] for name in paths.drivers.ids},
        start=paths.months[0],
        seed=paths.seed,
    )


def _write(
    directory: Path,
    run_id: str,
    request: RunRequest,
    spec: Spec,
    stored: dict,
    paths,
    batch,
    portfolio,
    base_currency: str,
    started: str,
    forecast_rows: dict,
    *,
    s0_forecast: dict | None = None,
    checks: dict | None = None,
    contributions: list | None = None,
    workspace: store.Workspace | None = None,
) -> dict:
    """The paths, the results of every position, the report and the meta that names them."""
    months = [month.isoformat() for month in paths.months]
    # A position that was not calculated has no column: NaN stands in, so nobody reads a zero
    # profit where there is no number at all.
    changes = np.full((paths.draws, len(batch.results)), np.nan)
    for index, result in enumerate(batch.results):
        if result.value_change:
            changes[:, index] = result.value_change
    np.savez(directory / "paths.npz", macro=paths.macro, drivers=paths.drivers.paths)
    np.savez(directory / "changes.npz", changes=changes)
    store.write_json(
        directory / "results.json",
        {
            "positions": [
                result.model_dump(mode="json", exclude={"value_change"}) for result in batch.results
            ]
        },
    )
    summary = (
        portfolio_report.build(
            portfolio,
            batch.results,
            dist_found=paths.dist,
            seed=paths.seed,
            horizon=paths.months[-1].isoformat(),
        )
        if portfolio is not None
        else None
    )
    if summary is not None:
        store.write_json(directory / "summary.json", summary)
    store.write_json(
        directory / "forecast.json", forecast_rows | {"s0": s0_forecast}
    )
    if contributions is not None:
        store.write_json(directory / "contributions.json", {"contributions": contributions})
    written = {
        "id": run_id,
        "scenario": request.scenario,
        "narrative": spec.narrative,
        "macro": spec.macro,
        "satellites": spec.satellites,
        "as_of": stored["as_of"],
        "started": started,
        "finished": _now(),
        "horizon": len(paths.months) - 1,
        "draws": paths.draws,
        "seed": paths.seed,
        "base_currency": base_currency,
        "portfolio": request.portfolio,
        "months": months,
        "variables": list(paths.ids),
        "drivers": list(paths.drivers.ids),
        "driver_flags": {
            driver: sorted(paths.drivers.flags[driver]) for driver in paths.drivers.ids
        },
        "dist": _dist_meta(paths.dist),
        "basis": _basis(spec),
        "checks": checks,
        "git_commit": _git_commit(),
        "data": {"series": str(workspace.series)} if workspace is not None else None,
        "models": {
            result.position_id: {
                "model": result.model_id,
                "version": result.model_version,
            }
            for result in batch.results
        },
        "position_ids": [result.position_id for result in batch.results],
        "positions": len(batch.results),
        "targets": forecast_rows["targets"],
        "status": summary["status"] if summary is not None else batch.status,
    }
    store.write_json(directory / "meta.json", written)
    return written


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _without_everything(spec: Spec) -> Spec:
    """The empty scenario ``s_0`` of the same versions and horizon: no conditions, no impacts."""
    return spec.model_copy(
        update={
            "constraints": [],
            "impacts": [],
            "expectations": [],
            "probabilities": [],
            "causes": [],
            "regime_event": None,
            "inner": 1,
        }
    )


def _variants(spec: Spec):
    """``(name, scenario)`` of every element of the scenario taken out alone."""
    for kind, field in (
        ("constraint", "constraints"),
        ("impact", "impacts"),
        ("expectation", "expectations"),
        ("probability", "probabilities"),
    ):
        elements = getattr(spec, field)
        for k in range(len(elements)):
            without = list(elements)
            del without[k]
            yield f"{kind}#{k + 1}", spec.model_copy(update={field: without})


def _checks(spec: Spec, macro, paths, empty, satellites) -> dict:
    """What the run says about its own conditions and paths (the canon, section 7).

    The probability of every interval condition under the base run — a scenario resting on a
    rare tail is seen as such, and a point condition is not applicable, its probability would
    measure the tolerance — and the two admission rules on the paths of the scenario, macro
    variables and drivers alike, because the conditions can widen the fan beyond what the
    history has ever seen. The shares are over the horizon months: a month whose band leaves
    the admissible domain or grows wider than the history counts against its variable.
    """
    from .drivers.registry import from_model
    from .macro.registry import load_registry as load_macro_registry
    from .metrics.baselines import (
        CONSTRAINT_THRESHOLD,
        constraint_probabilities,
        macro_domain,
        quantity_domain,
    )

    checks: dict = {}
    registry = load_macro_registry()
    plan = scenario_build.plan_of(spec, macro)
    if plan.constraints:
        model_scale = _macro_model_scale(registry, empty)
        checks["constraint_probabilities"] = constraint_probabilities(
            model_scale, list(empty.ids), plan.constraints
        )
        checks["constraint_threshold"] = CONSTRAINT_THRESHOLD
    fan: dict[str, dict[str, float]] = {}
    scenario_scale = _macro_model_scale(registry, paths)
    panel = {id_: macro.panel.y[:, j] for j, id_ in enumerate(macro.panel.ids)}
    for k, id_ in enumerate(paths.ids):
        fan[f"macro/{id_}"] = _fan_problems(
            scenario_scale[:, 1:, k], panel.get(id_), macro_domain()
        )
    for k, driver in enumerate(paths.drivers.ids):
        quantity, node = satellites.node(driver)
        declared = satellites.registry.quantity(quantity)
        own = satellites.arrays.get(f"{quantity}.d")
        column = own[:, node] if own is not None else np.array([])
        fan[f"driver/{driver}"] = _fan_problems(
            from_model(declared.transform, paths.drivers.paths[:, 1:, k]),
            column,
            quantity_domain(declared),
        )
    checks["paths"] = fan
    return checks


def _macro_model_scale(registry, paths) -> np.ndarray:
    """The macro paths of a run back in the model scale every condition reads."""
    return np.stack(
        [
            registry.variable(id_).to_model(paths.macro[..., k])
            for k, id_ in enumerate(paths.ids)
        ],
        axis=-1,
    )


def _fan_problems(draws, history, domain) -> dict[str, float]:
    """The shares of the horizon months whose band breaks each admission rule."""
    from .metrics.baselines import DOMAIN, WIDTH, band_problems
    from .metrics.cells import history_range

    if draws.size == 0:
        return {}
    spread = history_range(history) if history is not None else float("nan")
    months = draws.shape[1]
    broken = {rule: 0 for rule in (DOMAIN, WIDTH)}
    for t in range(months):
        for rule in band_problems(
            draws[:, t], float("nan"), domain=domain, history_range=spread
        ):
            broken[rule] += 1
    return {rule: count / months for rule, count in broken.items()}


def _basis(spec: Spec) -> list[dict]:
    """What every condition and impact of the scenario rests on, element by element."""
    rows = []
    for kind, elements in (
        ("constraint", spec.constraints),
        ("impact", spec.impacts),
        ("expectation", spec.expectations),
        ("probability", spec.probabilities),
    ):
        for k, element in enumerate(elements):
            rows.append(
                {
                    "element": f"{kind}#{k + 1}",
                    "basis": element.basis
                    if not isinstance(element.basis, str)
                    else [element.basis],
                }
            )
    return rows


def _git_commit() -> str | None:
    """The commit of the checkout, read from ``.git`` without running git; best effort."""
    from .metrics.backtest import _git_commit as read

    return read()


def _dist_meta(found) -> dict | None:
    """The price of the scenario with everything the reader needs to judge it.

    The definition is the canon's: ``dist(s) = KL(P_s || P) = E_P[-log P(S | R)]``, the sum
    of the support price and the price of the tilt (docs/concept/formalization.md). The
    standard error reads the spread over the groups; an infinite distance is passed as is,
    with the message naming the contradiction.
    """
    if found is None:
        return None
    return {
        "definition": "dist(s) = KL(P_s || P) (docs/concept/formalization.md, sections 4-5)",
        "distance": float(found.distance),
        "log_probability": float(found.log_probability),
        "standard_error": float(found.standard_error),
        "causes": list(found.causes),
        "tolerances": list(found.tolerances),
        "by_level": {str(level): float(price) for level, price in sorted(found.by_level.items())},
        "message": found.message,
    }


def main(argv: list[str] | None = None) -> None:
    """``python -m mst.runs <run id> --workspace DIR``: the job's process."""
    parser = argparse.ArgumentParser(prog="mst.runs")
    parser.add_argument("run")
    parser.add_argument("--workspace", required=True, type=Path)
    arguments = parser.parse_args(argv)
    summary = execute(store.Workspace(arguments.workspace), arguments.run)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
