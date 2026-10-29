"""A run on the paths of a scenario: the files of one run in the workspace.

``runs/<id>/`` holds ``request.json`` (what to run, written before the job starts), ``meta.json``
(the scenario, the two versions, the seed and the horizon), ``paths.npz`` (the macro variables and
drivers of every repetition) and the results of every position: ``results.json`` with the metrics
and the reason of a position that was not calculated, and ``changes.npz`` with the value change
of each repetition. A run of a portfolio also writes ``summary.json`` — its loss at P5, P50 and
P95 and the severity ``delta`` of the scenario.

A run is a job: the tool writes the request and starts ``python -m mst.runs <id>``; the job builds
the paths from the scenario and writes the rest. Repeating a request with the same seed gives the
same numbers.
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
from .drivers import version as satellite_version
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

    @model_validator(mode="after")
    def one_source(self):
        if (self.positions is None) == (self.portfolio is None):
            raise ValueError("a run names positions or a portfolio, not both and not neither")
        ids = [position.id for position in self.positions or []]
        if len(set(ids)) != len(ids):
            raise ValueError("two positions share one id")
        return self


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
    """Check ``request`` and write it as the run's ``request.json``; the job computes the rest."""
    checked = RunRequest.model_validate(request).model_dump(mode="json")
    store.write_json(folder(ws, run_id) / "request.json", checked)
    return checked


def execute(ws: store.Workspace, run_id: str) -> dict:
    """Compute the run of its request and write the paths and the results; the job's entry.

    The run's ``meta.json`` says ``running`` from the first moment, so a failed job leaves the
    reason next to the request instead of an empty folder.
    """
    directory = folder(ws, run_id)
    request = RunRequest.model_validate(store.read_json(directory / "request.json"))
    meta = {
        "id": run_id,
        "scenario": request.scenario,
        "draws": request.draws,
        "seed": request.seed,
        "base_currency": request.base_currency,
        "status": "running",
        "started": _now(),
    }
    store.write_json(directory / "meta.json", meta)
    try:
        spec = scenario_tool.load(ws.scenarios, request.scenario)
        model, stored = macro_version.load(ws.models, spec.macro)
        satellites = satellite_version.load(ws.models, spec.satellites)
        paths = scenario_build.build(
            spec, model, satellites, draws=request.draws, seed=request.seed
        )
        portfolio = (
            portfolio_tool.load(ws.portfolios, request.portfolio) if request.portfolio else None
        )
        positions = portfolio.positions if portfolio is not None else request.positions
        base_currency = (
            portfolio.base_currency if portfolio is not None else request.base_currency
        )
        batch = calculate_run(
            positions,
            readable(paths),
            drivers=satellites.registry,
            served=satellites.served(),
            base_currency=base_currency,
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
    written = _write(
        directory, run_id, request, spec, stored, paths, batch, portfolio, base_currency
    )
    return {"id": run_id, "status": written["status"], "positions": len(batch.results)}


def meta(ws: store.Workspace, run_id: str) -> dict:
    """The meta of a run; a run whose job has not written it yet has only its request."""
    stored = store.read_json(folder(ws, run_id) / "meta.json")
    if stored is None:
        raise FileNotFoundError(f"run {run_id!r} has not started")
    return stored


def result(ws: store.Workspace, run_id: str) -> dict:
    """The meta, the per-position results of a run and the summary of a portfolio run."""
    stored = meta(ws, run_id)
    positions = store.read_json(folder(ws, run_id) / "results.json")
    return {
        "run": stored,
        "positions": [] if positions is None else positions["positions"],
        "summary": store.read_json(folder(ws, run_id) / "summary.json"),
    }


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
) -> dict:
    """The paths, the results of every position, the report and the meta that names them."""
    months = [month.isoformat() for month in paths.months]
    # An invalid position keeps its column of zeros: the report reads the status, never the column.
    changes = np.zeros((paths.draws, len(batch.results)))
    for index, result in enumerate(batch.results):
        if result.value_change:
            changes[:, index] = result.value_change
    np.savez(directory / "paths.npz", macro=paths.macro, drivers=paths.drivers.paths)
    np.savez(directory / "changes.npz", changes=changes)
    store.write_json(
        directory / "results.json",
        {
            "positions": [
                result.model_dump(mode="json", exclude={"value_change"})
                for result in batch.results
            ]
        },
    )
    summary = (
        portfolio_report.build(portfolio, batch.results, severity_found=paths.severity)
        if portfolio is not None
        else None
    )
    if summary is not None:
        store.write_json(directory / "summary.json", summary)
    written = {
        "id": run_id,
        "scenario": request.scenario,
        "narrative": spec.narrative,
        "macro": spec.macro,
        "satellites": spec.satellites,
        "as_of": stored["as_of"],
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
        "severity": portfolio_report.severity(paths.severity),
        "models": {
            result.position_id: {
                "model": result.model_id,
                "version": result.model_version,
            }
            for result in batch.results
        },
        "position_ids": [result.position_id for result in batch.results],
        "positions": len(batch.results),
        "status": summary["status"] if summary is not None else batch.status,
        "finished": _now(),
    }
    store.write_json(directory / "meta.json", written)
    return written


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


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
