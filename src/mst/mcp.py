"""The MCP server: a tool parses its arguments and calls one function of the core."""

import argparse
from datetime import date
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from . import jobs as tasks
from . import models, runs, store
from .corpus import tool as corpus_tool
from .data import sources
from .data.store import SeriesStore
from .drivers import bundle as satellite_bundle
from .macro import version as macro_version
from .portfolio import tool as portfolio_tool
from .scenario import tool as scenario_tool


def build(ws: store.Workspace) -> FastMCP:
    """The server of one workspace; every tool closes over it."""
    server = FastMCP("macrostress")

    @server.tool
    def jobs(action: str, id: str | None = None, tail: int = 50) -> dict:
        """Workspace and task state. Actions: state, status, log, cancel."""
        if action == "state":
            return tasks.state(ws)
        if action == "status":
            return tasks.status(ws, _need(id, "id"))
        if action == "log":
            job_id = _need(id, "id")
            return {"id": job_id, "log": tasks.log(ws, job_id, tail)}
        if action == "cancel":
            return tasks.cancel(ws, _need(id, "id"))
        raise ValueError(f"jobs: unknown action {action!r}, try state, status, log, cancel")

    @server.tool
    def series(
        action: str,
        id: str | None = None,
        as_of: str | None = None,
        text: str | None = None,
        entry: dict[str, Any] | None = None,
        ids: list[str] | None = None,
        quantity: str | None = None,
    ) -> dict:
        """Stored series. Actions: search, get, add, update.

        ``update`` fetches the named ``ids``; with ``quantity`` it instead discovers the
        objects of that driver quantity and fetches their series.
        """
        if action == "search":
            return {"sources": sources.search(_need(text, "text"), extra=ws.sources)}
        if action == "get":
            return _rows(ws, _need(id, "id"), as_of)
        if action == "add":
            return {"id": sources.add(_need(entry, "entry"), ws.sources)}
        if action == "update":
            if quantity:
                if ids:
                    raise ValueError("series update: give ids or a quantity, not both")
                return _update_drivers(ws, quantity)
            return _update(ws, ids or [])
        raise ValueError(f"series: unknown action {action!r}, try search, get, add, update")

    @server.tool
    def corpus(
        action: str,
        query: str | None = None,
        as_of: str | None = None,
        id: str | None = None,
        kind: str | None = None,
        region: str | None = None,
        factors: list[str] | None = None,
        k: int = 8,
        key: str | None = None,
        file: str | None = None,
        url: str | None = None,
        ids: list[str] | None = None,
    ) -> dict:
        """The scenario corpus. Actions: search, get, list, add, index, fetch.

        ``search`` and ``get`` read the index as it was on ``as_of``; ``list`` shows what the
        manifest holds; ``add`` registers a file downloaded by hand under its ``key``. ``index``
        and ``fetch`` are tasks on the corpus resource.
        """
        if action == "search":
            return corpus_tool.search(
                ws.corpus,
                _need(query, "query"),
                _day(_need(as_of, "as_of")),
                kind=kind,
                region=region,
                factors=factors,
                k=k,
            )
        if action == "get":
            return corpus_tool.get(ws.corpus, _need(id, "id"), _day(_need(as_of, "as_of")))
        if action == "list":
            return corpus_tool.catalogue(ws.corpus)
        if action == "add":
            return corpus_tool.add(ws.corpus, _need(key, "key"), Path(_need(file, "file")), url)
        if action == "index":
            return _start(ws, "corpus", _corpus_argv(ws, "index"))
        if action == "fetch":
            return _start(ws, "corpus", _corpus_argv(ws, "fetch", *(ids or [])))
        raise ValueError(
            f"corpus: unknown action {action!r}, try search, get, list, add, index, fetch"
        )

    @server.tool
    def scenario(
        action: str,
        id: str | None = None,
        spec: dict[str, Any] | None = None,
    ) -> dict:
        """Stored scenarios. Actions: list, get, save, validate.

        ``save`` checks the contract and writes ``scenarios/<id>.json``; ``validate`` loads the
        macro and satellite versions the file names and names every problem that keeps the
        scenario from being computed, including corpus fragments not visible on the
        calculation day when the index exists.
        """
        if action == "list":
            return scenario_tool.catalogue(ws.scenarios)
        if action == "get":
            return scenario_tool.get(ws.scenarios, _need(id, "id"))
        if action == "save":
            return scenario_tool.save(ws.scenarios, _need(id, "id"), _need(spec, "spec"))
        if action == "validate":
            return scenario_tool.validate(
                ws.scenarios, _need(id, "id"), ws.models, _indexed(ws)
            )
        raise ValueError(f"scenario: unknown action {action!r}, try list, get, save, validate")

    @server.tool
    def run(
        action: str,
        id: str | None = None,
        scenario: str | None = None,
        portfolio: str | None = None,
        positions: list[dict[str, Any]] | None = None,
        draws: int | None = None,
        seed: int | None = None,
        base_currency: str = "RUB",
        targets: list[str] | None = None,
    ) -> dict:
        """Runs on the paths of a scenario. Actions: start, status, result, list.

        ``start`` draws the scenario's paths and calculates a portfolio's positions — or the
        ``positions`` given in the call — on them, as a job on the ``runs`` resource; ``status``
        reports that job and what the run has written, ``result`` the per-position results and
        the report of a portfolio run, ``list`` every run of the workspace. ``base_currency`` is
        the base of ``positions``; a portfolio brings its own.
        """
        if action == "start":
            if portfolio and positions:
                raise ValueError("run start: give portfolio or positions, not both")
            request = {
                "scenario": _need(scenario, "scenario"),
                "draws": _need(draws, "draws"),
                "seed": _need(seed, "seed"),
            }
            if portfolio:
                request["portfolio"] = portfolio
            else:
                request["positions"] = _need(positions, "positions")
                request["base_currency"] = base_currency
            if targets:
                request["targets"] = targets
            run_id = runs.new_id()
            runs.begin(ws, run_id, request)
            try:
                job = tasks.start(
                    ws, "runs", ["-m", "mst.runs", run_id, "--workspace", str(ws.root)]
                )
            except Exception as error:
                store.write_json(
                    runs.folder(ws, run_id) / "meta.json",
                    {"id": run_id, "status": "failed", "error": f"{type(error).__name__}: {error}"},
                )
                raise
            store.write_json(runs.folder(ws, run_id) / "job.json", {"job": job})
            return {"run": run_id, "job": job}
        if action == "status":
            run_id = _need(id, "id")
            link = store.read_json(runs.folder(ws, run_id) / "job.json") or {}
            return {
                "run": run_id,
                "meta": _meta(ws, run_id),
                "job": tasks.status(ws, link["job"]) if link.get("job") else None,
            }
        if action == "result":
            return runs.result(ws, _need(id, "id"))
        if action == "list":
            return runs.catalogue(ws)
        raise ValueError(f"run: unknown action {action!r}, try start, status, result, list")

    @server.tool
    def portfolio(
        action: str,
        id: str | None = None,
        spec: dict[str, Any] | None = None,
        scenario: str | None = None,
    ) -> dict:
        """Stored portfolios. Actions: list, get, save, delete, check, exposure.

        ``save`` checks the contract and writes ``portfolios/<id>.json``; ``check`` names every
        position the scenario's macro and satellite versions cannot serve, with the reason of a
        driver the out-of-sample check rejected; ``exposure`` values the positions at the
        calculation month of the scenario's macro version.
        """
        if action == "list":
            return portfolio_tool.catalogue(ws.portfolios)
        if action == "get":
            return portfolio_tool.get(ws.portfolios, _need(id, "id"))
        if action == "save":
            return portfolio_tool.save(ws.portfolios, _need(id, "id"), _need(spec, "spec"))
        if action == "delete":
            return portfolio_tool.delete(ws.portfolios, _need(id, "id"))
        if action == "check":
            return portfolio_tool.check(ws, _need(id, "id"), _need(scenario, "scenario"))
        if action == "exposure":
            return portfolio_tool.exposure(ws, _need(id, "id"), _need(scenario, "scenario"))
        raise ValueError(
            f"portfolio: unknown action {action!r}, try list, get, save, delete, check, exposure"
        )

    @server.tool
    def model(
        kind: str,
        action: str,
        version: str | None = None,
        as_of: str | None = None,
        only: list[str] | None = None,
        model_id: str | None = None,
        assignment: dict[str, str] | None = None,
    ) -> dict:
        """Model versions and plugins. kind: macro, satellites, catalog.

        The episode check of the models is ``backtest`` (``python -m mst.metrics``).
        Actions for ``macro``: list, describe, estimate, backtest; for ``satellites``: list,
        describe, estimate; for ``catalog``: list, the installed model plugins.
        ``model_id`` names the macro model to estimate or backtest — the only one installed
        when omitted; ``assignment`` maps a quantity to the satellite model that estimates
        it in an ``estimate`` of the satellites, overriding the model that declares it.
        ``version`` is the version hash ``describe`` reads and the macro version the
        satellites of an ``estimate`` are built on; ``list`` takes no version.
        """
        if kind == "catalog":
            if action != "list":
                raise ValueError("model: unknown action for catalog, try list")
            return models.catalog()
        if kind == "satellites":
            if action == "list":
                return {"versions": satellite_bundle.catalogue(ws.models)}
            if action == "describe":
                return satellite_bundle.describe(ws.models, _need(version, "version"))
            if action == "estimate":
                argv = _satellites_argv(ws, _need(version, "version"), assignment)
                return _start(ws, "models", argv)
            raise ValueError(
                f"model: unknown action {action!r} for satellites, try list, describe, estimate"
            )
        if kind != "macro":
            raise ValueError(f"model: unknown kind {kind!r}, try macro, satellites, catalog")
        if action == "list":
            return {"versions": macro_version.catalogue(ws.models)}
        if action == "describe":
            return macro_version.describe(ws.models, _need(version, "version"))
        if action == "estimate":
            return _start(
                ws,
                "models",
                _macro_argv(
                    ws,
                    "estimate-macro",
                    _need(as_of, "as_of"),
                    models.selected(model_id).model_id,
                ),
            )
        if action == "backtest":
            model = models.selected(model_id).model_id
            return _start(
                ws, "models", _backtest_argv(ws, _need(as_of, "as_of"), model, only or [])
            )
        raise ValueError(
            f"model: unknown action {action!r}, try list, describe, estimate, backtest"
        )

    return server


def _corpus_argv(ws: store.Workspace, command: str, *ids: str) -> list[str]:
    """The command line of ``python -m mst.corpus`` for this workspace."""
    return ["-m", "mst.corpus", command, *ids, "--corpus", str(ws.corpus)]


def _day(as_of: str) -> date:
    """The day an action works as of."""
    return date.fromisoformat(as_of)


def _indexed(ws: store.Workspace) -> Path | None:
    """The corpus root when its index exists, so a scenario's sources can be checked."""
    return ws.corpus if (ws.corpus / "index" / "chunks.jsonl").is_file() else None


def _macro_argv(
    ws: store.Workspace,
    command: str,
    as_of: str,
    model_id: str,
    only: list[str] | None = None,
) -> list[str]:
    """The command line of ``python -m mst.models`` for this workspace."""
    day = date.fromisoformat(as_of).isoformat()
    argv = [
        "-m",
        "mst.models",
        command,
        model_id,
        "--as-of",
        day,
        "--store",
        str(ws.series),
        "--models",
        str(ws.models),
    ]
    if command == "backtest":
        argv += ["--corpus", str(ws.corpus / "manifest.json")]
        if only:
            argv += ["--only", *only]
    return argv


def _backtest_argv(
    ws: store.Workspace, as_of: str, model_id: str, only: list[str] | None = None
) -> list[str]:
    """The command line of ``python -m mst.metrics backtest`` for this workspace."""
    day = date.fromisoformat(as_of).isoformat()
    argv = [
        "-m",
        "mst.metrics",
        "backtest",
        model_id,
        "--as-of",
        day,
        "--store",
        str(ws.series),
        "--models",
        str(ws.models),
        "--corpus",
        str(ws.corpus / "manifest.json"),
        "--drivers",
        str(ws.drivers / "catalog.json"),
    ]
    if only:
        argv += ["--only", *only]
    return argv


def _meta(ws: store.Workspace, run_id: str) -> dict | None:
    """The meta a run has written so far; a run that has not started has none."""
    try:
        return runs.meta(ws, run_id)
    except FileNotFoundError:
        return None


def _need(value: Any, name: str) -> Any:
    """The argument an action cannot work without."""
    if value is None or value == "":
        raise ValueError(f"{name} is required for this action")
    return value


def _satellites_argv(
    ws: store.Workspace, macro: str, assignment: dict[str, str] | None = None
) -> list[str]:
    """The command line of ``python -m mst.models estimate-satellites`` for this workspace."""
    argv = [
        "-m",
        "mst.models",
        "estimate-satellites",
        "--macro",
        macro,
        "--store",
        str(ws.series),
        "--models",
        str(ws.models),
        "--drivers",
        str(ws.drivers / "catalog.json"),
    ]
    for quantity, model_id in sorted((assignment or {}).items()):
        argv += ["--model", f"{quantity}={model_id}"]
    return argv


def _update_drivers(ws: store.Workspace, quantity: str) -> dict:
    """Discover the objects of ``quantity`` and fetch their series, as a task on series."""
    return _start(
        ws,
        "series",
        [
            "-m",
            "mst.drivers",
            "update",
            quantity,
            "--store",
            str(ws.series),
            "--drivers",
            str(ws.drivers / "catalog.json"),
        ],
    )


def _rows(ws: store.Workspace, series_id: str, as_of: str | None) -> dict:
    """The values of ``series_id`` known on ``as_of``, one row per month."""
    day = date.fromisoformat(as_of) if as_of else date.today()
    observations = SeriesStore(ws.series).as_of(series_id, day)
    return {
        "id": series_id,
        "as_of": day.isoformat(),
        "rows": [
            {
                "month": row.month.isoformat(),
                "value": row.value,
                "published": row.published.isoformat(),
                "approximate": row.approximate,
            }
            for row in observations
        ],
    }


def _update(ws: store.Workspace, ids: list[str]) -> dict:
    """Fetch the named series, or every one of them, as a task on the series resource."""
    return _start(ws, "series", ["-m", "mst.data", "update", *ids, "--workspace", str(ws.root)])


def _start(ws: store.Workspace, resource: str, argv: list[str]) -> dict:
    """Start ``argv`` as a task on ``resource`` and name it."""
    return {"job": tasks.start(ws, resource, argv), "argv": argv}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mst.mcp")
    parser.add_argument("--workspace", required=True, type=Path)
    ws = store.Workspace(parser.parse_args(argv).workspace)
    ws.root.mkdir(parents=True, exist_ok=True)
    store.write_state(ws)
    build(ws).run()


if __name__ == "__main__":
    main()
