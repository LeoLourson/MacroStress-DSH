"""The MCP server: a tool parses its arguments and calls one function of the core."""

import argparse
from datetime import date
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from . import jobs as tasks
from . import store
from .corpus import tool as corpus_tool
from .data import sources
from .data.store import SeriesStore
from .drivers import version as satellite_version
from .macro import version as macro_version
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
    def model(
        kind: str,
        action: str,
        version: str | None = None,
        as_of: str | None = None,
        only: list[str] | None = None,
    ) -> dict:
        """Model versions. kind: macro, satellites. Actions: list, describe, estimate, backtest.

        ``version`` names the version: the macro one to describe or backtest, or, for
        ``satellites``, the macro version whose panel the satellites are estimated on.
        """
        if kind == "satellites":
            if action == "list":
                return {"versions": satellite_version.catalogue(ws.models)}
            if action == "describe":
                return satellite_version.describe(ws.models, _need(version, "version"))
            if action == "estimate":
                return _start(ws, "models", _satellites_argv(ws, _need(version, "version")))
            raise ValueError(
                f"model: unknown action {action!r} for satellites, try list, describe, estimate"
            )
        if kind != "macro":
            raise ValueError(f"model: unknown kind {kind!r}, try macro, satellites")
        if action == "list":
            return {"versions": macro_version.catalogue(ws.models)}
        if action == "describe":
            return macro_version.describe(ws.models, _need(version, "version"))
        if action == "estimate":
            return _start(ws, "models", _macro_argv(ws, "estimate", _need(as_of, "as_of")))
        if action == "backtest":
            return _start(
                ws, "models", _macro_argv(ws, "backtest", _need(as_of, "as_of"), only or [])
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
    ws: store.Workspace, command: str, as_of: str, only: list[str] | None = None
) -> list[str]:
    """The command line of ``python -m mst.macro`` for this workspace."""
    day = date.fromisoformat(as_of).isoformat()
    argv = [
        "-m",
        "mst.macro",
        command,
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


def _need(value: Any, name: str) -> Any:
    """The argument an action cannot work without."""
    if value is None or value == "":
        raise ValueError(f"{name} is required for this action")
    return value


def _satellites_argv(ws: store.Workspace, macro: str) -> list[str]:
    """The command line of ``python -m mst.drivers estimate`` for this workspace."""
    return [
        "-m",
        "mst.drivers",
        "estimate",
        "--macro",
        macro,
        "--store",
        str(ws.series),
        "--models",
        str(ws.models),
        "--drivers",
        str(ws.drivers / "catalog.json"),
    ]


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
