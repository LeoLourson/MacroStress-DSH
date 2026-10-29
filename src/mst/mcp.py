"""The MCP server: a tool parses its arguments and calls one function of the core."""

import argparse
from datetime import date
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from . import jobs as tasks
from . import store
from .data import sources
from .data.store import SeriesStore
from .drivers import version as satellite_version
from .macro import version as macro_version


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
            return {"sources": sources.search(_need(text, "text"))}
        if action == "get":
            return _rows(ws, _need(id, "id"), as_of)
        if action == "add":
            return {"id": sources.add(_need(entry, "entry"))}
        if action == "update":
            if quantity:
                if ids:
                    raise ValueError("series update: give ids or a quantity, not both")
                return _update_drivers(ws, quantity)
            return _update(ws, ids or [])
        raise ValueError(f"series: unknown action {action!r}, try search, get, add, update")

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
