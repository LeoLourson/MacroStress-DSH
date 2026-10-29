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
    ) -> dict:
        """Stored series. Actions: search, get, add, update."""
        if action == "search":
            return {"sources": sources.search(_need(text, "text"))}
        if action == "get":
            return _rows(ws, _need(id, "id"), as_of)
        if action == "add":
            return {"id": sources.add(_need(entry, "entry"))}
        if action == "update":
            return _update(ws, ids or [])
        raise ValueError(f"series: unknown action {action!r}, try search, get, add, update")

    return server


def _need(value: Any, name: str) -> Any:
    """The argument an action cannot work without."""
    if value is None or value == "":
        raise ValueError(f"{name} is required for this action")
    return value


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
    argv = ["-m", "mst.data", "update", *ids, "--workspace", str(ws.root)]
    return {"job": tasks.start(ws, "series", argv), "argv": argv}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mst.mcp")
    parser.add_argument("--workspace", required=True, type=Path)
    ws = store.Workspace(parser.parse_args(argv).workspace)
    ws.root.mkdir(parents=True, exist_ok=True)
    store.write_state(ws)
    build(ws).run()


if __name__ == "__main__":
    main()
