"""The MCP server: a tool parses its arguments and calls one function of the core."""

import argparse
from pathlib import Path

from fastmcp import FastMCP

from . import jobs as tasks
from . import store


def build(ws: store.Workspace) -> FastMCP:
    """The server of one workspace; every tool closes over it."""
    server = FastMCP("macrostress")

    @server.tool
    def jobs(action: str, id: str | None = None, tail: int = 50) -> dict:
        """Workspace and task state. Actions: state, status, log, cancel."""
        if action == "state":
            return tasks.state(ws)
        if action == "status":
            return tasks.status(ws, _required(id, action))
        if action == "log":
            job_id = _required(id, action)
            return {"id": job_id, "log": tasks.log(ws, job_id, tail)}
        if action == "cancel":
            return tasks.cancel(ws, _required(id, action))
        raise ValueError(f"jobs: unknown action {action!r}, try state, status, log, cancel")

    return server


def _required(value: str | None, action: str) -> str:
    """The argument an action cannot work without."""
    if not value:
        raise ValueError(f"jobs: action {action!r} needs id")
    return value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mst.mcp")
    parser.add_argument("--workspace", required=True, type=Path)
    ws = store.Workspace(parser.parse_args(argv).workspace)
    ws.root.mkdir(parents=True, exist_ok=True)
    store.write_state(ws)
    build(ws).run()


if __name__ == "__main__":
    main()
