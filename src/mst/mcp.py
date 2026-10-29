"""The MCP server: a tool parses its arguments and calls one function of the core."""

import argparse
from pathlib import Path

from fastmcp import FastMCP

from . import store


def build(ws: store.Workspace) -> FastMCP:
    """The server of one workspace; every tool closes over it."""
    server = FastMCP("macrostress")

    @server.tool
    def jobs(action: str) -> dict:
        """Workspace and task state. Actions: state (refresh workspace.json and return it)."""
        if action != "state":
            raise ValueError(f"jobs: unknown action {action!r}, try state")
        return store.write_state(ws)

    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mst.mcp")
    parser.add_argument("--workspace", required=True, type=Path)
    ws = store.Workspace(parser.parse_args(argv).workspace)
    ws.root.mkdir(parents=True, exist_ok=True)
    store.write_state(ws)
    build(ws).run()


if __name__ == "__main__":
    main()
