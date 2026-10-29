# MacroStress over MCP

MacroStress is a stdio MCP server. Any client that speaks the Model Context Protocol can run it:
Claude Desktop, Claude Code, Cursor, or the DeepSeek Harness plugin in `dsh/`. The tools are the
same everywhere; the DSH plugin adds skills and a prompt section on top.

The server owns one directory on disk, the workspace. Series, model versions, the corpus index,
scenarios, portfolios, and runs live there as plain files, so a run can be reproduced from the
workspace and the sources.

## Requirements

- Python 3.12 or newer.
- [uv](https://docs.astral.sh/uv/) to install and run the package, or a plain `pip install`.
- A free [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html) if you need the global
  series (`gl.brent`, `gl.fed_rate`, `gl.ust_10y`, `gl.vix`). Other sources work without a key.

## From the repository

```powershell
git clone https://github.com/LeoLourson/MacroStress-DSH
cd MacroStress-DSH
uv sync
```

The server is then one command away:

```powershell
uv run macrostress --workspace C:/data/macrostress
```

It speaks over stdio and prints nothing until a client connects. The workspace is created on the
first start together with `workspace.json`.

## Client configuration

The client starts the command itself. Pass an absolute path to the workspace, and put the FRED
key in `env` so the server passes it to download jobs.

### Claude Desktop

Edit `claude_desktop_config.json`:

- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "macrostress": {
      "command": "uv",
      "args": [
        "run",
        "--project", "C:/src/MacroStress-DSH",
        "macrostress",
        "--workspace", "C:/data/macrostress"
      ],
      "env": {
        "FRED_API_KEY": "your-key"
      }
    }
  }
}
```

### Cursor

Add the same block to `mcpServers` in `~/.cursor/mcp.json` or in `.cursor/mcp.json` inside a
project.

### Any client, without a checkout

`uvx` runs the package from PyPI:

```json
{
  "mcpServers": {
    "macrostress": {
      "command": "uvx",
      "args": ["macrostress", "--workspace", "C:/data/macrostress"]
    }
  }
}
```

The same works straight from the repository, without cloning or publishing:

```json
{
  "mcpServers": {
    "macrostress": {
      "command": "uvx",
      "args": [
        "--from", "git+https://github.com/LeoLourson/MacroStress-DSH",
        "macrostress",
        "--workspace", "C:/data/macrostress"
      ]
    }
  }
}
```

Use forward slashes in JSON paths on Windows; a backslash starts an escape sequence.

### DeepSeek Harness

The plugin in `dsh/` starts the same server, registers three skills, and adds the workspace state
to the agent prompt. The connection steps are in [install.md](install.md).

## First session

1. `jobs(action="state")` — what is in the workspace: model versions, corpus index, portfolios,
   scenarios, runs.
2. `series(action="update")` — download the macro series. This starts a job and returns its id.
3. `jobs(action="status", id=...)` and `jobs(action="log", id=...)` — wait for the job.
4. `model(kind="macro", action="estimate", as_of="2026-10-29")` — estimate the BVAR on the data
   known at that date. The version is a hash.
5. `series(action="update", quantity="share_price")` — find the objects of a driver quantity and
   download their series. The same for `bond_spread`, `overdue_share`,
   `housing_price_primary`, `housing_price_secondary`.
6. `model(kind="satellites", action="estimate", version=<macro hash>)` — estimate the driver
   models on that macro version.
7. `corpus(action="search", query=..., as_of=...)` — fragments of the corpus known at the date.
8. `scenario(action="save", id=..., spec={...})` — save a scenario, then `scenario(action="validate", ...)`.
9. `portfolio(action="save", id=..., spec={...})` — save a portfolio and check it against the
   scenario: `portfolio(action="check", id=..., scenario=...)`.
10. `run(action="start", scenario=..., portfolio=..., draws=..., seed=...)` and
    `run(action="result", id=...)` — the loss distribution, $P_5$, $P_{50}$, $P_{95}$, ES95,
    $\delta$, and coverage.

The full reference is in [tools.md](tools.md); the contracts behind the numbers are in
[platform.md](platform.md); the method is in
[scenario-formalization.md](scenario-formalization.md).

## What is where

- `workspace/series/` — one CSV per series, with a release date for every value.
- `workspace/models/` — model versions named by hash.
- `workspace/corpus/` — converted documents, episode cards, the index.
- `workspace/scenarios/`, `workspace/portfolios/`, `workspace/runs/` — files and results of a run.
- `workspace/workspace.json` — the snapshot the prompt section reads.

## Troubleshooting

- **No tools in the client.** The server did not start. Run the command from the client
  configuration by hand in a terminal: an import error or a wrong path shows up there.
- **`FRED needs an API key ...`** in a job log. Add `FRED_API_KEY` to the `env` block or to the
  system environment of the client.
- **Certificate error on `rosstat.gov.ru`.** Rosstat is signed by the Ministry of Digital
  Development root; the bundle in `src/mst/data/mintsifry-roots.pem` is used for that host only.
  If the root changed, update the file.
- **A job is stuck or the resource is busy.** `jobs(action="cancel", id=...)` releases the
  resource; a second update cannot start until the first finishes.
