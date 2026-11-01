# MacroStress over MCP

MacroStress is a stdio MCP server. Any client that speaks the Model Context Protocol can connect to it: Claude Desktop, Claude Code, Cursor or the DeepSeek Harness plugin in `dsh/`. The tools are the same everywhere; the DSH plugin adds skills and a prompt section on top. The general scheme of the system is in [../concept/formalization.en.md](../concept/formalization.en.md).

The server owns one directory on disk, the workspace. Series, model versions, the corpus index, scenarios, portfolios and runs live there as plain files, so a run can be reproduced from the workspace and the sources.

## Requirements

- Python 3.12 or newer.
- [uv](https://docs.astral.sh/uv/) to install and run the package, or plain `pip install`.
- A free [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html) if you need the global series (`gl.brent`, `gl.fed_rate`, `gl.ust_10y`, `gl.vix`). The other sources work without a key.

## From the repository

```powershell
git clone https://github.com/LeoLourson/MacroStress-DSH
cd MacroStress-DSH
uv sync
```

The server is then one command:

```powershell
uv run macrostress --workspace C:/data/macrostress
```

The server speaks over stdio and prints nothing until a client connects. The workspace is created on the first start together with `workspace.json`.

## Client configuration

The client starts the command itself. Without `--workspace` the server takes `MACROSTRESS_WORKSPACE`, else `~/.macrostress/workspace`. Pass the absolute path to the workspace and put the FRED key in `env`, so the server hands it to the download jobs.

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

Add the same block to `mcpServers` in `~/.cursor/mcp.json` or in `.cursor/mcp.json` inside a project.

### Any client, without cloning the repository

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

The same works directly from the repository:

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

In JSON paths on Windows use forward slashes, because a backslash starts an escape sequence.

### DeepSeek Harness

The plugin in `dsh/` starts the same server, registers three skills and adds the workspace state to the agent's prompt. The connection steps are in [install.en.md](install.en.md).

## First session

A session follows the scheme: prepare the history from series, estimate the models, write a scenario, calculate.

1. `series(action="update")`: the history series.
2. `model(kind="macro", action="estimate", as_of=...)`: the macro model, level 0.
3. `series(action="update", quantity=...)` and `model(kind="satellites", action="estimate", version=...)`: the satellites, the levels above.
4. `scenario(action="save")` and `scenario(action="validate")`: the scenario.
5. `run(action="start")` and `run(action="result")`: the calculation: the conditional forecast of the target indicators and the portfolio report.

A step-by-step session with full calls is in [first-session.en.md](first-session.en.md). The full reference is [../reference/tools.en.md](../reference/tools.en.md), the number contracts are in [../reference/platform.en.md](../reference/platform.en.md), the method in [../concept/scenario-formalization.en.md](../concept/scenario-formalization.en.md).

## What lives where

- `workspace/series/`: one CSV per series; every value carries its publication date.
- `workspace/models/`: model versions named by hash.
- `workspace/corpus/`: texts in Markdown, episode cards, the index.
- `workspace/scenarios/`, `workspace/portfolios/`, `workspace/runs/`: files and run results.
- `workspace/workspace.json`: the snapshot the prompt section reads.

## Common errors

- **The client has no tools.** The server did not start. Run the command from the client configuration by hand in a terminal: an import error or a wrong path shows at once.
- **`FRED needs an API key ...`** in the job log. Add `FRED_API_KEY` to the `env` block or to the client's system environment.
- **Certificate error on `rosstat.gov.ru`.** Rosstat is signed by the root of the Russian Ministry of Digital Development; the bundle in `src/mst/data/mintsifry-roots.pem` applies only to this host. If the ministry has changed the certificate, the file needs an update.
- **A job hangs or a resource is busy.** `jobs(action="cancel", id=...)` frees the resource; a second update does not start until the first has ended.
