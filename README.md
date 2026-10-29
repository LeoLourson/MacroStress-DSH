# MacroStress

[English](README.md) | [Русский](README.ru.md) | [中文](README.zh.md)

[![CI](https://github.com/LeoLourson/MacroStress-DSH/actions/workflows/ci.yml/badge.svg)](https://github.com/LeoLourson/MacroStress-DSH/actions/workflows/ci.yml)
[![MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)

MacroStress answers one question: how much does a portfolio lose under a given macroeconomic
scenario? An analyst or an agent names paths for a few macro variables (say, the policy rate at
21 % for the next year) and the shocks behind them. A BVAR estimated on history completes
consistent paths for the other variables, satellite models turn those paths into drivers (equity
prices, bond spreads, industry overdue shares, housing prices), and every position is repriced
along the same draws. The output is a loss distribution with $P_5$, $P_{50}$, $P_{95}$, ES95 and
the scenario severity $\delta$. The method is described in
[docs/scenario-formalization.md](docs/scenario-formalization.md) (Russian).

The server speaks MCP, so any client can drive it: Claude Desktop, Cursor, Claude Code. A
DeepSeek Harness plugin in `dsh/` adds three skills and a workspace snapshot for the agent
prompt. All state lives in one `workspace/` directory; there is no web app, no database, and no
scheduler.

## What it does

| Part | What it does |
|---|---|
| Series | Monthly series from MOEX, the Bank of Russia, Rosstat, and FRED, with the release date of every value |
| Macro model | BVAR with stochastic volatility, sign restrictions, conditional forecasts, scenario severity, and checks on historical episodes |
| Drivers | Object catalogs, hierarchical satellite models, out-of-sample checks, driver paths on the macro draws |
| Scenario | A $(N, C, D, O)$ file with validation and paths: conditional forecast and drivers on the same draws |
| Positions | Equity, bond, credit or deposit, housing |
| Run and report | Positions along scenario paths, with $P_5$, $P_{50}$, $P_{95}$, ES95, and $\delta$ |
| Extensions | A new asset type is a plugin: `python -m mst.plugin new`, contract checks, a demo run |
| DSH plugin | Scenario from the corpus, report, drivers; the prompt section built from `workspace.json` |

MCP tools: `jobs`, `series`, `corpus`, `scenario`, `run`, `portfolio`, `model`. The reference is
in [docs/tools.md](docs/tools.md) (Russian).

## Install

Python 3.12 or newer and [uv](https://docs.astral.sh/uv/) are enough.

```bash
git clone https://github.com/LeoLourson/MacroStress-DSH
cd MacroStress-DSH
uv sync
uv run macrostress --workspace C:/data/macrostress
```

The server speaks over stdio and prints only errors; it waits for a client. Configuration for
Claude Desktop, Cursor, and any other MCP client is in [docs/mcp.md](docs/mcp.md). The DeepSeek
Harness plugin is covered in [docs/install.md](docs/install.md).

## First session

Once a client is connected, it sees seven tools. Long operations start jobs: a tool returns a
`job` at once, and `jobs` shows the progress.

1. `jobs(action="state")` — what is in the workspace: model versions, corpus index, jobs.
2. `series(action="update")` — download the macro series. FRED needs a key in `FRED_API_KEY`.
3. `jobs(action="status", id=...)` and `jobs(action="log", id=...)` — wait for the job.
4. `model(kind="macro", action="estimate", as_of="2026-10-29")` — estimate the macro model on the
   data known at that date. The version is a hash.
5. `series(action="update", quantity="share_price")` — find the objects of a quantity and
   download their series; the same for `bond_spread`, `overdue_share`, `housing_price_primary`,
   `housing_price_secondary`.
6. `model(kind="satellites", action="estimate", version=<macro hash>)` — estimate the satellites
   on that macro version.
7. `model(kind="satellites", action="describe", version=<hash>)` — which drivers passed the
   out-of-sample check and why the rest were rejected.
8. `corpus(action="search", query=..., as_of=...)` — fragments known at the date;
   `corpus(action="index")` rebuilds the index as a job.
9. `scenario(action="save", id=..., spec={...})` — save a scenario;
   `scenario(action="validate", id=...)` — list everything that blocks the calculation.
10. `portfolio(action="save", id=..., spec={...})` — save a portfolio;
    `portfolio(action="check", id=..., scenario=...)` — check it against the scenario.
11. `run(action="start", scenario=..., portfolio=..., draws=..., seed=...)` — start a run;
    `run(action="result", id=...)` — read $P_5$/$P_{50}$/$P_{95}$, $\delta$, and coverage.

## Documentation

The docs are in Russian for now; [docs/mcp.md](docs/mcp.md) is in English.

- [docs/mcp.md](docs/mcp.md) — any MCP client, the first session, troubleshooting;
- [docs/install.md](docs/install.md) — the Python package and the DeepSeek Harness plugin;
- [docs/architecture.md](docs/architecture.md) — packages, workspace, versions, jobs;
- [docs/tools.md](docs/tools.md) — MCP tools and `python -m mst.*` commands;
- [docs/plugins.md](docs/plugins.md) — a new asset type as a plugin;
- [docs/platform.md](docs/platform.md) — the contracts: position, paths, model, registry;
- [docs/scenario-formalization.md](docs/scenario-formalization.md) — the method and what the code
  implements.

Every package folder has its own README that names its files.

## Development

```bash
uv sync
uv run pytest -q -m "not slow"
uv run ruff check .
```

The `slow` tests estimate the BVAR and take minutes; the night workflow runs the full suite.
CI runs the fast suite on Linux and Windows with Python 3.12 and 3.13.

## Status

0.1.0, beta. Interfaces may change before 1.0. The data sources are the Russian market (MOEX,
the Bank of Russia, Rosstat) plus FRED for global series.

## Contributing and license

Contributing rules and the release process are in [CONTRIBUTING.md](CONTRIBUTING.md);
vulnerabilities go through [SECURITY.md](SECURITY.md). The code is MIT-licensed, see
[LICENSE](LICENSE). Series from MOEX, the Bank of Russia, Rosstat, FRED, and BIS are not part of
this repository: you load them under the terms of each source, and the license does not cover
that data. Citations are described in [CITATION.cff](CITATION.cff).
