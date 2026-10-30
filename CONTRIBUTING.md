# Contributing

Issues and pull requests are welcome. Small fixes can go straight to a pull request; for a new
driver, asset type, or data source, open an issue first so the contract stays coherent.

## Setup

Python 3.12 or newer and [uv](https://docs.astral.sh/uv/) are enough:

```powershell
uv sync
uv run pytest -q -m "not slow"
uv run ruff check .
```

`uv sync` creates `.venv` and installs the `mst` package in editable mode. The full suite adds
the BVAR estimation tests, which take minutes:

```powershell
uv run pytest -q
```

Those tests are marked `slow`. The daily CI runs everything except them; the nightly workflow
runs the full set.

## Layout

`src/mst/` holds the package: `data` (sources and series store), `macro` (BVAR), `drivers`
(satellites), `corpus`, `scenario`, `portfolio`, `engine`, `plugins`, `plugin_sdk`, and `mcp`
(the server). `dsh/` is the DeepSeek Harness plugin around the same server. `docs/` explains the
contracts; each package folder has its own README.

## Style

- `ruff` with the rules in `pyproject.toml`, line length 100.
- Type hints on public functions; dataclasses or pydantic models for structures that cross a
  package boundary.
- No new dependency unless the standard library and the current ones cannot do the job; explain
  the reason in the pull request.
- Tests go next to the package they cover and never reach the network: sources are mocked, models
  are checked on data generated from known parameters.
- A new asset type belongs in a plugin, not in `mst.engine`. See [docs/howto/plugins.md](docs/howto/plugins.md).
- Russian is the language of the docs and comments; code, tool names, and commit messages are in
  English.

## Commits and pull requests

Write the commit subject in the imperative and explain the reason in the body when it is not
obvious. Keep one topic per commit: a fix, a feature, and a docs update are three commits.

Before opening a pull request, run the two fast checks from Setup and confirm that CI stays
green. A pull request that changes a tool or a contract updates `docs/` and adds an entry under
`Unreleased` in `CHANGELOG.md`.

## Releasing

1. Move the `Unreleased` entries in `CHANGELOG.md` under the new version and set the date.
2. Bump `version` in `pyproject.toml` and `dsh/package.json`.
3. Commit, tag `vX.Y.Z`, and push the tag. The `Publish` workflow builds the package and sends it
   to PyPI through trusted publishing, so no token lives in the repository.
4. Publish a GitHub release from the tag, with the changelog section as the body.

The PyPI project is `macrostress`; the trusted publisher is configured on PyPI for this
repository, the `publish.yml` workflow, and the `pypi` environment.

## License

Contributions are accepted under the MIT license in [LICENSE](LICENSE).
