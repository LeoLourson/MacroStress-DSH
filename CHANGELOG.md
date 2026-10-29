# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-29

### Added

- Series store with release dates and loading from MOEX ISS, the Bank of Russia, Rosstat, and
  FRED.
- Macro model: BVAR with stochastic volatility, sign restrictions, conditional forecasts,
  scenario severity, and checks on historical episodes.
- Drivers: object catalogs, hierarchical satellites, out-of-sample checks, and driver paths on
  the same draws as the macro forecast.
- Corpus of sources: manifest with hashes, downloads, Markdown conversion, episode cards, index,
  and search that respects the publication date.
- Scenario file $(N, C, D, O)$ with validation and paths.
- Asset models for equities, bonds, credit and deposits, and housing.
- Portfolio checks and a run over scenario paths with $P_5$, $P_{50}$, $P_{95}$, ES95, and
  $\delta$ in the report.
- Asset types as plugins: `python -m mst.plugin new|check|demo-run`, contract tests, and a
  registry that keeps `mst.engine` free of asset-specific branches.
- MCP server with the tools `jobs`, `series`, `corpus`, `scenario`, `model`, `portfolio`, and
  `run`.
- DeepSeek Harness plugin: three skills, the prompt section built from `workspace.json`, and
  Russian and English locales.
