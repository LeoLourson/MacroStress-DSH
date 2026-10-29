# Security policy

## Supported versions

The latest release receives fixes. Versions before 0.1.0 are not supported.

| Version | Supported |
|---|---|
| 0.1.x | yes |

## Reporting a vulnerability

Use GitHub private reporting: **Security** tab → **Report a vulnerability**. Do not open a public
issue for an unfixed vulnerability.

Include the version, the platform, a minimal reproduction, and what an attacker gains. You will
get an answer within a week; a fix and a release follow when the report is valid.

## Scope

MacroStress runs locally and talks to data sources over HTTPS. Its attack surface:

- **Untrusted documents.** The corpus loader parses PDFs and Excel files from URLs named in
  `corpus/manifest.json`, and pass-through fetching follows links on those pages. A malicious
  manifest or a compromised source can feed the parser hostile input.
- **Series sources.** Responses from MOEX, the Bank of Russia, Rosstat, and FRED are parsed into
  rows; malformed or oversized responses are handled as errors, not as code.
- **Secrets.** `FRED_API_KEY` and other credentials live in the environment. They are never
  written to `workspace/`, logs, or the prompt section built from `workspace.json`.
- **MCP clients.** The server speaks over stdio and trusts its client; it is not exposed on a
  network port.

Out of scope: vulnerabilities in the upstream data providers, in MCP clients, and in the
DeepSeek Harness itself; reports about a scenario producing different numbers than you expect
(open a regular issue with the run id).
