# Documentation

[Русская версия](README.md)

The documentation answers three questions of a reader and keeps the accepted decisions;
each has its own folder.

- **[concept/](concept/)**: what the system does and why it is built this way: the general scheme, the method, the layout of the code.
- **[howto/](howto/)**: how to install and use it: installation, connection, the first session, a plugin.
- **[reference/](reference/)**: what corresponds to what: the platform contracts and the call reference.
- **[adr/](adr/)**: the accepted decisions with their context and consequences: why exactly this way.

| File | The question it answers |
|---|---|
| [concept/formalization.en.md](concept/formalization.en.md) | The general scheme: levels, the structural form of a model, a scenario of interventions and observations, the overlay, the conditional forecast, quality metrics |
| [concept/formalization-legacy.en.md](concept/formalization-legacy.en.md) | The previous edition of the general scheme, not the canon: a scenario with model versions and causes, coordinates of a model's outcome |
| [concept/bvar-method-legacy.en.md](concept/bvar-method-legacy.en.md) | Outdated description of the first BVAR version of the method, not the canon: details of estimating the BVAR and the satellites |
| [concept/architecture.en.md](concept/architecture.en.md) | How the scheme is implemented in code: packages and data flow, the workspace, model versions, jobs, the system prompt section |
| [howto/install.en.md](howto/install.en.md) | How to install the Python part, connect the plugin to a DSH profile and check that everything is visible |
| [howto/mcp.en.md](howto/mcp.en.md) | How to connect the server to any MCP client and resolve common errors |
| [howto/first-session.en.md](howto/first-session.en.md) | How to run a first session, from series to the report, with the tool calls |
| [howto/plugins.en.md](howto/plugins.en.md) | How to add a position model, a macro model or a satellite as a plugin: declaration, model, demo case, checks |
| [reference/tools.en.md](reference/tools.en.md) | Which MCP tools and `python -m mst.*` commands exist, what they accept and return |
| [reference/platform.en.md](reference/platform.en.md) | The calculation contracts: layers, position, paths, model, dispatcher, registry, run and report |
| [adr/0001-tool-schema-descriptions.en.md](adr/0001-tool-schema-descriptions.en.md) | Why every tool parameter is described and every tool annotated |
| [adr/0002-job-wait.en.md](adr/0002-job-wait.en.md) | Why the server holds a waiting call instead of the client polling |
| [adr/0003-plugin-stdio-isolation.en.md](adr/0003-plugin-stdio-isolation.en.md) | Why plugin printing goes to stderr at the Python level, not the descriptor |
| [adr/0004-atomic-writes-windows.en.md](adr/0004-atomic-writes-windows.en.md) | Why an atomic write retries when a Windows reader holds the file |
| [adr/0005-file-state-no-sqlite-arrow.en.md](adr/0005-file-state-no-sqlite-arrow.en.md) | Why state lives in files rather than SQLite and Arrow |

Reading order for a new reader: formalization → architecture → install → mcp → first-session → tools → platform → plugins.

Each package is described in the README of its folder (`src/mst/*/README.en.md`), the corpus in [../corpus/README.en.md](../corpus/README.en.md), the DSH plugin in [../dsh/README.en.md](../dsh/README.en.md).
