# dsh

The DSH plugin package `dsh-macrostress`. It connects the MCP server `macrostress`, registers the skills and adds a section on the workspace state to the agent's system prompt. There are no calculations here: the Python package `mst` from [../src](../src/README.en.md) does them. How to connect the plugin to a profile is described in [../docs/howto/install.en.md](../docs/howto/install.en.md).

| File | What it does |
|---|---|
| `package.json` | The package manifest; `dsh.bundle.patch` points DSH to `cordis.patch.yml` |
| `cordis.patch.yml` | A patch of the profile tree: it inserts `@deepseek-ai/dsh-mcp-client` with the server `python -m mst.mcp --workspace ...` and the plugin itself. Its paths are absolute and must be replaced on another machine |
| `index.js` | The plugin: on start it registers the skills from `skills/*.md` and the prompt section `macrostress` of five lines from `workspace.json`; on shutdown it removes the skills |
| `locale/` | The plugin's name and description in Russian and English |
| `skills/` | Three agent skills: `scenario-from-corpus`, `run-report`, `drivers` |
| `test/smoke.mjs` | A check without DSH: a fake context, the skills, five lines from a temporary `workspace.json`, the one-line fallback, the rereading of the file and the removal of the skills |

The workspace path is written once in the patch, in the anchor `&workspace`: both the server (`--workspace` in `args`) and `index.js` (the config parameter `workspace`) take it from there. Without a config the plugin takes `<repository>/workspace`.

A check of the plugin without starting DSH:

```powershell
node dsh/test/smoke.mjs
```

It calls `apply` with a fake context and checks the three skills, the card in both locales, the five section lines from a temporary `workspace.json`, the one-line fallback, the rereading of the file and the removal of the skills on shutdown. After a DSH restart the skills are visible in the list, the section in the system prompt, and the plugin card in Settings → Plugins.
