# Installation

The plugin has two parts. The Python package `mst` in `src/` is the MCP server and all the calculations. The folder `dsh/` is the DSH plugin package: it starts this server, registers the skills and adds a snapshot of the workspace to the agent's system prompt.

## 1. The Python part

You need Python 3.12 or newer and [uv](https://docs.astral.sh/uv/); `requires-python` in `pyproject.toml` fixes the minimum Python version.

```powershell
cd <path to the repository>
uv sync
.venv\Scripts\python.exe -m pytest -q -m "not slow"
```

`uv sync` creates `.venv` and installs the package `mst` there with the dependencies from `uv.lock`. To check that the server starts:

```powershell
uv run macrostress --workspace workspace
```

The server speaks over stdio and waits for a client, so it prints nothing after the start; Ctrl+C exits. On start it creates `workspace/` and writes `workspace.json` there.

## 2. Paths in `dsh/cordis.patch.yml`

The patch holds two absolute paths: the interpreter from this repository's `.venv` and the data directory.

```yaml
command: 'C:\Users\LeoLeon\Documents\MacroStress\macrostress-dsh\.venv\Scripts\python.exe'
args:
  - ...
  - '--workspace'
  - &workspace 'C:\Users\LeoLeon\Documents\MacroStress\macrostress-dsh\workspace'
...
- id: dsh-macrostress
  config:
    workspace: *workspace
```

On another machine replace both paths with your own. The data directory can live anywhere. The patch writes it once, in the anchor `&workspace`, and both the server and the plugin (`dsh/index.js`) take it from there, so the prompt and the tools always describe one directory.

## 3. FRED key

The global series (`gl.brent`, `gl.fed_rate`, `gl.ust_10y`, `gl.vix`) come from FRED and need a free key in the environment variable `FRED_API_KEY`. The other sources work without a key. Download jobs inherit the server's environment, so set the key in the Windows system variables or in the `env` field of the `mcp-macrostress` entry of the patch.

## 4. Connecting to a DSH profile

A DSH profile is the directory `~/.dsh/profiles/<name>/`. The plugin connects to it in these steps.

1. A link to the folder `dsh/` from the DSH plugin directory (a junction on Windows):

   ```powershell
   New-Item -ItemType Junction -Path "$HOME\.dsh\plugins\dsh-macrostress" `
       -Target "<path to the repository>\dsh"
   ```

2. A dependency and a bundle in `~/.dsh/profiles/web/package.json`:

   ```json
   {
     "dependencies": {
       "dsh-macrostress": "link:../../plugins/dsh-macrostress"
     },
     "dsh": {
       "profile": {
         "bundles": ["@deepseek-ai/dsh-base", "...", "dsh-macrostress"]
       }
     }
   }
   ```

   The name in `bundles` is the `name` field of `dsh/package.json`. The bundle attaches `dsh/cordis.patch.yml` (the field `dsh.bundle.patch`), and the patch inserts into the profile tree the MCP client `@deepseek-ai/dsh-mcp-client` with the server `macrostress` and the plugin itself.

3. A link `~/.dsh/profiles/web/node_modules/dsh-macrostress` → `~/.dsh/plugins/dsh-macrostress`. If it is missing after a restart, create it the same way as in step 1.

4. Restart DSH with this profile.

The machine-wide patch `~/.dsh/cordis.patch.yml` needs no change: a plugin connected by a link is described in the profile's manifest and patch.

## 5. Check

In a new DSH session:

- the system prompt has a MacroStress section of five lines with the workspace path and the model versions (dashes before the first estimate); the fourth line gives the number of portfolios, scenarios, runs and jobs in progress, the fifth says where to start;
- the skill list has three plugin skills: `scenario-from-corpus`, `run-report`, `drivers`;
- the plugin card (Settings → Plugins) shows the name and description in the interface language;
- the call `jobs(action="state")` returns a dictionary with the keys `workspace`, `macro`, `satellites`, `corpus_index`, `portfolios`, `scenarios`, `runs`, `jobs`.

A check without starting DSH:

```powershell
node dsh/test/smoke.mjs
```

It supplies a fake context, registers the skills, builds the five lines of the prompt section from a temporary `workspace.json`, and checks the one-line fallback without the file, the rereading of the file and the removal of the skills on shutdown; the card is checked in both locales.

If the tools are missing, the server probably did not start. The patch sets `failOnStartupError: false`, so DSH does not fail in that case. Run the command from `command` and `args` of the patch by hand in a terminal: an import error or a wrong path shows at once.

## Common errors

| Symptom | Cause |
|---|---|
| `resource 'series' is busy with job ...` | A job on this resource is still running; wait for it or cancel it with `jobs(action="cancel", id=...)` |
| `FRED needs an API key ...` in the download log | `FRED_API_KEY` is missing from the server's environment |
| Certificate error on rosstat.gov.ru | Rosstat is signed by the root of the Russian Ministry of Digital Development; the roots are in `src/mst/data/mintsifry-roots.pem` and apply only to this host. If the ministry has changed the certificate, the file needs an update |
| Satellite estimation takes hours | The command was not started through `python -m mst.drivers`, and BLAS runs in many threads; see [../reference/tools.en.md](../reference/tools.en.md) |
