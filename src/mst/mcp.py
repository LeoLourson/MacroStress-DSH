"""The MCP server: a tool parses its arguments and calls one function of the core."""

import argparse
import asyncio
import os
from datetime import date
from pathlib import Path
from typing import Annotated, Any

from fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from . import guides, models, runs, store
from . import jobs as tasks
from .corpus import tool as corpus_tool
from .data import sources
from .data.store import SeriesStore
from .drivers import bundle as satellite_bundle
from .drivers.objects import series_id as driver_series
from .macro import version as macro_version
from .portfolio import tool as portfolio_tool
from .scenario import tool as scenario_tool

INSTRUCTIONS = """\
MacroStress answers one question: how the target indicators of an object are distributed if a \
given event happens. A scenario s = (N, m, O, D, C) is built from the text of the event and the \
corpus; the conditions extend the description of the event, and the models compute its \
consequences.

Read a guide before the work it covers: guide(name="scenario-from-corpus") when the user names \
an event or asks what happens if it occurs, even without the words scenario or corpus; \
guide(name="run-report") before a calculation or its report; guide(name="drivers") before work \
on drivers and satellites. guide() lists them.

Search the corpus before writing any number of a scenario. Mark a condition or an impact with \
"event" when the text of the event states its number, with a corpus fragment only when the \
fragment gives its number, and with "assumption" otherwise. When \
neither the event nor the corpus gives a number, propose one from the nearest episode and mark \
it "assumption": a scenario with no conditions and no impacts records no event, and questions \
to the user come after the scenario, not instead of it. A human approves a scenario: never \
fill approved_by yourself and never start a calculation that was not asked for.

A tool that starts a job returns its id at once. Wait with jobs(action="wait", id=<job>): the \
server holds the call until the job ends, and jobs(action="log", id=<job>) shows what it \
printed.\
"""

WAIT_STEP = 0.2
"""Seconds between the file reads of a ``wait``, the only cost of waiting server-side."""


def build(ws: store.Workspace) -> FastMCP:
    """The server of one workspace; every tool closes over it."""
    server = FastMCP("macrostress", instructions=INSTRUCTIONS)

    @server.tool(
        annotations=ToolAnnotations(
            title="Working guides",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        )
    )
    def guide(
        name: Annotated[
            str | None,
            Field(
                description="The guide to read: scenario-from-corpus, run-report or drivers. "
                "The list when omitted."
            ),
        ] = None,
    ) -> dict:
        """The working guides of the agent: how to do the work, whatever the client.

        Without ``name``, the list. ``scenario-from-corpus`` turns an event into a scenario;
        read it before ``scenario save``.
        """
        return guides.get(name) if name else {"guides": guides.catalogue()}

    @server.prompt
    def scenario_from_event(event: str) -> str:
        """Build a scenario from the text of an event, by the corpus, for a human to approve."""
        return f"{guides.get('scenario-from-corpus')['content']}\n\n## Event\n\n{event}"

    @server.tool(
        annotations=ToolAnnotations(
            title="Workspace and job state",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=False,
        )
    )
    async def jobs(
        action: Annotated[
            str,
            Field(
                description="What to do with the workspace or one job: state, status, log, "
                "cancel or wait. wait holds the call until the job ends or timeout_sec "
                "passes; cancel stops the job's process tree and frees its resource."
            ),
        ],
        id: Annotated[
            str | None,
            Field(
                description="The job id that status, log, cancel and wait read: the id a "
                "starting tool returned."
            ),
        ] = None,
        tail: Annotated[
            int,
            Field(
                description="How many lines of the log to return, counted from the end; "
                "the whole log when 0."
            ),
        ] = 50,
        timeout_sec: Annotated[
            float,
            Field(
                ge=0.0,
                le=120.0,
                description="How long wait holds the call, in seconds. The job's record "
                "comes back when the time is out; call wait again to keep waiting.",
            ),
        ] = 30.0,
    ) -> dict:
        """Workspace and task state. Actions: state, status, log, cancel, wait.

        ``wait`` polls the job server-side, so one call replaces the loop of ``status`` and
        ``log`` calls an agent would otherwise run while a long job works; it returns the
        record ``status`` returns, and ``running`` says whether to call it again.
        """
        if action == "state":
            return tasks.state(ws)
        if action == "status":
            return tasks.status(ws, _need(id, "id"))
        if action == "log":
            job_id = _need(id, "id")
            return {"id": job_id, "log": tasks.log(ws, job_id, tail)}
        if action == "cancel":
            return tasks.cancel(ws, _need(id, "id"))
        if action == "wait":
            return await _wait_for_job(ws, _need(id, "id"), timeout_sec)
        raise ValueError(
            f"jobs: unknown action {action!r}, try state, status, log, cancel, wait"
        )

    @server.tool(
        annotations=ToolAnnotations(
            title="Stored series",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=True,
        )
    )
    def series(
        action: Annotated[
            str,
            Field(
                description="What to do: search for series, get one as known on a date, add "
                "a source, or update (fetch) as a job."
            ),
        ],
        id: Annotated[
            str | None,
            Field(
                description="The series id that get reads: a macro variable like "
                "ru.key_rate, or a driver like share_price:sibn."
            ),
        ] = None,
        as_of: Annotated[
            str | None,
            Field(
                description="The date get works as of, YYYY-MM-DD: only values published "
                "by then are returned. Today when omitted."
            ),
        ] = None,
        text: Annotated[
            str | None,
            Field(
                description="The words search looks for, in the ids and loaders of the "
                "known sources."
            ),
        ] = None,
        entry: Annotated[
            dict[str, Any] | None,
            Field(
                description="The source entry add writes to the workspace sources.toml: "
                "the table of one source plus its id."
            ),
        ] = None,
        ids: Annotated[
            list[str] | None,
            Field(
                description="The series ids update fetches; every known series when "
                "omitted. FRED needs the FRED_API_KEY environment variable."
            ),
        ] = None,
        quantity: Annotated[
            str | None,
            Field(
                description="A driver quantity, e.g. share_price: update discovers its "
                "objects and fetches their series instead of ids."
            ),
        ] = None,
    ) -> dict:
        """Stored series. Actions: search, get, add, update.

        ``update`` fetches the named ``ids``; with ``quantity`` it instead discovers the
        objects of that driver quantity and fetches their series.
        """
        if action == "search":
            return {"sources": sources.search(_need(text, "text"), extra=ws.sources)}
        if action == "get":
            return _rows(ws, _need(id, "id"), as_of)
        if action == "add":
            return {"id": sources.add(_need(entry, "entry"), ws.sources)}
        if action == "update":
            if quantity:
                if ids:
                    raise ValueError("series update: give ids or a quantity, not both")
                return _update_drivers(ws, quantity)
            return _update(ws, ids or [])
        raise ValueError(f"series: unknown action {action!r}, try search, get, add, update")

    @server.tool(
        annotations=ToolAnnotations(
            title="Scenario corpus",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=True,
        )
    )
    def corpus(
        action: Annotated[
            str,
            Field(
                description="What to do: search the index, get one fragment, list the "
                "manifest, add a file, or index and fetch as jobs."
            ),
        ],
        query: Annotated[
            str | None,
            Field(description="The words search looks for in the fragments of the corpus."),
        ] = None,
        as_of: Annotated[
            str | None,
            Field(
                description="The date search and get work as of, YYYY-MM-DD: only "
                "fragments visible on that date."
            ),
        ] = None,
        id: Annotated[
            str | None,
            Field(
                description="The fragment id that get returns, as search reported it."
            ),
        ] = None,
        kind: Annotated[
            str | None,
            Field(
                description="The kind of document search limits itself to, e.g. a report "
                "or a speech."
            ),
        ] = None,
        region: Annotated[
            str | None,
            Field(description="The region search limits itself to."),
        ] = None,
        factors: Annotated[
            list[str] | None,
            Field(description="The factors search limits itself to."),
        ] = None,
        k: Annotated[
            int,
            Field(description="How many fragments search returns, best first."),
        ] = 8,
        key: Annotated[
            str | None,
            Field(
                description="The manifest key add registers the file under."
            ),
        ] = None,
        file: Annotated[
            str | None,
            Field(description="The path of the downloaded file add registers."),
        ] = None,
        url: Annotated[
            str | None,
            Field(
                description="The address the file was downloaded from, which add records."
            ),
        ] = None,
        ids: Annotated[
            list[str] | None,
            Field(
                description="The documents fetch downloads, by manifest key; every "
                "missing one when omitted."
            ),
        ] = None,
    ) -> dict:
        """The scenario corpus. Actions: search, get, list, add, index, fetch.

        ``search`` and ``get`` read the index as it was on ``as_of``; ``list`` shows what the
        manifest holds; ``add`` registers a file downloaded by hand under its ``key``. ``index``
        and ``fetch`` are tasks on the corpus resource.
        """
        if action == "search":
            return corpus_tool.search(
                ws.corpus,
                _need(query, "query"),
                _day(_need(as_of, "as_of")),
                kind=kind,
                region=region,
                factors=factors,
                k=k,
            )
        if action == "get":
            return corpus_tool.get(ws.corpus, _need(id, "id"), _day(_need(as_of, "as_of")))
        if action == "list":
            return corpus_tool.catalogue(ws.corpus)
        if action == "add":
            return corpus_tool.add(ws.corpus, _need(key, "key"), Path(_need(file, "file")), url)
        if action == "index":
            return _start(ws, "corpus", _corpus_argv(ws, "index"))
        if action == "fetch":
            return _start(ws, "corpus", _corpus_argv(ws, "fetch", *(ids or [])))
        raise ValueError(
            f"corpus: unknown action {action!r}, try search, get, list, add, index, fetch"
        )

    @server.tool(
        annotations=ToolAnnotations(
            title="Stored scenarios",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=False,
        )
    )
    def scenario(
        action: Annotated[
            str,
            Field(
                description="What to do: list, get, save, validate or reverse."
            ),
        ],
        id: Annotated[
            str | None,
            Field(
                description="The scenario id, which is also the name of its file: "
                "lowercase letters, digits and dashes."
            ),
        ] = None,
        spec: Annotated[
            dict[str, Any] | None,
            Field(
                description="The scenario specification save checks against the contract "
                "before it writes the file."
            ),
        ] = None,
        portfolio: Annotated[
            str | None,
            Field(
                description="The portfolio whose losses the reverse search measures."
            ),
        ] = None,
        kappa: Annotated[
            float | None,
            Field(
                description="The price the reverse search fixes, for the worst loss "
                "under it."
            ),
        ] = None,
        level: Annotated[
            float | None,
            Field(
                description="The loss level the reverse search finds the lowest price "
                "for."
            ),
        ] = None,
        draws: Annotated[
            int | None,
            Field(
                description="How many paths the reverse search draws on every point of "
                "the family."
            ),
        ] = None,
        seed: Annotated[
            int | None,
            Field(
                description="The seed of the reverse search: the same inputs give the "
                "same answer."
            ),
        ] = None,
    ) -> dict:
        """Stored scenarios. Actions: list, get, save, validate, reverse.

        Read ``guide(name="scenario-from-corpus")`` before building one. ``save`` checks the
        contract and writes ``scenarios/<id>.json``; ``validate`` loads the macro and satellite
        versions the file names and names every problem that keeps the scenario from being
        computed, including corpus fragments not visible on the calculation day when the index
        exists, and reports how the elements are marked and what the scenario leaves unsaid
        under ``labels`` and ``warnings``. ``reverse`` searches the scenario's family on
        ``portfolio``: the worst loss under the price ``kappa``, or the lowest price whose loss
        reaches ``level``.
        """
        if action == "list":
            return scenario_tool.catalogue(ws.scenarios)
        if action == "get":
            return scenario_tool.get(ws.scenarios, _need(id, "id"))
        if action == "save":
            return scenario_tool.save(ws.scenarios, _need(id, "id"), _need(spec, "spec"))
        if action == "reverse":
            from mst import plugins as _plugins
            from mst.scenario.reverse import inf_dist, sup_rho, width_family

            loaded = scenario_tool.load(ws.scenarios, _need(id, "id"))
            installed = _plugins.registry()
            macro = macro_version.load(ws.models, loaded.macro, installed.macro)
            satellites = satellite_bundle.load(
                ws.models, loaded.satellites, installed.satellites
            )
            found = portfolio_tool.load(ws.portfolios, _need(portfolio, "portfolio"))
            family = width_family(loaded, macro)
            if kappa is not None:
                return sup_rho(
                    family,
                    kappa,
                    macro=macro,
                    satellites=satellites,
                    positions=found.positions,
                    draws=_need(draws, "draws"),
                    seed=_need(seed, "seed"),
                )
            return inf_dist(
                family,
                _need(level, "level"),
                macro=macro,
                satellites=satellites,
                positions=found.positions,
                draws=_need(draws, "draws"),
                seed=_need(seed, "seed"),
            )
        if action == "validate":
            return scenario_tool.validate(ws.scenarios, _need(id, "id"), ws.models, _indexed(ws))
        raise ValueError(
            f"scenario: unknown action {action!r}, try list, get, save, reverse, validate"
        )

    @server.tool(
        annotations=ToolAnnotations(
            title="Scenario runs",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=False,
        )
    )
    def run(
        action: Annotated[
            str,
            Field(
                description="What to do: start a run, read its status or result, or list "
                "the runs."
            ),
        ],
        id: Annotated[
            str | None,
            Field(description="The run id that status and result read."),
        ] = None,
        scenario: Annotated[
            str | None,
            Field(description="The scenario the run calculates."),
        ] = None,
        portfolio: Annotated[
            str | None,
            Field(
                description="The stored portfolio the run values; give portfolio or "
                "positions, not both."
            ),
        ] = None,
        positions: Annotated[
            list[dict[str, Any]] | None,
            Field(
                description="The positions the run values, given in the call; give "
                "portfolio or positions, not both."
            ),
        ] = None,
        draws: Annotated[
            int | None,
            Field(description="How many paths the run draws."),
        ] = None,
        seed: Annotated[
            int | None,
            Field(
                description="The seed of the run: the same scenario, versions and seed "
                "give the same numbers."
            ),
        ] = None,
        base_currency: Annotated[
            str,
            Field(
                description="The currency the notional of positions is stated in; a "
                "portfolio brings its own."
            ),
        ] = "RUB",
        targets: Annotated[
            list[str] | None,
            Field(
                description="The drivers the forecast reports, by id; every driver when "
                "omitted. An empty list is refused."
            ),
        ] = None,
        contributions: Annotated[
            bool,
            Field(
                description="Whether the report also breaks the loss down by position."
            ),
        ] = False,
    ) -> dict:
        """Runs on the paths of a scenario. Actions: start, status, result, list.

        Read ``guide(name="run-report")`` first. ``start`` draws the scenario's paths and
        calculates a portfolio's positions — or the ``positions`` given in the call — on them,
        as a job on the ``runs`` resource; ``status`` reports that job and what the run has
        written, ``result`` the per-position results, the conditional forecast of the
        ``targets`` (``forecast.json``: P5/P50/P95 of every target driver by month, in its own
        unit; every driver when ``targets`` is omitted) and the report of a portfolio run,
        ``list`` every run of the workspace. ``base_currency`` is the base of ``positions``; a
        portfolio brings its own.
        """
        if action == "start":
            if portfolio and positions:
                raise ValueError("run start: give portfolio or positions, not both")
            request = {
                "scenario": _need(scenario, "scenario"),
                "draws": _need(draws, "draws"),
                "seed": _need(seed, "seed"),
            }
            if portfolio:
                request["portfolio"] = portfolio
            else:
                request["positions"] = _need(positions, "positions")
                request["base_currency"] = base_currency
            if targets is not None:
                request["targets"] = targets
            if contributions:
                request["contributions"] = True
            run_id = runs.new_id()
            runs.begin(ws, run_id, request)
            try:
                job = tasks.start(
                    ws, "runs", ["-m", "mst.runs", run_id, "--workspace", str(ws.root)]
                )
            except Exception as error:
                store.write_json(
                    runs.folder(ws, run_id) / "meta.json",
                    {"id": run_id, "status": "failed", "error": f"{type(error).__name__}: {error}"},
                )
                raise
            store.write_json(runs.folder(ws, run_id) / "job.json", {"job": job})
            return {"run": run_id, "job": job}
        if action == "status":
            run_id = _need(id, "id")
            link = store.read_json(runs.folder(ws, run_id) / "job.json") or {}
            return {
                "run": run_id,
                "meta": _meta(ws, run_id),
                "job": tasks.status(ws, link["job"]) if link.get("job") else None,
            }
        if action == "result":
            return runs.result(ws, _need(id, "id"))
        if action == "list":
            return runs.catalogue(ws)
        raise ValueError(f"run: unknown action {action!r}, try start, status, result, list")

    @server.tool(
        annotations=ToolAnnotations(
            title="Stored portfolios",
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=False,
            openWorldHint=False,
        )
    )
    def portfolio(
        action: Annotated[
            str,
            Field(
                description="What to do: list, get, save, delete, check against a "
                "scenario, or value the exposure. delete removes the portfolio's file."
            ),
        ],
        id: Annotated[
            str | None,
            Field(
                description="The portfolio id, which is also the name of its file."
            ),
        ] = None,
        spec: Annotated[
            dict[str, Any] | None,
            Field(
                description="The portfolio specification save checks against the contract "
                "before it writes the file."
            ),
        ] = None,
        scenario: Annotated[
            str | None,
            Field(
                description="The scenario whose macro and satellite versions check and "
                "exposure work with."
            ),
        ] = None,
    ) -> dict:
        """Stored portfolios. Actions: list, get, save, delete, check, exposure.

        ``save`` checks the contract and writes ``portfolios/<id>.json``; ``check`` names every
        position the scenario's macro and satellite versions cannot serve, with the reason of a
        driver the out-of-sample check rejected; ``exposure`` values the positions at the
        calculation month of the scenario's macro version.
        """
        if action == "list":
            return portfolio_tool.catalogue(ws.portfolios)
        if action == "get":
            return portfolio_tool.get(ws.portfolios, _need(id, "id"))
        if action == "save":
            return portfolio_tool.save(ws.portfolios, _need(id, "id"), _need(spec, "spec"))
        if action == "delete":
            return portfolio_tool.delete(ws.portfolios, _need(id, "id"))
        if action == "check":
            return portfolio_tool.check(ws, _need(id, "id"), _need(scenario, "scenario"))
        if action == "exposure":
            return portfolio_tool.exposure(ws, _need(id, "id"), _need(scenario, "scenario"))
        raise ValueError(
            f"portfolio: unknown action {action!r}, try list, get, save, delete, check, exposure"
        )

    @server.tool(
        annotations=ToolAnnotations(
            title="Model versions and plugins",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=False,
        )
    )
    def model(
        kind: Annotated[
            str,
            Field(
                description="What to work with: macro versions, satellite bundles, or "
                "the catalog of installed plugins."
            ),
        ],
        action: Annotated[
            str,
            Field(
                description="What to do; the valid actions depend on kind, as the tool "
                "description lists."
            ),
        ],
        version: Annotated[
            str | None,
            Field(
                description="The version hash: what describe and base read, and the "
                "macro version the satellites of an estimate are built on."
            ),
        ] = None,
        as_of: Annotated[
            str | None,
            Field(
                description="The date estimate and backtest work as of, YYYY-MM-DD: only "
                "data published by then enters."
            ),
        ] = None,
        only: Annotated[
            list[str] | None,
            Field(
                description="The macro variables backtest limits itself to, and the ones "
                "base reports; all when omitted."
            ),
        ] = None,
        model_id: Annotated[
            str | None,
            Field(
                description="The macro plugin estimate and backtest run; the only one "
                "installed when omitted."
            ),
        ] = None,
        assignment: Annotated[
            dict[str, str] | None,
            Field(
                description='A mapping of a quantity to the satellite plugin that '
                'estimates it, e.g. {"bond_spread": "hier.gibbs"}, overriding the plugin '
                "that declares it."
            ),
        ] = None,
        horizon: Annotated[
            int,
            Field(
                description="How many months of the base paths base reports."
            ),
        ] = 12,
    ) -> dict:
        """Model versions and plugins. kind: macro, satellites, catalog.

        The episode check of the models is ``backtest`` (``python -m mst.metrics``).
        Actions for ``macro``: list, describe, base, estimate, backtest; for ``satellites``:
        list, describe, estimate; for ``catalog``: list, the installed model plugins.
        ``base`` gives p5, p50 and p95 of the macro variables under the base s_0 by month, up
        to ``horizon``, in the model scale, for ``only`` the variables named; it starts no job.
        ``model_id`` names the macro model to estimate or backtest — the only one installed
        when omitted; ``assignment`` maps a quantity to the satellite model that estimates
        it in an ``estimate`` of the satellites, overriding the model that declares it.
        ``version`` is the version hash ``describe`` and ``base`` read and the macro version the
        satellites of an ``estimate`` are built on; ``list`` takes no version.
        """
        if kind == "catalog":
            if action != "list":
                raise ValueError("model: unknown action for catalog, try list")
            return models.catalog()
        if kind == "satellites":
            if action == "list":
                return {"versions": satellite_bundle.catalogue(ws.models)}
            if action == "describe":
                return satellite_bundle.describe(ws.models, _need(version, "version"))
            if action == "estimate":
                argv = _satellites_argv(ws, _need(version, "version"), assignment)
                return _start(ws, "models", argv)
            raise ValueError(
                f"model: unknown action {action!r} for satellites, try list, describe, estimate"
            )
        if kind != "macro":
            raise ValueError(f"model: unknown kind {kind!r}, try macro, satellites, catalog")
        if action == "list":
            return {"versions": macro_version.catalogue(ws.models)}
        if action == "describe":
            return macro_version.describe(ws.models, _need(version, "version"))
        if action == "base":
            from mst import plugins as _plugins

            return macro_version.base(
                ws.models,
                _need(version, "version"),
                _plugins.registry().macro,
                horizon=horizon,
                only=only,
            )
        if action == "estimate":
            return _start(
                ws,
                "models",
                _macro_argv(
                    ws,
                    "estimate-macro",
                    _need(as_of, "as_of"),
                    models.selected(model_id).model_id,
                ),
            )
        if action == "backtest":
            model = models.selected(model_id).model_id
            return _start(
                ws, "models", _backtest_argv(ws, _need(as_of, "as_of"), model, only or [])
            )
        raise ValueError(
            f"model: unknown action {action!r}, try list, describe, base, estimate, backtest"
        )

    return server


def _corpus_argv(ws: store.Workspace, command: str, *ids: str) -> list[str]:
    """The command line of ``python -m mst.corpus`` for this workspace."""
    return ["-m", "mst.corpus", command, *ids, "--corpus", str(ws.corpus)]


async def _wait_for_job(ws: store.Workspace, job_id: str, timeout_sec: float) -> dict:
    """The job's record once it ends, or the record when ``timeout_sec`` runs out.

    The loop sleeps between file reads, so the server answers other calls while one client
    waits; the record is the one ``status`` returns, with ``running`` telling the caller
    whether to wait again.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(timeout_sec, 0.0)
    while True:
        record = tasks.status(ws, job_id)
        if not record["running"] or loop.time() >= deadline:
            return record
        await asyncio.sleep(WAIT_STEP)


def _day(as_of: str) -> date:
    """The day an action works as of."""
    return date.fromisoformat(as_of)


def _indexed(ws: store.Workspace) -> Path | None:
    """The corpus root when its index exists, so a scenario's sources can be checked."""
    return ws.corpus if (ws.corpus / "index" / "chunks.jsonl").is_file() else None


def _macro_argv(
    ws: store.Workspace,
    command: str,
    as_of: str,
    model_id: str,
    only: list[str] | None = None,
) -> list[str]:
    """The command line of ``python -m mst.models`` for this workspace."""
    day = date.fromisoformat(as_of).isoformat()
    argv = [
        "-m",
        "mst.models",
        command,
        model_id,
        "--as-of",
        day,
        "--store",
        str(ws.series),
        "--models",
        str(ws.models),
    ]
    if command == "backtest":
        argv += ["--corpus", str(ws.corpus / "manifest.json")]
        if only:
            argv += ["--only", *only]
    return argv


def _backtest_argv(
    ws: store.Workspace, as_of: str, model_id: str, only: list[str] | None = None
) -> list[str]:
    """The command line of ``python -m mst.metrics backtest`` for this workspace."""
    day = date.fromisoformat(as_of).isoformat()
    argv = [
        "-m",
        "mst.metrics",
        "backtest",
        model_id,
        "--as-of",
        day,
        "--store",
        str(ws.series),
        "--models",
        str(ws.models),
        "--corpus",
        str(ws.corpus / "manifest.json"),
        "--drivers",
        str(ws.drivers / "catalog.json"),
    ]
    if only:
        argv += ["--only", *only]
    return argv


def _meta(ws: store.Workspace, run_id: str) -> dict | None:
    """The meta a run has written so far; a run that has not started has none."""
    try:
        return runs.meta(ws, run_id)
    except FileNotFoundError:
        return None


def _need(value: Any, name: str) -> Any:
    """The argument an action cannot work without."""
    if value is None or value == "":
        raise ValueError(f"{name} is required for this action")
    return value


def _satellites_argv(
    ws: store.Workspace, macro: str, assignment: dict[str, str] | None = None
) -> list[str]:
    """The command line of ``python -m mst.models estimate-satellites`` for this workspace."""
    argv = [
        "-m",
        "mst.models",
        "estimate-satellites",
        "--macro",
        macro,
        "--store",
        str(ws.series),
        "--models",
        str(ws.models),
        "--drivers",
        str(ws.drivers / "catalog.json"),
    ]
    for quantity, model_id in sorted((assignment or {}).items()):
        argv += ["--model", f"{quantity}={model_id}"]
    return argv


def _update_drivers(ws: store.Workspace, quantity: str) -> dict:
    """Discover the objects of ``quantity`` and fetch their series, as a task on series."""
    return _start(
        ws,
        "series",
        [
            "-m",
            "mst.drivers",
            "update",
            quantity,
            "--store",
            str(ws.series),
            "--drivers",
            str(ws.drivers / "catalog.json"),
        ],
    )


def _rows(ws: store.Workspace, series_id: str, as_of: str | None) -> dict:
    """The values of ``series_id`` known on ``as_of``, one row per month.

    A driver ``<quantity>:<object>`` reads the series of its history.
    """
    day = date.fromisoformat(as_of) if as_of else date.today()
    quantity, on_driver, object_ = series_id.partition(":")
    stored = driver_series(quantity, object_) if on_driver else series_id
    observations = SeriesStore(ws.series).as_of(stored, day)
    return {
        "id": series_id,
        "as_of": day.isoformat(),
        "rows": [
            {
                "month": row.month.isoformat(),
                "value": row.value,
                "published": row.published.isoformat(),
                "approximate": row.approximate,
            }
            for row in observations
        ],
    }


def _update(ws: store.Workspace, ids: list[str]) -> dict:
    """Fetch the named series, or every one of them, as a task on the series resource."""
    return _start(ws, "series", ["-m", "mst.data", "update", *ids, "--workspace", str(ws.root)])


def _start(ws: store.Workspace, resource: str, argv: list[str]) -> dict:
    """Start ``argv`` as a task on ``resource`` and name it."""
    return {"job": tasks.start(ws, resource, argv), "argv": argv}


def default_workspace() -> Path:
    """The workspace without ``--workspace``: the rule ``dsh/index.js`` follows too."""
    named = os.environ.get("MACROSTRESS_WORKSPACE")
    return Path(named) if named else Path.home() / ".macrostress" / "workspace"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mst.mcp")
    parser.add_argument("--workspace", type=Path, default=None)
    ws = store.Workspace(parser.parse_args(argv).workspace or default_workspace())
    ws.root.mkdir(parents=True, exist_ok=True)
    store.write_state(ws)
    build(ws).run()


if __name__ == "__main__":
    main()
