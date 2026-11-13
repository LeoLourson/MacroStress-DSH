"""The MCP server: a tool parses its arguments and calls one function of the core."""

import argparse
import asyncio
import contextlib
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Annotated, Any

from fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from . import bench, guides, models, runs, store
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
given event happens. A scenario s = (N, I, O, A) is built from the text of the event and the \
corpus: the narrative names the event, the sources back its numbers, the interventions state \
what the event decides (a rate, a price, a volume), the observations state what the event \
constrains or expects, and a human approves the file. The models compute its consequences; the \
model versions, the horizon and the number of paths belong to the calculation request, not to \
the scenario.

Read a guide before the work it covers: guide(name="scenario-from-corpus") when the user names \
an event or asks what happens if it occurs, even without the words scenario or corpus; \
guide(name="run-report") before a calculation or its report; guide(name="drivers") before work \
on drivers and satellites; guide(name="model-inputs") before answering the questions of a model \
that declares inputs; guide(name="elicitation-orchestration") before running that interview with \
several agents, one per shock. guide() lists them.

When the user asks you to answer a model's questions or to act as its expert, start with \
elicit(action="next", as_of=..., model_id=...). It gives ONE small question with its schema \
and answer_call. Copy answer_call, add payload with your own answer object and call elicit \
again. Each answer saves a part and returns the next question; continue until complete=true. \
Do not ask the human for numbers, substitute placeholders, or build a full shock table. \
Partial answers survive restarts but cannot be estimated until all parts pass the full check. \
elicit(action="list", ...) lists input keys and action="status" says which are complete. \
Full question/write actions remain under model(kind="inputs"). When complete, \
model(kind="macro", action="estimate", ...) builds the version from the answers. \
The questions and the answers live in that tool: reading sources or searching the files of the \
workspace does not answer them, and several installed models that ask questions need model_id. \
The agent is the expert of that interview: the numbers are its own quantitative judgements from \
the state table the question prints and from the model corpus; it does not ask the human for \
them and does not take them from tests or fixtures. The market variant reads the model corpus, \
so its answers need that corpus indexed and fetched first.

Search the corpus before writing any number of a scenario. Mark an intervention or an \
observation with "event" when the text of the event states its number, with a corpus fragment \
only when the fragment gives its number, and with "assumption" otherwise. When \
neither the event nor the corpus gives a number, propose one from the nearest episode and mark \
it "assumption": a scenario with no interventions and no observations records no event, and \
questions to the user come after the scenario, not instead of it. A human approves a scenario: \
never fill approved_by yourself and never start a calculation that was not asked for.

A tool that starts a job returns its id at once. Wait with jobs(action="wait", id=<job>): the \
server holds the call until the job ends, and jobs(action="log", id=<job>) shows what it \
printed.\
"""

WAIT_STEP = 0.2
"""Seconds between the file reads of a ``wait``, the only cost of waiting server-side."""

EPISODE_PLAN = Path(__file__).resolve().parents[2] / "bench" / "model-backtest" / "episodes.toml"
"""The episode plan of the backtest, found next to the sources of a repository checkout; a
package installed without the repository has none."""


@contextmanager
def _plugin_output_to_stderr() -> Iterator[None]:
    """A plugin's stray ``print()`` goes to stderr, and the stdio channel stays JSON-RPC.

    The tools this wraps — scenario, portfolio, model — load and run plugin code in the
    server's process (the heavy estimate and run jobs are separate processes and do not
    need it). The stdio transport took its own wrapper of ``sys.stdout.buffer`` at startup, so
    rebinding ``sys.stdout`` for the duration of a call cannot touch the channel: the
    plugin's output reaches the client's log instead of corrupting the protocol. An
    fd-level redirect was rejected because a concurrent response written through the
    transport while the redirect held would land on stderr and be lost.
    """
    with contextlib.redirect_stdout(sys.stderr):
        yield


def build(ws: store.Workspace) -> FastMCP:
    """The server of one workspace; every tool closes over it."""
    server = FastMCP("macrostress", instructions=INSTRUCTIONS)
    session: dict[str, Any] = {}
    """The bench event the agent is building a scenario for, and its corpus calls so far."""

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
            destructiveHint=True,
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
                description="The kind of fragment search limits itself to: methodology, "
                "scenario_set, transmission, episode or market (market expectations, the "
                "model corpus alone)."
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
        corpus: Annotated[
            str,
            Field(
                description='Which corpus to work with: "scenario", the default, or '
                '"model" — the corpus of the models, with its own manifest and index.'
            ),
        ] = "scenario",
    ) -> dict:
        """The scenario corpus and the model corpus. Actions: search, get, list, add, index, fetch.

        ``corpus="model"`` works with the corpus of the models — the material a model's
        elicitation reads: episodes, the state at the date, transmission reviews, market
        expectations — which has its own manifest and index under ``corpus-model/``.
        ``search`` and ``get`` read the index as it was on ``as_of``; ``list`` shows what the
        manifest holds; ``add`` registers a file downloaded by hand under its ``key``. ``index``
        and ``fetch`` are tasks on the corpus resource. Inside a bench session every call and
        its answer go to the journal of the event.
        """
        if corpus not in {"scenario", "model"}:
            raise ValueError(
                f"corpus: unknown corpus {corpus!r}, try scenario or model"
            )
        found = _corpus_call(
            action, query, as_of, id, kind, region, factors, k, key, file, url, ids, corpus
        )
        if session:
            session["calls"].append(
                {"action": action, "corpus": corpus, "query": query, "as_of": as_of, "id": id,
                 "kind": kind, "region": region, "factors": factors, "k": k, "result": found}
            )
        return found

    def _corpus_call(action, query, as_of, id, kind, region, factors, k, key, file, url, ids,
                     corpus):
        root = ws.corpus if corpus == "scenario" else ws.model_corpus
        if action == "search":
            return corpus_tool.search(
                root,
                _need(query, "query"),
                _day(_need(as_of, "as_of")),
                kind=kind,
                region=region,
                factors=factors,
                k=k,
            )
        if action == "get":
            return corpus_tool.get(root, _need(id, "id"), _day(_need(as_of, "as_of")))
        if action == "list":
            return corpus_tool.catalogue(root)
        if action == "add":
            return corpus_tool.add(root, _need(key, "key"), Path(_need(file, "file")), url)
        if action == "index":
            return _start(ws, "corpus", _corpus_argv(ws, "index", root=root))
        if action == "fetch":
            return _start(ws, "corpus", _corpus_argv(ws, "fetch", *(ids or []), root=root))
        raise ValueError(
            f"corpus: unknown action {action!r}, try search, get, list, add, index, fetch"
        )

    @server.tool(
        annotations=ToolAnnotations(
            title="Stored scenarios",
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=False,
            openWorldHint=False,
        )
    )
    def scenario(
        action: Annotated[
            str,
            Field(
                description="What to do: list, get, save, validate or reverse. save "
                "overwrites a scenario stored under the same id. bench starts the "
                "scenario of a bench event, bench_save keeps it in the bench results."
            ),
        ],
        event: Annotated[
            str | None,
            Field(
                description="The bench event bench and bench_save work on, like E017."
            ),
        ] = None,
        notes: Annotated[
            str | None,
            Field(
                description="What bench_save adds to the journal of the event: how the "
                "numbers of the scenario were chosen."
            ),
        ] = None,
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
                "before it writes the file: narrative, sources, interventions, observations, "
                "approved_by. Neither the model versions nor the horizon are part of it."
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
        contract and writes ``scenarios/<id>.json``; ``validate`` names every problem that keeps
        the scenario from being computed on the newest finished macro and satellite versions, or
        on the versions the call names, including corpus fragments not visible on the calculation
        day when the index exists, and reports how the elements are marked and what the scenario
        leaves unsaid under ``labels`` and ``warnings``. ``reverse`` searches the scenario's family
        on ``portfolio``: the worst loss under the price ``kappa``, or the lowest price whose loss
        reaches ``level``. ``bench`` returns the text of the bench ``event`` and starts its
        journal of corpus calls; build the scenario by the guide, then ``bench_save`` checks
        ``spec`` and keeps it with the journal and ``notes`` in the bench results, beside an
        empty review the expert fills. A bench scenario is never approved and never computed.
        """
        with _plugin_output_to_stderr():
            if action == "bench":
                found = bench.event(_need(event, "event"))
                session.clear()
                session.update(event=found["event"], calls=[])
                return found
            if action == "bench_save":
                if session.get("event") != _need(event, "event"):
                    raise ValueError(
                        f"scenario bench_save: the session builds {session.get('event')}, "
                        f"not {event}; start it with action bench"
                    )
                saved = bench.save_scenario(event, _need(spec, "spec"), session["calls"], notes)
                session.clear()
                return saved
            if action == "list":
                return scenario_tool.catalogue(ws.scenarios)
            if action == "get":
                return scenario_tool.get(ws.scenarios, _need(id, "id"))
            if action == "save":
                return scenario_tool.save(ws.scenarios, _need(id, "id"), _need(spec, "spec"))
            if action == "reverse":
                from mst.scenario.reverse import inf_dist, sup_rho, width_family

                loaded = scenario_tool.load(ws.scenarios, _need(id, "id"))
                macro, satellites = scenario_tool.versions_of(ws.models)
                found = portfolio_tool.load(ws.portfolios, _need(portfolio, "portfolio"))
                family = width_family(loaded, macro, satellites)
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
                return scenario_tool.validate(
                    ws.scenarios, _need(id, "id"), ws.models, _indexed(ws)
                )
            raise ValueError(
                f"scenario: unknown action {action!r}, try list, get, save, reverse, validate, "
                "bench, bench_save"
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
                description="The currency the notional of positions is stated in, RUB "
                "only: the models price against the rouble. A portfolio brings its own."
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
                description="Whether to compare the full scenario with calculations without "
                "each intervention or observation, using the same random numbers."
            ),
        ] = False,
        horizon: Annotated[
            int | None,
            Field(
                description="The horizon of the calculation in months; twelve by default "
                "and never shorter than the last month a statement names."
            ),
        ] = None,
        inner: Annotated[
            int | None,
            Field(
                description="How many paths share one draw of the parameters and the "
                "interventions; one by default."
            ),
        ] = None,
        macro: Annotated[
            str | None,
            Field(
                description="The macro version hash; the newest finished one when omitted."
            ),
        ] = None,
        satellites: Annotated[
            str | None,
            Field(
                description="The satellite bundle hash estimated on that macro version; "
                "the newest one when omitted."
            ),
        ] = None,
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
            if horizon is not None:
                request["horizon"] = horizon
            if inner is not None:
                request["inner"] = inner
            if macro is not None:
                request["macro"] = macro
            if satellites is not None:
                request["satellites"] = satellites
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
                "scenario, or value the exposure. save overwrites a portfolio stored under "
                "the same id; delete removes the portfolio's file."
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
        driver the out-of-sample check rejected, and warns what an impact on a macro variable
        reaches through the positions' drivers, with the long-run loading of each one (below 0.1
        it barely reacts); ``exposure`` values the positions at the calculation month of the
        scenario's macro version.
        """
        with _plugin_output_to_stderr():
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
                return portfolio_tool.exposure(
                    ws, _need(id, "id"), _need(scenario, "scenario")
                )
            raise ValueError(
                f"portfolio: unknown action {action!r}, "
                "try list, get, save, delete, check, exposure"
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
                description="What to work with: macro versions, satellite bundles, the "
                "inputs a model declares (the answers the agent writes), or the catalog of "
                "installed plugins."
            ),
        ],
        action: Annotated[
            str,
            Field(
                description="What to do. macro: list, describe, base, estimate, backtest; "
                "satellites: list, describe, estimate; inputs: list, question, write, "
                "status; catalog: list."
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
                description="The episode ids backtest limits itself to, or the macro variables "
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
        key: Annotated[
            str | None,
            Field(
                description="The input key question and write read: one of the keys a "
                "model declares for the date."
            ),
        ] = None,
        payload: Annotated[
            dict[str, Any] | None,
            Field(
                description="The answer write checks and stores, as the question's "
                "schema asks."
            ),
        ] = None,
        form: Annotated[
            str | None,
            Field(
                description="The structural form a macro estimate stores: 'default' or "
                "'recursive'; omitted keeps the model's own default."
            ),
        ] = None,
        horizon: Annotated[
            int,
            Field(
                description="How many months of the base paths base reports."
            ),
        ] = 12,
    ) -> dict:
        """Model versions and plugins. kind: macro, satellites, inputs, catalog.

        The episode check of the models is ``backtest`` (``python -m mst.metrics``).
        Actions for ``macro``: list, describe, base, estimate, backtest; for ``satellites``:
        list, describe, estimate; for ``catalog``: list, the installed model plugins.
        ``base`` gives p5, p50 and p95 of the macro variables under the base s_0 by month, up
        to ``horizon``, in the model scale, for ``only`` the variables named; it starts no job.
        ``model_id`` names the macro model to estimate or backtest — the only one installed
        when omitted; ``assignment`` maps a quantity to the satellite model that estimates
        it in an ``estimate`` of the satellites, overriding the model that declares it.
        ``version`` is the version hash ``describe`` and ``base`` read and the macro version the
        satellites of an ``estimate`` are built on; ``list`` takes no version. ``form`` is the
        structural form a macro ``estimate`` stores — ``default`` or ``recursive`` — and any
        other value is refused by name; omitted keeps the model's own default.

        ``inputs`` is the elicitation a model that declares inputs is estimated from; the
        ``elicit`` tool serves the same four actions and is the one to call when the user asks
        for the answers of a model's questions. ``list`` and ``status`` name every key written
        or missing, ``question`` gives the text the agent reads before answering and ``write``
        checks the answer's form and stores it. The inputs are written by the agent itself and
        no human approves them — read guide(name="model-inputs") first — and ``model_id`` and
        ``as_of`` name the model and the date; a refused answer comes back with the field it
        was refused for.
        """
        if form not in (None, "default", "recursive"):
            raise ValueError(f"model: unknown form {form!r}, try default or recursive")
        with _plugin_output_to_stderr():
            if kind == "catalog":
                if action != "list":
                    raise ValueError("model: unknown action for catalog, try list")
                return models.catalog()
            if kind == "inputs":
                return _model_inputs(ws, action, model_id, as_of, key, payload)
            if kind == "satellites":
                if action == "list":
                    return {"versions": satellite_bundle.catalogue(ws.models)}
                if action == "describe":
                    return satellite_bundle.describe(ws.models, _need(version, "version"))
                if action == "estimate":
                    argv = _satellites_argv(ws, _need(version, "version"), assignment)
                    return _start(ws, "models", argv)
                raise ValueError(
                    f"model: unknown action {action!r} for satellites, "
                    "try list, describe, estimate"
                )
            if kind != "macro":
                raise ValueError(
                    f"model: unknown kind {kind!r}, try macro, satellites, catalog"
                )
            if action == "list":
                return {"versions": macro_version.catalogue(ws.models)}
            if action == "describe":
                return macro_version.describe(
                    ws.models, version or _newest_version(ws, model_id)
                )
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
                        form=form,
                    ),
                )
            if action == "backtest":
                model = models.selected(model_id).model_id
                return _start(
                    ws, "models", _backtest_argv(ws, _need(as_of, "as_of"), model, only or [])
                )
            raise ValueError(
                f"model: unknown action {action!r}, "
                "try list, describe, base, estimate, backtest"
                + (
                    '; the answers of a model\'s questions are the elicit tool, or kind="inputs" '
                    "with action list, question, write or status"
                    if action in {"inputs", "question", "write", "status", "elicit"}
                    else ""
                )
            )

    @server.tool(
        annotations=ToolAnnotations(
            title="The questions of a model and the answers the agent writes",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=False,
        )
    )
    def elicit(
        action: Annotated[
            str,
            Field(
                description="Use next to start or resume one small question; answer saves its "
                "payload using the returned ticket and gives the next question. list and status "
                "only report completed inputs. Do not request full question/write here."
            ),
        ],
        as_of: Annotated[
            str | None,
            Field(
                description="The calculation date, YYYY-MM-DD: required for next and list; "
                "answer reads the date from its ticket."
            ),
        ] = None,
        model_id: Annotated[
            str | None,
            Field(
                description="The macro model that asks the questions; every installed model "
                "that does when omitted and one of them when several."
            ),
        ] = None,
        key: Annotated[
            str | None,
            Field(
                description="The input key question and write read: one of the keys list names."
            ),
        ] = None,
        agent: Annotated[
            str | None,
            Field(
                description="The name of the answering agent: it is written to the journal "
                "and to the claim of the key, so progress names its owner."
            ),
        ] = None,
        payload: Annotated[
            dict[str, Any] | None,
            Field(
                description="The answer object write checks and stores, as the question's "
                "schema asks; the object itself, not a string of JSON."
            ),
        ] = None,
        ticket: Annotated[
            str | None,
            Field(description="Copy the ticket from next's answer_call unchanged; it addresses "
                  "the model, date, shock and exact step without repeating those fields."),
        ] = None,
    ) -> dict:
        """Answer a macro model's questions in small resumable steps.

        Start with ``next`` and model_id/as_of. It gives one small question, its schema and
        answer_call. Copy answer_call with your answer object as payload to ``answer``;
        the result saves it and gives the next question. Continue until complete=true.
        Each response step asks nine numbers, not a full shock table. Partial answers survive
        a restart and never count as a completed input. Never substitute placeholders or ask
        the human for numbers. ``next`` resumes the first missing step.

        A model that declares inputs is estimated from the agent's own answers, not from the
        series: ``list`` names every input key on the date and ``status`` repeats the summary.
        Full question/write operations belong to model(kind="inputs"), not this interview.
        The agent is the expert: the numbers are its own judgements, not the
        human's and not the fixtures of the tests. When nothing is missing, build the version:
        model(kind="macro", action="estimate", model_id=..., as_of=...) starts a job, and
        jobs(action="wait", id=...) holds until it ends. The same actions are kind="inputs" of
        the model tool.
        """
        with _plugin_output_to_stderr():
            from mst.plugin_sdk import interview

            if action in {"next", "question"}:
                return _interview(ws, as_of, model_id, key).next(key, agent)
            if action in {"answer", "write"}:
                found = interview.address(ws.elicitation, ticket)
                model = models.selected(found["model"])
                day = date.fromisoformat(found["date"])
                return interview.Session(model, _macro_context(ws, day)).answer(
                    ticket, payload, agent
                )
            return _model_inputs(ws, action, model_id, as_of, key, payload)

    @server.tool(
        annotations=ToolAnnotations(
            title="The interview of a model: who answers which input and how far",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=False,
        )
    )
    def interview(
        action: Annotated[
            str,
            Field(
                description="progress, claim, release, restart, audit, accept, reopen, history "
                "or journal: the state of an elicitation, not its answers."
            ),
        ],
        as_of: Annotated[
            str,
            Field(description="The calculation date, YYYY-MM-DD: the interview of this date."),
        ],
        model_id: Annotated[
            str,
            Field(
                description="The macro model that asks the questions, for example "
                "expert.elicited.rw; model(kind=\"catalog\") lists the installed ones."
            ),
        ],
        key: Annotated[
            str | None,
            Field(
                description="The input key, one shock or market: named by every action that "
                "touches one answer, left out by progress and by the whole-model audit."
            ),
        ] = None,
        agent: Annotated[
            str | None,
            Field(
                description="The name of the answering agent: it owns the claim and is "
                "written to the journal; required by claim."
            ),
        ] = None,
        steps: Annotated[
            list[str] | None,
            Field(
                description="The parts to ask again, by step id from the question; required "
                "by reopen and refused when a step is unknown."
            ),
        ] = None,
        reason: Annotated[
            str | None,
            Field(
                description="Why a published answer is accepted or reopened: written to the "
                "journal and to the accepted record."
            ),
        ] = None,
        minutes: Annotated[
            int,
            Field(description="How many minutes a claim stays alive without renewal, 45 by "
                  "default; a stopped agent does not block its shock forever."),
        ] = 45,
        limit: Annotated[
            int,
            Field(description="How many last journal events journal returns, 50 by default."),
        ] = 50,
    ) -> dict:
        """Track the interview of a model: ownership, progress, review and corrections.

        ``progress`` names every input key with its state, owner, answered parts and
        publication; call it before handing a shock to an agent. ``claim`` gives one key to
        one agent; a live claim of another agent refuses a second writer. ``release`` drops
        the caller's claim. ``restart`` resumes an agent whose run stopped — a network error,
        a timeout, a crashed client: it renews that agent's claim and returns the first
        missing part, so the same agent continues where the saved parts end and nothing is
        asked twice. ``audit`` lists the advisory findings of one published answer or
        of every one. ``accept`` records that the warnings were reviewed, with the reason;
        ``reopen`` asks named parts again, keeps the old answer as a version and makes the
        key missing until it is complete again. ``history`` lists the versions of one answer
        and ``journal`` the last events of the interview.

        This tool never writes an answer: the numbers come only through ``elicit``. The
        orchestrating agent owns the process: it hands one shock to one subagent, reads the
        warnings and either accepts the answer or reopens the steps they name.
        """
        with _plugin_output_to_stderr():
            day = _day(as_of)
            model = models.selected(model_id)
            if not model.input_keys(day):
                raise ValueError(
                    f"model {model_id!r} declares no inputs on {day.isoformat()}: it is "
                    "estimated from the series alone, so it has no interview"
                )
            from mst.plugin_sdk.interview import Session

            session = Session(model, _macro_context(ws, day))
            if action == "progress":
                return session.progress()
            if action == "claim":
                return session.claim(
                    _named(key, "claim", "key"), _named(agent, "claim", "agent"), minutes
                )
            if action == "release":
                return session.release(
                    _named(key, "release", "key"), _named(agent, "release", "agent")
                )
            if action == "restart":
                return session.restart(
                    _named(key, "restart", "key"), _named(agent, "restart", "agent"), minutes
                )
            if action == "journal":
                return session.journal(key, limit)
            if action == "audit":
                return session.audit(key)
            if action == "accept":
                return session.accept(
                    _named(key, "accept", "key"),
                    _named(agent, "accept", "agent"),
                    _named(reason, "accept", "reason"),
                )
            if action == "history":
                return session.history(_named(key, "history", "key"))
            if action == "reopen":
                return session.reopen(
                    _named(key, "reopen", "key"),
                    _named(steps, "reopen", "steps"),
                    _named(agent, "reopen", "agent"),
                    _named(reason, "reopen", "reason"),
                )
            raise ValueError(
                f"interview: unknown action {action!r}, try progress, claim, release, restart, "
                "audit, accept, reopen, history or journal"
            )

    return server


def _named(value, action: str, field: str) -> str:
    """A field one action cannot work without; the message names both."""
    if not value:
        raise ValueError(f"interview: {action} needs {field}")
    return value


def _macro_context(ws: store.Workspace, day: date):
    """The context a model's questions and checks see: series, registry, date, answers, corpus.

    It is the one an estimate builds, so a question never depends on a second definition.
    """
    from mst.macro.registry import load_registry as load_macro_registry
    from mst.model_sdk.contracts import MacroEstimateContext

    return MacroEstimateContext(
        store=SeriesStore(ws.series),
        registry=load_macro_registry(),
        as_of=day,
        inputs=ws.elicitation,
        fragment=_model_fragment(ws, day),
    )


def _interview(ws: store.Workspace, as_of, model_id, key):
    """The interview session of the model the caller named, or of the only one asking."""
    from mst.plugin_sdk.interview import Session

    if not as_of:
        raise ValueError("as_of is required for next: the calculation date YYYY-MM-DD")
    day = _day(as_of)
    if model_id is None:
        candidates = [model for model in models.registry().values() if model.input_keys(day)]
        if len(candidates) != 1:
            raise ValueError("next needs model_id: " + ", ".join(m.model_id for m in candidates))
        model = candidates[0]
    else:
        model = models.selected(model_id)
    if key is not None and key not in model.input_keys(day):
        raise ValueError(f"model declares no input {key!r}")
    return Session(model, _macro_context(ws, day))


def _model_inputs(
    ws: store.Workspace, action: str, model_id: str | None, as_of: str | None, key, payload
) -> dict:
    """The actions of ``model(kind="inputs")``: the answers a model declares for a date.

    The context the model's ``question`` and ``check_input`` see is the one an estimate
    would build: the series store, the registries, the calculation date, the folder of the
    answers and the visibility of the model corpus fragments on that date.
    """
    from mst import inputs as model_inputs

    if action not in {"list", "question", "write", "status"}:
        raise ValueError(
            f"model: unknown action {action!r} for inputs, try list, question, write, status"
        )
    if not as_of:
        raise ValueError(
            "model: as_of is required for inputs: the calculation date, YYYY-MM-DD"
        )
    day = _day(as_of)
    declaring = [model for model in models.registry().values() if model.input_keys(day)]
    if model_id is not None:
        model = models.selected(model_id)
        if model.model_id not in {candidate.model_id for candidate in declaring}:
            raise ValueError(
                f"model: {model.model_id} declares no inputs on {day.isoformat()}: it is "
                "estimated from the series alone; the models that ask questions are "
                + (", ".join(candidate.model_id for candidate in declaring) or "none installed")
            )
    elif len(declaring) == 1:
        model = declaring[0]
    elif not declaring:
        raise ValueError(
            f"model: no installed macro model declares inputs on {day.isoformat()}"
        )
    elif action == "list":
        # The caller asks what there is to answer: name every model that asks and its keys.
        return {
            "as_of": day.isoformat(),
            "models": [
                {
                    "model": candidate.model_id,
                    **model_inputs.status(ws.elicitation, candidate, day),
                }
                for candidate in declaring
            ],
        }
    else:
        raise ValueError(
            f"model: several macro models declare inputs on {day.isoformat()}, name one with "
            "model_id: " + ", ".join(candidate.model_id for candidate in declaring)
        )
    if action == "list":
        return {
            "model": model.model_id,
            "as_of": day.isoformat(),
            **model_inputs.status(ws.elicitation, model, day),
        }
    context = _macro_context(ws, day)
    if action in {"question", "write"}:
        # The key becomes a file name: only the model's own keys reach the folder.
        if not key:
            raise ValueError(
                "model: key is required for question and write: a key of "
                'model(kind="inputs", action="list")'
            )
        the_key = key
        if the_key not in model.input_keys(day):
            raise ValueError(
                f"model: {model.model_id} declares no input {the_key!r} on {day.isoformat()}"
            )
    if action == "question":
        return model.question(context, the_key)
    if action == "write":
        if payload is None:
            raise ValueError(
                "model: payload is required for write: the answer object the question's "
                'schema describes, not a string of it'
            )
        normal = model.check_input(context, the_key, payload)
        digest = model_inputs.write(ws.elicitation, day, the_key, normal)
        found = model_inputs.status(ws.elicitation, model, day)
        return {
            "model": model.model_id,
            "as_of": day.isoformat(),
            "key": the_key,
            "hash": digest,
            "missing": [row["key"] for row in found["keys"] if not row["written"]],
        }
    found = model_inputs.status(ws.elicitation, model, day)
    return {
        "model": model.model_id,
        "as_of": day.isoformat(),
        "missing": [row["key"] for row in found["keys"] if not row["written"]],
        "written": [row["key"] for row in found["keys"] if row["written"]],
    }


def _model_fragment(ws: store.Workspace, day: date):
    """Whether a fragment of the model corpus is visible on ``day``; ``None`` without an index."""
    if not (ws.model_corpus / "index" / "chunks.jsonl").is_file():
        return None

    def visible(fragment_id: str) -> bool:
        try:
            corpus_tool.get(ws.model_corpus, fragment_id, day)
        except ValueError:
            return False
        return True

    return visible


def _corpus_argv(
    ws: store.Workspace, command: str, *ids: str, root: Path | None = None
) -> list[str]:
    """The command line of ``python -m mst.corpus`` for this workspace."""
    return [
        "-m",
        "mst.corpus",
        command,
        *ids,
        "--corpus",
        str(root if root is not None else ws.corpus),
    ]


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
    form: str | None = None,
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
    if command == "estimate-macro":
        argv += ["--inputs", str(ws.elicitation)]
        if form is not None:
            argv += ["--form", form]
    if command == "backtest":
        argv += ["--corpus", str(ws.corpus / "manifest.json")]
        if only:
            argv += ["--only", *only]
    return argv


def _backtest_argv(
    ws: store.Workspace, as_of: str, model_id: str, only: list[str] | None = None
) -> list[str]:
    """The command line of ``python -m mst.metrics backtest`` for this workspace."""
    if not EPISODE_PLAN.is_file():
        raise ValueError(
            f"model: the episode plan {EPISODE_PLAN} is not there; run "
            "python -m mst.metrics backtest --episodes FILE with the plan of the repository"
        )
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
        "--episodes",
        str(EPISODE_PLAN),
        "--drivers",
        str(ws.drivers / "catalog.json"),
        "--inputs",
        str(ws.elicitation),
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


def _newest_version(ws: store.Workspace, model_id: str | None) -> str:
    """The newest installed version, of ``model_id`` when it is named.

    ``describe`` reads it when the caller names no version: the catalogue is newest first, so
    the first row of the model is the one a description of that model means. A model that asks
    questions has no version until its answers are estimated, and the error says so.
    """
    rows = macro_version.catalogue(ws.models)
    if model_id is not None:
        rows = [row for row in rows if row["model"] == model_id]
    if rows:
        return rows[0]["version"]
    if model_id is None:
        raise ValueError(
            "model: no macro version is installed yet: estimate one with "
            'model(kind="macro", action="estimate", as_of=...)'
        )
    raise ValueError(
        f"model: no version of {model_id!r} is installed yet: if that model asks questions, "
        f'answer them first — elicit(action="list", as_of=..., model_id={model_id!r}) — and '
        "build the version with model(kind=\"macro\", action=\"estimate\", ...)"
    )


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
    """Start ``argv`` as a task on ``resource`` and name it.

    The record carries the call that awaits the job: this server's jobs are not the background
    jobs of the client, so an agent that has just started one is told how to wait for it.
    """
    job = tasks.start(ws, resource, argv)
    return {"job": job, "argv": argv, "wait": f'jobs(action="wait", id="{job}")'}


LOCAL_WORKSPACE = Path("workspace")


def default_workspace() -> Path:
    """The workspace without ``--workspace``: the rule ``dsh/index.js`` follows too.

    ``MACROSTRESS_WORKSPACE``, else the first existing of ``~/.macrostress/workspace``, the user
    data directory of the OS and ``./workspace``, else the user data directory of the OS.
    ``main`` falls back to ``./workspace`` when that one cannot be created: managed machines often
    forbid writing outside the project.
    """
    named = os.environ.get("MACROSTRESS_WORKSPACE")
    if named:
        return Path(named)
    home = Path.home()
    if sys.platform == "win32":
        data = Path(os.environ.get("LOCALAPPDATA") or home / "AppData" / "Local")
    elif sys.platform == "darwin":
        data = home / "Library" / "Application Support"
    else:
        data = Path(os.environ.get("XDG_DATA_HOME") or home / ".local" / "share")
    shared = data / "macrostress" / "workspace"
    for found in (home / ".macrostress" / "workspace", shared, LOCAL_WORKSPACE):
        if found.is_dir():
            return found
    return shared


def _create(root: Path) -> None:
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        # Before the MCP handshake a traceback reaches the client only as "connection closed".
        sys.exit(
            f"macrostress: cannot create the workspace {root.resolve()}: {error}. "
            "Set MACROSTRESS_WORKSPACE (or pass --workspace) to a directory you can write to."
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mst.mcp")
    parser.add_argument("--workspace", type=Path, default=None)
    named = parser.parse_args(argv).workspace
    root = named or default_workspace()
    if named is None and not os.environ.get("MACROSTRESS_WORKSPACE"):
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError:
            root = LOCAL_WORKSPACE
    _create(root)
    ws = store.Workspace(root.resolve())
    store.write_state(ws)
    _load_compiled_code()
    build(ws).run()


def _load_compiled_code() -> None:
    """Import the plugins and scipy before the stdio transport starts reading stdin.

    On Windows the first load of scipy's compiled extensions inside a running server never
    returned while the transport's thread waited on stdin, so the first tool that loaded them
    hung until the client closed the pipe. Loaded here, they are in memory before any call.
    """
    from mst import plugins
    from mst.scenario import overlay  # noqa: F401  scipy.optimize

    plugins.registry()


if __name__ == "__main__":
    main()
