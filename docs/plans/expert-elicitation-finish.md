# Finish the expert.elicited.rw estimation

Written 2026-11-09. Workspace: the `workspace` folder of the repository.
Outcome: one stored macro version of `expert.elicited.rw` built from eighteen answered shocks,
with a report of what was accepted, what was reopened and what the estimate produced.

This file is the whole brief. Do not re-plan from scratch; verify the state, run the waves,
review the answers, estimate. Hand-off to a new session: open it in this repository with the
`macrostress` MCP server and say "follow `plans/expert-elicitation-finish.md`".

## 0. Read first

- `guide(name="model-inputs")` — how one part is answered.
- `guide(name="elicitation-orchestration")` — ownership, review, accept, reopen, estimate.
- This file.

## 1. State on 2026-11-09 — verify before touching anything

The interview lives on **2026-11-05**, not on the workspace date 2026-10-29. Parts are stored
per date; a new date starts an empty interview.

```
interview(action="progress", as_of="2026-11-05", model_id="expert.elicited.rw")
```

Expected result (637 of 1674 parts saved, one key accepted):

| key | state | answered / 93 |
|---|---|---:|
| `gl.oil_demand` | accepted, hash `fc315ee7d38a451db28faeda9a6a430243bf282faec0a52e7e5d3066abac6f33` | 93 |
| `gl.gas` | in_progress | 85 |
| `gl.financial` | in_progress | 86 |
| `gl.us_rates` | in_progress | 84 |
| `gl.metals` | in_progress | 83 |
| `gl.dollar` | in_progress | 78 |
| `gl.grain` | in_progress | 74 |
| `ru.monetary` | in_progress | 29 |
| `gl.oil` | in_progress, generation 3; the version folder keeps three replaced answers | 25 |
| `ru.inflation`, `ru.inflation_expect`, `ru.fx_risk`, `ru.sanctions_trade`, `ru.sanctions_finance`, `ru.fiscal`, `ru.credit`, `ru.labor`, `ru.capital_controls` | missing | 0 |

No key exists in `model(kind="macro", action="list")` for either expert model: the interview is
the only missing piece. The run of 2026-11-05 stopped at 21:33:35 MSK mid-step; the claims have
expired; `jobs(action="state")` shows no running job.

Each shock is 93 parts: one `law` (frequency, signed size quantiles, `rho`, nonlinearity,
sources) and 92 `response:<variable>:<horizon>` parts (23 variables × horizons 1, 3, 6, 12),
nine quantiles each. Remaining: 1037 parts, about nine thousand numbers.

Evidence: `workspace/elicitation/.steps/expert.elicited.rw/2026-11-05/_journal.jsonl` and
`_session.json`; published answers in `workspace/elicitation/2026-11-05/`.

## 2. Ground rules

- One key, one worker. A worker answers only its own key; the orchestrator never writes a number.
- Date and model id are always `2026-11-05` and `expert.elicited.rw`.
- The answers are the expert's quantitative judgement from the state table and envelope of each
  question. No fixture values, no values copied from earlier experiments, no placeholders, no
  values asked from the human.
- One model answers the whole interview. The model assumes one expert with constant preferences,
  so do not mix gateway models across shocks; record the chosen model in the report. The 637
  parts already saved were answered on 2026-11-05 by agents named `terra-*`; the language model
  behind them is not recorded in the journal or the answer files. Keeping them means the version
  mixes that unrecorded expert with the gateway model chosen now. Default: keep them — saved
  parts are never asked again, and reopening 637 parts costs as much as the whole remaining work —
  and say so in the report. If one recorded expert throughout is required, that is a separate
  decision to raise before wave 1, not a silent reopen.
- Publication is not acceptance. Read `interview(action="audit")` after each publication and
  either `accept` with an economic reason or `reopen` the named steps.
- Never write inside `workspace/elicitation/**` by hand. Only the elicit tools write there.
- The model corpus is not indexed (`workspace/corpus-model/manifest.json` is absent), so every
  `sources` field must be `[]`; the law question already says so.

## 3. Mechanism: gateway workers with only the elicit tool

The answering agents run on the `gateway` provider (registered in the DSH profile:
`http://185.246.65.84:4500/v1`, key in `~/.dsh/.credentials.yaml` under `refs.GATEWAY_API_KEY`,
never printed or written to a file). Gateway models available: `gpt-6.1-sol`, `gpt-6-astra`,
`claude-fable-5-1`, `claude-opus-5-5`, `claude-opus-4-8`, `gemini-4-argon`, `grok-4.7`.

Use the narrow executor, not `workflow` + `agent()`. The prototype is
`workspace/elicitation-behavior/agent_check.py` (see `RESULTS.md` next to it): a gateway worker
gets **only** the real `elicit` tool schema from `FastMCP.Client.list_tools`, and the runner
executes every call with `Client.call_tool`. No shell, no file tools — that removes the failure
class of the 2026-11-05 session, where agents with full tools copied fixtures or fought the shell.
`workflow` agents inherit the whole DSH toolset and each costs a full context; do not use them
for the numbers.

Runner shape (write `workspace/elicitation-behavior/finish_runner.py`, Python from `.venv`):

```python
from mst import store
from mst.mcp import build
from fastmcp import Client

server = build(store.Workspace(r"...\workspace"))
# one server, one Client, N concurrent workers; the SDK serializes writes with a lock
```

Worker loop for its key `K` and agent name `A`:

1. The orchestrator has already called
   `interview(action="restart", key=K, agent=A, as_of="2026-11-05", model_id="expert.elicited.rw")`
   for a partial key, or `interview(action="claim", ...)` for an empty one. The runner itself
   does no claiming.
2. Runner calls `elicit(action="next", as_of="2026-11-05", model_id="expert.elicited.rw", key=K, agent=A)`
   and pins `key` and `agent` on every such call, whatever the model writes. This fixes the
   2026-11-05 failure where the agent omitted `key` and the server silently started with `gl.oil`.
3. It sends the question to the gateway and executes the model's tool call. Only `action` in
   `{"answer"}` is accepted; any other action, a ticket that is not the current one, or a payload
   outside the schema is rejected, and the model is told to call again with the current
   `answer_call` unchanged plus `payload`.
4. The tool result (which is the next question, or `published`, or `complete`) goes back to the
   model as the tool message. Keep the context bounded: system prompt, the initial user message,
   and only the last assistant `tool_calls` message with its tool result. Do not resend the whole
   transcript.
5. Stop when the result has `complete: true` or `published`. Turn cap 130 per key, 60 minutes
   per key; save `trace-<key>.json` (each request, reply, usage) and append one summary line to
   `workspace/elicitation-behavior/finish-summary.jsonl`.

Gateway prompt (English, model-facing):

```text
You are the expert answering an interview for a macro model. The only tool is `elicit`.
The tool result is your next question: read its text, schema and envelope, judge the numbers
yourself from the state table, and answer with `action="answer"`, the exact `ticket` from
`answer_call` (copy it unchanged) and `payload` as the answer object itself, not a JSON string.

Rules: no placeholders; do not ask the human for numbers; use the units stated in the question
(log100 for log variables, percentage points for rates, differences not levels); q5 <= q50 <= q95;
a zero signed size has three zero response quantiles; keep earlier horizons of the same variable
consistent; distinguish the measured move of the affected price from its causal contribution to
other variables; an individual response may be proportional, but the full shock answer needs a
justified departure, so the `nonlinearity` field must say where and why; `sources` must be []
because the model corpus is not indexed here. Do not estimate, do not write files, do not stop
before the tool returns `complete=true`.
```

Gateway call: `POST {baseURL}/chat/completions`, `Authorization: Bearer <GATEWAY_API_KEY>`,
`model` set to the chosen id, and `tools` holding one entry built from the real `elicit` schema:
`{"type": "function", "function": {"name": "elicit", "description": <tool description>,
"parameters": <tool inputSchema>}}`. Retries: a 429 or a timeout is retried with backoff 5, 15,
45 s, three attempts, then the worker parks for 10 minutes; record it and continue the others.
On 2026-11-09 both `gpt-6-astra` and `gemini-4-argon` answered a direct probe; `gpt-6-astra`
returned 429 on the first attempt.

## 4. Preflight (all before wave 1)

1. `interview(action="progress", ...)` matches the table in section 1. If it does not, stop and
   report the difference; the state is the source of truth, not this file.
2. Gateway probe with the chosen model: one request `Reply with exactly one word: ok`. No text —
   do not start; report the ids tried.
3. Smoke test on a copy: copy `workspace` to `workspace/elicitation-behavior/finish-smoke/`,
   point the runner at the copy, answer two parts of one key, stop, verify both parts are saved
   and the question advanced. Delete the copy. Only then point the runner at the real workspace.
4. One writer at a time on the real workspace: during a wave the runner owns the writes; the
   orchestrator uses only `interview(action="progress")` for polling. Claims, accepts and reopens
   happen between waves, when the runner is stopped.

## 5. Waves

Wave 1 — close the four nearly finished keys: `gl.gas` (8), `gl.financial` (7), `gl.us_rates` (9),
`gl.metals` (10) = 34 answers; four workers. This is cheap and gives four publications to audit
before spending the rest.

Review wave 1 (section 6), then:

Wave 2a — `gl.dollar` (15), `gl.grain` (19), `gl.oil` (68), `ru.monetary` (64) = 166 answers.

Wave 2b — the nine empty keys, 837 answers, in two batches of four and five workers.

At the pace of the 2026-11-05 session (642 answers in 12 minutes with eighteen agents) 34 answers
is minutes; 837 with five or six workers is roughly an hour of wall clock. The real cost is
tokens: each turn resends the system prompt and one question, of the order of 3–8 thousand input
tokens per answered part, so the remaining work is several million input tokens. Report usage
from the traces.

## 6. Review

For every published key: `interview(action="audit", key=K, as_of=..., model_id=...)`. Codes:
`same_cell`, `flat_medians`, `zero_width`, `large_first`, `copied_reference`, `same_shock`, `rule`.

Read economically, not mechanically:

- a median profile pasted across unrelated variables, or one template across six variables, is a
  reopen of the whole key — patch cells would keep the copied transmission inside the model
  (exactly the reason `gl.oil` was reopened on 2026-11-05);
- `same_cell` between two quotes of the same currency (`ru.usdrub`, `ru.cnyrub`) is a real defect,
  not a style issue;
- `zero_width` on a size node that is not zero, `large_first` (a first-month response larger than
  the later ones for a persistent shock), or a `rule` finding means read the named rule first;
- `copied_reference` usually means the model copied the anchor instead of judging the variable.

Accept: `interview(action="accept", key=K, agent="lead", reason=<economic reason>, as_of=...,
model_id=...)`. Reopen: `interview(action="reopen", key=K, steps=[<step ids>], agent="lead",
reason=<what is wrong>, ...)` and hand the key back to a gateway worker with the same agent name.

An answer without findings is accepted automatically; nothing to do for it.

## 7. Estimate and the residual gate

Only when all eighteen keys are published and accepted:

```
model(kind="macro", action="estimate", model_id="expert.elicited.rw", as_of="2026-11-05")
jobs(action="wait", id=<job>)
jobs(action="log", id=<job>)
```

Success: a version in `model(kind="macro", action="list")` with `model=expert.elicited.rw`,
`shocks` = the eighteen ids plus `residual`, a non-empty `months` panel from `history_start`
(`2012-01` in `spec.toml`), and a `base` path that is finite.
Check `model(kind="macro", action="base", version=<hash>, horizon=12)` for finite values and no
explosion, and read `fit_misses` in the version metadata.

The known refusal is `ResidualNotAdmitted`: the answers attribute more movement to the shocks
than history shows (`σ_history − Σ_shocks` not positive semidefinite; the message names the
variable, e.g. `ru.housing_price`). This is a real gate, not a formality. Do not edit the
covariance and do not retry unchanged. Reopen the steps of the shocks whose response profile
dominates that variable, with a reason that names the over-attribution, re-answer, and estimate
again. Two failed diagnoses are a stop condition: report the variable, the shocks and the
measured over-attribution instead of guessing a third time.

## 8. Market variant — out of scope here

`expert.elicited.market` needs the model corpus (`workspace/corpus-model/manifest.json`), which
does not exist; the index job of 2026-11-05 failed with `FileNotFoundError`, and the `market`
key cannot be answered without it. Published shock answers are shared, so finishing the rw
interview leaves market with seventeen shocks and the market key missing. Do not start market
in the same session; it is a separate decision with its own corpus work.

## 9. Report

Leave in `workspace/elicitation-behavior/`:

- `finish-summary.jsonl` — one line per key: worker, model, saved parts, errors, tokens, stop reason;
- `trace-<key>.json` per key;
- `finish-report.md` — keys published and accepted with hashes, what was reopened and why, the
  gateway model used, the estimate job id and version hash, the gate message if any, timings and
  token totals.

Report the same in the session: the version (or the exact refusal), the keys still missing, and
the model that answered. Do not claim economic validity of the interview beyond what the audit
and the residual gate checked.

## 10. Stop conditions

- Progress differs from section 1 in a way the journal does not explain.
- The gateway gives no answer after the probe retries, or 429 persists for twenty minutes.
- A worker loops on the same error three times, or a ticket keeps being rejected.
- The same key is reopened twice more with the same finding (the model cannot answer it as asked;
  report it instead of reopening a third time).
- `ResidualNotAdmitted` twice after two different diagnoses.
- Any temptation to fill a number by hand, copy a value from a fixture, or silence a check.
