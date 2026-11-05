---
name: elicitation-orchestration
description: Run the expert elicitation as an orchestrator: one agent per shock, claims, review, reopen
whenToUse: when a model that declares inputs should be estimated from the answers of several agents
---

# Orchestrating an elicitation

The orchestrator owns the process and the review, not the numbers. One shock is one subagent;
the orchestrator reads `progress`, claims a key, starts the agent, reads the findings and
accepts the answer or reopens parts of it. Every action is recorded in the journal of the
interview, so a later reader can see who answered what and why it was accepted.

This instruction names no particular client: a subagent is whatever mechanism of your client
runs a separate agent with the same MCP server.

## 1. Survey before starting

`interview(action="progress", as_of=..., model_id=...)` returns one row per input key: its
state, owner, answered and total parts, published hash and finding count. The states are
`missing`, `in_progress`, `claimed`, `published`, `needs_review` and `accepted`; `counts`
sums them. First obtain the installed id from `model(kind="catalog", action="list")`, then
copy the model id returned by `progress` into subsequent calls.

## 2. One shock, one subagent

For every key that is `missing` or `in_progress`:

1. `interview(action="claim", key=<key>, agent=<short name>, as_of=..., model_id=...)`.
2. Start a subagent with this brief: give it only the key and its agent name, tell it to read
   `guide(name="model-inputs")`, to call `elicit(action="next", as_of=..., model_id=...,
   key=<key>, agent=<name>)` and to answer until the result has `complete=true`. Do not give
   it another key and do not let it answer other keys.
3. Record the key and the agent name. A claim expires after 45 minutes without renewal;
   `progress` shows the owner and the expiry, so a stopped agent does not hide its shock.

Keep the interview out of the parent context: the orchestrator needs the key, the progress and
the findings, not the numbers.

## 3. Review every published answer

A publication is not an acceptance. `interview(action="audit", key=<key>, ...)` lists the
findings of one answer; `audit` without a key lists every published one. The codes are
`same_cell`, `flat_medians`, `zero_width`, `large_first`, `copied_reference`, `same_shock` and
`rule`; each finding names the cells and the message, and a `rule` finding carries the note of
the economic rule it broke.

Judge the findings economically, not mechanically. Then either

- `interview(action="accept", key=<key>, agent=<your name>, reason=<why the findings are
  acceptable>, ...)`, or
- `interview(action="reopen", key=<key>, steps=[<step ids>], agent=<your name>,
  reason=<what is wrong>, ...)`: the published answer is kept as a version, the named parts are
  asked again and the key becomes `in_progress` until it is complete again. Hand the reopened
  parts back to an agent with the brief of section 2.

The answer without findings is accepted automatically, and `accept` is not needed for it.

## 4. Finish

Estimate only when every key is `accepted`:
`model(kind="macro", action="estimate", model_id=..., as_of=...)`, then
`jobs(action="wait", id=...)` and `jobs(action="log", id=...)`. A key that is missing or
`needs_review` keeps the model from being the one the orchestrator approved.

## 5. Do not

- Do not answer the numbers of a key that belongs to a subagent.
- Do not accept without reading the findings: the reason is stored with the acceptance.
- Do not hand two shocks to one subagent or let two agents answer one key.
- Do not ask the human for numbers and do not fill placeholders.
- Do not reopen a key to fix a finding the model itself would refuse: a form error is fixed by
  answering the current question, not by reopening.

## 6. Reading the journal

`interview(action="journal", key=..., limit=..., ...)` returns the last events: `claim`,
`release`, `next`, `answer`, `publish`, `warn`, `accept` and `reopen`, with the agent, the
step, the hash and the number of findings. `interview(action="history", key=...)` lists the
kept versions of a reopened answer with their hashes.
