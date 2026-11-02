# ADR 0002: `jobs wait` holds the call server-side instead of the client polling

Date: 2026-11-03
Status: accepted

## Context

An MCP server cannot push a message to a client on its own: the initiative is always the
client's. A tool that starts a long job returns its id at once, and an agent that waits for
a minutes-long estimation had to loop `jobs(action="status")` and `jobs(action="log")` —
one round-trip per step, each carrying the whole prompt history, or it hit the runtime's
step limit before the job finished.

## Decision

The `jobs` tool gained a `wait` action: the server holds the call until the job ends or
`timeout_sec` (30 by default, 120 at most) passes, then returns the record `status`
returns; `running` tells the caller whether to wait again. The loop sleeps between cheap
file reads, so the event loop keeps answering other calls while one client waits. A fast
job now costs one call, a long one a call per timeout.

A `wait_sec` parameter on every job-starting action was considered and rejected: it would
grow every tool's contract with a flag that changes what the tool returns, while `wait`
after `start` reaches the same round-trip count for fast jobs and keeps each tool's return
shape fixed.

## Consequences

The agent flow for a long job is `start` then `wait` (again, while `running` is true), with
`log` for the output. The server instructions, the `run-report` guide, the tools reference
and the first-session document teach `wait`. A client with a short request timeout sees the
call return at its own timeout, exactly as before; the default 30 s fits under common
client timeouts, and the cap of 120 s keeps a forgotten call from pinning a worker forever.
