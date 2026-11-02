# ADR 0003: Plugin printing goes to stderr, not to the stdio channel

Date: 2026-11-03
Status: accepted

## Context

The server speaks MCP over stdio: stdout carries JSON-RPC and nothing else. Estimate and
run jobs are separate processes, but the `scenario`, `portfolio` and `model` tools load and
run plugin code in the server's own process — `validate`, `reverse`, `check`, `exposure`,
`base` and the plugin catalog all import and call plugins in place. A stray `print()` in a
third-party plugin would land in the middle of the JSON-RPC stream and break every
following message of the session.

## Decision

The bodies of those three tools are wrapped in a redirect of `sys.stdout` to `sys.stderr`,
so plugin output reaches the client's log. The redirect is at the Python level on purpose:
the stdio transport takes its own wrapper of `sys.stdout.buffer` at startup, so rebinding
`sys.stdout` during a call cannot touch the channel — not even when the server answers a
concurrent call in the middle of the redirect. An fd-level `dup2` redirect, which also
catches C extensions writing to descriptor 1, was rejected for exactly that case: a
response the transport writes while the redirect holds would land on stderr and be lost.

## Consequences

A plugin author debugging with `print()` no longer breaks the protocol; the output is
visible in the client's log instead. A C extension writing to descriptor 1 directly would
still corrupt the channel — none of the shipped dependencies does, and the estimate and run
jobs, where such extensions work hardest, run in child processes whose stdout is a log
file. The guard covers the three tools as a whole, so a future action that calls plugin
code is covered without thinking about it.
