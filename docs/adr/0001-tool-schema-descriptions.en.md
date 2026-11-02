# ADR 0001: Tool schemas describe every parameter and annotate every tool

Date: 2026-11-02
Status: accepted

## Context

The JSON Schema of an MCP tool is the part of the contract an agent and an external scanner
read first. The eight tools are multiplexors that branch on `action`, and their parameters
carried types but no descriptions: a client had to guess which argument an action needs and
what it means. Scanners that rate tool definitions saw 0 % of parameters described and no
MCP annotations at all.

## Decision

Every parameter of every tool carries a `Field(description=...)` — what the field is for,
its format, and what it interacts with. The description of an `action` field names every
action the dispatch knows. Every tool declares `ToolAnnotations`: `guide` is a pure read
(`readOnlyHint`), `series` and `corpus` touch the network (`openWorldHint`), `portfolio`
and `scenario` overwrite or delete a file and `jobs cancel` stops a job (`destructiveHint`),
`run start` spawns a new run each call (no
`idempotentHint`). `tests/test_mcp_schema.py` holds the coverage at 100 % of parameters and
fails when a new action is not named in its field description.

## Consequences

A client no longer guesses valid argument combinations from the tool docstring alone, and
can tell a read from a mutation before calling. The hints describe the tool as a whole, not
each action, which is the price of the multiplexor design: an action more destructive than
its tool's hint must say so in its own field description, as `portfolio save` does.
