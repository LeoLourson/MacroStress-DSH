"""The schemas a client sees: every parameter described, every tool annotated.

The JSON Schema of a tool is the part of the contract an agent and an external scanner
read first; a field without a description is a field the caller has to guess. These tests
keep the coverage at every parameter of every tool, and keep the description of an
``action`` field naming every action the tool dispatches.
"""

import asyncio

from mst import store
from mst.mcp import build

MINIMUM = 15
"""Characters a description must carry to count as one."""

ACTIONS = {
    "jobs": ("state", "status", "log", "cancel", "wait"),
    "series": ("search", "get", "add", "update"),
    "corpus": ("search", "get", "list", "add", "index", "fetch"),
    "scenario": ("list", "get", "save", "validate", "reverse"),
    "run": ("start", "status", "result", "list"),
    "portfolio": ("list", "get", "save", "delete", "check", "exposure"),
}
"""Every action the tool's dispatch knows; the description of the field must name them all."""


def tools(tmp_path):
    """Every tool of the server, by name."""

    async def read():
        return await build(store.Workspace(tmp_path)).get_tools()

    return asyncio.run(read())


def test_every_parameter_of_every_tool_carries_a_description(tmp_path):
    found = tools(tmp_path)
    missing = [
        f"{name}.{field}"
        for name, tool in sorted(found.items())
        for field, spec in tool.parameters.get("properties", {}).items()
        if len(str(spec.get("description", "")).strip()) < MINIMUM
    ]
    assert not missing, missing


def test_every_tool_declares_the_hints_a_client_reads(tmp_path):
    found = tools(tmp_path)
    assert len(found) == 8, sorted(found)
    problems = []
    for name, tool in sorted(found.items()):
        annotations = tool.annotations
        if annotations is None:
            problems.append(f"{name}: no annotations")
            continue
        for hint in ("readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint"):
            if getattr(annotations, hint) is None:
                problems.append(f"{name}: {hint} is not set")
        if not annotations.title:
            problems.append(f"{name}: no title")
    assert not problems, problems


def test_the_description_of_an_action_field_names_every_action(tmp_path):
    found = tools(tmp_path)
    problems = []
    for name, actions in ACTIONS.items():
        description = found[name].parameters["properties"]["action"]["description"]
        for action in actions:
            if action not in description:
                problems.append(f"{name}: the action field does not name {action}")
    assert not problems, problems
