"""elicitation-context: the tokens an expert interview spends as the kept history changes.

An opt-in experiment, run by hand: it asks a language model behind an OpenAI-compatible
gateway the same law question and the same response questions of one shock through the real
MCP server, in three modes that differ only in how much of the conversation each request
keeps. Every mode answers in its own workspace under ``root``, so the production interview is
never written; the law the first mode elicited is given to the others, so the comparison does
not mix the length of the history with another shock size.

The gateway address and key come from ``GATEWAY_BASE_URL`` and ``GATEWAY_API_KEY`` only and
never reach a trace. ``describe`` reads the saved answers back: the medians of the responses
and the cells whose quantiles collapsed — a description for an expert, not an acceptance rule.
"""

import asyncio
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from mst import store

MODES = ("history", "bounded", "per-variable")
"""``history`` keeps the whole conversation, ``bounded`` starts every question afresh,
``per-variable`` keeps the horizons of one variable and starts afresh at the next one."""
RETRIED = {429, 502, 503, 504}
"""Gateway answers that are tried again, at most three times with a growing pause."""
ATTEMPTS = 3

SYSTEM = (
    "You are a macroeconomic expert, not a programmer. Answer the single current question "
    "with your own quantitative judgement through elicit. No placeholders, invented citations "
    "or questions to a human. Copy the supplied ticket. Use the schema exactly. "
    "A zero size gives zero response. Response is the causal level difference at the horizon, "
    "not an extra monthly change. Distinguish measured price changes from causal contributions. "
    "Respect log100 versus ordinary percentages and rate percentage points. "
    "Earlier horizons and the saved law are supplied explicitly when needed. "
    "Do not repeat already saved answers."
)


@dataclass(frozen=True)
class Experiment:
    """What one experiment asks and where it keeps its workspaces and traces."""

    root: Path
    """A folder of its own: a workspace per mode, the traces, checkpoints and the summary."""
    source: Path = Path("workspace")
    """The workspace whose series every mode copies."""
    gateway_model: str = "gpt-6-astra"
    model_id: str = "expert.elicited.rw"
    as_of: str = "2026-11-05"
    key: str = "gl.oil"
    responses: int = 8
    """The response questions after the law question."""


def compact(value) -> str:
    """JSON without spaces: what the conversation carries."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def reset_context(mode: str, previous_id: str | None, variable: str, previous: str | None) -> bool:
    """Whether the next question starts a new conversation; every horizon is still asked."""
    return (
        mode == "bounded"
        or previous_id is None
        or (mode == "per-variable" and variable != previous)
    )


async def _gateway(http, schema: dict, model: str, messages: list[dict]):
    """One chat completion that must call ``elicit``; transient failures are tried again."""
    import httpx

    for attempt in range(ATTEMPTS):
        try:
            response = await http.post(
                os.environ["GATEWAY_BASE_URL"].rstrip("/") + "/chat/completions",
                headers={"Authorization": "Bearer " + os.environ["GATEWAY_API_KEY"]},
                json={
                    "model": model,
                    "messages": messages,
                    "tools": [schema],
                    "tool_choice": {"type": "function", "function": {"name": "elicit"}},
                    "max_tokens": 1600,
                },
            )
            if response.status_code == 200:
                return response
            if response.status_code not in RETRIED:
                raise RuntimeError(f"gateway HTTP {response.status_code}")
            print("gateway transient HTTP", response.status_code, flush=True)
        except httpx.TimeoutException:
            print("gateway timeout; usage unavailable for the interrupted request", flush=True)
        if attempt < ATTEMPTS - 1:
            await asyncio.sleep(15 * (attempt + 1))
    raise RuntimeError(f"gateway unavailable after {ATTEMPTS} attempts; resume later")


async def _answer(http, client, schema, model, question, messages, trace):
    """The next question once the model answered this one, and the id of its tool call."""
    for attempt in range(ATTEMPTS):
        body = (await _gateway(http, schema, model, messages)).json()
        message = body["choices"][0]["message"]
        messages.append(message)
        calls = message.get("tool_calls") or []
        if len(calls) != 1:
            raise ValueError("expected exactly one tool call")
        call = calls[0]
        arguments = json.loads(call["function"]["arguments"])
        if (
            arguments.get("action") != "answer"
            or arguments.get("ticket") != question["answer_call"]["ticket"]
        ):
            result, failed = {"error": "Copy answer_call and answer ONLY the current ticket."}, True
        else:
            reply = await client.call_tool("elicit", arguments, raise_on_error=False)
            failed = reply.is_error
            result = reply.data if not failed else {"error": [c.text for c in reply.content]}
        trace.append(
            {
                "question": question["step"],
                "attempt": attempt,
                "message_count": len(messages) - 1,
                "usage": body.get("usage", {}),
                "request": arguments,
                "is_error": failed,
                "result": result,
            }
        )
        if not failed:
            return result, call["id"]
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": compact(result)})
    return None, None


def usage(calls: list[dict]) -> dict:
    """The questions, calls, errors and tokens of a mode, from its trace."""
    return {
        "response_questions": len({t["question"] for t in calls}),
        "accepted_questions": len({t["question"] for t in calls if not t["is_error"]}),
        "calls": len(calls),
        "errors": sum(t["is_error"] for t in calls),
        "prompt_tokens": sum(t["usage"].get("prompt_tokens", 0) for t in calls),
        "completion_tokens": sum(t["usage"].get("completion_tokens", 0) for t in calls),
        "cached_tokens": sum(
            t["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0) for t in calls
        ),
    }


async def _mode(experiment: Experiment, name: str, http, law: dict | None = None):
    """The saved answers and the usage of one mode; a stopped mode resumes from its files."""
    from fastmcp import Client

    from mst.mcp import build

    root = experiment.root
    ws = store.Workspace(root / name)
    if not ws.series.exists():
        shutil.copytree(experiment.source / "series", ws.series)
    trace_path = root / f"{name}-trace.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8")) if trace_path.exists() else []
    checkpoint_path = root / f"{name}-checkpoint.json"
    checkpoint = store.read_json(checkpoint_path)
    messages = checkpoint["messages"] if checkpoint else [{"role": "system", "content": SYSTEM}]
    previous_id = checkpoint["previous_id"] if checkpoint else None
    previous = checkpoint["variable"] if checkpoint else None
    async with Client(build(ws)) as client:
        tool = next(t for t in await client.list_tools() if t.name == "elicit")
        schema = {
            "type": "function",
            "function": {
                "name": "elicit",
                "description": tool.description,
                "parameters": tool.inputSchema,
            },
        }
        question = (
            await client.call_tool(
                "elicit",
                {
                    "action": "next",
                    "model_id": experiment.model_id,
                    "as_of": experiment.as_of,
                    "key": experiment.key,
                },
            )
        ).data
        if law is not None and question["step"] == "law":
            # The shared law was elicited by the first mode, not made up by this executor.
            question = (
                await client.call_tool("elicit", {**question["answer_call"], "payload": law})
            ).data
        accepted = {row["question"] for row in trace if not row["is_error"]}
        limit = experiment.responses + (0 if law is not None else 1)
        for index in range(len(accepted), limit):
            variable = question["step"].rsplit(":", 1)[0]
            if reset_context(name, previous_id, variable, previous):
                messages = [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": compact(question)},
                ]
            else:
                messages.append(
                    {"role": "tool", "tool_call_id": previous_id, "content": compact(question)}
                )
            previous = variable
            answered, previous_id = await _answer(
                http, client, schema, experiment.gateway_model, question, messages, trace
            )
            trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
            if answered is None:
                print(name, f"stopped after {ATTEMPTS} refused answers", flush=True)
                break
            question = answered
            store.write_json(
                checkpoint_path,
                {"messages": messages, "previous_id": previous_id, "variable": previous},
            )
            print(name, index + 1, "saved", question.get("progress"), flush=True)
    saved = store.read_json(
        ws.elicitation
        / ".steps"
        / experiment.model_id
        / experiment.as_of
        / f"{experiment.key}.json"
    )
    calls = trace if law is not None else [t for t in trace if t["question"] != "law"]
    return saved, usage(calls)


async def run(experiment: Experiment) -> dict:
    """The three modes in turn, on one gateway limit, and the summary they compare to."""
    import httpx

    experiment.root.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=120) as http:
        baseline, first = await _mode(experiment, "history", http)
        bounded, second = await _mode(experiment, "bounded", http, baseline["law"])
        local, third = await _mode(experiment, "per-variable", http, baseline["law"])
    steps = set(baseline) - {"law"}
    complete = steps == set(bounded) - {"law"} == set(local) - {"law"}
    if not complete:
        print("Incomplete modes: compare only matched question prefixes", flush=True)
    result = {
        "model": "gateway/" + experiment.gateway_model,
        "shared_law": baseline["law"],
        "history": first,
        "bounded": second,
        "per_variable": third,
        "input_reduction": 1 - second["prompt_tokens"] / first["prompt_tokens"],
        "local_input_reduction": 1 - third["prompt_tokens"] / first["prompt_tokens"],
        "question_ids": sorted(steps),
        "complete": complete,
    }
    store.write_json(experiment.root / "summary.json", result)
    return result


def describe(parts: dict) -> dict:
    """The medians of the saved responses and the cells whose quantiles collapsed."""
    sizes = parts["law"]["size"]
    oil, rate = {}, {}
    for horizon in (1, 3, 6, 12):
        for variable, target in (("gl.brent", oil), ("gl.fed_rate", rate)):
            step = f"response:{variable}:{horizon}"
            if step in parts:
                target[str(horizon)] = [
                    parts[step][f"at_{node}"]["q50"] for node in ("q5", "q50", "q95")
                ]
    reference = [
        100 * math.log1p(sizes[node] / 100) if sizes[node] > -100 else None
        for node in ("q5", "q50", "q95")
    ]
    return {
        "own_impact_reference_if_size_is_causal_contribution": reference,
        "brent_median_log100": oil,
        "brent_median_percent": {
            h: [100 * math.expm1(x / 100) for x in values] for h, values in oil.items()
        },
        "fed_rate_median_percentage_points": rate,
        "reference_is_not_hard_validation": "Measured total price movement and isolated causal "
        "contribution are not the same conditioning object; definition remains unresolved.",
        "uncertainty_collapsed_cells": [
            f"{step}/{node}"
            for step, answer in parts.items()
            if step.startswith("response:")
            for node, quantiles in answer.items()
            if quantiles["q5"] == quantiles["q50"] == quantiles["q95"]
        ],
    }
