"""Opt-in bounded gateway experiment; never writes production elicitation answers.

Set GATEWAY_API_KEY and GATEWAY_BASE_URL. This is a benchmark, not an LLM model plugin.
Every mode asks the same law and eight response questions with the same model and tool.
"""

import asyncio
import json
import os
import shutil
from pathlib import Path

import httpx
from fastmcp import Client

from mst import store
from mst.mcp import build

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


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


async def gateway(http, schema, model, messages):
    for retry in range(3):
        try:
            response = await http.post(
                os.environ["GATEWAY_BASE_URL"].rstrip("/") + "/chat/completions",
                headers={"Authorization": "Bearer " + os.environ["GATEWAY_API_KEY"]},
                json={"model": model, "messages": messages, "tools": [schema],
                      "tool_choice": {"type": "function", "function": {"name": "elicit"}},
                      "max_tokens": 1600},
            )
            if response.status_code == 200:
                return response
            if response.status_code not in {429, 502, 503, 504}:
                raise RuntimeError(f"gateway HTTP {response.status_code}")
            print("gateway transient HTTP", response.status_code, flush=True)
        except httpx.TimeoutException:
            print("gateway timeout; usage unavailable for interrupted request", flush=True)
        if retry < 2:
            await asyncio.sleep(15 * (retry + 1))
    raise RuntimeError("gateway unavailable after three bounded retries; resume later")


async def answer_question(http, client, schema, model, question, messages, trace):
    for attempt in range(3):
        response = await gateway(http, schema, model, messages)
        body = response.json()
        message = body["choices"][0]["message"]
        messages.append(message)
        calls = message.get("tool_calls") or []
        if len(calls) != 1:
            raise ValueError("expected exactly one tool call")
        call = calls[0]
        arguments = json.loads(call["function"]["arguments"])
        if (arguments.get("action") != "answer"
                or arguments.get("ticket") != question["answer_call"]["ticket"]):
            result = {"error": "Copy answer_call and answer ONLY the current ticket."}
            failed = True
        else:
            reply = await client.call_tool("elicit", arguments, raise_on_error=False)
            failed = reply.is_error
            result = reply.data if not failed else {"error": [c.text for c in reply.content]}
        trace.append({"question": question["step"], "attempt": attempt,
                      "message_count": len(messages) - 1,
                      "usage": body.get("usage", {}), "request": arguments,
                      "is_error": failed, "result": result})
        if not failed:
            return result, call["id"]
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": compact(result)})
    return None, None


def reset_context(name, previous_id, variable, previous_variable):
    """Only the lifetime of the conversation changes; every horizon is still asked."""
    return name == "bounded" or previous_id is None or (
        name == "per-variable" and variable != previous_variable
    )


async def mode(root, source, name, model, http, law=None):
    ws = store.Workspace(root / name)
    if not ws.series.exists():
        shutil.copytree(source / "series", ws.series)
    trace_path = root / (name + "-trace.json")
    trace = json.loads(trace_path.read_text(encoding="utf-8")) if trace_path.exists() else []
    checkpoint_path = root / (name + "-checkpoint.json")
    checkpoint = store.read_json(checkpoint_path)
    messages = checkpoint["messages"] if checkpoint else [{"role": "system", "content": SYSTEM}]
    previous_id = checkpoint["previous_id"] if checkpoint else None
    async with Client(build(ws)) as client:
        tool = next(t for t in await client.list_tools() if t.name == "elicit")
        schema = {"type": "function", "function": {
            "name": "elicit", "description": tool.description, "parameters": tool.inputSchema}}
        q = (await client.call_tool("elicit", {
            "action": "next", "model_id": "expert.elicited.rw", "as_of": "2026-11-05",
            "key": "gl.oil",
        })).data
        if law is not None and q["step"] == "law":
            # Controlled shared law: it was elicited, not generated by this executor.
            q = (await client.call_tool("elicit", {**q["answer_call"], "payload": law})).data
        previous_variable = checkpoint["variable"] if checkpoint else None
        accepted = {row["question"] for row in trace if not row["is_error"]}
        limit = 8 if law is not None else 9
        for index in range(len(accepted), limit):
            variable = q["step"].rsplit(":", 1)[0]
            reset = reset_context(name, previous_id, variable, previous_variable)
            if reset:
                messages = [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": compact(q)}]
            else:
                messages.append({"role": "tool", "tool_call_id": previous_id,
                                 "content": compact(q)})
            if name == "bounded":
                assert len(messages) == 2
            previous_variable = variable
            answered, previous_id = await answer_question(
                http, client, schema, model, q, messages, trace
            )
            trace_path.write_text(
                json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
            if answered is None:
                print(name, "stopped after three refused answers", flush=True)
                break
            q = answered
            store.write_json(checkpoint_path, {"messages": messages,
                             "previous_id": previous_id, "variable": previous_variable})
            print(name, index + 1, "saved", q.get("progress"), flush=True)
    saved = store.read_json(ws.elicitation / ".steps/expert.elicited.rw/2026-11-05/gl.oil.json")
    calls = trace if law is not None else [t for t in trace if t["question"] != "law"]
    summary = {"response_questions": len({t["question"] for t in calls}),
               "accepted_questions": len({t["question"] for t in calls if not t["is_error"]}),
               "calls": len(calls), "errors": sum(t["is_error"] for t in calls),
               "prompt_tokens": sum(t["usage"].get("prompt_tokens", 0) for t in calls),
               "completion_tokens": sum(t["usage"].get("completion_tokens", 0) for t in calls),
               "cached_tokens": sum(t["usage"].get("prompt_tokens_details", {}).get(
                   "cached_tokens", 0) for t in calls)}
    return saved, summary


async def main():
    root = Path(os.environ["ELICIT_BENCH_ROOT"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = Path(os.environ.get("ELICIT_BENCH_SOURCE", "workspace"))
    model = os.environ.get("ELICIT_BENCH_MODEL", "gpt-6-astra")
    async with httpx.AsyncClient(timeout=120) as http:
        baseline, first = await mode(root, source, "history", model, http)
        bounded, second = await mode(root, source, "bounded", model, http, baseline["law"])
        local, third = await mode(root, source, "per-variable", model, http, baseline["law"])
    steps = set(baseline) - {"law"}
    if steps != set(bounded) - {"law"} or steps != set(local) - {"law"}:
        print("Incomplete modes: compare only matched question prefixes", flush=True)
    result = {"model": "gateway/" + model, "shared_law": baseline["law"],
              "history": first, "bounded": second, "per_variable": third,
              "input_reduction": 1 - second["prompt_tokens"] / first["prompt_tokens"],
              "local_input_reduction": 1 - third["prompt_tokens"] / first["prompt_tokens"],
              "question_ids": sorted(steps)}
    (root / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(compact(result), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
