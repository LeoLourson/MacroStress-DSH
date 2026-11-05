"""Small elicitation steps that assemble the unchanged complete shock answer."""

import math
from typing import Annotated

from pydantic import Field

from mst.contracts import Contract, Identifier, NonNegative, Probability

from .answers import HORIZONS, NODES, Quantiles, _check_fragments, _state_of, check, question
from .model import MARKET, load_shocks


class ShockLaw(Contract):
    """The shock law and its explanation, without responses or a repeated shock id."""

    frequency: NonNegative
    size: Quantiles
    rho: Probability
    nonlinearity: Annotated[str, Field(min_length=1)]
    sources: list[Identifier]


class Response(Contract):
    """Response quantiles at each of the three answered shock size quantiles."""

    at_q5: Quantiles
    at_q50: Quantiles
    at_q95: Quantiles


def _coordinates(context):
    return [
        (f"response:{variable.id}:{horizon}", variable, horizon)
        for variable in context.registry.variables for horizon in HORIZONS
    ]


def input_steps(context, key: str) -> list[dict]:
    """Ask one shock law, then one variable and horizon at a time; keep market whole."""
    if key == MARKET:
        asked = question(context, key)
        return [{"id": "answer", "text": asked["text"], "schema": asked["schema"]}]
    shocks = {shock.id: shock for shock in load_shocks()}
    if key not in shocks:
        raise KeyError(f"the model declares no input {key!r}")
    shock = shocks[key]
    steps = [{"id": "law", "text": _law_question(context, shock),
              "schema": ShockLaw.model_json_schema()}]
    schema = Response.model_json_schema()
    for step_id, variable, horizon in _coordinates(context):
        steps.append({"id": step_id,
                      "text": _response_question(context, shock, variable, horizon),
                      "schema": schema})
    return steps


def _law_question(context, shock) -> str:
    return f"""## The law of shock {shock.id}

{shock.meaning}. Answer for {context.as_of.isoformat()}.

- `frequency`: how many times a year the shock occurs, a non-negative number;
- `size`: the 5, 50 and 95 % quantiles of signed size conditional on occurrence,
  in {shock.size_unit}. A shock that goes both ways may have a negative q5 and positive q95;
- `rho` in [0, 1]: how consistently different variables' responses to one occurrence
  come out stronger or weaker than their medians. At 1 they share one quantile level;
  at 0 the levels are uncorrelated, but joint extremes remain more frequent than under
  independence;
- `nonlinearity`: where responses depart from proportionality to shock size and why.
  The complete answer is refused if all responses are proportional to size;
- `sources`: ids actually returned by a search of the model corpus, never names of reports,
  URLs or invented ids. For your own judgement return `sources: []`.
  {('The model corpus is not indexed on this workspace: sources must be [].'
     if context.fragment is None else
     'Search corpus(action="search", corpus="model", as_of='
     + context.as_of.isoformat() + ', query=...).')}

Return only these five fields, with no shock id or response table. Quantiles objects
have required numeric `q5`, `q50`, `q95` with q5 <= q50 <= q95. Subsequent steps ask
for responses at exactly these three size quantiles; no response is inferred from size.

The variables, their units, model scales and last values known on the date:

{_state_of(context)}"""


def _response_question(context, shock, variable, horizon: int) -> str:
    observed = context.store.as_of(variable.id, context.as_of)
    last = observed[-1] if observed else None
    state = f"{last.value:g} ({last.month.isoformat()})" if last else "none"
    scale = (
        "log100 = 100 ln X; response differences are approximately percentage changes"
        if variable.transform == "log100"
        else "identity; response in percentage points, not relative percent or basis points"
        if variable.unit in {"%", "п. п."}
        else f"identity; the response is a difference in {variable.unit}, not a level"
    )
    converting = shock.size_scale == "percent_change" and variable.transform == "log100"
    conversion = (
        "The law's size is a percentage change of a price, and this variable is in log100. "
        "The size converted to this variable's scale is in `size_in_response_scale` of this "
        "envelope: a size of -12 % is -12.78 log100. The response is the effect of that "
        "size, not the size itself: a pass-through of 1 % is 1 log100 point, not 1 % of the "
        "response.\n"
        if converting
        else ""
    )
    return f"""## Response of {variable.id}, horizon {horizon}

Shock {shock.id}: {shock.meaning}. Signed shock size is measured in {shock.size_unit}.
Answer for {context.as_of.isoformat()} about this shock's response, not a new shock law.

Variable: {variable.id}; observed unit: {variable.unit}; model scale: {scale}.
Last known state in the observed unit: {state}.
Horizon: {horizon} months. The month of the shock counts as the first: horizon 1 is
its response in the month of the shock itself. Give the response relative to the
same path without this occurrence, in the variable's model scale, not its total level.

{conversion}Return exactly three quantiles objects:
- `at_q5`: response conditional on the signed shock size equal to the law's size.q5;
- `at_q50`: response conditional on the signed shock size equal to the law's size.q50;
- `at_q95`: response conditional on the signed shock size equal to the law's size.q95.

Each object has numeric `q5`, `q50`, `q95` (q5 <= q50 <= q95), describing uncertainty
of the response at that fixed size. These are response quantiles, not shock sizes.
Use the size values supplied with the answered law. Judge all nine response numbers;
do not infer missing responses, rescale a single response or assume proportionality.
This is the level effect of ONE occurrence at that horizon, not an additional monthly
change or a forecast including future shocks. A zero signed size has all three response
quantiles equal to zero in this model. Keep the nine answers even for a zero size.
For log100, a causal price change of s percent is 100*ln(1+s/100), not s log points
for large s. If size describes the directly affected price, distinguish its measured
monthly change from its causal contribution relative to the no-shock path; these are
not automatically identical. For rates, 1 percentage point = 100 basis points.
Non-proportional responses are allowed, but do not invent curvature merely to pass a
validator: an individual response can be proportional; the full answer needs a justified
departure somewhere, as explained in the law's `nonlinearity`."""


def step_context(context, key: str, step: str, parts: dict[str, dict]) -> dict:
    """Retain the law, earlier horizons and one transmission anchor, not the whole interview."""
    if not step.startswith("response:") or "law" not in parts:
        return {}
    _, variable_id, horizon = step.split(":")
    variable = next(row for row in context.registry.variables if row.id == variable_id)
    shock = next(row for row in load_shocks() if row.id == key)
    earlier = {
        str(h): parts[f"response:{variable_id}:{h}"]
        for h in HORIZONS if h < int(horizon) and f"response:{variable_id}:{h}" in parts
    }
    first = context.registry.variables[0].id
    reference = f"response:{first}:{horizon}"
    return {
        "shock_law": parts["law"],
        "size_in_response_scale": _sizes_in_scale(shock, parts["law"]["size"], variable),
        "zero_size_nodes": [
            node for node in NODES if parts["law"]["size"][node] == 0.0
        ],
        "earlier_horizons": earlier,
        "reference_response": {reference: parts[reference]}
        if first != variable_id and reference in parts else {},
        "context_rule": "Earlier horizons are saved judgements for this variable, not new "
        "questions. Preserve their units and explain dynamics consistently. The reference "
        "response is a transmission cross-check, not a value to copy: a cell equal to it is "
        "flagged as copied by the review. The nodes in `zero_size_nodes` have a zero size, "
        "so their three response quantiles are zero. Do not change saved answers.",
    }


def _sizes_in_scale(shock, size: dict, variable) -> dict:
    """The answered sizes in the variable's model scale, when the units differ."""
    if shock.size_scale != "percent_change" or variable.transform != "log100":
        return {}
    return {
        node: {
            "percent": size[node],
            "log100": 100.0 * math.log1p(size[node] / 100.0),
        }
        for node in NODES
    }


def check_part(context, key: str, step: str, payload: dict) -> dict:
    """Check only the supplied part, without fabricating missing responses."""
    if key == MARKET:
        return check(context, key, payload)
    if step == "law":
        law = ShockLaw.model_validate(payload)
        shock = next(row for row in load_shocks() if row.id == key)
        if shock.size_scale == "percent_change" and law.size.q5 <= -100.0:
            raise ValueError("size.q5: a percentage change of a price must be greater than -100")
        _check_fragments(context, set(law.sources), "sources")
        return law.model_dump(mode="json")
    if step not in {name for name, _, _ in _coordinates(context)}:
        raise ValueError("unknown response step")
    return Response.model_validate(payload).model_dump(mode="json")


def _require_parts(parts: dict[str, dict], expected: set[str]) -> None:
    missing = sorted(expected - parts.keys())
    extra = sorted(parts.keys() - expected)
    if missing or extra:
        raise ValueError(f"parts: missing {missing}; extra {extra}")


def assemble_input(context, key: str, parts: dict[str, dict]) -> dict:
    """Repack supplied judgements without arithmetic, defaults or source resolution."""
    if key == MARKET:
        _require_parts(parts, {"answer"})
        return parts["answer"]
    if key not in {shock.id for shock in load_shocks()}:
        raise KeyError(f"the model declares no input {key!r}")
    coordinates = _coordinates(context)
    _require_parts(parts, {"law", *(step_id for step_id, _, _ in coordinates)})
    ShockLaw.model_validate(parts["law"])
    responses = {}
    for step_id, variable, horizon in coordinates:
        part = parts[step_id]
        Response.model_validate(part)
        responses.setdefault(variable.id, {})[str(horizon)] = {
            node: part[f"at_{node}"] for node in NODES
        }
    return {"shock": key, **parts["law"], "responses": responses}


def input_parts(context, key: str, payload: dict) -> dict[str, dict]:
    """Split a published answer back into the parts the interview asked for."""
    if key == MARKET:
        return {"answer": payload}
    if key not in {shock.id for shock in load_shocks()}:
        raise KeyError(f"the model declares no input {key!r}")
    law = {name: payload[name] for name in ShockLaw.model_fields}
    parts = {"law": law}
    for variable, by_horizon in payload["responses"].items():
        for horizon in HORIZONS:
            parts[f"response:{variable}:{horizon}"] = {
                f"at_{node}": by_horizon[str(horizon)][node] for node in NODES
            }
    return parts
