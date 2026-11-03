"""The answers of the elicitation and the questions the agent answers.

The expert does not build a model: it answers predefined questions about quantities with
quantiles, and the plugin fixes the form the answers enter (the SHELF protocol, Cooke's
classical model). One question per shock, all sizes of a shock asked together so the
answers are consistent; the market expectations are asked in the same elicitation, because
the series store holds no futures, forwards, OFZ curve or consensus.

``check`` refuses what the estimate cannot read: a malformed answer (the message names the
field), a linear answer (the question is reworded), a market number without a visible
corpus fragment (a market number that is not in the corpus is the expert's judgement, and
judgements enter the model only through the shocks). The normal form ``check`` returns is
what the platform stores and what the estimate reads; its hash enters the model version.
"""

from typing import Annotated

from pydantic import Field, ValidationError, model_validator

from mst.contracts import Contract, Identifier, NonNegative, Number

from .model import MARKET, Shock, load_shocks

#: The months the responses are asked at; the month of the shock itself counts as the first.
HORIZONS = (1, 3, 6, 12)
#: The size nodes of the responses: exactly the quantiles of the size law.
NODES = ("q5", "q50", "q95")

Quantile = Number


class Quantiles(Contract):
    """The 5, 50 and 95 % quantiles of one answered quantity, in order."""

    q5: Quantile
    q50: Quantile
    q95: Quantile

    @model_validator(mode="after")
    def ordered(self):
        if not (self.q5 <= self.q50 <= self.q95):
            raise ValueError("the quantiles must satisfy q5 <= q50 <= q95")
        return self


class ShockAnswer(Contract):
    """The answer about one shock: its law, its responses and where they are not proportional.

    ``frequency`` is how many times a year the shock occurs; ``size`` the quantiles of the
    signed size given that it occurs, in the shock's own unit; ``responses`` the quantiles
    of the response of every variable of the registry 1, 3, 6 and 12 months after the
    shock, at the three sizes the ``size`` quantiles name, in the model scale of each
    variable; ``rho`` how consistently the responses of different variables to one
    occurrence come out stronger or weaker than their medians; ``nonlinearity`` where the
    response departs from proportionality to the size and why; ``sources`` the fragments of
    the model corpus the numbers come from.
    """

    shock: Identifier
    frequency: NonNegative
    size: Quantiles
    responses: dict[str, dict[int, dict[str, Quantiles]]]
    rho: Annotated[Number, Field(ge=0, le=1)]
    nonlinearity: Annotated[str, Field(min_length=1)]
    sources: list[Identifier] = []

    @model_validator(mode="after")
    def shaped(self):
        for variable, by_horizon in self.responses.items():
            if set(by_horizon) != set(HORIZONS):
                raise ValueError(
                    f"responses.{variable}: the horizons must be exactly "
                    f"{', '.join(map(str, HORIZONS))}"
                )
            for horizon, by_node in by_horizon.items():
                if set(by_node) != set(NODES):
                    raise ValueError(
                        f"responses.{variable}.{horizon}: the size nodes must be exactly "
                        f"{', '.join(NODES)}"
                    )
        return self


class MarketPath(Contract):
    """The market expectations of one variable: its path and the fragment it comes from."""

    horizons: dict[int, Quantile]
    fragment: Identifier

    @model_validator(mode="after")
    def shaped(self):
        if set(self.horizons) != set(HORIZONS):
            raise ValueError(
                f"the horizons must be exactly {', '.join(map(str, HORIZONS))}"
            )
        return self


class MarketAnswer(Contract):
    """The market expectations of the date: the value of every variable that has them."""

    values: dict[str, MarketPath] = Field(min_length=1)


def linear(answer: ShockAnswer) -> bool:
    """Whether the responses are proportional to the sizes for every variable and horizon.

    A size node of zero size is skipped; with fewer than two non-zero sizes the response is
    a line through the origin whatever it says, so the answer counts as linear and the
    question is reworded.
    """
    sizes = (answer.size.q5, answer.size.q50, answer.size.q95)
    nodes = [node for node, size in enumerate(sizes) if size != 0.0]
    if len(nodes) < 2:
        return True
    for by_horizon in answer.responses.values():
        for by_node in by_horizon.values():
            for quantile in ("q5", "q50", "q95"):
                values = {node: getattr(by_node[NODES[node]], quantile) for node in nodes}
                for first, second in zip(nodes, nodes[1:], strict=False):
                    left = values[first] * sizes[second]
                    right = values[second] * sizes[first]
                    if abs(left - right) > 1e-6 * max(abs(left), abs(right), 1e-12):
                        return False
    return True


def check(context, key: str, payload: dict) -> dict:
    """The answer of ``key`` in its normal form, or ``ValueError`` naming the field."""
    known = [variable.id for variable in context.registry.variables]
    if key == MARKET:
        answer = _parse(MarketAnswer, payload, "market")
        unknown = sorted(set(answer.values) - set(known))
        if unknown:
            raise ValueError(f"values: no variables {', '.join(unknown)} in the registry")
        _check_fragments(context, {path.fragment for path in answer.values.values()}, "values")
        return answer.model_dump(mode="json")
    shocks = {shock.id: shock for shock in load_shocks()}
    if key not in shocks:
        raise KeyError(f"the model declares no input {key!r}")
    answer = _parse(ShockAnswer, payload, key)
    if answer.shock != key:
        raise ValueError(f"shock: the answer names {answer.shock!r}, the key is {key!r}")
    missing = [variable for variable in known if variable not in answer.responses]
    if missing:
        raise ValueError(f"responses: no response for {', '.join(missing)}")
    unknown = sorted(set(answer.responses) - set(known))
    if unknown:
        raise ValueError(f"responses: no variables {', '.join(unknown)} in the registry")
    if linear(answer):
        raise ValueError(
            "responses: the response is proportional to the size everywhere; proportionality "
            "is not required — write where the response departs from it and why in "
            "`nonlinearity`, with sizes whose responses are not their ratios"
        )
    _check_fragments(context, set(answer.sources), "sources")
    return answer.model_dump(mode="json")


def question(context, key: str) -> dict:
    """What the agent reads before answering: the text and the JSON schema of the answer."""
    shocks = {shock.id: shock for shock in load_shocks()}
    if key in shocks:
        return {
            "key": key,
            "text": _shock_question(context, shocks[key]),
            "schema": ShockAnswer.model_json_schema(),
        }
    if key == MARKET:
        return {
            "key": key,
            "text": _market_question(context),
            "schema": MarketAnswer.model_json_schema(),
        }
    raise KeyError(f"the model declares no input {key!r}")


def _shock_question(context, shock: Shock) -> str:
    state = _state_of(context)
    return f"""## The shock {shock.id}

{shock.meaning}. Answer for the date {context.as_of.isoformat()} about its law.

- `frequency`: how many times a year the shock occurs, a non-negative number;
- `size`: the 5, 50 and 95 % quantiles of the signed size given that the shock occurs, in
  {shock.size_unit}; a shock that goes both ways has its 5 % quantile below zero and its
  95 % above, so separate up and down frequencies are not needed;
- `responses`: for every variable of the table below, {', '.join(map(str, HORIZONS))}
  months after the shock, at three sizes — exactly the three `size` quantiles you give —
  the 5, 50 and 95 % quantiles of the response, in the model scale of the variable. The
  month of the shock counts as the first: the response "1 month after" is the change of
  the variable in the month of the shock itself;
- `rho` in [0, 1]: how consistently the responses of different variables to one occurrence
  of the shock come out stronger or weaker than their medians — at 1 every variable takes
  one level, at 0 the levels are uncorrelated but joint extremes still come out more often
  than under independence;
- `nonlinearity`: where the response departs from proportionality to the size and why.
  Proportionality is not required; an answer whose responses are proportional to the sizes
  everywhere is refused — write sizes and responses that are not their ratios;
- `sources`: the ids of the model corpus fragments the numbers come from; may be empty.
  Search the corpus with corpus(action="search", corpus="model", as_of=\
{context.as_of.isoformat()}, query=...).

The variables, their unit, their model scale and the last value known on the date:

{state}"""


def _market_question(context) -> str:
    known = ", ".join(variable.id for variable in context.registry.variables)
    return f"""## The market expectations

Answer for the date {context.as_of.isoformat()}. For every variable that has market
expectations — oil from futures, the exchange rate from forwards or rate parity, the
policy rate from the OFZ curve and ROISfix, inflation and GDP from the consensus — the
expected value 1, 3, 6 and 12 months out in the model scale of the variable, and the id
of the model corpus fragment the number comes from. A value without a fragment is refused:
a market number that is not in the corpus is the expert's judgement, and judgements enter
the model only through the shocks. Between the horizons the values are interpolated
linearly; the variables without a market value keep the random walk with drift.

Only these variables may enter `values`: {known}."""


def _state_of(context) -> str:
    """Every variable with its unit, its model scale and its last value known on the date."""
    rows = ["| variable | unit | model scale | last value |", "|---|---|---|---|"]
    for variable in context.registry.variables:
        observed = context.store.as_of(variable.id, context.as_of)
        if observed:
            last = observed[-1]
            rows.append(
                f"| {variable.id} | {variable.unit} | {variable.transform} "
                f"| {last.value:g} ({last.month.isoformat()}) |"
            )
        else:
            rows.append(f"| {variable.id} | {variable.unit} | {variable.transform} | none |")
    return "\n".join(rows)


def _check_fragments(context, fragments: set[str], field: str) -> None:
    """Every cited fragment of the model corpus must be visible on the calculation date."""
    if not fragments:
        return
    if context.fragment is None:
        raise ValueError(
            f"{field}: the model corpus is not indexed, so fragments cannot be checked; "
            "index it with corpus(action=\"index\", corpus=\"model\") or cite none"
        )
    for fragment in sorted(fragments):
        if not context.fragment(fragment):
            raise ValueError(
                f"{field}: no model corpus fragment {fragment} is visible on "
                f"{context.as_of.isoformat()}"
            )


def _parse(model, payload: dict, key: str):
    """The payload as ``model``; a malformed answer names its field."""
    try:
        return model.model_validate(payload)
    except ValidationError as error:
        problems = []
        for issue in error.errors():
            where = ".".join(str(part) for part in issue["loc"]) or key
            problems.append(f"{where}: {issue['msg']}")
        raise ValueError(f"{key}: " + "; ".join(problems)) from None
