---
name: model-inputs
description: Answer the elicitation questions of a model in small parts, as the expert
whenToUse: when a model that declares inputs is to be estimated and its answers are not all written yet
---

# Model inputs

Some macro models are estimated not from history alone: they read answers the agent writes to
predefined questions (an elicitation). The model fixes the form of every answer; the agent fills
numbers in. No human approves the answers: their form is checked when they are written, and the
model checks the complete answer at the estimate.

## 1. You are the expert

The numbers are your own quantitative judgements. They rest on the state table the question
prints, on the model corpus where it has material for the date
(`corpus(action="search", corpus="model", as_of=..., query=...)`) and on what is known about
the shock. Do not ask the human for numbers, do not copy fixtures, do not generate the answer
with a script. `sources` names the fragments actually returned by the corpus search and is
empty when the number is your own judgement.

## 2. One part at a time

Start with
`elicit(action="next", as_of=..., model_id=..., key=<your key>, agent=<your name>)`.
The result is one small question with its schema, its envelope and `answer_call`. Copy
`answer_call` unchanged (including its `agent` when present), add `payload` with the answer
object itself (not a JSON string), and
call `elicit` again: the result contains the next question. Continue until `complete=true`.

The envelope of a response step carries what the question depends on:

- `shock_law` — the law you answered first, including the three signed sizes;
- `size_in_response_scale` — those sizes converted to the variable's model scale when the
  shock's size is a percentage change of a price while the variable is in log100. A size of
  -12 % is -12.78 log100; the response is the effect of that size, not the size itself;
- `zero_size_nodes` — the size nodes equal to zero: their three response quantiles are zero;
- `earlier_horizons` — your own saved answers for the same variable at shorter horizons;
- `reference_response` — one saved response at the same horizon, a cross-check of scale, not
  a value to copy. A cell equal to it is flagged as copied.

Units: for a log100 variable the response is 100 ln X, so a 1 % pass-through is 1 log100
point, not 1 % of the response; for a rate the response is in percentage points, where one
point is 100 basis points.

## 3. Ownership

When several agents answer at once, the orchestrator claims each key for one agent. Pass
`agent=<your name>` to `next` and `answer`; a key claimed by another agent refuses the call
and names its owner. Do not answer a key you do not own.

## 4. Failures

Read the error and change the call; a refused part is not saved and the question stays.
Never repeat a failed call unchanged. A stale ticket answers `step already answered` or
`no longer current`: call `next` again instead of reusing a ticket from earlier in the
conversation. If the last part is refused by the full check, the answer is asked again with
the same ticket.

## 5. When the key is complete

The publication returns `published` with the hash and the advisory findings. Do not accept
them: the orchestrator reviews them and either accepts the answer with a reason or reopens
named steps. Report back only the key, the hash and the findings.

## 6. The market variant

`expert.elicited.market` asks in the same interview for the market expectations of the date:
for every variable that has them, the expected value 1, 3, 6 and 12 months out in the model
scale and the id of the model corpus fragment the number comes from. A value without a
fragment is refused. Index and fetch the model corpus first —
`corpus(action="index", corpus="model")`, `corpus(action="fetch", corpus="model")` — and
search it with `kind="market"`. `expert.elicited.rw` asks about the shocks only.
