---
name: model-inputs
description: Answer the elicitation questions of a model and see what is missing before an estimate
whenToUse: when a model that declares inputs is to be estimated and its answers are not all written yet
---

# Model inputs

Some macro models are estimated not from history alone: they read answers the agent writes
to predefined questions (an elicitation). The model fixes the form of every answer; the agent
fills numbers in. No human approves the answers — they are checked by their form, at the
moment of writing, and by the model itself at the estimate.

## 1. See what is missing

`model(kind="inputs", action="list", model_id=..., as_of=...)` names every key the model
declares for the date, written or missing, the missing ones first. A key is one question.

## 2. Answer one question

`model(kind="inputs", action="question", model_id=..., as_of=..., key=...)` gives the text of
the question and the JSON Schema of the answer. The text states everything the answer needs:
the quantities asked, their units, the state of the series on the date, and the
model-specific instructions. Search the model corpus for material with the same date:
`corpus(action="search", corpus="model", as_of=<the same date>, query=...)`.

`model(kind="inputs", action="write", model_id=..., as_of=..., key=..., payload=...)` checks
the answer against the schema and the question's rules and stores it. A refused answer comes
back with the field it was refused for; correct that field and write again. The answer of a
key replaces the previous one of the same date.

## 3. Until nothing is missing

`model(kind="inputs", action="status", ...)` repeats the summary. When no key is missing,
estimate: `model(kind="macro", action="estimate", model_id=..., as_of=...)`. The estimate
reads the answers of the date as they stand; an answer written later is picked up by
estimating again, which gives a new version.
