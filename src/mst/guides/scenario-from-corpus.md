---
name: scenario-from-corpus
description: Turn an event described in text into a stress-test scenario s=(N,m,O,D,C) using the corpus and hand it to a human for approval
whenToUse: when the user names an event or asks what happens if it occurs and no scenario file exists yet, even without the words scenario or corpus
---

# Scenario from an event

Input: an event described in text, with or without numbers. Output: a scenario $s=(N,m,O,D,\mathcal C)$ saved with the `scenario` tool and shown to a human. A human approves the scenario; the agent does not compute it and does not start `run`.

| Element | Meaning | File field |
|---|---|---|
| $N$ | the event in text with links to sources | `narrative`, `sources` |
| $m$ | the model versions | `macro`, `satellites` |
| $\mathcal C$ | requirements on the measure $Q$: the event $A$ written on paths | `constraints`, `expectations`, `probabilities` |
| $D$ | the causes of the event: coordinates of $\omega$ other than parameters | `causes`, `regime_event` |
| $O$ | effects absent from the models: given shifts of paths | `impacts` |

The models compute the consequences of the event. The scenario states only the event itself and what the models lack.

## 1. Versions and the start month

1. `jobs(action="state")`: the versions to use and the state of the corpus. Take the macro and satellite versions it names. Take another version only if the user asks; `model(kind=..., action="list")` shows them.
2. `model(kind="macro", action="describe", version=...)`: `as_of`, shocks `shocks`, hidden coordinates `latents`, variable scales `variables`, the start point `origin`.
3. `model(kind="satellites", action="describe", version=...)`: scales and regressors `quantities`, drivers `served`, and `rejected` with the reason.
4. `model(kind="macro", action="base", version=..., horizon=..., only=[...])`: the base path under $s_0$ by month, `p5`, `p50`, `p95` in the model scale. This is not a calculation and needs no approval.

**Calendar.** Month $t=0$ is `origin.month`, not the month of `as_of`. Scenario month $k$ is $t=0$ plus $k$ calendar months. With `as_of` 2026-10-29 and `origin.month` 2026-09, month 1 is October 2026 even though it has passed. An event with no date starts in the first month after `as_of`, here month 2, never in a month that has passed. Convert the dates of the event to month numbers by this rule and list the mapping in the report.

## 2. What the event asserts

Numbers come from the event and the corpus, not from memory.

**Clauses.** Split the event into clauses and make a table for the report: each clause, then the element of the scenario that carries it, or "not expressed" with the reason. A clause with no quantity of its own needs the nearest proxy among `served`, for example the shares of gold miners for the price of gold, or a stated reason why no proxy fits. Scale, duration and start that the event leaves open are listed there too.

**Outcomes.** Name the main quantity the event moves and the sign of its move, with the reason. If the event, the corpus and the models do not fix the sign, the outcomes exclude each other: build one scenario per sign, each computed separately, with ids ending in `-a` and `-b`. Uncertainty about scale alone is the width of an interval, not a new scenario.

## 3. Corpus

Searching the corpus is mandatory, even when the event looks clear.

`corpus(action="search", query="...", as_of=<as_of of the version>, k=8)` searches the index and does not use the network. Corpus texts are in Russian, so query in Russian. Make several queries: the event, its mechanism, similar episodes, also with the `kind` filter. Read a fragment in full before citing it: `corpus(action="get", id=<chunk_id>, as_of=...)`. Report every number the fragment gives; do not call a number absent after reading only part of the text.

The history of a quantity in its own units is available with `series(action="get", id=..., as_of=...)`, for a macro variable (`ru.usdrub`) and for a driver (`share_price:sibn`). Compare the event with the base forecast in the corpus: if it already contains a similar path, say so in the report.

**When there is no number.** You build the scenario, and the human approves or corrects it. Questions to the user come after the scenario and do not replace it. For each quantity that neither the event nor the corpus gives:

1. Find the nearest episode in the corpus, even with other payers or another trigger.
2. Propose a scale, duration and interval width from it. When the text gives no number, measure the move of the quantity around the episode date with `series(action="get", ...)`.
3. Record the measurement next to the number: in `reason` for an impact, in `narrative` for a condition. Write the episode, the values before and after from `series`, and what else was in force in those months. Without any episode, write "no episode" and your reasoning. The same placeholder percentage on every driver is not a proposal: each number needs its own reasoning.
4. Set `"basis": "assumption"` as a string, add the fragment of the episode to `sources`, and name in the report how the episode differs from the event.

If the event has no variable of its own in the models, write its consequences in the quantities that exist, by the rule in section 5.

A scenario with no conditions and no impacts is the base $s_0$ and records no event. Work on an event cannot end there: if `validate` warns about this, the scenario is not finished.

## 4. Conditions $\mathcal C$

A condition records the event, or its scale and duration taken from the corpus. Do not set as a condition what the models must derive: that is the result of the calculation, not its input. A quantity fixed by a condition matches the forecast up to the tolerance, so it cannot serve as a target.

| What the event asserts | Field |
|---|---|
| a quantity lies within bounds | `constraints`, `low` and `high` |
| an exact value | `constraints`, `value`; the platform expands it to `value ± tau`, where `tau` is a tenth of the typical monthly move |
| only a mean | `expectations`, `low` and/or `high` |
| only a probability | `probabilities`, `low`, `high`, `at_least` |

```json
{"terms": [{"variable": "ru.key_rate", "month": 1, "weight": 1.0}], "low": 17.0, "high": 18.0, "basis": ["<chunk_id>"]}
```

**A deviation from the base.** A condition sets a level. To say that output is 1.2 % below the base in month 6, read `p50` of `ru.output` in month 6 from `base`, subtract 1.2 in the `log100` scale, and put the interval around that number. Take its width from your uncertainty about the size, not from the price. A move of a level over several months is a condition with weights on two months of the variable.

- `month` runs from 1 to `horizon`. Month 0 cannot appear in a condition.
- A variable with a month appears once in a condition, and a weight is not zero.
- `terms` read either macro variables only or drivers `<quantity>:<object>` only: a mixed condition is refused.
- Conditions on drivers and conditions on means and probabilities need `inner` of at least 2.

**Scale.** Bounds are in the model scale $y=g(X)$. `variables` gives the scale of each variable and `quantities` of each quantity: `identity` is own units, `log100` is $100\ln X$, `logit` is $\ln\big(d/(1-d)\big)$ for a share $d$.

Measure a shift from `origin.last`, which holds the last value of each variable in the model scale and its month. Values from `series` do not work for this: the model reads variables with `seasonal: true` seasonally adjusted, so a raw series gives another number in their scale.

**Comparability.** The price grows as an interval narrows, and for exact values as $-k\log(2\tau)$, where $k$ is the number of such conditions. Scenarios are comparable only at equal tolerances, so take the width from the spread of the source and do not tune it.

## 5. Impacts $O$

An impact records an effect absent from the models that stays where you put it, for example an export ban that hits one company. Test: if the models carry this effect from the conditions, no impact is needed, and a shift on top of such a channel counts the effect twice.

**Macro effects are conditions.** An impact on a macro variable moves that variable alone: the other macro variables do not react, and it reaches only the satellites with it in `regressors`. `ru.cpi` and `ru.output`, for example, move no share price. So when an effect on a macro variable must spread through the economy, or the event asserts the path of a macro variable, write a condition: the models then derive the other variables, and the scenario gets a price. This holds even when the models do not know the cause of the effect. Use an impact on a macro variable only if you want no reaction, and say why in `reason`. `validate` warns which quantities read each macro variable you shift.

```json
{"target": "share_price:<object>", "start": 1, "end": 6, "delta": -10.54, "reason": "why", "basis": ["<chunk_id>"]}
```

- `delta` is in the model scale of the quantity: a 10 % fall under `log100` is $100\ln 0.9\approx -10.54$, not $-0.1$.
- An impact on a macro variable enters the path before the satellites and reaches only those with it in `regressors`.
- An impact on a driver shifts that driver alone. A sector shift (`share_price:oil_gas`) does not reach the shares of its companies (`share_price:lkoh`): to move the companies, list them, and give each its own number when the event hits them differently. `validate` names the served drivers a shift does not reach. Lower levels feel a shift only through their models. The driver must be in `served`; name in the report any important driver found in `rejected`.
- A shift acts only in months `start..end` and vanishes entirely after `end`. Hold a lasting effect until the end of the horizon.
- Overlapping impacts on one target add up. Write a measure phased in over months as several impacts that add a step each, not as one constant shift.
- A condition on the same quantity holds on the final path: do not add `delta` to the bounds by hand.
- An impact is not random and has no price of its own. A scenario of impacts alone has $Q=P$, so its price is zero, may be missing from the result, and says nothing about how plausible the scenario is. Say so in the report.

## 6. Causes $D$

Causes are the coordinates through which the scenario explains the event: shocks from `shocks` and hidden coordinates from `latents`. The laws of the other coordinates and of the parameters stay as they were.

- With no causes named, `causes` is empty and $\omega_D$ holds everything except the parameters. Only then does the price split by level in `dist.by_level`.
- Name a cause only if the event or the corpus points to it. Fewer causes make the explanation more expensive, so do not tune `causes` to the price or to pass the check.
- If the event is unreachable at some value of the remaining coordinates, the scenario contradicts the models. `validate` or the calculation will say so; report it instead of silently removing the condition.
- `regime_event` is a regime the event itself asserts over the whole horizon, for example `regime.stress` of `msvar.markov`. That differs from naming a regime among the causes. It needs `regime_path` among the causes, or empty `causes`.

## 7. Labels

Each element of $O$ and $\mathcal C$ is marked in `basis` with the word `event`, a source, or the word `assumption`.

- Use `event` when the text of the event states the number or a direct conversion to it, for example Brent at 130 dollars or a cut of 5 points from the current rate.
- Use a source only if the fragment gives the number of that element or a direct conversion to it.
- If the fragment confirms only the mechanism or the direction and you chose the size, use `assumption` and add the fragment to `sources`.
- A number from another event is an analogy: name the episode in the report and how it differs.
- `validate` checks that fragments are visible on the calculation date. Nobody but the human checks that the labels are true.

`assumption` is an honest label on a number you propose, not a refusal to propose it. A scenario of assumptions only is allowed, but the report must show it.

## 8. Save and check

`narrative` is the full text of $N$: what happened, which outcome was chosen and why. Write it in the language of the user.

```json
{
  "narrative": "The Bank of Russia raises the key rate at an unscheduled meeting ...",
  "sources": ["<chunk_id>"],
  "horizon": 12,
  "macro": "<64 hex>",
  "satellites": "<64 hex>",
  "constraints": [],
  "expectations": [],
  "probabilities": [],
  "causes": [],
  "impacts": []
}
```

1. `scenario(action="save", id="...", spec={...})` checks the contract and writes the file. An id is lowercase letters, digits and hyphens. Do not write the file around `scenario`.
2. `scenario(action="validate", id="...")` checks the versions, causes, impacts, the feasibility of the conditions and the visibility of fragments. At this stage the only item left in `problems` should be the missing approver. Do not change the content of a scenario only to pass the check without saying so.

`labels` shows how many elements carry the event, a source and an assumption, and `warnings` what the scenario leaves unsupported and where its impacts do not go. Act on a warning about reach or say why you keep the element. Carry both into the report.

Fill `approved_by` only with a name the human gave, after their explicit consent. Then call `save` and `validate` again.

## 9. Report to the human

Write in the language of the user. For each scenario show:

- $N$ and the mapping of month numbers to the calendar;
- the table of clauses and the main quantity with the sign of its move;
- a table of the elements of $\mathcal C$ and $O$: own units, model scale, months, the event, the source and what it says, or `assumption` with its measurement or "no episode";
- $D$ and why these causes;
- how many elements rest on the event, on a number from a source, and only on a mechanism or an assumption;
- where confidence is lowest.
