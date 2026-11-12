---
name: scenario-from-corpus
description: Turn an event described in text into a stress-test scenario s=(N,I,A) of interventions and observations, using the corpus, and hand it to a human for approval
whenToUse: when the user names an event or asks what happens if it occurs and no scenario file exists yet, even without the words scenario or corpus
---

# Scenario from an event

Input: the event in text, with or without numbers. Output: a scenario $s=(N,I,\mathcal A)$ saved with the `scenario` tool and shown to a human. A human approves it; the agent does not compute it and does not start `run`.

| Element | Meaning | File field |
|---|---|---|
| $N$ | the event in text with links to sources | `narrative`, `sources` |
| $I$ | values that a subject outside the models sets | `interventions` |
| $\mathcal A$ | outcomes stated or expected about the world | `observations` |

The file states the event alone. It names no model, no horizon and no number of paths: the calculation request carries them, and the same file is computed on any model system.

## 1. Split the clauses: intervention or observation

Read the event as a list of statements and give each one a kind.

A statement is an **intervention** when a subject outside the models sets the value and the value would not follow from the other quantities: a central bank sets the key rate at its meeting, a regulator sets a tariff, a company announces a buyback, a strait closure sets the price of Brent for the Russian economy.

A statement is an **observation** when it asserts an outcome that the system produces itself: the rouble does not weaken past 100, output falls by 2 % by the end of the year, inflation stays near 4 %.

The test is who chose the value, not which quantity it names. Brent at 130 dollars "because the strait closed" is an intervention for the Russian economy; Brent at 130 dollars "amid a global boom" is an observation, and the two have different consequences.

| Clause of the event | Kind | How it is written |
|---|---|---|
| the Bank of Russia raises the key rate to 18 % at the meeting on 2026-12-18 | intervention | `ru.key_rate`, month `2026-12`, `unit: level`, value 18 |
| the regulator raises the regulated price by 9 % from 2027-01 | intervention | the regulated price quantity (or the nearest proxy), months from `2027-01`, `unit: percent`, value 9 |
| the company announces a buyback on 2027-02 | intervention on the quantity the decision controls; where the models carry only the share price, the announced price move is the intervention and the model carries it to the other quantities | `share_price:<object>`, month `2027-02` |
| the rouble does not weaken past 100 in 2027 | observation | a band on `ru.usdrub` with `high: 100`; an interval needs both bounds, so the lower one comes from the event, the corpus or the last observed value |
| inflation stays near 4 % in 2027 | observation | a path of `ru.cpi` in its own units; the registry holds the price index, not the inflation rate |

A clause with no quantity of its own takes the nearest proxy among the served drivers, for example the shares of gold miners for the price of gold, or a stated reason in the narrative why no proxy fits. A clause with no number in the event and no number in the corpus gets a proposed number from the nearest episode, marked `assumption` by section 6.

## 2. Write values in the quantity's own units and calendar months

Months are calendar months `YYYY-MM`, one for each value. Names such as "in three months", "month 4" or "the second quarter" are not written into the file: the platform counts horizon months from the last month of the model panel, never from the calculation date.

- If the event gives a delay ("within three months"), count it from the month the event names. If the event names no month at all, take the month after the calculation date, say so in the narrative, and mark the element `assumption`.
- A month before the last month of the panel cannot be stated: the panel already holds it, and `validate` names the month. Do not convert dates to month numbers yourself.
- The number is the one the event gives, in the quantity's own units: a rate of 18 is 18, an exchange rate of 100 is 100, a price of 95.4 is 95.4. It is never the model scale (no $100\ln X$).

| `unit` | The number is | Example |
|---|---|---|
| `level` | the value itself, in the quantity's own units | the key rate is cut to 11.5: `11.5` |
| `change` | the move from the last known value, in the same units | a cut of 5 p.p. from 16.5: `-5` |
| `percent` | the relative move in per cent; prices and indices only, which the registry keeps in `log100` | the rouble weakens by 25 %: `25` |

A 25 % rise of `ru.usdrub` is written `percent: 25`, and the platform applies it once to the last observed level: the result is that level times 1.25. Do not pre-convert it, do not write the resulting level yourself, and do not apply the per cent to a value you have already converted into the model scale. `percent` on a quantity the registry keeps in its own units (`ru.key_rate`, `ru.unemployment`) is refused: write such a move as `level` or `change`.

## 3. Collect the evidence and the quantity names

Searching the corpus is mandatory, even when the event looks clear.

`jobs(action="state")` gives the macro and satellite versions the platform will use and the date of the corpus index. `corpus(action="search", query="...", as_of=<that date>, k=8)` searches the index and does not use the network; corpus texts are in Russian, so query in Russian. Make several queries: the event, its mechanism, similar episodes, also with the `kind` filter. Read a fragment in full before citing it: `corpus(action="get", id=<chunk_id>, as_of=...)`. Report every number the fragment gives; do not call a number absent after reading only part of the text.

`series(action="get", id=..., as_of=...)` gives the history of a quantity in its own units, for a macro variable (`ru.usdrub`) and for a driver (`share_price:sibn`): the last observed value is what the percentage and the change are counted from, and the report shows it beside the imposed value.

Quantity names come from the registry `src/mst/macro/registry.toml` and the driver catalog `workspace/drivers/catalog.json`; `guide(name="drivers")` explains them. Do not read `model(kind=..., action="describe")` and do not choose or name a model version: the file has no versions, and the calculation request names them. Do not read the model scale of a quantity, either: values are written in own units.

**When there is no number.** You build the scenario, and the human approves or corrects it. Questions to the user come after the scenario and do not replace it. For each quantity that neither the event nor the corpus gives:

1. Find the nearest episode in the corpus, even with other payers or another trigger.
2. Propose a value from it. When the text gives no number, measure the move of the quantity around the episode date with `series(action="get", ...)`.
3. Record the measurement in the `reason` of an intervention or in `narrative` for an observation: the episode, the values before and after from `series`, and what else was in force in those months. Without any episode, write "no episode" and your reasoning. The same placeholder percentage on every quantity is not a proposal: each number needs its own reasoning.
4. Set `"basis": "assumption"`, add the fragment of the episode to `sources`, and name in the report how the episode differs from the event.

A scenario with no interventions and no observations is the base $s_0$ and records no event. Work on an event cannot end there: if `validate` warns about this, the scenario is not finished.

## 4. Interventions

An intervention sets or shifts the value of one quantity in the months named. A decision is written as a `set`: the quantity takes that value, and the model's own value in those months is discarded. A `shift` displaces the model's own value by an amount; use it when the event adds a displacement to the path, not when the event chooses the value.

```json
{"quantity": "ru.key_rate", "months": ["2026-12"], "kind": "set", "values": [18.0],
 "unit": "level", "reason": "unscheduled meeting raises the rate", "basis": ["<chunk_id>"]}
```

```json
{"quantity": "share_price:lkoh", "months": ["2027-02"], "kind": "shift", "values": [-10.0],
 "unit": "percent", "reason": "export duty effect absent from the model", "basis": "assumption"}
```

- `months` and `values` have one entry per month each and the same length; a constant value over several months is repeated, not written once.
- The pair (quantity, month) belongs to at most one intervention: write one intervention with several months, never two on the same month.
- A value may carry its law instead of a single number: `{"uniform": [low, high]}` draws the number per group of paths, and `{"with_probability": p, "value": x}` imposes `x` in a share `p` of the groups. The law is drawn from the seed of the calculation, so the scenario price averages it and the report shows in how many groups the intervention was active. Write a law when the event states a range or a chance ("a cut of 3 to 5 p.p.", "with a probability of one half"); a number the event states stays a number.
- In the default form the seeded value does not move the other macro variables of the same month; later months read it through the model's lags. When the event states the reaction of the market, write that reaction as its own intervention or observation instead of expecting the same-month spread.
- `reason` is a short identifier of the decision, `basis` the label of section 6.

## 5. Observations

An observation is a requirement on the measure over outcomes: a quantity lies in a band, a mean lies in a band, or a probability is at least some value.

| What the event asserts | `kind` | Fields |
|---|---|---|
| a quantity lies within bounds in a month | `constraint` | `low` and `high` |
| a quantity has an exact value | `constraint` | `value`; the platform expands it to `value ± tau`, where `tau` is a tenth of the typical monthly move |
| only a mean over several months lies within bounds | `expectation` | `low` and/or `high` |
| a quantity falls in a band with a probability of at least $p$ | `probability` | `low`, `high`, `at_least` |

```json
{"kind": "constraint", "terms": [{"quantity": "ru.usdrub", "month": "2027-06", "weight": 1.0}],
 "unit": "level", "low": 90.0, "high": 100.0, "basis": ["<chunk_id>"]}
```

```json
{"kind": "expectation", "terms": [
   {"quantity": "ru.key_rate", "month": "2027-01", "weight": 1.0},
   {"quantity": "ru.key_rate", "month": "2027-02", "weight": 1.0}],
 "unit": "level", "low": 14.0, "high": 16.0, "basis": "assumption"}
```

- `terms` is the functional: `weight` times the quantity in that month, summed over the terms. A statement about one quantity in one month has one term with weight 1.0. A statement about several quantities or months is one weighted sum, not several observations.
- Every observation names its `kind`: `constraint`, `expectation` or `probability`.
- A quantity with a month appears once in an observation, and a weight is not zero.
- `terms` read either macro variables only or drivers `<quantity>:<object>` only: a mixed observation is refused.
- A `constraint` needs both `low` and `high`. A one-sided statement ("not weaker than 100") becomes a band, and the second bound comes from the event, the corpus or the last observed value; when you chose it, it is part of the `assumption`. Only an `expectation` takes a single bound, and it bounds the mean over the months, not every path.
- For quantities in `log100`, the weights of one observation sum to one (an aggregate level, in `level` or `change`) or to zero (a ratio, in `percent`); any other sum is refused. A statement about a change of an index over time is usually such a ratio.
- An exact `value` is expanded to a tolerance, so its price grows as the band narrows. Prefer the interval the source gives; do not tune the width to the price. An `expectation` is available for quantities the registry keeps in their own units; on `log100` quantities the mean of the logarithm is not the logarithm of the mean and is refused.
- An observation on a driver, on a mean or on a probability needs `inner` of at least 2 in the calculation request: those requirements are met by reweighting the paths inside a group of one parameter draw.
- Do not add an observation of a quantity in a month where an intervention sets it: the platform substitutes the installed value, and an observation outside it contradicts the models at every path.
- A statement about a period covers its months: an observation on each month or on their mean, or a stated reason in the narrative why the effect ends earlier. A single month for a year-long event describes another event.

## 6. Labels

Each intervention and observation is marked in `basis` with the word `event`, a list of fragment ids, or the word `assumption`. A word inside the list is refused: `["event", "<chunk_id>"]` is not a label.

- `event` when the text of the event states the number or a direct conversion to it, for example Brent at 130 dollars or a cut of 5 p.p. from the current rate.
- A fragment id only when the fragment gives the number of that element or a direct conversion to it.
- `assumption` when you chose the number, even if a fragment fixes the mechanism or the direction: then add the fragment to `sources`.
- A number from another event is an analogy: name the episode in the report and how it differs.

`assumption` is an honest label on a number you propose, not a refusal to propose it. A scenario of assumptions only is allowed, but the report must show it. `validate` checks that cited fragments are visible on the calculation date; nobody but the human checks that the labels are true.

## 7. Save, check and approve

`narrative` is the full text of $N$: what happened, which statements were classified how and why. Write it in the language of the user. A number in the narrative is the number of the field it explains, or the narrative names its units: 444.73 in the text and 444.10 in the observation are two answers to one question.

```json
{
  "narrative": "The Bank of Russia raises the key rate at an unscheduled meeting ...",
  "sources": ["<chunk_id>"],
  "interventions": [
    {"quantity": "ru.key_rate", "months": ["2026-12"], "kind": "set", "values": [18.0],
     "unit": "level", "reason": "unscheduled meeting", "basis": "event"}
  ],
  "observations": [],
  "approved_by": null
}
```

1. `scenario(action="save", id="...", spec={...})` checks the contract and writes the file. An id is lowercase letters, digits and hyphens. Do not write the file around `scenario`.
2. `scenario(action="validate", id="...")` checks the quantity names, the units, the calendar months, that every observation translates into one scale, the corpus fragments and the approval. It runs on the newest finished macro version and the satellites estimated on it, over twelve months and one inner path, so a statement later than that horizon is reported as a problem even though a run whose request names a longer horizon computes it. The calculation request carries the horizon, the models and `inner` the run will really use.
3. At this stage the only item left in `problems` should be the missing approver. A month, a name or a unit the platform refuses is a mistake in the file: correct it. Do not change the content of a scenario only to pass the check without saying so.

`labels` shows how many elements carry the event, a source and an assumption; `warnings` says when the narrative cites no source, when every element is an assumption and when the file records no event at all. Carry both into the report.

Show the scenario to the human. Fill `approved_by` only with a name the human gave, after their explicit consent: the scenario is approved by a person, never by the agent. Then call `save` and `validate` again.

The horizon, the macro model and the satellites are chosen in the calculation request, not here. The same file is computed unchanged on another model, and the difference between the runs shows how much the conclusion depends on the choice.

## 8. Report to the human

Write in the language of the user. For each scenario show:

- $N$ and the calendar months it names, with what happens in each;
- the table of clauses: each clause, whether it is an intervention or an observation, the element that carries it, the number with its unit and month, or "not expressed" with the reason;
- for each intervention, the last observed value from `series` beside the imposed one, so the move is visible;
- the elements that rest on the event, on a number from a source and only on an assumption;
- where confidence is lowest, and which statements the models cannot express at all.

The calculation report shows how far each intervention stood from the base forecast (`alpha` in the `interventions` block) and calls an intervention-only scenario's price zero; do not promise either number before the run.
