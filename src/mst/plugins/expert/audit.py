"""Advisory findings about a published expert answer: never a refusal.

The form checks say whether the estimate can read an answer; these say whether the answer is
economically credible. A finding names a code, the cells it is about and a message, and the
orchestrating agent decides: accept with a reason, or reopen the parts the cells belong to.

The generic codes need no theory: an answer that repeats one cell across horizons has no
dynamics, a cell of zero width claims certainty, a first-month response many standard
deviations above the variable's own history is implausible, a cell equal to the reference the
question printed was copied, and a response table equal to another shock's means one of them
was not judged. The rules of ``audit.toml`` add the economic expectations; each carries its
own note and is a warning, not a verdict.
"""

import math
import tomllib
from pathlib import Path

from mst import inputs as answers_folder

from .answers import HORIZONS, NODES

RULES = Path(__file__).parent / "audit.toml"
#: Responses further than this many historical standard deviations at the first month.
WIDE_FIRST = 8.0


def _median(answer, variable, horizon, node="q50"):
    return answer["responses"][variable][str(horizon)][node]["q50"]


def _sigma(context, variable):
    """The standard deviation of the variable's monthly change, in its model scale."""
    observed = context.store.as_of(variable.id, context.as_of)
    if len(observed) < 24:
        return None
    values = [item.value for item in observed[-61:]]
    if variable.transform == "log100":
        if any(value <= 0 for value in values):
            return None
        changes = [
            100.0 * (math.log(second) - math.log(first))
            for first, second in zip(values, values[1:], strict=False)
        ]
    else:
        changes = [second - first for first, second in zip(values, values[1:], strict=False)]
    mean = sum(changes) / len(changes)
    spread = sum((change - mean) ** 2 for change in changes) / (len(changes) - 1)
    return math.sqrt(spread) or None


def _duplicates(answer):
    """One cell repeated across horizons, and one median repeated across the whole answer."""
    found = []
    for variable, by_horizon in answer["responses"].items():
        cells = {}
        for horizon in HORIZONS:
            cell = by_horizon[str(horizon)]["q50"]
            cells.setdefault((cell["q5"], cell["q50"], cell["q95"]), []).append(horizon)
        repeated = {cell: rows for cell, rows in cells.items() if len(rows) > 1}
        for rows in repeated.values():
            found.append(
                {
                    "code": "same_cell",
                    "cells": [f"{variable}@{horizon}" for horizon in rows],
                    "message": f"{variable}: the same quantiles at horizons "
                    + ", ".join(map(str, rows))
                    + " leave the response without dynamics",
                }
            )
        if not repeated:
            medians = {by_horizon[str(horizon)]["q50"]["q50"] for horizon in HORIZONS}
            if len(medians) == 1:
                found.append(
                    {
                        "code": "flat_medians",
                        "cells": [f"{variable}@{horizon}" for horizon in HORIZONS],
                        "message": f"{variable}: one median for every horizon is a "
                        "constant response, not a transmission",
                    }
                )
    return found


def _certainty(answer):
    """A cell whose quantiles coincide while its median is not zero."""
    found = []
    for variable, by_horizon in answer["responses"].items():
        for horizon in HORIZONS:
            for node in NODES:
                cell = by_horizon[str(horizon)][node]
                if cell["q5"] == cell["q95"] and cell["q50"] != 0.0:
                    found.append(
                        {
                            "code": "zero_width",
                            "cells": [f"{variable}@{horizon}/{node}"],
                            "message": f"{variable} at {horizon}m at size {node}: q5 equals "
                            f"q95, so the answer claims a certain response",
                        }
                    )
    return found


def _large_first(answer, context):
    """A first-month response far outside the variable's own history."""
    found = []
    for variable_row in context.registry.variables:
        if variable_row.id not in answer["responses"]:
            continue
        sigma = _sigma(context, variable_row)
        if sigma is None:
            continue
        value = _median(answer, variable_row.id, 1)
        if abs(value) > WIDE_FIRST * sigma:
            found.append(
                {
                    "code": "large_first",
                    "cells": [f"{variable_row.id}@1"],
                    "message": f"{variable_row.id} at 1m: {value:.4g} is "
                    f"{abs(value) / sigma:.1f} standard deviations of its monthly change "
                    f"({sigma:.4g})",
                }
            )
    return found


def _copied_reference(answer, context):
    """A cell equal to the reference response the question printed for that horizon."""
    first = context.registry.variables[0].id
    found = []
    if first not in answer["responses"]:
        return found
    for horizon in HORIZONS:
        reference = answer["responses"][first][str(horizon)]["q50"]
        for variable in answer["responses"]:
            if variable == first:
                continue
            cell = answer["responses"][variable][str(horizon)]["q50"]
            if cell == reference:
                found.append(
                    {
                        "code": "copied_reference",
                        "cells": [f"{variable}@{horizon}"],
                        "message": f"{variable} at {horizon}m is exactly the reference "
                        f"response of {first}; a transmission check was answered as a copy",
                    }
                )
    return found


def _same_shock(answer, context, key):
    """A response table equal to another shock's: one of the two was not judged."""
    found = []
    written = answers_folder.read(context.inputs, context.as_of)
    for other, payload in written.items():
        if other != key and payload.get("responses") == answer["responses"]:
            found.append(
                {
                    "code": "same_shock",
                    "cells": [],
                    "message": f"the responses of {key} equal the responses of {other}: "
                    "two shocks cannot share one transmission",
                }
            )
    return found


def load_rules() -> list[dict]:
    """The economic rules of ``audit.toml``; a missing file means no rules."""
    if not RULES.is_file():
        return []
    return tomllib.loads(RULES.read_text(encoding="utf-8")).get("rule", [])


def _value(answer, variable, horizon):
    return _median(answer, variable, horizon)


def _violates(rule, answer) -> bool:
    """Whether the answer breaks one rule; a rule it cannot judge is not broken."""
    variable, other = rule["variable"], rule.get("other")
    horizon = rule.get("horizon", "all")
    horizons = HORIZONS if horizon == "all" else (int(horizon),)
    for current in horizons:
        try:
            left = _value(answer, variable, current)
            right = _value(answer, other, current) if other else None
        except KeyError:
            return False
        op = rule["op"]
        if op in {"le", "ge"}:
            size = answer["size"]["q50"]
            if size == 0.0:
                continue
            if size < 0.0:
                left = -left
        if op == "same_sign" and (left == 0 or right == 0):
            continue
        if op == "opposite_sign" and (left == 0 or right == 0):
            continue
        if op == "opposite_sign" and abs(left) < rule.get("min_abs", 0.0):
            continue
        if op == "le" and left <= rule["value"]:
            continue
        if op == "ge" and left >= rule["value"]:
            continue
        if op == "same_sign" and left * right > 0:
            continue
        if op == "opposite_sign" and left * right < 0:
            continue
        if (op == "at_most_times" and left * right >= 0
                and abs(left) <= rule["value"] * abs(right)):
            continue
        return True
    return False


def _rules(answer, context, key):
    """The broken economic rules, one finding each."""
    found = []
    for rule in load_rules():
        if rule.get("shock", "any") not in {"any", key}:
            continue
        if _violates(rule, answer):
            horizon = rule.get("horizon", "all")
            found.append(
                {
                    "code": "rule",
                    "rule": rule["id"],
                    "cells": [f"{rule['variable']}@{horizon}"],
                    "message": f"{rule['id']}: {rule['note']}",
                }
            )
    return found


def findings(context, key: str, answer: dict) -> list[dict]:
    """Every advisory finding about one published answer, in a stable order."""
    if key == "market":
        return []
    found = []
    found += _duplicates(answer)
    found += _certainty(answer)
    found += _large_first(answer, context)
    found += _copied_reference(answer, context)
    found += _same_shock(answer, context, key)
    found += _rules(answer, context, key)
    return [{"severity": "warning", **row} for row in found]
