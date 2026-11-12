"""Read-only conversion of historical scenario files.

The caller resolves the original panel context. This module never loads a model,
infers a panel origin from as_of, writes a file, or hides old model coordinates in
new metadata. A number that cannot round-trip exactly stops the conversion.
"""

import math
from collections.abc import Callable
from decimal import localcontext

from pydantic import ValidationError

from .contracts import Scenario
from .translate import (
    TranslationData,
    _number,
    _own_level,
    calendar_month,
    observation_bounds,
    shift_to_scale,
    to_scale,
)


class LegacyTranslationError(ValueError):
    """A legacy statement needs a decision instead of a numerical approximation."""


def _inverse(target, quantity, unit, anchor, *, shift=False):
    with localcontext() as context:
        context.prec = 80
        value = _number(target)
        last = _number(anchor)
        if quantity.transform == "identity":
            return float(value if shift or unit == "level" else value - last)
        if shift:
            if unit == "percent":
                return float(100 * ((value / 100).exp() - 1))
            return float(_own_level(quantity, last) * ((value / 100).exp() - 1))
        if unit == "percent":
            return float(100 * (((value - last) / 100).exp() - 1))
        own = _own_level(quantity, value)
        return float(own if unit == "level" else own - _own_level(quantity, last))


def _exact(target, candidate, forward):
    lower = upper = candidate
    for _ in range(9):
        if forward(lower) == target:
            return lower
        if forward(upper) == target:
            return upper
        if forward(lower) < target < forward(upper):
            break
        lower = math.nextafter(lower, -math.inf)
        upper = math.nextafter(upper, math.inf)
    raise LegacyTranslationError(f"no exactly representable own-unit value for {target!r}")


def _values(targets, quantity, unit, anchor, *, shift=False):
    converter = shift_to_scale if shift else to_scale
    return [
        _exact(
            target,
            _inverse(target, quantity, unit, anchor, shift=shift),
            lambda value: converter(quantity, value, unit, anchor),
        )
        for target in targets
    ]


def _impact(raw, data):
    quantity = data.quantity(raw["target"])
    start = raw["start"]
    if "values" in raw:
        if "delta" in raw or "end" in raw:
            raise LegacyTranslationError("impact mixes values and delta/end formats")
        values = list(raw["values"])
    else:
        end = raw["end"]
        if end < start:
            raise LegacyTranslationError("impact end is before start")
        values = [raw["delta"]] * (end - start + 1)
    anchor = data.last_model.get(quantity.id, 0.0)
    if quantity.transform == "logit":
        raise LegacyTranslationError(
            f"{quantity.id}: an old logit shift has no physical additive equivalent"
        )
    units = ("change",) if quantity.transform == "identity" else ("percent", "change")
    for unit in units:
        try:
            if (
                unit == "change"
                and quantity.transform != "identity"
                and quantity.id not in data.last_model
            ):
                raise LegacyTranslationError(f"no last known value for {quantity.id}")
            converted = _values(values, quantity, unit, anchor, shift=True)
            return {
                "quantity": quantity.id,
                "months": [calendar_month(start + k, data.origin) for k in range(len(values))],
                "kind": "shift",
                "values": converted,
                "unit": unit,
                "reason": raw["reason"],
                "basis": raw["basis"],
            }
        except LegacyTranslationError:
            continue
    raise LegacyTranslationError(
        f"impact on {quantity.id}: no shared unit preserves every value {values!r}"
    )


def _observation(raw, kind, data):
    from .contracts import Constraint, Expectation, ProbabilityCondition

    terms = [
        {
            "quantity": term["variable"],
            "month": calendar_month(term["month"], data.origin),
            "weight": term["weight"],
        }
        for term in raw["terms"]
    ]
    quantities = [data.quantity(term["quantity"]) for term in terms]
    transforms = {quantity.transform for quantity in quantities}
    if len(transforms) != 1:
        raise LegacyTranslationError(
            "mixed registry transforms cannot become one own-unit observable"
        )
    quantity = quantities[0]
    if kind == "expectation" and quantity.transform != "identity":
        raise LegacyTranslationError(f"expectations on {quantity.id}: E(log X) is not log E(X)")
    if quantity.transform == "logit" and len(terms) != 1:
        raise LegacyTranslationError("multi-term logit observation cannot be translated exactly")
    total = sum(term["weight"] for term in terms)
    if quantity.transform == "log100" and total not in (0.0, 1.0):
        raise LegacyTranslationError(f"log100 observation on {quantity.id} has unsupported weights")
    anchor = sum(term["weight"] * data.last_model.get(term["quantity"], 0.0) for term in terms)
    units = ("level",) if quantity.transform == "identity" else ("level", "percent", "change")
    if quantity.transform == "log100" and total == 0:
        units = ("percent",)
    bounds = {name: raw[name] for name in ("low", "high", "value") if raw.get(name) is not None}
    cls = {
        "constraint": Constraint,
        "expectation": Expectation,
        "probability": ProbabilityCondition,
    }[kind]
    for unit in units:
        try:
            if unit != "level" and any(term["quantity"] not in data.last_model for term in terms):
                continue
            converted = _values(list(bounds.values()), quantity, unit, anchor)
            row = {
                "kind": kind,
                "terms": terms,
                "unit": unit,
                "basis": raw["basis"],
                **dict(zip(bounds, converted, strict=True)),
            }
            if kind == "probability":
                row["at_least"] = raw["at_least"]
            observation = cls.model_validate(row)
            recovered = observation_bounds(observation, data)
            if any(recovered[name] != value for name, value in bounds.items()):
                continue
            return row
        except (LegacyTranslationError, ValueError, ValidationError):
            continue
    raise LegacyTranslationError(
        f"observation on {quantity.id}: no shared unit preserves bounds {bounds!r}"
    )


def read_legacy(
    raw: dict,
    *,
    context: TranslationData | None = None,
    resolver: Callable[[dict], TranslationData] | None = None,
) -> Scenario:
    """Convert one old file; the resolver supplies its actual old panel origin.

    Approval is copied, never invented. Old causes and regime assertions remain
    visible in the narrative for reclassification; they do not influence a plan.
    """
    if "interventions" in raw or "observations" in raw:
        return Scenario.model_validate(raw)
    allowed = {
        "narrative",
        "sources",
        "horizon",
        "macro",
        "satellites",
        "constraints",
        "expectations",
        "probabilities",
        "impacts",
        "causes",
        "regime_event",
        "inner",
        "approved_by",
    }
    extra = set(raw) - allowed
    if extra:
        raise LegacyTranslationError(f"unknown legacy fields: {', '.join(sorted(extra))}")
    dated = any(raw.get(key) for key in ("impacts", "constraints", "expectations", "probabilities"))
    if dated and context is None:
        if resolver is None:
            raise LegacyTranslationError(
                "legacy numeric statements need panel context or a context resolver"
            )
        context = resolver(raw)
    result = {
        "narrative": raw["narrative"],
        "sources": raw.get("sources", []),
        "approved_by": raw.get("approved_by"),
        "interventions": [],
        "observations": [],
    }
    previous = list(raw.get("causes", []))
    if raw.get("regime_event") is not None:
        previous.append(f"regime_event={raw['regime_event']}")
    if previous:
        result["narrative"] += "\nпрежние причины: " + ", ".join(previous)
        result["approved_by"] = None
    for index, impact in enumerate(raw.get("impacts", []), 1):
        try:
            result["interventions"].append(_impact(impact, context))
        except (ValueError, KeyError) as error:
            raise LegacyTranslationError(f"impacts[{index}]: {error}") from error
    for family, kind in (
        ("constraints", "constraint"),
        ("expectations", "expectation"),
        ("probabilities", "probability"),
    ):
        for index, observation in enumerate(raw.get(family, []), 1):
            try:
                result["observations"].append(_observation(observation, kind, context))
            except (ValueError, KeyError) as error:
                raise LegacyTranslationError(f"{family}[{index}]: {error}") from error
    try:
        return Scenario.model_validate(result)
    except ValidationError as error:
        raise LegacyTranslationError(str(error)) from error
