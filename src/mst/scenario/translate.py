"""Translate calendar statements into registry coordinates.

``last_model`` and ``history`` contain panel coordinates, not physical levels. Keeping
that exact anchor avoids a lossy exp/log round trip. Decimal scalar arithmetic rounds
once at the boundary; model arrays remain float64.
"""

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import Literal

import numpy as np

from ..metrics.cells import typical_movement
from ..model_sdk.contracts import LinearConstraint, LinearTerm
from .contracts import Expectation, Observation, ProbabilityValue, Scenario, UniformValue, Unit

CALENDAR_MONTH = re.compile(r"^[0-9]{4}-(0[1-9]|1[0-2])$")

INTERVENTION_STREAM = 31
"""The stream the laws of the interventions are drawn on, apart from the models' own.

A group of paths is one draw of the parameters, so the law of the event is drawn once per
group: ``uniform`` gives a value, ``with_probability`` gives whether the stated value is
imposed. The stream depends on the seed and on nothing else in the file, so an observation
added later moves no drawn value.
"""


@dataclass(frozen=True)
class QuantityData:
    id: str
    transform: Literal["identity", "log100", "logit"]


@dataclass(frozen=True)
class TranslationData:
    origin: str
    quantities: Mapping[str, QuantityData]
    last_model: Mapping[str, float]
    history: Mapping[str, np.ndarray]

    def quantity(self, name: str) -> QuantityData:
        try:
            return self.quantities[name]
        except KeyError as error:
            raise ValueError(f"unknown quantity {name}") from error


def _calendar_number(month: str) -> int:
    if not isinstance(month, str) or not CALENDAR_MONTH.fullmatch(month):
        raise ValueError(f"invalid calendar month {month!r}; expected YYYY-MM")
    year, number = map(int, month.split("-"))
    return year * 12 + number - 1


def month_index(month: str, origin: str) -> int:
    index = _calendar_number(month) - _calendar_number(origin)
    if index < 1:
        raise ValueError(f"month {month} is before t=1 of panel origin {origin}")
    return index


def calendar_month(index: int, origin: str) -> str:
    if isinstance(index, bool) or not isinstance(index, int) or index < 1:
        raise ValueError(f"invalid horizon month {index!r}")
    year, month = divmod(_calendar_number(origin) + index, 12)
    return f"{year:04d}-{month + 1:02d}"


def _number(value: float) -> Decimal:
    if isinstance(value, bool) or not math.isfinite(value):
        raise ValueError("a translated value must be a finite number")
    return Decimal.from_float(float(value))


def _own_level(quantity: QuantityData, last: Decimal) -> Decimal:
    if quantity.transform == "identity":
        return last
    if quantity.transform == "log100":
        return (last / 100).exp()
    if quantity.transform == "logit":
        return 1 / (1 + (-last).exp())
    raise ValueError(f"unknown transform {quantity.transform!r} for {quantity.id}")


def _coordinate(quantity: QuantityData, level: Decimal) -> Decimal:
    if quantity.transform == "identity":
        return level
    if quantity.transform == "log100":
        if level <= 0:
            raise ValueError(f"{quantity.id} needs a positive level for log100")
        return 100 * level.ln()
    if quantity.transform == "logit":
        if not 0 < level < 1:
            raise ValueError(f"{quantity.id} needs a level strictly between 0 and 1")
        return (level / (1 - level)).ln()
    raise ValueError(f"unknown transform {quantity.transform!r} for {quantity.id}")


def percent_delta(value: float) -> float:
    """Correctly rounded ``100 ln(1 + p/100)`` for a scalar percentage."""
    with localcontext() as context:
        context.prec = 80
        ratio = 1 + _number(value) / 100
        if ratio <= 0:
            raise ValueError("a percent change must be greater than -100")
        return float(100 * ratio.ln())


def to_scale(quantity: QuantityData, value: float, unit: Unit, last: float) -> float:
    """An absolute model coordinate from a level or a reference-relative change.

    ``last`` is the last finite panel coordinate. Percent is reserved for log100;
    its displacement is added once to the stored coordinate.
    """
    if unit == "percent":
        if quantity.transform != "log100":
            raise ValueError(f"percent requires log100: {quantity.id}")
        return last + percent_delta(value)
    with localcontext() as context:
        context.prec = 80
        level = _number(value)
        if unit == "change":
            level += _own_level(quantity, _number(last))
        elif unit != "level":
            raise ValueError(f"unknown unit {unit!r}")
        return float(_coordinate(quantity, level))


def shift_to_scale(quantity: QuantityData, value: float, unit: Unit, last: float) -> float:
    """A constant coordinate displacement; changes use the last known reference.

    A log100 shift expressed in physical units is the relative move from that
    reference, not an additive physical move recalculated on every simulated path.
    """
    if unit == "percent":
        if quantity.transform != "log100":
            raise ValueError(f"percent requires log100: {quantity.id}")
        return percent_delta(value)
    if unit not in ("level", "change"):
        raise ValueError(f"unknown unit {unit!r}")
    if quantity.transform == "identity":
        return float(_number(value))
    with localcontext() as context:
        context.prec = 80
        anchor = _number(last)
        own = _own_level(quantity, anchor)
        if quantity.transform == "log100":
            ratio = 1 + _number(value) / own
            if ratio <= 0:
                raise ValueError(f"{quantity.id} shift reaches a non-positive level")
            return float(100 * ratio.ln())
        return float(_coordinate(quantity, own + _number(value)) - anchor)


def _functional(observation, data: TranslationData) -> tuple[QuantityData, float]:
    quantities = [data.quantity(term.quantity) for term in observation.terms]
    transforms = {quantity.transform for quantity in quantities}
    if len(transforms) != 1:
        raise ValueError("an observation mixes registry transforms; no exact scalar translation")
    quantity = quantities[0]
    if isinstance(observation, Expectation) and quantity.transform != "identity":
        names = ", ".join(term.quantity for term in observation.terms)
        raise ValueError(
            f"expectation on {names} needs a nonlinear mean evaluator; E(log X) is not log E(X)"
        )
    if quantity.transform == "logit" and len(quantities) != 1:
        raise ValueError("a multi-term logit observation has no exact scalar translation")
    if quantity.transform == "log100":
        total = sum(term.weight for term in observation.terms)
        if total not in (0.0, 1.0):
            raise ValueError("a log100 aggregate needs weights summing to zero or one")
        if total == 0 and observation.unit != "percent":
            raise ValueError("a zero-sum log100 aggregate requires percent units")
    anchor = sum(
        term.weight * data.last_model.get(term.quantity, 0.0) for term in observation.terms
    )
    if observation.unit != "level":
        missing = [
            term.quantity for term in observation.terms if term.quantity not in data.last_model
        ]
        if missing:
            raise ValueError(f"no last known value for {', '.join(missing)}")
    return quantity, anchor


def observation_bounds(observation, data: TranslationData) -> dict[str, float | None]:
    """Bounds of a scalar observable in registry coordinates.

    A homogeneous log100 support/probability aggregate is geometric: zero-sum
    weights describe a ratio (percent units), unit-sum weights describe a level.
    This is not an arithmetic sum of physical prices, nor an arithmetic mean.
    """
    quantity, anchor = _functional(observation, data)
    return {
        name: None
        if getattr(observation, name, None) is None
        else to_scale(quantity, getattr(observation, name), observation.unit, anchor)
        for name in ("low", "high", "value")
    }


def tolerance_of(terms, data: TranslationData) -> float:
    movements = [typical_movement(history, 1) for history in data.history.values()]
    finite = [value for value in movements if np.isfinite(value) and value > 0]
    fallback = float(np.median(finite)) if finite else 1e-6
    total = 0.0
    for term in terms:
        history = data.history.get(term.variable)
        movement = typical_movement(history, 1) if history is not None else float("nan")
        if not np.isfinite(movement) or movement <= 0:
            movement = fallback
        total += abs(term.weight) * 0.1 * movement
    return float(total)


def _terms(observation, data: TranslationData, horizon: int) -> tuple[LinearTerm, ...]:
    terms = []
    for term in observation.terms:
        month = month_index(term.month, data.origin)
        if month > horizon:
            raise ValueError(f"month {term.month} is beyond the horizon of {horizon}")
        terms.append(LinearTerm(term.quantity, month, term.weight))
    return tuple(terms)


@dataclass(frozen=True)
class TranslatedObservation:
    """One numeric observable; support points have not yet been widened."""

    kind: Literal["constraint", "expectation", "probability"]
    terms: tuple[LinearTerm, ...]
    low: float | None
    high: float | None
    value: float | None
    source: Observation
    at_least: float | None = None

    @property
    def on_driver(self) -> bool:
        return any(":" in term.variable for term in self.terms)

    @property
    def point(self) -> bool:
        return self.value is not None


def observations_of(
    scenario: Scenario, data: TranslationData, horizon: int
) -> tuple[TranslatedObservation, ...]:
    rows = []
    for observation in scenario.observations:
        terms = _terms(observation, data, horizon)
        if len({term.on_driver for term in observation.terms}) != 1:
            raise ValueError("a condition mixes macro variables and drivers")
        bounds = observation_bounds(observation, data)
        rows.append(
            TranslatedObservation(
                kind=observation.kind,
                terms=terms,
                **bounds,
                source=observation,
                at_least=getattr(observation, "at_least", None),
            )
        )
    return tuple(rows)


def observation_constraints(
    scenario: Scenario, data: TranslationData, horizon: int, *, on_driver=False
):
    rows = []
    for observation in observations_of(scenario, data, horizon):
        if observation.kind != "constraint" or observation.on_driver != on_driver:
            continue
        tolerance = tolerance_of(observation.terms, data)
        rows.append(
            LinearConstraint(
                observation.terms,
                observation.value - tolerance if observation.point else observation.low,
                observation.value + tolerance if observation.point else observation.high,
                point=observation.point,
                tolerance=tolerance,
            )
        )
    return tuple(rows)


def draw_interventions(
    scenario: Scenario, groups: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray | None]]:
    """The value and the activity of every intervention month, one row per group.

    The numbers are returned in the unit the file states, before the translation to the model
    scale: a ``uniform`` interval is a law of that very number, not of its translation. The
    second member is the activity mask for ``with_probability`` and ``None`` when the
    intervention is certain.
    """
    rng = np.random.default_rng([seed, INTERVENTION_STREAM])
    drawn = []
    for intervention in scenario.interventions:
        values, active = [], []
        for value in intervention.values:
            if isinstance(value, UniformValue):
                low, high = value.uniform
                values.append(rng.uniform(low, high, size=groups))
                active.append(None)
            elif isinstance(value, ProbabilityValue):
                values.append(np.full(groups, float(value.value)))
                active.append(rng.random(groups) < float(value.with_probability))
            else:
                values.append(np.full(groups, float(value)))
                active.append(None)
        mask = (
            None
            if all(item is None for item in active)
            else np.stack(
                [np.ones(groups, dtype=bool) if item is None else item for item in active], axis=1
            )
        )
        drawn.append((np.stack(values, axis=1), mask))
    return drawn


def plan_of(scenario: Scenario, data: TranslationData, horizon: int, *, groups=1, inner=1, seed=0):
    """Produce neutral numeric instructions; no model or model version is read."""
    from ..model_sdk.contracts import Intervention, ScenarioPlan

    if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 360:
        raise ValueError("horizon must be an integer from 1 to 360")
    if isinstance(groups, bool) or not isinstance(groups, int) or groups < 1:
        raise ValueError("groups must be a positive integer")
    drawn = draw_interventions(scenario, groups, seed)
    interventions = []
    for intervention, (units, active) in zip(scenario.interventions, drawn, strict=True):
        quantity = data.quantity(intervention.quantity)
        months = tuple(month_index(month, data.origin) for month in intervention.months)
        for calendar, month in zip(intervention.months, months, strict=True):
            if month > horizon:
                raise ValueError(f"month {calendar} is beyond the horizon of {horizon}")
        if intervention.unit != "level" and quantity.id not in data.last_model:
            raise ValueError(f"no last known value for {quantity.id}")
        converter = to_scale if intervention.kind == "set" else shift_to_scale
        last = data.last_model.get(quantity.id, 0.0)
        certain = not any(
            isinstance(value, UniformValue | ProbabilityValue) for value in intervention.values
        )
        if certain:
            converted = [
                converter(quantity, value, intervention.unit, last) for value in intervention.values
            ]
            values = np.tile(np.asarray(converted, dtype=float), (groups, 1))
        else:
            values = np.empty(units.shape, dtype=float)
            for row in range(groups):
                for position in range(units.shape[1]):
                    values[row, position] = converter(
                        quantity, float(units[row, position]), intervention.unit, last
                    )
        interventions.append(
            Intervention(
                target=quantity.id,
                months=months,
                kind=intervention.kind,
                values=values,
                active=active,
            )
        )
    return ScenarioPlan(
        horizon=horizon,
        constraints=observation_constraints(scenario, data, horizon),
        interventions=tuple(interventions),
        inner=inner,
    )
