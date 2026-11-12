"""Scenario statements in calendar months and the quantities' own units.

An observation names a scalar functional, not the coordinates of a model. The
translation boundary resolves its units and the panel's calendar origin.
"""

from typing import Annotated, Literal

from pydantic import BeforeValidator, Field, model_validator

from ..contracts import Contract, Identifier, Number, Probability

Month = Annotated[str, Field(strict=True, pattern=r"^[0-9]{4}-(0[1-9]|1[0-2])$")]
Unit = Literal["level", "change", "percent"]


def _the_word(value):
    if value in (["assumption"], ["event"]):
        return value[0]
    if isinstance(value, list):
        for word in ("event", "assumption"):
            if word in value:
                raise ValueError(
                    f"the word {word} inside a list of fragments: basis is the word event, "
                    "the word assumption, or a list of fragment ids"
                )
    return value


Basis = Annotated[
    list[Identifier] | Literal["assumption", "event"],
    BeforeValidator(_the_word),
    Field(min_length=1),
]


class UniformValue(Contract):
    uniform: tuple[Number, Number]

    @model_validator(mode="after")
    def ordered(self):
        if self.uniform[0] > self.uniform[1]:
            raise ValueError("uniform requires low <= high")
        return self


class ProbabilityValue(Contract):
    with_probability: Probability
    value: Number


InterventionValue = Number | UniformValue | ProbabilityValue


class Intervention(Contract):
    quantity: Identifier
    months: list[Month] = Field(min_length=1)
    kind: Literal["set", "shift"]
    values: list[InterventionValue] = Field(min_length=1)
    unit: Unit
    reason: Identifier
    basis: Basis

    @model_validator(mode="after")
    def aligned(self):
        if len(self.months) != len(self.values):
            raise ValueError("an intervention needs one value for each month")
        if len(set(self.months)) != len(self.months):
            raise ValueError("an intervention names a calendar month twice")
        return self

    @property
    def on_driver(self) -> bool:
        return ":" in self.quantity


class Term(Contract):
    quantity: Identifier
    month: Month
    weight: Number

    @property
    def on_driver(self) -> bool:
        return ":" in self.quantity

    @model_validator(mode="after")
    def non_zero(self):
        if self.weight == 0:
            raise ValueError("a term needs a non-zero weight")
        return self


def _unique_terms(terms):
    seen = set()
    for term in terms:
        key = (term.quantity, term.month)
        if key in seen:
            raise ValueError(f"{term.quantity} in month {term.month} appears twice")
        seen.add(key)


class Constraint(Contract):
    kind: Literal["constraint"] = "constraint"
    terms: list[Term] = Field(min_length=1)
    unit: Unit
    low: Number | None = None
    high: Number | None = None
    value: Number | None = None
    basis: Basis

    @model_validator(mode="after")
    def well_formed(self):
        interval = self.low is not None or self.high is not None
        if interval and self.value is not None:
            raise ValueError("a constraint names either an interval or a value, not both")
        if not interval and self.value is None:
            raise ValueError("a constraint needs either low and high, or a value")
        if interval and (self.low is None or self.high is None):
            raise ValueError("an interval constraint needs both low and high")
        if interval and self.low >= self.high:
            raise ValueError("Require low < high: an exact value belongs in `value`")
        _unique_terms(self.terms)
        return self

    @property
    def point(self) -> bool:
        return self.value is not None


class Expectation(Contract):
    """Bounds on an arithmetic expectation; nonlinear means need an evaluator."""

    kind: Literal["expectation"] = "expectation"
    terms: list[Term] = Field(min_length=1)
    unit: Unit
    low: Number | None = None
    high: Number | None = None
    basis: Basis

    @model_validator(mode="after")
    def well_formed(self):
        if self.low is None and self.high is None:
            raise ValueError("an expectation needs low, high, or both")
        if self.low is not None and self.high is not None and self.low >= self.high:
            raise ValueError("Require low < high")
        _unique_terms(self.terms)
        return self


class ProbabilityCondition(Contract):
    kind: Literal["probability"] = "probability"
    terms: list[Term] = Field(min_length=1)
    unit: Unit
    low: Number
    high: Number
    at_least: Probability
    basis: Basis

    @model_validator(mode="after")
    def well_formed(self):
        if self.low >= self.high:
            raise ValueError("Require low < high")
        _unique_terms(self.terms)
        return self


Observation = Annotated[
    Constraint | Expectation | ProbabilityCondition, Field(discriminator="kind")
]


class Scenario(Contract):
    narrative: Identifier
    sources: list[Identifier] = []
    interventions: list[Intervention] = []
    observations: list[Observation] = []
    approved_by: str | None = None

    @model_validator(mode="after")
    def unique_interventions(self):
        seen = set()
        for intervention in self.interventions:
            for month in intervention.months:
                key = (intervention.quantity, month)
                if key in seen:
                    raise ValueError(
                        f"{intervention.quantity} in month {month} has two interventions"
                    )
                seen.add(key)
        return self


class Impact(Contract):
    """Temporary import bridge while the calculation layer moves to interventions."""

    target: Identifier
    start: Annotated[int, Field(strict=True, ge=1, le=360)]
    values: list[Number] = Field(min_length=1)
    reason: Identifier
    basis: Basis

    @property
    def end(self) -> int:
        return self.start + len(self.values) - 1

    def value_at(self, month: int) -> float:
        if self.start <= month <= self.end:
            return float(self.values[month - self.start])
        return 0.0

    @property
    def on_driver(self) -> bool:
        return ":" in self.target
