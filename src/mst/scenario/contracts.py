"""The scenario of a stress test: a narrative, constraints, driving shocks and overlays.

``Scenario`` is the scenario (N, C, D, O) of docs/scenario-formalization.md, section 3; the
bounds of a constraint are in the model scale ``y = g(X)`` of each variable. Variable and
shock names are checked against the registry by ``mst.macro.registry.Registry.check``.
"""

from typing import Annotated

from pydantic import Field, model_validator

from ..contracts import Contract, Horizon, Identifier, Number

#: A month of the scenario horizon, 1..H; month 0 is the calculation date.
Month = Annotated[int, Field(strict=True, ge=1, le=360)]


class Term(Contract):
    variable: Identifier
    month: Month
    weight: Number

    @model_validator(mode="after")
    def non_zero(self):
        if self.weight == 0:
            raise ValueError("a term needs a non-zero weight")
        return self


class Constraint(Contract):
    """``low <= sum(weight * y[variable, month]) <= high``; ``low == high`` fixes the value."""

    terms: list[Term] = Field(min_length=1)
    low: Number
    high: Number
    note: str = ""

    @model_validator(mode="after")
    def well_formed(self):
        if self.low > self.high:
            raise ValueError("Require low <= high")
        seen = set()
        for term in self.terms:
            key = (term.variable, term.month)
            if key in seen:
                raise ValueError(f"{term.variable} in month {term.month} appears twice")
            seen.add(key)
        return self


class Overlay(Contract):
    """Adds ``delta`` to driver ``driver`` in months ``start..end``, with its reason."""

    driver: Identifier
    start: Month
    end: Month
    delta: Number
    reason: Identifier
    sources: list[Identifier] = Field(min_length=1)

    @model_validator(mode="after")
    def ordered(self):
        if self.start > self.end:
            raise ValueError("Require start <= end")
        return self


class Scenario(Contract):
    narrative: Identifier
    sources: list[Identifier] = []
    horizon: Horizon
    constraints: list[Constraint] = []
    #: Shocks that cause the constrained paths; empty means any shock may.
    driving_shocks: list[Identifier] = []
    overlays: list[Overlay] = []

    @model_validator(mode="after")
    def within_horizon(self):
        months = [term.month for c in self.constraints for term in c.terms]
        months += [overlay.end for overlay in self.overlays]
        for month in months:
            if month > self.horizon:
                raise ValueError(f"month {month} is beyond the horizon of {self.horizon}")
        seen = set()
        for shock in self.driving_shocks:
            if shock in seen:
                raise ValueError(f"{shock} is named twice")
            seen.add(shock)
        return self
