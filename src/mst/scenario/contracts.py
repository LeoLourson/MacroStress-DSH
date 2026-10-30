"""The scenario of a stress test: narrative, constraints, driving shocks and impacts.

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


class Impact(Contract):
    """A direct impact ``o_l``: ``delta`` added to ``target`` in months ``start..end``.

    The target is a macro variable (level 0, its id names it) or a driver
    ``<quantity>:<object>`` of a level above, in the model scale of its level. The impact of
    a level is applied before the levels below it are simulated, so an impact on a macro
    variable reaches every satellite; an impact on a driver moves that driver alone. Every
    impact carries its reason and the corpus sources it rests on.
    """

    target: Identifier
    start: Month
    end: Month
    delta: Number
    reason: Identifier
    sources: list[Identifier] = Field(min_length=1)

    @property
    def on_driver(self) -> bool:
        """Whether the impact targets a driver rather than a macro variable."""
        return ":" in self.target

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
    impacts: list[Impact] = []
    """The direct impacts ``O`` of the scenario, on the macro variables or on drivers."""

    @model_validator(mode="before")
    @classmethod
    def old_overlays(cls, data: object) -> object:
        """Read the pre-0.2 field ``overlays`` as ``impacts``, so the old files still load.

        The old name is deprecated: a scenario that has ``impacts`` too is refused instead of
        silently preferring one of the two, and every other field is validated as usual.
        """
        if not isinstance(data, dict) or "overlays" not in data:
            return data
        if "impacts" in data:
            raise ValueError(
                "the scenario names both `overlays` (deprecated) and `impacts`: keep `impacts`"
            )
        mapped = []
        for item in data["overlays"] if isinstance(data["overlays"], list) else []:
            if isinstance(item, dict) and "driver" in item:
                if "target" in item:
                    raise ValueError(
                        "the old impact names both `driver` and `target`: keep `target`"
                    )
                driver = item["driver"]
                item = {key: value for key, value in item.items() if key != "driver"} | {
                    "target": driver
                }
            mapped.append(item)
        return {key: value for key, value in data.items() if key != "overlays"} | {
            "impacts": mapped
        }

    @model_validator(mode="after")
    def within_horizon(self):
        months = [term.month for c in self.constraints for term in c.terms]
        months += [impact.end for impact in self.impacts]
        for month in months:
            if month > self.horizon:
                raise ValueError(f"month {month} is beyond the horizon of {self.horizon}")
        seen = set()
        for shock in self.driving_shocks:
            if shock in seen:
                raise ValueError(f"{shock} is named twice")
            seen.add(shock)
        return self
