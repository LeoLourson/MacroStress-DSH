"""The scenario of a stress test: narrative, constraints, causes, impacts and approval.

``Scenario`` is the scenario ``(N, m, O, D, C)`` of docs/concept/formalization.md, section 3:
the narrative with its sources, the model versions, the direct impacts, the causes the
scenario names among the coordinates of ``omega`` beyond the parameters, and the conditions.
The bounds of a constraint are in the model scale ``y = g(X)`` of each variable; a point is a
``value`` the platform expands to ``value ± tau``. Variable names are checked against the
registry by ``mst.macro.registry.Registry.check``; cause names — shocks and the hidden
coordinates the macro model declares — against the macro version by
``mst.scenario.build.validate``, and a scenario is not computed until a human has approved it.
"""

from typing import Annotated, Literal

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
    """One condition on the paths: an interval ``low <= w'y <= high`` or a point ``value``.

    A point is a value the scenario fixes exactly; the platform expands it to
    ``value ± tau`` with the tolerance of the variable's typical monthly movement
    (``mst.scenario.build.plan_of``), because an exact value has no probability under a
    continuous law and the canon asks for a half-width ``tau > 0``. Exactly one of the two
    forms is allowed: ``low`` with ``high`` strictly above it, or ``value`` alone — an
    interval of zero width is a point and belongs in ``value``.
    """

    terms: list[Term] = Field(min_length=1)
    low: Number | None = None
    high: Number | None = None
    value: Number | None = None
    basis: list[Identifier] | Literal["assumption"] = Field(min_length=1)
    """The corpus fragments behind the condition, or ``assumption`` when it is the author's."""

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
            raise ValueError(
                "Require low < high: an exact value belongs in `value`, which the platform "
                "expands to value ± tau"
            )
        seen = set()
        for term in self.terms:
            key = (term.variable, term.month)
            if key in seen:
                raise ValueError(f"{term.variable} in month {term.month} appears twice")
            seen.add(key)
        return self

    @property
    def point(self) -> bool:
        """Whether the constraint fixes its value exactly, as a point."""
        return self.value is not None


class Impact(Contract):
    """A direct impact ``o_l``: ``delta`` added to ``target`` in months ``start..end``.

    The target is a macro variable (level 0, its id names it) or a driver
    ``<quantity>:<object>`` of a level above, in the model scale of its level. The impact of
    a level is applied before the levels below it are simulated, so an impact on a macro
    variable reaches every satellite; an impact on a driver moves that driver alone. Every
    impact carries its reason and what it rests on: the corpus fragments it cites, or the
    word ``assumption`` when it is the author's own.
    """

    target: Identifier
    start: Month
    end: Month
    delta: Number
    reason: Identifier
    basis: list[Identifier] | Literal["assumption"] = Field(min_length=1)

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
    #: The causes ``D``: coordinates of ``omega`` beyond the parameters that the scenario
    #: holds responsible for the event — shock names and the hidden coordinates the macro
    #: model declares (a regime path, a volatility path). Empty means every coordinate
    #: except the parameters.
    causes: list[Identifier] = []
    #: The regime element the event asserts, fixed over the whole horizon; ``None`` when the
    #: event does not fix a regime. Naming a regime as a cause is a different thing: it says
    #: the scenario explains the event through the regime, without fixing it.
    regime_event: Identifier | None = None
    impacts: list[Impact] = []
    """The direct impacts ``O`` of the scenario, on the macro variables or on drivers."""
    approved_by: str | None = None
    """Who approved the scenario; the canon has a human assert it before it is computed."""

    @model_validator(mode="after")
    def within_horizon(self):
        months = [term.month for c in self.constraints for term in c.terms]
        months += [impact.end for impact in self.impacts]
        for month in months:
            if month > self.horizon:
                raise ValueError(f"month {month} is beyond the horizon of {self.horizon}")
        seen = set()
        for cause in self.causes:
            if cause in seen:
                raise ValueError(f"{cause} is named twice")
            seen.add(cause)
        return self
