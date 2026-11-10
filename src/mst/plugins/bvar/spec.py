"""The estimation specification of the macro model: ``spec.toml``."""

import tomllib
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from mst.contracts import Contract, Positive

SPEC_PATH = Path(__file__).with_name("spec.toml")

Count = Annotated[int, Field(strict=True, ge=1)]


def first_of_month(value: date) -> date:
    if value.day != 1:
        raise ValueError(f"{value} is not the first day of a month")
    return value


class Dummy(Contract):
    month: date
    kind: Literal["impulse", "step"]
    note: str = ""

    _month = field_validator("month")(first_of_month)

    def value(self, month: date) -> float:
        if self.kind == "impulse":
            return float(month == self.month)
        return float(month >= self.month)


class Sampler(Contract):
    draws: Count
    burn: Annotated[int, Field(strict=True, ge=0)]
    thin: Count
    seed: Annotated[int, Field(strict=True, ge=0, le=2**32 - 1)]


class Spec(Contract):
    start: date
    lags: Count
    tightness: Positive
    volatility: Literal["constant", "stochastic"]
    stationary: list[str] = []
    volvol_mean: Positive
    volvol_df: Annotated[float, Field(strict=True, gt=2)]
    sign_lag: Annotated[int, Field(strict=True, ge=0)]
    rotation_tries: Count
    rho_max: Annotated[float | None, Field(strict=True, ge=1.0)] = None
    """The ceiling on the companion-matrix spectral radius a parameter draw must meet to
    survive. ``None`` is no ceiling — the versions estimated before the filter carry it."""
    mu: Positive | None = None
    """The weight of the sum-of-coefficients dummy rows (Sims and Zha, 1998; Giannone,
    Lenza and Primiceri, 2015); ``None`` turns them off. Smaller is tighter: the sums of
    own-lag coefficients pull toward one, and the system away from explosive roots."""
    delta: Positive | None = None
    """The weight of the initial-observation dummy row; ``None`` turns it off. Smaller is
    tighter: the no-change forecast at the beginning of the sample gets better."""
    dof_min: Annotated[float, Field(strict=True, gt=2)] = 3.0
    """The floor of the degrees-of-freedom grid of the Student tails (stochastic only).
    The shipped 5 sits above the kurtosis boundary at 4; the versions estimated before the
    field carried the 3."""
    differences: list[str] = []
    """The variables whose equation runs on monthly changes ``y_t - y_{t-1}`` against the
    changes of the others in the list and the levels of the rest. Their unit root is exact
    instead of a near-one root the sampling noise pushes above one, and the constant of their
    equation is a monthly drift. The version stores the equivalent VAR in levels with one
    lag more, so forecasting, conditions and identification read it unchanged. Empty keeps
    every equation in levels, as the versions estimated before the field."""
    sampler: Sampler
    dummies: list[Dummy] = []

    _start = field_validator("start")(first_of_month)

    @model_validator(mode="after")
    def _dummy_rows_act_on_levels(self):
        # The dummy rows state a no-change forecast at the levels of the first months; on a
        # variable in changes they would state it about the changes and pull its drift.
        if self.differences and (self.mu is not None or self.delta is not None):
            raise ValueError(
                "differences and the dummy rows mu, delta do not mix: the rows are written "
                "for levels; set mu and delta off when variables run on changes"
            )
        return self


def load_spec(path: Path = SPEC_PATH) -> Spec:
    with path.open("rb") as file:
        return Spec.model_validate(tomllib.load(file))
