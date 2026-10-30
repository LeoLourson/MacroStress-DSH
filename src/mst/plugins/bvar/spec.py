"""The estimation specification of the macro model: ``spec.toml``."""

import tomllib
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator

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
    seasonal: list[str] = []
    volvol_mean: Positive
    volvol_df: Annotated[float, Field(strict=True, gt=2)]
    sign_lag: Annotated[int, Field(strict=True, ge=0)]
    rotation_tries: Count
    sampler: Sampler
    dummies: list[Dummy] = []

    _start = field_validator("start")(first_of_month)


def load_spec(path: Path = SPEC_PATH) -> Spec:
    with path.open("rb") as file:
        return Spec.model_validate(tomllib.load(file))
