"""Validated parameters of the joint Student random walk."""

import tomllib
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Break(BaseModel):
    model_config = ConfigDict(extra="forbid")

    month: date
    variables: list[str] = Field(min_length=1)

    @field_validator("month")
    @classmethod
    def first_of_month(cls, value: date) -> date:
        if value.day != 1:
            raise ValueError("break month must be the first day of a month")
        return value

    @field_validator("variables")
    @classmethod
    def distinct_variables(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("break variables must be distinct")
        return value


class Spec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: date = date(2012, 1, 1)
    drifts: list[str] = Field(
        default_factory=lambda: [
            "ru.cpi",
            "ru.usdrub",
            "ru.cnyrub",
            "ru.output",
            "ru.real_wage",
            "ru.retail",
            "ru.credit",
            "ru.housing_price",
            "ru.imoex",
        ]
    )
    breaks: list[Break] = Field(
        default_factory=lambda: [
            Break(
                month=date(2021, 1, 1),
                variables=["ru.spread_aa", "ru.spread_a", "ru.spread_bbb"],
            )
        ]
    )
    nu_bounds: tuple[float, float] = (2.5, 60.0)
    em_tol: float = Field(default=1e-8, gt=0, allow_inf_nan=False)
    em_iterations: int = Field(default=500, ge=1, strict=True)
    eig_floor: float = Field(default=1e-8, gt=0, lt=1, allow_inf_nan=False)
    min_changes: int = Field(default=3, ge=1, strict=True)
    seed: int = Field(default=20261029, ge=0, strict=True)

    @field_validator("start")
    @classmethod
    def first_of_month(cls, value: date) -> date:
        if value.day != 1:
            raise ValueError("start must be the first day of a month")
        return value

    @field_validator("nu_bounds")
    @classmethod
    def valid_nu_bounds(cls, value: tuple[float, float]) -> tuple[float, float]:
        low, high = value
        if not 2 < low < high < float("inf"):
            raise ValueError("nu_bounds must be finite, increasing and above 2")
        return value

    @field_validator("drifts")
    @classmethod
    def distinct_drifts(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("drifts must be distinct")
        return value

    def validate_ids(self, ids: list[str]) -> None:
        requested = set(self.drifts)
        for item in self.breaks:
            requested.update(item.variables)
        unknown = requested - set(ids)
        if unknown:
            raise ValueError(f"spec names unknown variables: {', '.join(sorted(unknown))}")


def load_spec(path: str | Path | None = None) -> Spec:
    source = Path(path) if path is not None else Path(__file__).with_name("spec.toml")
    with source.open("rb") as stream:
        return Spec.model_validate(tomllib.load(stream))
