"""The plan of the forecast check set: what is fixed before any result is read."""

import tomllib
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Splice(_Strict):
    """A month whose change of the named series is a statistical splice, not a move."""

    month: date
    variables: list[str] = Field(min_length=1)


class Gates(_Strict):
    """The thresholds every verdict reads; changing one is a change of the plan."""

    width: tuple[float, float] = (0.75, 1.40)
    """The model's quantile width over the historical one."""
    width_share: float = Field(default=0.8, gt=0, le=1)
    """The share of the macro variables of one horizon whose width must pass."""
    correlation: float = Field(default=0.15, gt=0)
    """The largest distance of the mean pair correlation of the shares from history."""

    @model_validator(mode="after")
    def ordered(self):
        for name in ("width",):
            low, high = getattr(self, name)
            if not 0 < low < high:
                raise ValueError(f"{name} needs 0 < low < high")
        return self


class Events(_Strict):
    """The scenarios computed as ordinary runs of the workspace, and their base."""

    base: str
    """The scenario without an event the medians by month are compared with."""
    scenarios: list[str] = Field(min_length=1)
    draws: int = Field(default=2000, ge=100, strict=True)
    contributions: bool = True


class Plan(_Strict):
    portfolio: str
    """The stored portfolio of the workspace whose losses are checked."""
    history_start: date
    """The first change month of every historical comparison."""
    horizons: list[int] = Field(default=[1, 3, 12], min_length=1)
    band: float = Field(default=0.90, gt=0, lt=1)
    """The central band: its quantiles give the width and the coverage."""
    draws: int = Field(default=2000, ge=100, strict=True)
    seed: int = Field(default=20261115, ge=0, strict=True)
    splices: list[Splice] = []
    gates: Gates = Gates()
    events: Events | None = None

    @model_validator(mode="after")
    def horizons_positive(self):
        if any(h < 1 for h in self.horizons) or len(set(self.horizons)) != len(self.horizons):
            raise ValueError("horizons are distinct positive months")
        return self

    @property
    def quantiles(self) -> tuple[float, float]:
        tail = (1 - self.band) / 2
        return tail, 1 - tail

    @property
    def horizon(self) -> int:
        return max(self.horizons)


def load_plan(path: Path) -> Plan:
    with Path(path).open("rb") as stream:
        return Plan.model_validate(tomllib.load(stream))
