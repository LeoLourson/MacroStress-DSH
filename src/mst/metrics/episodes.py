"""The episodes of the backtest: historical events the models are judged on.

An episode names the variables whose realized paths fix the scenario (they are constrained
month by month, exactly as they happened), the shocks the sources of the time held
responsible, and the corpus keys of those sources. The window ends ``after_months`` months
after the start. An exclusion drops the months of a variable from its ``start`` on when the
history ends before it — an index splice makes the realized level incomparable.
"""

import tomllib
from datetime import date, timedelta
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator

from mst.contracts import Contract, Identifier, Seed

EPISODES_PATH = Path(__file__).with_name("episodes.toml")
Count = Annotated[int, Field(strict=True, ge=1)]


def first_of_month(value: date) -> date:
    if value.day != 1:
        raise ValueError(f"{value} is not the first day of a month")
    return value


class Exclusion(Contract):
    variables: list[Identifier] = Field(min_length=1)
    start: date
    reason: Identifier

    _start = field_validator("start")(first_of_month)


class EpisodeCheck(Contract):
    id: Identifier
    constrained: list[Identifier] = Field(min_length=1)
    driving: list[Identifier] = Field(min_length=1)
    reason: Identifier
    """Why these shocks drove the episode, with the corpus keys of the sources of the time."""
    sources: list[Identifier] = Field(min_length=1)


class Episodes(Contract):
    """The plan of the backtest: the episodes, the window and the simulation of each one."""

    draws: Count
    seed: Seed
    after_months: Annotated[int, Field(strict=True, ge=1)]
    """The window ends this many months after the start month, as in the episode cards."""
    exclusions: list[Exclusion] = []
    episodes: list[EpisodeCheck] = Field(min_length=1)
    path: Path | None = Field(default=None, exclude=True)
    """The file the plan was loaded from, for the report's hash; not a TOML key."""


def load_episodes(path: Path = EPISODES_PATH) -> Episodes:
    with path.open("rb") as file:
        return Episodes.model_validate({**tomllib.load(file), "path": path})


def calculation_date(start: date) -> date:
    """The day before the episode starts: the date the models are estimated on."""
    return start - timedelta(days=1)
