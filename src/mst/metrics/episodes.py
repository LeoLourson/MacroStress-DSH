"""The episodes of the backtest: historical events the models are judged on.

An episode names the variables whose realized paths fix the scenario (they are constrained
month by month, exactly as they happened), the shocks the sources of the time held
responsible, and the corpus keys of those sources. Each episode also names its ``half`` of the
plan — tuning fits the models, validation checks them, both halves non-empty — and the
``source`` of its scenario, ``manual`` or ``llm``, which the scenario utility reports apart.
The set of episodes is a plan, not a fixed list: it grows as new historical events are carded,
and the marks are data of this file, not code. The window ends ``after_months`` months after
the start. An exclusion drops the months of a variable from its ``start`` on when the history
ends before it — an index splice makes the realized level incomparable.
"""

import tomllib
from datetime import date, timedelta
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator

from mst.contracts import Contract, Identifier, Seed

EPISODES_PATH = Path(__file__).with_name("episodes.toml")
Count = Annotated[int, Field(strict=True, ge=1)]
Half = Literal["tuning", "validation"]
"""Which half of the episode plan an episode belongs to: tuning fits the models, validation checks
them. The split is fixed before the models are fitted; the final table of the quality metrics
is built over all episodes regardless of it."""
Source = Literal["manual", "llm"]
"""Who formalised the episode's scenario: a human, or an LLM reading the corpus. The scenario
utility reports the two apart, because an LLM that has seen the outcomes of past
episodes is only trusted in calculations forward of the date it was formalised."""


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
    half: Half
    source: Source
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

    @field_validator("episodes")
    @classmethod
    def _both_halves(cls, episodes: list[EpisodeCheck]) -> list[EpisodeCheck]:
        """Both halves of the episode plan are non-empty.

        A plan marked entirely as tuning has no validation half, so fitting could not be told
        from checking; the check is over the plan as a whole, whatever its size.
        """
        named = {episode.half for episode in episodes}
        if missing := {"tuning", "validation"} - named:
            raise ValueError(f"no episodes are marked {', '.join(sorted(missing))}")
        return episodes


def load_episodes(path: Path = EPISODES_PATH) -> Episodes:
    with path.open("rb") as file:
        return Episodes.model_validate({**tomllib.load(file), "path": path})


def calculation_date(start: date) -> date:
    """The day before the episode starts: the date the models are estimated on."""
    return start - timedelta(days=1)
