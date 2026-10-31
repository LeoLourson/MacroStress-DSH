"""The named shocks and the regimes of the MS-VAR: the ``J`` set of every version.

The signs are the identification of this model, so they belong to it, exactly as the BVAR's
belong to the BVAR. The ids of the five economic shocks match the BVAR's on purpose: a
scenario's driving shocks are checked against the version's ``J`` set, and the same names
keep a scenario portable between the two macro models. The regimes are elements of the same
set — a scenario names ``regime.stress`` to hold the regime path of the event fixed over
the horizon — and their declaration order is the volatility order the sampler enforces
(``estimate`` keeps the regime declared first the least volatile).
"""

import tomllib
from pathlib import Path
from typing import Annotated

from pydantic import Field, model_validator

from mst.contracts import Contract, Identifier
from mst.macro.registry import BlockId, block_of, load_registry

SIGNS_PATH = Path(__file__).with_name("signs.toml")

RegimeId = Annotated[str, Field(pattern=r"^regime\.[a-z0-9_]+$")]


class Shock(Contract):
    """A named shock moves ``rises`` up and ``falls`` down when it occurs."""

    id: BlockId
    label: Identifier
    rises: list[BlockId] = []
    falls: list[BlockId] = []

    @model_validator(mode="after")
    def consistent_signs(self):
        if not self.rises and not self.falls:
            raise ValueError(f"{self.id} restricts no variable")
        for both in sorted(set(self.rises) & set(self.falls)):
            raise ValueError(f"{self.id} both raises and lowers {both}")
        if block_of(self.id) == "ru":
            for moved in self.rises + self.falls:
                if block_of(moved) == "gl":
                    raise ValueError(f"Russian shock {self.id} cannot move {moved}")
        return self


class Regime(Contract):
    """A discrete state of the macro model, addressable by a scenario like a shock."""

    id: RegimeId
    label: Identifier
    note: str = ""


class Signs(Contract):
    """The sign table of the model and its regimes: the whole ``J`` set of a version."""

    shocks: list[Shock] = []
    regimes: list[Regime] = []

    @model_validator(mode="after")
    def known_names(self):
        seen: set[str] = set()
        for shock in self.shocks:
            if shock.id in seen:
                raise ValueError(f"shock {shock.id} is registered twice")
            seen.add(shock.id)
        if len(self.regimes) < 2:
            raise ValueError("a switching model needs at least two regimes")
        for regime in self.regimes:
            if regime.id in seen:
                raise ValueError(f"{regime.id} is registered twice")
            seen.add(regime.id)
        return self

    @property
    def ids(self) -> list[str]:
        return [shock.id for shock in self.shocks]

    @property
    def regime_ids(self) -> list[str]:
        return [regime.id for regime in self.regimes]


def check_variables(signs: Signs, known: set[str] | None = None) -> Signs:
    """Raises ``ValueError`` naming a shock that restricts a variable the registry lacks."""
    if known is None:
        known = {variable.id for variable in load_registry().variables}
    for shock in signs.shocks:
        for moved in shock.rises + shock.falls:
            if moved not in known:
                raise ValueError(
                    f"{shock.id} restricts unknown variable {moved}: "
                    "the macro registry does not hold it"
                )
    return signs


def load_signs(path: Path = SIGNS_PATH) -> Signs:
    with path.open("rb") as file:
        return check_variables(Signs.model_validate(tomllib.load(file)))
