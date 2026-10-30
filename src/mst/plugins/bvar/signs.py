"""The named shocks of the BVAR and their sign restrictions (formalization, section 4.2).

The signs are the identification of this model, so they belong to it: a different macro model
identifies its own shocks. The ids are the ``J`` set of every version of this model — a
scenario's driving shocks are checked against them — and ``identify`` keeps the posterior
draws whose rotation moves the restricted variables the required way.
"""

import tomllib
from pathlib import Path

from pydantic import model_validator

from mst.contracts import Contract, Identifier
from mst.macro.registry import BlockId, block_of, load_registry

SIGNS_PATH = Path(__file__).with_name("signs.toml")


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


class Signs(Contract):
    """The sign table of the model: every shock it is identified by."""

    shocks: list[Shock] = []

    @model_validator(mode="after")
    def known_names(self):
        seen = set()
        for shock in self.shocks:
            if shock.id in seen:
                raise ValueError(f"shock {shock.id} is registered twice")
            seen.add(shock.id)
        return self

    @property
    def ids(self) -> list[str]:
        return [shock.id for shock in self.shocks]


def check_variables(signs: Signs, known: set[str] | None = None) -> Signs:
    """Raises ``ValueError`` naming a shock that restricts a variable the registry lacks.

    The table is the model's own file, so an unknown id would otherwise surface as a bare
    ``KeyError`` from ``identify``, deep inside estimation.
    """
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
