"""Macro variables: the data contract of level 0
(docs/concept/bvar-method-legacy.md, section 2).

An id starts with its block, ``gl.`` or ``ru.``. Global variables move Russian ones and never
the reverse. Models work with ``y = g(X)``: ``identity`` keeps rates and spreads in their
units, ``log100`` takes ``100 ln X`` of prices, indices and volumes, so a difference of two
values is about a percentage change.

The registry names the variables every macro model is estimated on and every scenario
constraint refers to. The named shocks — the ``J`` set with the signs a model is identified
by — belong to the macro model plugin that uses them (``mst.plugins.bvar.signs``), and a
scenario's driving shocks are checked against the version it runs on.
"""

import tomllib
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, model_validator

from mst.contracts import Contract, Identifier
from mst.scenario.contracts import Scenario

REGISTRY_PATH = Path(__file__).with_name("registry.toml")

BlockId = Annotated[str, Field(pattern=r"^(gl|ru)\.[a-z0-9_]+$")]
Block = Literal["gl", "ru"]


def block_of(id_: str) -> Block:
    return id_.split(".", 1)[0]


class Variable(Contract):
    id: BlockId
    label: Identifier
    unit: Identifier
    transform: Literal["identity", "log100"]
    source: Identifier
    seasonal: bool = False
    """The series is stored with its calendar pattern, which the panel build removes."""

    @property
    def block(self) -> Block:
        return block_of(self.id)

    def to_model(self, x):
        """``g(X)``: the value in the scale the models use."""
        if self.transform == "identity":
            return x
        if np.any(np.asarray(x) <= 0):
            raise ValueError(f"{self.id} takes 100 ln X and needs positive values")
        return 100 * np.log(x)

    def from_model(self, y):
        """``X = g^-1(y)``: the value in the variable's own unit."""
        return y if self.transform == "identity" else np.exp(np.asarray(y) / 100)


class Registry(Contract):
    """The macro variables: what every macro model estimates on and a scenario constrains."""

    variables: list[Variable] = Field(min_length=1)

    @model_validator(mode="after")
    def known_names(self):
        known = set()
        for item in self.variables:
            if item.id in known:
                raise ValueError(f"{item.id} is registered twice")
            known.add(item.id)
        return self

    def variable(self, id_: str) -> Variable:
        for item in self.variables:
            if item.id == id_:
                return item
        raise KeyError(id_)

    def check(self, scenario: Scenario) -> None:
        """Raises ``ValueError`` naming every variable the registry does not know."""
        variables = {item.id for item in self.variables}
        problems = []
        for constraint in scenario.constraints:
            for term in constraint.terms:
                if term.variable not in variables:
                    problems.append(f"unknown variable {term.variable}")
        if problems:
            raise ValueError("; ".join(dict.fromkeys(problems)))


def load_registry(path: Path = REGISTRY_PATH) -> Registry:
    with path.open("rb") as file:
        return Registry.model_validate(tomllib.load(file))
