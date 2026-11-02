"""The estimation specification of the satellite models: ``spec.toml``."""

import tomllib
from pathlib import Path
from typing import Annotated

from pydantic import Field

from mst.contracts import Contract, Identifier, Positive

SPEC_PATH = Path(__file__).with_name("spec.toml")

Count = Annotated[int, Field(strict=True, ge=1)]


class Sampler(Contract):
    draws: Count
    burn: Annotated[int, Field(strict=True, ge=0)]
    seed: Annotated[int, Field(strict=True, ge=0, le=2**32 - 1)]


class Spec(Contract):
    spread: Positive
    constant_spread: Positive
    rho_max: Annotated[float, Field(strict=True, gt=0, lt=1)]
    holdout: Count
    reach: Count
    differences: list[Identifier] = []
    """The quantities fitted on monthly changes: the long-run level of a trending driver is
    spurious, so its equation runs on Δd and Δx and the simulation accumulates the level.
    A quantity not named stays in levels."""
    sampler: Sampler


def load_spec(path: Path = SPEC_PATH) -> Spec:
    with path.open("rb") as file:
        return Spec.model_validate(tomllib.load(file))
