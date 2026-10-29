"""The estimation specification of the satellite models: ``spec.toml``."""

import tomllib
from pathlib import Path
from typing import Annotated

from pydantic import Field

from mst.contracts import Contract, Positive

SPEC_PATH = Path(__file__).with_name("spec.toml")

Count = Annotated[int, Field(strict=True, ge=1)]


class Sampler(Contract):
    draws: Count
    burn: Annotated[int, Field(strict=True, ge=0)]
    seed: Annotated[int, Field(strict=True, ge=0, le=2**32 - 1)]


class Spec(Contract):
    spread: Positive
    constant_spread: Positive
    holdout: Count
    reach: Count
    sampler: Sampler


def load_spec(path: Path = SPEC_PATH) -> Spec:
    with path.open("rb") as file:
        return Spec.model_validate(tomllib.load(file))
