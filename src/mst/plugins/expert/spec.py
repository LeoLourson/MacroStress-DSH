"""The specification of the expert model: ``spec.toml``, a contract like ``bvar.spec``."""

import tomllib
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator

from mst.contracts import Contract
from mst.model_sdk.contracts import LATENT_CANDIDATES

SPEC_PATH = Path(__file__).with_name("spec.toml")


def first_of_month(value: date) -> date:
    if value.day != 1:
        raise ValueError(f"{value} is not the first day of a month")
    return value


class Spec(Contract):
    """What the estimate of the expert model depends on beyond the answers.

    ``family`` builds the law of every answered quantity from its three quantiles;
    ``alpha`` is the tail index — the Pareto index of the ``pareto`` family and the degrees
    of freedom of the ``skew_t`` family and of the t copula, so one parameter sets the
    weight of the tails. ``levels`` is the level grid of the quantile tables,
    ``stream_draws`` the months of the stationary shock stream behind the covariance the
    shocks contribute, ``candidates`` the candidates of the named causes per value of R
    and ``subset_samples`` the samples per level of subset simulation.
    """

    family: Literal["pareto", "skew_t"]
    alpha: Annotated[float, Field(strict=True, gt=2)] = 4.0
    levels: Annotated[int, Field(strict=True, ge=33)] = 513
    stream_draws: Annotated[int, Field(strict=True, ge=1000)] = 200_000
    candidates: Annotated[int, Field(strict=True, ge=1)] = LATENT_CANDIDATES
    subset_samples: Annotated[int, Field(strict=True, ge=100)] = 2000
    history_start: date
    seed: Annotated[int, Field(strict=True, ge=0, le=2**32 - 1)]

    _history_start = field_validator("history_start")(first_of_month)


def load_spec(path: Path = SPEC_PATH) -> Spec:
    with path.open("rb") as file:
        return Spec.model_validate(tomllib.load(file))
