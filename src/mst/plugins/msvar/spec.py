"""The estimation specification of the MS-VAR: ``spec.toml``."""

import tomllib
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator

from mst.contracts import Contract, Positive

SPEC_PATH = Path(__file__).with_name("spec.toml")

Count = Annotated[int, Field(strict=True, ge=1)]


def first_of_month(value: date) -> date:
    if value.day != 1:
        raise ValueError(f"{value} is not the first day of a month")
    return value


class Sampler(Contract):
    draws: Count
    burn: Annotated[int, Field(strict=True, ge=0)]
    thin: Count
    seed: Annotated[int, Field(strict=True, ge=0, le=2**32 - 1)]


class Spec(Contract):
    """The specification of one estimation; a version stores it as it was.

    There are no intervention dummies and no volatility block: the regime carries both. The
    crisis months of the panel are the months of the volatile regime, and the covariance of
    a regime is constant over its months — the heavy tails of the forecast come from the
    mixture over regimes, not from stochastic volatility inside one.
    """

    start: date
    lags: Count
    tightness: Positive
    stationary: list[str] = []
    sign_lag: Annotated[int, Field(strict=True, ge=0)]
    rotation_tries: Count
    rho_max: Annotated[float | None, Field(strict=True, ge=1.0)] = None
    """The ceiling on the companion-matrix spectral radius a parameter draw must meet in
    every regime to survive. ``None`` is no ceiling — the versions estimated before the
    filter carry it."""
    mu: Positive | None = None
    """The weight of the sum-of-coefficients dummy rows (Sims and Zha, 1998; Giannone,
    Lenza and Primiceri, 2015); ``None`` turns them off. Smaller is tighter: the sums of
    own-lag coefficients pull toward one, and the system away from explosive roots."""
    delta: Positive | None = None
    """The weight of the initial-observation dummy row; ``None`` turns it off. Smaller is
    tighter: the no-change forecast at the beginning of the sample gets better."""
    switching: Literal["all", "mean_covariance"] = "all"
    """What switches with the regime: every coefficient, or the mean and the covariance
    alone with one common lag block. ``all`` is the behaviour the versions estimated before
    the field carried."""
    transition_stay: Positive
    transition_switch: Positive
    sampler: Sampler

    _start = field_validator("start")(first_of_month)


def load_spec(path: Path = SPEC_PATH) -> Spec:
    with path.open("rb") as file:
        return Spec.model_validate(tomllib.load(file))
