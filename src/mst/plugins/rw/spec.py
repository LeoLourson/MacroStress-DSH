"""Validated parameters of the joint Student random walk."""

import tomllib
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Break(BaseModel):
    model_config = ConfigDict(extra="forbid")

    month: date
    variables: list[str] = Field(min_length=1)

    @field_validator("month")
    @classmethod
    def first_of_month(cls, value: date) -> date:
        if value.day != 1:
            raise ValueError("break month must be the first day of a month")
        return value

    @field_validator("variables")
    @classmethod
    def distinct_variables(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("break variables must be distinct")
        return value


class Spec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: date = date(2012, 1, 1)
    drifts: list[str] = Field(
        default_factory=lambda: [
            "ru.cpi",
            "ru.usdrub",
            "ru.cnyrub",
            "ru.output",
            "ru.real_wage",
            "ru.retail",
            "ru.credit",
            "ru.housing_price",
            "ru.imoex",
        ]
    )
    breaks: list[Break] = Field(
        default_factory=lambda: [
            Break(
                month=date(2021, 1, 1),
                variables=["ru.spread_aa", "ru.spread_a", "ru.spread_bbb"],
            )
        ]
    )
    nu_bounds: tuple[float, float] = (2.5, 60.0)
    em_tol: float = Field(default=1e-8, gt=0, allow_inf_nan=False)
    em_iterations: int = Field(default=500, ge=1, strict=True)
    eig_floor: float = Field(default=1e-8, gt=0, lt=1, allow_inf_nan=False)
    min_changes: int = Field(default=3, ge=1, strict=True)
    seed: int = Field(default=20261029, ge=0, strict=True)
    reversion: list[str] = Field(default_factory=list)
    """Variables whose level mean-reverts with an estimated diagonal AR(1) coefficient.

    A variable not named here keeps ``A = 1``: a random walk with drift. A variable that is
    also in ``drifts`` uses the fitted AR(1) law, so its fixed neutral replaces the drift in
    the forecast mean; the drift still centres the changes before the scale estimate. The
    price block keeps ``A = 1``, because an AR(1) on a trending log level fits the trend
    rather than a return to a fixed level.
    """
    stochastic_scale: bool = True
    """Whether the monthly multiplier follows the estimated log AR(1) path from its state in
    the last month of the panel, so that the band widens after a turbulent month.
    """
    band_fit: bool = False
    """Fit the Student degrees of freedom to the central 90 percent band when sample_scale is on."""
    sample_scale: bool = True
    """Whether the level of the innovation covariance is the sample covariance of changes.

    The ECME t-scale downweights the clustered crisis months and understates the
    unconditional covariance; with this switch the scale is the sample covariance of the
    changes over ``regimes``, rescaled to the Student scale. The correlation
    and the variance of the transfer then come from the sample, not from the downweighted
    estimate.
    """
    scale_blocks: list[list[str]] = Field(default_factory=list)
    """Groups of variables whose large changes fall in months of their own.

    Every group has its own common monthly multiplier with its own ``nu``; the variables no
    group names share one more. Independent multipliers cap the correlation across groups,
    so a scenario carries less to the variables of other groups; with an empty list all the
    variables share one multiplier.
    """
    regimes: list[date] = Field(
        default_factory=lambda: [date(2014, 11, 1), date(2020, 3, 1), date(2022, 3, 1)]
    )
    """The first months of the regimes: the rouble float, the pandemic, the sanctions of 2022.

    The sample covariance and the momentum read the changes of the regime the panel ends in:
    a correlation estimated across a change of regime pulls the quantities a scenario does
    not fix in the direction of the old one.
    """
    regime_months: int = Field(default=36, ge=2, strict=True)
    """The shortest window: a regime younger than this many months is read together with the
    end of the one before it, since three years is the least the momentum can be fitted on."""

    momentum: bool = True
    """Whether the monthly changes keep a share of the previous change.

    The series are monthly means of daily values, and the mean of a walk over a month has
    changes that follow one another: a crisis spreads over two monthly means. With this switch
    every variable outside ``reversion`` gets a diagonal AR(1) on its changes,
    ``d_t - mu = phi (d_{t-1} - mu) + e_t``; ``phi`` is chosen so that the variance of
    ``momentum_horizon``-month changes over the scale window equals the sample one, while the
    monthly covariance stays the sample covariance.
    """
    momentum_horizon: int = Field(default=12, ge=2, strict=True)
    momentum_bound: float = Field(default=0.5, gt=0, lt=1, allow_inf_nan=False)
    """The largest absolute ``phi``: a short window cannot support a stronger persistence."""

    @field_validator("start")
    @classmethod
    def first_of_month(cls, value: date) -> date:
        if value.day != 1:
            raise ValueError("start must be the first day of a month")
        return value

    @field_validator("regimes")
    @classmethod
    def increasing_regimes(cls, value: list[date]) -> list[date]:
        if any(month.day != 1 for month in value):
            raise ValueError("a regime must start on the first day of a month")
        if any(later <= earlier for earlier, later in zip(value, value[1:], strict=False)):
            raise ValueError("regimes must be in increasing order")
        return value

    @field_validator("nu_bounds")
    @classmethod
    def valid_nu_bounds(cls, value: tuple[float, float]) -> tuple[float, float]:
        low, high = value
        if not 2 < low < high < float("inf"):
            raise ValueError("nu_bounds must be finite, increasing and above 2")
        return value

    @field_validator("drifts")
    @classmethod
    def distinct_drifts(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("drifts must be distinct")
        return value

    @field_validator("reversion")
    @classmethod
    def distinct_reversion(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("reversion variables must be distinct")
        return value

    @field_validator("scale_blocks")
    @classmethod
    def disjoint_blocks(cls, value: list[list[str]]) -> list[list[str]]:
        named = [variable for block in value for variable in block]
        if any(not block for block in value) or len(named) != len(set(named)):
            raise ValueError("scale blocks must be non-empty and disjoint")
        return value

    def validate_ids(self, ids: list[str]) -> None:
        requested = set(self.drifts) | set(self.reversion)
        requested.update(variable for block in self.scale_blocks for variable in block)
        for item in self.breaks:
            requested.update(item.variables)
        unknown = requested - set(ids)
        if unknown:
            raise ValueError(f"spec names unknown variables: {', '.join(sorted(unknown))}")


def load_spec(path: str | Path | None = None) -> Spec:
    source = Path(path) if path is not None else Path(__file__).with_name("spec.toml")
    with source.open("rb") as stream:
        return Spec.model_validate(tomllib.load(stream))
