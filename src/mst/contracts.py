"""Wire conventions: money in millions of its currency, rates as fractions, time in months/days."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Identifier = Annotated[str, Field(min_length=1, pattern=r"\S")]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Probability = Annotated[Number, Field(ge=0, le=1)]
NonNegative = Annotated[Number, Field(ge=0)]
Positive = Annotated[Number, Field(gt=0)]
Horizon = Annotated[int, Field(strict=True, ge=1, le=360)]
Seed = Annotated[int, Field(strict=True, ge=0, le=2**32 - 1)]
#: An ISO 4217 code: the currency of a position, a portfolio's base currency.
Currency = Annotated[str, Field(pattern=r"^[A-Z]{3}$")]
#: A kind of code a position carries and an object may cover (``Registry.match``).
CodeKind = Literal["secid", "isin", "inn", "okved", "okato"]
Status = Literal["ok", "needs_review"]
#: The status of a calculated position: ``invalid`` never enters portfolio arithmetic.
ResultStatus = Literal["ok", "needs_review", "invalid"]

__all__ = [
    "CodeKind",
    "Contract",
    "Currency",
    "Horizon",
    "Identifier",
    "NonNegative",
    "Number",
    "Positive",
    "Probability",
    "ResultStatus",
    "Seed",
    "Sha256",
    "Status",
]


class Contract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        validate_default=True,
        validate_assignment=True,
        revalidate_instances="always",
    )

