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
Status = Literal["ok", "needs_review"]

__all__ = [
    "Contract",
    "Horizon",
    "Identifier",
    "NonNegative",
    "Number",
    "Positive",
    "Probability",
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

