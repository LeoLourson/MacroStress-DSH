"""What every source of a series declares about its releases."""

from pydantic import Field

from mst.contracts import Contract


class Release(Contract):
    lag_days: int = Field(strict=True, ge=0)
    """Days from the end of a month to the release of its value."""
    revised: bool
    """Whether the source revises values after their first release."""
