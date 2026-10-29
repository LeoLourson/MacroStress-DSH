"""``portfolios/<id>.json``: the positions of one portfolio and the currency they are valued in.

The file is a name, the base currency of the portfolio's values and its positions. A position is
``mst.plugin_sdk.model.Position``: its instrument type, notional in millions of its currency, the
codes of the object it is exposed to and the terms its model reads. The id is the file name;
``mst.portfolio.tool`` reads and writes the files, a run takes the positions of one.
"""

from pydantic import Field, model_validator

from ..contracts import Contract, Currency
from ..plugin_sdk.model import Position


class Portfolio(Contract):
    """One portfolio: its name, its base currency and its positions."""

    name: str = ""
    base_currency: Currency = "RUB"
    positions: list[Position] = Field(min_length=1)

    @model_validator(mode="after")
    def unique(self):
        ids = [position.id for position in self.positions]
        if len(set(ids)) != len(ids):
            raise ValueError("two positions share one id")
        return self
