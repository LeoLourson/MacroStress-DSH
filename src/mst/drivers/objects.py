"""Objects and drivers: what the registry declares and what the catalogs discover.

An object is something a quantity is measured on: an activity, a region, a company, a bond
issue. ``codes`` holds the identifiers a position can be matched by, per kind (``okved``,
``okato``, ``secid``, ``isin``, ``inn``); a position names a kind and its own code, and the deepest
object one of whose codes is a prefix of it is the position's object (``Registry.match``).
"""

from typing import Annotated

from pydantic import Field

from mst.contracts import Contract, Identifier

Name = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]
SeriesId = Annotated[str, Field(pattern=r"^[a-z0-9_]+(\.[a-z0-9_]+)*$")]


class Object(Contract):
    id: Name
    class_: Name = Field(alias="class")
    label: Identifier
    parent: Name | None = None
    codes: dict[Name, list[Identifier]] = {}


class Driver(Contract):
    quantity: Name
    object: Name
    series: SeriesId | None = None

    @property
    def id(self) -> str:
        return f"{self.quantity}:{self.object}"


def series_id(quantity: str, object_: str) -> str:
    """The store series of a driver's history."""
    return f"drv.{quantity}.{object_}"
