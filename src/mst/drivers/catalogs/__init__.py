"""Catalogs of drivers, by ``loader``. A new source of drivers is a class added here."""

from typing import Annotated

from pydantic import Field

from .base import Catalog, Discovery, name, slug
from .cbr import CbrActivities
from .moex import MoexBonds, MoexShares
from .rosstat import RosstatRegions

CatalogSource = Annotated[
    MoexShares | MoexBonds | CbrActivities | RosstatRegions,
    Field(discriminator="loader"),
]

__all__ = ["Catalog", "CatalogSource", "Discovery", "name", "slug"]
