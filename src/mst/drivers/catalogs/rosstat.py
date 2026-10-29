"""The catalog of territories of a Rosstat table by subject: Russia, districts, subjects.

The rows of the latest workbook matching ``quarterly`` are labelled ``<OKATO code> - <name>``:
643 is the root, a three-digit code a federal district, an eleven-digit code a subject, which
sits under the district row above it. Each territory is fetched by ``RosstatHousingPrice`` with
its code. Objects are named ``okato_<code>``, so two tables of the same territories (primary
and secondary market) share them.
"""

import re
from typing import Literal

import httpx

from mst.data.rosstat import RosstatHousingPrice, _label, _workbooks, territory_code

from ..objects import Object
from .base import Catalog, Discovery


class RosstatRegions(Catalog):
    loader: Literal["rosstat_regions"]
    page: str
    quarterly: str
    """The file pattern of the yearly workbooks, its group the year."""

    def discover(self, http: httpx.Client) -> Discovery:
        books = _workbooks(http, self.page, self.quarterly)
        _, sheets = max(books, key=lambda book: int(book[0].group(1)))
        rows = next(rows for rows in sheets.values() if any("I квартал" in r for r in rows))
        found, district = Discovery(), None
        for row in rows:
            label = _label(row)
            code = territory_code(label)
            if code is None:
                continue
            name = re.sub(r"^\S+\s+-\s+", "", label)
            if code == "643":
                parent, kind = None, "country"
            elif len(code) == 3:
                parent, kind, district = "okato_643", "district", f"okato_{code}"
            else:
                parent, kind = district or "okato_643", "subject"
            object_id = f"okato_{code}"
            if any(o.id == object_id for o in found.objects):
                continue
            found.objects.append(
                Object.model_validate(
                    {
                        "id": object_id,
                        "class": kind,
                        "label": name,
                        "parent": parent,
                        "codes": {"okato": [code]},
                    }
                )  # fmt: skip
            )
            found.drivers[object_id] = RosstatHousingPrice(
                loader="rosstat_housing_price", page=self.page, quarterly=self.quarterly,
                region=code, lag_days=self.lag_days, revised=self.revised,
            )  # fmt: skip
        return found
