"""The catalog of activities of the Bank of Russia table of debt by activity (01_02_A).

Every row of the table is an activity; the indent of its label gives the level, so a part sits
under the section above it, and «ВСЕГО» is the root. ``okved`` gives the OKVED 2 codes an
activity covers (the table follows the OKVED 2007 grouping), by row label, so a borrower is
matched to the deepest activity of its code.
"""

import io
from typing import Literal

import httpx
import openpyxl

from mst.contracts import Identifier
from mst.data.cbr_stats import DEBT_BY_ACTIVITY, CbrOverdue, _activity

from ..objects import Object
from .base import Catalog, Discovery, slug

INDENT = 8
"""Spaces per level of the row labels."""


class CbrActivities(Catalog):
    loader: Literal["cbr_activities"]
    skip: list[str] = []
    """Row labels that are not activities, e.g. «на завершение расчетов»."""
    okved: dict[str, list[Identifier]] = {}

    def discover(self, http: httpx.Client) -> Discovery:
        response = http.get(DEBT_BY_ACTIVITY)
        response.raise_for_status()
        book = openpyxl.load_workbook(io.BytesIO(response.content), read_only=True, data_only=True)
        rows = list(book.worksheets[0].iter_rows(values_only=True))
        found, chain, seen = Discovery(), [], set()
        for row in rows[2:]:
            if not isinstance(row[0], str) or not row[0].strip():
                continue
            label = _activity(row[0])
            if label in self.skip:
                found.skipped[label] = "declared not an activity"
                continue
            root = label == "ВСЕГО"
            level = 0 if root else 1 + (len(row[0]) - len(row[0].lstrip(" "))) // INDENT
            object_id = "economy" if root else slug(label)
            while object_id in seen:
                object_id += "_x"
            seen.add(object_id)
            chain = chain[:level] + [object_id]
            fields = {"id": object_id, "class": f"activity_{level}", "label": label,
                      "parent": chain[level - 1] if level else None}  # fmt: skip
            if label in self.okved:
                fields["codes"] = {"okved": self.okved[label]}
            found.objects.append(Object.model_validate(fields))
            found.drivers[object_id] = CbrOverdue(
                loader="cbr_overdue", activity=label, lag_days=self.lag_days, revised=self.revised
            )
        unknown = sorted(set(self.okved) - {o.label for o in found.objects})
        if unknown:
            raise ValueError(f"okved names rows the table lacks: {', '.join(unknown)}")
        return found
