"""The series store: one CSV per series, one row per published version of a month.

A row ``month, value, published, approximate`` says that the value of ``month`` became known
on ``published``. A revision adds a row and keeps the old one, so ``as_of(day)`` recovers what
was known on that day. ``approximate`` marks a row whose publication date is estimated from
the release lag rather than observed.
"""

import calendar
import csv
import os
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

SERIES_ID = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)*$")
COLUMNS = ["month", "value", "published", "approximate"]


@dataclass(frozen=True)
class Observation:
    month: date
    """The first day of the month."""
    value: float
    published: date
    approximate: bool


def month_end(month: date) -> date:
    return month.replace(day=calendar.monthrange(month.year, month.month)[1])


def shift(month: date, k: int) -> date:
    """The first day of the month ``k`` months after the month of ``month``; ``k`` may be < 0."""
    index = month.year * 12 + month.month - 1 + k
    return date(index // 12, index % 12 + 1, 1)


def new_versions(
    known: list[Observation],
    fetched: dict[date, float],
    *,
    lag_days: int,
    revised: bool,
    fetched_on: date,
) -> list[Observation]:
    """Rows that ``fetched`` adds to a series whose stored rows are ``known``.

    A month that has not ended by ``fetched_on`` is skipped: its value is not final. The first
    fetch of a series dates each month by its end plus ``lag_days``; if the source revises the
    series, these rows carry the latest version under an estimated date and are approximate.
    Later fetches date new months and changed values by ``fetched_on``.
    """
    latest = {row.month: row.value for row in sorted(known, key=lambda row: row.published)}
    backfill = not known
    rows = []
    for month, value in sorted(fetched.items()):
        if month_end(month) >= fetched_on or latest.get(month) == value:
            continue
        if month in latest or not backfill and revised:
            rows.append(Observation(month, value, fetched_on, False))
        else:
            estimated = month_end(month) + timedelta(days=lag_days)
            rows.append(Observation(month, value, min(estimated, fetched_on), backfill and revised))
    return rows


class SeriesStore:
    def __init__(self, root: Path):
        self.root = root

    def path(self, series: str) -> Path:
        if not SERIES_ID.match(series):
            raise ValueError(f"{series!r} is not a series id")
        return self.root / f"{series}.csv"

    def read(self, series: str) -> list[Observation]:
        path = self.path(series)
        if not path.exists():
            return []
        with path.open(newline="", encoding="utf-8") as file:
            return [
                Observation(
                    date.fromisoformat(row["month"] + "-01"),
                    float(row["value"]),
                    date.fromisoformat(row["published"]),
                    row["approximate"] == "1",
                )
                for row in csv.DictReader(file)
            ]

    def write(self, series: str, rows: list[Observation]) -> None:
        """Replaces the series with ``rows``; a crash leaves either the old file or the new one."""
        path = self.path(series)
        path.parent.mkdir(parents=True, exist_ok=True)
        draft = path.with_suffix(".csv.tmp")
        with draft.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerow(COLUMNS)
            for row in sorted(rows, key=lambda row: (row.month, row.published)):
                writer.writerow(
                    [
                        row.month.strftime("%Y-%m"),
                        repr(row.value),
                        row.published.isoformat(),
                        int(row.approximate),
                    ]
                )
        os.replace(draft, path)

    def as_of(self, series: str, day: date) -> list[Observation]:
        """For each month, the last version published on or before ``day``, sorted by month."""
        known = {}
        for row in sorted(self.read(series), key=lambda row: row.published):
            if row.published <= day:
                known[row.month] = row
        return [known[month] for month in sorted(known)]
