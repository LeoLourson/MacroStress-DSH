"""Rosstat (rosstat.gov.ru): monthly series read from the Excel workbooks it publishes.

A workbook's file name changes with every release (``ipc_mes_09-2026.xlsx``), so a source
names the page that links it and a pattern of the file name. Workbooks are parsed in memory.
Values are stored as published, without seasonal adjustment. A source with a ``base`` month
publishes percent changes to the previous month; they are chained into a level that is 100 in
that month. No key is needed.
"""

import io
import re
import weakref
import zipfile
from datetime import date
from typing import Literal

import httpx
import openpyxl
from openpyxl.utils.exceptions import InvalidFileException
from pydantic import model_validator

from .release import Release
from .store import shift

ROOT = "https://rosstat.gov.ru"
MONTHS = {
    name: number
    for number, name in enumerate(
        ["январь", "февраль", "март", "апрель", "май", "июнь",
         "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"],
        1,
    )
}  # fmt: skip
QUARTERS = {"I квартал": 3, "II квартал": 6, "III квартал": 9, "IV квартал": 12}
MISSING = {"", "-", "…", "..."}


_BOOKS: "weakref.WeakKeyDictionary[httpx.Client, dict]" = weakref.WeakKeyDictionary()


def _workbooks(http: httpx.Client, page: str, pattern: str) -> list[tuple[re.Match, dict]]:
    """The workbooks linked from ``page`` whose file name matches ``pattern``: sheets of rows.

    They are read once per client: the series of every territory come from the same books.
    """
    books = _BOOKS.setdefault(http, {})
    if (page, pattern) not in books:
        books[page, pattern] = _read_workbooks(http, page, pattern)
    return books[page, pattern]


def _read_workbooks(http: httpx.Client, page: str, pattern: str) -> list[tuple[re.Match, dict]]:
    response = http.get(page)
    response.raise_for_status()
    names = dict.fromkeys(re.findall(r'href="/storage/mediabank/([^"/]+)"', response.text))
    found = [(match, name) for name in names if (match := re.fullmatch(pattern, name))]
    if not found:
        raise ValueError(f"no file matching {pattern} is linked from {page}")
    books = []
    for match, name in found:
        content = http.get(f"{ROOT}/storage/mediabank/{name}")
        content.raise_for_status()
        try:
            book = openpyxl.load_workbook(
                io.BytesIO(content.content), read_only=True, data_only=True
            )
        except (zipfile.BadZipFile, InvalidFileException, KeyError) as error:
            raise ValueError(f"{name} is not a workbook") from error
        sheets = {sheet.title.strip(): list(sheet.iter_rows(values_only=True)) for sheet in book}
        books.append((match, sheets))
    return books


def _number(cell) -> float | None:
    """A cell's value; text cells may carry a one-digit footnote mark (``71,33)`` is 71.3)."""
    if isinstance(cell, int | float):
        return float(cell)
    if cell is None or str(cell).strip() in MISSING:
        return None
    try:
        return float(re.sub(r"(\d\))+$", "", str(cell).strip()).replace(",", "."))
    except ValueError:
        raise ValueError(f"{cell!r} is not a number") from None


def _first(items, what: str):
    """The first of ``items``; a workbook whose layout changed lacks it, which is an error."""
    for item in items:
        return item
    raise ValueError(f"the workbook has no {what}")


def _single(books: list, pattern: str) -> dict:
    if len(books) > 1:
        raise ValueError(f"several files match {pattern}")
    return books[0][1]


def _label(row: tuple) -> str:
    return "" if not row or row[0] is None else str(row[0]).strip()


def _year(cell) -> int | None:
    """The year a cell heads, footnote marks after it ignored (``2022г.2)`` is 2022)."""
    if isinstance(cell, float) and not cell.is_integer():
        return None
    match = re.match(r"(\d{4})", str(cell).strip()) if cell is not None else None
    year = int(match.group(1)) if match else None
    return year if year and 1990 <= year <= 2100 else None


def _month(cell) -> int | None:
    return MONTHS.get(re.sub(r"[\d)*\s]+$", "", str(cell or "")).strip().lower())


def chain(changes: dict[date, float], base: date) -> dict[date, float]:
    """Levels with 100 in ``base`` from percent changes to the previous month after it."""
    levels, month = {base: 100.0}, base
    while (following := shift(month, 1)) in changes:
        levels[following] = levels[month] * changes[following] / 100
        month = following
    gap = [later for later in changes if later > month]
    if gap:
        raise ValueError(
            f"no change for {shift(month, 1):%Y-%m} breaks the chain to {max(gap):%Y-%m}"
        )
    return levels


class Rosstat(Release):
    page: str
    """The page that links the workbook."""
    file: str
    """A regular expression the workbook's file name matches in full."""
    base: date | None = None
    """The month whose level is 100; absent when the table publishes levels."""

    def _finish(self, values: dict[date, float], start: date) -> dict[date, float]:
        if self.base is not None:
            values = chain({m: v for m, v in values.items() if m > self.base}, self.base)
        return {month: value for month, value in sorted(values.items()) if month >= start}


class RosstatMonthly(Rosstat):
    """A table with years across and months down, as the CPI and ИВБО workbooks lay it out.

    Rows that are not months (quarters, headings) are skipped; a month met a second time ends
    the table, since the rows below it belong to another one.
    """

    loader: Literal["rosstat_monthly"]
    sheet: str

    def fetch(self, http: httpx.Client, start: date) -> dict[date, float]:
        rows = iter(_single(_workbooks(http, self.page, self.file), self.file)[self.sheet])
        for row in rows:
            years = {
                column: year for column, cell in enumerate(row[1:], 1) if (year := _year(cell))
            }
            if years:
                break
        values, seen = {}, set()
        for row in rows:
            month = _month(row[0] if row else None)
            if month is None:
                continue
            if month in seen:
                break
            seen.add(month)
            for column, year in years.items():
                value = _number(row[column]) if column < len(row) else None
                if value is not None:
                    values[date(year, month, 1)] = value
        return self._finish(values, start)


class RosstatShortTerm(Rosstat):
    """Sections of «Краткосрочные экономические показатели Российской Федерации».

    A section is a heading numbered like ``4.1.5.`` and rows of years with the months in the
    columns headed ``Янв.`` onwards. ``block`` picks the rows under the first line after the
    heading that starts with it, e.g. ``в % к предыдущему периоду``. When ``sections`` name
    several headings, a later one overrides an earlier one for the months both give.
    """

    loader: Literal["rosstat_short_term"]
    sheet: str
    sections: list[str]
    block: str | None = None

    def fetch(self, http: httpx.Client, start: date) -> dict[date, float]:
        rows = _single(_workbooks(http, self.page, self.file), self.file)[self.sheet]
        january = _first((row.index("Янв.") for row in rows if "Янв." in row), "column Янв.")
        values = {}
        for section in self.sections:
            heading = (i for i, row in enumerate(rows) if _label(row).split()[:1] == [section])
            at = _first(heading, f"section {section}")
            if self.block is not None:
                lines = (
                    i for i in range(at + 1, len(rows)) if _label(rows[i]).startswith(self.block)
                )
                at = _first(lines, f"{self.block!r} after {section}")
            for row in rows[at + 1 :]:
                year = _year(_label(row))
                if year is None and _label(row).startswith("/"):
                    continue
                if year is None:
                    break
                for month in range(12):
                    cell = row[january + month] if january + month < len(row) else None
                    if (value := _number(cell)) is not None:
                        values[date(year, month + 1, 1)] = value
        return self._finish(values, start)


class RosstatHousingPrice(Release):
    """Average price of a square metre at the end of a period, roubles, of one territory.

    ``annual`` gives the year ends of Russia since 1998, stored in December; ``quarterly``
    matches one workbook per year (its year is the pattern's group) whose quarters exist since
    2021 for every subject and are stored in a quarter's last month, overriding the December of
    the annual table. Months in between stay empty: the series is quarterly, then annual
    further back. ``region`` is the OKATO code a row of the quarterly table starts with: 643
    for Russia, 030 for a federal district, eleven digits for a subject. The annual table
    covers Russia only.
    """

    loader: Literal["rosstat_housing_price"]
    page: str
    annual: str | None = None
    quarterly: str
    region: str = "643"

    @model_validator(mode="after")
    def annual_of_russia(self):
        if self.annual is not None and self.region != "643":
            raise ValueError("the annual table covers Russia only")
        return self

    def fetch(self, http: httpx.Client, start: date) -> dict[date, float]:
        values = self._annual(http) if self.annual is not None else {}
        found = False
        for match, sheets in _workbooks(http, self.page, self.quarterly):
            year = int(match.group(1))
            rows = _first(
                (rows for rows in sheets.values() if any("I квартал" in r for r in rows)),
                f"sheet of quarters for {year}",
            )
            header = _first((row for row in rows if "I квартал" in row), "quarter header")
            territory = next(
                (row for row in rows if territory_code(_label(row)) == self.region), None
            )
            if territory is None:
                continue  # a subject the table did not show that year
            found = True
            for column, cell in enumerate(header):
                month = QUARTERS.get(str(cell or "").strip())
                if month and (value := _number(territory[column])) is not None:
                    values[date(year, month, 1)] = value
        if not found:
            raise ValueError(f"no workbook {self.quarterly} has row {self.region}")
        return {month: value for month, value in sorted(values.items()) if month >= start}

    def _annual(self, http: httpx.Client) -> dict[date, float]:
        values = {}
        sheets = _single(_workbooks(http, self.page, self.annual), self.annual)
        rows = _first(
            (rows for rows in sheets.values() if any("Все типы квартир" in r for r in rows)),
            "sheet of year ends",
        )
        column = next(row for row in rows if "Все типы квартир" in row).index("Все типы квартир")
        for row in rows:
            year = _year(_label(row))
            if year is not None and (value := _number(row[column])) is not None:
                values[date(year, 12, 1)] = value
        return values


def territory_code(label: str) -> str | None:
    """The OKATO code of a row labelled ``<code> - <name>``: 643, 030 or eleven digits."""
    match = re.match(r"^(\d{3}|\d{11})\s+-\s+", label)
    return match.group(1) if match else None
