"""Moscow Exchange ISS (iss.moex.com): index closes, share closes and corporate bond spreads.
No key is needed."""

import math
import weakref
from datetime import date, timedelta
from typing import Literal

import httpx
from pydantic import model_validator

from mst.contracts import Contract, Positive

from .cbr import curve_at, zero_coupon_curve
from .daily import monthly_means, monthly_medians
from .release import Release

INDEX_URL = "https://iss.moex.com/iss/history/engines/stock/markets/index/securities/{}.json"
SHARE_URL = "https://iss.moex.com/iss/history/engines/stock/markets/shares/boards/{board}/securities/{}.json"
BOND_URL = "https://iss.moex.com/iss/history/engines/stock/markets/bonds/securities/{}.json"
JUMP = math.log(3)
"""A day-to-day move of a share beyond a factor of three needs a declared split."""
_CURVES: "weakref.WeakKeyDictionary[httpx.Client, dict]" = weakref.WeakKeyDictionary()


def curve(http: httpx.Client, start: date) -> dict[date, dict[float, float]]:
    """The zero-coupon curve from ``start``, read once per client: many bonds share it."""
    curves = _CURVES.setdefault(http, {})
    if start not in curves:
        curves[start] = zero_coupon_curve(http, start)
    return curves[start]


def spread(curve_day: dict[float, float], yield_: float, duration_days: float) -> float:
    """A yield over the OFZ curve at the duration, p.p."""
    return yield_ - curve_at(curve_day, duration_days / 365)


def _history(
    http: httpx.Client, secid: str, start: date, till: date | None = None, url: str = INDEX_URL
) -> list[dict]:
    """Daily rows of a security's history, one dict per row keyed by ISS column names."""
    rows, offset, total = [], 0, 1
    while offset < total:
        params = {"from": start.isoformat(), "start": str(offset), "iss.meta": "off"}
        if till is not None:
            params["till"] = till.isoformat()
        response = http.get(url.format(secid), params=params)
        response.raise_for_status()
        body = response.json()
        columns = body["history"]["columns"]
        rows += [dict(zip(columns, row, strict=True)) for row in body["history"]["data"]]
        _, total, size = body["history.cursor"]["data"][0]
        offset += size
    return rows


class MoexIndex(Release):
    """Monthly means of the daily close of an index."""

    loader: Literal["moex_index"]
    secid: str

    def fetch(self, http: httpx.Client, start: date) -> dict[date, float]:
        return monthly_means(self.daily(http, start))

    def daily(self, http: httpx.Client, start: date) -> dict[date, float]:
        return {
            date.fromisoformat(row["TRADEDATE"]): float(row["CLOSE"])
            for row in _history(http, self.secid, start)
            if row["CLOSE"] is not None
        }


class Split(Contract):
    day: date
    """The first trading day at the new number of shares."""
    ratio: Positive
    """New shares per old share: 100 for a 1:100 split, 1/5000 for a 5000:1 consolidation."""


class MoexShare(Release):
    """Monthly means of the daily close of a share, roubles per share at today's share count.

    Days come from ``boards`` in order, a later board replacing an earlier one on common days
    (EQBR until 2013, TQBR since). Closes before a split are divided by its ratio. A move from
    one trading day to the next beyond a factor of three that no split explains stops the
    fetch: it is an undeclared split or a broken quote.
    """

    loader: Literal["moex_share"]
    secid: str
    boards: list[str]
    splits: list[Split] = []

    def fetch(self, http: httpx.Client, start: date) -> dict[date, float]:
        return monthly_means(self.daily(http, start))

    def daily(self, http: httpx.Client, start: date) -> dict[date, float]:
        closes = {}
        for board in self.boards:
            url = SHARE_URL.replace("{board}", board)
            for row in _history(http, self.secid, start, url=url):
                if row["CLOSE"] is not None:
                    closes[date.fromisoformat(row["TRADEDATE"])] = float(row["CLOSE"])
        days = sorted(closes)
        for split in self.splits:
            for day in days:
                if day < split.day:
                    closes[day] /= split.ratio
        for before, day in zip(days, days[1:], strict=False):
            if abs(math.log(closes[day] / closes[before])) > JUMP:
                raise ValueError(
                    f"{self.secid} moves from {closes[before]} on {before} to {closes[day]} on "
                    f"{day}: declare the split or check the quote"
                )
        return {day: closes[day] for day in days}


class MoexSpread(Release):
    """Monthly medians of a bond index's yield over the OFZ curve at the index's duration, p.p.

    Months before ``splice`` come from the ``old`` index, later ones from ``new``. Days whose
    yield or duration is missing or zero (ISS fills the first day of an index with zeros) are
    left out, and so are the days in ``skip``. The median keeps a few days of a defaulting bond
    out of the month; ``skip`` names the longer episodes.
    """

    loader: Literal["moex_spread"]
    old: str
    new: str
    splice: date
    """The first month taken from ``new``."""
    skip: list[tuple[date, date]] = []
    """Inclusive ranges of days whose index yield is not a market yield."""

    @model_validator(mode="after")
    def check_dates(self):
        if self.splice.day != 1:
            raise ValueError(f"splice {self.splice} must be the first day of a month")
        for first, last in self.skip:
            if last < first:
                raise ValueError(f"skipped range {first}..{last} ends before it starts")
        return self

    def fetch(self, http: httpx.Client, start: date) -> dict[date, float]:
        return monthly_medians(self.daily(http, start))

    def daily(self, http: httpx.Client, start: date) -> dict[date, float]:
        days = curve(http, start)
        rows = _history(http, self.old, start, self.splice - timedelta(days=1))
        rows += _history(http, self.new, max(start, self.splice))
        daily = {}
        for row in rows:
            day = date.fromisoformat(row["TRADEDATE"])
            skipped = any(first <= day <= last for first, last in self.skip)
            if row["YIELD"] and row["DURATION"] and day in days and not skipped:
                daily[day] = spread(days[day], row["YIELD"], row["DURATION"])
        return daily


class MoexBondSpread(Release):
    """Monthly medians of a bond's yield over the OFZ curve at its duration, p.p.

    A day takes the board with the most trades; days without trades, yield or duration are
    left out, so a month nobody traded the bond stays empty.
    """

    loader: Literal["moex_bond_spread"]
    secid: str

    def fetch(self, http: httpx.Client, start: date) -> dict[date, float]:
        return monthly_medians(self.daily(http, start))

    def daily(self, http: httpx.Client, start: date) -> dict[date, float]:
        days = curve(http, start)
        best: dict[date, dict] = {}
        for row in _history(http, self.secid, start, url=BOND_URL):
            day = date.fromisoformat(row["TRADEDATE"])
            if row["NUMTRADES"] and row["YIELDCLOSE"] and row["DURATION"] and day in days:
                if day not in best or row["NUMTRADES"] > best[day]["NUMTRADES"]:
                    best[day] = row
        return {
            day: spread(days[day], row["YIELDCLOSE"], row["DURATION"])
            for day, row in sorted(best.items())
        }
