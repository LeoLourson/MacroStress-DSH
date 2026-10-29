"""Bank of Russia (www.cbr.ru): key rate, RUONIA, official exchange rates, the OFZ curve.

The key rate and RUONIA come from the ``DailyInfoWebServ`` SOAP service, exchange rates from
``XML_dynamic.asp``, the zero-coupon OFZ yields of the Moscow Exchange from the Bank's
``hd_base/zcyc_params`` table. No key is needed.
"""

import re
import xml.etree.ElementTree as ET
from datetime import date
from itertools import pairwise
from typing import Literal

import httpx

from mst.contracts import Contract, Number

from .daily import monthly_means, step_days, step_means
from .release import Release

SOAP_URL = "https://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx"
CURRENCY_URL = "https://www.cbr.ru/scripts/XML_dynamic.asp"
ZCYC_URL = "https://www.cbr.ru/hd_base/zcyc_params/"


def _soap(http: httpx.Client, method: str, start: date, tag: str) -> list[ET.Element]:
    envelope = (
        '<?xml version="1.0" encoding="utf-8"?><soap12:Envelope '
        'xmlns:soap12="http://www.w3.org/2003/05/soap-envelope"><soap12:Body>'
        f'<{method} xmlns="http://web.cbr.ru/"><fromDate>{start.isoformat()}</fromDate>'
        f"<ToDate>{date.today().isoformat()}</ToDate></{method}></soap12:Body></soap12:Envelope>"
    )
    response = http.post(
        SOAP_URL,
        content=envelope.encode(),
        headers={"Content-Type": "application/soap+xml; charset=utf-8"},
    )
    response.raise_for_status()
    return ET.fromstring(response.content).iter(tag)


def _day(text: str) -> date:
    return date.fromisoformat(text[:10])


class Step(Contract):
    since: date
    value: Number


class CbrKeyRate(Release):
    """Monthly means over calendar days of the rate in force.

    The service starts on 13 September 2013, when the key rate was introduced; ``before``
    gives the rate that played its part earlier, step by step.
    """

    loader: Literal["cbr_key_rate"]
    before: list[Step]

    def fetch(self, http: httpx.Client, start: date) -> dict[date, float]:
        return step_means(self._steps(http, start), start)

    def daily(self, http: httpx.Client, start: date) -> dict[date, float]:
        """The rate in force on each calendar day from ``start`` to the end of this month."""
        steps = self._steps(http, start)
        steps.setdefault(date.today(), steps[max(steps)])
        return step_days(steps, start)

    def _steps(self, http: httpx.Client, start: date) -> dict[date, float]:
        steps = {step.since: step.value for step in self.before}
        for row in _soap(http, "KeyRateXML", start, "KR"):
            steps[_day(row.findtext("DT"))] = float(row.findtext("Rate"))
        return steps


class CbrRuonia(Release):
    """Monthly means of the published daily RUONIA."""

    loader: Literal["cbr_ruonia"]

    def fetch(self, http: httpx.Client, start: date) -> dict[date, float]:
        return monthly_means(self.daily(http, start))

    def daily(self, http: httpx.Client, start: date) -> dict[date, float]:
        rows = _soap(http, "RuoniaXML", start, "ro")
        return {_day(row.findtext("D0")): float(row.findtext("ruo")) for row in rows}


class CbrCurrency(Release):
    """Monthly means of the official rate, in roubles per one unit of the currency."""

    loader: Literal["cbr_currency"]
    code: str
    """The Bank of Russia currency code, e.g. R01235 for the US dollar."""

    def fetch(self, http: httpx.Client, start: date) -> dict[date, float]:
        return monthly_means(self.daily(http, start))

    def daily(self, http: httpx.Client, start: date) -> dict[date, float]:
        params = {
            "date_req1": start.strftime("%d/%m/%Y"),
            "date_req2": date.today().strftime("%d/%m/%Y"),
            "VAL_NM_RQ": self.code,
        }
        response = http.get(CURRENCY_URL, params=params)
        response.raise_for_status()
        daily = {}
        for record in ET.fromstring(response.content).iter("Record"):
            day = date(*map(int, reversed(record.get("Date").split("."))))
            value = float(record.findtext("Value").replace(",", "."))
            daily[day] = value / int(record.findtext("Nominal"))
        return daily


def zero_coupon_curve(http: httpx.Client, start: date) -> dict[date, dict[float, float]]:
    """Daily zero-coupon OFZ yields, % per year, by term in years, from ``start`` to today.

    The table refuses long ranges, so it is read one calendar year at a time.
    """
    curve = {}
    for year in range(start.year, date.today().year + 1):
        first = max(start, date(year, 1, 1))
        curve |= _curve_year(http, first, min(date.today(), date(year, 12, 31)))
    return curve


def _curve_year(http: httpx.Client, first: date, last: date) -> dict[date, dict[float, float]]:
    params = {
        "UniDbQuery.Posted": "True",
        "UniDbQuery.From": first.strftime("%d.%m.%Y"),
        "UniDbQuery.To": last.strftime("%d.%m.%Y"),
    }
    response = http.get(ZCYC_URL, params=params)
    response.raise_for_status()
    table = re.search(r'<table class="data spaced">(.*?)</table>', response.text, re.S)
    if table is None:
        raise ValueError(f"the curve table for {first.year} is missing from the page")
    rows = re.findall(r"<tr>(.*?)</tr>", table.group(1), re.S)
    terms = [_number(term) for term in re.findall(r"<th>([^<]*)</th>", rows[1])]
    curve = {}
    for row in rows[2:]:
        cells = re.findall(r"<td>(?:<a[^>]*>)?([^<]*)", row)
        day = date(*map(int, reversed(cells[0].split("."))))
        curve[day] = dict(zip(terms, map(_number, cells[1:]), strict=True))
    return curve


def _number(text: str) -> float:
    return float(text.replace(",", "."))


def curve_at(terms: dict[float, float], years: float) -> float:
    """The yield at ``years``: linear between the table's terms, flat beyond its ends."""
    points = sorted(terms.items())
    if years <= points[0][0]:
        return points[0][1]
    for (left, low), (right, high) in pairwise(points):
        if years <= right:
            return low + (high - low) * (years - left) / (right - left)
    return points[-1][1]


class CbrZeroCoupon(Release):
    """Monthly means of the daily zero-coupon OFZ yield for one term, % per year."""

    loader: Literal["cbr_zcyc"]
    years: str
    """The term as the table heads it, e.g. ``1`` or ``0,25``."""

    def fetch(self, http: httpx.Client, start: date) -> dict[date, float]:
        return monthly_means(self.daily(http, start))

    def daily(self, http: httpx.Client, start: date) -> dict[date, float]:
        term = _number(self.years)
        return {day: terms[term] for day, terms in zero_coupon_curve(http, start).items()}
