"""FRED (api.stlouisfed.org): US and world market series, monthly means or as observed.

The key comes from the ``FRED_API_KEY`` environment variable and is sent only to FRED.
"""

import os
from datetime import date
from typing import Literal

import httpx

from .release import Release

URL = "https://api.stlouisfed.org/fred/series/observations"


class Fred(Release):
    loader: Literal["fred"]
    series_id: str

    def fetch(self, http: httpx.Client, start: date) -> dict[date, float]:
        return self._observations(http, start, frequency="m", aggregation_method="avg")

    def daily(self, http: httpx.Client, start: date) -> dict[date, float]:
        """Observations at the series' own frequency: daily, weekly or monthly."""
        return self._observations(http, start)

    def _observations(self, http: httpx.Client, start: date, **extra: str) -> dict[date, float]:
        key = os.environ.get("FRED_API_KEY")
        if not key:
            raise RuntimeError("FRED needs an API key in the FRED_API_KEY environment variable")
        params = {
            "series_id": self.series_id,
            "api_key": key,
            "file_type": "json",
            "observation_start": start.isoformat(),
            **extra,
        }
        response = http.get(URL, params=params)
        response.raise_for_status()
        return {
            date.fromisoformat(item["date"]): float(item["value"])
            for item in response.json()["observations"]
            if item["value"] != "."
        }
