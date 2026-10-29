"""Series for the movements tables of the episode cards: D1–D5 of the manifest.

``<corpus>/series.toml`` names each series, its loader and unit, and which series measures a
factor in a country. A loader is one of ``mst.data`` (Bank of Russia, Moscow Exchange, FRED),
read at its own frequency instead of monthly means, or the BIS statistics API. Downloads are
cached in ``<corpus>/series/<id>.csv``. A series that has no cache and cannot be fetched
falls back to its monthly version in the series store, if ``monthly`` names one.
"""

import csv
import io
import tomllib
from datetime import date, timedelta
from pathlib import Path
from typing import Annotated, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from mst.data.cbr import CbrCurrency, CbrKeyRate, CbrRuonia, CbrZeroCoupon
from mst.data.fred import Fred
from mst.data.moex import MoexIndex, MoexSpread
from mst.data.store import SeriesStore

BIS_URL = "https://stats.bis.org/api/v1/data/{flow}/{key}/all"
SDMX_CSV = "application/vnd.sdmx.data+csv;version=1.0.0"


class Bis(BaseModel):
    """A series of the BIS statistics SDMX API; quarters are dated by their last month."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    loader: Literal["bis"]
    flow: str
    """The dataflow, e.g. WS_CBPOL (policy rates) or WS_XRU (exchange rates)."""
    key: str
    """The series key, e.g. D.TR for the daily policy rate of Turkey."""

    def daily(self, http: httpx.Client, start: date) -> dict[date, float]:
        response = http.get(
            BIS_URL.format(flow=self.flow, key=self.key),
            params={"startPeriod": start.isoformat(), "detail": "dataonly"},
            headers={"Accept": SDMX_CSV},
        )
        response.raise_for_status()
        values = {}
        for row in csv.DictReader(io.StringIO(response.text)):
            value = row.get("OBS_VALUE", "")
            if value not in ("", "NaN"):
                values[period(row["TIME_PERIOD"])] = float(value)
        return values


def period(text: str) -> date:
    """``2020-01-02``, ``2020-01``, ``2020-Q1`` or ``2020`` as a day."""
    if "-Q" in text:
        year, quarter = text.split("-Q")
        return date(int(year), int(quarter) * 3, 1)
    parts = [int(part) for part in text.split("-")]
    return date(*parts, *[1] * (3 - len(parts)))


Loader = Annotated[
    Fred | CbrKeyRate | CbrRuonia | CbrCurrency | CbrZeroCoupon | MoexIndex | MoexSpread | Bis,
    Field(discriminator="loader"),
]


class Series(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    title: str
    unit: Literal["pp", "pct"]
    """How a move is measured: the difference in percentage points, or the change in percent."""
    source: dict | None = None
    """The loader of the daily series; none for a series that exists only in the monthly store."""
    since: date
    """Where the download starts: the beginning of the history worth reading."""
    monthly: str | None = None
    """The same series in the monthly store, used when the daily one is not available."""
    transform: Literal["yoy"] | None = None
    """``yoy``: the percent change over twelve months of an index, in p.p. moves."""
    minus: str | None = None
    """Another series subtracted from this one on common dates, e.g. a Bund yield for a spread."""
    redenominated: date | None = None
    """The day the currency lost three zeros (the rouble on 1 January 1998). Values before it
    are divided by 1000 if the source gives them in old units, which shows as a jump of more
    than a hundred times across the day."""

    def loader(self) -> Loader | None:
        if self.source is None:
            return None
        config = dict(self.source)
        if config["loader"] != "bis":
            config.setdefault("lag_days", 0)
            config.setdefault("revised", False)
        return TypeAdapter(Loader).validate_python(config)


class Catalog(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    series: dict[str, Series]
    factors: dict[str, dict[str, list[str]]]
    """By country (``*`` for any), the series that measure a factor, best first."""
    episodes: dict[str, dict[str, list[str]]] = {}
    """Overrides by episode id."""

    def candidates(self, episode: str, country: str, factor: str) -> list[str]:
        for table in (
            self.episodes.get(episode, {}),
            self.factors.get(country, {}),
            self.factors.get("*", {}),
        ):
            if factor in table:
                return table[factor]
        return []


def load_catalog(path: Path) -> Catalog:
    with path.open("rb") as file:
        catalog = Catalog.model_validate(tomllib.load(file))
    named = {
        sid
        for tables in (*catalog.factors.values(), *catalog.episodes.values())
        for ids in tables.values()
        for sid in ids
    }
    named |= {series.minus for series in catalog.series.values() if series.minus}
    unknown = sorted(named - set(catalog.series))
    if unknown:
        raise ValueError(f"series.toml names unknown series {unknown}")
    for series_id, series in catalog.series.items():
        if series.source is None and series.monthly is None:
            raise ValueError(f"{series_id} has neither a source nor a monthly series")
        series.loader()
    return catalog


class Cache:
    def __init__(self, root: Path, store_root: Path):
        self.root = root
        self.store = SeriesStore(store_root)

    def path(self, series_id: str) -> Path:
        return self.root / f"{series_id}.csv"

    def read(self, series_id: str) -> dict[date, float] | None:
        path = self.path(series_id)
        if not path.exists():
            return None
        with path.open(newline="", encoding="utf-8") as file:
            return {
                date.fromisoformat(row["date"]): float(row["value"]) for row in csv.DictReader(file)
            }

    def write(self, series_id: str, values: dict[date, float]) -> None:
        path = self.path(series_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        draft = path.with_suffix(".csv.tmp")
        with draft.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerow(["date", "value"])
            for day in sorted(values):
                writer.writerow([day.isoformat(), repr(values[day])])
        draft.replace(path)

    def monthly(self, series_id: str) -> dict[date, float]:
        return {row.month: row.value for row in self.store.as_of(series_id, date.max)}


def fetch_series(
    catalog: Catalog, cache: Cache, http: httpx.Client, ids: list[str]
) -> dict[str, str]:
    """Download the named series into the cache; the failures by id with their reasons."""
    failed = {}
    for series_id in ids:
        series = catalog.series[series_id]
        if series.source is None:
            continue
        try:
            values = series.loader().daily(http, series.since)
        except (httpx.HTTPError, RuntimeError, ValueError, KeyError) as error:
            failed[series_id] = f"{type(error).__name__}: {error}"
            continue
        if not values:
            failed[series_id] = "no observations"
            continue
        cache.write(series_id, values)
    return failed


def observations(catalog: Catalog, cache: Cache, series_id: str) -> tuple[dict[date, float], str]:
    """The values of a series as the movements use them, and where they came from.

    The origin is ``cache`` or ``store`` (the monthly fallback); no values gives ``{}, "none"``.
    """
    series = catalog.series[series_id]
    values, origin = cache.read(series_id), "cache"
    if values is None and series.monthly:
        values, origin = cache.monthly(series.monthly), "store"
    if not values:
        return {}, "none"
    if series.redenominated:
        values = redenominate(values, series.redenominated)
    if series.minus:
        other, other_origin = observations(catalog, cache, series.minus)
        values = {day: value - other[day] for day, value in values.items() if day in other}
        if other_origin == "store":
            origin = "store"
    if series.transform == "yoy":
        values = year_on_year(values)
    return values, origin


def redenominate(values: dict[date, float], day: date) -> dict[date, float]:
    before = [d for d in values if d < day]
    after = [d for d in values if d >= day]
    if before and after and values[max(before)] > 100 * values[min(after)]:
        return {d: v / 1000 if d < day else v for d, v in values.items()}
    return values


def year_on_year(values: dict[date, float]) -> dict[date, float]:
    """Percent change of each value over the value twelve months before (± 3 days)."""
    days = sorted(values)
    out = {}
    for day in days:
        try:
            target = day.replace(year=day.year - 1)
        except ValueError:  # 29 February
            target = day.replace(year=day.year - 1, day=28)
        near = [d for d in (target + timedelta(days=k) for k in range(-3, 4)) if d in values]
        if near and values[near[0]]:
            out[day] = (values[day] / values[near[0]] - 1) * 100
    return out
