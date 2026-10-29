"""Where each series comes from: ``sources.toml``, keyed by series id."""

import json
import tomllib
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field, TypeAdapter

from .cbr import CbrCurrency, CbrKeyRate, CbrRuonia, CbrZeroCoupon
from .cbr_stats import CbrCredit, CbrMortgageRate, CbrOverdue
from .fred import Fred
from .moex import MoexBondSpread, MoexIndex, MoexShare, MoexSpread
from .rosstat import RosstatHousingPrice, RosstatMonthly, RosstatShortTerm
from .store import SERIES_ID

SOURCES_PATH = Path(__file__).with_name("sources.toml")

Source = Annotated[
    Fred
    | CbrKeyRate
    | CbrRuonia
    | CbrCurrency
    | CbrZeroCoupon
    | MoexIndex
    | MoexSpread
    | MoexShare
    | MoexBondSpread
    | CbrCredit
    | CbrMortgageRate
    | CbrOverdue
    | RosstatMonthly
    | RosstatShortTerm
    | RosstatHousingPrice,
    Field(discriminator="loader"),
]


def load_sources(path: Path = SOURCES_PATH) -> dict[str, Source]:
    with path.open("rb") as file:
        return TypeAdapter(dict[str, Source]).validate_python(tomllib.load(file))


def search(text: str, path: Path = SOURCES_PATH) -> list[dict[str, str]]:
    """Sources whose id or loader contains ``text``, case-insensitive."""
    needle = text.casefold()
    return [
        {"id": series_id, "loader": source.loader}
        for series_id, source in load_sources(path).items()
        if needle in series_id.casefold() or needle in source.loader.casefold()
    ]


def add(entry: dict[str, Any], path: Path = SOURCES_PATH) -> str:
    """Validate one source, append it to ``sources.toml`` and return its id.

    ``entry`` is the table the file would hold plus ``id``: the id is the table name and
    the rest is the source itself, checked by the same union the file is read with.
    """
    series_id = entry.get("id")
    if not isinstance(series_id, str) or not SERIES_ID.match(series_id):
        raise ValueError(f"not a series id: {series_id!r}")
    if series_id in load_sources(path):
        raise ValueError(f"series {series_id!r} already has a source; update it instead")
    source = TypeAdapter(Source).validate_python(
        {key: value for key, value in entry.items() if key != "id"}
    )
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write("\n" + table(series_id, source.model_dump(exclude_none=True)))
    return series_id


def table(name: str, fields: dict[str, Any]) -> str:
    """One ``["id"]`` table of ``sources.toml``."""
    lines = [f'["{name}"]']
    lines.extend(f"{key} = {_value(value)}" for key, value in fields.items())
    return "\n".join(lines) + "\n"


def _value(value: Any) -> str:
    """A value of ``sources.toml``: str, bool, number, date, array or inline table."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, date | datetime):
        return value.isoformat()
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{key} = {_value(item)}" for key, item in value.items()) + " }"
    raise TypeError(f"cannot write {type(value).__name__} to sources.toml")
