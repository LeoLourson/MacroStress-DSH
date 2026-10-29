"""Where each series comes from: ``sources.toml``, keyed by series id.

The shipped file lives next to this module and is part of the code. Sources added from the
workspace (``series(action="add")``) go to ``<workspace>/sources.toml``, which is read on top of
it; an id may appear in only one of the two.
"""

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


def _read(path: Path) -> dict[str, Source]:
    with path.open("rb") as file:
        return TypeAdapter(dict[str, Source]).validate_python(tomllib.load(file))


def load_sources(path: Path = SOURCES_PATH, extra: Path | None = None) -> dict[str, Source]:
    """The sources of ``path`` and, when it exists, of the workspace file ``extra``."""
    found = _read(path)
    if extra is not None and extra.is_file():
        added = _read(extra)
        repeated = sorted(set(found) & set(added))
        if repeated:
            raise ValueError(f"{extra} repeats the sources of {', '.join(repeated)}")
        found |= added
    return found


def search(text: str, path: Path = SOURCES_PATH, extra: Path | None = None) -> list[dict[str, str]]:
    """Sources whose id or loader contains ``text``, case-insensitive."""
    needle = text.casefold()
    return [
        {"id": series_id, "loader": source.loader}
        for series_id, source in load_sources(path, extra).items()
        if needle in series_id.casefold() or needle in source.loader.casefold()
    ]


def add(entry: dict[str, Any], path: Path, base: Path = SOURCES_PATH) -> str:
    """Validate one source, append it to the file ``path`` and return its id.

    ``entry`` is the table the file would hold plus ``id``: the id is the table name and
    the rest is the source itself, checked by the same union the file is read with. An id
    that ``base`` or ``path`` already holds is refused.
    """
    series_id = entry.get("id")
    if not isinstance(series_id, str) or not SERIES_ID.match(series_id):
        raise ValueError(f"not a series id: {series_id!r}")
    known = set(load_sources(base)) | (set(load_sources(path)) if path.is_file() else set())
    if series_id in known:
        raise ValueError(f"series {series_id!r} already has a source; update it instead")
    source = TypeAdapter(Source).validate_python(
        {key: value for key, value in entry.items() if key != "id"}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
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
