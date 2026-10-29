"""What every catalog is: a source that lists the objects of a quantity and how to fetch each.

A catalog is declared under its quantity in ``registry.toml`` as a table with ``loader``, as a
data source is in ``sources.toml``, and carries the release terms (``lag_days``, ``revised``) its
series share. ``discover`` returns the objects it found, each after the one above it, and one
data source (``mst.data.sources.Source``) per driver, so the series are fetched by the loaders
of ``mst.data`` and stored like any other series. A new source of drivers is a new subclass
added to ``mst.drivers.catalogs.Catalog``; a new object of an existing source needs nothing.
"""

import re
import unicodedata
from dataclasses import dataclass, field

import httpx

from mst.data.release import Release

from ..objects import Object

LATIN = dict(
    zip(
        "абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
        "a b v g d e e zh z i y k l m n o p r s t u f kh ts ch sh shch _ y _ e yu ya".split(),
        strict=True,
    )
)


def slug(text: str, words: int = 4) -> str:
    """A stable object id from a Russian or English label: transliterated, first ``words``."""
    text = unicodedata.normalize("NFC", text.lower())
    latin = "".join(LATIN.get(char, char) for char in text)
    parts = [p for p in re.split(r"[^a-z0-9]+", latin.replace("_", "")) if p][:words]
    name = "_".join(parts)
    return name if re.match(r"^[a-z]", name) else f"x_{name}"


def name(code: str) -> str:
    """An object id from a source's code (``VEON-RX`` is ``veon_rx``)."""
    text = re.sub(r"[^a-z0-9]+", "_", code.lower()).strip("_")
    return text if re.match(r"^[a-z]", text) else f"x_{text}"


@dataclass
class Discovery:
    objects: list[Object] = field(default_factory=list)
    """Objects found, each after the object above it."""
    drivers: dict[str, object] = field(default_factory=dict)
    """Object id -> the data source of its history."""
    skipped: dict[str, str] = field(default_factory=dict)
    """What the catalog left out and why."""


class Catalog(Release):
    def discover(self, http: httpx.Client) -> Discovery:
        raise NotImplementedError
