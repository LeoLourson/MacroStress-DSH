"""``corpus/manifest.json``: what the corpus holds and where each file comes from.

A source is one document or a series of editions. Every file of a source, a ``Document``, has
a Markdown path in the corpus and, once it is known, a ``FileSpec`` that says how to download
the original: a direct ``url``, or a ``page`` whose first link matching ``match`` is the file.
"""

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Kind = Literal["methodology", "scenario_set", "transmission", "episode"]
Region = Literal["ru", "em", "dm"]
SourceStatus = Literal["planned", "raw", "ready"]
"""``planned``: no text yet. ``raw``: text extracted from the original by ``msa-corpus convert``
and not proofread; the index takes it. ``ready``: proofread, every edition present."""
EpisodeStatus = Literal["planned", "draft", "ready"]
"""``draft``: the card exists with its movements table; the chronology is not written yet."""


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FileSpec(Model):
    url: str | None = None
    """The file itself."""
    page: str | None = None
    """A page that links to the file, for files whose address changes or is not known."""
    match: str | None = None
    """A regular expression; the first link of ``page`` whose address or text matches it."""
    format: Literal["pdf", "html"] = "pdf"
    """``html``: the document is a web page, ``url`` or the matched link is the page itself."""
    published: date | None = None
    """The day of publication when it is known from the publisher."""
    note: str | None = None

    @model_validator(mode="after")
    def one_way(self):
        if (self.url is None) == (self.page is None):
            raise ValueError("give either url or page")
        if (self.page is None) != (self.match is None):
            raise ValueError("page and match go together")
        if self.match is not None:
            try:
                re.compile(self.match)
            except re.error as error:
                raise ValueError(f"match is not a regular expression: {error}") from error
        return self


class Source(Model):
    id: str = Field(pattern=r"^[MST]\d+$")
    kind: Literal["methodology", "scenario_set", "transmission"]
    title: str
    publisher: str
    authors: list[str] = []
    year: int | None = None
    editions: list[str] | None = None
    path: str
    language: Literal["ru", "en"]
    jurisdiction: str
    url: str | None
    url_verified: bool
    text_policy: Literal["full", "summary"]
    factors: list[str]
    purpose: str
    status: SourceStatus
    verify: bool
    notes: str | None = None
    file: FileSpec | None = None
    """The original of a single-file source."""
    files: dict[str, FileSpec] = {}
    """Originals of a series by edition; an edition without one is not downloadable yet."""

    @model_validator(mode="after")
    def files_match_editions(self):
        if self.editions is None:
            if self.files:
                raise ValueError(f"{self.id}: files need editions")
            if "<edition>" in self.path:
                raise ValueError(f"{self.id}: path has <edition> but there are no editions")
        else:
            if self.file is not None:
                raise ValueError(f"{self.id}: a series takes files, not file")
            if "<edition>" not in self.path:
                raise ValueError(f"{self.id}: a series path needs <edition>")
            unknown = sorted(set(self.files) - set(self.editions))
            if unknown:
                raise ValueError(f"{self.id}: files for unknown editions {unknown}")
        return self


class Episode(Model):
    id: str = Field(pattern=r"^[RXG]\d+$")
    start: str = Field(pattern=r"^\d{4}-\d{2}$")
    region: Region
    country: str
    title: str
    factors: list[str]
    data: list[str]
    path: str
    status: EpisodeStatus
    flags: list[str] = []

    @property
    def start_month(self) -> date:
        year, month = map(int, self.start.split("-"))
        return date(year, month, 1)


class DataSeries(Model):
    id: str
    title: str
    publisher: str
    series: list[str]
    factors: list[str]


class Manifest(Model):
    model_config = ConfigDict(extra="forbid", frozen=False)

    corpus_id: str
    version: str
    updated: str
    description: str
    vocabularies: dict[str, dict[str, str]]
    sources: list[Source]
    data_series: list[DataSeries]
    episodes: list[Episode]

    @model_validator(mode="after")
    def consistent(self):
        ids = [item.id for item in self.sources] + [item.id for item in self.episodes]
        repeated = sorted({i for i in ids if ids.count(i) > 1})
        if repeated:
            raise ValueError(f"repeated ids {repeated}")
        factors = set(self.vocabularies["factors"])
        for item in [*self.sources, *self.episodes]:
            unknown = sorted(set(item.factors) - factors)
            if unknown:
                raise ValueError(f"{item.id}: unknown factors {unknown}")
        data = {series.id for series in self.data_series}
        for episode in self.episodes:
            if set(episode.data) - data:
                raise ValueError(f"{episode.id}: unknown data {sorted(set(episode.data) - data)}")
            if set(episode.flags) - set(self.vocabularies["flags"]):
                raise ValueError(f"{episode.id}: unknown flags")
        return self


@dataclass(frozen=True)
class Document:
    """One file of a source: a single document or one edition of a series."""

    source: Source
    edition: str | None
    spec: FileSpec | None

    @property
    def key(self) -> str:
        """``M1`` or ``S1/2023``: the name of the document in downloads and chunk ids."""
        return self.source.id if self.edition is None else f"{self.source.id}/{self.edition}"

    @property
    def path(self) -> str:
        """The Markdown file, relative to the corpus root."""
        return self.source.path.replace("<edition>", self.edition or "")

    def raw_path(self, root: Path) -> Path:
        suffix = ".html" if self.spec is not None and self.spec.format == "html" else ".pdf"
        name = (self.edition or self.source.id) + suffix
        return root / self.source.id / name


def documents(source: Source) -> list[Document]:
    if source.editions is None:
        return [Document(source, None, source.file)]
    return [Document(source, edition, source.files.get(edition)) for edition in source.editions]


def load(path: Path) -> Manifest:
    return Manifest.model_validate(json.loads(path.read_text(encoding="utf-8")))


def save(manifest: Manifest, path: Path) -> None:
    """Write the manifest in its own layout: two-space JSON, UTF-8 as is, unset keys left out."""
    data = manifest.model_dump(mode="json", exclude_defaults=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
