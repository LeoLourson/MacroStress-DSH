"""Where a corpus lives: the authored part under version control, the derived part in a workspace.

The authored part is what people write and review: the manifest, the series catalog, the
templates and the episode cards, whose chronology and mechanism are written by hand from the
sources of the time. The scenario corpus keeps it in ``corpus/`` of the repository, one copy
under version control. The data root of a workspace (``workspace/corpus``) holds what the
commands download and derive: the originals in ``raw/``, the converted texts the manifest paths
name, ``text/``, ``series/``, ``index/`` and ``downloads.json``.

The corpus of the models has no authored copy in the repository: its two roots are the same
folder of the workspace.
"""

from dataclasses import dataclass
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[3] / "corpus"
"""The authored part of the scenario corpus in the repository."""


@dataclass(frozen=True)
class Roots:
    rules: Path
    """The authored part: ``manifest.json``, ``series.toml``, ``templates/`` and the cards."""
    data: Path
    """The downloaded and derived part: originals, texts, series, index and downloads."""

    @property
    def manifest(self) -> Path:
        return self.rules / "manifest.json"

    @property
    def catalog(self) -> Path:
        return self.rules / "series.toml"

    def card(self, episode) -> Path:
        """The card of an episode of the manifest."""
        return self.rules / episode.path

    def text(self, document) -> Path:
        """The converted text of a document of the manifest."""
        return self.data / document.path

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def private(self) -> Path:
        """The extracted texts a summary is written from, never indexed as they are."""
        return self.data / "text"

    @property
    def series(self) -> Path:
        return self.data / "series"

    @property
    def index(self) -> Path:
        return self.data / "index"

    @property
    def downloads(self) -> Path:
        return self.data / "downloads.json"


def scenario(data: Path) -> Roots:
    """The scenario corpus: the authored part of the repository, the data of ``data``."""
    return Roots(rules=REPOSITORY, data=data)


def model(data: Path) -> Roots:
    """The corpus of the models: one folder of the workspace holds both parts."""
    return Roots(rules=data, data=data)
