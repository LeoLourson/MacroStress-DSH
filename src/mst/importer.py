"""One-off import of the old project's data into a workspace."""

from pathlib import Path

from .data.store import SeriesStore
from .store import Workspace


def import_series(old_root: Path, ws: Workspace) -> dict[str, str | int]:
    """Copy ``<old_root>/data/series/*.csv`` into ``<workspace>/series``.

    Every file is read through the store first, so a row the store cannot parse stops the
    import instead of landing in the new workspace.
    """
    source = Path(old_root) / "data" / "series"
    if not source.is_dir():
        raise FileNotFoundError(f"no series under {source}")
    files = sorted(source.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no series files under {source}")
    old = SeriesStore(source)
    target = SeriesStore(ws.series)
    for path in files:
        target.write(path.stem, old.read(path.stem))
    return {"copied": len(files), "into": str(ws.series)}
