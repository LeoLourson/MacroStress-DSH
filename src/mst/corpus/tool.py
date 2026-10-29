"""The operations of the ``corpus`` tool: search, get, list and add.

The tool only parses arguments; the corpus root (``workspace/corpus``) is closed over by the
server. Search and get read the index that ``python -m mst.corpus index`` wrote and never touch
the network; add registers a file downloaded by hand, as ``python -m mst.corpus add`` does.
"""

from datetime import date
from pathlib import Path

from . import manifest as manifest_module
from .fetch import add as add_by_hand
from .fetch import load_downloads, save_downloads
from .index import Index
from .manifest import documents


def search(
    root: Path,
    query: str,
    as_of: date,
    *,
    kind: str | None = None,
    region: str | None = None,
    factors: list[str] | None = None,
    k: int = 8,
) -> dict:
    """The ``k`` best chunks visible on ``as_of``, with their provenance."""
    index = Index.load(root / "index")
    return {
        "query": query,
        "as_of": as_of.isoformat(),
        "chunks": [_chunk(chunk) for chunk in index.search(query, as_of, kind, region, factors, k)],
    }


def get(root: Path, chunk_id: str, as_of: date) -> dict:
    """One chunk by id, if it is visible on ``as_of``."""
    index = Index.load(root / "index")
    for chunk in index.chunks:
        if chunk.chunk_id == chunk_id:
            if chunk.visible_from > as_of.isoformat():
                raise ValueError(f"{chunk_id} is published on {chunk.visible_from}, after {as_of}")
            return _chunk(chunk)
    raise ValueError(f"no chunk {chunk_id!r} in the index")


def catalogue(root: Path) -> dict:
    """What the manifest holds: the corpus, its sources and episode cards with their status."""
    manifest = manifest_module.load(root / "manifest.json")
    downloads = load_downloads(root / "downloads.json")
    return {
        "corpus_id": manifest.corpus_id,
        "version": manifest.version,
        "updated": manifest.updated,
        "sources": [
            {
                "id": source.id,
                "title": source.title,
                "kind": source.kind,
                "status": source.status,
                "text_policy": source.text_policy,
                "factors": list(source.factors),
                "files": [
                    {
                        "key": document.key,
                        "path": document.path,
                        "has_address": document.spec is not None,
                        "downloaded": document.key in downloads,
                        "converted": (root / document.path).exists(),
                    }
                    for document in documents(source)
                ],
            }
            for source in manifest.sources
        ],
        "episodes": [
            {
                "id": episode.id,
                "title": episode.title,
                "start": episode.start,
                "region": episode.region,
                "country": episode.country,
                "status": episode.status,
                "factors": list(episode.factors),
                "card": (root / episode.path).exists(),
            }
            for episode in manifest.episodes
        ],
        "data_series": [series.id for series in manifest.data_series],
    }


def add(root: Path, key: str, file: Path, url: str | None = None) -> dict:
    """Register a file downloaded by hand under the manifest key ``key``."""
    manifest = manifest_module.load(root / "manifest.json")
    found = [document for source in manifest.sources for document in documents(source)]
    document = next((item for item in found if item.key == key), None)
    if document is None:
        raise ValueError(f"no document {key!r} in the manifest")
    address = url or (document.spec.url or document.spec.page if document.spec else None)
    if address is None:
        raise ValueError(f"{key} has no address in the manifest: give url")
    records = load_downloads(root / "downloads.json")
    records[document.key] = add_by_hand(document, Path(file), address, root / "raw")
    save_downloads(records, root / "downloads.json")
    return {"key": document.key, "raw": str(document.raw_path(root / "raw"))}


def _chunk(chunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "edition": chunk.edition,
        "kind": chunk.kind,
        "region": chunk.region,
        "jurisdiction": chunk.jurisdiction,
        "visible_from": chunk.visible_from,
        "title": chunk.title,
        "heading_path": list(chunk.heading_path),
        "pages": list(chunk.pages),
        "factors": list(chunk.factors),
        "text": chunk.text,
    }
