"""Download the originals of the corpus sources and record what was downloaded.

Each file goes to ``<corpus>/raw/<id>/<edition>.pdf`` and gets a line in
``<corpus>/downloads.json``: the address it came from, its SHA-256, its size and the day it was
downloaded. ``convert`` copies the hash and the day into the Markdown header. A file already
on disk with the recorded hash is not downloaded again.
"""

import hashlib
import html
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import urljoin

import httpx

from .manifest import Document, FileSpec, Manifest, documents

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) mst-corpus/1.0"
LINK = re.compile(r"<a\b[^>]*?\bhref\s*=\s*[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", re.S | re.I)
DOCUMENT = re.compile(r"\.(pdf|ashx|docx?|xlsx?)(\?|#|$)", re.I)


def client() -> httpx.Client:
    return httpx.Client(timeout=120, follow_redirects=True, headers={"User-Agent": USER_AGENT})


@dataclass(frozen=True)
class Link:
    url: str
    text: str


def links(page_url: str, body: str) -> list[Link]:
    """Every link of an HTML page, with absolute addresses and the link text without tags."""
    found = []
    for href, inner in LINK.findall(body):
        text = " ".join(re.sub(r"<[^>]+>", " ", html.unescape(inner)).split())
        found.append(Link(urljoin(page_url, html.unescape(href.strip())), text))
    return found


def resolve(http: httpx.Client, spec: FileSpec) -> str:
    """The address of the file: ``url`` itself, or the first link of ``page`` matching ``match``."""
    if spec.url is not None:
        return spec.url
    response = http.get(spec.page)
    response.raise_for_status()
    pattern = re.compile(spec.match)
    for link in links(str(response.url), response.text):
        if pattern.search(link.url) or pattern.search(link.text):
            return link.url
    raise LookupError(f"no link on {spec.page} matches {spec.match!r}")


def discover(http: httpx.Client, page: str, pattern: str | None = None) -> list[Link]:
    """Links to documents on ``page``, to choose editions and write ``files`` in the manifest."""
    response = http.get(page)
    response.raise_for_status()
    chosen = re.compile(pattern) if pattern else None
    found = []
    for link in links(str(response.url), response.text):
        if chosen is not None:
            if chosen.search(link.url) or chosen.search(link.text):
                found.append(link)
        elif DOCUMENT.search(link.url):
            found.append(link)
    return found


def load_downloads(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_downloads(records: dict[str, dict], path: Path) -> None:
    text = json.dumps(
        dict(sorted(records.items(), key=lambda item: _order(item[0]))),
        ensure_ascii=False,
        indent=2,
    )
    path.write_text(text + "\n", encoding="utf-8")


def _order(key: str) -> tuple:
    head, _, edition = key.partition("/")
    return head[0], int(head[1:]), edition


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_content(spec: FileSpec, content: bytes, content_type: str) -> None:
    if spec.format == "pdf" and not content.lstrip()[:5].startswith(b"%PDF"):
        raise ValueError(f"expected a PDF, got {content_type or 'no content type'}")
    if spec.format == "html" and b"<" not in content[:2048]:
        raise ValueError(f"expected an HTML page, got {content_type or 'no content type'}")


def download(http: httpx.Client, document: Document, root: Path) -> dict:
    """Download one document into ``root`` and return its record for ``downloads.json``."""
    url = resolve(http, document.spec)
    response = http.get(url)
    response.raise_for_status()
    content = response.content
    check_content(document.spec, content, response.headers.get("content-type", ""))
    target = document.raw_path(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    partial.write_bytes(content)
    partial.replace(target)
    record = {
        "url": str(response.url),
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
        "retrieved": date.today().isoformat(),
    }
    if document.spec.page is not None:
        record["page"] = document.spec.page
    return record


def add(document: Document, local: Path, url: str, root: Path) -> dict:
    """Register a file downloaded by hand (a site that refuses scripts) as if ``fetch`` got it."""
    content = local.read_bytes()
    check_content(document.spec or FileSpec(url=url), content, "the local file")
    target = document.raw_path(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return {
        "url": url,
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
        "retrieved": date.today().isoformat(),
        "by_hand": True,
    }


@dataclass
class Outcome:
    fetched: list[str]
    kept: list[str]
    failed: dict[str, str]
    unspecified: list[str]
    """Documents whose original has no address in the manifest yet."""


def fetch(
    manifest: Manifest,
    http: httpx.Client,
    only: set[str] | None = None,
    *,
    root: Path,
    downloads_path: Path,
    force: bool = False,
) -> Outcome:
    """Download every document of the manifest (or of the sources in ``only``) not yet on disk.

    The record of each new download is saved at once, so an interrupted run keeps its work.
    """
    records = load_downloads(downloads_path)
    outcome = Outcome([], [], {}, [])
    for source in manifest.sources:
        if only and source.id not in only:
            continue
        for document in documents(source):
            if document.spec is None:
                outcome.unspecified.append(document.key)
                continue
            target = document.raw_path(root)
            known = records.get(document.key)
            if not force and known and target.exists() and sha256(target) == known["sha256"]:
                outcome.kept.append(document.key)
                continue
            try:
                records[document.key] = download(http, document, root)
            except (httpx.HTTPError, LookupError, ValueError) as error:
                outcome.failed[document.key] = f"{type(error).__name__}: {error}"
                continue
            save_downloads(records, downloads_path)
            outcome.fetched.append(document.key)
    return outcome
