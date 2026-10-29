"""Turn a downloaded original into a corpus Markdown file (corpus/README.md, «Документ»).

The text comes from the PDF text layer as pypdf reads it; there is no OCR, so a scanned page
comes out empty and is marked so. Each page starts with ``<!-- p. N -->``, where N is the
page label the PDF shows (the printed number when the PDF declares it). Headings come from the
PDF outline: a bookmark of depth d becomes a heading of level d + 2, down to ``####``, at the
top of its page. Running headers and footers, lines that repeat on most pages, are dropped.
Tables come out as lines of text, not Markdown tables; proofreading them is what moves a source
from ``raw`` to ``ready``.

A source with ``text_policy: summary`` (a book or a subscription journal) keeps its text out of
git: it is written to ``<corpus>/text/`` as material for a hand-written summary.
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from html.parser import HTMLParser
from pathlib import Path

import yaml
from pypdf import PdfReader

from .manifest import Document

# pypdf warns on every font it cannot fully decode; the text it gets is kept as it is.
logging.getLogger("pypdf").setLevel(logging.ERROR)
EMPTY_PAGE = "нет текстового слоя"


@dataclass(frozen=True)
class Page:
    label: str
    lines: list[str]


@dataclass(frozen=True)
class Text:
    pages: list[Page]
    headings: dict[int, list[tuple[int, str]]]
    """Outline entries by page index: (level 2–4, title)."""
    created: date | None
    """The creation date in the PDF metadata."""

    @property
    def empty_pages(self) -> int:
        return sum(1 for page in self.pages if not page.lines)


def read_pdf(path: Path) -> Text:
    reader = PdfReader(path)
    try:
        labels = list(reader.page_labels)
    except Exception:  # noqa: BLE001 - a broken label tree falls back to page numbers
        labels = []
    if len(labels) != len(reader.pages):
        labels = [str(i + 1) for i in range(len(reader.pages))]
    pages = [Page(labels[i], _lines(_page_text(page))) for i, page in enumerate(reader.pages)]
    return Text(_strip_running(pages), _outline(reader), _created(reader))


def _page_text(page) -> str:
    """The page's text layer; a page pypdf cannot read comes out empty, like a scan."""
    try:
        return page.extract_text() or ""
    except Exception:  # noqa: BLE001 - one unreadable page does not lose the document
        return ""


def _lines(text: str) -> list[str]:
    return [" ".join(line.split()) for line in text.splitlines() if line.strip()]


def _strip_running(pages: list[Page]) -> list[Page]:
    """Drop lines that open or close more than half of the pages, and bare page numbers."""
    edges = Counter()
    for page in pages:
        for line in {*page.lines[:2], *page.lines[-2:]}:
            edges[_shape(line)] += 1
    running = {
        shape for shape, count in edges.items() if len(pages) >= 4 and count > len(pages) / 2
    }
    cleaned = []
    for page in pages:
        lines = list(page.lines)
        head = [line for line in lines[:2] if _shape(line) not in running]
        tail = (
            [line for line in lines[-2:] if _shape(line) not in running] if len(lines) > 2 else []
        )
        lines = (
            head + lines[2:-2] + tail
            if len(lines) > 4
            else [line for line in lines if _shape(line) not in running]
        )
        if lines and re.fullmatch(r"[-–—\s]*\d{1,4}[-–—\s]*", lines[-1]):
            lines = lines[:-1]
        if lines and re.fullmatch(r"[-–—\s]*\d{1,4}[-–—\s]*", lines[0]):
            lines = lines[1:]
        cleaned.append(Page(page.label, lines))
    return cleaned


def _shape(line: str) -> str:
    """A line with its digits blanked, so «Page 3» and «Page 4» count as one running line."""
    return re.sub(r"\d+", "#", line.strip().lower())


def _outline(reader: PdfReader) -> dict[int, list[tuple[int, str]]]:
    headings: dict[int, list[tuple[int, str]]] = {}

    def walk(items, depth):
        for item in items:
            if isinstance(item, list):
                walk(item, depth + 1)
                continue
            try:
                index = reader.get_destination_page_number(item)
            except Exception:  # noqa: BLE001 - a bookmark without a page is skipped
                continue
            title = " ".join(str(item.title).split())
            if index is not None and index >= 0 and title:
                headings.setdefault(index, []).append((min(depth + 2, 4), title))

    try:
        walk(reader.outline, 0)
    except Exception:  # noqa: BLE001 - a broken outline gives no headings
        return {}
    return headings


def _created(reader: PdfReader) -> date | None:
    try:
        created = reader.metadata.creation_date if reader.metadata else None
    except Exception:  # noqa: BLE001 - malformed dates are ignored
        return None
    return created.date() if isinstance(created, datetime) else None


def paragraphs(lines: list[str]) -> list[str]:
    """Join the lines of a page into paragraphs.

    A line is joined to the previous one unless the previous ends a sentence and this one starts
    with a capital, a digit or a list mark; a word broken by a hyphen at the line end is rejoined.
    """
    out: list[str] = []
    for line in lines:
        if out and not _breaks(out[-1], line):
            previous = out[-1]
            if re.search(r"\w-$", previous) and line[:1].islower():
                out[-1] = previous[:-1] + line
            else:
                out[-1] = previous + " " + line
        else:
            out.append(line)
    return out


def _breaks(previous: str, line: str) -> bool:
    if re.match(
        r"^([•▪●◦\-–—*]|\d+[.)]\s|[a-zа-я][.)]\s|Таблица|Рисунок|Table|Figure|Chart|Box)", line
    ):
        return True
    return bool(re.search(r"[.:;!?»”)]$", previous)) and not line[:1].islower()


def published(document: Document, created: date | None) -> tuple[date, str]:
    """The publication day and how it was set: manifest, PDF metadata, or the end of the year.

    The end of the year is the latest the document can have appeared, so ``as_of`` never shows
    it early; the metadata date is used only when its year is the edition's.
    """
    if document.spec is not None and document.spec.published is not None:
        return document.spec.published, "manifest"
    year = _year(document)
    if created is not None and (year is None or created.year == year):
        return created, "pdf_metadata"
    if year is None:
        raise ValueError(f"{document.key}: no year to date the document by")
    return date(year, 12, 31), "year_end"


def _year(document: Document) -> int | None:
    if document.edition and (m := re.match(r"\d{4}", document.edition)):
        return int(m.group())
    return document.source.year


def header(document: Document, record: dict, day: date, basis: str, extraction: str) -> str:
    source = document.source
    fields = {
        "id": source.id,
        "edition": document.edition,
        "kind": source.kind,
        "title": source.title,
        "publisher": source.publisher,
        "authors": list(source.authors),
        "published": day,
        "published_basis": basis,
        "language": source.language,
        "jurisdiction": source.jurisdiction,
        "url": record["url"],
        "retrieved": date.fromisoformat(record["retrieved"]),
        "source_sha256": record["sha256"],
        "text_policy": source.text_policy,
        "extraction": extraction,
        "factors": list(source.factors),
    }
    if document.edition is None:
        del fields["edition"]
    body = yaml.safe_dump(fields, allow_unicode=True, sort_keys=False, width=1000)
    return f"---\n{body}---\n"


def pdf_markdown(document: Document, record: dict, text: Text) -> str:
    day, basis = published(document, text.created)
    title = document.source.title + (f" ({document.edition})" if document.edition else "")
    parts = [header(document, record, day, basis, "pdf-text-layer"), f"# {title}\n"]
    for index, page in enumerate(text.pages):
        if not page.lines:
            parts.append(f"<!-- p. {page.label}: {EMPTY_PAGE} -->\n")
            continue
        parts.append(f"<!-- p. {page.label} -->\n")
        for level, heading in text.headings.get(index, []):
            parts.append(f"{'#' * level} {heading}\n")
        parts.extend(f"{paragraph}\n" for paragraph in paragraphs(page.lines))
    return "\n".join(parts)


class _HtmlText(HTMLParser):
    """Headings, paragraphs, list items and table rows of a page's main part, as Markdown."""

    SKIP = {"script", "style", "nav", "header", "footer", "noscript", "form", "button", "svg"}
    BLOCKS = {"p", "li", "div", "section", "article", "br", "tr", "dd", "dt", "blockquote"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self.current: list[str] = []
        self.skip = 0
        self.heading = 0
        self.row: list[str] | None = None
        self.cell: list[str] | None = None

    def flush(self):
        text = " ".join("".join(self.current).split())
        self.current = []
        if text:
            self.blocks.append(("#" * self.heading + " " if self.heading else "") + text)

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif self.skip:
            return
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self.flush()
            self.heading = min(int(tag[1]) + 1, 4)
        elif tag == "tr":
            self.flush()
            self.row = []
        elif tag in {"td", "th"} and self.row is not None:
            self.cell = []
        elif tag in self.BLOCKS:
            self.flush()

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif self.skip:
            return
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self.flush()
            self.heading = 0
        elif tag in {"td", "th"} and self.cell is not None and self.row is not None:
            self.row.append(" ".join("".join(self.cell).split()).replace("|", "/"))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            if any(self.row):
                self.blocks.append("| " + " | ".join(self.row) + " |")
            self.row = None
        elif tag in self.BLOCKS:
            self.flush()

    def handle_data(self, data):
        if self.skip:
            return
        (self.cell if self.cell is not None else self.current).append(data)


def html_blocks(body: str) -> list[str]:
    main = re.search(r"<main\b.*?</main>", body, re.S | re.I) or re.search(
        r"<body\b.*?</body>", body, re.S | re.I
    )
    parser = _HtmlText()
    parser.feed(main.group() if main else body)
    parser.flush()
    blocks, table = [], []
    for block in parser.blocks:
        if block.startswith("| "):
            table.append(block)
            continue
        if table:
            blocks.append(_table(table))
            table = []
        blocks.append(block)
    if table:
        blocks.append(_table(table))
    return blocks


def _table(rows: list[str]) -> str:
    width = max(row.count(" | ") + 1 for row in rows)
    padded = [row[:-2] + " | " * (width - row.count(" | ") - 1) + " |" for row in rows]
    return "\n".join([padded[0], "|" + "---|" * width, *padded[1:]])


def html_markdown(document: Document, record: dict, body: str) -> str:
    day, basis = published(document, None)
    title = document.source.title + (f" ({document.edition})" if document.edition else "")
    parts = [header(document, record, day, basis, "html"), f"# {title}\n", "<!-- p. 1 -->\n"]
    parts.extend(f"{block}\n" for block in html_blocks(body))
    return "\n".join(parts)


@dataclass(frozen=True)
class Converted:
    path: Path
    pages: int
    empty_pages: int
    private: bool


def convert(
    document: Document,
    record: dict,
    *,
    raw_root: Path,
    corpus_root: Path,
    private_root: Path,
) -> Converted:
    raw = document.raw_path(raw_root)
    if document.spec.format == "html":
        markdown = html_markdown(
            document, record, raw.read_text(encoding="utf-8", errors="replace")
        )
        pages = empty = 0
    else:
        text = read_pdf(raw)
        markdown = pdf_markdown(document, record, text)
        pages, empty = len(text.pages), text.empty_pages
    private = document.source.text_policy == "summary"
    target = (private_root / f"{document.key}.md") if private else corpus_root / document.path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(markdown, encoding="utf-8")
    return Converted(target, pages, empty, private)
