"""The corpus index and search (corpus/README.md, «Фрагмент индекса», «Индексация»).

``build`` cuts every Markdown file of the manifest into chunks and writes
``<corpus>/index/chunks.jsonl``; ``write_vectors`` writes their e5 vectors (``embed.py``) next to
it and ``write_state`` records the day the index was built. ``Index.search`` ranks the chunks
that pass the filters twice, by BM25 over the heading path and the text with Russian and English
stemming, and by the cosine of the query vector, and adds the reciprocal ranks of the two lists.
Tokens are words: a run of letters or digits.
"""

import functools
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import snowballstemmer
import yaml

from .. import store
from .embed import MODEL, embed
from .manifest import Manifest, documents

MAX_TOKENS = 800
MIN_TOKENS = 300
PAGE = re.compile(r"<!--\s*p\.\s*([^\s:>]+)[^>]*-->")
COMMENT = re.compile(r"<!--.*?-->", re.S)
WORD = re.compile(r"\w+", re.U)
HEADING = re.compile(r"^(#{1,4})\s+(.*)$")
K1, B = 1.5, 0.75
RANKED = 100
"""How deep each ranking goes before the two are fused."""
RRF = 60
"""The constant of reciprocal rank fusion: a chunk at rank r of a list scores 1 / (RRF + r)."""


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    edition: str | None
    kind: str
    region: str | None
    jurisdiction: str | None
    visible_from: str
    title: str
    """The document title, the ``#`` heading; searched together with the heading path."""
    heading_path: list[str]
    pages: list[str]
    factors: list[str]
    text: str
    sha256: str


def normalize(text: str) -> str:
    """NFC, page comments removed, whitespace collapsed: the form the hash is taken of."""
    text = PAGE.sub(" ", unicodedata.normalize("NFC", text))
    return " ".join(text.split())


def tokens_count(text: str) -> int:
    return len(WORD.findall(text))


@dataclass
class Section:
    path: list[str]
    number: str
    blocks: list[str] = field(default_factory=list)


def front_matter(text: str) -> tuple[dict, str]:
    match = re.match(r"---\n(.*?)\n---\n(.*)", text, re.S)
    if not match:
        raise ValueError("the file has no YAML header")
    return yaml.safe_load(match.group(1)) or {}, match.group(2)


def sections(body: str) -> list[Section]:
    """The body cut at headings ``##``–``####``; ``#`` is the document title and opens nothing.

    A section is numbered by its position among its siblings: ``3.2`` is the second ``###`` of
    the third ``##``. Text before the first heading is section ``0``.
    """
    out = [Section([], "0")]
    counters = [0, 0, 0]
    stack: list[str] = []
    block: list[str] = []

    def close_block():
        if block:
            out[-1].blocks.append("\n".join(block).strip())
            block.clear()

    in_table = False
    for line in body.splitlines():
        heading = HEADING.match(line)
        if heading and len(heading.group(1)) >= 2:
            close_block()
            level = len(heading.group(1)) - 2
            counters[level] += 1
            for deeper in range(level + 1, 3):
                counters[deeper] = 0
            stack[level:] = [heading.group(2).strip()]
            out.append(Section(list(stack), ".".join(str(c) for c in counters[: level + 1])))
            continue
        if heading:
            continue
        if not line.strip():
            close_block()
            in_table = False
            continue
        is_table = line.lstrip().startswith("|")
        if block and is_table != in_table:
            close_block()
        block.append(line)
        in_table = is_table
    close_block()
    return [section for section in out if any(normalize(b) for b in section.blocks)]


def pieces(blocks: list[str]) -> list[list[int]]:
    """A section's blocks, by index, in pieces of at most ``MAX_TOKENS``, overlapping by one block.

    A table stays whole and in the piece of the paragraph before it. A section that fits stays
    one piece; a piece is cut only once it holds ``MIN_TOKENS``.
    """
    if sum(tokens_count(b) for b in blocks) <= MAX_TOKENS:
        return [list(range(len(blocks)))]
    units: list[list[int]] = []
    for i, block in enumerate(blocks):
        if block.lstrip().startswith("|") and units:
            units[-1].append(i)
        else:
            units.append([i])
    out, current, size = [], [], 0
    for unit in units:
        unit_size = sum(tokens_count(blocks[i]) for i in unit)
        if current and size >= MIN_TOKENS and size + unit_size > MAX_TOKENS:
            out.append(current)
            last = current[-1]
            current = [] if blocks[last].lstrip().startswith("|") else [last]
            size = sum(tokens_count(blocks[i]) for i in current)
        current = current + unit
        size += unit_size
    if current:
        out.append(current)
    return out


def chunks_of(
    text: str,
    *,
    doc_id: str,
    edition: str | None,
    kind: str,
    region: str | None,
    jurisdiction: str | None,
    visible_from: str,
    factors: list[str],
) -> list[Chunk]:
    _, body = front_matter(text)
    name = doc_id if edition is None else f"{doc_id}/{edition}"
    found_title = re.search(r"(?m)^# (.+)$", body)
    title = found_title.group(1).strip() if found_title else name
    parts = sections(body)
    # The page in effect where each block starts, in reading order.
    starts, page = [], None
    for section in parts:
        starts.append([])
        for block in section.blocks:
            starts[-1].append(page)
            found = PAGE.findall(block)
            if found:
                page = found[-1]
    out = []
    for section, section_starts in zip(parts, starts, strict=True):
        for n, indices in enumerate(pieces(section.blocks), start=1):
            raw = "\n\n".join(section.blocks[i] for i in indices)
            clean = COMMENT.sub("", raw).strip()
            if not normalize(clean):
                continue
            pages = PAGE.findall(raw)
            first = section_starts[indices[0]]
            if first is not None and not PAGE.match(section.blocks[indices[0]].lstrip()):
                pages = [first, *pages]
            out.append(
                Chunk(
                    f"{name}#{section.number}/{n}",
                    doc_id,
                    edition,
                    kind,
                    region,
                    jurisdiction,
                    visible_from,
                    title,
                    section.path,
                    list(dict.fromkeys(pages)),
                    factors,
                    clean,
                    hashlib.sha256(normalize(clean).encode()).hexdigest(),
                )
            )
    return out


@dataclass
class Built:
    chunks: list[Chunk]
    errors: list[str]
    skipped: list[str]


def build(manifest: Manifest, root: Path) -> Built:
    built = Built([], [], [])
    for source in manifest.sources:
        if source.status == "planned":
            continue
        region = "ru" if source.jurisdiction == "ru" else None
        for document in documents(source):
            path = root / document.path
            if not path.exists():
                (built.errors if source.status == "ready" else built.skipped).append(document.key)
                continue
            text = path.read_text(encoding="utf-8")
            fields, _ = front_matter(text)
            published = fields.get("published")
            if published is None:
                built.errors.append(f"{document.key}: no published date")
                continue
            built.chunks += chunks_of(
                text,
                doc_id=source.id,
                edition=document.edition,
                kind=source.kind,
                region=region,
                jurisdiction=source.jurisdiction,
                visible_from=str(published),
                factors=list(source.factors),
            )
    for episode in manifest.episodes:
        if episode.status == "planned":
            continue
        path = root / episode.path
        if not path.exists():
            built.errors.append(episode.id)
            continue
        text = path.read_text(encoding="utf-8")
        fields, _ = front_matter(text)
        built.chunks += chunks_of(
            text,
            doc_id=episode.id,
            edition=None,
            kind="episode",
            region=episode.region,
            jurisdiction=episode.country,
            visible_from=str(fields["known_from"]),
            factors=list(episode.factors),
        )
    return built


def write(chunks: list[Chunk], root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "chunks.jsonl"
    with path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    return path


def write_state(chunks: list[Chunk], root: Path, day: date) -> dict:
    """``<corpus>/index/state.json``: the day of the build, read by the prompt section."""
    path = root / "chunks.jsonl"
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    snapshot = {"as_of": day.isoformat(), "chunks": len(chunks), "sha256": digest}
    store.write_json(root / "state.json", snapshot)
    return snapshot


def passage(chunk: Chunk) -> str:
    """The text a chunk's vector is computed from: title, heading path and text."""
    return ". ".join([chunk.title, " / ".join(chunk.heading_path), chunk.text])


def _key(chunk: Chunk) -> str:
    return hashlib.sha256(passage(chunk).encode()).hexdigest()


def write_vectors(chunks: list[Chunk], root: Path, embed=embed) -> int:
    """``vectors.npy`` (float16, a row per line of ``chunks.jsonl``) and ``vectors.json``.

    A vector is reused when the previous index of the same model had the same passage; the
    number of vectors computed anew is returned.
    """
    keys = [_key(chunk) for chunk in chunks]
    known: dict[str, np.ndarray] = {}
    meta_path, array_path = root / "vectors.json", root / "vectors.npy"
    if meta_path.exists() and array_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["model"] == MODEL:
            known = dict(zip(meta["keys"], np.load(array_path), strict=True))
    missing = [i for i, key in enumerate(keys) if key not in known]
    fresh = embed([passage(chunks[i]) for i in missing], "passage: ") if missing else []
    known.update({keys[i]: vector for i, vector in zip(missing, fresh, strict=True)})
    vectors = np.array([known[key] for key in keys], dtype="float16")
    root.mkdir(parents=True, exist_ok=True)
    np.save(array_path, vectors)
    meta_path.write_text(json.dumps({"model": MODEL, "keys": keys}), encoding="utf-8")
    return len(missing)


def sources_missing(root: Path, ids: list[str], as_of: date) -> list[str]:
    """The ids that are not chunks of the index visible on ``as_of``."""
    path = root / "chunks.jsonl"
    if not path.is_file():
        return list(ids)
    known = set()
    with path.open(encoding="utf-8") as file:
        for line in file:
            chunk = json.loads(line)
            if chunk["visible_from"] <= as_of.isoformat():
                known.add(chunk["chunk_id"])
    return [id_ for id_ in ids if id_ not in known]


_STEMMERS = {"ru": snowballstemmer.stemmer("russian"), "en": snowballstemmer.stemmer("english")}


@functools.cache
def stem(word: str) -> str:
    language = "ru" if re.search(r"[а-я]", word) else "en"
    return _STEMMERS[language].stemWord(word)


def terms(text: str) -> list[str]:
    return [stem(word) for word in WORD.findall(text.lower().replace("ё", "е"))]


class Index:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray | None = None, embed=embed):
        """Without ``vectors`` the index ranks by BM25 alone."""
        self.chunks = chunks
        self.vectors = vectors
        self.embed = embed
        self.terms = [Counter(terms(" ".join([c.title, *c.heading_path, c.text]))) for c in chunks]
        self.lengths = [sum(t.values()) for t in self.terms]
        self.average = (sum(self.lengths) / len(self.lengths)) if chunks else 0
        frequency = Counter(term for counts in self.terms for term in counts)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in frequency.items()}

    @classmethod
    def load(cls, root: Path) -> "Index":
        with (root / "chunks.jsonl").open(encoding="utf-8") as file:
            chunks = [Chunk(**json.loads(line)) for line in file]
        meta_path = root / "vectors.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"{meta_path} is missing: run python -m mst.corpus index")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["model"] != MODEL or meta["keys"] != [_key(chunk) for chunk in chunks]:
            raise ValueError(
                f"{meta_path} does not match chunks.jsonl: run python -m mst.corpus index"
            )
        return cls(chunks, np.load(root / "vectors.npy").astype("float32"))

    def search(
        self,
        query: str,
        as_of: date,
        kind: str | None = None,
        region: str | None = None,
        factors: list[str] | None = None,
        k: int = 8,
    ) -> list[Chunk]:
        """The ``k`` best chunks visible on ``as_of`` that pass the filters."""
        allowed = [
            i
            for i, chunk in enumerate(self.chunks)
            if chunk.visible_from <= as_of.isoformat()
            and not (kind and chunk.kind != kind)
            and not (region and chunk.region != region)
            and not (factors and not set(factors) & set(chunk.factors))
        ]
        lexical = self._bm25(query, allowed)
        if self.vectors is None:
            return [self.chunks[i] for i in lexical[:k]]
        fused: dict[int, float] = {}
        for ranking in (lexical[:RANKED], self._nearest(query, allowed)):
            for rank, i in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + 1 / (RRF + rank)
        best = sorted(fused, key=lambda i: (-fused[i], i))
        return [self.chunks[i] for i in best[:k]]

    def _nearest(self, query: str, allowed: list[int]) -> list[int]:
        if not allowed:
            return []
        similarity = self.vectors[allowed] @ self.embed([query], "query: ")[0]
        order = np.argsort(-similarity, kind="stable")[:RANKED]
        return [allowed[j] for j in order]

    def _bm25(self, query: str, allowed: list[int]) -> list[int]:
        """``allowed`` chunks that share a term with the query, best BM25 first."""
        wanted = terms(query)
        scored = []
        for i in allowed:
            counts, length = self.terms[i], self.lengths[i]
            score = 0.0
            for term in wanted:
                f = counts.get(term, 0)
                if f:
                    score += (
                        self.idf[term]
                        * f
                        * (K1 + 1)
                        / (f + K1 * (1 - B + B * length / self.average))
                    )
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [i for _, i in scored]


@dataclass(frozen=True)
class Question:
    number: str
    text: str
    as_of: date
    kind: str | None
    expected: list[str]


def questions(text: str) -> list[Question]:
    """Questions of ``corpus/eval/questions.md``: ``## Q<n>. <question>`` and a list of fields."""
    out = []
    for part in re.split(r"(?m)^## ", text)[1:]:
        title, _, body = part.partition("\n")
        match = re.match(r"(Q\d+)\.\s*(.*)", title.strip())
        if not match:
            continue
        fields = dict(re.findall(r"(?m)^- (\w+):\s*(.*)$", body))
        expected = [
            item.strip(" `") for item in fields.get("expected", "").split(",") if item.strip(" `")
        ]
        out.append(
            Question(
                match.group(1),
                match.group(2),
                date.fromisoformat(fields["as_of"].strip()),
                fields.get("kind", "").strip() or None,
                expected,
            )
        )
    return out


def recall(index: Index, items: list[Question], k: int = 5) -> tuple[float | None, list[str]]:
    """Mean share of expected chunks in the top ``k``; questions not marked up are skipped."""
    shares, unmarked = [], []
    for question in items:
        if not question.expected:
            unmarked.append(question.number)
            continue
        found = {
            c.chunk_id for c in index.search(question.text, question.as_of, question.kind, k=k)
        }
        shares.append(len(found & set(question.expected)) / len(question.expected))
    return (sum(shares) / len(shares) if shares else None), unmarked
