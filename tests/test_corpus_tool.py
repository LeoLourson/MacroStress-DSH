"""The corpus tool: search, get and list read the index; index and fetch start a task."""

import asyncio
import json

from fastmcp import Client

from mst import store
from mst.corpus import index as ix
from mst.corpus import manifest as m
from mst.corpus import tool
from mst.mcp import build

DOC = """---
published: 2023-11-10
---

# ОДКП 2024

<!-- p. 1 -->

Введение.

## Сценарии

<!-- p. 12 -->

Рисковый сценарий: глобальный кризис и падение цен на нефть.
"""
MANIFEST = {
    "corpus_id": "test",
    "version": "1",
    "updated": "2026-10-29",
    "description": "a corpus of one document",
    "vocabularies": {"factors": {"fx": "Курс"}, "flags": {}},
    "sources": [
        {
            "id": "S1",
            "kind": "scenario_set",
            "title": "ОДКП",
            "publisher": "Банк России",
            "path": "sources/S1.md",
            "language": "ru",
            "jurisdiction": "ru",
            "url": None,
            "url_verified": False,
            "text_policy": "full",
            "factors": ["fx"],
            "purpose": "",
            "status": "raw",
            "verify": False,
            "file": {"url": "https://cbr.ru/s1.pdf"},
        }
    ],
    "data_series": [],
    "episodes": [],
}


class Bm25Only:
    """The index of the corpus without vectors: the ranking itself is tested in tests/corpus."""

    @staticmethod
    def load(root):
        with (root / "chunks.jsonl").open(encoding="utf-8") as file:
            return ix.Index([ix.Chunk(**json.loads(line)) for line in file])


def call(ws, name, arguments):
    """One tool call against a server built for ``ws``."""

    async def run():
        async with Client(build(ws)) as client:
            return await client.call_tool(name, arguments)

    return asyncio.run(run())


def corpus_root(tmp_path, monkeypatch):
    """A corpus of one document with its index, and the tool wired to it."""
    root = tmp_path / "corpus"
    (root / "sources").mkdir(parents=True)
    m.save(m.Manifest.model_validate(MANIFEST), root / "manifest.json")
    (root / "sources" / "S1.md").write_text(DOC, encoding="utf-8")
    built = ix.build(m.load(root / "manifest.json"), root)
    ix.write(built.chunks, root / "index")
    monkeypatch.setattr(tool, "Index", Bm25Only)
    return root


def test_list_shows_what_the_manifest_holds(tmp_path, monkeypatch):
    ws = store.Workspace(tmp_path)
    corpus_root(tmp_path, monkeypatch)
    result = call(ws, "corpus", {"action": "list"}).data
    assert result["corpus_id"] == "test"
    assert [source["id"] for source in result["sources"]] == ["S1"]
    assert result["sources"][0]["status"] == "raw"
    assert result["sources"][0]["files"][0]["converted"] is True
    assert result["sources"][0]["files"][0]["downloaded"] is False


def test_search_returns_chunks_visible_on_the_day(tmp_path, monkeypatch):
    ws = store.Workspace(tmp_path)
    corpus_root(tmp_path, monkeypatch)
    found = call(ws, "corpus", {"action": "search", "query": "нефть", "as_of": "2024-01-01"})
    assert [chunk["chunk_id"] for chunk in found.data["chunks"]] == ["S1#1/1"]
    earlier = call(ws, "corpus", {"action": "search", "query": "нефть", "as_of": "2023-01-01"})
    assert earlier.data["chunks"] == []


def test_get_names_a_chunk_by_id_and_its_publication_day(tmp_path, monkeypatch):
    ws = store.Workspace(tmp_path)
    corpus_root(tmp_path, monkeypatch)
    chunk = call(ws, "corpus", {"action": "get", "id": "S1#1/1", "as_of": "2024-01-01"}).data
    assert chunk["doc_id"] == "S1" and chunk["pages"] == ["12"]
    assert "падение цен на нефть" in chunk["text"]
    try:
        call(ws, "corpus", {"action": "get", "id": "S1#1/1", "as_of": "2023-01-01"})
    except Exception as error:
        assert "published on 2023-11-10" in str(error)
    else:
        raise AssertionError("a chunk must not be shown before its publication day")


def test_search_and_get_need_their_arguments(tmp_path, monkeypatch):
    ws = store.Workspace(tmp_path)
    corpus_root(tmp_path, monkeypatch)
    for arguments in ({"action": "search", "as_of": "2024-01-01"}, {"action": "get", "id": "x"}):
        try:
            call(ws, "corpus", arguments)
        except Exception as error:
            assert "is required for this action" in str(error)
        else:
            raise AssertionError(f"{arguments} must not be accepted")


def test_index_and_fetch_start_a_task_on_the_corpus_resource(tmp_path):
    ws = store.Workspace(tmp_path)
    expected = {
        "index": ["-m", "mst.corpus", "index", "--corpus", str(ws.corpus)],
        "fetch": ["-m", "mst.corpus", "fetch", "S1/2024", "--corpus", str(ws.corpus)],
    }
    for action, argv in expected.items():
        job = call(ws, "corpus", {"action": action, "ids": ["S1/2024"]}).data
        try:
            assert job["argv"] == argv
            assert call(ws, "jobs", {"action": "status", "id": job["job"]}).data["running"]
        finally:
            call(ws, "jobs", {"action": "cancel", "id": job["job"]})


def test_add_registers_a_file_by_hand(tmp_path, monkeypatch):
    ws = store.Workspace(tmp_path)
    root = corpus_root(tmp_path, monkeypatch)
    pdf = tmp_path / "s1.pdf"
    pdf.write_bytes(b"%PDF-1.4 body")
    added = call(
        ws,
        "corpus",
        {"action": "add", "key": "S1", "file": str(pdf), "url": "https://cbr.ru/s1.pdf"},
    ).data
    assert added["raw"] == str(root / "raw" / "S1" / "S1.pdf")
    assert (root / "raw" / "S1" / "S1.pdf").read_bytes() == b"%PDF-1.4 body"
    assert "S1" in json.loads((root / "downloads.json").read_text(encoding="utf-8"))
