import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from mst.corpus import manifest as m

CORPUS = Path(__file__).resolve().parents[2] / "corpus"
MANIFEST = CORPUS / "manifest.json"


def test_the_corpus_manifest_loads_and_saves_byte_for_byte(tmp_path):
    loaded = m.load(MANIFEST)
    m.save(loaded, tmp_path / "manifest.json")
    assert (tmp_path / "manifest.json").read_text(encoding="utf-8") == MANIFEST.read_text(
        encoding="utf-8"
    )


def test_documents_expand_editions_with_their_paths_and_files():
    source = next(s for s in m.load(MANIFEST).sources if s.id == "S1")
    docs = m.documents(source)
    assert [d.key for d in docs][:2] == ["S1/2014", "S1/2015"]
    assert docs[0].path == "sources/scenario-sets/S1/2014.md"
    assert docs[0].spec.url.endswith("on_2015(2016-2017).pdf")
    assert docs[0].raw_path(Path("raw")).name == "2014.pdf"


def test_a_file_spec_takes_a_url_or_a_page_with_a_pattern():
    assert m.FileSpec(url="https://x/a.pdf").url
    assert m.FileSpec(page="https://x/", match=r"a\.pdf").match
    for bad in (
        {},
        {"url": "u", "page": "p", "match": "m"},
        {"page": "p"},
        {"page": "p", "match": "("},
    ):
        with pytest.raises(ValidationError):
            m.FileSpec(**bad)


def test_files_for_an_edition_outside_the_list_are_refused():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    s1 = next(s for s in data["sources"] if s["id"] == "S1")
    s1["files"]["1999"] = {"url": "https://x/a.pdf"}
    with pytest.raises(ValidationError, match="unknown editions"):
        m.Manifest.model_validate(data)
