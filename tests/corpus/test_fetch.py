import json
from pathlib import Path

import httpx
import pytest

from mst.corpus import fetch as f
from mst.corpus import manifest as m

CORPUS = Path(__file__).resolve().parents[2] / "corpus"
MANIFEST = CORPUS / "manifest.json"
PDF = b"%PDF-1.4 body"


def source(**files):
    return m.Source(
        id="M90",
        kind="methodology",
        title="T",
        publisher="P",
        editions=list(files),
        path="sources/methodology/M90/<edition>.md",
        language="en",
        jurisdiction="intl",
        url=None,
        url_verified=False,
        text_policy="full",
        factors=[],
        purpose="",
        status="planned",
        verify=False,
        files=files,
    )


def manifest(src):
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    built = m.Manifest.model_validate(data)
    return built.model_copy(update={"sources": [src]})


def http(routes, seen=None):
    def respond(request):
        if seen is not None:
            seen.append(str(request.url))
        body = routes[str(request.url)]
        return httpx.Response(200, content=body if isinstance(body, bytes) else body.encode())

    return httpx.Client(transport=httpx.MockTransport(respond))


def test_links_are_absolute_and_carry_their_text():
    body = '<a class="x" href="/files/a%281%29.pdf"><span>Report</span> 2023</a>'
    assert f.links("https://site.org/page/", body) == [
        f.Link("https://site.org/files/a%281%29.pdf", "Report 2023")
    ]


def test_a_page_spec_takes_the_first_matching_link():
    spec = m.FileSpec(page="https://s.org/p", match=r"methodolog")
    page = (
        '<a href="other.pdf">x</a><a href="note.pdf">Methodological note</a><a href="m2.pdf">y</a>'
    )
    with pytest.raises(LookupError):
        f.resolve(http({"https://s.org/p": page}), m.FileSpec(page="https://s.org/p", match="zzz"))
    spec = m.FileSpec(page="https://s.org/p", match=r"(?i)methodolog")
    assert f.resolve(http({"https://s.org/p": page}), spec) == "https://s.org/note.pdf"


def test_fetch_downloads_records_and_then_keeps(tmp_path):
    src = source(
        **{
            "2020": m.FileSpec(url="https://s.org/a.pdf"),
            "2021": m.FileSpec(url="https://s.org/b.pdf"),
        }
    )
    src = src.model_copy(update={"editions": ["2020", "2021", "2022"]})
    routes = {"https://s.org/a.pdf": PDF, "https://s.org/b.pdf": "<html>not found</html>"}
    lock = tmp_path / "downloads.json"
    seen = []
    outcome = f.fetch(manifest(src), http(routes, seen), root=tmp_path, downloads_path=lock)
    assert outcome.fetched == ["M90/2020"]
    assert "expected a PDF" in outcome.failed["M90/2021"]
    assert outcome.unspecified == ["M90/2022"]
    record = json.loads(lock.read_text(encoding="utf-8"))["M90/2020"]
    assert record["bytes"] == len(PDF) and record["url"] == "https://s.org/a.pdf"
    assert (tmp_path / "M90" / "2020.pdf").read_bytes() == PDF
    seen.clear()
    again = f.fetch(manifest(src), http(routes, seen), root=tmp_path, downloads_path=lock)
    assert again.kept == ["M90/2020"] and "https://s.org/a.pdf" not in seen
