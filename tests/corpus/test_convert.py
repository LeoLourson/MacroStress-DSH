from datetime import date

import yaml

from mst.corpus import convert as c
from mst.corpus import manifest as m

from .pdfs import text_pdf


def document(tmp_path, **spec):
    src = m.Source(
        id="M90",
        kind="methodology",
        title="Stress tests",
        publisher="P",
        year=2019,
        path="sources/methodology/M90.md",
        language="en",
        jurisdiction="intl",
        url=None,
        url_verified=False,
        text_policy="full",
        factors=["fx"],
        purpose="",
        status="planned",
        verify=False,
        file=m.FileSpec(url="https://s.org/a.pdf", **spec),
    )
    return m.documents(src)[0]


RECORD = {"url": "https://s.org/a.pdf", "sha256": "a" * 64, "bytes": 1, "retrieved": "2026-10-29"}


def test_paragraphs_join_lines_and_rejoin_hyphenated_words():
    lines = [
        "The scenario is con-",
        "structed from paths",
        "of 28 variables.",
        "Table 1. Paths",
        "• first item",
    ]
    assert c.paragraphs(lines) == [
        "The scenario is constructed from paths of 28 variables.",
        "Table 1. Paths",
        "• first item",
    ]


def test_running_lines_and_page_numbers_are_dropped():
    bodies = ["Rates rise.", "Credit slows.", "Prices fall.", "Banks lose.", "Policy reacts."]
    pages = [
        c.Page(str(i), ["Financial Stability Review", body, str(i)])
        for i, body in enumerate(bodies, start=1)
    ]
    assert [p.lines for p in c._strip_running(pages)] == [[body] for body in bodies]


def test_a_pdf_becomes_markdown_with_a_header_and_page_marks(tmp_path):
    doc = document(tmp_path, published=date(2019, 3, 1))
    raw = doc.raw_path(tmp_path / "raw")
    raw.parent.mkdir(parents=True)
    raw.write_bytes(text_pdf([["First page text."], []]))
    result = c.convert(
        doc,
        RECORD,
        raw_root=tmp_path / "raw",
        corpus_root=tmp_path / "corpus",
        private_root=tmp_path / "text",
    )
    text = result.path.read_text(encoding="utf-8")
    header = yaml.safe_load(text.split("---\n")[1])
    assert header["published"] == date(2019, 3, 1) and header["published_basis"] == "manifest"
    assert header["source_sha256"] == "a" * 64 and header["extraction"] == "pdf-text-layer"
    assert "<!-- p. 1 -->\n\nFirst page text." in text
    assert "<!-- p. 2: нет текстового слоя -->" in text
    assert (result.pages, result.empty_pages, result.private) == (2, 1, False)


def test_without_a_known_day_the_end_of_the_year_is_taken(tmp_path):
    assert c.published(document(tmp_path), None) == (date(2019, 12, 31), "year_end")
    assert c.published(document(tmp_path), date(2019, 5, 6)) == (date(2019, 5, 6), "pdf_metadata")
    assert c.published(document(tmp_path), date(2021, 5, 6))[1] == "year_end"


def test_html_keeps_headings_paragraphs_and_tables():
    body = """<html><body><nav>menu</nav><main><h2>Scenario</h2><p>Rates rise.</p>
    <table><tr><th>Year</th><th>Rate</th></tr><tr><td>2023</td><td>6</td></tr></table></main>"""
    assert c.html_blocks(body) == [
        "### Scenario",
        "Rates rise.",
        "| Year | Rate |\n|---|---|\n| 2023 | 6 |",
    ]
