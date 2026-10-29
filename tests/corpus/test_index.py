from datetime import date

import numpy as np
import pytest

from mst.corpus import index as ix

DOC = """---
published: 2023-11-10
---

# ОДКП 2024

<!-- p. 1 -->

Введение.

## Сценарии

<!-- p. 12 -->

Рисковый сценарий: глобальный кризис и падение цен на нефть.

### Глобальный кризис

Ключевая ставка повышается.

| Показатель | 2024 |
|---|---|
| Ставка, % | 16 |

## Риски

<!-- p. 14 -->

Санкции.
"""


def chunks():
    return ix.chunks_of(
        DOC,
        doc_id="S1",
        edition="2023",
        kind="scenario_set",
        region="ru",
        jurisdiction="ru",
        visible_from="2023-11-10",
        factors=["rate.policy"],
    )


def test_sections_are_numbered_and_carry_pages_and_heading_paths():
    found = {c.chunk_id: c for c in chunks()}
    assert list(found) == ["S1/2023#0/1", "S1/2023#1/1", "S1/2023#1.1/1", "S1/2023#2/1"]
    assert found["S1/2023#1.1/1"].heading_path == ["Сценарии", "Глобальный кризис"]
    assert found["S1/2023#1.1/1"].pages == ["12"]
    assert found["S1/2023#2/1"].pages == ["14"]
    assert "| Ставка, % | 16 |" in found["S1/2023#1.1/1"].text
    assert found["S1/2023#0/1"].title == "ОДКП 2024"


def test_long_sections_are_cut_with_tables_kept_whole():
    blocks = [" ".join(["слово"] * 250) for _ in range(5)] + ["| a | b |\n|---|---|\n| 1 | 2 |"]
    parts = ix.pieces(blocks)
    assert all(sum(ix.tokens_count(blocks[i]) for i in p) <= 800 for p in parts[:-1])
    assert parts[-1][-2:] == [4, 5]
    assert parts[1][0] == parts[0][-1]


def test_search_hides_chunks_published_after_as_of_and_ranks_by_stems():
    index = ix.Index(chunks())
    assert index.search("нефть", date(2023, 1, 1)) == []
    top = index.search("падения цены нефти", date(2024, 1, 1), k=1)
    assert top[0].chunk_id == "S1/2023#1/1"
    assert index.search("нефть", date(2024, 1, 1), kind="episode") == []


def test_questions_are_parsed_and_recall_counts_marked_ones():
    text = """## Q1. Как падение нефти входит в рисковый сценарий?
- as_of: 2024-01-01
- expected: S1/2023#1/1

## Q2. Не размечен
- as_of: 2024-01-01
"""
    items = ix.questions(text)
    assert [q.expected for q in items] == [["S1/2023#1/1"], []]
    assert ix.recall(ix.Index(chunks()), items) == (1.0, ["Q2"])


def fake_embed(calls):
    """Two-dimensional vectors: oil-like texts point one way, everything else the other."""

    def embed(texts, prefix):
        calls.append((prefix, list(texts)))
        rows = [[1.0, 0.0] if ("нефт" in t or "oil" in t) else [0.0, 1.0] for t in texts]
        return np.array(rows, dtype="float32")

    return embed


def test_vectors_find_a_chunk_that_shares_no_word_with_the_query(tmp_path):
    found = chunks()
    vectors = np.array([[1.0, 0.0] if "нефт" in c.text else [0.0, 1.0] for c in found])
    index = ix.Index(found, vectors.astype("float32"), embed=fake_embed([]))
    assert index.search("oil price shock", date(2024, 1, 1), k=1)[0].chunk_id == "S1/2023#1/1"
    assert ix.Index(found).search("oil price shock", date(2024, 1, 1)) == []


def test_vectors_are_reused_for_unchanged_passages_and_checked_on_load(tmp_path):
    found, calls = chunks(), []
    ix.write(found, tmp_path)
    assert ix.write_vectors(found, tmp_path, embed=fake_embed(calls)) == len(found)
    assert calls[0][0] == "passage: "
    assert ix.write_vectors(found, tmp_path, embed=fake_embed(calls)) == 0
    assert len(calls) == 1
    assert ix.Index.load(tmp_path).vectors.shape == (len(found), 2)
    ix.write(found[:-1], tmp_path)
    with pytest.raises(ValueError, match="does not match"):
        ix.Index.load(tmp_path)


def test_the_index_state_names_the_day_and_the_chunks(tmp_path):
    found = chunks()
    ix.write(found, tmp_path)
    snapshot = ix.write_state(found, tmp_path, date(2024, 3, 1))
    assert snapshot["as_of"] == "2024-03-01" and snapshot["chunks"] == len(found)
    assert (tmp_path / "state.json").is_file()


def test_sources_are_missing_when_the_index_is_not_visible_on_the_day(tmp_path):
    ix.write(chunks(), tmp_path)
    assert ix.sources_missing(tmp_path, ["S1/2023#1/1"], date(2023, 12, 1)) == []
    assert ix.sources_missing(tmp_path, ["S1/2023#1/1"], date(2023, 1, 1)) == ["S1/2023#1/1"]
    assert ix.sources_missing(tmp_path, ["S1/2023#9/9"], date(2024, 1, 1)) == ["S1/2023#9/9"]
