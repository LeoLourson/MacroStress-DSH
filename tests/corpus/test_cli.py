import json

from mst.corpus import cli
from mst.corpus import index as ix

DOC = """---
published: 2023-11-10
---

# ОДКП 2024

## Сценарии

Рисковый сценарий: глобальный кризис и падение цен на нефть.
"""

QUESTIONS = """## Q1. Как падение нефти входит в рисковый сценарий?
- as_of: 2024-01-01
- expected: S1/2023#1/1
"""


def test_eval_reads_the_questions_it_is_given_outside_the_corpus(tmp_path, monkeypatch, capsys):
    chunks = ix.chunks_of(
        DOC,
        doc_id="S1",
        edition="2023",
        kind="scenario_set",
        region="ru",
        jurisdiction="ru",
        visible_from="2023-11-10",
        factors=["rate.policy"],
    )
    monkeypatch.setattr(cli.Index, "load", classmethod(lambda cls, root: cls(chunks)))
    questions = tmp_path / "bench" / "questions.md"
    questions.parent.mkdir()
    questions.write_text(QUESTIONS, encoding="utf-8")

    code = cli.main(["eval", "--corpus", str(tmp_path / "corpus"), "--questions", str(questions)])

    assert code == 0
    assert capsys.readouterr().out.splitlines() == [
        "questions: 1, marked up: 1",
        "recall@5: 1.000",
    ]


def test_eval_saves_the_recall_beside_the_questions(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli.Index, "load", classmethod(lambda cls, root: cls([])))
    questions = tmp_path / "bench" / "questions.md"
    questions.parent.mkdir()
    questions.write_text(QUESTIONS, encoding="utf-8")

    cli.main(
        ["eval", "--corpus", str(tmp_path), "--questions", str(questions), "--save"]
    )

    saved = tmp_path / "bench" / "results"
    report = json.loads((saved / "recall.json").read_text(encoding="utf-8"))
    assert report == {"questions": 1, "marked_up": 1, "recall_at_5": 0.0, "unmarked": []}
    assert (saved / "recall.md").read_text(encoding="utf-8") == capsys.readouterr().out
