"""The elicitation answers of the workspace: atomic writes, reads and the status."""

from datetime import date

from mst import inputs, plugins
from tests.plugins import fakes

DAY = date(2026, 11, 1)


def test_a_write_then_a_read_returns_the_normal_form(tmp_path):
    first = inputs.write(tmp_path, DAY, "q1", {"value": 1.0})
    assert inputs.read(tmp_path, DAY) == {"q1": {"value": 1.0}}
    assert inputs.path(tmp_path, DAY, "q1") == tmp_path / "2026-11-01" / "q1.json"

    second = inputs.write(tmp_path, DAY, "q1", {"value": 2.0})
    assert inputs.read(tmp_path, DAY) == {"q1": {"value": 2.0}}
    assert first != second


def test_the_same_answer_written_again_keeps_its_hash(tmp_path):
    once = inputs.write(tmp_path, DAY, "q1", {"value": 1.0})
    twice = inputs.write(tmp_path, DAY, "q1", {"value": 1.0})
    assert once == twice


def test_an_interrupted_write_is_no_answer(tmp_path):
    folder = tmp_path / DAY.isoformat()
    folder.mkdir(parents=True)
    (folder / ".draft-broken.json").write_text("{}", encoding="utf-8")

    assert inputs.read(tmp_path, DAY) == {}
    model = fakes.InputMacroModel()
    assert [row["key"] for row in inputs.status(tmp_path, model, DAY)["keys"]] == ["q1", "q2"]


def test_status_lists_exactly_the_keys_missing_first(tmp_path):
    model = fakes.InputMacroModel()
    inputs.write(tmp_path, DAY, "q2", {"value": 5.0})

    rows = inputs.status(tmp_path, model, DAY)["keys"]

    assert [row["key"] for row in rows] == ["q1", "q2"]
    assert rows[0]["written"] is False and rows[0]["hash"] is None
    assert rows[1]["written"] is True and rows[1]["hash"] is not None
    assert rows[1]["written_at"] is not None
    assert inputs.missing(tmp_path, model, DAY) == ["q1"]


def test_the_answers_of_one_date_do_not_leak_into_another(tmp_path):
    inputs.write(tmp_path, DAY, "q1", {"value": 1.0})
    assert inputs.read(tmp_path, date(2026, 12, 1)) == {}


def test_the_fake_model_is_registered_through_the_plugin(monkeypatch):
    registry = plugins.build([fakes.model_plugin(macro=(fakes.InputMacroModel(),))])
    monkeypatch.setattr(plugins, "registry", lambda: registry)
    assert "gmodel.inputs.v1" in registry.macro
