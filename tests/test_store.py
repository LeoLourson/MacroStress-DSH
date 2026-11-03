"""The state summary the prompt section and the jobs tool both read."""

import json
import os

from mst import store


def test_hash_ignores_key_order_and_unicode_escapes():
    first = {"b": [1, 2], "a": "цена"}
    second = {"a": "цена", "b": [1, 2]}
    assert store.sha256_of(first) == store.sha256_of(second)
    assert store.sha256_of({"a": "цена"}) == store.sha256_of({"a": "цена"})


def test_hash_changes_with_a_value():
    assert store.sha256_of({"a": 1}) != store.sha256_of({"a": 2})


def test_written_json_reads_back_and_leaves_no_draft(tmp_path):
    path = tmp_path / "nested" / "state.json"
    store.write_json(path, {"b": 1, "a": "цена"})
    assert store.read_json(path) == {"a": "цена", "b": 1}
    assert sorted(entry.name for entry in path.parent.iterdir()) == ["state.json"]
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": "цена", "b": 1}


def test_a_replace_retries_when_a_reader_holds_the_target(monkeypatch, tmp_path):
    # On Windows os.replace fails with PermissionError while another process has the
    # target open — the DSH prompt section reads workspace.json that way. The retries
    # must carry the write past a short holder.
    real = os.replace
    left = {"count": 2}

    def flaky(source, target):
        if left["count"] > 0:
            left["count"] -= 1
            raise PermissionError("the file is open in another process")
        real(source, target)

    monkeypatch.setattr(os, "replace", flaky)
    store.write_json(tmp_path / "state.json", {"a": 1})
    assert store.read_json(tmp_path / "state.json") == {"a": 1}
    assert left["count"] == 0


def test_a_replace_gives_up_and_leaves_no_draft_when_the_holder_never_lets_go(
    monkeypatch, tmp_path
):
    def stuck(source, target):
        raise PermissionError("the file is open in another process")

    monkeypatch.setattr(os, "replace", stuck)
    try:
        store.write_json(tmp_path / "state.json", {"a": 1})
    except PermissionError:
        pass
    else:
        raise AssertionError("a permanent holder must still get the error")
    assert list(tmp_path.iterdir()) == []


def test_state_counts_entries_and_names_the_newest_version(tmp_path):
    ws = store.Workspace(tmp_path)
    assert store.state(ws)["portfolios"] == 0
    assert store.state(ws)["macro"] is None

    (ws.portfolios / "demo").mkdir(parents=True)
    ws.scenarios.mkdir(parents=True)
    (ws.scenarios / "a.json").write_text("{}", encoding="utf-8")
    (ws.scenarios / "a.md").write_text("", encoding="utf-8")
    older = ws.macro_models / "aaa"
    newer = ws.macro_models / "bbb"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    (newer / "meta.json").write_text("{}", encoding="utf-8")
    draft = ws.macro_models / ".draft-xyz"
    draft.mkdir()
    (draft / "meta.json").write_text("{}", encoding="utf-8")
    # A version being written is newer than any finished one.
    later = newer.stat().st_mtime + 60
    os.utime(draft, (later, later))

    snapshot = store.state(ws)
    assert snapshot["portfolios"] == 1
    assert snapshot["scenarios"] == 1
    assert snapshot["macro"] == "bbb"
    assert snapshot["workspace"] == str(tmp_path.resolve())


def test_write_state_puts_the_same_dict_on_disk(tmp_path):
    ws = store.Workspace(tmp_path / "workspace")
    written = store.write_state(ws)
    assert store.read_json(ws.state_file) == written
