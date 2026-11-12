"""The scenario tool on the event contract: what it lists and the versions validate selects."""

import json
import os
import shutil
from datetime import date
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from mst import plugins, store
from mst.data.store import shift
from mst.scenario import tool


def intervention(**changes):
    return {
        "quantity": "ru.key_rate",
        "months": ["2026-11"],
        "kind": "set",
        "values": [11.5],
        "unit": "level",
        "reason": "Policy decision",
        "basis": "event",
    } | changes


def observation():
    return {
        "kind": "constraint",
        "terms": [{"quantity": "ru.key_rate", "month": "2026-11", "weight": 1.0}],
        "value": 11.5,
        "unit": "level",
        "basis": "event",
    }


def spec(**changes):
    return {
        "narrative": "The key rate is set",
        "sources": ["S1#1/1"],
        "interventions": [intervention()],
        "approved_by": "И. Иванов",
    } | changes


def old(**changes):
    return {
        "narrative": "An old scenario",
        "macro": "a" * 64,
        "satellites": "b" * 64,
        "horizon": 6,
        "approved_by": "И. Иванов",
    } | changes


OLDER, NEWER = "2" * 64, "3" * 64
SATELLITES, OTHER = "5" * 64, "6" * 64


def version_folder(directory, name, *, mtime, **meta):
    """A finished version folder: the meta is all ``versions.folders`` reads to find it."""
    folder = directory / name
    folder.mkdir(parents=True)
    (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    os.utime(folder, (mtime, mtime))
    return folder


def fake_models(tmp_path):
    """Two macro versions and two bundles; the newest bundle belongs to the older macro."""
    models = tmp_path / "models"
    version_folder(models / "macro", OLDER, mtime=1_000, as_of="2026-01-01")
    version_folder(models / "macro", NEWER, mtime=2_000, as_of="2026-05-01")
    version_folder(models / "satellites", SATELLITES, mtime=2_500, macro=NEWER)
    version_folder(models / "satellites", OTHER, mtime=3_000, macro=OLDER)
    return models


def no_installed(monkeypatch):
    """The selection reads names only: the installed plugins are not part of it."""
    monkeypatch.setattr(plugins, "registry", lambda: SimpleNamespace(macro={}, satellites={}))


def fake_loaders(monkeypatch, *, bundle_macro=NEWER):
    """The two model loaders of the selection, recording the names they receive."""
    calls = []
    macro = SimpleNamespace(as_of=date(2026, 5, 1))
    bundle = SimpleNamespace(macro=bundle_macro)

    def load_macro(root, name, installed):
        calls.append(("macro", name))
        macro.name = name
        return macro

    def load_satellites(root, name, installed):
        calls.append(("satellites", name))
        return bundle

    monkeypatch.setattr(tool.macro_version, "load", load_macro)
    monkeypatch.setattr(tool.satellite_bundle, "load", load_satellites)
    return calls, macro, bundle


def fake_check(monkeypatch):
    """The builder's check of a scenario is the other agent's layer: record what it receives."""
    seen = {}

    def check(spec, macro, satellites, *, as_of, corpus=None):
        seen.update(macro=macro, satellites=satellites, as_of=as_of, corpus=corpus)
        return {"ok": True, "problems": [], "labels": {}, "warnings": []}

    monkeypatch.setattr(tool, "check", check)
    return seen


def test_catalogue_shows_the_counts_and_the_approval_of_the_event_contract(tmp_path):
    folder = tmp_path / "scenarios"
    tool.save(folder, "event", spec(observations=[observation()]))
    listed = tool.catalogue(folder)
    assert listed["broken"] == []
    assert listed["scenarios"] == [
        {
            "id": "event",
            "narrative": "The key rate is set",
            "sources": 1,
            "observations": 1,
            "interventions": 1,
            "approved_by": "И. Иванов",
        }
    ]


def test_catalogue_names_an_old_file_as_broken_and_save_refuses_it(tmp_path):
    folder = tmp_path / "scenarios"
    folder.mkdir()
    (folder / "old.json").write_text(json.dumps(old()), encoding="utf-8")
    (folder / "garbage.json").write_text("{", encoding="utf-8")
    listed = tool.catalogue(folder)
    assert listed["scenarios"] == []
    broken = {row["id"]: row["error"] for row in listed["broken"]}
    assert sorted(broken) == ["garbage", "old"]
    assert "macro" in broken["old"]
    with pytest.raises(ValidationError, match="macro"):
        tool.save(folder, "copy", old())
    with pytest.raises(ValidationError):
        tool.get(folder, "old")


def test_validate_selects_the_newest_finished_macro_and_its_satellites(tmp_path, monkeypatch):
    models = fake_models(tmp_path)
    no_installed(monkeypatch)
    calls, macro, bundle = fake_loaders(monkeypatch)
    seen = fake_check(monkeypatch)
    folder = tmp_path / "scenarios"
    tool.save(folder, "event", spec())

    report = tool.validate(folder, "event", models)

    assert calls == [("macro", NEWER), ("satellites", SATELLITES)]
    assert seen["macro"] is macro and seen["satellites"] is bundle
    assert seen["as_of"] == date(2026, 5, 1) and seen["corpus"] is None
    assert report["id"] == "event" and report["ok"] is True


def test_validate_honours_an_explicit_macro_and_satellites(tmp_path, monkeypatch):
    models = fake_models(tmp_path)
    no_installed(monkeypatch)
    calls, _, bundle = fake_loaders(monkeypatch, bundle_macro=OLDER)
    fake_check(monkeypatch)
    folder = tmp_path / "scenarios"
    tool.save(folder, "event", spec())

    report = tool.validate(folder, "event", models, macro=OLDER, satellites=OTHER)

    assert calls == [("macro", OLDER), ("satellites", OTHER)]
    assert report["id"] == "event" and report["ok"] is True


def test_validate_refuses_satellites_estimated_on_another_macro(tmp_path, monkeypatch):
    models = fake_models(tmp_path)
    no_installed(monkeypatch)
    fake_loaders(monkeypatch, bundle_macro=OLDER)
    fake_check(monkeypatch)
    folder = tmp_path / "scenarios"
    tool.save(folder, "event", spec())

    with pytest.raises(ValueError, match="estimated on macro version"):
        tool.validate(folder, "event", models, macro=NEWER, satellites=OTHER)


def test_validate_names_the_missing_macro_version(tmp_path, monkeypatch):
    no_installed(monkeypatch)
    fake_check(monkeypatch)
    folder = tmp_path / "scenarios"
    tool.save(folder, "event", spec())

    with pytest.raises(ValueError, match="no finished macro version"):
        tool.validate(folder, "event", tmp_path / "models")


def test_validate_names_the_macro_that_has_no_satellites(tmp_path, monkeypatch):
    models = tmp_path / "models"
    version_folder(models / "macro", NEWER, mtime=2_000, as_of="2026-05-01")
    version_folder(models / "satellites", OTHER, mtime=3_000, macro=OLDER)
    no_installed(monkeypatch)
    fake_loaders(monkeypatch)
    fake_check(monkeypatch)
    folder = tmp_path / "scenarios"
    tool.save(folder, "event", spec())

    with pytest.raises(ValueError, match=f"no satellite bundle.*{NEWER}"):
        tool.validate(folder, "event", models)


def test_validate_computes_against_the_installed_pair(tmp_path, versions):
    ws = store.Workspace(tmp_path)
    for kind, name in (("macro", versions.macro), ("satellites", versions.satellites)):
        shutil.copytree(versions.models / kind / name, ws.models / kind / name)
    month = shift(versions.loaded.panel.months[-1], 1).strftime("%Y-%m")
    tool.save(ws.scenarios, "event", spec(interventions=[intervention(months=[month])]))

    report = tool.validate(ws.scenarios, "event", ws.models)

    assert report["id"] == "event"
    assert report["ok"] is True and report["problems"] == []
