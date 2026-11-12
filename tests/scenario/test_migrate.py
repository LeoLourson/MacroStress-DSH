import json
from datetime import date
from types import SimpleNamespace

import numpy as np

from mst.scenario import __main__ as cli


def old(**extra):
    return {"narrative": "event", "sources": [], "macro": "v1", "satellites": "s1", **extra}


def put(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_migration_preflights_all_files_before_writing(tmp_path, capsys):
    folder = tmp_path / "scenarios"
    folder.mkdir()
    put(folder / "a.json", old())
    put(folder / "b.json", old(unexpected=7))
    before = (folder / "a.json").read_bytes()
    assert cli.main(["migrate", str(folder), "--models", str(tmp_path)]) != 0
    assert (folder / "a.json").read_bytes() == before
    assert "b.json" in capsys.readouterr().err


def test_migration_preserves_approval_and_lists_causes(tmp_path):
    folder = tmp_path / "scenarios"
    folder.mkdir()
    put(folder / "a.json", old(approved_by="human"))
    put(folder / "b.json", old(causes=["oil"], approved_by="human"))
    assert cli.main(["migrate", str(folder), "--workspace", str(tmp_path)]) == 0
    a = json.loads((folder / "a.json").read_text(encoding="utf-8"))
    b = json.loads((folder / "b.json").read_text(encoding="utf-8"))
    assert a["approved_by"] == "human"
    assert b["approved_by"] is None
    assert "oil" in b["narrative"]
    manifests = list(tmp_path.glob("*migration*"))
    assert len(manifests) == 1 and "b.json" in manifests[0].read_text(encoding="utf-8")
    saved = {p: p.read_bytes() for p in folder.glob("*.json")}
    assert cli.main(["migrate", str(folder), "--workspace", str(tmp_path)]) == 0
    assert {p: p.read_bytes() for p in saved} == saved


def test_migration_translates_numeric_statements(tmp_path, monkeypatch):
    from mst.scenario import migrate
    from mst.scenario.translate import QuantityData, TranslationData

    folder = tmp_path / "scenarios"
    folder.mkdir()
    put(
        folder / "a.json",
        old(
            approved_by="human",
            impacts=[
                {
                    "target": "ru.rate",
                    "start": 1,
                    "end": 2,
                    "delta": -2.5,
                    "reason": "policy",
                    "basis": "event",
                }
            ],
        ),
    )
    data = TranslationData(
        "2020-02", {"ru.rate": QuantityData("ru.rate", "identity")}, {"ru.rate": 10.0}, {}
    )
    monkeypatch.setattr(migrate, "context_of", lambda models, raw: data)
    assert cli.main(["migrate", str(folder), "--models", str(tmp_path)]) == 0
    result = json.loads((folder / "a.json").read_text(encoding="utf-8"))
    assert result["interventions"][0]["months"] == ["2020-03", "2020-04"]
    assert result["interventions"][0]["values"] == [-2.5, -2.5]
    assert result["approved_by"] == "human"


def test_migration_rolls_back_when_replacement_fails(tmp_path, monkeypatch, capsys):
    from mst.scenario import migrate

    folder = tmp_path / "scenarios"
    folder.mkdir()
    for name in ("a.json", "b.json"):
        put(folder / name, old())
    originals = {p: p.read_bytes() for p in folder.glob("*.json")}
    replace = migrate.os.replace
    calls = 0

    def fail_once(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated failure")
        return replace(source, target)

    monkeypatch.setattr(migrate.os, "replace", fail_once)
    assert cli.main(["migrate", str(folder), "--models", str(tmp_path)]) != 0
    assert {p: p.read_bytes() for p in originals} == originals
    assert "simulated failure" in capsys.readouterr().err


def test_resolver_uses_panel_origin_and_driver_history(tmp_path, monkeypatch):
    from mst import plugins
    from mst.drivers import bundle
    from mst.macro import registry, version
    from mst.scenario.migrate import context_of

    panel = SimpleNamespace(
        ids=["ru.rate"], months=[date(2020, 1, 1), date(2020, 2, 1)], y=np.array([[1.0], [2.0]])
    )
    monkeypatch.setattr(version, "panel_of", lambda root, name: (panel, date(2020, 5, 1)))
    monkeypatch.setattr(
        registry,
        "load_registry",
        lambda: SimpleNamespace(variables=[SimpleNamespace(id="ru.rate", transform="identity")]),
    )
    driver_registry = SimpleNamespace(quantity=lambda name: SimpleNamespace(transform="log100"))
    sat = SimpleNamespace(
        macro="v1",
        parts={"share_price": {"driver": ["share_price:abc"]}},
        arrays={"share_price.d": np.array([[4.0], [5.0]])},
        registry=driver_registry,
    )
    monkeypatch.setattr(plugins, "registry", lambda: SimpleNamespace(satellites={}))
    monkeypatch.setattr(bundle, "load", lambda root, name, models: sat)
    context = context_of(tmp_path, old(impacts=[{"target": "share_price:abc"}]))
    assert context.origin == "2020-02"
    assert context.last_model["share_price:abc"] == 5.0
    np.testing.assert_array_equal(context.history["share_price:abc"], [4.0, 5.0])
