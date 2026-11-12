"""The portfolio tool on the event contract: the chosen versions and the intervention warnings."""

from datetime import date
from types import SimpleNamespace

import numpy as np

from mst import store
from mst.portfolio import tool
from mst.scenario import tool as scenario_tool
from mst.scenario.spec import Spec

MACRO, SATELLITES = "3" * 64, "5" * 64


def position():
    return {"id": "sber", "type": "common_stock", "notional": 10.0, "codes": {"secid": "SBER"}}


def intervention(quantity, unit="level", **changes):
    return {
        "quantity": quantity,
        "months": ["2026-11"],
        "kind": "shift",
        "values": [1.0],
        "unit": unit,
        "reason": "Policy decision",
        "basis": "event",
    } | changes


def chosen(monkeypatch, *, selected=None):
    """The versions the portfolio's selection reads, recorded with every argument."""
    selected = {} if selected is None else selected
    macro = SimpleNamespace(
        name=MACRO, as_of=date(2026, 5, 1), panel=SimpleNamespace(ids=[], y=np.empty((0, 0)))
    )
    bundle = SimpleNamespace(macro=MACRO)

    def versions_of(models, *, macro=None, satellites=None):
        selected.update(models=models, macro=macro, satellites=satellites)
        return loaded_macro, bundle

    loaded_macro = macro
    monkeypatch.setattr(scenario_tool, "versions_of", versions_of)
    return macro, bundle


def saved(workspace):
    """One new-format scenario and one portfolio of a single rouble position."""
    scenario_tool.save(
        workspace.scenarios,
        "event",
        {"narrative": "The key rate is set", "approved_by": "И. Иванов"},
    )
    tool.save(workspace.portfolios, "demo", {"positions": [position()]})


def test_check_runs_on_the_selected_versions_and_the_requested_horizon(tmp_path, monkeypatch):
    ws = store.Workspace(tmp_path)
    saved(ws)
    selection = {}
    macro, satellites = chosen(monkeypatch, selected=selection)
    seen = {}
    monkeypatch.setattr(tool.scenario_build, "validate", lambda *args, **kwargs: {"problems": []})
    monkeypatch.setattr(tool, "_reach_warnings", lambda *args: [])

    def problems(item, options, loaded, bundle, catalog):
        seen.update(options=options, macro=loaded, satellites=bundle)
        return []

    monkeypatch.setattr(tool, "_position_problems", problems)

    report = tool.check(
        ws, "demo", "event", macro=MACRO, satellites=SATELLITES, horizon=6, inner=2
    )

    assert selection == {"models": ws.models, "macro": MACRO, "satellites": SATELLITES}
    assert seen["macro"] is macro and seen["satellites"] is satellites
    assert seen["options"].as_of == date(2026, 5, 1)
    assert seen["options"].horizon == 6
    assert seen["options"].base_currency == "RUB"
    assert report == {"id": "demo", "scenario": "event", "ok": True, "problems": [], "warnings": []}


def test_check_uses_the_twelve_month_horizon_by_default(tmp_path, monkeypatch):
    ws = store.Workspace(tmp_path)
    saved(ws)
    chosen(monkeypatch)
    seen = {}
    monkeypatch.setattr(tool.scenario_build, "validate", lambda *args, **kwargs: {"problems": []})
    monkeypatch.setattr(tool, "_reach_warnings", lambda *args: [])

    def problems(item, options, loaded, bundle, catalog):
        seen.update(options=options)
        return []

    monkeypatch.setattr(tool, "_position_problems", problems)

    tool.check(ws, "demo", "event")

    assert seen["options"].horizon == 12


def test_exposure_values_the_positions_at_the_selected_macro(tmp_path, monkeypatch):
    ws = store.Workspace(tmp_path)
    saved(ws)
    selection = {}
    macro, _ = chosen(monkeypatch, selected=selection)

    report = tool.exposure(ws, "demo", "event", macro=MACRO, satellites=SATELLITES)

    assert selection["macro"] == MACRO and selection["satellites"] == SATELLITES
    assert report["as_of"] == "2026-05-01"
    assert report["problems"] == []
    row = report["exposures"][0]
    assert (row["position"], row["notional"], row["fx"], row["value"]) == ("sber", 10.0, 1.0, 10.0)


def test_reach_warnings_read_the_macro_interventions_of_the_scenario(monkeypatch):
    spec = Spec.model_validate(
        {
            "narrative": "The key rate is set",
            "interventions": [
                intervention("ru.key_rate"),
                intervention("share_price:sber", unit="change"),
            ],
        }
    )
    requirement = SimpleNamespace(quantity="share_price", object=None, codes={})
    catalog = SimpleNamespace(
        model_for=lambda item: SimpleNamespace(
            requirements=lambda item, options: SimpleNamespace(drivers=[requirement])
        )
    )
    declared = SimpleNamespace(regressors=["ru.key_rate"], features=[])
    registry = SimpleNamespace(
        resolve=lambda quantity, object_=None, codes=None, among=None: "share_price:sber",
        quantity=lambda name: declared,
    )
    satellites = SimpleNamespace(registry=registry, served=lambda: {"share_price:sber"})
    options = SimpleNamespace()
    monkeypatch.setattr(
        tool,
        "_loading_warning",
        lambda item_id, driver, variable, bundle: f"{item_id} reads {variable} through {driver}",
    )

    warnings = tool._reach_warnings(
        SimpleNamespace(id="sber", codes={"secid": "SBER"}), options, satellites, spec, catalog
    )

    assert warnings == ["sber reads ru.key_rate through share_price:sber"]


def test_reach_warnings_say_nothing_when_only_drivers_are_intervened(monkeypatch):
    spec = Spec.model_validate(
        {"narrative": "Shares", "interventions": [intervention("share_price:sber", "change")]}
    )
    catalog = SimpleNamespace(model_for=lambda item: None)
    monkeypatch.setattr(
        tool,
        "_loading_warning",
        lambda *args: (_ for _ in ()).throw(AssertionError("no macro variable is intervened")),
    )

    warnings = tool._reach_warnings(
        SimpleNamespace(id="sber"), SimpleNamespace(), None, spec, catalog
    )

    assert warnings == []
