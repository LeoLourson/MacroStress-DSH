"""mst.portfolio.tool: the files of the portfolios, their check and their exposure."""

import numpy as np
import pytest
from pydantic import ValidationError

from mst.macro import version as macro_version
from mst.portfolio import tool
from mst.scenario import tool as scenario_tool


def test_the_tool_writes_lists_reads_and_deletes_a_portfolio(workspace, positions):
    saved = tool.save(workspace.portfolios, "demo", {"name": "Демо", "positions": positions})
    assert saved["portfolio"]["positions"][0]["id"] == positions[0]["id"]
    listed = tool.catalogue(workspace.portfolios)
    assert [item["id"] for item in listed["portfolios"]] == ["demo"]
    assert listed["broken"] == []
    assert tool.get(workspace.portfolios, "demo")["portfolio"]["name"] == "Демо"
    assert tool.delete(workspace.portfolios, "demo") == {"id": "demo", "deleted": True}
    with pytest.raises(FileNotFoundError):
        tool.get(workspace.portfolios, "demo")


def test_a_broken_file_does_not_hide_the_others(workspace, positions):
    tool.save(workspace.portfolios, "good", {"positions": positions})
    (workspace.portfolios / "bad.json").write_text("{", encoding="utf-8")
    listed = tool.catalogue(workspace.portfolios)
    assert [item["id"] for item in listed["portfolios"]] == ["good"]
    assert [item["id"] for item in listed["broken"]] == ["bad"]


def test_a_bad_id_and_an_absent_file_are_refused(workspace):
    with pytest.raises(ValueError, match="not a portfolio id"):
        tool.get(workspace.portfolios, "../../etc")
    with pytest.raises(FileNotFoundError):
        tool.delete(workspace.portfolios, "nowhere")


def test_save_refuses_two_positions_with_one_id(workspace, positions):
    with pytest.raises(ValidationError, match="share one id"):
        tool.save(workspace.portfolios, "demo", {"positions": [positions[0], positions[0]]})


def test_check_accepts_a_portfolio_the_versions_serve(workspace, portfolio):
    report = tool.check(workspace, portfolio, "oil")
    assert report == {"id": "demo", "scenario": "oil", "ok": True, "problems": []}


def test_check_names_an_unknown_instrument_type(workspace, positions):
    positions[0]["type"] = "option"
    tool.save(workspace.portfolios, "demo", {"positions": positions})
    report = tool.check(workspace, "demo", "oil")
    assert report["ok"] is False
    assert "option" in report["problems"][0]


def test_check_names_terms_the_model_cannot_read(workspace, positions):
    positions[1]["terms"] = {}
    tool.save(workspace.portfolios, "demo", {"positions": positions})
    report = tool.check(workspace, "demo", "oil")
    assert report["ok"] is False
    assert "terms /duration" in report["problems"][0]


def test_check_names_a_code_no_object_covers(workspace, positions):
    positions[0]["codes"] = {"secid": "NOWHERE"}
    tool.save(workspace.portfolios, "demo", {"positions": positions})
    report = tool.check(workspace, "demo", "oil")
    assert report["ok"] is False
    assert "no object of share_price covers secid NOWHERE" in report["problems"][0]


def test_check_names_a_driver_the_check_rejected_with_its_reason(workspace, positions):
    positions[1]["codes"] = {"secid": "SBER"}
    tool.save(workspace.portfolios, "demo", {"positions": positions})
    report = tool.check(workspace, "demo", "oil")
    assert report["ok"] is False
    assert "bond_spread:sber is not served" in report["problems"][0]
    assert "no checked driver above it" in report["problems"][0]


def test_exposure_values_the_positions_at_the_calculation_month(workspace, portfolio):
    report = tool.exposure(workspace, portfolio, "oil")
    assert report["base_currency"] == "RUB" and report["problems"] == []
    assert [row["position"] for row in report["exposures"]] == ["sber", "corp", "mining", "flat"]
    for row in report["exposures"]:
        assert row["fx"] == 1.0
        assert row["value"] == pytest.approx(row["notional"])


def test_exposure_converts_a_foreign_position_at_the_panel_rate(workspace, positions, versions):
    positions[0]["currency"] = "USD"
    tool.save(workspace.portfolios, "demo", {"positions": positions})
    report = tool.exposure(workspace, "demo", "oil")
    row = report["exposures"][0]
    panel, _ = macro_version.panel_of(workspace.models, versions.macro)
    rate = float(np.exp(panel.y[-1, panel.ids.index("ru.usdrub")] / 100))
    assert row["fx"] == pytest.approx(rate)
    assert row["value"] == pytest.approx(row["notional"] * rate)


def test_exposure_names_a_currency_with_no_macro_path(workspace, positions):
    positions[0]["currency"] = "GBP"
    tool.save(workspace.portfolios, "demo", {"positions": positions})
    report = tool.exposure(workspace, "demo", "oil")
    assert "no macro path from GBP to RUB" in report["problems"][0]
    assert len(report["exposures"]) == len(positions) - 1


def test_check_names_a_currency_the_models_cannot_price(workspace, positions):
    positions[0]["currency"] = "GBP"
    tool.save(workspace.portfolios, "demo", {"positions": positions})
    report = tool.check(workspace, "demo", "oil")
    assert report["ok"] is False
    assert "no macro path from GBP to RUB" in report["problems"][0]


def test_check_names_an_infeasible_scenario(workspace, portfolio, versions):
    target = float(versions.data.y[-1, 0])
    fixed = {
        "terms": [{"variable": "gl.brent", "month": 1, "weight": 1.0}],
        "basis": "assumption",
    }
    scenario_tool.save(
        workspace.scenarios,
        "bad",
        {
            "narrative": "дважды",
            "horizon": 3,
            "macro": versions.macro,
            "satellites": versions.satellites,
            "approved_by": "И. Иванов",
            "constraints": [
                {**fixed, "value": target},
                {**fixed, "value": target + 1.0},
            ],
        },
    )
    report = tool.check(workspace, portfolio, "bad")
    assert report["ok"] is False
    # Two disjoint bands on one functional: the candidate weights read an empty event, and
    # the run refuses the scenario by lemma 2 — the driving-shock check would name the
    # second constraint the same way.
    assert any("contradicts the models" in problem for problem in report["problems"])
