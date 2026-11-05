"""Advisory findings about an expert answer, and the hard rule a zero size carries."""

import copy

import pytest

from mst.plugins.expert import answers as answers_module
from mst.plugins.expert import audit as audit_module
from tests.expert import fakes

KEY = "gl.oil"


def settings(tmp_path):
    return fakes.context(tmp_path, inputs=tmp_path / "elicitation")


def answer():
    return fakes.shock_payload(KEY)


def codes(found):
    return [row["code"] for row in found]


def test_a_cell_repeated_across_horizons_has_no_dynamics(tmp_path):
    payload = answer()
    payload["responses"]["ru.imoex"]["3"] = copy.deepcopy(payload["responses"]["ru.imoex"]["1"])
    found = audit_module.findings(settings(tmp_path), KEY, payload)
    same = [row for row in found if row["code"] == "same_cell"]
    assert same and "ru.imoex@1" in same[0]["cells"] and "ru.imoex@3" in same[0]["cells"]


def test_one_median_for_every_horizon_is_a_constant_response(tmp_path):
    payload = answer()
    for horizon in (1, 3, 6, 12):
        payload["responses"]["ru.retail"][str(horizon)]["q50"] = {
            "q5": -horizon, "q50": 2.0, "q95": horizon,
        }
    found = audit_module.findings(settings(tmp_path), KEY, payload)
    assert "flat_medians" in codes(found)


def test_a_cell_with_equal_quantiles_claims_certainty(tmp_path):
    payload = answer()
    payload["responses"]["ru.cpi"]["6"]["q50"] = {"q5": 1.0, "q50": 1.0, "q95": 1.0}
    found = audit_module.findings(settings(tmp_path), KEY, payload)
    assert "zero_width" in codes(found)


def test_a_first_month_far_above_history_is_flagged(tmp_path):
    payload = answer()
    payload["responses"]["ru.imoex"]["1"]["q50"] = {"q5": 900.0, "q50": 1000.0, "q95": 1100.0}
    found = audit_module.findings(settings(tmp_path), KEY, payload)
    assert "large_first" in codes(found)


def test_a_cell_equal_to_the_reference_is_flagged_as_copied(tmp_path):
    payload = answer()
    reference = payload["responses"]["gl.brent"]["1"]["q50"]
    payload["responses"]["ru.imoex"]["1"]["q50"] = copy.deepcopy(reference)
    found = audit_module.findings(settings(tmp_path), KEY, payload)
    copied = [row for row in found if row["code"] == "copied_reference"]
    assert "ru.imoex@1" in {cell for row in copied for cell in row["cells"]}


def test_two_shocks_with_one_transmission_are_flagged(tmp_path):
    from mst import inputs

    context = settings(tmp_path)
    payload = answer()
    inputs.write(context.inputs, context.as_of, "gl.oil_demand", payload)
    found = audit_module.findings(context, KEY, payload)
    assert "same_shock" in codes(found)


def test_a_broken_economic_rule_is_a_warning_with_its_note(tmp_path):
    payload = answer()
    payload["responses"]["ru.cnyrub"]["3"]["q50"] = {"q5": -9.0, "q50": -5.0, "q95": -1.0}
    payload["responses"]["ru.usdrub"]["3"]["q50"] = {"q5": 1.0, "q50": 5.0, "q95": 9.0}
    found = audit_module.findings(settings(tmp_path), KEY, payload)
    rules = [row for row in found if row["code"] == "rule"]
    assert "one_dollar_two_quotes" in {row["rule"] for row in rules}
    assert all(row["message"].startswith("one_dollar_two_quotes") for row in rules
               if row["rule"] == "one_dollar_two_quotes")


def test_directional_rules_reverse_with_a_negative_signed_size(tmp_path):
    payload = answer()
    payload["size"] = {"q5": -12.0, "q50": -5.0, "q95": -1.0}
    payload["responses"]["ru.usdrub"]["1"]["q50"] = {"q5": 1.0, "q50": 2.0, "q95": 3.0}
    found = audit_module.findings(settings(tmp_path), KEY, payload)
    assert "oil_supply_rouble" not in {row.get("rule") for row in found}


def test_the_zero_size_requires_a_zero_response(tmp_path):
    payload = fakes.shock_payload(KEY, one_sided=True)
    payload["responses"]["ru.imoex"]["1"]["q5"] = {"q5": -1.0, "q50": 0.0, "q95": 1.0}
    with pytest.raises(ValueError, match="size is zero"):
        answers_module.check(settings(tmp_path), KEY, payload)


def test_the_zero_size_rule_accepts_zero_responses(tmp_path):
    payload = fakes.shock_payload(KEY, one_sided=True)
    normal = answers_module.check(settings(tmp_path), KEY, payload)
    assert normal["responses"]["ru.imoex"]["1"]["q5"] == {
        "q5": 0.0, "q50": 0.0, "q95": 0.0,
    }
