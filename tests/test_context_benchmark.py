"""Context policies keep the same questions without bringing back whole history."""

import pytest

from mst.bench import elicitation


def test_reset_policies_keep_four_separate_horizon_questions():
    for _h in (1, 3, 6, 12):
        assert elicitation.reset_context("bounded", "call", "gl.brent", "gl.brent")
        assert not elicitation.reset_context("history", "call", "gl.brent", "gl.brent")
        assert not elicitation.reset_context("per-variable", "call", "gl.brent", "gl.brent")
    assert elicitation.reset_context("per-variable", "call", "gl.fed_rate", "gl.brent")
    assert elicitation.reset_context("history", None, "gl.brent", None)


def test_benchmark_uses_exactly_one_current_tool_and_separate_horizon_questions():
    assert "single current question" in elicitation.SYSTEM
    assert "rate percentage points" in elicitation.SYSTEM
    assert elicitation.compact({"q5": -1, "q50": 0, "q95": 1}) == '{"q5":-1,"q50":0,"q95":1}'


def test_usage_counts_questions_errors_and_tokens_of_a_trace():
    calls = [
        {"question": "a", "is_error": True, "usage": {"prompt_tokens": 10}},
        {
            "question": "a",
            "is_error": False,
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 3,
                "prompt_tokens_details": {"cached_tokens": 8},
            },
        },
        {"question": "b", "is_error": False, "usage": {}},
    ]
    assert elicitation.usage(calls) == {
        "response_questions": 2,
        "accepted_questions": 2,
        "calls": 3,
        "errors": 1,
        "prompt_tokens": 22,
        "completion_tokens": 3,
        "cached_tokens": 8,
    }


def test_describe_reads_medians_and_names_collapsed_quantiles():
    flat = {"q5": 1.0, "q50": 1.0, "q95": 1.0}
    wide = {"q5": 0.0, "q50": 2.0, "q95": 5.0}
    parts = {
        "law": {"size": {"q5": -10.0, "q50": 20.0, "q95": 50.0}},
        "response:gl.brent:1": {"at_q5": wide, "at_q50": flat, "at_q95": wide},
    }
    found = elicitation.describe(parts)
    assert found["brent_median_log100"] == {"1": [2.0, 1.0, 2.0]}
    assert found["uncertainty_collapsed_cells"] == ["response:gl.brent:1/at_q50"]
    assert found["own_impact_reference_if_size_is_causal_contribution"][1] == pytest.approx(
        100 * 0.1823215567939546
    )
