"""The question text: the size in the response's scale, and the zero-size nodes."""

import pytest

from mst.plugins.expert import steps
from mst.plugins.expert.model import load_shocks
from tests.expert import fakes

LAW = {
    "frequency": 1.0,
    "size": {"q5": -12.0, "q50": 5.0, "q95": 20.0},
    "rho": 0.5,
    "nonlinearity": "the pass-through saturates at the upper node",
    "sources": [],
}


def context(tmp_path):
    return fakes.context(tmp_path)


def variable(context, name):
    return next(row for row in context.registry.variables if row.id == name)


def test_a_percent_size_is_converted_to_the_variables_log100(tmp_path):
    ctx = context(tmp_path)
    part = steps.step_context(ctx, "gl.oil", "response:ru.imoex:3", {"law": LAW})
    converted = part["size_in_response_scale"]
    assert converted["q5"] == {"percent": -12.0, "log100": pytest.approx(-12.783, abs=0.001)}
    assert converted["q95"]["log100"] == pytest.approx(18.232, abs=0.001)


def test_a_percentage_point_size_is_not_converted(tmp_path):
    ctx = context(tmp_path)
    part = steps.step_context(ctx, "ru.monetary", "response:ru.imoex:3", {"law": LAW})
    assert part["size_in_response_scale"] == {}


def test_a_zero_size_node_is_named(tmp_path):
    ctx = context(tmp_path)
    law = {**LAW, "size": {"q5": 0.0, "q50": 2.0, "q95": 5.0}}
    part = steps.step_context(ctx, "gl.oil", "response:ru.imoex:3", {"law": law})
    assert part["zero_size_nodes"] == ["q5"]


def test_the_response_question_states_the_conversion(tmp_path):
    ctx = context(tmp_path)
    shock = next(row for row in load_shocks() if row.id == "gl.oil")
    text = steps._response_question(ctx, shock, variable(ctx, "ru.imoex"), 3)
    assert "size_in_response_scale" in text and "-12.78" in text

    points = next(row for row in load_shocks() if row.id == "ru.monetary")
    other = steps._response_question(ctx, points, variable(ctx, "ru.imoex"), 3)
    assert "size_in_response_scale" not in other


@pytest.mark.parametrize("size", [-100.0, -120.0])
def test_impossible_percent_change_is_refused_before_next_step(tmp_path, size):
    ctx = context(tmp_path)
    law = {**LAW, "size": {**LAW["size"], "q5": size}}
    with pytest.raises(ValueError, match="greater than -100"):
        steps.check_part(ctx, "gl.oil", "law", law)


def test_the_every_shock_declares_how_to_read_its_size():
    scales = {row.id: row.size_scale for row in load_shocks()}
    assert len(scales) == 18
    assert scales["gl.oil"] == "percent_change" and scales["ru.monetary"] == "percentage_points"
    assert scales["gl.financial"] == "index_points" and scales["ru.fiscal"] == "percent_of_stock"
    assert all(scales.values())
