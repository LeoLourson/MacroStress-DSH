"""Fresh agent contexts retain only bounded, economically relevant saved judgements."""

import pytest

from mst.plugins.expert.model import ExpertModel
from tests.expert import fakes
from tests.expert.test_steps import ctx as shared_ctx
from tests.expert.test_steps import split_answer


@pytest.fixture
def ctx():
    return shared_ctx.__wrapped__()


def test_response_context_keeps_law_and_earlier_horizons_of_same_variable(ctx):
    model = ExpertModel('rw')
    parts = split_answer(fakes.shock_payload('gl.oil'))
    # Pretend all other variables were already answered; their history must not leak in.
    context = model.input_step_context(ctx, 'gl.oil', 'response:gl.fed_rate:6', parts)
    assert context['shock_law'] == parts['law']
    assert context['earlier_horizons'] == {
        '1': parts['response:gl.fed_rate:1'], '3': parts['response:gl.fed_rate:3']
    }
    assert '6' not in context['earlier_horizons'] and '12' not in context['earlier_horizons']
    assert len(context['reference_response']) == 1
    assert 'response:gl.brent:6' in context['reference_response']


def test_law_has_no_previous_responses_and_context_is_bounded(ctx):
    model = ExpertModel('rw')
    parts = split_answer(fakes.shock_payload('gl.oil'))
    assert model.input_step_context(ctx, 'gl.oil', 'law', parts) == {}
    context = model.input_step_context(ctx, 'gl.oil', 'response:ru.cpi:12', parts)
    assert len(context['earlier_horizons']) == 3
    assert len(context['reference_response']) <= 1


def test_zero_size_and_unit_rules_are_explicit_without_reducing_questions(ctx):
    model = ExpertModel('rw')
    oil = model.input_steps(ctx, 'gl.oil')
    assert len(oil) == 93
    assert '100*ln(1+s/100)' in oil[1]['text']
    assert 'zero' in oil[1]['text'] and 'all three' in oil[1]['text']
    monetary = model.input_steps(ctx, 'ru.monetary')
    assert '1 percentage point = 100 basis points' in monetary[1]['text']
