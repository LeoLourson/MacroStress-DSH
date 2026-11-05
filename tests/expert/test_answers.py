"""The answer contract of the elicitation: what passes, what is refused and with which field."""

from datetime import date

import pytest

from mst.data.store import Observation, SeriesStore
from mst.plugins.expert.answers import check, linear, question
from mst.plugins.expert.model import MARKET
from tests.expert import fakes


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    return fakes.context(tmp_path_factory.mktemp("series"), fragment=fakes.visible)


def _refused(context, key, payload, field):
    with pytest.raises(ValueError, match=field):
        check(context, key, payload)


def test_the_fixture_of_eighteen_answers_passes(ctx):
    for key, payload in fakes.answers().items():
        normal = check(ctx, key, payload)
        assert normal["shock"] == key
    assert check(ctx, MARKET, fakes.market_payload())["values"]["gl.brent"]["fragment"] == "M1"


def test_a_crossed_quantile_is_refused_with_the_field(ctx):
    payload = fakes.shock_payload("gl.oil")
    payload["size"]["q5"] = 6.0
    _refused(ctx, "gl.oil", payload, "size")


def test_a_missing_variable_is_refused_with_the_field(ctx):
    payload = fakes.shock_payload("gl.oil")
    del payload["responses"]["ru.cpi"]
    _refused(ctx, "gl.oil", payload, "ru.cpi")


def test_a_horizon_beyond_the_nodes_is_refused_with_the_field(ctx):
    payload = fakes.shock_payload("gl.oil")
    payload["responses"]["ru.cpi"]["2"] = payload["responses"]["ru.cpi"]["3"]
    _refused(ctx, "gl.oil", payload, "horizon")


def test_a_rho_above_one_is_refused_with_the_field(ctx):
    payload = fakes.shock_payload("gl.oil", rho=1.5)
    _refused(ctx, "gl.oil", payload, "rho")


def test_a_negative_frequency_is_refused_with_the_field(ctx):
    payload = fakes.shock_payload("gl.oil", frequency=-0.5)
    _refused(ctx, "gl.oil", payload, "frequency")


def test_a_proportional_answer_is_refused(ctx):
    payload = fakes.shock_payload("gl.oil")
    sizes = payload["size"]
    for variable in payload["responses"]:
        for horizon in payload["responses"][variable]:
            for name in ("q5", "q50", "q95"):
                share = sizes[name] / sizes["q95"]
                payload["responses"][variable][horizon][name] = {
                    part: 0.1 * share for part in ("q5", "q50", "q95")
                }
    assert linear(_parse(payload))
    _refused(ctx, "gl.oil", payload, "proportional")


def _parse(payload):
    from mst.plugins.expert.answers import ShockAnswer

    return ShockAnswer.model_validate(payload)


def test_a_market_fragment_that_is_not_visible_is_refused(ctx):
    payload = fakes.market_payload()
    payload["values"]["gl.brent"]["fragment"] = "M9"
    _refused(ctx, MARKET, payload, "fragment")


def test_a_context_without_fragments_refuses_answers_that_cite_them(tmp_path):
    bare = fakes.context(tmp_path)
    payload = fakes.shock_payload("gl.oil", sources=["M1"])
    _refused(bare, "gl.oil", payload, "not indexed")
    _refused(bare, MARKET, fakes.market_payload(), "not indexed")
    # ... while the same answers with no fragments pass.
    assert check(bare, "gl.oil", fakes.shock_payload("gl.oil"))["sources"] == []


def test_an_unknown_key_is_refused(ctx):
    with pytest.raises(KeyError):
        check(ctx, "nope", {})


def test_the_question_names_the_unit_the_values_and_nothing_later(tmp_path):
    ctx = fakes.context(tmp_path, fragment=fakes.visible)
    asked = question(ctx, "gl.oil")
    assert asked["key"] == "gl.oil"
    assert "percent change of the Brent price" in asked["text"]
    assert "OPEC+" in asked["text"]
    assert "rho" in asked["text"] and "nonlinearity" in asked["text"]
    assert asked["schema"]["title"] == "ShockAnswer"
    for variable in ctx.registry.variables:
        assert variable.id in asked["text"], "every variable is named in the question"
    last = ctx.store.as_of("gl.brent", fakes.AS_OF)[-1]
    assert f"{last.value:g}" in asked["text"] and last.month.isoformat() in asked["text"]

    # A revision published after the date and a value of a later month stay out.
    rows = list(ctx.store.read("gl.brent"))
    SeriesStore(ctx.store.root).write(
        "gl.brent",
        rows
        + [
            Observation(date(2026, 10, 1), 777.0, date(2026, 11, 20), False),
            Observation(date(2026, 11, 1), 80.0, date(2026, 11, 20), False),
        ],
    )
    asked = question(ctx, "gl.oil")
    row = next(line for line in asked["text"].splitlines() if line.startswith("| gl.brent"))
    assert f"{last.value:g}" in row and "2026-10-01" in row
    assert "777" not in row and "80" not in row


def test_the_question_states_the_shape_of_the_responses(ctx):
    text = question(ctx, "gl.oil")["text"]
    for horizon in ("1", "3", "6", "12"):
        assert f'"{horizon}"' in text, "the horizons are named as the keys of responses"
    for node in ("q5", "q50", "q95"):
        assert f'"{node}"' in text, "the size nodes are named as the keys of a horizon"
    assert "quantiles object" in text, "one cell is described in words"
    assert "proportional" in text, "the linear answer is named before it is refused"


def test_the_market_question_names_the_corpus_the_fragments_come_from(ctx):
    text = question(ctx, MARKET)["text"]
    assert "corpus-model" in text, "the market answer names the corpus it needs"
    assert 'corpus(action="index", corpus="model")' in text


def test_the_market_question_names_the_fragment_rule(ctx):
    asked = question(ctx, MARKET)
    assert asked["key"] == MARKET
    assert "fragment" in asked["text"] and "model scale" in asked["text"]
    assert asked["schema"]["title"] == "MarketAnswer"
