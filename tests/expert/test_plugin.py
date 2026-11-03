"""The declaration of the expert plugin: two models, the variants of one law."""

from datetime import date

from mst import models, plugins
from mst.plugins.expert.model import MARKET, RESIDUAL, ExpertModel, load_shocks

DAY = date(2026, 11, 1)


def test_the_plugin_check_passes():
    from mst.plugin import check

    assert check("expert") == []


def test_the_registry_lists_both_variants():
    macro = plugins.registry().macro
    assert "expert.elicited.rw" in macro and "expert.elicited.market" in macro
    assert isinstance(macro["expert.elicited.rw"], ExpertModel)
    assert models.selected("expert.elicited.market").model_id == "expert.elicited.market"


def test_the_models_differ_by_the_market_answer_only():
    rw = models.selected("expert.elicited.rw")
    market = models.selected("expert.elicited.market")
    keys = rw.input_keys(DAY)
    assert len(keys) == 18 and len(set(keys)) == 18
    assert market.input_keys(DAY) == (*keys, MARKET)


def test_the_shocks_are_eighteen_and_carry_their_units():
    shocks = load_shocks()
    assert len(shocks) == 18
    assert all(shock.meaning and shock.size_unit for shock in shocks)
    assert {shock.id for shock in shocks} >= {
        "gl.oil", "gl.financial", "ru.monetary", "ru.inflation", "ru.fx_risk"
    }, "the names the other macro models know must fit this model unchanged"
    assert RESIDUAL not in {shock.id for shock in shocks}
