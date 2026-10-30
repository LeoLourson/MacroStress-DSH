"""The sign table of the BVAR: the named shocks and what they move."""

import pytest
from pydantic import ValidationError

from mst.plugins.bvar.signs import Shock, Signs, load_signs


def shock(id_, rises=(), falls=()):
    return {"id": id_, "label": id_, "rises": list(rises), "falls": list(falls)}


def signs(rows):
    return Signs.model_validate({"shocks": [Shock.model_validate(row) for row in rows]})


def test_shipped_signs_hold_the_formalization_table():
    table = load_signs()
    assert table.ids == ["ru.monetary", "ru.inflation", "ru.fx_risk", "gl.oil", "gl.financial"]
    assert table.shocks[0].rises[:2] == ["ru.key_rate", "ru.ruonia"]


def test_the_shipped_signs_restrict_registered_variables():
    from mst.macro.registry import load_registry

    known = {variable.id for variable in load_registry().variables}
    for item in load_signs().shocks:
        assert set(item.rises + item.falls) <= known


def test_a_shock_cannot_move_a_variable_both_ways():
    with pytest.raises(ValidationError, match=r"both raises and lowers ru\.key_rate"):
        signs([shock("ru.monetary", rises=["ru.key_rate"], falls=["ru.key_rate"])])


def test_a_russian_shock_cannot_move_a_global_variable():
    with pytest.raises(ValidationError, match=r"Russian shock ru\.monetary cannot move gl\.vix"):
        signs([shock("ru.monetary", falls=["gl.vix"])])


def test_a_shock_needs_a_sign():
    with pytest.raises(ValidationError, match=r"gl\.oil restricts no variable"):
        signs([shock("gl.oil")])


def test_shock_ids_are_unique():
    with pytest.raises(ValidationError, match=r"shock gl\.oil is registered twice"):
        signs([shock("gl.oil", rises=["gl.brent"]), shock("gl.oil", rises=["gl.brent"])])
