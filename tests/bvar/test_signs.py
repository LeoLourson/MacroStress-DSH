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


def test_an_unknown_variable_is_named_at_load(tmp_path):
    path = tmp_path / "signs.toml"
    path.write_text(
        '[[shocks]]\nid = "ru.monetary"\nlabel = "t"\nrises = ["ru.key_rat"]\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"ru\.monetary restricts unknown variable ru\.key_rat"):
        load_signs(path)


def test_the_monetary_shock_carries_a_share_of_the_key_rate_surprise():
    monetary = load_signs().shocks[0]
    assert monetary.shares == {"ru.key_rate": 0.1}


def test_the_global_shocks_carry_a_share_of_their_variable():
    shares = {shock.id: shock.shares for shock in load_signs().shocks}
    assert shares["gl.oil"] == {"gl.brent": 0.1}
    assert shares["gl.financial"] == {"gl.vix": 0.1}


def test_a_share_needs_a_signed_variable():
    row = shock("ru.monetary", rises=["ru.key_rate"]) | {"shares": {"ru.cpi": 0.1}}
    with pytest.raises(ValidationError, match=r"ru\.monetary bounds the share of ru\.cpi"):
        signs([row])


@pytest.mark.parametrize("share", [0.0, 1.5])
def test_a_share_lies_in_the_unit_interval(share):
    row = shock("ru.monetary", rises=["ru.key_rate"]) | {"shares": {"ru.key_rate": share}}
    with pytest.raises(ValidationError, match=r"share of ru\.key_rate .* in \(0, 1\]"):
        signs([row])
