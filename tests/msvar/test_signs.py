"""The sign table of the MS-VAR: the named shocks, their signs and the regimes."""

import pytest
from pydantic import ValidationError

from mst.plugins.msvar.signs import Regime, Shock, Signs, load_signs


def shock(id_, rises=(), falls=()):
    return {"id": id_, "label": id_, "rises": list(rises), "falls": list(falls)}


def regime(id_):
    return {"id": id_, "label": id_}


def signs(rows, regimes=()):
    return Signs.model_validate(
        {
            "shocks": [Shock.model_validate(row) for row in rows],
            "regimes": [Regime.model_validate(row) for row in regimes],
        }
    )


def test_shipped_signs_hold_the_formalization_table():
    table = load_signs()
    assert table.ids == ["ru.monetary", "ru.inflation", "ru.fx_risk", "gl.oil", "gl.financial"]
    assert table.regime_ids == ["regime.calm", "regime.stress"]


def test_the_shipped_signs_restrict_registered_variables():
    from mst.macro.registry import load_registry

    known = {variable.id for variable in load_registry().variables}
    for item in load_signs().shocks:
        assert set(item.rises + item.falls) <= known


def test_a_shock_needs_a_sign():
    with pytest.raises(ValidationError, match=r"gl\.oil restricts no variable"):
        signs([shock("gl.oil")], [regime("regime.calm"), regime("regime.stress")])


def test_regime_ids_are_separate_from_shock_ids():
    with pytest.raises(ValidationError, match=r"regime\.calm is registered twice"):
        signs(
            [shock("gl.oil", rises=["gl.brent"])],
            [regime("regime.calm"), regime("regime.calm")],
        )


def test_a_switching_model_needs_two_regimes():
    with pytest.raises(ValidationError, match="at least two regimes"):
        signs([shock("gl.oil", rises=["gl.brent"])], [regime("regime.calm")])


def test_a_regime_id_names_itself_a_regime():
    with pytest.raises(ValidationError, match="String should match pattern"):
        Regime.model_validate({"id": "calm", "label": "спокойный"})


def test_an_unknown_variable_is_named_at_load(tmp_path):
    path = tmp_path / "signs.toml"
    path.write_text(
        '[[shocks]]\nid = "ru.monetary"\nlabel = "t"\nrises = ["ru.key_rat"]\n'
        '[[regimes]]\nid = "regime.calm"\nlabel = "c"\n'
        '[[regimes]]\nid = "regime.stress"\nlabel = "s"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"ru\.monetary restricts unknown variable ru\.key_rat"):
        load_signs(path)
