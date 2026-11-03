"""The model lifecycle: which macro model is chosen and which model estimates a quantity."""

import numpy as np
import pytest

from mst import models as models_pkg
from mst import plugins
from tests.plugins import fakes


def test_an_empty_registry_says_no_macro_model_is_installed(monkeypatch):
    monkeypatch.setattr(plugins, "registry", lambda: plugins.Registry())
    with pytest.raises(ValueError, match="no macro model is installed"):
        models_pkg.selected()


def test_one_installed_macro_model_is_the_default(monkeypatch):
    model = fakes.DemoMacroModel()
    registry = plugins.build([fakes.model_plugin(macro=(model,), satellites=())])
    monkeypatch.setattr(plugins, "registry", lambda: registry)
    assert models_pkg.selected() is model


def test_several_macro_models_are_named_one(monkeypatch):
    class Other(fakes.DemoMacroModel):
        model_id = "gmodel.other.v1"

    registry = plugins.build(
        [fakes.model_plugin(macro=(fakes.DemoMacroModel(), Other()))]
    )
    monkeypatch.setattr(plugins, "registry", lambda: registry)
    with pytest.raises(ValueError, match="several macro models are installed, name one"):
        models_pkg.selected()
    assert models_pkg.selected("gmodel.other.v1").model_id == "gmodel.other.v1"


def test_a_named_unknown_macro_model_is_refused(monkeypatch):
    monkeypatch.setattr(plugins, "registry", lambda: plugins.Registry())
    with pytest.raises(ValueError, match="no installed macro model 'nope'; installed: none"):
        models_pkg.selected("nope")


def test_an_override_backs_any_installed_satellite_model(monkeypatch):
    # The contract says an explicit assignment may hand a model a quantity it does not
    # declare: the context's ``quantities`` then hold the assignment, not the declaration.
    registry = plugins.build([fakes.model_plugin()])
    monkeypatch.setattr(plugins, "registry", lambda: registry)
    chosen, unserved = models_pkg.assignment(
        ["bond_spread"], {"bond_spread": "gmodel.sat.v1"}
    )
    assert chosen == {"bond_spread": "gmodel.sat.v1"}
    assert unserved == []


def test_estimate_macro_hands_the_model_its_answers_and_checkpoints(tmp_path, monkeypatch):
    from datetime import date

    from mst.model_sdk.contracts import MacroEstimate

    seen = {}

    class Answering(fakes.DemoMacroModel):
        model_id = "gmodel.answers.v1"

        def estimate(self, context):
            seen["inputs"] = context.inputs
            seen["checkpoints"] = context.checkpoints
            return MacroEstimate(
                meta={
                    "as_of": context.as_of.isoformat(),
                    "ids": ["gl.brent"],
                    "n_global": 1,
                    "months": [date(2026, 1, 1).isoformat()],
                    "shocks": ["gl.oil"],
                },
                arrays={
                    "panel_y": np.zeros((1, 1)),
                    "panel_approximate": np.zeros((1, 1), dtype=bool),
                },
            )

    model = Answering()
    registry = plugins.build([fakes.model_plugin(macro=(model,), satellites=())])
    monkeypatch.setattr(plugins, "registry", lambda: registry)

    answers = tmp_path / "elicitation"
    name = models_pkg.estimate_macro(
        "gmodel.answers.v1",
        store=tmp_path / "series",
        models=tmp_path / "models",
        as_of=date(2026, 1, 31),
        inputs=answers,
    )
    assert seen["inputs"] == answers
    assert seen["checkpoints"] == tmp_path / "models" / ".checkpoints" / "macro" / model.model_id
    # The version is saved and the checkpoint folder is gone.
    assert (tmp_path / "models" / "macro" / name / "meta.json").is_file()
    assert not seen["checkpoints"].exists()


def test_estimate_macro_without_inputs_hands_the_model_none(tmp_path, monkeypatch):
    from datetime import date

    from mst.model_sdk.contracts import MacroEstimate

    seen = {}

    class Silent(fakes.DemoMacroModel):
        model_id = "gmodel.silent.v1"

        def estimate(self, context):
            seen["inputs"] = context.inputs
            return MacroEstimate(
                meta={
                    "as_of": context.as_of.isoformat(),
                    "ids": ["gl.brent"],
                    "n_global": 1,
                    "months": [date(2026, 1, 1).isoformat()],
                    "shocks": ["gl.oil"],
                },
                arrays={
                    "panel_y": np.zeros((1, 1)),
                    "panel_approximate": np.zeros((1, 1), dtype=bool),
                },
            )

    registry = plugins.build([fakes.model_plugin(macro=(Silent(),), satellites=())])
    monkeypatch.setattr(plugins, "registry", lambda: registry)
    models_pkg.estimate_macro(
        "gmodel.silent.v1",
        store=tmp_path / "series",
        models=tmp_path / "models",
        as_of=date(2026, 1, 31),
    )
    assert seen["inputs"] is None
