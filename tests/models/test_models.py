"""The model lifecycle: which macro model is chosen and which model estimates a quantity."""

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
