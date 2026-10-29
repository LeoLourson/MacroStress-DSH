"""Discovery of ``mst.plugins.<id>`` packages and the checks of their declarations."""

import sys

import pytest

from mst import plugins
from mst.engine import Catalog
from mst.plugin_sdk import Position
from tests.plugins import fakes


def package(root, name: str, text: str) -> None:
    """A package ``root/<name>`` with ``__init__.py`` holding ``text``."""
    target = root / name
    target.mkdir(parents=True, exist_ok=True)
    (target / "__init__.py").write_text(text, encoding="utf-8")


def forget(*names: str) -> None:
    """Drop modules from ``sys.modules`` so a temporary package is imported afresh."""
    for name in list(sys.modules):
        if any(name == item or name.startswith(item + ".") for item in names):
            del sys.modules[name]


def test_the_installed_packages_are_the_four_position_models():
    registry = plugins.registry()
    assert sorted(registry.plugins) == ["bond", "credit", "equity", "housing"]
    assert sorted(registry.models) == [
        "bond.paths.v1",
        "credit.paths.v1",
        "equity.paths.v1",
        "housing.paths.v1",
    ]
    assert registry.owner["equity.paths.v1"] == "equity"


def test_the_catalog_answers_by_id_and_by_type():
    catalog = Catalog()
    assert catalog.model("equity.paths.v1").instrument_types == ("common_stock",)
    position = Position(id="x", type="common_stock", notional=1.0)
    assert catalog.model_for(position).model_id == "equity.paths.v1"
    with pytest.raises(KeyError, match="unknown model"):
        catalog.model("gold.paths.v1")


def test_a_package_without_a_declaration_is_named(tmp_path, monkeypatch):
    package(tmp_path, "brokenplugins", "")
    package(tmp_path, "brokenplugins/bad", "value = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        with pytest.raises(plugins.PluginError, match="brokenplugins.bad exports no"):
            plugins.discover("brokenplugins")
    finally:
        forget("brokenplugins")


def test_a_package_that_cannot_be_imported_is_named(tmp_path, monkeypatch):
    package(tmp_path, "brokenimports", "")
    package(tmp_path, "brokenimports/bad", "raise RuntimeError('boom')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        with pytest.raises(plugins.PluginError, match="cannot be imported: RuntimeError: boom"):
            plugins.discover("brokenimports")
    finally:
        forget("brokenimports")


def test_a_plugin_declared_twice_is_rejected():
    with pytest.raises(plugins.PluginError, match="declared twice"):
        plugins.build([fakes.plugin(), fakes.plugin()])


def test_a_model_must_live_in_the_namespace_of_its_plugin():
    class Foreign(fakes.DemoModel):
        model_id = "other.paths.v1"

    with pytest.raises(plugins.PluginError, match="must start with 'gold.'"):
        plugins.build([fakes.plugin(models=(Foreign(),))])


def test_a_metric_must_live_in_the_namespace_of_its_plugin():
    class Foreign(fakes.DemoModel):
        metrics = ("other.loss",)

    with pytest.raises(plugins.PluginError, match="must start with 'gold.'"):
        plugins.build([fakes.plugin(models=(Foreign(),))])


def test_two_models_must_not_serve_one_instrument_type():
    class Second(fakes.DemoModel):
        model_id = "gold.second.v1"

    with pytest.raises(plugins.PluginError, match="instrument type 'gold' is served by"):
        plugins.build([fakes.plugin(models=(fakes.DemoModel(), Second()))])


def test_a_plugin_must_require_a_compatible_sdk():
    with pytest.raises(plugins.PluginError, match="requires SDK"):
        plugins.build([fakes.plugin(sdk=">=9.0,<10.0")])
