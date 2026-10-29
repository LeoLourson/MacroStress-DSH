"""Discovery of ``mst.plugins.<id>`` packages and the checks of their declarations."""

import sys

import pytest

from mst import plugins
from mst.engine import Catalog
from mst.plugin_sdk import Position
from mst.plugin_sdk.testing import check_plugin
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


def test_a_plugin_version_has_three_parts():
    with pytest.raises(plugins.PluginError, match="x.y.z"):
        plugins.build([fakes.plugin(version="1.0")])


@pytest.mark.parametrize(
    "changes",
    [{"sdk": 1.0}, {"models": None}, {"label": None}, {"demo": (None,)}],
)
def test_a_malformed_declaration_is_a_finding_not_a_crash(changes):
    findings = check_plugin(fakes.plugin(**changes))
    assert findings
    assert all(finding.plugin_id == "gold" for finding in findings)


def test_a_package_must_declare_its_own_name(tmp_path, monkeypatch):
    package(
        tmp_path,
        "wrong",
        "from mst.plugin_sdk import Plugin\n"
        "plugin = Plugin(plugin_id='other', version='0.1.0', sdk='>=1.0,<2.0', "
        "label='Wrong')\n",
    )
    monkeypatch.setattr(plugins, "__path__", [*plugins.__path__, str(tmp_path)])
    try:
        with pytest.raises(plugins.PluginError, match="declares plugin_id 'other'"):
            plugins.discover()
    finally:
        forget("mst.plugins.wrong")


def test_the_default_catalog_is_built_only_when_asked(monkeypatch):
    from mst.engine import model_catalog

    calls = []

    class Empty:
        models = {}

    monkeypatch.setattr(model_catalog, "registry", lambda: calls.append(1) or Empty())
    catalog = model_catalog.Catalog()
    assert calls == []
    with pytest.raises(KeyError, match="unknown model"):
        catalog.model("gold.paths.v1")
    assert calls == [1]
