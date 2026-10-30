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


def test_the_installed_packages_are_the_position_models_and_the_models():
    registry = plugins.registry()
    assert sorted(registry.plugins) == ["bond", "bvar", "credit", "equity", "hier", "housing"]
    assert sorted(registry.models) == [
        "bond.paths.v1",
        "credit.paths.v1",
        "equity.paths.v1",
        "housing.paths.v1",
    ]
    assert registry.owner["equity.paths.v1"] == "equity"
    assert sorted(registry.macro) == ["bvar.sv"]
    assert registry.macro_owner["bvar.sv"] == "bvar"
    assert sorted(registry.satellites) == ["hier.gibbs"]
    assert registry.quantity_model["share_price"] == ["hier.gibbs"]


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


def test_a_model_plugin_without_position_models_is_registered_by_kind():
    registry = plugins.build([fakes.model_plugin()])
    assert list(registry.plugins) == ["gmodel"]
    assert registry.models == {}
    assert sorted(registry.macro) == ["gmodel.macro.v1"]
    assert sorted(registry.satellites) == ["gmodel.sat.v1"]
    assert registry.quantity_model == {"share_price": ["gmodel.sat.v1"]}
    assert registry.macro_owner["gmodel.macro.v1"] == "gmodel"
    assert registry.satellite_owner["gmodel.sat.v1"] == "gmodel"


def test_a_model_plugin_needs_the_model_sdk_range():
    with pytest.raises(plugins.PluginError, match="model SDK range in model_sdk"):
        plugins.build([fakes.model_plugin(model_sdk="")])
    with pytest.raises(plugins.PluginError, match="requires model SDK"):
        plugins.build([fakes.model_plugin(model_sdk=">=9.0,<10.0")])


def test_a_satellite_model_serves_known_quantities_only():
    class FarAway(fakes.DemoSatelliteModel):
        quantities = ("unobtainium",)

    with pytest.raises(plugins.PluginError, match="names unknown quantities: unobtainium"):
        plugins.build([fakes.model_plugin(satellites=(FarAway(),))])


def test_a_macro_model_names_known_variables_only():
    class FarAway(fakes.DemoMacroModel):
        variables = ("gl.unobtainium",)

    with pytest.raises(plugins.PluginError, match="unknown macro variables: gl.unobtainium"):
        plugins.build([fakes.model_plugin(macro=(FarAway(),))])


def test_a_satellite_model_in_the_macro_slot_is_refused():
    problems = plugins.declaration_problems(
        fakes.model_plugin(macro=(fakes.DemoSatelliteModel(),), satellites=())
    )
    assert any("not a MacroModel" in problem for problem in problems)
    with pytest.raises(plugins.PluginError, match="not a MacroModel"):
        plugins.build(
            [fakes.model_plugin(macro=(fakes.DemoSatelliteModel(),), satellites=())]
        )


def test_a_macro_model_in_the_satellite_slot_is_refused():
    problems = plugins.declaration_problems(
        fakes.model_plugin(macro=(), satellites=(fakes.DemoMacroModel(),))
    )
    assert any("not a SatelliteModel" in problem for problem in problems)
    with pytest.raises(plugins.PluginError, match="not a SatelliteModel"):
        plugins.build(
            [fakes.model_plugin(macro=(), satellites=(fakes.DemoMacroModel(),))]
        )


def test_one_quantity_is_served_by_one_satellite_model(monkeypatch):
    class Other(fakes.DemoSatelliteModel):
        model_id = "gmodel.other.v1"

    registry = plugins.build(
        [fakes.model_plugin(satellites=(fakes.DemoSatelliteModel(), Other()))]
    )
    # Both may declare it — a plugin under development beside the installed one — but the
    # assignment refuses the ambiguity until a name resolves it.
    assert registry.quantity_model["share_price"] == ["gmodel.sat.v1", "gmodel.other.v1"]
    from mst import models as models_pkg

    monkeypatch.setattr(plugins, "registry", lambda: registry)
    with pytest.raises(ValueError, match="several satellite models declare share_price"):
        models_pkg.assignment(["share_price"])
    chosen, unserved = models_pkg.assignment(
        ["share_price"], {"share_price": "gmodel.other.v1"}
    )
    assert chosen == {"share_price": "gmodel.other.v1"} and unserved == []


def test_a_plugin_without_any_model_is_rejected():
    with pytest.raises(plugins.PluginError, match="no model is declared"):
        plugins.build(
            [fakes.model_plugin(models=(), demo=(), macro=(), satellites=())]
        )


def test_a_plugin_with_a_position_model_and_no_demo_is_rejected():
    with pytest.raises(plugins.PluginError, match="must ship a demo case"):
        plugins.build([fakes.plugin(demo=())])


class _Point:
    """An entry point stand-in: a name and what its load returns."""

    def __init__(self, name: str, declared):
        self.name, self._declared = name, declared

    def load(self):
        return self._declared


def test_entry_point_plugins_are_discovered(monkeypatch):
    monkeypatch.setattr(
        plugins.importlib.metadata,
        "entry_points",
        lambda **kwargs: [_Point("gmodel", fakes.model_plugin())]
        if kwargs == {"group": plugins.ENTRY_POINTS}
        else [],
    )
    found = plugins.discover()
    ids = [plugin.plugin_id for plugin in found]
    assert "gmodel" in ids
    assert ids.count("gmodel") == 1
    assert {"bond", "credit", "equity", "housing"} <= set(ids)


def test_an_entry_point_that_names_another_plugin_is_rejected(monkeypatch):
    monkeypatch.setattr(
        plugins.importlib.metadata,
        "entry_points",
        lambda **kwargs: [_Point("other", fakes.model_plugin())],
    )
    with pytest.raises(plugins.PluginError, match="declares plugin_id 'gmodel'"):
        plugins.discover()


def test_plugins_of_the_extra_paths_are_discovered(tmp_path, monkeypatch):
    (tmp_path / "gmodel").mkdir()
    (tmp_path / "gmodel" / "__init__.py").write_text(
        "from mst.plugin_sdk import Plugin\n"
        "plugin = Plugin(plugin_id='gmodel', version='0.1.0', label='Gmodel', "
        "model_sdk='>=1.0,<2.0')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(plugins.PATH_ENV, str(tmp_path))
    try:
        found = plugins.discover()
        ids = [plugin.plugin_id for plugin in found]
        assert "gmodel" in ids and ids.count("gmodel") == 1
        # A plugin of the paths that repeats a namespace plugin id is a problem.
        (tmp_path / "equity").mkdir()
        (tmp_path / "equity" / "__init__.py").write_text(
            "from mst.plugin_sdk import Plugin\n"
            "plugin = Plugin(plugin_id='equity', version='0.1.0', label='Not equity')\n",
            encoding="utf-8",
        )
        with pytest.raises(plugins.PluginError, match="declared twice"):
            plugins.discover()
    finally:
        monkeypatch.delenv(plugins.PATH_ENV, raising=False)


def test_a_duplicate_plugin_id_names_both_sources(tmp_path, monkeypatch):
    (tmp_path / "equity").mkdir()
    (tmp_path / "equity" / "__init__.py").write_text(
        "from mst.plugin_sdk import Plugin\n"
        "plugin = Plugin(plugin_id='equity', version='0.1.0', label='Not equity')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(plugins.PATH_ENV, str(tmp_path))
    with pytest.raises(plugins.PluginError) as caught:
        plugins.discover()
    message = str(caught.value)
    assert "plugin 'equity' is declared twice" in message
    assert "mst.plugins.equity" in message
    assert f"{plugins.PATH_ENV} {tmp_path / 'equity'}" in message


def test_a_missing_plugins_path_root_is_named(tmp_path, monkeypatch):
    missing = tmp_path / "nope"
    monkeypatch.setenv(plugins.PATH_ENV, str(missing))
    with pytest.raises(plugins.PluginError, match="is not a directory") as caught:
        plugins.discover()
    assert f"{plugins.PATH_ENV}: {missing}" in str(caught.value)


def test_load_package_reads_a_package_outside_the_namespace(tmp_path):
    (tmp_path / "gmodel").mkdir()
    (tmp_path / "gmodel" / "__init__.py").write_text(
        "from mst.plugin_sdk import Plugin\n"
        "plugin = Plugin(plugin_id='gmodel', version='0.1.0', label='Gmodel')\n",
        encoding="utf-8",
    )
    assert plugins.load_package(tmp_path / "gmodel").plugin_id == "gmodel"
    with pytest.raises(plugins.PluginLoadError, match="is not a Python package"):
        plugins.load_package(tmp_path / "nowhere")
