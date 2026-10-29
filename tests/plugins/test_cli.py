"""``python -m mst.plugin``: scaffold, check and calculate a demonstration position."""

import sys

import pytest

from mst import plugin


def test_a_fresh_package_passes_check_and_demo_run(tmp_path):
    target = plugin.new("gold", dest=tmp_path / "plugins")
    assert (target / "plugin.py").is_file()
    assert (target / "model.py").is_file()
    assert (target / "tests").is_dir()

    assert plugin.check(path=target) == []
    text = plugin.demo_run(path=target)
    assert "P5 / P50 / P95" in text
    assert "gold.paths.v1" in text


def test_main_check_and_demo_run_return_exit_codes(tmp_path, capsys):
    target = plugin.new("gold", dest=tmp_path / "plugins")
    assert plugin.main(["check", "--path", str(target)]) == 0
    assert plugin.main(["demo-run", "--path", str(target)]) == 0
    out = capsys.readouterr().out
    assert "P5 / P50 / P95" in out


def test_a_broken_package_names_the_contract_problem(tmp_path):
    target = plugin.new("gold", dest=tmp_path / "plugins")
    path = target / "plugin.py"
    path.write_text(
        path.read_text(encoding="utf-8").replace('">=1.0,<2.0"', '">=9.0,<10.0"'),
        encoding="utf-8",
    )
    problems = plugin.check(path=target)
    assert any("SDK" in problem for problem in problems)


def test_a_package_that_cannot_be_imported_is_named(tmp_path):
    target = tmp_path / "broken"
    target.mkdir()
    (target / "__init__.py").write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    problems = plugin.check(path=target)
    assert any("cannot be imported: RuntimeError: boom" in problem for problem in problems)


def test_main_check_names_a_missing_path(tmp_path, capsys):
    assert plugin.main(["check", "--path", str(tmp_path / "nope")]) == 1
    assert "is not a Python package" in capsys.readouterr().out


def test_a_path_package_is_read_afresh(tmp_path):
    target = plugin.new("gold", dest=tmp_path / "plugins")
    assert plugin.check(path=target) == []
    path = target / "plugin.py"
    path.write_text(
        path.read_text(encoding="utf-8").replace('">=1.0,<2.0"', '">=9.0,<10.0"'),
        encoding="utf-8",
    )
    assert any("SDK" in problem for problem in plugin.check(path=target))


def test_new_refuses_to_overwrite_a_package(tmp_path):
    plugin.new("gold", dest=tmp_path / "plugins")
    with pytest.raises(SystemExit, match="already exists"):
        plugin.new("gold", dest=tmp_path / "plugins")


def test_a_label_with_quotes_is_escaped(tmp_path):
    target = plugin.new("gold", label='G"old', dest=tmp_path / "plugins")
    assert plugin.check(path=target) == []
    text = (target / "plugin.py").read_text(encoding="utf-8")
    assert 'G\\"old' in text


def test_a_label_with_braces_is_kept(tmp_path):
    target = plugin.new("gold", label="{{x}}", dest=tmp_path / "plugins")
    assert plugin.check(path=target) == []


def test_check_reports_a_broken_installed_package(tmp_path, monkeypatch, capsys):
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "__init__.py").write_text("plugin = None\n", encoding="utf-8")
    monkeypatch.setattr(plugin.plugins, "__path__", [*plugin.plugins.__path__, str(tmp_path)])
    try:
        assert plugin.main(["check"]) == 1
    finally:
        for key in list(sys.modules):
            if key == "mst.plugins.broken":
                del sys.modules[key]
    assert "broken exports no" in capsys.readouterr().out


def test_check_a_named_plugin_reports_a_broken_registry(monkeypatch):
    def broken():
        raise plugin.plugins.PluginError(
            ["mst.plugins.bad exports no `plugin = Plugin(...)` declaration"]
        )

    monkeypatch.setattr(plugin.plugins, "registry", broken)
    assert plugin.check("bad") == [
        "mst.plugins.bad exports no `plugin = Plugin(...)` declaration"
    ]


def test_demo_run_names_a_broken_registry(monkeypatch):
    def broken():
        raise plugin.plugins.PluginError(
            ["mst.plugins.bad exports no `plugin = Plugin(...)` declaration"]
        )

    monkeypatch.setattr(plugin.plugins, "registry", broken)
    with pytest.raises(SystemExit, match="mst.plugins.bad exports no"):
        plugin.demo_run("gold")
