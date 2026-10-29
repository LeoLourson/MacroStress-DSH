"""``python -m mst.plugin``: scaffold, check and calculate a demonstration position."""

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


def test_a_broken_package_names_the_contract_problem(tmp_path):
    target = plugin.new("gold", dest=tmp_path / "plugins")
    path = target / "plugin.py"
    path.write_text(
        path.read_text(encoding="utf-8").replace('">=1.0,<2.0"', '">=9.0,<10.0"'),
        encoding="utf-8",
    )
    problems = plugin.check(path=target)
    assert any("SDK" in problem for problem in problems)


def test_new_refuses_to_overwrite_a_package(tmp_path):
    plugin.new("gold", dest=tmp_path / "plugins")
    with pytest.raises(SystemExit, match="already exists"):
        plugin.new("gold", dest=tmp_path / "plugins")
