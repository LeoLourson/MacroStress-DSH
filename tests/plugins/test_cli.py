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


def test_a_fresh_satellite_package_passes_check(tmp_path):
    target = plugin.new("arlink", kind="satellite", quantity="bond_spread", dest=tmp_path)
    assert (target / "model.py").is_file() and (target / "tests").is_dir()
    assert plugin.check(path=target) == []


def test_a_fresh_macro_package_passes_check_and_meets_constraints(tmp_path):
    target = plugin.new("walk", kind="macro", dest=tmp_path)
    assert plugin.check(path=target) == []


    import numpy as np

    loaded = plugin.load(target)
    model = next(iter(loaded.macro))
    simulation = model.simulate(
        _walk_model(),
        _plan(),
        draws=4,
        seed=1,
    )
    assert simulation.paths.shape == (4, 4, 2)
    assert np.allclose(simulation.paths[:, 2, 1], 18.0, atol=1e-9)


def test_a_scaffolded_macro_model_starts_from_the_last_known_value(tmp_path):
    """The last panel month need not know every variable: no ``NaN`` may enter the paths."""
    import numpy as np

    target = plugin.new("walk", kind="macro", dest=tmp_path)
    model = next(iter(plugin.load(target).macro))
    panel = np.array(
        [
            [1.0, 10.0, 100.0],
            [2.0, 11.0, np.nan],
            [3.0, np.nan, np.nan],
        ]
    )
    loaded = model.load(
        {
            "as_of": "2026-10-01",
            "ids": ["gl.brent", "ru.key_rate", "ru.usdrub"],
            "months": ["2026-08-01", "2026-09-01", "2026-10-01"],
        },
        {
            "panel_y": panel,
            "drift": np.zeros(3),
            "chol": np.eye(3),
        },
    )
    assert np.array_equal(loaded.last, np.array([3.0, 11.0, 100.0]))

    from mst.model_sdk import ScenarioPlan

    simulation = model.simulate(loaded, ScenarioPlan(horizon=2), draws=3, seed=1)
    assert simulation.paths.shape == (3, 3, 3)
    assert np.all(np.isfinite(simulation.paths))


def test_a_scaffolded_macro_model_names_a_variable_without_observations(tmp_path):
    import numpy as np

    target = plugin.new("walk", kind="macro", dest=tmp_path)
    model = next(iter(plugin.load(target).macro))
    panel = np.array([[1.0, np.nan], [np.nan, np.nan]])
    with pytest.raises(ValueError, match="no observation of b"):
        model.load(
            {"as_of": "2026-10-01", "ids": ["a", "b"], "months": ["2026-09-01", "2026-10-01"]},
            {"panel_y": panel, "drift": np.zeros(2), "chol": np.eye(2)},
        )


def test_a_scaffolded_satellite_model_starts_from_the_object_above(tmp_path):
    """A served object without its own history starts from the nearest one that has it."""
    import numpy as np

    from mst.drivers.registry import load_registry
    from mst.model_sdk import SatellitePart

    target = plugin.new("arlink", kind="satellite", quantity="bond_spread", dest=tmp_path)
    model = next(iter(plugin.load(target).satellites))
    quantity = load_registry().quantity("bond_spread")
    width = 2 + len(quantity.regressors)
    part = SatellitePart(
        meta={
            "quantity": quantity.model_dump(mode="json", by_alias=True),
            "months": ["2026-08-01", "2026-09-01"],
            "nodes": ["top", "child"],
            "parent": [-1, 0],
            "classes": ["group", "group"],
            "driver": [None, "bond_spread:child"],
            "rows": [2, 0],
            "checks": {
                "1": {
                    "satellite": None,
                    "baseline": None,
                    "forecasts": 0,
                    "served": True,
                    "reason": "",
                },
            },
        },
        arrays={
            "coef": np.zeros((2, width)),
            "sigma": np.zeros(2),
            "d": np.array([[1000.0, np.nan], [2000.0, np.nan]]),
            "approximate": np.zeros((2, 2), dtype=bool),
            "x": np.zeros((2, width - 2)),
        },
    )
    loaded = model.load({"bond_spread": part})
    simulation = model.simulate(
        loaded,
        _satellite_macro(quantity.regressors, horizon=2),
        ["bond_spread:child"],
        [],
        correlation=None,
        shocks=np.zeros((2, 2, 0)),
        seed=1,
        stream=0,
    )
    assert np.allclose(simulation.paths[:, 0, 0], 2000.0)
    assert np.all(np.isfinite(simulation.paths))


def test_a_scaffolded_satellite_model_names_a_driver_without_any_history(tmp_path):
    import numpy as np

    from mst.drivers.registry import load_registry
    from mst.model_sdk import SatellitePart

    target = plugin.new("arlink", kind="satellite", quantity="bond_spread", dest=tmp_path)
    model = next(iter(plugin.load(target).satellites))
    quantity = load_registry().quantity("bond_spread")
    width = 2 + len(quantity.regressors)
    part = SatellitePart(
        meta={
            "quantity": quantity.model_dump(mode="json", by_alias=True),
            "months": ["2026-08-01", "2026-09-01"],
            "nodes": ["top", "child"],
            "parent": [-1, 0],
            "classes": ["group", "group"],
            "driver": [None, "bond_spread:child"],
            "rows": [0, 0],
            "checks": {},
        },
        arrays={
            "coef": np.zeros((2, width)),
            "sigma": np.zeros(2),
            "d": np.full((2, 2), np.nan),
            "approximate": np.zeros((2, 2), dtype=bool),
            "x": np.zeros((2, width - 2)),
        },
    )
    loaded = model.load({"bond_spread": part})
    with pytest.raises(ValueError, match="no history of its own"):
        model.simulate(
            loaded,
            _satellite_macro(quantity.regressors, horizon=1),
            ["bond_spread:child"],
            [],
            correlation=None,
            shocks=np.zeros((2, 1, 0)),
            seed=1,
            stream=0,
        )


def _satellite_macro(ids, *, horizon: int):
    """A macro simulation of zeros with the current and the next ``horizon`` months."""
    from datetime import date

    import numpy as np

    from mst.macro.panel import shift
    from mst.model_sdk import MacroSimulation

    months = [shift(date(2026, 9, 1), k) for k in range(horizon + 1)]
    return MacroSimulation(
        ids=list(ids),
        months=months,
        paths=np.zeros((2, horizon + 1, len(ids))),
        history=np.zeros((2, 1, len(ids))),
    )


def _walk_model():
    from datetime import date

    import numpy as np

    class Fake:
        ids = ["gl.brent", "ru.key_rate"]
        last_month = date(2026, 8, 1)
        last = np.array([100.0, 16.0])
        drift = np.array([0.1, 0.0])
        chol = np.eye(2)

    return Fake()


def _plan():
    from mst.model_sdk.contracts import LinearConstraint, LinearTerm, ScenarioPlan

    return ScenarioPlan(
        horizon=3,
        constraints=(
            LinearConstraint(
                terms=(LinearTerm(variable="ru.key_rate", month=2, weight=1.0),),
                low=18.0,
                high=18.0,
            ),
        ),
        causes=("ru.key_rate",),
    )


def test_a_kind_must_be_known(tmp_path, capsys):
    with pytest.raises(SystemExit) as caught:
        plugin.main(["new", "odd", "--kind", "strange", "--dest", str(tmp_path)])
    assert caught.value.code == 2  # argparse refuses the choice before the scaffold runs
