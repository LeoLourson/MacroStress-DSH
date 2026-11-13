"""The plugin's public lifecycle, neutral contracts and stored-version round trip."""

import ast
import json
from datetime import date
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from mst.macro.panel import Panel, shift
from mst.macro.registry import Registry, Variable
from mst.model_sdk import MacroEstimateContext, ScenarioPlan, check_interventions
from mst.plugins import build, declaration_problems
from mst.plugins.rw import MODEL, plugin
from mst.plugins.rw.spec import Spec


def panel():
    rng = np.random.default_rng(12)
    steps = rng.standard_t(8, (180, 2))
    y = np.vstack([np.zeros(2), np.cumsum(steps, axis=0)])
    y[-2:, 1] = np.nan
    return Panel(
        ["gl.x", "ru.y"],
        1,
        [shift(date(2012, 1, 1), t) for t in range(len(y))],
        y,
        np.zeros_like(y, dtype=bool),
    )


def test_plugin_is_discoverable_and_conforms_without_other_plugin_imports():
    assert not declaration_problems(plugin)
    registry = build([plugin])
    assert registry.macro["rw.joint"] is MODEL
    assert MODEL.latents == ("scale_path",)
    allowed = ("mst.model_sdk", "mst.plugin_sdk", "mst.macro.panel", "mst.macro.registry")
    for path in Path("src/mst/plugins/rw").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                assert not node.module.startswith("mst.") or node.module.startswith(allowed)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("mst.") or alias.name.startswith(allowed)


def test_estimate_load_round_trip_and_intervention_contract(monkeypatch, tmp_path):
    from mst.plugins.rw import model as adapter

    data = panel()
    monkeypatch.setattr(adapter, "build_panel", lambda *args, **kwargs: data)
    monkeypatch.setattr(
        adapter,
        "load_spec",
        lambda: Spec(drifts=["gl.x"], breaks=[], stochastic_scale=False, sample_scale=False),
    )
    registry = Registry(
        variables=[
            Variable(id=id_, label=id_, unit="unit", transform="identity", source="test")
            for id_ in data.ids
        ]
    )
    context = MacroEstimateContext(store=None, registry=registry, as_of=date(2026, 11, 1))
    estimate = MODEL.estimate(context)
    meta = json.loads(json.dumps(estimate.meta))
    assert meta["as_of"] == "2026-11-01"
    assert meta["shocks"] == data.ids
    assert meta["structural_form"] == {"order_within_month": data.ids, "time_recursion": True}
    assert_array_equal(estimate.arrays["panel_y"], data.y)
    loaded = MODEL.load(meta, {key: value.copy() for key, value in estimate.arrays.items()})
    form = MODEL.structural_form(loaded)
    assert form.order_within_month == tuple(data.ids)
    assert form.time_recursion
    first = MODEL.simulate(loaded, ScenarioPlan(3, inner=2), draws=3, seed=5)
    from mst.macro.version import load, save

    name = save(MODEL.model_id, estimate.meta, estimate.arrays, tmp_path)
    restored = load(tmp_path, name, {MODEL.model_id: MODEL})
    reloaded = restored.model
    assert_array_equal(
        first.paths, MODEL.simulate(reloaded, ScenarioPlan(3, inner=2), draws=3, seed=5).paths
    )
    assert not check_interventions(MODEL, loaded)
    assert loaded.start_sd[0] == 0
    assert loaded.start_sd[1] > 0
    assert_allclose(loaded.start_mean[1], data.y[-3, 1])
    assert_array_equal(loaded.persistence, np.ones(2))
    assert_array_equal(loaded.intercept, loaded.drift)
    assert loaded.scale_persistence == 0
    assert loaded.scale_volatility == 0


def test_load_of_a_version_without_the_new_arrays_keeps_the_random_walk():
    meta = {
        "ids": ["gl.x", "ru.y"],
        "n_global": 1,
        "months": ["2026-01-01"],
        "nu": 8.0,
        "diagnostics": {},
    }
    arrays = {
        "drift": np.array([0.1, 0.0]),
        "scale": np.eye(2),
        "weights": np.array([1.0]),
        "start_mean": np.zeros(2),
        "start_sd": np.zeros(2),
    }
    model = MODEL.load(meta, arrays)
    assert_array_equal(model.persistence, np.ones(2))
    assert_array_equal(model.intercept, arrays["drift"])
    assert model.scale_persistence == 0.0
    assert model.scale_volatility == 0.0
