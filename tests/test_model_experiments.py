"""The experiment runner keeps the tuning and validation datasets apart."""

import importlib.util
from argparse import Namespace
from pathlib import Path

import pytest


@pytest.fixture
def runner(tmp_path, monkeypatch):
    path = Path("bench/model-backtest/run_experiment.py")
    spec = importlib.util.spec_from_file_location("model_experiments", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "OUTPUT", tmp_path)
    return module


@pytest.mark.parametrize("half", ["tuning", "validation", "all"])
def test_runner_selects_only_requested_half_and_passes_explicit_model(runner, monkeypatch, half):
    from mst import models
    from mst.metrics import backtest, rolling
    from mst.metrics.episodes import load_episodes

    calls = []

    def grid(**kwargs):
        selected = models.selected("rw.joint")
        assert selected._spec.band_fit
        assert not selected._spec.stochastic_scale
        calls.append(("grid", kwargs))
        return 1

    def episodes(**kwargs):
        calls.append(("episodes", kwargs))
        return 1

    monkeypatch.setattr(rolling, "run", grid)
    monkeypatch.setattr(backtest, "run", episodes)
    args = Namespace(
        variant="band", half=half, checks=["rolling", "episodes"], workers=1,
        drivers=None, portfolio=None, models=None,
    )
    assert runner.execute(args) == 0
    plan = load_episodes(runner.ROOT / "bench/model-backtest/episodes.toml")
    assert calls[0][1]["half"] == (None if half == "all" else half)
    assert calls[1][1]["rolling"] is False
    assert calls[1][1]["only"] == (
        [] if half == "all" else [ep.id for ep in plan.episodes if ep.half == half]
    )
    assert calls[0][1]["models"] == calls[1][1]["models"]
    assert calls[0][1]["details"] != calls[1][1]["details"]


def test_runner_refuses_reusing_directory_with_changed_configuration(runner, monkeypatch):
    from mst.metrics import rolling

    monkeypatch.setattr(rolling, "run", lambda **kwargs: 0)
    args = Namespace(
        variant="band", half="tuning", checks=["rolling"], workers=1,
        drivers=None, portfolio=None, models=None,
    )
    runner.execute(args)
    runner.VARIANTS["band"] = {"band_fit": True, "momentum": False}
    with pytest.raises(ValueError, match="configuration changed"):
        runner.execute(args)
