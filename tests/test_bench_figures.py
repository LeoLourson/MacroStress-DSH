"""The slide charts of a forecast result read their numbers from the result JSON: the base
fan, the history quantiles and the event shifts match the committed result."""

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "bench/forecast/results/2026-10-29-6e980206-62c73426.json"


@pytest.fixture(scope="module")
def figures():
    spec = importlib.util.spec_from_file_location("figures", ROOT / "bench/forecast/figures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def result():
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_base_fan_is_the_return_of_the_portfolio(figures, result):
    base = next(e for e in result["events"] if e.get("base"))
    year = figures.returns_by_month(base, result["value"])[-1]
    assert year["month"] == "2027-09-01"
    assert year["p05"] == pytest.approx(-21.00, abs=0.01)
    assert year["p50"] == pytest.approx(8.19, abs=0.01)
    assert year["p95"] == pytest.approx(52.86, abs=0.01)


def test_history_quantiles_come_from_the_portfolio_rows(figures, result):
    year = figures.history_quantiles(result)[12]
    assert year["p05"] == pytest.approx(-26.3, abs=0.05)
    assert year["p50"] == pytest.approx(10.1, abs=0.05)
    assert year["p95"] == pytest.approx(54.1, abs=0.05)


def test_event_shifts_name_every_event(figures, result):
    shifts = {r["scenario"]: r for r in figures.event_shifts(result)}
    assert len(shifts) == 12
    assert all(r["name"] != r["scenario"] for r in shifts.values())
    assert shifts["hormuz-closure-2m-random"]["h3"] == pytest.approx(2.30, abs=0.01)
    assert shifts["hormuz-closure-2m-random"]["h12"] == pytest.approx(-1.81, abs=0.01)


def test_main_writes_four_pictures(figures, tmp_path):
    paths = figures.main([str(RESULT), str(tmp_path)])
    assert [p.name for p in paths] == ["fan.png", "backtest.png", "events.png", "event-path.png"]
    assert all(p.stat().st_size > 10_000 for p in paths)
