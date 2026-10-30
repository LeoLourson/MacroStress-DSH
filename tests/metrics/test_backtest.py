"""The episode backtest: the runner over fake models, the cache and the report."""

import json
from datetime import date

import pytest

from mst import models as models_pkg
from mst.data.store import Observation
from mst.macro.panel import months_between, shift
from mst.metrics import backtest as runner
from mst.metrics.episodes import calculation_date, load_episodes

from . import simulated

KNOWN_ON = date(2030, 1, 1)


def manifest(tmp_path, episodes):
    """A corpus manifest with the named episodes, each starting on its month."""
    rows = [
        {
            "id": id_,
            "start": f"{start:%Y-%m}",
            "region": "ru",
            "country": "RU",
            "title": id_,
            "factors": [],
            "data": [],
            "path": f"episodes/{id_}.md",
            "status": "ready",
        }
        for id_, start in episodes
    ]
    file = tmp_path / "manifest.json"
    file.write_text(
        json.dumps(
            {
                "corpus_id": "c",
                "version": "1",
                "updated": "2026-10-29",
                "description": "",
                "vocabularies": {"factors": {}, "flags": {}},
                "sources": [],
                "data_series": [],
                "episodes": rows,
            }
        ),
        encoding="utf-8",
    )
    return file


def plan_file(tmp_path, episodes):
    """An episodes.toml with the named checks: one constrained variable, one driving shock."""
    text = [
        "draws = 40",
        "seed = 3",
        "after_months = 6",
        "",
        "",
    ]
    for id_, variable, shock in episodes:
        text += [
            "[[episodes]]",
            f'id = "{id_}"',
            f'constrained = ["{variable}"]',
            f'driving = ["{shock}"]',
            'reason = "test"',
            'sources = ["S1"]',
            "",
            "",
        ]
    file = tmp_path / "episodes.toml"
    file.write_text("\n".join(text), encoding="utf-8")
    return file


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A store, a corpus manifest, an episode plan and the fake models installed."""
    store = simulated.series_store(tmp_path / "series")
    macro = simulated.FakeMacro()
    satellite = simulated.FakeSatellite()
    monkeypatch.setattr(models_pkg, "selected", lambda model_id=None: macro)
    monkeypatch.setattr(models_pkg, "satellites", lambda: {satellite.model_id: satellite})
    monkeypatch.setattr(
        models_pkg, "assignment", lambda quantities, overrides=None: (
            {quantity: satellite.model_id for quantity in quantities},
            [],
        )
    )
    return {
        "store": store,
        "models": tmp_path / "models",
        "macro": macro,
        "satellite": satellite,
        "tmp": tmp_path,
    }


def test_the_runner_scores_the_cells_of_every_episode(workspace, tmp_path, capsys, monkeypatch):
    corpus = manifest(tmp_path, [("R1", date(2023, 6, 1)), ("R2", date(2024, 3, 1))])
    plan = plan_file(
        tmp_path, [("R1", "gl.brent", "fake.demand"), ("R2", "ru.key_rate", "fake.rate")]
    )
    monkeypatch.setattr(runner, "load_episodes", lambda path=None: load_episodes(plan))

    code = runner.run(
        store=workspace["store"].root,
        models=workspace["models"],
        corpus=corpus,
        as_of=KNOWN_ON,
    )

    assert code == 0
    out = capsys.readouterr().out
    assert "## Episodes" in out and "R1" in out and "R2" in out
    assert "## Quantities" in out and "ru.key_rate" in out
    # The constrained variable is reported but never scored.
    assert "gl.brent" in out


def test_the_episode_versions_are_cached_by_the_data(workspace, tmp_path, monkeypatch):
    corpus = manifest(tmp_path, [("R1", date(2023, 6, 1))])
    plan = plan_file(tmp_path, [("R1", "gl.brent", "fake.demand")])
    monkeypatch.setattr(runner, "load_episodes", lambda path=None: load_episodes(plan))
    arguments = dict(
        store=workspace["store"].root,
        models=workspace["models"],
        corpus=corpus,
        as_of=KNOWN_ON,
    )

    assert runner.run(**arguments) == 0
    estimated_once = len(workspace["macro"].calls)
    assert estimated_once == 1

    assert runner.run(**arguments) == 0
    assert len(workspace["macro"].calls) == estimated_once  # the cache answered
    assert (workspace["models"] / "backtest" / "macro").is_dir()


def test_an_unknown_driving_shock_is_reported_not_raised(workspace, tmp_path, monkeypatch, capsys):
    corpus = manifest(tmp_path, [("R1", date(2023, 6, 1))])
    plan = plan_file(tmp_path, [("R1", "gl.brent", "no.such.shock")])
    monkeypatch.setattr(runner, "load_episodes", lambda path=None: load_episodes(plan))

    code = runner.run(
        store=workspace["store"].root, models=workspace["models"], corpus=corpus, as_of=KNOWN_ON
    )

    assert code == 1
    assert "the macro model knows no shocks no.such.shock" in capsys.readouterr().out


def test_an_episode_missing_from_the_manifest_is_reported(workspace, tmp_path, monkeypatch, capsys):
    corpus = manifest(tmp_path, [])
    plan = plan_file(tmp_path, [("R9", "gl.brent", "fake.demand")])
    monkeypatch.setattr(runner, "load_episodes", lambda path=None: load_episodes(plan))

    code = runner.run(
        store=workspace["store"].root, models=workspace["models"], corpus=corpus, as_of=KNOWN_ON
    )

    assert code == 1
    assert "R9: not in the corpus manifest" in capsys.readouterr().out


def test_the_details_file_holds_the_full_report(workspace, tmp_path, monkeypatch):
    corpus = manifest(tmp_path, [("R1", date(2023, 6, 1))])
    plan = plan_file(tmp_path, [("R1", "gl.brent", "fake.demand")])
    monkeypatch.setattr(runner, "load_episodes", lambda path=None: load_episodes(plan))
    details = tmp_path / "details.json"

    assert (
        runner.run(
            store=workspace["store"].root,
            models=workspace["models"],
            corpus=corpus,
            as_of=KNOWN_ON,
            details=details,
        )
        == 0
    )

    report = json.loads(details.read_text(encoding="utf-8"))
    assert report["problems"] == []
    assert len(report["episodes"]) == 1
    row = report["quantities"]["ru.key_rate"]
    assert row["scored"] > 0 and row["crps"] > 0
    assert row["baseline_crps"] is not None
    assert row["calibration"]["shares"] and row["sharpness"] > 0
    assert row["admission"]["share"] < 0.05 or row["admission"]["problems"]
    # The constrained variable keeps its cells, marked as constrained.
    constrained = report["quantities"]["gl.brent"]
    assert constrained["constrained"] > 0


def test_the_satellites_score_the_served_drivers(workspace, tmp_path, monkeypatch):
    corpus = manifest(tmp_path, [("R1", date(2023, 6, 1))])
    plan = plan_file(tmp_path, [("R1", "gl.brent", "fake.demand")])
    monkeypatch.setattr(runner, "load_episodes", lambda path=None: load_episodes(plan))
    # The driver catalog of the workspace: one object, one driver, with a realized series.
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "quantities": {
                    "share_price": {
                        "objects": [
                            {
                                "id": "sber",
                                "class": "company",
                                "label": "Сбербанк",
                                "codes": {"secid": ["SBER"]},
                            }
                        ],
                        "drivers": ["sber"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    store = workspace["store"]
    grid = months_between(simulated.START, shift(simulated.START, 59))
    store.write(
        "drv.share_price.sber",
        [
            Observation(month, float(5.0 + 0.01 * t), month, False)
            for t, month in enumerate(grid)
        ],
    )
    details = tmp_path / "details.json"

    code = runner.run(
        store=workspace["store"].root,
        models=workspace["models"],
        corpus=corpus,
        as_of=KNOWN_ON,
        drivers=catalog,
        satellites=True,
        details=details,
    )

    assert code == 0
    report = json.loads(details.read_text(encoding="utf-8"))
    row = report["quantities"]["share_price:sber"]
    assert row["level"] == "driver/share_price"
    assert row["scored"] > 0
    # The naive forecast of a target is scored alongside the random walk.
    assert row["naive_crps"] is not None
    assert row["baseline_crps"] is not None


def test_the_calculation_date_is_the_day_before_the_episode():
    assert calculation_date(date(2023, 6, 1)) == date(2023, 5, 31)


def test_the_shipped_episodes_fit_the_shocked_variables():
    from mst.macro.registry import load_registry
    from mst.plugins.bvar.signs import load_signs

    plan = load_episodes()
    variables = {v.id for v in load_registry().variables}
    shocks = set(load_signs().ids)
    for check in plan.episodes:
        assert set(check.constrained) <= variables, check.id
        assert set(check.driving) <= shocks, check.id
