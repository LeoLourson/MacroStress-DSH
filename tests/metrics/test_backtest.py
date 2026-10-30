"""The episode backtest: the runner over fake models, the cache, the gates and the report."""

import json
from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest

from mst import models as models_pkg
from mst.data.store import Observation, SeriesStore
from mst.drivers.registry import Quantity
from mst.macro.panel import months_between, seasonal_profile, shift
from mst.macro.registry import load_registry
from mst.metrics import backtest as runner
from mst.metrics.baselines import DOMAIN, WIDTH
from mst.metrics.cells import Cell
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


def run_episodes(workspace, corpus, tmp_path, *, name="details.json", only=(), **extra):
    """Run the backtest and read its details file."""
    details = tmp_path / name
    code = runner.run(
        store=workspace["store"].root,
        models=workspace["models"],
        corpus=corpus,
        as_of=KNOWN_ON,
        only=list(only),
        details=details,
        **extra,
    )
    return code, json.loads(details.read_text(encoding="utf-8"))


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

    code, report = run_episodes(workspace, corpus, tmp_path)

    assert code == 0
    assert report["problems"] == []
    assert len(report["episodes"]) == 1
    row = report["quantities"]["ru.key_rate"]
    assert row["scored"] > 0 and row["crps"] > 0
    assert row["baseline_crps"] is not None and row["base_crps"] is not None
    assert row["calibration"]["shares"] and row["sharpness"] > 0
    assert row["admission"]["scored"] == row["scored"]
    assert row["admission"][DOMAIN] == 0 and row["admission"][WIDTH] == 0
    assert row["intervals"]["crps"] is not None
    assert row["intervals"]["normalized"] is not None
    assert row["intervals"]["vs_baseline"] is not None
    # The constrained variable keeps its cells, marked as constrained.
    constrained = report["quantities"]["gl.brent"]
    assert constrained["constrained"] > 0


def test_the_random_walk_is_not_the_base_run(workspace, tmp_path, monkeypatch):
    corpus = manifest(tmp_path, [("R1", date(2023, 6, 1))])
    plan = plan_file(tmp_path, [("R1", "gl.brent", "fake.demand")])
    monkeypatch.setattr(runner, "load_episodes", lambda path=None: load_episodes(plan))

    _, report = run_episodes(workspace, corpus, tmp_path)

    row = report["quantities"]["ru.key_rate"]
    # The fake model ignores the scenario, so its base run is its conditional forecast.
    assert row["base_crps"] == pytest.approx(row["crps"])
    # The random walk is a different benchmark, and it is the one labelled as such.
    assert row["baseline_crps"] != pytest.approx(row["crps"], rel=1e-9)
    # The conditional-minus-base comparison of the scheme is reported apart from the RW one.
    assert row["base_normalized_crps"] == pytest.approx(row["normalized_crps"])
    assert row["baseline_normalized_crps"] != pytest.approx(
        row["normalized_crps"], rel=1e-9
    )


def test_an_episode_scores_the_same_alone_and_with_others(workspace, tmp_path, monkeypatch):
    corpus = manifest(tmp_path, [("R1", date(2023, 6, 1)), ("R2", date(2024, 3, 1))])
    plan = plan_file(
        tmp_path, [("R1", "gl.brent", "fake.demand"), ("R2", "ru.key_rate", "fake.rate")]
    )
    monkeypatch.setattr(runner, "load_episodes", lambda path=None: load_episodes(plan))

    _, alone = run_episodes(workspace, corpus, tmp_path, name="alone.json", only=["R1"])
    _, together = run_episodes(workspace, corpus, tmp_path, name="together.json")

    def cells_of(report):
        return {
            quantity: [row for row in data["months"] if row["episode"] == "R1"]
            for quantity, data in report["quantities"].items()
        }

    # The normaliser, the history range and the random walk all come from R1's own calculation
    # date, so loading R2 changes nothing about R1's cells.
    assert cells_of(alone) == cells_of(together)
    assert cells_of(alone)["ru.key_rate"]


def test_the_default_macro_domain_rejects_a_band_below_zero():
    registry = load_registry()
    assert runner._macro_domain(registry.variable("ru.key_rate")) == (0.0, np.inf)
    cell = Cell(
        "e",
        1,
        "ru.key_rate",
        0.0,
        np.array([-10.0, -2.0, 0.0, 1.0]),
        sigma=1.0,
        history_range=100.0,
        domain=runner._macro_domain(registry.variable("ru.key_rate")),
    )
    admission = runner._admission([cell])
    assert admission["rules"][DOMAIN]["cells"] == 1
    assert admission["rules"][DOMAIN]["failed"] is True
    assert admission["rules"][WIDTH]["cells"] == 0


def test_the_driver_domain_maps_the_natural_endpoints():
    def quantity(**fields):
        return Quantity(
            id="q", label="q", unit="u", regressors=["ru.cpi"], lags=1, **fields
        )

    assert runner._model_domain(quantity(transform="log100")) == (0.0, np.inf)
    assert runner._model_domain(quantity(transform="log100", domain=[0.0, 1e6]))[0] == 0.0
    assert runner._model_domain(quantity(transform="log100", domain=[0.0, 1e6]))[1] == (
        pytest.approx(100 * np.log(1e6))
    )
    # A logit declared on [0, 1] is unbounded in the model scale, not silently "no gate".
    assert runner._model_domain(quantity(transform="logit", domain=[0.0, 1.0])) is None
    low, high = runner._model_domain(quantity(transform="logit", domain=[0.01, 0.99]))
    assert low == pytest.approx(np.log(0.01 / 0.99))
    assert high == pytest.approx(np.log(0.99 / 0.01))
    assert runner._model_domain(quantity(transform="identity", domain=[0.0, 100.0])) == (
        0.0,
        100.0,
    )


def test_the_admission_shares_are_pooled_per_rule():
    def cell(month, draws, quantity="q"):
        return Cell(
            "e",
            month,
            quantity,
            0.0,
            np.asarray(draws, dtype=float),
            sigma=1.0,
            history_range=10.0,
            domain=(0.0, np.inf),
        )

    domain_cells = [cell(month, [-2.0, -1.0, 0.0, 1.0]) for month in range(1, 5)]
    width_cells = [cell(month, [0.0, 0.0, 100.0, 100.0]) for month in range(1, 5)]
    fine = [cell(month, [1.0, 2.0, 3.0, 4.0]) for month in range(1, 93)]
    # Four cells break each rule on its own: the union is 8 %, but neither rule reaches 5 %.
    admission = runner._admission(domain_cells + width_cells + fine)
    assert admission["cells"] == 100
    assert admission["rules"][DOMAIN] == {
        "cells": 4,
        "share": pytest.approx(0.04),
        "failed": False,
        "episodes": ["e"],
    }
    assert admission["rules"][WIDTH]["share"] == pytest.approx(0.04)
    assert admission["failed"] is False
    # Six cells breaking the width rule fail that rule and the version.
    extra = [cell(month, [0.0, 0.0, 100.0, 100.0]) for month in range(1, 3)]
    failed = runner._admission(domain_cells + width_cells + extra + fine)
    assert failed["rules"][WIDTH]["share"] == pytest.approx(6 / 102)
    assert failed["rules"][WIDTH]["failed"] is True
    assert failed["rules"][DOMAIN]["failed"] is False
    assert failed["failed"] is True
    # The threshold is strict: exactly five percent does not pass ("fewer than 5 %").
    exact = [cell(month, [0.0, 0.0, 100.0, 100.0]) for month in range(1, 6)]
    enough = [cell(month, [1.0, 2.0, 3.0, 4.0]) for month in range(1, 96)]
    threshold = runner._admission(exact + enough)
    assert threshold["cells"] == 100 and threshold["rules"][WIDTH]["share"] == 0.05
    assert threshold["rules"][WIDTH]["failed"] is True and threshold["failed"] is True


def test_the_constraint_gate_needs_more_than_one_percent():
    shares = {
        "R1": {0: None, 1: 0.02},
        "R2": {0: 0.005},
    }
    gate = runner._constraint_gate(shares)
    assert gate["applicable"] == 2
    assert gate["not_applicable"] == 1
    assert gate["failed_constraints"] == ["R2#0"]
    assert gate["failed"] is True
    assert runner._constraint_gate({"R1": {0: 0.02, 1: None}})["failed"] is False
    # The gate is strict: exactly one percent is not "above 1 %".
    assert runner._constraint_gate({"R1": {0: 0.01}})["failed"] is True


def test_a_fact_episode_reports_the_constraint_probability_as_not_applicable(
    workspace, tmp_path, monkeypatch, capsys
):
    corpus = manifest(tmp_path, [("R1", date(2023, 6, 1))])
    plan = plan_file(tmp_path, [("R1", "gl.brent", "fake.demand")])
    monkeypatch.setattr(runner, "load_episodes", lambda path=None: load_episodes(plan))

    code, report = run_episodes(workspace, corpus, tmp_path)
    out = capsys.readouterr().out

    assert code == 0
    row = report["episodes"][0]
    assert row["constraint_probabilities"]
    assert all(share is None for share in row["constraint_probabilities"].values())
    gate = report["constraint_gate"]
    assert gate["applicable"] == 0
    assert gate["not_applicable"] == len(row["constraint_probabilities"])
    assert gate["failed"] is False
    # Not a meaningless 0.00: the episode table says the probability is not applicable.
    episode_line = next(line for line in out.splitlines() if line.startswith("| R1 |"))
    assert "n/a" in episode_line and "0.00" not in episode_line


def test_the_interval_resamples_the_reported_statistic():
    rows = [
        {"episode": "e1", "crps": 8.0},
        {"episode": "e2", "crps": 1.0},
        {"episode": "e2", "crps": 1.0},
        {"episode": "e3", "crps": 1.0},
        {"episode": "e3", "crps": 1.0},
    ]
    plan = SimpleNamespace(seed=3)
    interval = runner._interval(rows, "crps", plan)
    assert interval is not None
    low, high = interval
    assert low < 12 / 5 < high
    assert runner._interval(rows, "crps", plan) == interval
    assert runner._interval([], "crps", plan) is None


def test_the_report_meta_records_the_versions_and_the_vintage(workspace, tmp_path, monkeypatch):
    corpus = manifest(tmp_path, [("R1", date(2023, 6, 1))])
    plan = plan_file(tmp_path, [("R1", "gl.brent", "fake.demand")])
    monkeypatch.setattr(runner, "load_episodes", lambda path=None: load_episodes(plan))

    _, report = run_episodes(workspace, corpus, tmp_path)
    meta = report["meta"]

    assert meta["draws"] == 40 and meta["seed"] == 3
    assert meta["known_on"] == "2030-01-01"
    assert meta["episodes_toml"] == runner._file_digest(plan)
    assert len(meta["macro_registry"]) == 64
    assert meta["macro_models"] == [{"model_id": "fake.macro", "version": "0.1.0"}]
    assert meta["macro_versions"] == [report["episodes"][0]["macro_version"]]
    assert meta["store"] and meta["models"] and meta["backtest_root"]
    assert meta["git_commit"] is None or len(meta["git_commit"]) == 40
    assert any("equal-weight" in item for item in meta["missing"])
    assert any("Acerbi" in item for item in meta["missing"])
    assert any("wild cluster" in item for item in meta["missing"])
    assert any("halves" in item for item in meta["missing"])


def test_the_cache_key_covers_the_model_configuration(workspace, tmp_path, monkeypatch):
    config = tmp_path / "spec.toml"
    config.write_text("tightness = 0.2\n", encoding="utf-8")
    monkeypatch.setattr(runner, "_configuration_paths", lambda plugin: [config])
    plugin, data, as_of = workspace["macro"], workspace["store"], date(2023, 5, 31)
    registry = load_registry()

    first = runner._data_key(plugin, data, registry, as_of)
    assert runner._data_key(plugin, data, registry, as_of) == first

    config.write_text("tightness = 0.4\n", encoding="utf-8")
    assert runner._data_key(plugin, data, registry, as_of) != first


def test_the_configuration_paths_include_the_spec_and_the_signs():
    from mst.plugins.bvar.plugin import MODEL

    names = {path.name for path in runner._configuration_paths(MODEL)}
    assert {"spec.toml", "signs.toml", "registry.toml"} <= names


def test_seasonal_facts_use_the_calculation_date_profile(tmp_path):
    from mst.macro.registry import Registry, Variable

    variable = Variable(
        id="ru.cpi",
        label="cpi",
        unit="index",
        transform="log100",
        source="test",
        seasonal=True,
    )
    registry = Registry(variables=[variable])
    store = SeriesStore(tmp_path / "series")
    months = months_between(date(2019, 1, 1), date(2020, 12, 1))
    factor = np.array([1.0, 1.05, 1.10, 1.15, 1.20, 1.10, 1.0, 0.95, 1.10, 1.05, 1.0, 0.90])
    original = {
        month: float(100.0 * 1.01**t * factor[month.month - 1])
        for t, month in enumerate(months)
    }
    revised = {
        date(2019, 6, 1): original[date(2019, 6, 1)] * 1.5,
        date(2019, 7, 1): original[date(2019, 7, 1)] * 0.5,
    }
    rows = [
        Observation(month, value, date(2021, 1, 10), False)
        for month, value in original.items()
    ]
    rows += [
        Observation(month, value, date(2022, 6, 1), False)
        for month, value in revised.items()
    ]
    horizon = months_between(date(2021, 1, 1), date(2021, 2, 1))
    # The facts of the horizon are published after the calculation date, before known_on.
    facts = {
        month: float(100.0 * 1.01 ** (24 + t) * factor[month.month - 1])
        for t, month in enumerate(horizon)
    }
    rows += [
        Observation(month, value, date(2021, 3 + t, 10), False)
        for t, (month, value) in enumerate(facts.items())
    ]
    store.write("ru.cpi", rows)
    plan = SimpleNamespace(exclusions=[])
    panel = SimpleNamespace(ids=["ru.cpi"])

    result = runner._realized(
        store,
        registry,
        panel,
        months,
        horizon,
        estimated_on=date(2021, 1, 15),
        known_on=date(2023, 1, 1),
        plan=plan,
    )

    known = {**original, **revised, **facts}
    calendar = [month.month - 1 for month in horizon]
    calculated = seasonal_profile(
        np.array([100 * np.log(original[month]) for month in months]), months
    )
    expected = np.array([100 * np.log(known[month]) for month in horizon]) - calculated[calendar]
    assert np.allclose(result[:, 0], expected)
    # A profile re-estimated from today's revised data would give a different scale.
    today = seasonal_profile(
        np.array([100 * np.log(known[month]) for month in months]), months
    )
    alternative = np.array([100 * np.log(known[month]) for month in horizon]) - today[calendar]
    assert not np.allclose(result[:, 0], alternative)


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

    code, report = run_episodes(
        workspace, corpus, tmp_path, drivers=catalog, satellites=True
    )

    assert code == 0
    row = report["quantities"]["share_price:sber"]
    assert row["level"] == "driver/share_price"
    assert row["scored"] > 0
    # The naive forecast of a target is scored alongside the random walk; the base run has no
    # driver paths, so its cell score is absent rather than copied from the model.
    assert row["naive_crps"] is not None
    assert row["baseline_crps"] is not None
    assert row["base_crps"] is None


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
