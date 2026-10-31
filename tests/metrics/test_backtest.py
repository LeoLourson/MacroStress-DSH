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


@pytest.fixture(autouse=True)
def fast_intervals(monkeypatch):
    """A hundred repetitions keep the tests quick; the report uses a thousand."""
    monkeypatch.setattr(runner, "INTERVAL_REPETITIONS", 100)


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


def plan_file(tmp_path, episodes, *, swap=False):
    """An episodes.toml with the named checks: one constrained variable, one driving shock.

    The halves alternate tuning and validation by position, so every plan this helper writes
    has both non-empty — the plan contract requires it; ``swap`` flips the marks, for the test
    that the full-sample table does not depend on them.
    """
    text = [
        "draws = 40",
        "seed = 3",
        "after_months = 6",
        "",
        "",
    ]
    for number, (id_, variable, shock) in enumerate(episodes):
        tuning = number % 2 == 0
        half = "tuning" if tuning != swap else "validation"
        text += [
            "[[episodes]]",
            f'id = "{id_}"',
            f'half = "{half}"',
            'source = "manual"',
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
    corpus = manifest(tmp_path, [("R1", date(2023, 6, 1)), ("R2", date(2024, 3, 1))])
    plan = plan_file(
        tmp_path, [("R1", "gl.brent", "fake.demand"), ("R2", "ru.key_rate", "fake.rate")]
    )
    monkeypatch.setattr(runner, "load_episodes", lambda path=None: load_episodes(plan))
    arguments = dict(
        store=workspace["store"].root,
        models=workspace["models"],
        corpus=corpus,
        as_of=KNOWN_ON,
    )

    assert runner.run(**arguments) == 0
    estimated_once = len(workspace["macro"].calls)
    assert estimated_once == 2  # one version per episode calculation date

    assert runner.run(**arguments) == 0
    assert len(workspace["macro"].calls) == estimated_once  # the cache answered
    assert (workspace["models"] / "backtest" / "macro").is_dir()


def test_an_unknown_driving_shock_is_reported_not_raised(workspace, tmp_path, monkeypatch, capsys):
    corpus = manifest(tmp_path, [("R1", date(2023, 6, 1)), ("R2", date(2024, 3, 1))])
    plan = plan_file(
        tmp_path, [("R1", "gl.brent", "no.such.shock"), ("R2", "ru.key_rate", "fake.rate")]
    )
    monkeypatch.setattr(runner, "load_episodes", lambda path=None: load_episodes(plan))

    code = runner.run(
        store=workspace["store"].root, models=workspace["models"], corpus=corpus, as_of=KNOWN_ON
    )

    assert code == 1
    assert "the macro model knows no shocks no.such.shock" in capsys.readouterr().out


def test_an_episode_missing_from_the_manifest_is_reported(workspace, tmp_path, monkeypatch, capsys):
    corpus = manifest(tmp_path, [("R2", date(2024, 3, 1))])
    plan = plan_file(
        tmp_path, [("R9", "gl.brent", "fake.demand"), ("R2", "ru.key_rate", "fake.rate")]
    )
    monkeypatch.setattr(runner, "load_episodes", lambda path=None: load_episodes(plan))

    code = runner.run(
        store=workspace["store"].root, models=workspace["models"], corpus=corpus, as_of=KNOWN_ON
    )

    assert code == 1
    assert "R9: not in the corpus manifest" in capsys.readouterr().out


def test_a_run_without_a_single_episode_reports_it_instead_of_raising(
    workspace, tmp_path, monkeypatch, capsys
):
    corpus = manifest(tmp_path, [("R1", date(2023, 6, 1)), ("R2", date(2024, 3, 1))])
    # Unknown driving shocks drop both episodes: the report has no scored cells at all.
    plan = plan_file(
        tmp_path, [("R1", "gl.brent", "fake.one"), ("R2", "ru.key_rate", "fake.two")]
    )
    monkeypatch.setattr(runner, "load_episodes", lambda path=None: load_episodes(plan))

    code, report = run_episodes(workspace, corpus, tmp_path)

    assert code == 1
    assert report["episodes"] == []
    assert report["aggregate"]["levels"] == {}
    assert "no scored cells" in report["aggregate"]["note"]
    assert report["problems"]
    assert "No episode could be judged" in capsys.readouterr().out


def test_the_monte_carlo_option_moves_no_other_field(workspace, tmp_path, monkeypatch):
    corpus = manifest(tmp_path, [("R1", date(2023, 6, 1)), ("R2", date(2024, 3, 1))])
    plan = plan_file(
        tmp_path, [("R1", "gl.brent", "fake.demand"), ("R2", "ru.key_rate", "fake.rate")]
    )
    monkeypatch.setattr(runner, "load_episodes", lambda path=None: load_episodes(plan))

    _, plain = run_episodes(workspace, corpus, tmp_path, name="plain.json")
    _, monte_carlo = run_episodes(
        workspace, corpus, tmp_path, name="mc.json", mc_bootstrap=True
    )

    # The Monte-Carlo error is reported beside the sampling error, never in place of it.
    for key in ("episodes", "quantities", "admission", "constraint_gate", "halves"):
        assert plain[key] == monte_carlo[key]
    assert "monte_carlo" not in plain["aggregate"]["intervals"]
    interval = monte_carlo["aggregate"]["intervals"]["monte_carlo"]
    assert interval["note"].startswith("bootstrap over the draws")
    assert interval["interval"] is not None


def test_the_details_file_holds_the_full_report(workspace, tmp_path, monkeypatch):
    corpus = manifest(tmp_path, [("R1", date(2023, 6, 1)), ("R2", date(2024, 3, 1))])
    plan = plan_file(
        tmp_path, [("R1", "gl.brent", "fake.demand"), ("R2", "ru.key_rate", "fake.rate")]
    )
    monkeypatch.setattr(runner, "load_episodes", lambda path=None: load_episodes(plan))

    code, report = run_episodes(workspace, corpus, tmp_path)

    assert code == 0
    assert report["problems"] == []
    assert len(report["episodes"]) == 2
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


def test_the_report_holds_the_aggregate_the_usefulness_and_the_halves(
    workspace, tmp_path, monkeypatch, capsys
):
    corpus = manifest(tmp_path, [("R1", date(2023, 6, 1)), ("R2", date(2024, 3, 1))])
    plan = plan_file(
        tmp_path, [("R1", "gl.brent", "fake.demand"), ("R2", "ru.key_rate", "fake.rate")]
    )
    monkeypatch.setattr(runner, "load_episodes", lambda path=None: load_episodes(plan))

    _, report = run_episodes(workspace, corpus, tmp_path)
    out = capsys.readouterr().out

    aggregate = report["aggregate"]
    assert report["target_level"] == "macro"
    macro = aggregate["levels"]["macro"]
    assert macro["by_horizon"]
    assert macro["overall"]["model"] is not None
    assert macro["counters"]["model"]["cells"] >= macro["counters"]["model"]["normalized"]
    sampling = aggregate["intervals"]["sampling"]["all"]
    assert sampling["model"]["pair"] is not None
    assert sampling["model"]["wild"] is not None
    # Two episodes: the ordinary bootstrap understates the interval, the wild one is primary.
    assert sampling["primary"] == "wild"
    assert "the wild one is read as primary" in out
    assert report["halves"]["all"]["episodes"] == ["R1", "R2"]
    assert report["halves"]["tuning"]["episodes"] == ["R1"]
    assert report["halves"]["validation"]["episodes"] == ["R2"]
    assert report["halves"]["tuning"]["admission"]["cells"] > 0
    assert aggregate["utility"]["by_source"]["manual"]["target"]["episodes"] == 2
    # The tables are printed, not only written to the JSON.
    assert "## Aggregate (equal weights across quantities; target level `macro`)" in out
    assert "## Scenario usefulness" in out
    assert "## Halves" in out
    assert "| macro | all |" in out


def test_the_full_table_does_not_depend_on_the_halves(workspace, tmp_path, monkeypatch):
    corpus = manifest(tmp_path, [("R1", date(2023, 6, 1)), ("R2", date(2024, 3, 1))])
    checks = [("R1", "gl.brent", "fake.demand"), ("R2", "ru.key_rate", "fake.rate")]
    plan = plan_file(tmp_path, checks)
    monkeypatch.setattr(runner, "load_episodes", lambda path=None: load_episodes(plan))
    _, first = run_episodes(workspace, corpus, tmp_path, name="first.json")

    swapped = plan_file(tmp_path, checks, swap=True)
    monkeypatch.setattr(runner, "load_episodes", lambda path=None: load_episodes(swapped))
    _, second = run_episodes(workspace, corpus, tmp_path, name="second.json")

    # The final table is built over all episodes: the marks change the halves, nothing else.
    assert first["aggregate"]["levels"] == second["aggregate"]["levels"]
    assert first["aggregate"]["rows"] == second["aggregate"]["rows"]
    assert first["quantities"] == second["quantities"]
    assert first["halves"]["tuning"]["episodes"] == second["halves"]["validation"]["episodes"]


def test_the_decision_section_prints_the_steps(capsys):
    report = {
        "decision": {
            "target_level": "macro",
            "half": "validation",
            "cells": 12,
            "cells_not_compared": 1,
            "candidate_crps": 0.9,
            "benchmark_crps": 1.1,
            "incumbent_crps": 1.0,
            "se_vs_benchmark": 0.05,
            "se_vs_incumbent": 0.04,
            "diff_vs_baseline": -0.2,
            "pair_interval": (-0.3, -0.1),
            "wild_interval": (-0.31, -0.09),
            "wild_disagrees": False,
            "admission_shares": {DOMAIN: 0.0, WIDTH: 0.01},
            "admission_grown": [],
            "primary": "wild",
            "verdict": "accepted",
            "step": 2,
            "note": "better by more than two SE",
            "notes": ["a single episode reads as zero"],
        }
    }
    runner._print_decision(report)
    out = capsys.readouterr().out
    assert "## Decision (two standard errors)" in out
    assert "**accepted** (step 2)" in out
    assert "12 paired cells, 1 not compared" in out
    assert "fewer than 8 episodes: the wild interval is read as primary" in out
    assert "a single episode reads as zero" in out


def test_the_tail_section_prints_the_horizon_and_the_dropped_episodes(capsys):
    report = {
        "tail": {
            "z": 0.4,
            "interval": (0.1, 0.7),
            "wild": (0.05, 0.75),
            "wild_disagrees": False,
            "trimmed_z": 0.3,
            "trim": 0.01,
            "episodes": [{"id": "R1"}],
            "dropped": [{"id": "R2", "reason": "no fact at t = H"}],
            "horizon": {"kind": "episode window", "month": 6, "note": ""},
        }
    }
    runner._print_tail(report)
    out = capsys.readouterr().out
    assert "## Tail (Acerbi-Székely)" in out
    assert "month 6" in out
    assert "dropped R2: no fact at t = H" in out


def test_the_random_walk_is_not_the_base_run(workspace, tmp_path, monkeypatch):
    corpus = manifest(tmp_path, [("R1", date(2023, 6, 1)), ("R2", date(2024, 3, 1))])
    plan = plan_file(
        tmp_path, [("R1", "gl.brent", "fake.demand"), ("R2", "ru.key_rate", "fake.rate")]
    )
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
    corpus = manifest(tmp_path, [("R1", date(2023, 6, 1)), ("R2", date(2024, 3, 1))])
    plan = plan_file(
        tmp_path, [("R1", "gl.brent", "fake.demand"), ("R2", "ru.key_rate", "fake.rate")]
    )
    monkeypatch.setattr(runner, "load_episodes", lambda path=None: load_episodes(plan))

    code, report = run_episodes(workspace, corpus, tmp_path)
    out = capsys.readouterr().out

    assert code == 0
    row = report["episodes"][0]
    assert row["constraint_probabilities"]
    assert all(share is None for share in row["constraint_probabilities"].values())
    gate = report["constraint_gate"]
    assert gate["applicable"] == 0
    total_constraints = sum(len(e["constraint_probabilities"]) for e in report["episodes"])
    assert gate["not_applicable"] == total_constraints
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
    corpus = manifest(tmp_path, [("R1", date(2023, 6, 1)), ("R2", date(2024, 3, 1))])
    plan = plan_file(
        tmp_path, [("R1", "gl.brent", "fake.demand"), ("R2", "ru.key_rate", "fake.rate")]
    )
    monkeypatch.setattr(runner, "load_episodes", lambda path=None: load_episodes(plan))

    _, report = run_episodes(workspace, corpus, tmp_path)
    meta = report["meta"]

    assert meta["draws"] == 40 and meta["seed"] == 3
    assert meta["known_on"] == "2030-01-01"
    assert meta["episodes_toml"] == runner._file_digest(plan)
    assert len(meta["macro_registry"]) == 64
    assert meta["target_level"] == "macro"
    assert meta["model_sdk"] == runner.MODEL_SDK_VERSION
    assert meta["assumptions"]
    assert meta["macro_models"] == [{"model_id": "fake.macro", "version": "0.1.0"}]
    # Two episodes at two calculation dates: two estimated versions, unless one is cached.
    assert len(meta["macro_versions"]) == 2
    assert meta["store"] and meta["models"] and meta["backtest_root"]
    assert meta["git_commit"] is None or len(meta["git_commit"]) == 40
    # Phases 1-3 are implemented: nothing of section 5 is named as missing any more, and the
    # tail is reported on the portfolio when one is given.
    assert meta["missing"] == []


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
    corpus = manifest(tmp_path, [("R1", date(2023, 6, 1)), ("R2", date(2024, 3, 1))])
    plan = plan_file(
        tmp_path, [("R1", "gl.brent", "fake.demand"), ("R2", "ru.key_rate", "fake.rate")]
    )
    monkeypatch.setattr(runner, "load_episodes", lambda path=None: load_episodes(plan))
    # The driver catalog of the workspace: one object, one driver, with a realized series.
    catalog = _catalog(tmp_path)
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
    # The fake part records no training range: the flag says the whole column was used.
    assert row["legacy_training_range"] is True


def test_the_portfolio_tail_is_reported_with_its_horizon(workspace, tmp_path, monkeypatch, capsys):
    corpus = manifest(tmp_path, [("R1", date(2023, 6, 1)), ("R2", date(2024, 3, 1))])
    plan = plan_file(
        tmp_path, [("R1", "gl.brent", "fake.demand"), ("R2", "ru.key_rate", "fake.rate")]
    )
    monkeypatch.setattr(runner, "load_episodes", lambda path=None: load_episodes(plan))
    catalog = _catalog(tmp_path)
    store = workspace["store"]
    grid = months_between(simulated.START, shift(simulated.START, 59))
    store.write(
        "drv.share_price.sber",
        [
            Observation(month, float(5.0 + 0.01 * t), month, False)
            for t, month in enumerate(grid)
        ],
    )
    from mst.portfolio import tool as portfolio_tool

    portfolio_tool.save(
        tmp_path / "portfolios",
        "tail-test",
        {
            "name": "tail",
            "base_currency": "RUB",
            "positions": [
                {"id": "sber", "type": "common_stock", "notional": 10.0, "codes": {"secid": "SBER"}}
            ],
        },
    )

    _, report = run_episodes(
        workspace,
        corpus,
        tmp_path,
        name="tail.json",
        drivers=catalog,
        satellites=True,
        portfolio="tail-test",
    )

    tail = report["tail"]
    assert tail["horizon"]["kind"] == "episode window"
    assert tail["horizon"]["month"] == report["episodes"][0]["horizon"]
    assert len(tail["episodes"]) + len(tail["dropped"]) == 2
    assert "## Tail (Acerbi-Székely)" in capsys.readouterr().out


def test_an_unknown_portfolio_is_a_named_error(workspace, tmp_path, monkeypatch):
    corpus = manifest(tmp_path, [("R1", date(2023, 6, 1)), ("R2", date(2024, 3, 1))])
    plan = plan_file(
        tmp_path, [("R1", "gl.brent", "fake.demand"), ("R2", "ru.key_rate", "fake.rate")]
    )
    monkeypatch.setattr(runner, "load_episodes", lambda path=None: load_episodes(plan))

    with pytest.raises(ValueError, match="no portfolio 'absent'"):
        run_episodes(workspace, corpus, tmp_path, portfolio="absent")


def test_the_decision_compares_two_reports_on_the_same_cells(workspace, tmp_path, monkeypatch):
    corpus = manifest(tmp_path, [("R1", date(2023, 6, 1)), ("R2", date(2024, 3, 1))])
    plan = plan_file(
        tmp_path, [("R1", "gl.brent", "fake.demand"), ("R2", "ru.key_rate", "fake.rate")]
    )
    monkeypatch.setattr(runner, "load_episodes", lambda path=None: load_episodes(plan))
    _, report = run_episodes(workspace, corpus, tmp_path, name="incumbent.json")
    against = tmp_path / "incumbent.json"
    halves = {episode["id"]: episode["half"] for episode in report["episodes"]}

    def decide_on(path):
        return runner._decision_block(
            report["aggregate"]["rows"],
            path,
            admission=report["admission"],
            halves=halves,
            plan=load_episodes(plan),
            target_level="macro",
        )

    # The candidate against itself: no difference from the incumbent is found, and the verdict
    # repeats. The random walk here is better than the fake model, so step 3 may reject it.
    verdict = decide_on(against)
    assert verdict["candidate_crps"] == pytest.approx(verdict["incumbent_crps"])
    assert verdict["diff_vs_baseline"] == pytest.approx(
        verdict["candidate_crps"] - verdict["benchmark_crps"]
    )
    assert verdict["verdict"] in {"rejected_worse", "inconclusive"}
    # The verdict is spoken on the validation half, naming it.
    assert verdict["half"] == "validation"
    assert verdict["cells"] > 0
    assert verdict["cells_not_compared"] == 0
    assert verdict["primary"] == "wild"
    assert verdict["wild_interval"] is not None
    assert decide_on(against) == verdict

    # A cell the two reports do not share is an error naming the missing cell, not a silent
    # comparison on the intersection.
    broken = json.loads(against.read_text(encoding="utf-8"))
    dropped = broken["aggregate"]["rows"].pop()
    file = tmp_path / "broken.json"
    file.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="do not share cells") as error:
        decide_on(file)
    assert f"{dropped['episode']}/{dropped['quantity']}" in str(error.value)


def test_the_satellite_bundle_is_cached_by_its_own_inputs(workspace, tmp_path, monkeypatch):
    corpus = manifest(tmp_path, [("R1", date(2023, 6, 1)), ("R2", date(2024, 3, 1))])
    plan = plan_file(
        tmp_path, [("R1", "gl.brent", "fake.demand"), ("R2", "ru.key_rate", "fake.rate")]
    )
    monkeypatch.setattr(runner, "load_episodes", lambda path=None: load_episodes(plan))
    catalog = _catalog(tmp_path)
    store = workspace["store"]
    grid = months_between(simulated.START, shift(simulated.START, 59))
    store.write(
        "drv.share_price.sber",
        [
            Observation(month, float(5.0 + 0.01 * t), month, False)
            for t, month in enumerate(grid)
        ],
    )
    calls = []
    original = models_pkg.estimate_satellites

    def counted(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(models_pkg, "estimate_satellites", counted)
    _, first = run_episodes(
        workspace, corpus, tmp_path, name="first.json", drivers=catalog, satellites=True
    )
    estimated = len(calls)
    # One bundle per episode: the episodes are estimated on different macro versions.
    assert estimated == 2
    root = workspace["models"] / "backtest"
    legacy = root / f"satellites-{first['meta']['macro_versions'][0]}"
    legacy.write_text("stale\n", encoding="utf-8")

    _, second = run_episodes(
        workspace, corpus, tmp_path, name="second.json", drivers=catalog, satellites=True
    )

    # The same inputs reuse both bundles: the estimation is not repeated.
    assert len(calls) == estimated
    assert first["meta"]["satellite_versions"] == second["meta"]["satellite_versions"]
    assert first["episodes"] == second["episodes"]
    # The pointer keyed by the macro version alone is gone: the new key is the digest.
    assert not legacy.exists()


def test_the_satellite_key_covers_the_inputs_that_shape_the_bundle(
    workspace, tmp_path, monkeypatch
):
    from mst.drivers.registry import load_registry as load_drivers

    catalog = _catalog(tmp_path)
    store = workspace["store"]
    grid = months_between(simulated.START, shift(simulated.START, 59))
    series = [
        Observation(month, float(5.0 + 0.01 * t), month, False)
        for t, month in enumerate(grid)
    ]
    store.write("drv.share_price.sber", series)
    registry = load_drivers(catalog=catalog)
    installed = {"fake.sat": workspace["satellite"]}
    episode = SimpleNamespace(macro=SimpleNamespace(name="macro-1"), as_of=date(2024, 1, 31))

    def key():
        return runner._satellite_key(episode, store, catalog, installed, registry)

    baseline = key()
    assert key() == baseline
    # Another macro version, another catalog, another series, another plugin version or
    # another assignment is another key — a stale bundle is never reused.
    other = SimpleNamespace(macro=SimpleNamespace(name="macro-2"), as_of=episode.as_of)
    assert runner._satellite_key(other, store, catalog, installed, registry) != baseline

    catalog.write_bytes(catalog.read_bytes() + b" ")
    assert key() != baseline
    catalog.write_bytes(catalog.read_bytes()[:-1])
    assert key() == baseline

    store.write(
        "drv.share_price.sber",
        [Observation(row.month, row.value + 1.0, row.published, row.approximate) for row in series],
    )
    assert key() != baseline
    store.write("drv.share_price.sber", series)
    assert key() == baseline

    with monkeypatch.context() as patch:
        patch.setattr(type(workspace["satellite"]), "version", "0.2.0")
        assert key() != baseline
    with monkeypatch.context() as patch:
        patch.setattr(
            models_pkg,
            "assignment",
            lambda quantities, overrides=None: ({"share_price": "other.model"}, []),
        )
        assert key() != baseline


def _catalog(tmp_path):
    """The driver catalog of the tests: one quantity, one object, one driver."""
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
    return catalog


def test_an_episode_the_satellite_estimation_cannot_serve_is_skipped_by_name(
    workspace, tmp_path, monkeypatch
):
    corpus = manifest(tmp_path, [("R1", date(2023, 6, 1)), ("R2", date(2024, 3, 1))])
    plan = plan_file(
        tmp_path, [("R1", "gl.brent", "fake.demand"), ("R2", "ru.key_rate", "fake.rate")]
    )
    monkeypatch.setattr(runner, "load_episodes", lambda path=None: load_episodes(plan))
    catalog = _catalog(tmp_path)
    original = models_pkg.estimate_satellites
    calls = []

    def failing(*args, **kwargs):
        calls.append(args[0])
        if len(calls) == 1:
            raise ValueError(
                "share_price:it: series drv.share_price.it has no value known on 2023-05-31"
            )
        return original(*args, **kwargs)

    monkeypatch.setattr(models_pkg, "estimate_satellites", failing)

    code, report = run_episodes(
        workspace, corpus, tmp_path, name="late.json", drivers=catalog, satellites=True
    )

    # The first episode is named and skipped; the check goes on with the second.
    assert code == 1
    assert [episode["id"] for episode in report["episodes"]] == ["R2"]
    assert any(problem.startswith("R1: satellites:") for problem in report["problems"])


def test_the_calculation_date_is_the_day_before_the_episode():
    assert calculation_date(date(2023, 6, 1)) == date(2023, 5, 31)


def test_an_unmarked_or_misguided_plan_is_a_validation_error(tmp_path):
    from pydantic import ValidationError

    plan = plan_file(
        tmp_path, [("R1", "gl.brent", "fake.demand"), ("R2", "ru.key_rate", "fake.rate")]
    )
    text = plan.read_text(encoding="utf-8")

    def check(broken: str) -> str:
        file = tmp_path / "broken.toml"
        file.write_text(broken, encoding="utf-8")
        with pytest.raises(ValidationError):
            load_episodes(file)
        return broken

    # The half and the source of the scenario are marks of the plan, not optional fields.
    check(text.replace('half = "tuning"\n', ""))
    check(text.replace('source = "manual"\n', "", 1))
    check(text.replace('source = "manual"', 'source = "human"', 1))
    # A plan marked entirely as one half has nothing to check on.
    check(text.replace('half = "validation"', 'half = "tuning"'))


def test_the_shipped_episodes_fit_the_shocked_variables():
    from mst.macro.registry import load_registry
    from mst.plugins.bvar.signs import load_signs

    plan = load_episodes()
    variables = {v.id for v in load_registry().variables}
    shocks = set(load_signs().ids)
    for check in plan.episodes:
        assert set(check.constrained) <= variables, check.id
        assert set(check.driving) <= shocks, check.id
