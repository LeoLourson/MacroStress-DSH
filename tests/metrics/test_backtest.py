"""The episode backtest: the runner over fake models, the cache, the gates and the report."""

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mst import models as models_pkg
from mst.data.store import Observation, SeriesStore
from mst.drivers.registry import Quantity
from mst.macro.panel import months_between, seasonal_profile, shift
from mst.macro.registry import load_registry
from mst.metrics import backtest as runner
from mst.metrics.baselines import DOMAIN, WIDTH, macro_domain, quantity_domain
from mst.metrics.cells import Cell
from mst.metrics.episodes import calculation_date, load_episodes
from mst.metrics.sanity import check as sanity_check

from . import simulated

KNOWN_ON = date(2030, 1, 1)
STARTS = {"R1": "2023-06", "R2": "2024-03"}
"""The start months of the episodes the tests plan."""
PLAN = Path(__file__).resolve().parents[2] / "bench" / "model-backtest" / "episodes.toml"


@pytest.fixture(autouse=True)
def fast_intervals(monkeypatch):
    """A hundred repetitions keep the tests quick; the report uses a thousand."""
    monkeypatch.setattr(runner, "INTERVAL_REPETITIONS", 100)


def plan_file(tmp_path, episodes, *, swap=False):
    """An episodes.toml with the named checks: ``(id, observations, interventions)``.

    A string is one name. The halves alternate tuning and validation by position, so every plan
    this helper writes has both non-empty — the plan contract requires it; ``swap`` flips the
    marks, for the test that the full-sample table does not depend on them.
    """
    text = [
        "draws = 40",
        "seed = 3",
        "after_months = 6",
        "",
        "",
    ]
    for number, item in enumerate(episodes):
        id_, observations = item[0], item[1]
        interventions = item[2] if len(item) > 2 else []
        observed = [observations] if isinstance(observations, str) else list(observations)
        decided = [interventions] if isinstance(interventions, str) else list(interventions)
        tuning = number % 2 == 0
        half = "tuning" if tuning != swap else "validation"
        text += [
            "[[episodes]]",
            f'id = "{id_}"',
            f'start = "{STARTS[id_]}"',
            f'half = "{half}"',
            'source = "manual"',
            f"observations = {json.dumps(observed)}",
            f"interventions = {json.dumps(decided)}",
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
    """A store and the fake models installed."""
    store = simulated.series_store(tmp_path / "series")
    macro = simulated.FakeMacro()
    satellite = simulated.FakeSatellite()
    monkeypatch.setattr(models_pkg, "selected", lambda model_id=None: macro)
    monkeypatch.setattr(models_pkg, "satellites", lambda: {satellite.model_id: satellite})
    monkeypatch.setattr(
        models_pkg,
        "assignment",
        lambda quantities, overrides=None: (
            {quantity: satellite.model_id for quantity in quantities},
            [],
        ),
    )
    return {
        "store": store,
        "models": tmp_path / "models",
        "macro": macro,
        "satellite": satellite,
        "tmp": tmp_path,
    }


def run_episodes(workspace, plan, tmp_path, *, name="details.json", only=(), **extra):
    """Run the backtest and read its details file."""
    details = tmp_path / name
    code = runner.run(
        store=workspace["store"].root,
        models=workspace["models"],
        episodes=plan,
        as_of=KNOWN_ON,
        only=list(only),
        details=details,
        **extra,
    )
    return code, json.loads(details.read_text(encoding="utf-8"))


def test_the_runner_scores_the_cells_of_every_episode(workspace, tmp_path, capsys):
    plan = plan_file(tmp_path, [("R1", "gl.brent"), ("R2", "ru.key_rate")])

    code = runner.run(
        store=workspace["store"].root,
        models=workspace["models"],
        episodes=plan,
        as_of=KNOWN_ON,
    )

    assert code == 0
    out = capsys.readouterr().out
    assert "## Episodes" in out and "R1" in out and "R2" in out
    assert "## Quantities" in out and "ru.key_rate" in out
    # The constrained variable is reported but never scored.
    assert "gl.brent" in out


def test_the_episode_versions_are_cached_by_the_data(workspace, tmp_path):
    plan = plan_file(tmp_path, [("R1", "gl.brent"), ("R2", "ru.key_rate")])
    arguments = dict(
        store=workspace["store"].root,
        models=workspace["models"],
        episodes=plan,
        as_of=KNOWN_ON,
    )

    assert runner.run(**arguments) == 0
    estimated_once = len(workspace["macro"].calls)
    assert estimated_once == 2  # one version per episode calculation date

    assert runner.run(**arguments) == 0
    assert len(workspace["macro"].calls) == estimated_once  # the cache answered
    assert (workspace["models"] / "backtest" / "macro").is_dir()


def test_a_model_with_inputs_reads_the_answers_of_each_episode_date(
    workspace, tmp_path, monkeypatch
):
    from mst import inputs as answers

    class Answered(simulated.FakeMacro):
        def input_keys(self, as_of):
            return ("q1",)

        def estimate(self, context):
            assert context.inputs is not None, "the backtest passes the answers folder"
            assert answers.read(context.inputs, context.as_of)["q1"]
            return super().estimate(context)

    macro = Answered()
    monkeypatch.setattr(models_pkg, "selected", lambda model_id=None: macro)
    folder = tmp_path / "elicitation"
    dates = [calculation_date(date.fromisoformat(start + "-01")) for start in STARTS.values()]
    for as_of in dates:
        answers.write(folder, as_of, "q1", {"value": 1.0})
    arguments = dict(
        store=workspace["store"].root,
        models=workspace["models"],
        episodes=plan_file(tmp_path, [("R1", "gl.brent"), ("R2", "ru.key_rate")]),
        as_of=KNOWN_ON,
        inputs=folder,
    )

    assert runner.run(**arguments) == 0
    assert sorted(macro.calls) == dates
    assert runner.run(**arguments) == 0
    assert sorted(macro.calls) == dates, "the same answers reuse the cached versions"
    answers.write(folder, dates[0], "q1", {"value": 2.0})
    assert runner.run(**arguments) == 0
    assert sorted(macro.calls) == sorted([*dates, dates[0]]), "an edited answer is re-estimated"


def test_an_unknown_variable_is_reported_not_raised(workspace, tmp_path, capsys):
    plan = plan_file(tmp_path, [("R1", "no.such.variable"), ("R2", "ru.key_rate")])

    code = runner.run(
        store=workspace["store"].root,
        models=workspace["models"],
        episodes=plan,
        as_of=KNOWN_ON,
    )

    assert code == 1
    assert "the macro panel knows no variables no.such.variable" in capsys.readouterr().out


def test_the_calculation_date_is_the_day_before_the_start_the_plan_names(workspace, tmp_path):
    plan = plan_file(tmp_path, [("R1", "gl.brent"), ("R2", "ru.key_rate")])

    code, report = run_episodes(workspace, plan, tmp_path)

    assert code == 0
    assert [row["as_of"] for row in report["episodes"]] == ["2023-05-31", "2024-02-29"]


def test_a_run_without_a_single_episode_reports_it_instead_of_raising(workspace, tmp_path, capsys):
    # Unknown variables drop both episodes: the report has no scored cells at all.
    plan = plan_file(tmp_path, [("R1", "fake.one"), ("R2", "fake.two")])

    code, report = run_episodes(workspace, plan, tmp_path)

    assert code == 1
    assert report["episodes"] == []
    assert report["scores"] == {}
    assert set(report["dropped"]["episodes"]) == {"R1", "R2"}
    out = capsys.readouterr().out
    assert "0 of 2 episodes judged." in out
    assert "- R1: the macro panel knows no variables fake.one" in out


def test_the_monte_carlo_option_moves_no_other_field(workspace, tmp_path):
    plan = plan_file(tmp_path, [("R1", "gl.brent"), ("R2", "ru.key_rate")])

    _, plain = run_episodes(workspace, plan, tmp_path, name="plain.json")
    _, monte_carlo = run_episodes(workspace, plan, tmp_path, name="mc.json", mc_bootstrap=True)

    # The Monte-Carlo error is reported beside the sampling error, never in place of it.
    for key in ("episodes", "quantities", "sanity", "scores", "constraint_gate", "halves"):
        assert plain[key] == monte_carlo[key]
    assert "monte_carlo" not in plain
    interval = monte_carlo["monte_carlo"]
    assert interval["note"].startswith("bootstrap over the draws")
    low, high = interval["interval"]
    assert low <= interval["point"] <= high


def test_the_details_file_holds_the_full_report(workspace, tmp_path):
    plan = plan_file(tmp_path, [("R1", "gl.brent"), ("R2", "ru.key_rate")])

    code, report = run_episodes(workspace, plan, tmp_path)

    assert code == 0
    assert report["dropped"] == {"episodes": {}, "left_out": {}, "non_finite": {}}
    assert len(report["episodes"]) == 2
    row = report["quantities"]["ru.key_rate"]
    assert row["scored"] > 0
    assert row["ratios"]["model/walk"] is not None and row["ratios"]["model/base"] is not None
    assert row["calibration"] and row["sharpness"] > 0
    # The cells are in the details file, every one with its band and its scores.
    cells = [cell for cell in report["rows"] if cell["quantity"] == "ru.key_rate"]
    assert len(cells) == row["scored"]
    assert all(cell["model"] > 0 and cell["walk"] is not None for cell in cells)
    # The constrained variable keeps its cells, marked as constrained.
    constrained = report["quantities"]["gl.brent"]
    assert constrained["constrained"] > 0


def test_save_keeps_the_report_without_its_cells_beside_the_plan(workspace, tmp_path, capsys):
    plan = plan_file(tmp_path, [("R1", "gl.brent"), ("R2", "ru.key_rate")])

    code, report = run_episodes(workspace, plan, tmp_path, save=True)

    assert code == 0
    saved = tmp_path / "results"
    kept = json.loads((saved / "fake.macro.json").read_text(encoding="utf-8"))
    assert "rows" not in kept
    assert kept == {key: value for key, value in report.items() if key != "rows"}
    tables = (saved / "fake.macro.md").read_text(encoding="utf-8")
    assert tables.startswith("## What fell out") and tables in capsys.readouterr().out


def test_a_part_of_the_plan_is_not_saved_as_its_result(workspace, tmp_path):
    plan = plan_file(tmp_path, [("R1", "gl.brent"), ("R2", "ru.key_rate")])

    with pytest.raises(ValueError, match="--only"):
        run_episodes(workspace, plan, tmp_path, only=["R1"], save=True)
    assert not (tmp_path / "results").exists()


def test_the_report_reads_what_fell_out_and_the_sanity_before_the_scores(
    workspace, tmp_path, capsys
):
    plan = plan_file(tmp_path, [("R1", "gl.brent"), ("R2", "ru.key_rate")])

    _, report = run_episodes(workspace, plan, tmp_path)
    out = capsys.readouterr().out

    assert report["target_level"] == "macro"
    macro = report["scores"]["macro"]
    assert set(macro) == {
        "model/conditional_walk",
        "model/walk",
        "base/walk",
        "model/base",
    }
    assert macro["model/walk"]["score"] > 0 and macro["model/walk"]["by_horizon"]
    assert report["halves"]["all"]["episodes"] == ["R1", "R2"]
    assert report["halves"]["tuning"]["episodes"] == ["R1"]
    assert report["halves"]["validation"]["episodes"] == ["R2"]
    assert report["halves"]["tuning"]["admission_failed"] is False
    assert report["utility"]["by_source"]["manual"]["episodes"] == 2
    order = [
        out.index(title)
        for title in (
            "## What fell out",
            "## Sanity of the quantities",
            "## Relative CRPS",
            "## Scenario usefulness",
            "## Halves",
            "## Quantities",
        )
    ]
    assert order == sorted(order)


def test_the_full_table_does_not_depend_on_the_halves(workspace, tmp_path):
    checks = [("R1", "gl.brent"), ("R2", "ru.key_rate")]
    plan = plan_file(tmp_path, checks)
    _, first = run_episodes(workspace, plan, tmp_path, name="first.json")

    swapped = plan_file(tmp_path, checks, swap=True)
    _, second = run_episodes(workspace, swapped, tmp_path, name="second.json")

    # The final table is built over all episodes: the marks change the halves, nothing else.
    assert first["scores"] == second["scores"]
    assert first["rows"] == second["rows"]
    assert first["quantities"] == second["quantities"]
    assert first["sanity"] == second["sanity"]
    assert first["halves"]["tuning"]["episodes"] == second["halves"]["validation"]["episodes"]


def test_the_decision_section_prints_the_steps(capsys):
    report = {
        "decision": {
            "target_level": "macro",
            "half": "validation",
            "cells": 12,
            "benchmark": "conditional_walk",
            "vs_benchmark": {"point": 0.9, "low": 0.8, "high": 1.0},
            "vs_incumbent": {"point": 0.95, "low": 0.9, "high": 1.01},
            "admission_shares": {DOMAIN: 0.0, WIDTH: 0.01},
            "admission_grown": [],
            "verdict": "inconclusive",
            "step": 4,
            "note": "the difference is undetected; the incumbent stays",
        }
    }
    runner._print_decision(report)
    out = capsys.readouterr().out
    assert "## Decision (two standard errors)" in out
    assert "**inconclusive** (step 4)" in out
    assert "half validation: 12 paired cells" in out
    assert "model/conditional_walk 0.900 [0.800, 1.000]" in out


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


def test_the_random_walk_is_not_the_base_run(workspace, tmp_path):
    plan = plan_file(tmp_path, [("R1", "gl.brent"), ("R2", "ru.key_rate")])

    _, report = run_episodes(workspace, plan, tmp_path)

    ratios = report["quantities"]["ru.key_rate"]["ratios"]
    # The fake model ignores the scenario, so its base run is its conditional forecast.
    assert ratios["model/base"] == pytest.approx(1.0)
    # The random walk is a different benchmark, and it is the one labelled as such.
    assert ratios["model/walk"] != pytest.approx(1.0, rel=1e-9)
    assert ratios["base/walk"] == pytest.approx(ratios["model/walk"])


def test_an_episode_scores_the_same_alone_and_with_others(workspace, tmp_path):
    plan = plan_file(tmp_path, [("R1", "gl.brent"), ("R2", "ru.key_rate")])

    _, alone = run_episodes(workspace, plan, tmp_path, name="alone.json", only=["R1"])
    _, together = run_episodes(workspace, plan, tmp_path, name="together.json")

    def cells_of(report):
        return [row for row in report["rows"] if row["episode"] == "R1"]

    # The history range and the random walk come from R1's own calculation date, so loading
    # R2 changes nothing about R1's cells.
    assert cells_of(alone) == cells_of(together)
    assert any(row["quantity"] == "ru.key_rate" for row in cells_of(alone))


def test_the_macro_domain_bounds_rates_and_leaves_spreads_and_logs_open():
    registry = load_registry()
    assert macro_domain(registry.variable("ru.key_rate")) == (0.0, np.inf)
    # A spread has been below zero in the history, and 100 ln X is any real number.
    assert macro_domain(registry.variable("ru.spread_aa")) is None
    assert macro_domain(registry.variable("ru.usdrub")) is None
    cell = Cell(
        "e",
        1,
        "ru.key_rate",
        0.0,
        np.array([-10.0, -2.0, 0.0, 1.0]),
        history_range=100.0,
        domain=macro_domain(registry.variable("ru.key_rate")),
    )
    (row,) = runner._cell_rows([cell], {"ru.key_rate": "macro"})
    assert row[DOMAIN] is True and row[WIDTH] is False


def test_the_driver_domain_maps_the_natural_endpoints():
    def quantity(**fields):
        return Quantity(id="q", label="q", unit="u", regressors=["ru.cpi"], lags=1, **fields)

    # A price of 0.05 is 100 ln 0.05 = -300 in the model scale: only a price of nought is
    # out of bounds, and it maps to minus infinity.
    assert quantity_domain(quantity(transform="log100")) is None
    assert quantity_domain(quantity(transform="log100", domain=[0.0, 1e6]))[0] == -np.inf
    assert quantity_domain(quantity(transform="log100", domain=[0.0, 1e6]))[1] == (
        pytest.approx(100 * np.log(1e6))
    )
    # A logit declared on [0, 1] is unbounded in the model scale, not silently "no gate".
    assert quantity_domain(quantity(transform="logit", domain=[0.0, 1.0])) is None
    low, high = quantity_domain(quantity(transform="logit", domain=[0.01, 0.99]))
    assert low == pytest.approx(np.log(0.01 / 0.99))
    assert high == pytest.approx(np.log(0.99 / 0.01))
    assert quantity_domain(quantity(transform="identity", domain=[0.0, 100.0])) == (
        0.0,
        100.0,
    )


def test_the_admission_rules_are_read_apart_for_every_quantity():
    def cell(month, draws, quantity="q"):
        return Cell(
            "e",
            month,
            quantity,
            0.0,
            np.asarray(draws, dtype=float),
            history_range=10.0,
            domain=(0.0, np.inf),
        )

    def admission(cells):
        levels = {cell.quantity: "macro" for cell in cells}
        return sanity_check(runner._cell_rows(cells, levels))["admission"]

    domain_cells = [cell(month, [-2.0, -1.0, 0.0, 1.0]) for month in range(1, 5)]
    width_cells = [cell(month, [0.0, 0.0, 100.0, 100.0]) for month in range(1, 5)]
    fine = [cell(month, [1.0, 2.0, 3.0, 4.0]) for month in range(1, 93)]
    # Four cells break each rule on its own: the union is 8 %, but neither rule reaches 5 %.
    found = admission(domain_cells + width_cells + fine)
    assert found["cells"] == 100
    assert found["rules"][DOMAIN]["worst"] == pytest.approx(0.04)
    assert found["rules"][WIDTH]["worst"] == pytest.approx(0.04)
    assert found["failed"] is False
    # Six cells breaking the width rule fail that rule and the version.
    extra = [cell(month, [0.0, 0.0, 100.0, 100.0]) for month in range(1, 3)]
    failed = admission(domain_cells + width_cells + extra + fine)
    assert failed["rules"][WIDTH]["worst"] == pytest.approx(6 / 102)
    assert failed["rules"][WIDTH]["quantities"] == ["q"]
    assert failed["rules"][DOMAIN]["quantities"] == []
    assert failed["failed"] is True
    # The threshold is strict: exactly five percent does not pass ("fewer than 5 %").
    exact = [cell(month, [0.0, 0.0, 100.0, 100.0]) for month in range(1, 6)]
    enough = [cell(month, [1.0, 2.0, 3.0, 4.0]) for month in range(1, 96)]
    assert admission(exact + enough)["failed"] is True
    # The same five broken cells inside a quiet second quantity still fail: the pool of two
    # quantities would hide them.
    other = [cell(month, [1.0, 2.0, 3.0, 4.0], quantity="r") for month in range(1, 200)]
    hidden = admission(width_cells[:3] + enough[:40] + other)
    assert hidden["rules"][WIDTH]["pooled"] < 0.05
    assert hidden["rules"][WIDTH]["quantities"] == ["q"]


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
    workspace, tmp_path, capsys
):
    plan = plan_file(tmp_path, [("R1", "gl.brent"), ("R2", "ru.key_rate")])

    code, report = run_episodes(workspace, plan, tmp_path)
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


def prepare(workspace, tmp_path, episodes):
    """The plan and its first episode, prepared as the runner prepares them."""
    plan = load_episodes(plan_file(tmp_path, episodes))
    check = plan.episodes[0]
    episode = runner._episode(
        workspace["macro"],
        workspace["store"],
        load_registry(),
        plan,
        check,
        workspace["models"] / "backtest",
        KNOWN_ON,
    )
    return plan, episode


def test_an_episode_without_interventions_keeps_its_constraint_rows(workspace, tmp_path):
    """The "nothing changed" proof of the plan's check 1, up to the rows of the scenario.

    An episode of observations alone builds the same point rows the old code built: one per
    finite realized month of every constrained variable, in the same order, with the same
    tolerance. The paths themselves are not compared: the old binary is not available, so the
    CRPS of check 1 cannot be reproduced here and no number is invented.
    """
    from mst.model_sdk.contracts import LinearTerm
    from mst.scenario.build import point_constraint

    _, episode = prepare(
        workspace, tmp_path, [("R1", ["gl.brent", "ru.key_rate"]), ("R2", "ru.key_rate")]
    )

    assert episode.scenario.interventions == ()
    assert episode.observations == ["gl.brent", "ru.key_rate"]
    assert episode.interventions == []
    expected = tuple(
        point_constraint(
            (LinearTerm(variable=name, month=month, weight=1.0),),
            float(episode.realized[month - 1, episode.panel.ids.index(name)]),
            episode.macro,
        )
        for name in episode.observations
        for month in range(1, episode.horizon + 1)
        if not np.isnan(episode.realized[month - 1, episode.panel.ids.index(name)])
    )
    assert episode.scenario.constraints == expected


def test_an_intervention_sets_the_realized_level_and_pins_its_cells(workspace, tmp_path):
    plan, episode = prepare(
        workspace, tmp_path, [("R1", "gl.brent", "ru.key_rate"), ("R2", "ru.key_rate")]
    )
    column = episode.panel.ids.index("ru.key_rate")
    realized = episode.realized[:, column]
    months = tuple(month for month, value in enumerate(realized, start=1) if not np.isnan(value))

    (item,) = episode.scenario.interventions
    assert item.target == "ru.key_rate" and item.kind == "set"
    assert item.months == months
    assert item.values.shape == (plan.draws, len(months))
    expected = np.tile([float(realized[month - 1]) for month in months], (plan.draws, 1))
    assert np.array_equal(item.values, expected)
    # The decision fixes its cells: they are reported, as a point observation's cell is, and
    # never scored.
    installed = runner._installed(episode.scenario.interventions)
    assert installed == {("ru.key_rate", month) for month in months}
    assert ("gl.brent", 1) not in installed


def test_an_intervened_quantity_is_reported_but_never_scored(workspace, tmp_path):
    plan = plan_file(tmp_path, [("R1", "gl.brent", "ru.key_rate"), ("R2", "ru.key_rate")])

    _, report = run_episodes(workspace, plan, tmp_path, only=["R1"])

    episode = report["episodes"][0]
    assert episode["interventions"] == ["ru.key_rate"]
    row = report["quantities"]["ru.key_rate"]
    assert row["constrained"] == episode["horizon"] == row["cells"]
    assert row["scored"] == 0 and row["calibration"] is None


def test_the_report_meta_records_the_versions_and_the_vintage(workspace, tmp_path):
    plan = plan_file(tmp_path, [("R1", "gl.brent"), ("R2", "ru.key_rate")])

    _, report = run_episodes(workspace, plan, tmp_path)
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
    # The report names no missing metric any more, and the tail is reported on the portfolio
    # when one is given.
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
        month: float(100.0 * 1.01**t * factor[month.month - 1]) for t, month in enumerate(months)
    }
    revised = {
        date(2019, 6, 1): original[date(2019, 6, 1)] * 1.5,
        date(2019, 7, 1): original[date(2019, 7, 1)] * 0.5,
    }
    rows = [
        Observation(month, value, date(2021, 1, 10), False) for month, value in original.items()
    ]
    rows += [Observation(month, value, date(2022, 6, 1), False) for month, value in revised.items()]
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
    today = seasonal_profile(np.array([100 * np.log(known[month]) for month in months]), months)
    alternative = np.array([100 * np.log(known[month]) for month in horizon]) - today[calendar]
    assert not np.allclose(result[:, 0], alternative)


def test_the_satellites_score_the_served_drivers(workspace, tmp_path):
    plan = plan_file(tmp_path, [("R1", "gl.brent"), ("R2", "ru.key_rate")])
    # The driver catalog of the workspace: one object, one driver, with a realized series.
    catalog = _catalog(tmp_path)
    store = workspace["store"]
    grid = months_between(simulated.START, shift(simulated.START, 59))
    store.write(
        "drv.share_price.sber",
        [Observation(month, float(5.0 + 0.01 * t), month, False) for t, month in enumerate(grid)],
    )

    code, report = run_episodes(workspace, plan, tmp_path, drivers=catalog, satellites=True)

    assert code == 0
    row = report["quantities"]["share_price:sber"]
    assert row["level"] == "driver/share_price"
    assert row["scored"] > 0
    # The naive forecast of a target is scored alongside the random walk; the base run has no
    # driver paths, so there is no ratio against it rather than one copied from the model.
    assert row["ratios"]["model/naive"] is not None
    assert row["ratios"]["model/walk"] is not None
    assert "model/base" not in row["ratios"]
    assert set(report["scores"]["driver/share_price"]) == {"model/walk", "model/naive"}
    # The fake part records no training range, so no range check runs on its paths.
    assert row["unverified_tail"] is False


def test_the_portfolio_tail_is_reported_with_its_horizon(workspace, tmp_path, capsys):
    plan = plan_file(tmp_path, [("R1", "gl.brent"), ("R2", "ru.key_rate")])
    catalog = _catalog(tmp_path)
    store = workspace["store"]
    grid = months_between(simulated.START, shift(simulated.START, 59))
    store.write(
        "drv.share_price.sber",
        [Observation(month, float(5.0 + 0.01 * t), month, False) for t, month in enumerate(grid)],
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
        plan,
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


def test_an_unknown_portfolio_is_a_named_error(workspace, tmp_path):
    plan = plan_file(tmp_path, [("R1", "gl.brent"), ("R2", "ru.key_rate")])

    # Without the driver paths there is nothing to value the positions on.
    with pytest.raises(ValueError, match="needs --satellites"):
        run_episodes(workspace, plan, tmp_path, portfolio="absent")

    catalog = _catalog(tmp_path)
    grid = months_between(simulated.START, shift(simulated.START, 59))
    workspace["store"].write(
        "drv.share_price.sber",
        [Observation(month, float(5.0 + 0.01 * t), month, False) for t, month in enumerate(grid)],
    )
    with pytest.raises(ValueError, match="no portfolio 'absent'"):
        run_episodes(
            workspace, plan, tmp_path, drivers=catalog, satellites=True, portfolio="absent"
        )


def test_the_decision_compares_two_reports_on_the_same_cells(workspace, tmp_path):
    plan = plan_file(tmp_path, [("R1", "gl.brent"), ("R2", "ru.key_rate")])
    _, report = run_episodes(workspace, plan, tmp_path, name="incumbent.json")
    against = tmp_path / "incumbent.json"
    halves = {episode["id"]: episode["half"] for episode in report["episodes"]}

    def decide_on(path):
        return runner._decision_block(
            report["rows"],
            path,
            sanity=report["sanity"],
            halves=halves,
            plan=load_episodes(plan),
            target_level="macro",
        )

    # The candidate against itself: no difference from the incumbent, and the verdict repeats.
    verdict = decide_on(against)
    assert verdict["vs_incumbent"]["point"] == pytest.approx(1.0)
    assert verdict["benchmark"] == "conditional_walk"
    # The verdict is spoken on the validation half, naming it; one episode leaves its error
    # unknown, so nothing is detected.
    assert verdict["half"] == "validation"
    assert verdict["cells"] > 0
    assert verdict["verdict"] == "inconclusive"
    assert decide_on(against) == verdict

    # A cell the two reports do not share is an error naming the missing cell, not a silent
    # comparison on the intersection.
    broken = json.loads(against.read_text(encoding="utf-8"))
    dropped = next(row for row in reversed(broken["rows"]) if row["episode"] == "R2")
    broken["rows"].remove(dropped)
    file = tmp_path / "broken.json"
    file.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="do not share 1 cells") as error:
        decide_on(file)
    assert f"{dropped['episode']}/{dropped['quantity']}" in str(error.value)

    # The saved result keeps no cells: a decision reads the details file instead.
    saved = tmp_path / "saved.json"
    saved.write_text(json.dumps(runner._compact(broken)), encoding="utf-8")
    with pytest.raises(ValueError, match="pass the --details file"):
        decide_on(saved)


def test_the_satellite_bundle_is_cached_by_its_own_inputs(workspace, tmp_path, monkeypatch):
    plan = plan_file(tmp_path, [("R1", "gl.brent"), ("R2", "ru.key_rate")])
    catalog = _catalog(tmp_path)
    store = workspace["store"]
    grid = months_between(simulated.START, shift(simulated.START, 59))
    store.write(
        "drv.share_price.sber",
        [Observation(month, float(5.0 + 0.01 * t), month, False) for t, month in enumerate(grid)],
    )
    calls = []
    original = models_pkg.estimate_satellites

    def counted(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(models_pkg, "estimate_satellites", counted)
    _, first = run_episodes(
        workspace, plan, tmp_path, name="first.json", drivers=catalog, satellites=True
    )
    estimated = len(calls)
    # One bundle per episode: the episodes are estimated on different macro versions.
    assert estimated == 2

    _, second = run_episodes(
        workspace, plan, tmp_path, name="second.json", drivers=catalog, satellites=True
    )

    # The same inputs reuse both bundles: the estimation is not repeated.
    assert len(calls) == estimated
    assert first["meta"]["satellite_versions"] == second["meta"]["satellite_versions"]
    assert first["episodes"] == second["episodes"]


def test_the_satellite_key_covers_the_inputs_that_shape_the_bundle(
    workspace, tmp_path, monkeypatch
):
    from mst.drivers.registry import load_registry as load_drivers

    catalog = _catalog(tmp_path)
    store = workspace["store"]
    grid = months_between(simulated.START, shift(simulated.START, 59))
    series = [
        Observation(month, float(5.0 + 0.01 * t), month, False) for t, month in enumerate(grid)
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
    plan = plan_file(tmp_path, [("R1", "gl.brent"), ("R2", "ru.key_rate")])
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
        workspace, plan, tmp_path, name="late.json", drivers=catalog, satellites=True
    )

    # The first episode is named and skipped; the check goes on with the second.
    assert code == 1
    assert [episode["id"] for episode in report["episodes"]] == ["R2"]
    assert report["dropped"]["episodes"]["R1"].startswith("satellites:")


def test_the_calculation_date_is_the_day_before_the_episode():
    assert calculation_date(date(2023, 6, 1)) == date(2023, 5, 31)


def test_an_episode_splits_its_variables_into_two_kinds():
    from pydantic import ValidationError

    from mst.metrics.episodes import EpisodeCheck

    fields = dict(
        id="R1",
        start="2014-03",
        half="tuning",
        source="manual",
        reason="the decision of the episode",
        sources=["S1"],
    )
    with pytest.raises(ValidationError, match="fixes no variable"):
        EpisodeCheck(**fields)
    with pytest.raises(ValidationError, match="both an observation and an intervention"):
        EpisodeCheck(**fields, observations=["ru.key_rate"], interventions=["ru.key_rate"])
    check = EpisodeCheck(**fields, observations=["ru.usdrub"], interventions=["ru.key_rate"])
    assert check.constrained == ["ru.usdrub", "ru.key_rate"]


def test_an_unmarked_or_misguided_plan_is_a_validation_error(tmp_path):
    from pydantic import ValidationError

    plan = plan_file(tmp_path, [("R1", "gl.brent"), ("R2", "ru.key_rate")])
    text = plan.read_text(encoding="utf-8")

    def check(broken: str) -> str:
        file = tmp_path / "broken.toml"
        file.write_text(broken, encoding="utf-8")
        with pytest.raises(ValidationError):
            load_episodes(file)
        return broken

    # The half and the source of the scenario are marks of the episode plan, not optional fields.
    check(text.replace('half = "tuning"\n', ""))
    check(text.replace('source = "manual"\n', "", 1))
    check(text.replace('source = "manual"', 'source = "human"', 1))
    # A plan marked entirely as one half has nothing to check on.
    check(text.replace('half = "validation"', 'half = "tuning"'))
    # An episode names its start month, written as YYYY-MM.
    check(text.replace('start = "2023-06"\n', ""))
    check(text.replace('start = "2023-06"', 'start = "2023-6"'))
    check(text.replace('start = "2023-06"', "start = 2023-06-01"))


def test_the_shipped_episodes_name_registry_variables():
    from mst.macro.registry import load_registry

    plan = load_episodes(PLAN)
    variables = {v.id for v in load_registry().variables}
    for check in plan.episodes:
        assert set(check.observations) <= variables, check.id
        assert set(check.interventions) <= variables, check.id
        assert not set(check.observations) & set(check.interventions), check.id


def test_every_point_pins_its_cell_whatever_its_rounding():
    from mst.model_sdk.contracts import LinearConstraint, LinearTerm

    rng = np.random.default_rng(3)
    for value, tolerance in zip(rng.uniform(0, 20, 500), rng.uniform(1e-4, 0.1, 500), strict=True):
        term = LinearTerm("x", 1, 1.0)
        point = LinearConstraint(
            terms=(term,),
            low=value - tolerance,
            high=value + tolerance,
            point=True,
            tolerance=tolerance,
        )
        assert runner._pinned([point]) == {("x", 1)}
    wide = LinearConstraint(terms=(term,), low=0.0, high=1.0, tolerance=0.01)
    assert runner._pinned([wide]) == set()


class _ShortRate(simulated.FakeMacro):
    """The fake model with a rate history it finds too short on every date."""

    def estimate(self, context):
        if any(variable.id == "ru.key_rate" for variable in context.registry.variables):
            from mst.model_sdk import ShortHistory

            raise ShortHistory({"ru.key_rate": "2 changes, the model needs 3"})
        assert context.left_out == {"ru.key_rate": "2 changes, the model needs 3"}
        return super().estimate(context)


def test_a_short_variable_leaves_out_its_cells_not_the_episode(workspace, tmp_path, monkeypatch):
    macro = _ShortRate()
    monkeypatch.setattr(models_pkg, "selected", lambda model_id=None: macro)
    plan = plan_file(tmp_path, [("R1", "gl.brent"), ("R2", "gl.brent")])

    code, report = run_episodes(workspace, plan, tmp_path, rolling=False)

    assert [row["id"] for row in report["episodes"]] == ["R1", "R2"]
    reason = {"ru.key_rate": "2 changes, the model needs 3"}
    assert all(row["left_out"] == reason for row in report["episodes"])
    assert report["dropped"]["left_out"] == {"R1": reason, "R2": reason}
    assert "ru.key_rate" not in report["quantities"]
    assert "gl.brent" in report["quantities"]
    assert report["dropped"]["episodes"] == {}
    assert code == 0


def test_an_episode_fixing_a_short_variable_is_dropped_with_the_reason(
    workspace, tmp_path, monkeypatch
):
    macro = _ShortRate()
    monkeypatch.setattr(models_pkg, "selected", lambda model_id=None: macro)
    plan = plan_file(tmp_path, [("R1", "gl.brent"), ("R2", [], "ru.key_rate")])

    code, report = run_episodes(workspace, plan, tmp_path, rolling=False)

    assert [row["id"] for row in report["episodes"]] == ["R1"]
    assert report["dropped"]["episodes"] == {
        "R2": "the model left out ru.key_rate, which the episode fixes: "
        "2 changes, the model needs 3"
    }
    assert code == 1


def test_a_short_history_that_names_nothing_is_no_contract():
    from mst.model_sdk import ShortHistory

    with pytest.raises(ValueError, match="names no variable"):
        ShortHistory({})
