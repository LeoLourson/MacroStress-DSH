"""The episodes: the scenario from the facts, what fell out, the report and its gates."""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from mst import models as models_pkg
from mst.macro.registry import load_registry
from mst.metrics import backtest as runner
from mst.metrics import scores
from mst.metrics.episodes import calculation_date, load_episodes
from mst.metrics.tables import print_decision

from . import simulated

KNOWN_ON = date(2030, 1, 1)
STARTS = {"R1": "2023-06", "R2": "2024-03"}
"""The start months of the episodes the tests plan."""
PLAN = Path(__file__).resolve().parents[2] / "bench" / "model-backtest" / "episodes.toml"


@pytest.fixture(autouse=True)
def fast_intervals(monkeypatch):
    """A hundred repetitions keep the tests quick; the report uses a thousand."""
    monkeypatch.setattr(scores, "REPETITIONS", 100)


def plan_file(tmp_path, episodes, *, swap=False):
    """An episodes.toml with the named checks: ``(id, observations, interventions)``.

    A string is one name. The halves alternate tuning and validation by position, so every plan
    this helper writes has both non-empty — the plan contract requires it; ``swap`` flips the
    marks, for the test that the full-sample table does not depend on them.
    """
    text = ["draws = 40", "seed = 3", "after_months = 6", "", ""]
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
    satellite = simulated.install(monkeypatch, macro)
    return {
        "store": store,
        "models": tmp_path / "models",
        "macro": macro,
        "satellite": satellite,
        "tmp": tmp_path,
    }


def run_episodes(workspace, plan, tmp_path, *, name="details.json", only=(), **extra):
    """Run the check and read its details file."""
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


def with_drivers(workspace, tmp_path):
    """The driver catalog of the tests, with the realized series of its driver."""
    simulated.share_price(workspace["store"])
    return simulated.catalog(tmp_path)


def test_the_runner_judges_the_cells_of_every_episode(workspace, tmp_path, capsys):
    plan = plan_file(tmp_path, [("R1", "gl.brent"), ("R2", "ru.key_rate")])

    code = runner.run(
        store=workspace["store"].root, models=workspace["models"], episodes=plan, as_of=KNOWN_ON
    )

    assert code == 0
    out = capsys.readouterr().out
    assert "| R1 |" in out and "| R2 |" in out
    assert "## Episodes: quantities" in out and "ru.key_rate" in out
    # One version per episode calculation date, estimated on the day before the start.
    assert sorted(workspace["macro"].calls) == [
        calculation_date(date.fromisoformat(start + "-01")) for start in STARTS.values()
    ]


def test_an_unknown_variable_is_reported_not_raised(workspace, tmp_path, capsys):
    plan = plan_file(tmp_path, [("R1", "no.such.variable"), ("R2", "ru.key_rate")])

    code, report = run_episodes(workspace, plan, tmp_path)

    assert code == 1
    assert report["dropped"]["refused"] == {
        "R1": "the macro panel knows no variables no.such.variable"
    }
    assert "- R1: the macro panel knows no variables no.such.variable" in capsys.readouterr().out


def test_a_run_without_a_single_episode_reports_it_instead_of_raising(workspace, tmp_path, capsys):
    # Unknown variables drop both episodes: the report has no scored cells at all.
    plan = plan_file(tmp_path, [("R1", "fake.one"), ("R2", "fake.two")])

    code, report = run_episodes(workspace, plan, tmp_path)

    assert code == 1
    assert report["episodes"] == [] and report["levels"] == {}
    assert set(report["dropped"]["refused"]) == {"R1", "R2"}
    assert "0 of 2 episodes judged." in capsys.readouterr().out


def test_the_monte_carlo_option_moves_no_other_field(workspace, tmp_path):
    plan = plan_file(tmp_path, [("R1", "gl.brent"), ("R2", "ru.key_rate")])

    _, plain = run_episodes(workspace, plan, tmp_path, name="plain.json")
    _, monte_carlo = run_episodes(workspace, plan, tmp_path, name="mc.json", mc_bootstrap=True)

    # The Monte-Carlo error is reported beside the sampling error, never in place of it.
    for key in ("episodes", "quantities", "sanity", "levels", "constraint_gate", "halves"):
        assert plain[key] == monte_carlo[key]
    assert "monte_carlo" not in plain
    found = monte_carlo["monte_carlo"]
    low, high = found["interval"]
    assert low <= found["point"] <= high


def test_the_details_file_holds_the_full_report(workspace, tmp_path):
    plan = plan_file(tmp_path, [("R1", "gl.brent"), ("R2", "ru.key_rate")])

    code, report = run_episodes(workspace, plan, tmp_path)

    assert code == 0
    assert report["dropped"] == {"refused": {}, "left_out": {}, "non_finite": {}}
    assert len(report["episodes"]) == 2
    row = report["quantities"]["ru.key_rate"]
    assert row["scored"] > 0
    assert row["ratios"]["model/walk"] is not None and row["ratios"]["model/base"] is not None
    assert row["calibration"] and row["sharpness"] > 0
    cells = [cell for cell in report["rows"] if cell["quantity"] == "ru.key_rate"]
    assert len(cells) == row["scored"]
    assert all(cell["model"] > 0 and cell["walk"] is not None for cell in cells)
    # The constrained variable keeps its cells, marked as constrained.
    assert report["quantities"]["gl.brent"]["constrained"] > 0


def test_save_keeps_the_report_without_its_cells_beside_the_plan(workspace, tmp_path, capsys):
    plan = plan_file(tmp_path, [("R1", "gl.brent"), ("R2", "ru.key_rate")])

    code, report = run_episodes(workspace, plan, tmp_path, save=True)

    assert code == 0
    saved = tmp_path / "results"
    kept = json.loads((saved / "fake.macro.json").read_text(encoding="utf-8"))
    assert "rows" not in kept
    assert kept == {key: value for key, value in report.items() if key != "rows"}
    tables = (saved / "fake.macro.md").read_text(encoding="utf-8")
    assert tables.startswith("## Episodes: what fell out") and tables in capsys.readouterr().out


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

    macro = report["levels"]["macro"]
    assert set(macro["pairs"]) == {
        "model/conditional_walk",
        "model/walk",
        "base/walk",
        "model/base",
    }
    assert macro["pairs"]["model/walk"]["score"] > 0
    assert report["halves"]["all"]["units"] == ["R1", "R2"]
    assert report["halves"]["tuning"]["units"] == ["R1"]
    assert report["halves"]["validation"]["units"] == ["R2"]
    assert report["halves"]["tuning"]["sanity"]["failed"] is False
    assert report["utility"]["by_source"]["manual"]["episodes"] == 2
    titles = (
        "## Episodes: what fell out",
        "## Episodes: sanity of the quantities",
        "## Episodes: relative CRPS",
        "## Episodes: halves",
        "## Episodes: quantities",
        "## Episodes: usefulness of the scenario",
    )
    order = [out.index(title) for title in titles]
    assert order == sorted(order)


def test_the_full_table_does_not_depend_on_the_halves(workspace, tmp_path):
    checks = [("R1", "gl.brent"), ("R2", "ru.key_rate")]
    _, first = run_episodes(workspace, plan_file(tmp_path, checks), tmp_path, name="first.json")
    swapped = plan_file(tmp_path, checks, swap=True)
    _, second = run_episodes(workspace, swapped, tmp_path, name="second.json")

    # The final table is built over all episodes: the marks change the halves, nothing else.
    assert first["levels"] == second["levels"]
    assert first["quantities"] == second["quantities"]
    assert first["sanity"] == second["sanity"]
    assert first["halves"]["tuning"]["units"] == second["halves"]["validation"]["units"]


def test_the_random_walk_is_not_the_base_run(workspace, tmp_path):
    plan = plan_file(tmp_path, [("R1", "gl.brent"), ("R2", "ru.key_rate")])

    _, report = run_episodes(workspace, plan, tmp_path)

    ratios = report["quantities"]["ru.key_rate"]["ratios"]
    # The fake model ignores the scenario, so its base run is its conditional forecast.
    assert ratios["model/base"] == pytest.approx(1.0)
    assert ratios["model/walk"] != pytest.approx(1.0, rel=1e-9)
    assert ratios["base/walk"] == pytest.approx(ratios["model/walk"])


def test_an_episode_scores_the_same_alone_and_with_others(workspace, tmp_path):
    plan = plan_file(tmp_path, [("R1", "gl.brent"), ("R2", "ru.key_rate")])

    _, alone = run_episodes(workspace, plan, tmp_path, name="alone.json", only=["R1"])
    _, together = run_episodes(workspace, plan, tmp_path, name="together.json")

    def cells_of(report):
        return [row for row in report["rows"] if row["episode"] == "R1"]

    # The history range and the walks come from R1's own calculation date, so loading R2
    # changes nothing about R1's cells.
    assert cells_of(alone) == cells_of(together)
    assert any(row["quantity"] == "ru.key_rate" for row in cells_of(alone))


def test_the_constraint_gate_needs_more_than_one_percent():
    gate = runner._constraint_gate({"R1": {0: None, 1: 0.02}, "R2": {0: 0.005}})
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
    assert gate["not_applicable"] == sum(
        len(episode["constraint_probabilities"]) for episode in report["episodes"]
    )
    assert gate["failed"] is False
    # Not a meaningless 0.00: the episode table says the probability is not applicable.
    episode_line = next(line for line in out.splitlines() if line.startswith("| R1 | 2023"))
    assert "n/a" in episode_line and "0.00 " not in episode_line


def prepare(workspace, tmp_path, episodes):
    """The plan and its first episode, prepared as the runner prepares them."""
    plan = load_episodes(plan_file(tmp_path, episodes))
    episode = runner._episode(
        workspace["macro"],
        workspace["store"],
        load_registry(),
        plan,
        plan.episodes[0],
        workspace["models"] / "backtest",
        KNOWN_ON,
        None,
        None,
    )
    return plan, episode


def test_an_observation_follows_the_realized_path_as_points(workspace, tmp_path):
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


def test_an_intervention_sets_the_realized_level(workspace, tmp_path):
    plan, episode = prepare(
        workspace, tmp_path, [("R1", "gl.brent", "ru.key_rate"), ("R2", "ru.key_rate")]
    )
    realized = episode.realized[:, episode.panel.ids.index("ru.key_rate")]
    months = tuple(month for month, value in enumerate(realized, start=1) if not np.isnan(value))

    (item,) = episode.scenario.interventions
    assert item.target == "ru.key_rate" and item.kind == "set"
    assert item.months == months
    expected = np.tile([float(realized[month - 1]) for month in months], (plan.draws, 1))
    assert np.array_equal(item.values, expected)


def test_an_intervened_quantity_is_reported_but_never_scored(workspace, tmp_path):
    plan = plan_file(tmp_path, [("R1", "gl.brent", "ru.key_rate"), ("R2", "ru.key_rate")])

    _, report = run_episodes(workspace, plan, tmp_path, only=["R1"])

    episode = report["episodes"][0]
    assert episode["interventions"] == ["ru.key_rate"]
    row = report["quantities"]["ru.key_rate"]
    assert row["constrained"] == episode["horizon"] == row["cells"]
    assert row["scored"] == 0 and row["calibration"] is None


def test_the_report_meta_records_the_versions_and_the_vintage(workspace, tmp_path):
    from mst.metrics.windows import file_digest
    from mst.model_sdk.version import MODEL_SDK_VERSION

    plan = plan_file(tmp_path, [("R1", "gl.brent"), ("R2", "ru.key_rate")])

    _, report = run_episodes(workspace, plan, tmp_path)
    meta = report["meta"]

    assert meta["draws"] == 40 and meta["seed"] == 3
    assert meta["known_on"] == "2030-01-01"
    assert meta["episodes_toml"] == file_digest(plan)
    assert len(meta["macro_registry"]) == 64
    assert meta["model_sdk"] == MODEL_SDK_VERSION
    assert meta["macro_model"] == {"model_id": "fake.macro", "version": "0.1.0"}
    # Two episodes at two calculation dates: two estimated versions.
    assert len(meta["macro_versions"]) == 2
    assert meta["git_commit"] is None or len(meta["git_commit"]) == 40
    assert meta["missing"] == []


def test_the_satellites_judge_the_drivers_with_their_own_base_run(workspace, tmp_path):
    plan = plan_file(tmp_path, [("R1", "gl.brent"), ("R2", "ru.key_rate")])
    catalog = with_drivers(workspace, tmp_path)

    code, report = run_episodes(workspace, plan, tmp_path, drivers=catalog, satellites=True)

    assert code == 0
    row = report["quantities"]["share_price:sber"]
    assert row["level"] == "driver/share_price"
    assert row["scored"] > 0
    # The naive forecast of a target and the walk are scored alongside, and the drivers on the
    # base run of the macro model give the base run of the level.
    assert set(report["levels"]["driver/share_price"]["pairs"]) == {
        "model/walk",
        "base/walk",
        "model/base",
        "model/naive",
    }
    # The fake part records no training range, so no range check runs on its paths.
    assert row["unverified_tail"] is False


def test_an_episode_the_satellite_estimation_cannot_serve_is_dropped_by_name(
    workspace, tmp_path, monkeypatch
):
    plan = plan_file(tmp_path, [("R1", "gl.brent"), ("R2", "ru.key_rate")])
    catalog = with_drivers(workspace, tmp_path)
    original = models_pkg.estimate_satellites
    calls = []

    def failing(*args, **kwargs):
        calls.append(args[0])
        if len(calls) == 1:
            raise ValueError("share_price:it: series drv.share_price.it has no value")
        return original(*args, **kwargs)

    monkeypatch.setattr(models_pkg, "estimate_satellites", failing)

    code, report = run_episodes(
        workspace, plan, tmp_path, name="late.json", drivers=catalog, satellites=True
    )

    # The first episode is named and dropped; the check goes on with the second.
    assert code == 1
    assert [episode["id"] for episode in report["episodes"]] == ["R2"]
    assert report["dropped"]["refused"]["R1"].startswith("satellites:")


def test_the_portfolio_tail_is_reported_with_its_horizon(workspace, tmp_path, capsys):
    from mst.portfolio import tool as portfolio_tool

    plan = plan_file(tmp_path, [("R1", "gl.brent"), ("R2", "ru.key_rate")])
    catalog = with_drivers(workspace, tmp_path)
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
        workspace, plan, tmp_path, drivers=catalog, satellites=True, portfolio="tail-test"
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
    catalog = with_drivers(workspace, tmp_path)
    with pytest.raises(ValueError, match="no portfolio 'absent'"):
        run_episodes(
            workspace, plan, tmp_path, drivers=catalog, satellites=True, portfolio="absent"
        )


def test_the_decision_compares_two_reports_on_the_same_cells(workspace, tmp_path):
    plan = plan_file(tmp_path, [("R1", "gl.brent"), ("R2", "ru.key_rate")])
    _, report = run_episodes(workspace, plan, tmp_path, name="incumbent.json")
    against = tmp_path / "incumbent.json"

    def decide_on(path):
        return runner._decision_block(report, path, load_episodes(plan), "macro")

    # The candidate against itself: no difference from the incumbent, and the verdict repeats.
    verdict = decide_on(against)
    assert verdict["vs_incumbent"]["point"] == pytest.approx(1.0)
    assert verdict["benchmark"] == "conditional_walk"
    # The verdict is spoken on the validation half; one episode leaves its error unknown, so
    # nothing is detected.
    assert verdict["half"] == "validation"
    assert verdict["cells"] > 0
    assert verdict["verdict"] == "inconclusive"
    assert decide_on(against) == verdict

    # A cell the two reports do not share is an error naming it, not a silent comparison on
    # the intersection.
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
    saved.write_text(json.dumps(runner.compact(broken)), encoding="utf-8")
    with pytest.raises(ValueError, match="pass the --details file"):
        decide_on(saved)


def test_the_decision_section_prints_the_steps(capsys):
    print_decision(
        {
            "level": "macro",
            "half": "validation",
            "cells": 12,
            "benchmark": "conditional_walk",
            "vs_benchmark": {"point": 0.9, "low": 0.8, "high": 1.0},
            "vs_incumbent": {"point": 0.95, "low": 0.9, "high": 1.01},
            "admission_shares": {"band_outside_domain": 0.0},
            "admission_grown": [],
            "verdict": "inconclusive",
            "step": 4,
            "note": "the difference is undetected; the incumbent stays",
        }
    )
    out = capsys.readouterr().out
    assert "**inconclusive** (step 4)" in out
    assert "half validation: 12 paired cells" in out
    assert "model/conditional_walk 0.900 [0.800, 1.000]" in out


def test_the_tail_section_prints_the_horizon_and_the_dropped_episodes(capsys):
    runner._print_tail(
        {
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
    )
    out = capsys.readouterr().out
    assert "## Tail (Acerbi-Székely)" in out
    assert "month 6" in out
    assert "dropped R2: no fact at t = H" in out


class _ShortRate(simulated.FakeMacro):
    """The fake model with a rate history it finds too short on every date."""

    def estimate(self, context):
        if any(variable.id == "ru.key_rate" for variable in context.registry.variables):
            from mst.model_sdk import ShortHistory

            raise ShortHistory({"ru.key_rate": "2 changes, the model needs 3"})
        return super().estimate(context)


def test_a_short_variable_leaves_out_its_cells_not_the_episode(workspace, tmp_path, monkeypatch):
    monkeypatch.setattr(models_pkg, "selected", lambda model_id=None: _ShortRate())
    plan = plan_file(tmp_path, [("R1", "gl.brent"), ("R2", "gl.brent")])

    code, report = run_episodes(workspace, plan, tmp_path)

    assert [row["id"] for row in report["episodes"]] == ["R1", "R2"]
    reason = {"ru.key_rate": "2 changes, the model needs 3"}
    assert report["dropped"]["left_out"] == {"R1": reason, "R2": reason}
    assert "ru.key_rate" not in report["quantities"]
    assert report["dropped"]["refused"] == {}
    assert code == 0


def test_an_episode_fixing_a_short_variable_is_dropped_with_the_reason(
    workspace, tmp_path, monkeypatch
):
    monkeypatch.setattr(models_pkg, "selected", lambda model_id=None: _ShortRate())
    plan = plan_file(tmp_path, [("R1", "gl.brent"), ("R2", [], "ru.key_rate")])

    code, report = run_episodes(workspace, plan, tmp_path)

    assert [row["id"] for row in report["episodes"]] == ["R1"]
    assert report["dropped"]["refused"] == {
        "R2": "the model left out ru.key_rate, which the episode fixes: "
        "2 changes, the model needs 3"
    }
    assert code == 1


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

    text = plan_file(tmp_path, [("R1", "gl.brent"), ("R2", "ru.key_rate")]).read_text(
        encoding="utf-8"
    )

    def check(broken: str) -> None:
        file = tmp_path / "broken.toml"
        file.write_text(broken, encoding="utf-8")
        with pytest.raises(ValidationError):
            load_episodes(file)

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
    plan = load_episodes(PLAN)
    variables = {v.id for v in load_registry().variables}
    for check in plan.episodes:
        assert set(check.observations) <= variables, check.id
        assert set(check.interventions) <= variables, check.id
        assert not set(check.observations) & set(check.interventions), check.id
