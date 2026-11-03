"""The rolling check: the uniformity at ``s_0`` over the dates and the price of events.

The checks follow the plan: on a simulated model that is right, the probability integral
transform of the facts is uniform over the grid and the logarithmic score of the model's
``P(S)`` beats the score of the historical frequency; on a model that is wrong — biased
paths — the transform drifts away from uniform and the deviation is seen; and the block
bootstrap is wider than the ordinary spread on an autocorrelated series, which is why the
overlapping windows of the price ask for it.
"""

import json
from datetime import date

import numpy as np
import pytest

from mst.macro.registry import load_registry
from mst.metrics.block import block_interval
from mst.metrics.episodes import load_episodes
from mst.metrics.rolling import rolling_report
from mst.model_sdk.contracts import MacroSimulation

from . import simulated

KNOWN_ON = date(2030, 1, 1)


class WalkMacro(simulated.FakeMacro):
    """A random-walk macro that knows its own step scale: the panel's monthly changes.

    The store of the tests is drawn from the same walk, so this model is the right one: the
    transform of the facts under its forecast is uniform, and its ``P(S)`` is calibrated.
    The panel is turned into the model scale the way a real model's panel is, so the facts
    and the forecast speak one scale.
    """

    model_id = "walk.macro"

    def estimate(self, context):
        from mst.macro.registry import load_registry

        registry = load_registry()
        months, columns = [], []
        for id_ in self.variables:
            rows = list(context.store.as_of(id_, context.as_of))
            months = [row.month for row in rows]
            variable = registry.variable(id_)
            columns.append([variable.to_model(row.value) for row in rows])
        y = np.array(columns, dtype=float).T
        from mst.model_sdk.contracts import MacroEstimate

        return MacroEstimate(
            meta={
                "as_of": context.as_of.isoformat(),
                "ids": list(self.variables),
                "n_global": 1,
                "months": [m.isoformat() for m in months],
                "shocks": list(simulated.SHOCKS),
                "diagnostics": {},
            },
            arrays={
                "panel_y": y,
                "panel_approximate": np.zeros(y.shape, dtype=bool),
            },
        )

    def load(self, meta, arrays):
        loaded = super().load(meta, arrays)
        steps = arrays["panel_y"][1:] - arrays["panel_y"][:-1]
        loaded.sigma = steps.std(axis=0)
        return loaded


def plan_file(tmp_path, *, step=6, events=(), validation_from=None):
    """An episodes.toml with a rolling grid and, optionally, a family of price events."""
    text = [
        "draws = 60",
        "seed = 3",
        "after_months = 6",
        "",
        "[rolling]",
        f"step_months = {step}",
    ]
    if validation_from is not None:
        text.append(f"validation_from = {validation_from}")
    for event in events:
        identifier, horizon, low, high = event
        text += [
            "",
            "[[rolling.price_events]]",
            f'id = "{identifier}"',
            f"horizon = {horizon}",
            f"low = {low}",
            f"high = {high}",
            'note = "test"',
            f"terms = [{{variable = 'gl.brent', month = {horizon}, weight = 1.0}}]",
        ]
    text += [
        "",
        "",
        "[[episodes]]",
        'id = "R1"',
        'start = "2023-06"',
        'half = "tuning"',
        'source = "manual"',
        'constrained = ["gl.brent"]',
        'driving = ["fake.demand"]',
        'reason = "test"',
        'sources = ["S1"]',
        "",
        "[[episodes]]",
        'id = "R2"',
        'start = "2024-03"',
        'half = "validation"',
        'source = "manual"',
        'constrained = ["ru.key_rate"]',
        'driving = ["fake.rate"]',
        'reason = "test"',
        'sources = ["S1"]',
        "",
    ]
    file = tmp_path / "episodes.toml"
    file.write_text("\n".join(text), encoding="utf-8")
    return file


def report_of(
    tmp_path,
    monkeypatch,
    macro=None,
    *,
    events=(),
    step=3,
    months=120,
    validation_from=None,
    only=None,
):
    """The rolling report of the fake macro on the simulated store."""
    from mst import models as models_pkg
    from mst.metrics import backtest as runner

    tmp_path.mkdir(parents=True, exist_ok=True)
    store = simulated.series_store(tmp_path / "series", months=months, seed=14)
    macro = macro or WalkMacro()
    file = plan_file(tmp_path, step=step, events=events, validation_from=validation_from)
    monkeypatch.setattr(models_pkg, "selected", lambda model_id=None: macro)
    monkeypatch.setattr(runner, "load_episodes", lambda path=None: load_episodes(file))
    registry = load_registry()
    root = tmp_path / "models"
    plan = load_episodes(file)
    return rolling_report(
        plugin=macro,
        store=store,
        registry=registry,
        root=root,
        plan=plan,
        known_on=KNOWN_ON,
        macro_of=runner._episode_macro,
        only=only,
    )


def test_the_transform_of_the_facts_is_uniform_on_a_right_model(tmp_path, monkeypatch):
    found = report_of(tmp_path, monkeypatch)
    assert found["dates"] >= 20
    shares = found["base_uniformity"]["shares"]
    expected = found["base_uniformity"]["expected"]
    # A random walk estimated on its own history: the transform of the facts sits in the
    # canon's buckets within the Monte-Carlo error of a few hundred cells.
    assert found["base_uniformity"]["cells"] >= 150
    for found_share, expected_share in zip(shares, expected, strict=True):
        assert found_share == pytest.approx(expected_share, abs=0.08)


def test_a_biased_model_drifts_away_from_uniform(tmp_path, monkeypatch):
    class Biased(simulated.FakeMacro):
        """A model whose forecast is shifted: the facts land systematically low."""

        def simulate(self, model, plan, *, draws, seed):
            found = super().simulate(model, plan, draws=draws, seed=seed)
            paths = found.paths.copy()
            paths[:, 1:] += 6.0 * model.sigma
            from mst.model_sdk.contracts import MacroSimulation

            return MacroSimulation(
                ids=found.ids,
                months=found.months,
                paths=paths,
                history=found.history,
            )

    found = report_of(tmp_path, monkeypatch, macro=Biased(), months=120)
    shares = found["base_uniformity"]["shares"]
    # The shift pushes the facts below the forecast: the lowest bucket alone carries far
    # more than its expected five percent.
    assert shares[0] > 0.3


def test_a_right_model_ties_the_walk_and_a_biased_one_loses(tmp_path, monkeypatch):
    class Biased(WalkMacro):
        def simulate(self, model, plan, *, draws, seed):
            found = super().simulate(model, plan, draws=draws, seed=seed)
            paths = found.paths.copy()
            paths[:, 1:] += 6.0 * model.sigma
            return MacroSimulation(
                ids=found.ids, months=found.months, paths=paths, history=found.history
            )

    right = report_of(tmp_path / "right", monkeypatch)["quality"]["halves"]["all"]
    assert right["vs_walk"] == pytest.approx(1.0, abs=0.1)
    assert right["interval"]["low"] < right["vs_walk"] < right["interval"]["high"]
    assert right["coverage_90"] == pytest.approx(0.9, abs=0.1)
    assert sum(right["calibration"]) == pytest.approx(1.0)
    assert set(right["by_horizon"]) == set(range(1, 7))
    biased = report_of(tmp_path / "biased", monkeypatch, macro=Biased())
    assert biased["quality"]["halves"]["all"]["vs_walk"] > 1.5


def test_the_grid_splits_into_halves_and_keeps_every_row_in_one(tmp_path, monkeypatch):
    found = report_of(tmp_path, monkeypatch, validation_from="2026-01-01")
    quality = found["quality"]
    assert set(quality["halves"]) == {"all", "tuning", "validation"}
    assert (
        quality["halves"]["tuning"]["cells"] + quality["halves"]["validation"]["cells"]
        == quality["halves"]["all"]["cells"]
    )
    assert all(
        (row["date"] >= "2026-01-01") == (row["half"] == "validation") for row in quality["rows"]
    )
    # A step of three months against windows of six: blocks of two dates.
    assert quality["block_dates"] == 2


class Late(WalkMacro):
    def estimate(self, context):
        if context.as_of.year < 2025:
            raise ValueError("no draw keeps its radius")
        return super().estimate(context)


def test_a_date_with_enough_data_where_the_model_fails_is_a_refusal(tmp_path, monkeypatch):
    found = report_of(tmp_path, monkeypatch, macro=Late())
    assert set(found["refused"].values()) == {"no draw keeps its radius"}
    assert not found["skipped"]
    assert all(row["date"] >= "2025-01-01" for row in found["quality"]["rows"])


def test_a_date_short_of_history_is_left_out_and_is_no_refusal(tmp_path, monkeypatch):
    from mst.metrics import rolling

    monkeypatch.setattr(rolling, "_short_of_history", lambda *_: True)
    found = report_of(tmp_path, monkeypatch, macro=Late())
    assert set(found["skipped"].values()) == {"no draw keeps its radius"}
    assert not found["refused"]
    assert found["dates"] > 0


def test_short_of_history_reads_the_data(tmp_path):
    from mst.macro.registry import load_registry
    from mst.metrics.rolling import _short_of_history

    store = simulated.series_store(tmp_path / "series", months=120, seed=14)
    registry = load_registry()
    assert _short_of_history(store, registry, date(2020, 2, 15))
    assert not _short_of_history(store, registry, date(2029, 1, 1))


def test_the_grid_can_be_narrowed_to_some_dates(tmp_path, monkeypatch):
    from mst.metrics import rolling

    everything = report_of(tmp_path / "all", monkeypatch)
    dates = sorted({row["date"] for row in everything["quality"]["rows"]})[:2]
    only = [date.fromisoformat(when) for when in dates]
    found = report_of(tmp_path / "some", monkeypatch, only=only)
    assert found["dates"] == 2
    assert sorted({row["date"] for row in found["quality"]["rows"]}) == dates
    assert rolling.half_of(only[0], load_episodes(plan_file(tmp_path))) == "tuning"


def test_the_decision_reads_the_two_versions_on_the_same_cells(tmp_path, monkeypatch):
    class Biased(WalkMacro):
        def simulate(self, model, plan, *, draws, seed):
            found = super().simulate(model, plan, draws=draws, seed=seed)
            paths = found.paths.copy()
            paths[:, 1:] += 6.0 * model.sigma
            return MacroSimulation(
                ids=found.ids, months=found.months, paths=paths, history=found.history
            )

    from mst.metrics.backtest import _jsonable
    from mst.metrics.rolling import rolling_decision

    right = report_of(tmp_path / "right", monkeypatch, validation_from="2026-01-01")
    biased = report_of(
        tmp_path / "biased", monkeypatch, macro=Biased(), validation_from="2026-01-01"
    )
    against = tmp_path / "incumbent.json"
    against.write_text(json.dumps(right, default=_jsonable), encoding="utf-8")
    verdict = rolling_decision(biased, against, seed=1)
    assert verdict["half"] == "validation"
    assert verdict["vs_incumbent"]["point"] > 1.5
    assert verdict["verdict"] in {"rejected_worse", "rejected_admission"}
    short = {**right, "quality": {**right["quality"], "rows": right["quality"]["rows"][:-1]}}
    against.write_text(json.dumps(short, default=_jsonable), encoding="utf-8")
    with pytest.raises(ValueError, match="do not share 1 cells"):
        rolling_decision(biased, against, seed=1)


def test_the_price_of_an_event_beats_the_historical_frequency(tmp_path, monkeypatch):
    # An event a fair share of windows sees: the model's probability carries the information
    # of the current level, the frequency only the base rate.
    # The threshold sits in the model scale of the middle of the store, so the event's
    # probability moves over the grid and the model's level carries information the base
    # rate has not.
    found = report_of(tmp_path, monkeypatch, events=[("brent-up", 1, 460.5, 1e6)], months=120)
    price = found["price"]
    assert price["events"] == 1
    assert price["scores"]["windows"] > 20
    # The model's logarithmic score is at least as good as the frequency's — on a sample of
    # dozens of windows the frequency can win by little, never by much.
    assert price["scores"]["model"] <= price["scores"]["frequency"] + 0.05
    # The groups table names the mean probability against the realized frequency.
    assert price["groups"]


def test_the_parameter_part_of_the_price_is_reported(tmp_path, monkeypatch):
    found = report_of(tmp_path, monkeypatch, events=[("brent-up", 1, 460.5, 1e6)], months=120)
    rows = found["price"]["rows"]
    # The fake model has one parameter set, so the price is all event: dist + log P(S)
    # stays at zero whatever the date — the parameter uncertainty the difference names is
    # absent. A date whose event is unreachable keeps no row at all.
    for row in rows:
        assert np.isfinite(row["parameter_part"])
        assert row["parameter_part"] == pytest.approx(0.0, abs=1e-6)


def test_the_block_bootstrap_is_wider_on_an_autocorrelated_series():
    rng = np.random.default_rng(5)
    keys = [f"d{k // 2}" for k in range(240)]
    plain = block_interval(rng.standard_normal(240), keys)
    walk = block_interval(np.cumsum(rng.standard_normal(240)), keys)
    assert walk[1] - walk[0] > 2 * (plain[1] - plain[0])
    # A single block carries everything: the interval closes on the mean.
    one = block_interval([1.0, 2.0, 3.0], ["a", "a", "a"])
    assert one[0] == one[1] == pytest.approx(2.0)


def test_an_event_of_two_terms_reads_their_sum():
    from types import SimpleNamespace

    from mst.metrics.rolling import Grid, _happened
    from mst.model_sdk.contracts import LinearConstraint, LinearTerm

    when = date(2020, 1, 1)
    grid = Grid(
        dates=[when],
        versions={when: SimpleNamespace(panel=SimpleNamespace(months=[date(2019, 12, 1)]))},
        forecasts={},
        facts={when: {(date(2020, 1, 1), "a"): 3.0, (date(2020, 1, 1), "b"): 4.0}},
    )
    terms = (LinearTerm("a", 1, 1.0), LinearTerm("b", 1, 1.0))
    assert _happened(grid, when, terms, LinearConstraint(terms=terms, low=6.0, high=8.0)) is True
    assert _happened(grid, when, terms, LinearConstraint(terms=terms, low=0.0, high=5.0)) is False
    missing = (LinearTerm("c", 1, 1.0),)
    absent = LinearConstraint(terms=missing, low=0.0, high=1.0)
    assert _happened(grid, when, missing, absent) is None
