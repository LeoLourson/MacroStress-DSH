"""The rolling grid: the forecast of every level over the dates, and the price of events.

The checks follow the plan: on a simulated model that is right, the probability integral
transform of the facts is uniform over the grid and the logarithmic score of the model's
``P(S)`` beats the score of the historical frequency; on a model that is wrong — biased
paths — the transform drifts away from uniform and the walk wins; and the block bootstrap is
wider than the ordinary spread on an autocorrelated series, which is why the overlapping
windows ask for it.
"""

import json
from datetime import date

import numpy as np
import pytest

from mst.macro.registry import load_registry
from mst.metrics import rolling, scores
from mst.metrics.backtest import jsonable
from mst.metrics.block import block_interval
from mst.metrics.episodes import load_episodes
from mst.metrics.rolling import rolling_decision, rolling_report
from mst.metrics.windows import Scope
from mst.model_sdk.contracts import MacroSimulation

from . import simulated

KNOWN_ON = date(2030, 1, 1)


@pytest.fixture(autouse=True)
def fast_intervals(monkeypatch):
    """A hundred repetitions keep the tests quick; the report uses a thousand."""
    monkeypatch.setattr(scores, "REPETITIONS", 100)


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


class Biased(WalkMacro):
    """A model whose forecast is shifted: the facts land systematically low."""

    def simulate(self, model, plan, *, draws, seed):
        found = super().simulate(model, plan, draws=draws, seed=seed)
        paths = found.paths.copy()
        paths[:, 1:] += 6.0 * model.sigma
        return MacroSimulation(
            ids=found.ids, months=found.months, paths=paths, history=found.history
        )


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
        "[[episodes]]",
        'id = "R1"',
        'start = "2023-06"',
        'half = "tuning"',
        'source = "manual"',
        'observations = ["gl.brent"]',
        "interventions = []",
        'reason = "test"',
        'sources = ["S1"]',
        "",
        "[[episodes]]",
        'id = "R2"',
        'start = "2024-03"',
        'half = "validation"',
        'source = "manual"',
        'observations = ["ru.key_rate"]',
        "interventions = []",
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
    drivers=False,
    portfolio=False,
):
    """The rolling report of the fake macro on the simulated store."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    store = simulated.series_store(tmp_path / "series", months=months, seed=14)
    macro = macro or WalkMacro()
    simulated.install(monkeypatch, macro)
    catalog = None
    if drivers or portfolio:
        simulated.share_price(store, months=months)
        catalog = simulated.catalog(tmp_path)
    held = simulated.portfolio(tmp_path) if portfolio else None
    file = plan_file(tmp_path, step=step, events=events, validation_from=validation_from)
    return rolling_report(
        plugin=macro,
        store=store,
        registry=load_registry(),
        root=tmp_path / "models",
        plan=load_episodes(file),
        known_on=KNOWN_ON,
        scope=Scope.of(store.root, catalog, held),
        only=only,
    )


def test_the_transform_of_the_facts_is_uniform_on_a_right_model(tmp_path, monkeypatch):
    found = report_of(tmp_path, monkeypatch)
    assert found["dates"] >= 20
    macro = found["levels"]["macro"]
    # A random walk estimated on its own history: the transform of the facts sits in the
    # canon's buckets within the Monte-Carlo error of a few hundred cells.
    assert macro["cells"] >= 150
    for share, expected in zip(macro["calibration"], (0.05, 0.45, 0.45, 0.05), strict=True):
        assert share == pytest.approx(expected, abs=0.08)


def test_a_right_model_ties_the_walk_and_a_biased_one_loses(tmp_path, monkeypatch):
    right = report_of(tmp_path / "right", monkeypatch)["levels"]["macro"]
    versus = right["pairs"]["model/walk"]
    assert versus["score"] == pytest.approx(1.0, abs=0.1)
    assert versus["interval"]["low"] < versus["score"] < versus["interval"]["high"]
    assert right["coverage"] == pytest.approx(0.9, abs=0.1)
    assert set(versus["by_horizon"]) >= set(range(1, 7))
    # The empty scenario has no base run apart from itself and fixes no path.
    assert set(right["pairs"]) == {"model/walk"}
    biased = report_of(tmp_path / "biased", monkeypatch, macro=Biased())["levels"]["macro"]
    assert biased["pairs"]["model/walk"]["score"] > 1.5
    # The shift pushes the facts below the forecast: the lowest bucket alone carries far
    # more than its expected five percent.
    assert biased["calibration"][0] > 0.3


def test_the_grid_splits_into_halves_and_keeps_every_row_in_one(tmp_path, monkeypatch):
    found = report_of(tmp_path, monkeypatch, validation_from="2026-01-01")
    halves = found["halves"]
    assert set(halves) == {"all", "tuning", "validation"}
    assert halves["tuning"]["cells"] + halves["validation"]["cells"] == halves["all"]["cells"]
    assert all(
        (row["date"] >= "2026-01-01") == (row["half"] == "validation") for row in found["rows"]
    )
    # A step of three months against windows of six: blocks of two dates.
    assert found["block_dates"] == 2


class Late(WalkMacro):
    def estimate(self, context):
        if context.as_of.year < 2025:
            raise ValueError("no draw keeps its radius")
        return super().estimate(context)


def test_a_date_with_enough_data_where_the_model_fails_is_a_refusal(tmp_path, monkeypatch):
    found = report_of(tmp_path, monkeypatch, macro=Late())
    assert set(found["dropped"]["refused"].values()) == {"no draw keeps its radius"}
    assert not found["dropped"]["skipped"]
    assert all(row["date"] >= "2025-01-01" for row in found["rows"])
    assert not rolling.passed(found)


def test_a_date_short_of_history_is_left_out_and_is_no_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(rolling, "_short_of_history", lambda *_: True)
    found = report_of(tmp_path, monkeypatch, macro=Late())
    assert set(found["dropped"]["skipped"].values()) == {"no draw keeps its radius"}
    assert not found["dropped"]["refused"]
    assert found["dates"] > 0


def test_short_of_history_reads_the_data(tmp_path):
    store = simulated.series_store(tmp_path / "series", months=120, seed=14)
    registry = load_registry()
    assert rolling._short_of_history(store, registry, date(2020, 2, 15))
    assert not rolling._short_of_history(store, registry, date(2029, 1, 1))


class ShortRate(WalkMacro):
    """The walk with a rate history it finds too short on every date."""

    def estimate(self, context):
        from mst.model_sdk import ShortHistory

        given = {variable.id for variable in context.registry.variables}
        if "ru.key_rate" in given:
            raise ShortHistory({"ru.key_rate": "too few changes"})
        self.variables = tuple(id_ for id_ in simulated.VARIABLES if id_ in given)
        return super().estimate(context)


def test_a_short_variable_is_left_out_of_its_dates_not_the_dates(tmp_path, monkeypatch):
    found = report_of(tmp_path, monkeypatch, macro=ShortRate())

    dropped = found["dropped"]
    assert not dropped["refused"] and not dropped["skipped"]
    assert found["dates"] > 0
    assert set(dropped["left_out"]) == {row["date"] for row in found["rows"]}
    assert all(
        reasons == {"ru.key_rate": "too few changes"} for reasons in dropped["left_out"].values()
    )
    assert {row["quantity"] for row in found["rows"]} == {"gl.brent"}


def test_the_drivers_are_judged_on_the_grid_as_on_the_episodes(tmp_path, monkeypatch):
    found = report_of(tmp_path, monkeypatch, drivers=True, months=72)

    driver = found["levels"]["driver/share_price"]
    assert driver["cells"] > 0
    assert set(driver["pairs"]) == {"model/walk", "model/naive"}
    entries = {entry["quantity"]: entry for entry in found["sanity"]["quantities"]}
    assert entries["share_price:sber"]["level"] == "driver/share_price"
    assert found["quantities"]["share_price:sber"]["scored"] == driver["cells"]


def test_the_positions_and_the_tail_are_judged_on_the_grid(tmp_path, monkeypatch):
    found = report_of(tmp_path, monkeypatch, portfolio=True, months=72)

    for level in ("position", "portfolio"):
        assert set(found["levels"][level]["pairs"]) == {"model/historical"}
    assert found["tail"]["z"] is not None
    assert len(found["tail"]["windows"]) == found["dates"] - len(found["tail"]["dropped"])


def test_the_grid_can_be_narrowed_to_some_dates(tmp_path, monkeypatch):
    everything = report_of(tmp_path / "all", monkeypatch)
    dates = sorted({row["date"] for row in everything["rows"]})[:2]
    only = [date.fromisoformat(when) for when in dates]
    found = report_of(tmp_path / "some", monkeypatch, only=only)
    assert found["dates"] == 2
    assert sorted({row["date"] for row in found["rows"]}) == dates
    assert rolling.half_of(only[0], load_episodes(plan_file(tmp_path))) == "tuning"


def test_the_decision_reads_the_two_versions_on_the_same_cells(tmp_path, monkeypatch):
    right = report_of(tmp_path / "right", monkeypatch, validation_from="2026-01-01")
    biased = report_of(
        tmp_path / "biased", monkeypatch, macro=Biased(), validation_from="2026-01-01"
    )
    against = tmp_path / "incumbent.json"
    against.write_text(json.dumps(right, default=jsonable), encoding="utf-8")
    verdict = rolling_decision(biased, against, seed=1)
    assert verdict["half"] == "validation"
    assert verdict["vs_incumbent"]["point"] > 1.5
    assert verdict["verdict"] in {"rejected_worse", "rejected_admission"}
    short = {**right, "rows": right["rows"][:-1]}
    against.write_text(json.dumps(short, default=jsonable), encoding="utf-8")
    with pytest.raises(ValueError, match="do not share 1 cells"):
        rolling_decision(biased, against, seed=1)
    kept = {key: value for key, value in right.items() if key != "rows"}
    against.write_text(json.dumps(kept, default=jsonable), encoding="utf-8")
    with pytest.raises(ValueError, match="pass the --details file"):
        rolling_decision(biased, against, seed=1)


def test_the_price_of_an_event_beats_the_historical_frequency(tmp_path, monkeypatch):
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
    # The fake model has one parameter set, so the price is all event: dist + log P(S)
    # stays at zero whatever the date.
    for row in found["price"]["rows"]:
        assert np.isfinite(row["parameter_part"])
        assert row["parameter_part"] == pytest.approx(0.0, abs=1e-6)


class Unreachable(WalkMacro):
    """A macro that measures its own boxes and gives the event no probability anywhere."""

    def constraint_law(self, model, plan, *, draws, seed):
        return None

    def log_box_probability(self, model, plan, low, high, *, draws, seed):
        return np.full(draws, -np.inf)


def test_an_event_the_model_cannot_reach_is_scored_not_dropped(tmp_path, monkeypatch):
    # A date where the model gives the event no probability is its worst miss: it stays in
    # the score with P(S) = 0, as it does when the probability comes from a law.
    found = report_of(
        tmp_path, monkeypatch, Unreachable(), events=[("brent-up", 1, 460.5, 1e6)], months=120
    )
    rows = found["price"]["rows"]
    assert rows and all(row["pi"] == 0.0 for row in rows)


def test_an_event_of_two_terms_reads_their_sum(tmp_path):
    from types import SimpleNamespace

    from mst.model_sdk.contracts import LinearConstraint, LinearTerm

    store = simulated.series_store(tmp_path / "series", months=24, seed=14)
    registry = load_registry()
    last = date(2020, 6, 1)
    window = SimpleNamespace(
        panel=SimpleNamespace(months=[last], ids=list(simulated.VARIABLES), y=np.zeros((1, 2))),
        store=store,
        as_of=date(2020, 7, 15),
        known_on=KNOWN_ON,
    )
    plan = load_episodes(plan_file(tmp_path))
    brent, rate = (
        registry.variable(id_).to_model(store.as_of(id_, KNOWN_ON)[7].value)
        for id_ in simulated.VARIABLES
    )
    terms = (LinearTerm("gl.brent", 2, 1.0), LinearTerm("ru.key_rate", 2, 1.0))
    total = brent + rate

    def held(low, high):
        constraint = LinearConstraint(terms=terms, low=low, high=high)
        return rolling._happened(window, registry, plan, terms, constraint)

    assert held(total - 1.0, total + 1.0) is True
    assert held(total + 1.0, total + 2.0) is False
    late = (LinearTerm("gl.brent", 40, 1.0),)
    beyond = LinearConstraint(terms=late, low=0.0, high=1e9)
    assert rolling._happened(window, registry, plan, late, beyond) is None


def test_the_block_bootstrap_is_wider_on_an_autocorrelated_series():
    rng = np.random.default_rng(5)
    keys = [f"d{k // 2}" for k in range(240)]
    plain = block_interval(rng.standard_normal(240), keys)
    walk = block_interval(np.cumsum(rng.standard_normal(240)), keys)
    assert walk[1] - walk[0] > 2 * (plain[1] - plain[0])
    # A single block carries everything: the interval closes on the mean.
    one = block_interval([1.0, 2.0, 3.0], ["a", "a", "a"])
    assert one[0] == one[1] == pytest.approx(2.0)
