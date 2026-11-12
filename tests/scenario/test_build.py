"""mst.scenario.build: the paths of an event and what validate tells about it."""

from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from mst import plugins
from mst.corpus.index import Chunk
from mst.corpus.index import write as write_index
from mst.macro.registry import load_registry
from mst.macro.version import load as load_macro
from mst.model_sdk.contracts import ConstraintLaw
from mst.model_sdk.interventions import apply_default
from mst.scenario.build import build, dist_of, plan_of, validate
from mst.scenario.contracts import Scenario
from mst.scenario.translate import calendar_month

DRAWS, HORIZON, SEED, INNER = 8, 6, 5, 2
DRIVER_INNER = 4


@pytest.fixture(scope="module")
def macro(versions):
    """The macro version loaded back, as a run receives it."""
    return load_macro(versions.models, versions.macro, plugins.registry().macro)


def month(versions, index: int) -> str:
    """The calendar month of index ``index`` of the panel, one month after its last."""
    return calendar_month(index, versions.data.months[-1].strftime("%Y-%m"))


def last(versions, variable: str) -> float:
    """The last panel coordinate of ``variable`` in the model scale."""
    return float(versions.data.y[-1, versions.data.ids.index(variable)])


def own(versions, variable: str, coordinate: float) -> float:
    """A model coordinate in the variable's own unit."""
    return float(load_registry().variable(variable).from_model(coordinate))


def scenario(**changes) -> Scenario:
    values = {"narrative": "нефть дорожает", "approved_by": "И. Иванов"}
    values.update(changes)
    return Scenario.model_validate(values)


def observation(quantity, at, *, value=None, low=None, high=None, unit="level", basis="assumption"):
    row = {
        "kind": "constraint",
        "terms": [{"quantity": quantity, "month": at, "weight": 1.0}],
        "unit": unit,
        "basis": basis,
    }
    if value is not None:
        row["value"] = value
    if low is not None or high is not None:
        row["low"] = low
        row["high"] = high
    return row


def intervention(quantity, at, kind, value, *, unit="level", basis="assumption"):
    return {
        "quantity": quantity,
        "months": [at],
        "kind": kind,
        "values": [value],
        "unit": unit,
        "reason": "test",
        "basis": basis,
    }


def paths_of(versions, macro, spec, **changes):
    options = {"draws": DRAWS, "seed": SEED, "horizon": HORIZON}
    options.update(changes)
    return build(spec, macro, versions.satellite, **options)


def test_the_paths_run_from_the_last_month_of_the_panel_for_the_horizon(versions, macro):
    paths = paths_of(versions, macro, scenario())
    registry = load_registry()
    assert paths.months[0] == versions.data.months[-1]
    assert len(paths.months) == HORIZON + 1
    assert paths.macro.shape == (DRAWS, HORIZON + 1, 3)
    assert paths.ids == versions.data.ids
    for k, name in enumerate(versions.data.ids):
        variable = registry.variable(name)
        expected = float(variable.from_model(versions.data.y[-1, k]))
        if variable.lower is not None:
            expected = max(expected, variable.lower)
        assert paths.macro[0, 0, k] == pytest.approx(expected, abs=1e-12)


def test_a_point_observation_holds_within_its_tolerance_in_every_path(versions, macro):
    # A point arrives as an interval of half-width tau, so the paths land inside the band
    # rather than on the exact value; the plan carries strict bounds with the point mark.
    target = own(versions, "gl.brent", last(versions, "gl.brent") + 5.0)
    spec = scenario(observations=[observation("gl.brent", month(versions, 2), value=target)])
    plan = plan_of(spec, macro, draws=DRAWS, seed=SEED, horizon=HORIZON)
    (constraint,) = plan.constraints
    assert constraint.point and constraint.low < constraint.high
    paths = paths_of(versions, macro, spec)
    moved = paths.shadow[:, 2, paths.ids.index("gl.brent")]
    assert np.all(moved >= constraint.low - 1e-9)
    assert np.all(moved <= constraint.high + 1e-9)
    assert constraint.high - constraint.low == pytest.approx(2 * constraint.tolerance)
    # The other variables still vary.
    assert paths.macro[:, 2, 1].std() > 0


def test_a_shift_moves_the_target_month_only(versions, macro):
    # The default structural form adds the shift to the month it names and leaves the rest of
    # the path, the calculation month included, bit for bit as the model drew it.
    plain = paths_of(versions, macro, scenario())
    shift = own(versions, "gl.brent", last(versions, "gl.brent") + 2.0) - own(
        versions, "gl.brent", last(versions, "gl.brent")
    )
    spec = scenario(interventions=[intervention("gl.brent", month(versions, 2), "shift", shift)])
    moved = paths_of(versions, macro, spec)
    column = moved.ids.index("gl.brent")
    ratio = moved.macro[:, :, column] / plain.macro[:, :, column]
    anchor = own(versions, "gl.brent", last(versions, "gl.brent"))
    assert np.allclose(ratio[:, 2], 1 + shift / anchor)
    assert np.allclose(ratio[:, [0, 1, 3, 4, 5, 6]], 1.0)


def test_a_set_installs_the_exact_value(versions, macro):
    plain = paths_of(versions, macro, scenario())
    target = own(versions, "gl.brent", last(versions, "gl.brent") + 1.0)
    spec = scenario(interventions=[intervention("gl.brent", month(versions, 2), "set", target)])
    moved = paths_of(versions, macro, spec)
    column = moved.ids.index("gl.brent")
    assert np.allclose(moved.macro[:, 2, column], target, rtol=1e-12)
    assert np.array_equal(
        moved.macro[:, [0, 1, 3, 4, 5, 6], column], plain.macro[:, [0, 1, 3, 4, 5, 6], column]
    )


class Recursive:
    """A macro plugin that applies the interventions itself, inside its own recursion."""

    def __init__(self, inner):
        self._inner = inner

    def structural_form(self, model):
        from mst.model_sdk.contracts import StructuralForm

        return StructuralForm(("gl.brent",), True)

    def simulate(self, model, plan, *, draws, seed):
        simulation = self._inner.simulate(model, plan, draws=draws, seed=seed)
        return replace(
            simulation,
            paths=apply_default(
                simulation.paths, simulation.ids, plan.interventions, inner=plan.inner
            ),
        )

    def constraint_law(self, model, plan, *, draws, seed):
        return self._inner.constraint_law(model, plan, draws=draws, seed=seed)

    def log_box_probability(self, model, plan, low, high, *, draws, seed):
        return None


def test_a_recursive_model_applies_the_intervention_itself(versions, macro):
    # A plugin that declares a structural form has already applied the intervention; the
    # platform must not add the default a second time.
    plain = paths_of(versions, macro, scenario(), inner=INNER)
    shift = own(versions, "gl.brent", last(versions, "gl.brent") + 2.0) - own(
        versions, "gl.brent", last(versions, "gl.brent")
    )
    spec = scenario(interventions=[intervention("gl.brent", month(versions, 2), "shift", shift)])
    recursive = replace(macro, plugin=Recursive(macro.plugin))
    moved = paths_of(versions, recursive, spec, inner=INNER)
    assert moved.macro.shape[0] == DRAWS * INNER
    column = moved.ids.index("gl.brent")
    anchor = own(versions, "gl.brent", last(versions, "gl.brent"))
    step = moved.macro[:, 2, column, None] / plain.macro[:, 2, column, None]
    assert np.allclose(step, 1 + shift / anchor)


def driver_of(versions) -> tuple[str, int]:
    """A served driver and the level of its quantity."""
    driver = sorted(versions.satellite.served())[0]
    quantity = driver.split(":", 1)[0]
    return driver, versions.satellite.registry.quantity(quantity).level


def driver_band(versions, macro, driver, first, second, inner=DRIVER_INNER):
    """A band of the driver's own unit in month 2, from two quantiles of its free paths."""
    plain = paths_of(versions, macro, scenario(), inner=inner)
    column = plain.drivers.paths[:, 2, plain.drivers.ids.index(driver)]
    return float(np.percentile(column, first)), float(np.percentile(column, second))


def test_a_condition_on_a_driver_keeps_the_survivors_and_prices_its_level(versions, macro):
    driver, level = driver_of(versions)
    low, high = driver_band(versions, macro, driver, 5, 95)
    spec = scenario(observations=[observation(driver, month(versions, 2), low=low, high=high)])
    paths = paths_of(versions, macro, spec, inner=DRIVER_INNER)
    column = paths.drivers.paths[:, 2, paths.drivers.ids.index(driver)]
    assert np.all(column >= low - 1e-9)
    assert np.all(column <= high + 1e-9)
    # The price counts on its own level of the hierarchy, above the macro variables.
    assert paths.dist is not None and paths.dist.distance > 0
    assert paths.dist.by_level == {level: pytest.approx(paths.dist.distance)}
    # The condition resamples the macro paths with the survivors instead of drawing new ones.
    plain = paths_of(versions, macro, scenario(), inner=DRIVER_INNER)
    kept = {row.tobytes() for row in plain.macro.reshape(-1, plain.macro.shape[-1])}
    assert all(row.tobytes() in kept for row in paths.macro.reshape(-1, paths.macro.shape[-1]))


def test_a_group_without_a_driver_survivor_contradicts_the_models(versions, macro):
    from mst.model_sdk.dist import ContradictsModels

    driver, _ = driver_of(versions)
    spec = scenario(observations=[observation(driver, month(versions, 2), low=1e6, high=1e7)])
    with pytest.raises(ContradictsModels, match="contradicts the models"):
        paths_of(versions, macro, spec, inner=DRIVER_INNER)


def test_a_scenario_without_observations_has_no_price(versions, macro):
    paths = paths_of(versions, macro, scenario())
    assert paths.dist is None  # no price, not a zero distance


def test_a_support_observation_carries_the_price_of_the_event(versions, macro):
    # The bounds are the event's own, in its own unit; the price is E_P[-log P(S | R)] over
    # the constraint law of the model, with the standard error and log P(S) beside it.
    anchor = last(versions, "gl.brent")
    spec = scenario(
        observations=[
            observation(
                "gl.brent",
                month(versions, 2),
                low=own(versions, "gl.brent", anchor - 1.0),
                high=own(versions, "gl.brent", anchor + 1.0),
            )
        ]
    )
    paths = paths_of(versions, macro, spec)
    assert paths.dist is not None and paths.dist.message is None
    assert np.isfinite(paths.dist.distance) and paths.dist.distance > 0
    assert paths.dist.standard_error >= 0
    assert -paths.dist.log_probability <= paths.dist.distance  # Jensen
    assert paths.dist.tolerances == ()
    assert paths.dist.one_sided is not None


def test_the_checks_report_the_share_inside_the_bounds(versions, macro):
    anchor = last(versions, "gl.brent")
    spec = scenario(
        observations=[
            observation(
                "gl.brent",
                month(versions, 2),
                low=own(versions, "gl.brent", anchor - 1.0),
                high=own(versions, "gl.brent", anchor + 1.0),
            )
        ]
    )
    paths = paths_of(versions, macro, spec)
    (constraint,) = plan_of(spec, macro, draws=DRAWS, seed=SEED, horizon=HORIZON).constraints
    column = paths.shadow[:, 2, paths.ids.index("gl.brent")]
    inside = np.mean((column >= constraint.low) & (column <= constraint.high))
    assert paths.checks["support_share"] == pytest.approx(float(inside))


def test_a_wide_observation_costs_less_than_a_narrow_one(versions, macro):
    anchor = last(versions, "gl.brent")

    def price(half):
        spec = scenario(
            observations=[
                observation(
                    "gl.brent",
                    month(versions, 2),
                    low=own(versions, "gl.brent", anchor - half),
                    high=own(versions, "gl.brent", anchor + half),
                )
            ]
        )
        found = dist_of(spec, macro, draws=40, seed=SEED, horizon=HORIZON)
        return found.distance

    assert price(50.0) < price(0.5)


def test_an_expectation_condition_tilts_the_paths_and_prices_them(versions, macro):
    # The mean of the key rate in month 2 is asked above its unconditional level: the tilt
    # raises it there, the price carries the tilt, and the checks report the mean and the
    # effective size. A mean condition needs an identity quantity; log100 prices would mix
    # E(log X) with log E(X).
    plain = paths_of(versions, macro, scenario(), inner=8)
    column = plain.macro[:, 2, plain.ids.index("ru.key_rate")]
    target = float(column.mean())
    spec = scenario(
        observations=[
            {
                "kind": "expectation",
                "terms": [{"quantity": "ru.key_rate", "month": month(versions, 2), "weight": 1.0}],
                "unit": "level",
                "low": target + 1.0,
                "basis": "assumption",
            }
        ]
    )
    tilted = paths_of(versions, macro, spec, inner=8)
    moved = float(tilted.macro[:, 2, tilted.ids.index("ru.key_rate")].mean())
    assert moved == pytest.approx(target + 1.0, abs=0.15)
    assert tilted.effective_size < tilted.draws * 8
    assert tilted.checks["means"][0]["active"] is True
    assert tilted.dist is not None and tilted.dist.distance > 0
    # Without the condition there is no tilt to report.
    assert plain.effective_size is None and plain.checks == {}


def test_a_probability_condition_raises_the_share_of_the_event(versions, macro):
    plain = paths_of(versions, macro, scenario(), inner=8)
    column = plain.macro[:, 2, plain.ids.index("gl.brent")]
    low, high = np.percentile(column, [10, 60])
    spec = scenario(
        observations=[
            {
                "kind": "probability",
                "terms": [{"quantity": "gl.brent", "month": month(versions, 2), "weight": 1.0}],
                "unit": "level",
                "low": float(low),
                "high": float(high),
                "at_least": 0.9,
                "basis": "assumption",
            }
        ]
    )
    tilted = paths_of(versions, macro, spec, inner=8)
    moved = tilted.macro[:, 2, tilted.ids.index("gl.brent")]
    inside = np.mean((moved >= low) & (moved <= high))
    assert inside >= 0.88
    assert tilted.checks["effective_size"] > 0


def test_an_expectation_no_tilt_can_meet_is_refused(versions, macro):
    plain = paths_of(versions, macro, scenario(), inner=4)
    column = plain.macro[:, 2, plain.ids.index("ru.key_rate")]
    spec = scenario(
        observations=[
            {
                "kind": "expectation",
                "terms": [{"quantity": "ru.key_rate", "month": month(versions, 2), "weight": 1.0}],
                "unit": "level",
                "low": float(column.max()) + 10.0,
                "basis": "assumption",
            }
        ]
    )
    with pytest.raises(ValueError, match="cannot be met"):
        paths_of(versions, macro, spec, inner=4)


def test_the_paths_carry_the_size_of_the_driving_shocks(versions, macro):
    target = own(versions, "gl.brent", last(versions, "gl.brent") + 5.0)
    paths = paths_of(
        versions,
        macro,
        scenario(observations=[observation("gl.brent", month(versions, 3), value=target)]),
    )
    assert len(paths.diagnostics["driving_sd"]["median_by_month"]) == HORIZON


def test_the_paths_repeat_bit_for_bit_for_the_same_seed(versions, macro):
    spec = scenario(
        observations=[
            observation("gl.brent", month(versions, 2), value=last(versions, "gl.brent") + 5.0)
        ]
    )
    first = paths_of(versions, macro, spec)
    again = paths_of(versions, macro, spec)
    assert np.array_equal(first.macro, again.macro)
    assert np.array_equal(first.drivers.paths, again.drivers.paths)
    other = paths_of(versions, macro, spec, seed=SEED + 1)
    assert not np.array_equal(first.macro, other.macro)


class FixedLaw:
    """A macro plugin whose constraint functional does not move: zero in every group."""

    def constraint_law(self, model, plan, *, draws, seed):
        return ConstraintLaw(
            means=np.zeros((draws, 1)),
            covs=np.zeros((draws, 1, 1)),
            log_weight=np.zeros(draws),
            group=np.arange(draws),
        )


class NoLaw:
    """A macro plugin that simulates but measures neither the law nor its own boxes."""

    def __init__(self, inner):
        self._inner = inner

    def simulate(self, model, plan, *, draws, seed):
        return self._inner.simulate(model, plan, draws=draws, seed=seed)

    def constraint_law(self, model, plan, *, draws, seed):
        return None

    def log_box_probability(self, model, plan, low, high, *, draws, seed):
        return None


def test_an_unreachable_box_has_an_infinite_price(versions, macro):
    # The functional is fixed by the model at zero and the box excludes it: the event has no
    # probability at all and the price is infinite with the message.
    spec = scenario(
        observations=[
            observation(
                "gl.brent",
                month(versions, 2),
                low=own(versions, "gl.brent", last(versions, "gl.brent") + 100.0),
                high=own(versions, "gl.brent", last(versions, "gl.brent") + 101.0),
            )
        ]
    )
    fixed = replace(macro, plugin=FixedLaw())
    found = dist_of(spec, fixed, draws=DRAWS, seed=SEED, horizon=HORIZON)
    assert found.distance == float("inf")
    assert "no probability" in found.message


def test_a_model_without_a_law_and_without_boxes_has_no_price(versions, macro):
    spec = scenario(
        observations=[
            observation(
                "gl.brent",
                month(versions, 2),
                low=own(versions, "gl.brent", last(versions, "gl.brent") - 1.0),
                high=own(versions, "gl.brent", last(versions, "gl.brent") + 1.0),
            )
        ]
    )
    stripped = replace(macro, plugin=NoLaw(macro.plugin))
    assert dist_of(spec, stripped, draws=DRAWS, seed=SEED, horizon=HORIZON) is None


def test_validate_accepts_a_scenario_the_model_can_compute(versions, macro):
    spec = scenario(
        observations=[
            observation("gl.brent", month(versions, 2), value=last(versions, "gl.brent") + 5.0),
            observation(
                "gl.brent",
                month(versions, 3),
                low=last(versions, "gl.brent"),
                high=last(versions, "gl.brent") + 10.0,
            ),
        ]
    )
    report = validate(spec, macro, versions.satellite, as_of=macro.as_of)
    assert report["ok"] is True and report["problems"] == []


def test_validate_names_a_scenario_without_an_approver(versions, macro):
    spec = scenario(
        observations=[observation("gl.brent", month(versions, 2), value=1.0)],
        approved_by=None,
    )
    report = validate(spec, macro, versions.satellite, as_of=macro.as_of)
    assert report["ok"] is False
    assert "names no approver" in "; ".join(report["problems"])


def test_validate_names_an_unknown_quantity(versions, macro):
    spec = scenario(observations=[observation("gl.nowhere", month(versions, 2), value=1.0)])
    report = validate(spec, macro, versions.satellite, as_of=macro.as_of)
    assert report["ok"] is False
    assert "unknown quantity gl.nowhere" in "; ".join(report["problems"])


def test_validate_names_a_month_beyond_the_horizon(versions, macro):
    spec = scenario(observations=[observation("gl.brent", month(versions, 13), value=1.0)])
    report = validate(spec, macro, versions.satellite, as_of=macro.as_of)
    assert report["ok"] is False
    assert any("beyond the horizon of 12" in problem for problem in report["problems"])


def test_validate_names_the_satellites_of_another_macro_version(versions, macro):
    elsewhere = replace(versions.satellite, macro="0" * 64)
    report = validate(scenario(), macro, elsewhere, as_of=macro.as_of)
    assert report["ok"] is False
    assert "estimated on macro version" in "; ".join(report["problems"])


def test_validate_labels_what_the_event_states_and_what_it_assumes(versions, macro):
    spec = scenario(
        sources=["S1#1/1"],
        observations=[
            observation("gl.brent", month(versions, 2), value=1.0, basis="event"),
            observation("gl.brent", month(versions, 3), value=1.0, basis=["S1#1/1"]),
        ],
        interventions=[
            intervention("ru.key_rate", month(versions, 2), "shift", 1.0, basis="assumption")
        ],
    )
    report = validate(spec, macro, versions.satellite, as_of=macro.as_of)
    assert report["labels"] == {"event": 1, "sourced": 1, "assumption": 1}
    assert report["warnings"] == []


def test_validate_warns_about_a_scenario_that_says_nothing(versions, macro):
    report = validate(scenario(), macro, versions.satellite, as_of=macro.as_of)
    assert report["warnings"] == [
        "the narrative cites no source",
        "no observation and no intervention: this is the base scenario",
    ]


def test_validate_checks_that_the_cited_fragments_are_visible_on_the_day(versions, macro, tmp_path):
    chunk = Chunk(
        chunk_id="S1/2020#1/1",
        doc_id="S1",
        edition="2020",
        kind="scenario_set",
        region="ru",
        jurisdiction="ru",
        visible_from="2020-03-01",
        title="ОДКП",
        heading_path=["Сценарии"],
        pages=["12"],
        factors=["fx"],
        text="Рисковый сценарий.",
        sha256="a" * 64,
    )
    write_index([chunk], tmp_path / "index")
    spec = scenario(sources=["S1/2020#1/1"])
    assert validate(spec, macro, versions.satellite, as_of=date(2020, 4, 1), corpus=tmp_path)["ok"]
    report = validate(spec, macro, versions.satellite, as_of=date(2020, 1, 1), corpus=tmp_path)
    assert report["ok"] is False and "S1/2020#1/1" in report["problems"][0]
    # A fragment an observation rests on is checked the same way as the scenario's sources.
    assumed = scenario(
        observations=[observation("gl.brent", month(versions, 2), value=1.0, basis=["S1/2020#1/1"])]
    )
    report = validate(assumed, macro, versions.satellite, as_of=date(2020, 1, 1), corpus=tmp_path)
    assert report["ok"] is False and "S1/2020#1/1" in report["problems"][0]
