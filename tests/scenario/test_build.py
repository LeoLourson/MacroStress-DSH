"""mst.scenario.build: the paths of a scenario and what validate tells about it."""

from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from mst.corpus.index import Chunk
from mst.corpus.index import write as write_index
from mst.macro.registry import load_registry
from mst.model_sdk import InfeasibleScenario
from mst.model_sdk.contracts import ConstraintLaw
from mst.plugins.hier.simulate import IMPACT
from mst.scenario.build import build, dist_of, plan_of, tolerance_of, validate
from mst.scenario.contracts import Impact
from mst.scenario.spec import Spec

DRAWS, HORIZON, SEED = 8, 6, 5


def scenario(versions, **changes) -> Spec:
    values = {
        "narrative": "нефть дорожает",
        "horizon": HORIZON,
        "macro": versions.macro,
        "satellites": versions.satellites,
        "approved_by": "И. Иванов",
    }
    values.update(changes)
    return Spec.model_validate(values)


def fixed(variable: str, month: int, value: float) -> dict:
    return {
        "terms": [{"variable": variable, "month": month, "weight": 1.0}],
        "value": value,
        "basis": "assumption",
    }


def interval(variable: str, month: int, low: float, high: float) -> dict:
    return {
        "terms": [{"variable": variable, "month": month, "weight": 1.0}],
        "low": low,
        "high": high,
        "basis": "assumption",
    }


def change(variable: str, month: int, other: int, value: float) -> dict:
    return {
        "terms": [
            {"variable": variable, "month": month, "weight": 1.0},
            {"variable": variable, "month": other, "weight": -1.0},
        ],
        "value": value,
        "basis": "assumption",
    }


def paths_of(versions, spec):
    return build(spec, versions.loaded, versions.satellite, draws=DRAWS, seed=SEED)


def test_the_paths_run_from_the_last_month_of_the_panel_for_the_horizon(versions):
    paths = paths_of(versions, scenario(versions))
    registry = load_registry()
    assert paths.months[0] == versions.data.months[-1]
    assert len(paths.months) == HORIZON + 1
    assert paths.macro.shape == (DRAWS, HORIZON + 1, 3)
    assert paths.ids == versions.data.ids
    for k, id_ in enumerate(versions.data.ids):
        assert paths.macro[0, 0, k] == pytest.approx(
            registry.variable(id_).from_model(versions.data.y[-1, k]), abs=1e-12
        )


def test_point_constraints_hold_within_the_tolerance_in_every_path(versions):
    # A point arrives as an interval of half-width tau, so the paths land inside the band
    # rather than on the exact value; the plan carries strict bounds with the point mark.
    registry = load_registry()
    target = float(versions.data.y[-1, 0] + 5.0)
    spec = scenario(versions, constraints=[fixed("gl.brent", 2, target)])
    plan = plan_of(spec, versions.loaded)
    (constraint,) = plan.constraints
    assert constraint.point and constraint.low < constraint.high
    assert constraint.tolerance == pytest.approx(
        tolerance_of(constraint.terms, versions.loaded)
    )
    paths = paths_of(versions, spec)
    moved = registry.variable("gl.brent").to_model(paths.macro[:, 2, 0])
    assert np.all(np.abs(moved - target) <= constraint.tolerance + 1e-9)
    assert paths.macro[:, 2, 1].std() > 0  # the other variables still vary


def test_an_impact_on_a_constrained_variable_meets_the_constraint(versions):
    # The impact is added after the model has drawn the paths, so the constraint C must hold on
    # the shifted trajectory F(omega, O) = G_0 + o_0: the target is met by the final path. The
    # impact varies by month, so the shift the plan reads is the value of that month alone.
    registry = load_registry()
    target = float(versions.data.y[-1, 0] + 5.0)
    impact = Impact(
        target="gl.brent", start=2, values=[2.0, 1.0], reason="r", basis=["s"]
    )
    spec = scenario(versions, constraints=[fixed("gl.brent", 3, target)], impacts=[impact])
    report = validate(spec, versions.loaded, versions.satellite, as_of=versions.data.months[-1])
    assert report["ok"] is True and report["problems"] == []
    paths = paths_of(versions, spec)
    moved = registry.variable("gl.brent").to_model(paths.macro[:, 3, 0])
    (constraint,) = plan_of(spec, versions.loaded).constraints
    assert np.all(np.abs(moved - target) <= constraint.tolerance + 1e-9)


def test_a_change_constraint_holds_across_months(versions):
    spec = scenario(versions, constraints=[change("ru.key_rate", 3, 1, 2.0)])
    (constraint,) = plan_of(spec, versions.loaded).constraints
    paths = paths_of(versions, spec)
    difference = paths.macro[:, 3, 1] - paths.macro[:, 1, 1]
    assert np.all(np.abs(difference - 2.0) <= constraint.tolerance + 1e-9)


def test_the_paths_repeat_bit_for_bit_for_the_same_seed(versions):
    spec = scenario(versions, constraints=[fixed("gl.brent", 2, 1.0)])
    first = paths_of(versions, spec)
    again = paths_of(versions, spec)
    assert np.array_equal(first.macro, again.macro)
    assert np.array_equal(first.drivers.paths, again.drivers.paths)
    other = build(spec, versions.loaded, versions.satellite, draws=DRAWS, seed=SEED + 1)
    assert not np.array_equal(first.macro, other.macro)


def test_a_scenario_without_constraints_has_no_price(versions):
    paths = paths_of(versions, scenario(versions))
    assert paths.dist is None  # no price, not a zero distance


def test_the_paths_carry_the_price_of_the_constraints(versions):
    # The price is E_P[-log P(S | R)] over the constraint law of the model, with the standard
    # error of the groups and log P(S) beside it; a plain interval constraint has no
    # tolerances and names no causes.
    spec = scenario(versions, constraints=[interval("gl.brent", 2, 1.0, 3.0)])
    paths = paths_of(versions, spec)
    assert paths.dist is not None
    assert paths.dist.message is None
    assert paths.dist.distance > 0 and np.isfinite(paths.dist.distance)
    assert paths.dist.standard_error >= 0
    assert -paths.dist.log_probability <= paths.dist.distance  # Jensen
    assert paths.dist.causes == ()
    assert paths.dist.tolerances == ()


def test_a_point_constraint_carries_its_tolerance_into_the_price(versions):
    spec = scenario(versions, constraints=[fixed("gl.brent", 2, 1.0)])
    found = dist_of(spec, versions.loaded, draws=DRAWS, seed=SEED)
    (tolerance,) = found.tolerances
    assert tolerance == pytest.approx(
        tolerance_of(plan_of(spec, versions.loaded).constraints[0].terms, versions.loaded)
    )


def test_named_causes_cost_no_less_than_unnamed(versions):
    # Explaining the event through one named shock narrows R, so the price cannot fall;
    # the difference is within the Monte-Carlo noise of the two estimates.
    spec = scenario(versions, constraints=[interval("gl.brent", 2, 0.0, 1.0)])
    unnamed = dist_of(spec, versions.loaded, draws=200, seed=SEED)
    named = dist_of(
        scenario(versions, constraints=spec.constraints, causes=["gl.oil"]),
        versions.loaded,
        draws=200,
        seed=SEED,
    )
    assert named.causes == ("gl.oil",)
    assert named.distance >= unnamed.distance - 2 * (
        named.standard_error + unnamed.standard_error
    )


def test_a_far_box_has_a_large_finite_price(versions):
    # A box far in the tail: the GHK sampler keeps the answer finite where a path share
    # would read zero, and the price is large.
    spec = scenario(versions, constraints=[interval("gl.brent", 2, 9.0, 9.5)])
    found = dist_of(spec, versions.loaded, draws=DRAWS, seed=SEED)
    assert np.isfinite(found.distance) and found.distance > 1.0
    assert found.message is None


def test_an_unreachable_box_contradicts_the_models(versions):
    # The functional the box constrains is fixed by the model at a value the box excludes:
    # the event has no probability at all and the price is infinite with the message.
    spec = scenario(versions, constraints=[interval("gl.brent", 2, 100.0, 101.0)])
    fixed = replace(versions.loaded, plugin=FixedLaw())
    found = dist_of(spec, fixed, draws=DRAWS, seed=SEED)
    assert found.distance == float("inf")
    assert "contradicts the models" in found.message


def test_a_far_box_of_a_gaussian_law_is_dear_but_has_a_price(versions):
    # A gaussian law gives every box a probability, however far: the price is large and finite
    # in the upper tail as in the lower one.
    above = scenario(versions, constraints=[interval("gl.brent", 2, 100.0, 101.0)])
    found = dist_of(above, versions.loaded, draws=DRAWS, seed=SEED)
    assert np.isfinite(found.distance) and found.distance > 100.0
    assert found.message is None


def test_a_model_without_a_law_has_no_price(versions):
    # A macro model that does not measure the constraint law leaves the price absent.
    spec = scenario(versions, constraints=[interval("gl.brent", 2, 1.0, 3.0)])
    stripped = replace(versions.loaded, plugin=NoLaw(versions.loaded.plugin))
    assert dist_of(spec, stripped, draws=DRAWS, seed=SEED) is None


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


def test_the_price_reads_the_impact_shifted_bounds(versions):
    # The impact moves the constrained variable, so the plan the model solves — and the price
    # with it — reads the box with the impact taken out: the same constraint with the impact
    # prices the event the paths actually carry, not the unshifted one.
    impact = Impact(
        target="gl.brent", start=1, values=[10.0, 10.0, 10.0], reason="шок", basis=["s"]
    )
    plain = scenario(versions, constraints=[interval("gl.brent", 2, 1.0, 3.0)])
    shifted = scenario(
        versions, constraints=plain.constraints, impacts=[impact]
    )
    found_plain = dist_of(plain, versions.loaded, draws=100, seed=SEED)
    found_shifted = dist_of(shifted, versions.loaded, draws=100, seed=SEED)
    assert found_shifted.distance > found_plain.distance + 0.5


def test_narrowing_an_interval_raises_the_price_and_not_the_one_sided_price(versions):
    # Both boxes share their upper bound and sit below the median of the functional, so
    # they open into the same ray: the price grows with the narrowing, the one-sided price
    # of the ray does not.
    wide = scenario(versions, constraints=[interval("gl.brent", 2, -6.0, -2.0)])
    narrow = scenario(versions, constraints=[interval("gl.brent", 2, -4.0, -2.0)])
    found_wide = dist_of(wide, versions.loaded, draws=200, seed=SEED)
    found_narrow = dist_of(narrow, versions.loaded, draws=200, seed=SEED)
    assert found_narrow.distance > found_wide.distance
    assert found_wide.one_sided is not None and found_narrow.one_sided is not None
    assert found_wide.one_sided.dropped == () and found_narrow.one_sided.dropped == ()
    assert found_narrow.one_sided.distance == pytest.approx(
        found_wide.one_sided.distance, abs=0.15
    )
    assert found_narrow.one_sided.distance <= found_narrow.distance + 2 * (
        found_narrow.standard_error + found_narrow.one_sided.standard_error
    )


def test_an_interval_that_holds_the_median_is_dropped_from_the_one_sided_event(versions):
    # A box that holds the median of the functional: the price is the box's, the one-sided
    # event holds nothing of it, so the one-sided price is zero with the coordinate named.
    spec = scenario(versions, constraints=[interval("gl.brent", 2, 1.0, 8.0)])
    found = dist_of(spec, versions.loaded, draws=200, seed=SEED)
    assert found.distance > 0.0
    assert found.one_sided is not None
    assert found.one_sided.dropped == (1,)
    assert found.one_sided.distance == 0.0
    assert found.one_sided.log_probability == 0.0
    assert found.one_sided.standard_error == 0.0


def test_the_one_sided_price_is_at_most_the_price(versions):
    # A+ holds A, so the price of the ray is below the price of the box, within the error
    # of the two estimates.
    spec = scenario(versions, constraints=[interval("gl.brent", 2, 1.0, 3.0)])
    found = dist_of(spec, versions.loaded, draws=200, seed=SEED)
    assert found.one_sided is not None
    assert found.one_sided.distance <= found.distance + 2 * (
        found.standard_error + found.one_sided.standard_error
    )


def test_a_scenario_of_conditions_on_means_has_no_one_sided_price(versions):
    # Conditions on means and probabilities carry no interval to open into a ray, so the
    # one-sided price is absent even beside a support condition.
    spec = scenario(
        versions,
        inner=4,
        constraints=[interval("gl.brent", 2, 1.0, 3.0)],
        expectations=[
            {
                "terms": [{"variable": "gl.brent", "month": 2, "weight": 1.0}],
                "low": 1.0,
                "basis": "assumption",
            }
        ],
    )
    found = dist_of(spec, versions.loaded, draws=200, seed=SEED)
    assert found is not None and found.one_sided is None


def test_a_wide_box_gives_a_price_near_zero(versions):
    wide = scenario(versions, constraints=[interval("gl.brent", 2, -50.0, 50.0)])
    narrow = scenario(versions, constraints=[interval("gl.brent", 2, 1.0, 3.0)])
    assert dist_of(wide, versions.loaded, draws=40, seed=SEED).distance < 0.1
    assert dist_of(narrow, versions.loaded, draws=40, seed=SEED).distance > dist_of(
        wide, versions.loaded, draws=40, seed=SEED
    ).distance


def test_an_expectation_condition_tilts_the_paths_and_prices_them(versions):
    # The mean of the oil price in month 2 is asked to sit above its unconditional level:
    # the tilt raises it there, the price carries the tilt beside the support price, and
    # the checks report the mean and the effective size. The bounds live in the model
    # scale, like every condition's.
    registry = load_registry()
    plain = paths_of(versions, scenario(versions))
    level = float(np.mean(registry.variable("gl.brent").to_model(plain.macro[:, 2, 0])))
    target = level + 1.0
    spec = scenario(
        versions,
        inner=8,
        expectations=[
            {
                "terms": [{"variable": "gl.brent", "month": 2, "weight": 1.0}],
                "low": target,
                "basis": "assumption",
            }
        ],
    )
    tilted = paths_of(versions, spec)
    moved = float(
        np.mean(registry.variable("gl.brent").to_model(tilted.macro[:, 2, 0]))
    )
    assert moved == pytest.approx(target, abs=0.15)
    assert tilted.effective_size < tilted.draws * spec.inner
    assert tilted.checks["means"][0]["active"] is True
    assert tilted.dist is not None and tilted.dist.distance > 0
    # Without the condition there is no tilt to report.
    assert plain.effective_size is None and plain.checks == {}


def test_a_macro_impact_keeps_the_groups_the_tilt_runs_in(versions):
    # The impact rebuilds the paths; the groups of R must survive it, or every path becomes its
    # own group and no tilt can move anything.
    registry = load_registry()
    impact = Impact(
        target="gl.brent", start=1, values=[1.0, 1.0, 1.0], reason="шок", basis=["S1#1/1"]
    )
    plain = paths_of(versions, scenario(versions, impacts=[impact]))
    level = float(np.mean(registry.variable("gl.brent").to_model(plain.macro[:, 2, 0])))
    spec = scenario(
        versions,
        inner=8,
        impacts=[impact],
        expectations=[
            {
                "terms": [{"variable": "gl.brent", "month": 2, "weight": 1.0}],
                "low": level + 1.0,
                "basis": "assumption",
            }
        ],
    )
    tilted = paths_of(versions, spec)
    moved = float(np.mean(registry.variable("gl.brent").to_model(tilted.macro[:, 2, 0])))
    assert moved == pytest.approx(level + 1.0, abs=0.15)


def test_a_probability_condition_raises_the_share_of_the_event(versions):
    registry = load_registry()
    plain = paths_of(versions, scenario(versions))
    column = registry.variable("gl.brent").to_model(plain.macro[:, 2, 0])
    low, high = np.percentile(column, [10, 60])
    spec = scenario(
        versions,
        inner=8,
        probabilities=[
            {
                "terms": [{"variable": "gl.brent", "month": 2, "weight": 1.0}],
                "low": float(low),
                "high": float(high),
                "at_least": 0.9,
                "basis": "assumption",
            }
        ],
    )
    tilted = paths_of(versions, spec)
    moved = registry.variable("gl.brent").to_model(tilted.macro[:, 2, 0])
    inside = np.mean((moved >= low) & (moved <= high))
    assert inside >= 0.88
    assert tilted.checks["effective_size"] > 0


def test_an_expectation_no_tilt_can_meet_is_refused(versions):
    registry = load_registry()
    plain = paths_of(versions, scenario(versions))
    highest = float(
        registry.variable("gl.brent").to_model(plain.macro[:, 2, 0]).max()
    )
    spec = scenario(
        versions,
        inner=4,
        expectations=[
            {
                "terms": [{"variable": "gl.brent", "month": 2, "weight": 1.0}],
                "low": highest + 10.0,
                "basis": "assumption",
            }
        ],
    )
    with pytest.raises(ValueError, match="cannot be met"):
        paths_of(versions, spec)


def test_driver_paths_are_in_the_drivers_own_units_and_carry_impacts(versions):
    impact = Impact(
        target="price:alpha", start=2, values=[10.0, 10.0], reason="перебой", basis=["S1#1/1"]
    )
    plain = paths_of(versions, scenario(versions))
    with_impact = paths_of(versions, scenario(versions, impacts=[impact]))
    assert with_impact.drivers.ids == sorted(versions.satellite.served())
    k = with_impact.drivers.ids.index("price:alpha")
    difference = with_impact.drivers.paths[:, :, k] - plain.drivers.paths[:, :, k]
    assert np.allclose(difference[:, 2:4], 10.0) and np.allclose(difference[:, [0, 4, 5, 6]], 0)
    assert IMPACT in with_impact.drivers.flags["price:alpha"]
    assert with_impact.drivers.flags["price:alpha"] <= {
        IMPACT, "outside_training_range"
    }  # fmt: skip


def test_an_impact_on_a_macro_variable_reaches_the_drivers(versions):
    # A rise of the oil price in the model scale moves the drivers that read it — the
    # impact enters level 0 before the satellites see the paths.
    impact = Impact(
        target="gl.brent", start=1, values=[2.0, 2.0, 2.0], reason="шок", basis=["S1#1/1"]
    )
    plain = paths_of(versions, scenario(versions))
    with_impact = paths_of(versions, scenario(versions, impacts=[impact]))
    macro_k = with_impact.ids.index("gl.brent")
    # Brent is log100 in the model scale: a shift of 2 is a ratio of exp(0.02) in own units.
    assert np.allclose(
        with_impact.macro[:, 1:4, macro_k] / plain.macro[:, 1:4, macro_k], np.exp(2.0 / 100)
    )
    assert np.allclose(with_impact.macro[:, [0, 4], macro_k], plain.macro[:, [0, 4], macro_k])
    changed = {
        driver
        for k, driver in enumerate(with_impact.drivers.ids)
        if not np.allclose(with_impact.drivers.paths[:, :, k], plain.drivers.paths[:, :, k])
    }
    assert changed, "an impact on the oil price must move the drivers that read it"


def test_an_impact_on_a_driver_the_satellites_do_not_know_is_refused(versions):
    impact = Impact(target="price:nowhere", start=1, values=[1.0, 1.0], reason="r", basis=["s"])
    with pytest.raises(ValueError, match="knows no such driver"):
        paths_of(versions, scenario(versions, impacts=[impact]))


def test_an_impact_on_a_variable_the_macro_version_lacks_is_refused(versions):
    impact = Impact(target="ru.nowhere", start=1, values=[1.0, 1.0], reason="r", basis=["s"])
    report = validate(
        scenario(versions, impacts=[impact]),
        versions.loaded,
        versions.satellite,
        as_of=versions.data.months[-1],
    )
    assert report["ok"] is False
    assert "the macro version has no such variable" in "; ".join(report["problems"])
    # build names the same variable instead of failing on a raw list index.
    with pytest.raises(ValueError, match="the macro version has no such variable"):
        paths_of(versions, scenario(versions, impacts=[impact]))


def test_a_decaying_path_on_a_driver_lands_on_exactly_those_months(versions):
    impact = Impact(
        target="price:alpha", start=2, values=[4.0, 2.0, 1.0], reason="r", basis=["s"]
    )
    plain = paths_of(versions, scenario(versions))
    with_impact = paths_of(versions, scenario(versions, impacts=[impact]))
    k = with_impact.drivers.ids.index("price:alpha")
    difference = with_impact.drivers.paths[:, :, k] - plain.drivers.paths[:, :, k]
    assert np.allclose(difference[:, 2], 4.0)
    assert np.allclose(difference[:, 3], 2.0)
    assert np.allclose(difference[:, 4], 1.0)
    assert np.allclose(difference[:, [0, 1, 5, 6]], 0.0)
    assert IMPACT in with_impact.drivers.flags["price:alpha"]


def test_a_decaying_path_on_a_macro_variable_lands_on_exactly_those_months(versions):
    impact = Impact(target="gl.brent", start=2, values=[2.0, 1.0], reason="r", basis=["s"])
    plain = paths_of(versions, scenario(versions))
    with_impact = paths_of(versions, scenario(versions, impacts=[impact]))
    macro_k = with_impact.ids.index("gl.brent")
    # Brent is log100 in the model scale: a shift of 2 is a ratio of exp(0.02) in own units.
    ratio = with_impact.macro[:, :, macro_k] / plain.macro[:, :, macro_k]
    assert np.allclose(ratio[:, 2], np.exp(2.0 / 100))
    assert np.allclose(ratio[:, 3], np.exp(1.0 / 100))
    assert np.allclose(ratio[:, [0, 1, 4, 5, 6]], 1.0)


def test_two_overlapping_impacts_add_up(versions):
    first = Impact(target="price:alpha", start=1, values=[3.0, 3.0], reason="r", basis=["s"])
    second = Impact(
        target="price:alpha", start=2, values=[4.0, 4.0, 4.0], reason="r", basis=["s"]
    )
    plain = paths_of(versions, scenario(versions))
    both = paths_of(versions, scenario(versions, impacts=[first, second]))
    k = both.drivers.ids.index("price:alpha")
    difference = both.drivers.paths[:, :, k] - plain.drivers.paths[:, :, k]
    assert np.allclose(difference[:, 1], 3.0)
    assert np.allclose(difference[:, 2], 7.0)
    assert np.allclose(difference[:, 3], 4.0)
    assert np.allclose(difference[:, 4], 4.0)
    assert np.allclose(difference[:, [0, 5, 6]], 0.0)


def test_an_impact_path_past_the_horizon_is_refused_by_name(versions):
    # A spec built by hand can carry an impact whose path outlives the horizon, as the family
    # of the reverse problem does; validate and build name the target.
    impact = Impact(target="price:alpha", start=2, values=[1.0] * 4, reason="r", basis=["s"])
    spec = scenario(versions, impacts=[impact]).model_copy(update={"horizon": 3})
    report = validate(spec, versions.loaded, versions.satellite, as_of=versions.data.months[-1])
    assert report["ok"] is False
    assert any("price:alpha" in problem for problem in report["problems"])
    with pytest.raises(ValueError, match="price:alpha"):
        paths_of(versions, spec)


def test_an_impact_on_a_rejected_driver_names_the_reason(versions):
    quantity, node = versions.satellite.node("price:alpha")
    rejected = dict(versions.satellite.checks[quantity])
    rejected[node] = replace(rejected[node], served=False, reason="not more accurate")
    satellite = replace(versions.satellite, checks={quantity: rejected})
    impact = Impact(target="price:alpha", start=1, values=[1.0, 1.0], reason="r", basis=["s"])
    with pytest.raises(ValueError, match="not more accurate"):
        build(
            scenario(versions, impacts=[impact]),
            versions.loaded,
            satellite,
            draws=DRAWS,
            seed=SEED,
        )


def test_validate_accepts_a_scenario_the_model_can_compute(versions):
    spec = scenario(versions, constraints=[fixed("gl.brent", 2, 1.0)])
    report = validate(spec, versions.loaded, versions.satellite, as_of=versions.data.months[-1])
    assert report["ok"] is True and report["problems"] == []


def _last(versions, variable: str) -> float:
    """The last observed value of a variable in the model scale: what the scale check reads."""
    column = versions.data.y[:, versions.data.ids.index(variable)]
    return float(column[np.isfinite(column)][-1])


def _scale_warnings(versions, **changes) -> list[str]:
    spec = scenario(versions, **changes)
    report = validate(spec, versions.loaded, versions.satellite, as_of=versions.data.months[-1])
    return [warning for warning in report["warnings"] if "check the scale" in warning]


def test_validate_warns_about_a_level_far_from_the_last_observed_one(versions):
    # 200 points of 100 ln X is a level 7.4 times the last one: a percentage applied twice.
    level = _last(versions, "gl.brent") + 200
    found = _scale_warnings(versions, constraints=[fixed("gl.brent", 2, level)])
    assert len(found) == 1
    assert "constraint on gl.brent in month 2" in found[0] and "7.39 times" in found[0]


def test_validate_keeps_quiet_about_a_stress_within_five_times_the_level(versions):
    # Brent from 114 to 45 dollars is 2.5 times: a stress, not a scale error.
    level = _last(versions, "gl.brent") - 100 * np.log(114 / 45)
    assert _scale_warnings(versions, constraints=[fixed("gl.brent", 2, level)]) == []


def test_validate_does_not_read_the_scale_of_a_rate(versions):
    # A rate keeps its own units: its level is a percentage, so any number may be meant.
    level = _last(versions, "ru.key_rate") + 500
    assert _scale_warnings(versions, constraints=[fixed("ru.key_rate", 2, level)]) == []


def test_validate_warns_about_a_one_sided_bound_written_as_a_percentage(versions):
    # A fall of 45 per cent written as the level -45: the mean is above it on every path, so the
    # condition holds trivially, and the bound alone gives the error away.
    bound = _last(versions, "gl.brent") - 100 * np.log(1 / 0.0003)
    expectation = {
        "terms": [{"variable": "gl.brent", "month": 2, "weight": 1.0}],
        "low": bound,
        "basis": "assumption",
    }
    found = _scale_warnings(versions, expectations=[expectation], inner=2)
    assert len(found) == 1 and found[0].startswith("expectation on gl.brent in month 2")


def test_validate_does_not_read_the_scale_of_a_change_between_months(versions):
    # Weights 1 and -1 make a move, not a level: 300 points of move is the event's business.
    assert _scale_warnings(versions, constraints=[change("gl.brent", 3, 1, 300.0)]) == []


def test_validate_reads_a_mean_over_two_months_as_one_level(versions):
    level = _last(versions, "gl.brent") - 200
    mean = {
        "terms": [
            {"variable": "gl.brent", "month": 2, "weight": 0.5},
            {"variable": "gl.brent", "month": 3, "weight": 0.5},
        ],
        "low": level - 1,
        "high": level + 1,
        "basis": "assumption",
    }
    found = _scale_warnings(versions, constraints=[mean])
    assert len(found) == 1 and "gl.brent in months 2, 3" in found[0]


def test_validate_names_a_scenario_without_an_approver(versions):
    spec = scenario(versions, constraints=[fixed("gl.brent", 2, 1.0)], approved_by="")
    report = validate(spec, versions.loaded, versions.satellite, as_of=versions.data.months[-1])
    assert report["ok"] is False
    assert "names no approver" in "; ".join(report["problems"])


def test_validate_names_every_unknown_variable_and_shock(versions):
    spec = scenario(
        versions,
        constraints=[fixed("gl.nowhere", 2, 1.0), fixed("ru.nowhere", 3, 1.0)],
        causes=["ru.nosuch"],
        approved_by=None,
    )
    report = validate(spec, versions.loaded, versions.satellite, as_of=versions.data.months[-1])
    problems = "; ".join(report["problems"])
    assert report["ok"] is False
    assert "unknown variable gl.nowhere" in problems
    assert "unknown variable ru.nowhere" in problems
    assert "unknown cause ru.nosuch" in problems
    # The scenario names no approver: that is a problem of its own.
    assert "names no approver" in problems


def test_a_constraint_the_causes_cannot_move_is_refused_by_name(versions):
    # The second constraint cannot move anything the first does not: one shock, one direction.
    spec = scenario(
        versions,
        constraints=[fixed("gl.brent", 1, 1.0), fixed("gl.brent", 1, 2.0)],
        causes=["gl.oil"],
    )
    with pytest.raises(InfeasibleScenario, match="cannot meet constraints 2"):
        paths_of(versions, spec)
    report = validate(spec, versions.loaded, versions.satellite, as_of=versions.data.months[-1])
    assert report["ok"] is False
    assert any("cannot meet constraints 2" in problem for problem in report["problems"])


def test_validate_names_an_impact_on_a_rejected_driver(versions):
    quantity, node = versions.satellite.node("price:alpha")
    rejected = dict(versions.satellite.checks[quantity])
    rejected[node] = replace(rejected[node], served=False, reason="not more accurate")
    satellite = replace(versions.satellite, checks={quantity: rejected})
    impact = Impact(target="price:alpha", start=1, values=[1.0, 1.0], reason="r", basis=["s"])
    report = validate(
        scenario(versions, impacts=[impact]),
        versions.loaded,
        satellite,
        as_of=versions.data.months[-1],
    )
    assert report["ok"] is False and "not more accurate" in report["problems"][0]


def test_validate_names_the_satellites_of_another_macro_version(versions):
    shorter = replace(
        versions.loaded,
        meta={**versions.loaded.meta, "months": versions.loaded.meta["months"][:-1]},
    )
    report = validate(scenario(versions), shorter, versions.satellite, as_of=date(2010, 1, 1))
    assert report["ok"] is False and "the satellites are estimated up to" in report["problems"][0]


def test_validate_checks_that_the_cited_fragments_are_visible_on_the_day(versions, tmp_path):
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
    spec = scenario(versions, sources=["S1/2020#1/1"])
    assert validate(
        spec, versions.loaded, versions.satellite, as_of=date(2020, 4, 1), corpus=tmp_path
    )["ok"]
    report = validate(
        spec, versions.loaded, versions.satellite, as_of=date(2020, 1, 1), corpus=tmp_path
    )
    assert report["ok"] is False and "S1/2020#1/1" in report["problems"][0]

    # The fragments an impact cites are checked the same way as the scenario's own sources;
    # an impact resting on an assumption cites nothing and is not checked.
    impact = Impact(
        target="price:alpha", start=1, values=[1.0, 1.0], reason="r", basis=["S1/2020#1/1"]
    )
    report = validate(
        scenario(versions, impacts=[impact]),
        versions.loaded,
        versions.satellite,
        as_of=date(2020, 1, 1),
        corpus=tmp_path,
    )
    assert report["ok"] is False and "S1/2020#1/1" in report["problems"][0]


def _floored(versions, monkeypatch):
    """The key rate of the simulated panel reads through a floor just under its fan."""
    registry = load_registry()
    level = float(versions.data.y[-1, 1])
    variables = [
        variable.model_copy(update={"lower": level if variable.id == "ru.key_rate" else None})
        for variable in registry.variables
    ]
    from mst.scenario import build as build_module

    floored = registry.model_copy(update={"variables": variables})
    monkeypatch.setattr(build_module, "load_registry", lambda: floored)
    return level


def test_an_impact_adds_to_the_shadow_and_the_floor_reads_the_sum(versions, monkeypatch):
    # The impact is added before the floor: the condition holds on the shadow with the impact
    # in it, whatever the floor does to the part the model drew below it.
    level = _floored(versions, monkeypatch)
    target = level + 1.0
    impact = Impact(target="ru.key_rate", start=2, values=[3.0], reason="r", basis=["s"])
    spec = scenario(versions, constraints=[fixed("ru.key_rate", 2, target)], impacts=[impact])
    paths = paths_of(versions, spec)
    (constraint,) = plan_of(spec, versions.loaded).constraints
    assert np.all(np.abs(paths.macro[:, 2, 1] - target) <= constraint.tolerance + 1e-9)
    assert paths.checks["support_share"] == 1.0


def test_a_negative_impact_cannot_take_the_reported_value_below_the_floor(versions, monkeypatch):
    level = _floored(versions, monkeypatch)
    impact = Impact(target="ru.key_rate", start=2, values=[-500.0], reason="r", basis=["s"])
    paths = paths_of(versions, scenario(versions, impacts=[impact]))
    assert np.all(paths.macro[:, 1:, 1] >= level)
    assert np.all(paths.shadow[:, 2, 1] < level)  # the shadow keeps the whole impact


def test_a_support_condition_is_checked_on_the_shadow(versions, monkeypatch):
    # A fall to below the floor: the floor would break the difference, and the shadow keeps it.
    level = _floored(versions, monkeypatch)
    spec = scenario(versions, constraints=[change("ru.key_rate", 3, 1, -5.0)])
    paths = paths_of(versions, spec)
    assert paths.checks["support_share"] == 1.0
    assert np.all(paths.macro[:, 3, 1] >= level)
    assert np.any(paths.macro[:, 3, 1] - paths.macro[:, 1, 1] > -5.0 + 1.0)


def test_a_condition_on_a_mean_is_met_on_the_shadow(versions, monkeypatch):
    # The mean the tilt reads is the shadow's: the mean of the floored values stands above it,
    # so a tilt on the floored values would have measured another functional than the price.
    level = _floored(versions, monkeypatch)
    free = paths_of(versions, scenario(versions, inner=8)).shadow[:, 2, 1]
    target = level - 0.5 * float(free.std())
    spec = scenario(
        versions,
        inner=8,
        expectations=[
            {
                "terms": [{"variable": "ru.key_rate", "month": 2, "weight": 1.0}],
                "high": target,
                "basis": "assumption",
            }
        ],
    )
    paths = paths_of(versions, spec)
    assert np.mean(paths.shadow[:, 2, 1]) == pytest.approx(target, abs=0.2 * float(free.std()))
    assert np.mean(paths.macro[:, 2, 1]) >= level > target


def test_the_paths_carry_the_size_of_the_driving_shocks(versions):
    target = float(versions.data.y[-1, 0] + 5.0)
    paths = paths_of(versions, scenario(versions, constraints=[fixed("gl.brent", 3, target)]))
    assert len(paths.diagnostics["driving_sd"]["median_by_month"]) == HORIZON


class MeasuredBoxes:
    """A macro plugin that measures its own box probabilities and reports no law.

    Its functionals move with the shocks the causes name, so no mixture of gaussian rows
    carries their law: the price comes from ``log_box_probability``, here the logarithm of
    a known gaussian probability around a fixed mean.
    """

    def __init__(self, inner, mean=0.0, sigma=1.0, logs=None):
        self._inner = inner
        self._mean = mean
        self._sigma = sigma
        self._logs = logs
        self.seen = []
        self.plans = []

    def simulate(self, model, plan, *, draws, seed):
        # Constant zero paths: the medians of the one-sided price are zero, whatever
        # the inner model would have simulated.
        from datetime import date

        from mst.model_sdk.contracts import MacroSimulation

        variables = sorted({term.variable for c in plan.constraints for term in c.terms})
        if not variables:
            variables = ["gl.brent"]
        return MacroSimulation(
            ids=variables,
            months=[date(2026, 10, 1)] * (plan.horizon + 1),
            paths=np.zeros((draws, plan.horizon + 1, len(variables))),
        )

    def constraint_law(self, model, plan, *, draws, seed):
        return None

    def log_box_probability(self, model, plan, low, high, *, draws, seed):
        self.seen.append((np.asarray(low, dtype=float), np.asarray(high, dtype=float)))
        self.plans.append(plan)
        if self._logs is not None:
            return np.asarray(self._logs, dtype=float)
        low = np.asarray(low, dtype=float)
        high = np.asarray(high, dtype=float)
        from scipy.stats import norm

        probability = norm.cdf((high[0] - self._mean) / self._sigma) - norm.cdf(
            (low[0] - self._mean) / self._sigma
        )
        return np.full(draws, float(np.log(probability)))


def test_a_model_measuring_its_own_boxes_is_priced_by_them(versions):
    from scipy.stats import norm

    # The two-sided price is the mean of -log P(S | R) over the values returned.
    plugin = MeasuredBoxes(versions.loaded.plugin, mean=0.0, sigma=2.0)
    measured = replace(versions.loaded, plugin=plugin)
    spec = scenario(versions, constraints=[interval("gl.brent", 2, 0.0, 2.0)])
    found = dist_of(spec, measured, draws=8, seed=SEED)
    exact = -np.log(norm.cdf(1.0) - norm.cdf(0.0))
    assert found.distance == pytest.approx(exact)
    assert found.log_probability == pytest.approx(np.log(norm.cdf(1.0) - norm.cdf(0.0)))
    assert found.standard_error == 0.0
    assert found.message is None


def test_a_model_measuring_its_own_boxes_has_a_one_sided_price(versions):
    from scipy.stats import norm

    # The medians of the unconstrained simulation decide the side: an interval above the
    # median opens upwards, and the one-sided price is the ray's own probability.
    plugin = MeasuredBoxes(versions.loaded.plugin, mean=0.0, sigma=2.0)
    measured = replace(versions.loaded, plugin=plugin)
    spec = scenario(versions, constraints=[interval("gl.brent", 2, 1.0, 2.0)])
    found = dist_of(spec, measured, draws=8, seed=SEED)
    assert found.one_sided is not None and len(plugin.seen) >= 2
    ray = plugin.seen[-1]
    assert ray[0] == pytest.approx(1.0) and np.isposinf(ray[1]), "the ray opens upwards"
    assert found.one_sided.distance == pytest.approx(-np.log(1.0 - norm.cdf(0.5)))
    assert found.one_sided.dropped == ()


def test_the_one_sided_price_of_a_model_reads_only_the_kept_constraints(versions):
    # The first interval holds the median zero and is dropped; the plan the model
    # measures the ray on carries the second constraint alone, matching the bounds.
    plugin = MeasuredBoxes(versions.loaded.plugin, mean=0.0, sigma=2.0)
    measured = replace(versions.loaded, plugin=plugin)
    spec = scenario(
        versions,
        constraints=[interval("gl.brent", 1, -1.0, 1.0), interval("gl.brent", 2, 1.0, 2.0)],
    )
    found = dist_of(spec, measured, draws=8, seed=SEED)
    assert found.one_sided is not None and found.one_sided.dropped == (1,)
    ray, plan = plugin.seen[-1], plugin.plans[-1]
    assert len(plan.constraints) == len(ray[0]) == 1
    assert plan.constraints[0].terms[0].month == 2
    assert ray[0] == pytest.approx(1.0) and np.isposinf(ray[1])


def test_a_model_whose_boxes_are_impossible_has_an_infinite_price(versions):
    plugin = MeasuredBoxes(versions.loaded.plugin, logs=[float("-inf"), 0.0])
    measured = replace(versions.loaded, plugin=plugin)
    spec = scenario(versions, constraints=[interval("gl.brent", 2, 1.0, 2.0)])
    found = dist_of(spec, measured, draws=8, seed=SEED)
    assert found.distance == float("inf")
    assert "contradicts the models" in found.message


def test_a_model_without_a_law_and_without_boxes_has_no_price(versions):
    class Neither(MeasuredBoxes):
        def log_box_probability(self, model, plan, low, high, *, draws, seed):
            return None

    stripped = replace(
        versions.loaded, plugin=Neither(versions.loaded.plugin)
    )
    spec = scenario(versions, constraints=[interval("gl.brent", 2, 1.0, 3.0)])
    assert dist_of(spec, stripped, draws=DRAWS, seed=SEED) is None
