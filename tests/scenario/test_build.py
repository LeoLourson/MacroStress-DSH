"""mst.scenario.build: the paths of a scenario and what validate tells about it."""

from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from mst.corpus.index import Chunk
from mst.corpus.index import write as write_index
from mst.macro.registry import load_registry
from mst.model_sdk import InfeasibleScenario
from mst.model_sdk.contracts import ScenarioPlan
from mst.model_sdk.dist import box_probability_mc, box_probability_mixture
from mst.plugins.hier.simulate import IMPACT
from mst.scenario.build import build, dist_of, plan_of, validate
from mst.scenario.contracts import Impact
from mst.scenario.spec import Spec

DRAWS, HORIZON, SEED = 8, 6, 5


def scenario(versions, **changes) -> Spec:
    values = {
        "narrative": "нефть дорожает",
        "horizon": HORIZON,
        "macro": versions.macro,
        "satellites": versions.satellites,
    }
    values.update(changes)
    return Spec.model_validate(values)


def fixed(variable: str, month: int, value: float) -> dict:
    return {
        "terms": [{"variable": variable, "month": month, "weight": 1.0}],
        "low": value,
        "high": value,
    }


def interval(variable: str, month: int, low: float, high: float) -> dict:
    return {
        "terms": [{"variable": variable, "month": month, "weight": 1.0}],
        "low": low,
        "high": high,
    }


def change(variable: str, month: int, other: int, value: float) -> dict:
    return {
        "terms": [
            {"variable": variable, "month": month, "weight": 1.0},
            {"variable": variable, "month": other, "weight": -1.0},
        ],
        "low": value,
        "high": value,
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


def test_exact_constraints_hold_in_every_path_to_machine_precision(versions):
    registry = load_registry()
    target = float(versions.data.y[-1, 0] + 5.0)
    spec = scenario(versions, constraints=[fixed("gl.brent", 2, target)])
    paths = paths_of(versions, spec)
    moved = registry.variable("gl.brent").to_model(paths.macro[:, 2, 0])
    assert np.allclose(moved, target, atol=1e-9, rtol=0)
    assert paths.macro[:, 2, 1].std() > 0  # the other variables still vary
    assert paths.severity is not None and paths.severity.distance > 0


def test_an_impact_on_a_constrained_variable_meets_the_constraint(versions):
    # The impact is added after the model has drawn the paths, so the constraint C must hold on
    # the shifted trajectory F(omega, O) = G_0 + o_0: the target is met by the final path.
    registry = load_registry()
    target = float(versions.data.y[-1, 0] + 5.0)
    impact = Impact(target="gl.brent", start=2, end=2, delta=2.0, reason="r", sources=["s"])
    spec = scenario(versions, constraints=[fixed("gl.brent", 2, target)], impacts=[impact])
    report = validate(spec, versions.loaded, versions.satellite, as_of=versions.data.months[-1])
    assert report == {"ok": True, "problems": []}
    paths = paths_of(versions, spec)
    moved = registry.variable("gl.brent").to_model(paths.macro[:, 2, 0])
    assert np.allclose(moved, target, atol=1e-9, rtol=0)


def test_a_change_constraint_holds_across_months(versions):
    spec = scenario(versions, constraints=[change("ru.key_rate", 3, 1, 2.0)])
    paths = paths_of(versions, spec)
    assert np.allclose(paths.macro[:, 3, 1] - paths.macro[:, 1, 1], 2.0, atol=1e-9, rtol=0)


def test_the_paths_repeat_bit_for_bit_for_the_same_seed(versions):
    spec = scenario(versions, constraints=[fixed("gl.brent", 2, 1.0)])
    first = paths_of(versions, spec)
    again = paths_of(versions, spec)
    assert np.array_equal(first.macro, again.macro)
    assert np.array_equal(first.drivers.paths, again.drivers.paths)
    other = build(spec, versions.loaded, versions.satellite, draws=DRAWS, seed=SEED + 1)
    assert not np.array_equal(first.macro, other.macro)


def test_a_scenario_without_constraints_has_no_severity(versions):
    paths = paths_of(versions, scenario(versions))
    assert paths.severity is None
    assert paths.dist is None  # no dist, not a zero distance


def test_the_paths_carry_the_platform_dist_of_the_constraints(versions):
    # The MC share of the base run's paths estimates P(C); the distance is -log P(C) of the
    # constraint box on the unconditional run, not the Mahalanobis delta of the plugin.
    spec = scenario(versions, constraints=[interval("gl.brent", 2, 1.0, 3.0)])
    paths = paths_of(versions, spec)
    assert paths.dist is not None
    assert paths.dist.way == "mc"
    assert paths.dist.message is None
    assert paths.dist.distance == pytest.approx(-np.log(paths.dist.probability))
    assert 0.0 < paths.dist.probability < 1.0


def test_the_mc_dist_agrees_with_the_exact_mixture(versions):
    # The share of the base run's paths (the same draws and seed the constraint gate reads)
    # must agree with the exact probability of the parameter mixture within the MC error.
    spec = scenario(versions, constraints=[interval("gl.brent", 2, 1.0, 4.0)])
    constraints = plan_of(spec).constraints
    base = versions.loaded.plugin.simulate(
        versions.loaded.model, ScenarioPlan(horizon=spec.horizon), draws=4000, seed=SEED
    )
    # The same 4000 drawings: the share of the dist is the share counted here, exactly.
    found = dist_of(spec, versions.loaded, draws=4000, seed=SEED)
    mc = box_probability_mc(base.paths, base.ids, constraints)
    assert found.way == "mc"
    assert found.probability == mc
    means, covs = versions.loaded.plugin.constraint_moments(
        versions.loaded.model, plan_of(spec), draws=100, seed=SEED
    )
    exact = box_probability_mixture(means, covs, constraints)
    assert exact == pytest.approx(mc, abs=0.02)


def test_a_zero_share_falls_back_to_the_exact_mixture(versions):
    # A box three and a half standard deviations out: none of the few drawn paths lands in it,
    # so the Monte-Carlo share is zero and the exact probability of the mixture takes over;
    # the distance is large but finite.
    spec = scenario(versions, constraints=[interval("gl.brent", 2, 9.0, 9.5)])
    found = dist_of(spec, versions.loaded, draws=DRAWS, seed=SEED)
    assert found.way == "exact"
    assert 0.0 < found.probability < 1.0
    assert np.isfinite(found.distance) and found.distance > 0


def test_a_model_without_moments_leaves_the_monte_carlo_zero(versions):
    # A macro model that does not measure the per-draw moments: a zero share stays a zero
    # share, and the dist is infinite with the failed gate named.
    spec = scenario(versions, constraints=[interval("gl.brent", 2, 100.0, 101.0)])
    stripped = replace(versions.loaded, plugin=NoMoments(versions.loaded.plugin))
    found = dist_of(spec, stripped, draws=DRAWS, seed=SEED)
    assert found.way == "mc"
    assert found.probability == 0.0
    assert found.distance == float("inf")
    assert "gate pi_c is failed" in found.message


class NoMoments:
    """A macro plugin that simulates but does not measure the constraint moments."""

    def __init__(self, inner):
        self._inner = inner

    def simulate(self, model, plan, *, draws, seed):
        return self._inner.simulate(model, plan, draws=draws, seed=seed)

    def severity(self, model, plan, *, draws, seed):
        return self._inner.severity(model, plan, draws=draws, seed=seed)

    def constraint_moments(self, model, plan, *, draws, seed):
        return None


def test_a_wide_box_gives_a_dist_near_zero(versions):
    wide = scenario(versions, constraints=[interval("gl.brent", 2, -50.0, 50.0)])
    narrow = scenario(versions, constraints=[interval("gl.brent", 2, 1.0, 3.0)])
    assert dist_of(wide, versions.loaded, draws=40, seed=SEED).probability > 0.99
    assert dist_of(wide, versions.loaded, draws=40, seed=SEED).distance < 0.1
    assert dist_of(narrow, versions.loaded, draws=40, seed=SEED).distance > dist_of(
        wide, versions.loaded, draws=40, seed=SEED
    ).distance


def test_driver_paths_are_in_the_drivers_own_units_and_carry_impacts(versions):
    impact = Impact(
        target="price:alpha", start=2, end=3, delta=10.0, reason="перебой", sources=["S1#1/1"]
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
    impact = Impact(target="gl.brent", start=1, end=3, delta=2.0, reason="шок", sources=["S1#1/1"])
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
    impact = Impact(target="price:nowhere", start=1, end=2, delta=1.0, reason="r", sources=["s"])
    with pytest.raises(ValueError, match="knows no such driver"):
        paths_of(versions, scenario(versions, impacts=[impact]))


def test_an_impact_on_a_variable_the_macro_version_lacks_is_refused(versions):
    impact = Impact(target="ru.nowhere", start=1, end=2, delta=1.0, reason="r", sources=["s"])
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


def test_an_impact_on_a_rejected_driver_names_the_reason(versions):
    quantity, node = versions.satellite.node("price:alpha")
    rejected = dict(versions.satellite.checks[quantity])
    rejected[node] = replace(rejected[node], served=False, reason="not more accurate")
    satellite = replace(versions.satellite, checks={quantity: rejected})
    impact = Impact(target="price:alpha", start=1, end=2, delta=1.0, reason="r", sources=["s"])
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
    assert report == {"ok": True, "problems": []}


def test_validate_names_every_unknown_variable_and_shock(versions):
    spec = scenario(
        versions,
        constraints=[fixed("gl.nowhere", 2, 1.0), fixed("ru.nowhere", 3, 1.0)],
        driving_shocks=["ru.nosuch"],
    )
    report = validate(spec, versions.loaded, versions.satellite, as_of=versions.data.months[-1])
    problems = "; ".join(report["problems"])
    assert report["ok"] is False
    assert "unknown variable gl.nowhere" in problems
    assert "unknown variable ru.nowhere" in problems
    assert "unknown shock ru.nosuch" in problems


def test_a_constraint_the_driving_shocks_cannot_move_is_refused_by_name(versions):
    # The second constraint cannot move anything the first does not: one shock, one direction.
    spec = scenario(
        versions,
        constraints=[fixed("gl.brent", 1, 1.0), fixed("gl.brent", 1, 2.0)],
        driving_shocks=["gl.oil"],
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
    impact = Impact(target="price:alpha", start=1, end=2, delta=1.0, reason="r", sources=["s"])
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

    # The fragments an impact cites are checked the same way as the scenario's own sources.
    impact = Impact(
        target="price:alpha", start=1, end=2, delta=1.0, reason="r", sources=["S1/2020#1/1"]
    )
    report = validate(
        scenario(versions, impacts=[impact]),
        versions.loaded,
        versions.satellite,
        as_of=date(2020, 1, 1),
        corpus=tmp_path,
    )
    assert report["ok"] is False and "S1/2020#1/1" in report["problems"][0]
