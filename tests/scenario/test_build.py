"""mst.scenario.build: the paths of a scenario and what validate tells about it."""

from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from mst.corpus.index import Chunk
from mst.corpus.index import write as write_index
from mst.drivers.simulate import OVERLAY
from mst.macro.forecast import InfeasibleScenario
from mst.macro.registry import load_registry
from mst.scenario.build import build, validate
from mst.scenario.contracts import Overlay
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
    return build(spec, versions.model, versions.satellite, draws=DRAWS, seed=SEED)


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
    other = build(spec, versions.model, versions.satellite, draws=DRAWS, seed=SEED + 1)
    assert not np.array_equal(first.macro, other.macro)


def test_a_scenario_without_constraints_has_no_severity(versions):
    paths = paths_of(versions, scenario(versions))
    assert paths.severity is None


def test_driver_paths_are_in_the_drivers_own_units_and_carry_overlays(versions):
    overlay = Overlay(
        driver="price:alpha", start=2, end=3, delta=10.0, reason="перебой", sources=["S1#1/1"]
    )
    plain = paths_of(versions, scenario(versions))
    with_overlay = paths_of(versions, scenario(versions, overlays=[overlay]))
    assert with_overlay.drivers.ids == sorted(versions.satellite.served())
    k = with_overlay.drivers.ids.index("price:alpha")
    difference = with_overlay.drivers.paths[:, :, k] - plain.drivers.paths[:, :, k]
    assert np.allclose(difference[:, 2:4], 10.0) and np.allclose(difference[:, [0, 4, 5, 6]], 0)
    assert with_overlay.drivers.flags["price:alpha"] == {OVERLAY}


def test_an_overlay_on_a_driver_the_satellites_do_not_know_is_refused(versions):
    overlay = Overlay(
        driver="price:nowhere", start=1, end=2, delta=1.0, reason="r", sources=["s"]
    )
    with pytest.raises(ValueError, match="knows no such driver"):
        paths_of(versions, scenario(versions, overlays=[overlay]))


def test_an_overlay_on_a_rejected_driver_names_the_reason(versions):
    quantity, node = versions.satellite.node("price:alpha")
    rejected = dict(versions.satellite.checks[quantity])
    rejected[node] = replace(rejected[node], served=False, reason="not more accurate")
    satellite = replace(versions.satellite, checks={quantity: rejected})
    overlay = Overlay(driver="price:alpha", start=1, end=2, delta=1.0, reason="r", sources=["s"])
    with pytest.raises(ValueError, match="not more accurate"):
        build(
            scenario(versions, overlays=[overlay]),
            versions.model,
            satellite,
            draws=DRAWS,
            seed=SEED,
        )


def test_validate_accepts_a_scenario_the_model_can_compute(versions):
    spec = scenario(versions, constraints=[fixed("gl.brent", 2, 1.0)])
    report = validate(spec, versions.model, versions.satellite, as_of=versions.data.months[-1])
    assert report == {"ok": True, "problems": []}


def test_validate_names_every_unknown_variable_and_shock(versions):
    spec = scenario(
        versions,
        constraints=[fixed("gl.nowhere", 2, 1.0), fixed("ru.nowhere", 3, 1.0)],
        driving_shocks=["ru.nosuch"],
    )
    report = validate(spec, versions.model, versions.satellite, as_of=versions.data.months[-1])
    assert report["ok"] is False
    assert "unknown variable gl.nowhere" in report["problems"][0]
    assert "unknown variable ru.nowhere" in report["problems"][0]
    assert "unknown shock ru.nosuch" in report["problems"][0]


def test_a_constraint_the_driving_shocks_cannot_move_is_refused_by_name(versions):
    # The second constraint cannot move anything the first does not: one shock, one direction.
    spec = scenario(
        versions,
        constraints=[fixed("gl.brent", 1, 1.0), fixed("gl.brent", 1, 2.0)],
        driving_shocks=["gl.oil"],
    )
    with pytest.raises(InfeasibleScenario, match="cannot meet constraints 2"):
        paths_of(versions, spec)
    report = validate(spec, versions.model, versions.satellite, as_of=versions.data.months[-1])
    assert report["ok"] is False
    assert any("cannot meet constraints 2" in problem for problem in report["problems"])


def test_validate_names_an_overlay_on_a_rejected_driver(versions):
    quantity, node = versions.satellite.node("price:alpha")
    rejected = dict(versions.satellite.checks[quantity])
    rejected[node] = replace(rejected[node], served=False, reason="not more accurate")
    satellite = replace(versions.satellite, checks={quantity: rejected})
    overlay = Overlay(driver="price:alpha", start=1, end=2, delta=1.0, reason="r", sources=["s"])
    report = validate(
        scenario(versions, overlays=[overlay]),
        versions.model,
        satellite,
        as_of=versions.data.months[-1],
    )
    assert report["ok"] is False and "not more accurate" in report["problems"][0]


def test_validate_names_the_satellites_of_another_macro_version(versions):
    shorter = replace(versions.model, months=versions.model.months[:-1])
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
        spec, versions.model, versions.satellite, as_of=date(2020, 4, 1), corpus=tmp_path
    )["ok"]
    report = validate(
        spec, versions.model, versions.satellite, as_of=date(2020, 1, 1), corpus=tmp_path
    )
    assert report["ok"] is False and "S1/2020#1/1" in report["problems"][0]
