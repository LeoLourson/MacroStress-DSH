import json
from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from mst.data.store import Observation, SeriesStore
from mst.drivers.check import check
from mst.drivers.estimate import correlation, fit, history
from mst.drivers.registry import Registry, from_model, load_registry, to_model
from mst.drivers.simulate import OVERLAY, PARAMETERS_FROM_ABOVE, NotServed, simulate
from mst.drivers.spec import load_spec
from mst.drivers.version import build, load, save
from mst.macro.forecast import Simulation
from mst.macro.registry import load_registry as load_macro_registry
from mst.scenario.contracts import Overlay

from . import simulated

AS_OF = date(2030, 1, 1)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    panel = simulated.panel(240)
    store = simulated.store(tmp_path_factory.mktemp("series"), panel)
    return simulated.registry(), panel, store


def test_the_registry_declares_a_catalog_per_quantity_and_known_macro_variables():
    registry = load_registry(catalog=None)
    registry.check(load_macro_registry())
    assert all(q.catalog is not None for q in registry.quantities)


def test_a_catalog_file_adds_objects_and_drivers_and_positions_match_by_code(tmp_path):
    objects = [
        {"id": "economy", "class": "activity_0", "label": "ВСЕГО", "codes": {"okved": ["4"]}},
        {"id": "construction", "class": "activity_1", "label": "строительство",
         "parent": "economy", "codes": {"okved": ["41", "42"]}},
        {"id": "building", "class": "activity_2", "label": "здания", "parent": "construction",
         "codes": {"okved": ["41"]}},
    ]  # fmt: skip
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps({"quantities": {"overdue_share": {"objects": objects,
                    "drivers": ["economy", "construction", "building"]}}}),  # fmt: skip
        encoding="utf-8",
    )
    registry = load_registry(catalog=catalog)
    assert registry.match("overdue_share", "okved", "41.20") == "overdue_share:building"
    assert registry.match("overdue_share", "okved", "42.11") == "overdue_share:construction"
    assert registry.match("overdue_share", "okved", "47.11") == "overdue_share:economy"
    assert registry.match("overdue_share", "okved", "41.20", among={"overdue_share:economy"}) == (
        "overdue_share:economy"
    )
    assert registry.match("overdue_share", "okved", "01.11") is None
    assert next(d for d in registry.drivers if d.object == "building").series == (
        "drv.overdue_share.building"
    )


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda o: o[1].update(parent="nowhere"), "unknown parent"),
        (lambda o: o[0].update(parent="alpha"), "cycle"),
    ],
)
def test_the_registry_refuses_broken_trees(change, message):
    raw = simulated.registry().model_dump(by_alias=True)
    change(raw["objects"])
    with pytest.raises(ValueError, match=message):
        Registry.model_validate(raw)


def test_transforms_invert():
    for transform, value in (("identity", -3.0), ("log100", 7.5), ("logit", 0.07)):
        assert from_model(transform, to_model(transform, value)) == pytest.approx(value)
    with pytest.raises(ValueError):
        to_model("logit", 1.0)


def test_the_posterior_covers_the_true_coefficients_of_every_object(data):
    registry, panel, store = data
    hist = history(registry, "price", panel, store, AS_OF)
    result = fit(hist, load_spec())
    covered = []
    for name, truth in {"sector": simulated.SECTOR, **simulated.COMPANIES}.items():
        draws = result.gamma[:, hist.nodes.index(name), 1:]
        low, high = np.percentile(draws, [5, 95], axis=0)
        covered += list((low <= truth[1:]) & (truth[1:] <= high))
        sigma = np.median(result.sigma[:, hist.nodes.index(name)])
        assert sigma == pytest.approx(simulated.SIGMA, rel=0.15)
    assert np.mean(covered) >= 0.75


def test_an_object_without_history_takes_parameters_from_the_level_above(data):
    registry, panel, store = data
    hist = history(registry, "price", panel, store, AS_OF)
    result = fit(hist, load_spec())
    delta = result.gamma[:, hist.nodes.index("delta")].mean(axis=0)
    sector = result.gamma[:, hist.nodes.index("sector")].mean(axis=0)
    assert result.rows[hist.nodes.index("delta")] == 0
    assert delta[1:] == pytest.approx(sector[1:], abs=0.15)


def test_a_satellite_that_uses_the_macro_variables_beats_the_model_without_them(data):
    registry, panel, store = data
    hist = history(registry, "price", panel, store, AS_OF)
    checks = check(hist, load_spec())
    for name in ["sector", *simulated.COMPANIES]:
        item = checks[hist.nodes.index(name)]
        assert item.served and item.satellite < item.baseline
    assert checks[hist.nodes.index("delta")].served


def test_the_residual_correlation_is_a_correlation_matrix():
    rng = np.random.default_rng(3)
    common = rng.standard_normal(100)
    a, b, c = common + rng.standard_normal(100), common, rng.standard_normal(100)
    c[:95] = np.nan
    corr = correlation([a, b, c])
    assert np.allclose(np.diag(corr), 1)
    assert corr[0, 1] == pytest.approx(np.corrcoef(a, b)[0, 1], abs=1e-6)
    assert corr[0, 2] == 0
    assert np.linalg.eigvalsh(corr).min() > 0


@pytest.fixture(scope="module")
def satellites(data):
    registry, panel, store = data
    return build(registry, load_spec(), panel, store, AS_OF, "macro-version")


def test_a_version_loads_as_saved_and_refuses_a_change(satellites, tmp_path):
    name = save(satellites, tmp_path)
    assert save(satellites, tmp_path) == name
    again = load(tmp_path, name)
    assert again.served() == satellites.served()
    assert np.array_equal(again.fits["price"].gamma, satellites.fits["price"].gamma)
    meta = tmp_path / "satellites" / name / "meta.json"
    meta.write_text(meta.read_text(encoding="utf-8").replace("macro-version", "x"), "utf-8")
    with pytest.raises(ValueError, match="does not match"):
        load(tmp_path, name)


def macro_paths(panel, x_future, draws=3):
    """A macro simulation whose global variable follows ``x_future`` after the panel."""
    p, horizon = 3, len(x_future)
    history = np.repeat(panel.y[None, -p:], draws, axis=0)
    future = np.zeros((draws, horizon, 2))
    future[:, :, 0] = x_future
    paths = np.concatenate([history[:, -1:], future], axis=1)
    months = [panel.months[-1]] + [date(2030, m, 1) for m in range(1, horizon + 1)]
    return Simulation(ids=panel.ids, months=months, paths=paths, history=history)


def test_driver_paths_follow_the_satellite_and_carry_overlays(satellites, data):
    _, panel, _ = data
    fixed = replace(satellites.fits["price"])
    gamma = np.tile(simulated.COMPANIES["alpha"], fixed.gamma.shape[:2] + (1,))
    fixed = replace(fixed, gamma=gamma, sigma=np.zeros_like(fixed.sigma))
    sat = replace(satellites, fits={"price": fixed})
    macro = macro_paths(panel, np.array([1.0, 2.0, 3.0]))
    overlay = Overlay(driver="price:alpha", start=2, end=3, delta=10.0, reason="r", sources=["s"])
    result = simulate(sat, macro, ["price:alpha", "price:delta"], [overlay], seed=7)

    hist = sat.fits["price"].history
    d, x = hist.d[-1, hist.nodes.index("alpha")], [panel.y[-1, 0], 1.0, 2.0, 3.0]
    expected = [d]
    for t in range(1, 4):
        expected.append(simulated.COMPANIES["alpha"] @ [1.0, expected[-1], x[t], x[t - 1]])
    expected = np.array(expected) + [0, 0, 10, 10]
    assert result.paths[:, :, 0] == pytest.approx(np.tile(expected, (3, 1)))
    assert result.flags["price:alpha"] == {OVERLAY}
    assert result.flags["price:delta"] == {PARAMETERS_FROM_ABOVE}
    sector = hist.d[-1, hist.nodes.index("sector")]
    assert result.paths[:, 0, 1] == pytest.approx(sector)


def test_a_driver_the_check_rejected_or_too_old_is_not_served(satellites, data):
    _, panel, _ = data
    macro = macro_paths(panel, np.array([1.0]))
    quantity, node = satellites.node("price:alpha")
    rejected = dict(satellites.checks[quantity])
    rejected[node] = replace(rejected[node], served=False, reason="not more accurate")
    sat = replace(satellites, checks={quantity: rejected})
    with pytest.raises(NotServed, match="not more accurate"):
        simulate(sat, macro, ["price:alpha"], seed=1)

    fit_ = satellites.fits["price"]
    x = fit_.history.x.copy()
    x[-5] = np.nan
    d = fit_.history.d.copy()
    d[-6:, node] = np.nan
    unknown = replace(fit_, history=replace(fit_.history, d=d, x=x))
    with pytest.raises(NotServed, match="x is unknown"):
        simulate(replace(satellites, fits={"price": unknown}), macro, ["price:alpha"], seed=1)


def test_a_driver_that_lags_the_panel_is_simulated_up_to_t0(satellites, data):
    _, panel, _ = data
    fit_ = satellites.fits["price"]
    quantity, node = satellites.node("price:alpha")
    gamma = np.tile(simulated.COMPANIES["alpha"], fit_.gamma.shape[:2] + (1,))
    d = fit_.history.d.copy()
    d[-5:, node] = np.nan
    lagging = replace(fit_, gamma=gamma, sigma=np.zeros_like(fit_.sigma),
                      history=replace(fit_.history, d=d))  # fmt: skip
    macro = macro_paths(panel, np.array([1.0]))
    result = simulate(replace(satellites, fits={"price": lagging}), macro, ["price:alpha"], seed=1)
    x, value = panel.y[:, 0], fit_.history.d[-6, node]
    for t in range(len(x) - 5, len(x)):
        value = simulated.COMPANIES["alpha"] @ [1.0, value, x[t], x[t - 1]]
    assert result.paths[0, 0, 0] == pytest.approx(value)


def test_a_quarterly_average_driver_is_estimated_with_its_months_drawn(tmp_path):
    panel = simulated.panel(240, seed=5)
    rng = np.random.default_rng(6)
    x = panel.y[:, 0]
    registry = simulated.registry(without_history=())
    raw = registry.model_dump(by_alias=True)
    raw["quantities"][0]["span"] = 3
    registry = Registry.model_validate(raw)
    store = SeriesStore(tmp_path)
    for name, gamma in {"sector": simulated.SECTOR, **simulated.COMPANIES}.items():
        values = simulated.path(gamma, x, rng)
        rows = [
            Observation(
                panel.months[t], float(values[t - 2 : t + 1].mean()), panel.months[t], False
            )
            for t in range(2, len(x), 3)
        ]
        store.write(f"drv.price.{name}", rows)
    hist = history(registry, "price", panel, store, AS_OF)
    result = fit(hist, load_spec())
    rho = result.gamma[:, hist.nodes.index("alpha"), 1]
    low, high = np.percentile(rho, [5, 95])
    assert low - 0.05 <= simulated.COMPANIES["alpha"][1] <= high + 0.05
    beta = result.gamma[:, hist.nodes.index("sector"), 2].mean()
    assert beta == pytest.approx(simulated.SECTOR[2], abs=0.3)


def test_driver_paths_repeat_bit_for_bit(satellites, data):
    _, panel, _ = data
    macro = macro_paths(panel, np.array([1.0, 0.5]))
    first = simulate(satellites, macro, ["price:alpha", "price:beta"], seed=11)
    again = simulate(satellites, macro, ["price:alpha", "price:beta"], seed=11)
    assert np.array_equal(first.paths, again.paths)
