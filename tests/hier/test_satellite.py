"""The hierarchical satellite: estimation, check, version and driver paths."""

from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from mst.data.store import Observation, SeriesStore
from mst.drivers import bundle as satellite_bundle
from mst.drivers.history import correlation, history
from mst.drivers.registry import Registry
from mst.model_sdk import AdditiveImpact
from mst.model_sdk.contracts import MacroSimulation as Simulation
from mst.plugins.hier import MODEL
from mst.plugins.hier.check import check
from mst.plugins.hier.estimate import fit, path_of
from mst.plugins.hier.simulate import IMPACT, PARAMETERS_FROM_ABOVE, NotServed
from mst.plugins.hier.simulate import simulate as simulate_hier
from mst.plugins.hier.spec import Sampler, Spec, load_spec
from tests.drivers import simulated

AS_OF = date(2030, 1, 1)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    panel = simulated.panel(240)
    store = simulated.store(tmp_path_factory.mktemp("series"), panel)
    return simulated.registry(), panel, store


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
def version(data, tmp_path_factory):
    """The estimate of the plugin on the synthetic data, assembled and saved as a bundle."""
    registry, panel, store = data
    estimate = MODEL.estimate(
        _context(registry, panel, store, quantities=("price",), macro="macro-version")
    )
    meta, arrays = satellite_bundle.assemble(
        registry, "macro-version", AS_OF, {MODEL.model_id: estimate}
    )
    root = tmp_path_factory.mktemp("models")
    name = satellite_bundle.save(meta, arrays, root)
    return satellite_bundle.load(root, name, {MODEL.model_id: MODEL})


def _context(registry, panel, store, *, quantities, macro):
    from mst.model_sdk.contracts import SatelliteEstimateContext

    return SatelliteEstimateContext(
        store=store,
        registry=registry,
        panel=panel,
        as_of=AS_OF,
        macro=macro,
        quantities=quantities,
    )


def test_a_version_loads_as_saved_and_refuses_a_change(version, data, tmp_path):
    registry, panel, store = data
    assert version.quantity_model == {"price": MODEL.model_id}
    assert set(version.served()) == {
        "price:sector", "price:alpha", "price:beta", "price:gamma", "price:delta",
    }  # fmt: skip
    again = version.models[MODEL.model_id]
    assert set(again.fits) == {"price"}

    estimate = MODEL.estimate(
        _context(registry, panel, store, quantities=("price",), macro="macro-version")
    )
    meta, arrays = satellite_bundle.assemble(
        registry, "macro-version", AS_OF, {MODEL.model_id: estimate}
    )
    name = satellite_bundle.save(meta, arrays, tmp_path)
    assert satellite_bundle.save(meta, arrays, tmp_path) == name
    stored = tmp_path / "satellites" / name / "meta.json"
    stored.write_text(stored.read_text(encoding="utf-8").replace("macro-version", "x"), "utf-8")
    with pytest.raises(ValueError, match="does not match"):
        satellite_bundle.load(tmp_path, name, {MODEL.model_id: MODEL})


def test_a_version_of_an_uninstalled_model_is_refused(version, tmp_path):
    meta = {
        "macro": version.macro,
        "as_of": version.as_of.isoformat(),
        "registry": version.registry.model_dump(mode="json", by_alias=True),
        "quantities": {
            quantity: {**part, "model": "gone.v1"}
            for quantity, part in version.parts.items()
        },
        "correlated": list(version.correlated),
    }
    name = satellite_bundle.save(meta, dict(version.arrays), tmp_path)
    with pytest.raises(KeyError, match="which is not installed"):
        satellite_bundle.load(tmp_path, name, {})


def macro_paths(panel, x_future, draws=3):
    """A macro simulation whose global variable follows ``x_future`` after the panel."""
    p, horizon = 3, len(x_future)
    history = np.repeat(panel.y[None, -p:], draws, axis=0)
    future = np.zeros((draws, horizon, 2))
    future[:, :, 0] = x_future
    paths = np.concatenate([history[:, -1:], future], axis=1)
    months = [panel.months[-1]] + [date(2030, m, 1) for m in range(1, horizon + 1)]
    return Simulation(ids=panel.ids, months=months, paths=paths, history=history)


def platform_shocks(model, macro, drivers, rows: int = 64, seed: int = 5) -> np.ndarray:
    """The standardized residuals the platform draws for these drivers' own histories."""
    fit_ = next(iter(model.fits.values()))
    own = [
        driver
        for driver in drivers
        if driver in fit_.history.driver
        and fit_.rows[fit_.history.driver.index(driver)] > 0
    ]
    draws = macro.paths.shape[0]
    return np.random.default_rng(seed).standard_normal((draws, rows, len(own)))


def _with_fit(version, fit_):
    """The version's model with the fit of ``price`` replaced."""
    model = version.models[MODEL.model_id]
    return replace(model, fits={"price": fit_})


def test_driver_paths_follow_the_satellite_and_carry_impacts(version, data):
    _, panel, _ = data
    fit_ = version.models[MODEL.model_id].fits["price"]
    gamma = np.tile(simulated.COMPANIES["alpha"], fit_.gamma.shape[:2] + (1,))
    fixed = replace(fit_, gamma=gamma, sigma=np.zeros_like(fit_.sigma))
    model = _with_fit(version, fixed)
    macro = macro_paths(panel, np.array([1.0, 2.0, 3.0]))
    impact = AdditiveImpact(target="price:alpha", start=2, end=3, delta=10.0)
    result = simulate_hier(
        model,
        macro,
        ["price:alpha", "price:delta"],
        [impact],
        correlation=(version.correlated, version.correlation),
        shocks=platform_shocks(model, macro, ["price:alpha", "price:delta"]),
        seed=7,
        stream=0,
    )

    hist = fit_.history
    d, x = hist.d[-1, hist.nodes.index("alpha")], [panel.y[-1, 0], 1.0, 2.0, 3.0]
    expected = [d]
    for t in range(1, 4):
        expected.append(simulated.COMPANIES["alpha"] @ [1.0, expected[-1], x[t], x[t - 1]])
    expected = np.array(expected) + [0, 0, 10, 10]
    assert result.paths[:, :, 0] == pytest.approx(np.tile(expected, (3, 1)))
    assert result.flags["price:alpha"] == {IMPACT}
    assert result.flags["price:delta"] == {PARAMETERS_FROM_ABOVE}
    sector = hist.d[-1, hist.nodes.index("sector")]
    assert result.paths[:, 0, 1] == pytest.approx(sector)


def test_a_driver_the_check_rejected_or_too_old_is_not_served(version, data):
    _, panel, _ = data
    macro = macro_paths(panel, np.array([1.0]))
    model = version.models[MODEL.model_id]
    quantity, node = "price", model.fits["price"].history.driver.index("price:alpha")
    rejected = dict(model.checks[quantity])
    rejected[node] = replace(rejected[node], served=False, reason="not more accurate")
    with pytest.raises(NotServed, match="not more accurate"):
        simulate_hier(
            replace(model, checks={quantity: rejected}), macro, ["price:alpha"], [],
            shocks=platform_shocks(model, macro, ["price:alpha"]), seed=1, stream=0,
        )

    fit_ = model.fits["price"]
    x = fit_.history.x.copy()
    x[-5] = np.nan
    d = fit_.history.d.copy()
    d[-6:, node] = np.nan
    unknown = replace(fit_, history=replace(fit_.history, d=d, x=x))
    with pytest.raises(NotServed, match="x is unknown"):
        simulate_hier(
            replace(model, fits={"price": unknown}), macro, ["price:alpha"], [],
            shocks=platform_shocks(model, macro, ["price:alpha"]), seed=1, stream=0,
        )


def test_a_driver_that_lags_the_panel_is_simulated_up_to_t0(version, data):
    _, panel, _ = data
    model = version.models[MODEL.model_id]
    fit_ = model.fits["price"]
    node = fit_.history.driver.index("price:alpha")
    gamma = np.tile(simulated.COMPANIES["alpha"], fit_.gamma.shape[:2] + (1,))
    d = fit_.history.d.copy()
    d[-5:, node] = np.nan
    lagging = replace(fit_, gamma=gamma, sigma=np.zeros_like(fit_.sigma),
                      history=replace(fit_.history, d=d))  # fmt: skip
    macro = macro_paths(panel, np.array([1.0]))
    result = simulate_hier(
        replace(model, fits={"price": lagging}), macro, ["price:alpha"], [],
        correlation=(version.correlated, version.correlation),
        shocks=platform_shocks(model, macro, ["price:alpha"]), seed=1, stream=0,
    )
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


def test_a_driver_whose_data_imply_an_explosive_root_gets_a_stationary_path(tmp_path):
    panel = simulated.panel(240, seed=5)
    registry = simulated.registry(without_history=())
    store = simulated.store(tmp_path, panel, seed=2)
    store.write(
        "drv.price.alpha",
        [
            Observation(month, 2.0 * 1.02**t, month, False)
            for t, month in enumerate(panel.months)
        ],
    )
    hist = history(registry, "price", panel, store, AS_OF)
    spec = load_spec()
    rho = fit(hist, spec).gamma[:, hist.driver.index("price:alpha"), 1]
    assert np.all(np.abs(rho) <= spec.rho_max + 1e-9)
    assert np.median(rho) > spec.rho_max - 0.05


def test_a_path_ends_at_the_last_month_the_panel_knows(tmp_path):
    panel = simulated.panel(120, seed=7)
    y = panel.y.copy()
    y[-2:] = np.nan
    panel = replace(panel, y=y)
    store = simulated.store(tmp_path, panel, seed=3)
    registry = simulated.registry()
    hist = history(registry, "price", panel, store, AS_OF)
    node = hist.driver.index("price:alpha")
    item = path_of(hist, node)
    assert item.first == 1
    assert item.observed[-1] == len(panel.months) - 3
    assert len(item.path) == len(panel.months) - 3
    assert fit(hist, load_spec()).rows[node] == len(item.path)


def test_driver_paths_repeat_bit_for_bit(version, data):
    _, panel, _ = data
    macro = macro_paths(panel, np.array([1.0, 0.5]))
    model = version.models[MODEL.model_id]
    shocks = platform_shocks(model, macro, ["price:alpha", "price:beta"])
    first = simulate_hier(
        model, macro, ["price:alpha", "price:beta"], [],
        correlation=(version.correlated, version.correlation), shocks=shocks,
        seed=11, stream=0,
    )
    again = simulate_hier(
        model, macro, ["price:alpha", "price:beta"], [],
        correlation=(version.correlated, version.correlation), shocks=shocks,
        seed=11, stream=0,
    )
    assert np.array_equal(first.paths, again.paths)


def test_the_platform_shocks_replace_the_models_correlated_noise(version, data):
    _, panel, _ = data
    macro = macro_paths(panel, np.array([1.0, 0.5]))
    model = version.models[MODEL.model_id]
    draws, steps = macro.paths.shape[0], 2
    shocks = np.random.default_rng(12).standard_normal((draws, steps + 3, 2))
    drivers = ["price:alpha", "price:beta"]

    given = simulate_hier(
        model, macro, drivers, [],
        correlation=(version.correlated, version.correlation),
        shocks=shocks, seed=11, stream=0,
    )
    # The platform gives more rows than this model needs: only the first ones are read.
    trimmed = simulate_hier(
        model, macro, drivers, [],
        correlation=(version.correlated, version.correlation),
        shocks=shocks[:, :steps], seed=11, stream=0,
    )
    other = simulate_hier(
        model, macro, drivers, [],
        correlation=(version.correlated, version.correlation),
        shocks=shocks[:, :steps] * 0.5, seed=11, stream=0,
    )
    assert np.array_equal(given.paths, trimmed.paths)
    assert not np.array_equal(given.paths, other.paths)

    with pytest.raises(ValueError, match="steps of shocks"):
        simulate_hier(
            model, macro, drivers, [],
            correlation=(version.correlated, version.correlation),
            shocks=shocks[:, : steps - 1], seed=11, stream=0,
        )


def test_the_hier_model_uses_the_platform_shocks(version, data):
    from mst.model_sdk.testing import check_shocks

    _, panel, _ = data
    model = version.models[MODEL.model_id]
    findings = check_shocks(
        MODEL,
        model,
        macro_paths(panel, np.array([1.0, 0.5])),
        ["price:alpha", "price:beta"],
        correlated=["price:alpha", "price:beta"],
        history=len(model.fits["price"].history.months),
        seed=11,
        stream=0,
    )
    assert [str(finding) for finding in findings] == []


def small_spec() -> Spec:
    """A quick sampler: the bounds do not need a converged posterior."""
    return Spec(
        spread=0.1,
        constant_spread=10.0,
        rho_max=0.99,
        holdout=24,
        reach=12,
        sampler=Sampler(draws=10, burn=5, seed=7),
    )


def test_the_training_bounds_cover_the_observations_of_the_nodes_window(tmp_path):
    panel = simulated.panel(120, seed=4)
    store = simulated.store(tmp_path, panel, seed=2)
    # A month before the node's window (its own lag is unknown there) holds an extreme value:
    # it is data, but the model was not trained on it, so it must not widen the trained range.
    rows = store.read("drv.price.alpha")
    rows[0] = replace(rows[0], value=rows[0].value + 500.0)
    store.write("drv.price.alpha", rows)
    registry = simulated.registry()
    hist = history(registry, "price", panel, store, AS_OF)
    node = hist.driver.index("price:alpha")
    item = path_of(hist, node)
    result = fit(hist, small_spec())

    assert item.first == 1 and 0 not in item.observed
    assert result.rows[node] == len(item.observed)
    assert result.train_low[node] == item.values.min()
    assert result.train_high[node] == item.values.max()
    assert result.train_high[node] < hist.d[0, node]

    # A node without a series of its own takes the bounds of the nearest object above it with
    # data, exactly the source the simulation takes its level from.
    delta, sector = hist.driver.index("price:delta"), hist.driver.index("price:sector")
    assert result.rows[delta] == 0
    assert result.train_low[delta] == result.train_low[sector]
    assert result.train_high[delta] == result.train_high[sector]


def test_rows_published_after_the_calculation_move_no_bound(tmp_path):
    panel = simulated.panel(120, seed=4)
    store = simulated.store(tmp_path, panel, seed=2)
    registry = simulated.registry()
    before = fit(history(registry, "price", panel, store, AS_OF), small_spec())

    rows = store.read("drv.price.alpha")
    later = replace(rows[-1], value=9999.0, published=date(2031, 1, 1))
    store.write("drv.price.alpha", rows + [later])
    again = fit(history(registry, "price", panel, store, AS_OF), small_spec())

    assert np.array_equal(before.train_low, again.train_low, equal_nan=True)
    assert np.array_equal(before.train_high, again.train_high, equal_nan=True)
