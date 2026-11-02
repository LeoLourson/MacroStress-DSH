"""The satellite in differences: no spurious long-run level, the level is the accumulated path."""

from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from mst.drivers import bundle as satellite_bundle
from mst.drivers.history import history
from mst.model_sdk import AdditiveImpact
from mst.model_sdk.contracts import MacroSimulation as Simulation
from mst.model_sdk.contracts import SatelliteEstimateContext
from mst.plugins.hier.check import check
from mst.plugins.hier.estimate import difference, fit
from mst.plugins.hier.model import HierModel
from mst.plugins.hier.simulate import IMPACT, PARAMETERS_FROM_ABOVE
from mst.plugins.hier.simulate import simulate as simulate_hier
from mst.plugins.hier.spec import Sampler, Spec, load_spec
from tests.drivers import simulated

AS_OF = date(2030, 1, 1)
MODEL_ID = "hier.gibbs"


def differenced_spec() -> Spec:
    """A quick sampler of the simulated quantity ``price``, fitted in differences."""
    return Spec(
        spread=0.1,
        constant_spread=10.0,
        rho_max=0.99,
        holdout=24,
        reach=12,
        differences=["price"],
        sampler=Sampler(draws=60, burn=30, seed=7),
    )


def macro_paths(panel, x_future, draws=3):
    """A macro simulation whose global variable follows ``x_future`` after the panel."""
    p, horizon = 3, len(x_future)
    history = np.repeat(panel.y[None, -p:], draws, axis=0)
    future = np.zeros((draws, horizon, 2))
    future[:, :, 0] = x_future
    paths = np.concatenate([history[:, -1:], future], axis=1)
    months = [panel.months[-1]] + [date(2030, m, 1) for m in range(1, horizon + 1)]
    return Simulation(ids=panel.ids, months=months, paths=paths, history=history)


def upper_of(macro):
    from mst.model_sdk.contracts import UpperPaths

    return UpperPaths(macro=macro, levels={})


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


def with_fit(version, fit_):
    """The version's model with the fit of ``price`` replaced."""
    return replace(version.models[MODEL_ID], fits={"price": fit_})


@pytest.fixture(scope="module")
def differenced(tmp_path_factory):
    """The plugin's estimate on data from the known model in differences, as a bundle."""
    panel = simulated.panel(240)
    store = simulated.changes_store(tmp_path_factory.mktemp("series"), panel)
    registry = simulated.registry()
    model = HierModel(spec=differenced_spec())
    estimate = model.estimate(
        SatelliteEstimateContext(
            store=store,
            registry=registry,
            panel=panel,
            as_of=AS_OF,
            macro="macro-version",
            quantities=("price",),
        )
    )
    meta, arrays = satellite_bundle.assemble(
        registry, "macro-version", AS_OF, {model.model_id: estimate}
    )
    root = tmp_path_factory.mktemp("models")
    name = satellite_bundle.save(meta, arrays, root)
    return panel, store, satellite_bundle.load(root, name, {model.model_id: model})


TRUTH = {"sector": simulated.CHANGE_SECTOR, **simulated.CHANGE_COMPANIES,
         "delta": simulated.CHANGE_SECTOR}


def _fixed(version):
    """The fit of the bundle's ``price`` with every draw set to the known truth, no noise."""
    fit_ = version.models[MODEL_ID].fits["price"]
    gamma = np.zeros_like(fit_.gamma)
    for node, name in enumerate(fit_.history.nodes):
        gamma[:, node, :] = TRUTH[name]
    return replace(fit_, gamma=gamma, sigma=np.zeros_like(fit_.sigma))


def _changes(panel, x_future):
    """The changes of the global variable over the panel months and the horizon."""
    return difference(np.concatenate([panel.y[:, 0], x_future]))


def test_differenced_data_recovers_the_known_elasticities(differenced):
    _, _, version = differenced
    fit_ = version.models[MODEL_ID].fits["price"]
    hist = fit_.history
    assert fit_.differenced is True
    covered = []
    for name, truth in TRUTH.items():
        if name == "delta":
            continue
        draws = fit_.gamma[:, hist.nodes.index(name)]
        low, high = np.percentile(draws, [5, 95], axis=0)
        covered += list((low <= truth) & (truth <= high))
        sigma = np.median(fit_.sigma[:, hist.nodes.index(name)])
        assert sigma == pytest.approx(simulated.CHANGE_SIGMA, rel=0.15)
    assert np.mean(covered) >= 0.75
    # The training bounds stay in levels, so the outside-range flag keeps its meaning.
    sector = hist.nodes.index("sector")
    levels = hist.d[np.isfinite(hist.d[:, sector]), sector]
    assert fit_.train_low[sector] == pytest.approx(levels.min())
    assert fit_.train_high[sector] == pytest.approx(levels.max())


def test_a_differenced_satellite_beats_the_levels_autoregression(differenced, tmp_path):
    panel, store, _ = differenced
    registry = simulated.registry()
    hist = history(registry, "price", panel, store, AS_OF)
    checks = check(hist, differenced_spec())
    for name in ["sector", *simulated.CHANGE_COMPANIES]:
        item = checks[hist.nodes.index(name)]
        assert item.served and item.satellite < item.baseline


def test_a_level_path_is_the_cumulative_sum_of_its_changes(differenced):
    """The simulated levels equal start + the accumulated changes, for every kind of start."""
    panel, _, version = differenced
    model = version.models[MODEL_ID]
    hist = model.fits["price"].history
    x_future = np.array([1.0, 2.0, 0.5, -1.0])
    macro = macro_paths(panel, x_future)
    result = simulate_hier(
        with_fit(version, _fixed(version)),
        upper_of(macro),
        ["price:alpha", "price:delta"],
        [],
        shocks=platform_shocks(model, macro, ["price:alpha", "price:delta"]),
        seed=7,
        stream=0,
    )
    dx = _changes(panel, x_future)
    zero = len(panel.months) - 1

    # A driver whose last value is the panel's last month: the level starts there and every
    # month adds the change the known equation draws.
    node = hist.nodes.index("alpha")
    change = hist.d[-1, node] - hist.d[-2, node]
    level, expected = hist.d[-1, node], [hist.d[-1, node]]
    for t in range(1, len(x_future) + 1):
        u = zero + t
        change = simulated.CHANGE_COMPANIES["alpha"] @ [1.0, change, dx[u], dx[u - 1]]
        level += change
        expected.append(level)
    assert result.paths[:, :, 0] == pytest.approx(np.tile(expected, (3, 1)))

    # A driver without history starts from the level of the sector above it and keeps the
    # sector's last change as its own start.
    sector = hist.nodes.index("sector")
    change = hist.d[-1, sector] - hist.d[-2, sector]
    level, expected = hist.d[-1, sector], [hist.d[-1, sector]]
    for t in range(1, len(x_future) + 1):
        u = zero + t
        change = simulated.CHANGE_SECTOR @ [1.0, change, dx[u], dx[u - 1]]
        level += change
        expected.append(level)
    assert result.paths[:, :, 1] == pytest.approx(np.tile(expected, (3, 1)))
    assert result.flags["price:delta"] == {PARAMETERS_FROM_ABOVE}


def test_a_driver_with_an_older_last_value_accumulates_from_it(differenced):
    panel, _, version = differenced
    model = version.models[MODEL_ID]
    fit_ = model.fits["price"]
    hist = fit_.history
    node = hist.nodes.index("alpha")
    count, zero = len(panel.months), len(panel.months) - 1
    d = hist.d.copy()
    d[-5:, node] = np.nan
    lagging = replace(_fixed(version), history=replace(hist, d=d))
    x_future = np.array([1.0, 2.0, 0.5, -1.0])
    macro = macro_paths(panel, x_future)
    result = simulate_hier(
        with_fit(version, lagging),
        upper_of(macro),
        ["price:alpha"],
        [],
        shocks=platform_shocks(model, macro, ["price:alpha"]),
        seed=1,
        stream=0,
    )
    dx = _changes(panel, x_future)
    change = hist.d[count - 6, node] - hist.d[count - 7, node]
    level, expected = hist.d[count - 6, node], {}
    for u in range(count - 5, count + len(x_future)):
        change = simulated.CHANGE_COMPANIES["alpha"] @ [1.0, change, dx[u], dx[u - 1]]
        level += change
        if u >= zero:
            expected[u - zero] = level
    assert result.paths[0, :, 0] == pytest.approx([expected[t] for t in range(len(x_future) + 1)])


def test_a_missing_month_before_the_last_value_starts_at_the_mean_change(differenced):
    panel, _, version = differenced
    model = version.models[MODEL_ID]
    fit_ = model.fits["price"]
    hist = fit_.history
    node = hist.nodes.index("alpha")
    count = len(panel.months)
    d = hist.d.copy()
    d[count - 10, node] = np.nan
    d[count - 8:, node] = np.nan
    gapped = replace(_fixed(version), history=replace(hist, d=d))
    x_future = np.array([1.0, 2.0])
    macro = macro_paths(panel, x_future)
    result = simulate_hier(
        with_fit(version, gapped),
        upper_of(macro),
        ["price:alpha"],
        [],
        shocks=platform_shocks(model, macro, ["price:alpha"]),
        seed=1,
        stream=0,
    )
    dx = _changes(panel, x_future)
    gamma = simulated.CHANGE_COMPANIES["alpha"]
    change = gamma[0] / (1 - gamma[1])  # the month before the last value is missing
    level, expected = hist.d[count - 9, node], []
    for u in range(count - 8, count + len(x_future)):
        change = gamma @ [1.0, change, dx[u], dx[u - 1]]
        level += change
        if u >= count - 1:
            expected.append(level)
    assert result.paths[0, :, 0] == pytest.approx(expected)


def test_an_impact_lands_on_the_level_and_does_not_accumulate(differenced):
    panel, _, version = differenced
    model = version.models[MODEL_ID]
    macro = macro_paths(panel, np.array([1.0, 2.0, 0.5, -1.0, 0.0]))
    drivers = ["price:alpha"]
    shocks = platform_shocks(model, macro, drivers)
    plain = simulate_hier(model, upper_of(macro), drivers, [], shocks=shocks, seed=3, stream=0)
    moved = simulate_hier(
        model,
        upper_of(macro),
        drivers,
        [AdditiveImpact(target="price:alpha", start=2, values=(10.0, 10.0))],
        shocks=shocks,
        seed=3,
        stream=0,
    )
    assert moved.flags["price:alpha"] == {IMPACT}
    shift = np.array([0, 0, 10, 10, 0, 0])[None, :, None]
    assert moved.paths == pytest.approx(plain.paths + shift)


def test_without_the_option_the_fit_is_bit_for_bit_as_before(tmp_path):
    """A quantity not named in ``differences`` keeps the exact numbers of the levels fit."""
    panel = simulated.panel(240, seed=5)
    store = simulated.store(tmp_path, panel, seed=2)
    registry = simulated.registry()
    hist = history(registry, "price", panel, store, AS_OF)
    spec = Spec(
        spread=0.1,
        constant_spread=10.0,
        rho_max=0.99,
        holdout=24,
        reach=12,
        sampler=Sampler(draws=25, burn=10, seed=7),
    )
    result = fit(hist, spec)
    assert result.differenced is False
    # The numbers the same code produced before the option existed, on the same data.
    assert float(result.gamma.sum()) == -55.413514720370415
    assert result.gamma[0, hist.nodes.index("sector")].tolist() == [
        2.3739325374713793,
        0.433567148962365,
        1.4416411514393548,
        -0.6411546563515367,
    ]
    assert float(np.nanmedian(result.sigma)) == 1.0011809768151583
    assert result.rows.tolist() == [239, 239, 239, 239, 0]
    assert result.train_low[hist.nodes.index("delta")] == result.train_low[
        hist.nodes.index("sector")
    ]
    # The shipped specification differences the shares and nothing else.
    assert load_spec().differences == ["share_price"]
    again = fit(hist, spec)
    assert np.array_equal(result.gamma, again.gamma)
    # On data from a model in levels the differenced fit is another model, not the same one.
    other = fit(hist, spec.model_copy(update={"differences": ["price"]}))
    assert other.differenced is True
    assert not np.array_equal(result.gamma, other.gamma)


def test_a_missing_level_gives_two_missing_changes():
    d = np.array([1.0, np.nan, 3.0, 4.0])
    assert np.isnan(difference(d)[1])
    assert np.isnan(difference(d)[2])
    assert difference(d)[3] == 1.0
