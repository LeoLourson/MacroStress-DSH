"""Variables whose equation runs on monthly changes: the version stores the levels form."""

import numpy as np
import pytest
from pydantic import ValidationError

from mst.model_sdk.stability import companion_radius
from mst.plugins.bvar.estimate import estimate, levels_form
from mst.plugins.bvar.var import iterate, split
from tests.bvar import simulated as sim

DRIFT = np.array([0.0, 0.4, 0.0])
"""ru.b is a random walk with this drift in its changes; gl.a and ru.c stay in levels."""


def test_the_levels_form_reproduces_the_accumulated_changes():
    rng = np.random.default_rng(0)
    n, p, d = 3, 2, 1
    s = np.array([0.0, 1.0, 1.0])
    changes = np.hstack(
        [rng.normal(size=(n, 1)), 0.2 * rng.normal(size=(n, n * p)), rng.normal(size=(n, d))]
    )
    const, lags, dummy = split(changes, p)
    y = rng.normal(size=(p + 1, n))  # the last p + 1 months of levels
    shocks = rng.normal(size=(6, n))
    future = np.ones((6, d))
    # The changes iterated on their own, then accumulated from the last level.
    w_history = y[1:] - s * y[:-1]
    w = iterate(w_history, lags, const + future @ dummy.T, shocks)
    expected = np.empty_like(w)
    previous = y[-1]
    for t in range(len(w)):
        expected[t] = w[t] + s * previous
        previous = expected[t]
    levels = levels_form(changes, s, p)
    const_l, lags_l, dummy_l = split(levels, p + 1)
    assert lags_l.shape == (p + 1, n, n)
    got = iterate(y, lags_l, const_l + future @ dummy_l.T, shocks)
    assert np.allclose(got, expected)


def test_without_differences_the_levels_form_is_the_coefficients():
    rng = np.random.default_rng(1)
    coefficients = rng.normal(size=(3, 1 + 3 * 2))
    assert np.array_equal(levels_form(coefficients, np.zeros(3), 2), coefficients)


def _walk(count: int, rng) -> np.ndarray:
    """gl.a and ru.c from the stationary VAR of the helper; ru.b replaced by a random walk
    with drift whose change loads on gl.a's shock."""
    y = sim.simulate(count, rng)
    steps = DRIFT[1] + 0.5 * (y[:, 0] - y[:, 0].mean()) + rng.normal(size=count)
    y[:, 1] = 100.0 + np.cumsum(steps)
    return y


@pytest.mark.slow
@pytest.mark.parametrize("volatility", ["constant", "stochastic"])
def test_a_differenced_variable_keeps_an_exact_unit_root_and_its_drift(volatility):
    rng = np.random.default_rng(2)
    y = _walk(240, rng)
    y[100:104, 1] = np.nan  # an unknown stretch is still drawn in levels
    posterior = estimate(
        sim.panel(y),
        sim.spec(
            volatility=volatility,
            stationary=["gl.a", "ru.c"],
            differences=["ru.b"],
            draws=200,
            burn=50,
            seed=5,
        ),
    )
    assert posterior.lags.shape[1:] == (2, 3, 3)
    assert posterior.lags_count == 2
    assert posterior.history.shape[1:] == (2, 3)
    # The level of ru.b carries its previous level one for one: the lag sum on itself is one
    # exactly and on every other variable nought, so its unit root is in every draw.
    sums = posterior.lags.sum(axis=1)
    assert np.allclose(sums[:, 1, 1], 1.0)
    assert np.allclose(sums[:, 0, 1], 0.0) and np.allclose(sums[:, 2, 1], 0.0)
    radius = companion_radius(posterior.lags)
    assert np.all(radius <= 1.0 + 1e-9)
    assert np.all(radius >= 1.0 - 1e-9)
    # The mean change of ru.b implied by each draw — the stationary mean of the system of
    # changes, G read back from the levels form — covers the drift it was simulated with.
    s = np.diag([0.0, 1.0, 0.0])
    means = []
    for const, lags in zip(posterior.const, posterior.lags, strict=True):
        g, previous = [], np.zeros((3, 3))
        for k in range(1, 3):
            previous = lags[k - 1] - (s if k == 1 else 0) + previous @ s
            g.append(previous)
        means.append(np.linalg.solve(np.eye(3) - g[0], const)[1])
    low, high = np.quantile(means, [0.005, 0.995])
    assert low <= DRIFT[1] <= high


def _drift_of(posterior, k: int, months: int = 240) -> np.ndarray:
    """The mean monthly change of variable ``k`` over ``months`` of the forecast without
    shocks, one value per draw: what the drift of the version is after its dummies."""
    from mst.plugins.bvar.forecast import _future

    future = _future(posterior, months)
    drifts = []
    for s in range(posterior.draws):
        intercept = posterior.const[s] + future @ posterior.dummy[s].T
        path = iterate(posterior.history[s], posterior.lags[s], intercept, np.zeros((months, 3)))
        drifts.append((path[-1, k] - posterior.history[s][-1, k]) / months)
    return np.array(drifts)


@pytest.mark.slow
def test_a_level_step_does_not_bend_the_drift_of_a_variable_on_changes():
    # A step of the spec shifts a level. On changes it is one impulse in the month of the
    # step: the drift of ru.b after the step is the drift of the whole sample.
    rng = np.random.default_rng(6)
    y = _walk(240, rng)
    step = sim.START.replace(year=sim.START.year + 15)  # month 180 of 240
    spec = sim.spec(
        volatility="stochastic",
        stationary=["gl.a", "ru.c"],
        differences=["ru.b"],
        dummies=[{"month": step, "kind": "step"}],
        draws=200,
        burn=50,
        seed=8,
    )
    posterior = estimate(sim.panel(y), spec)
    kinds = [(d.month, d.kind) for d in posterior.dummies]
    assert kinds == [(step, "step"), (step, "impulse")]
    # ru.b reads the impulse only, ru.c (in levels) the step only.
    assert np.all(posterior.dummy[:, 1, 0] == 0.0)
    assert np.all(posterior.dummy[:, 2, 1] == 0.0)
    low, high = np.quantile(_drift_of(posterior, 1), [0.005, 0.995])
    assert low <= DRIFT[1] <= high


@pytest.mark.slow
def test_a_dummy_over_unobserved_months_is_left_out_of_that_equation():
    # The impulse falls in the last month, where ru.b is not yet known: no observation can
    # fix its coefficient, so the equation of ru.b does not read it, and the drawn last
    # month stays on the scale of the walk.
    rng = np.random.default_rng(7)
    y = _walk(200, rng)
    y[-1, 1] = np.nan
    last = sim.START.replace(year=sim.START.year + 16, month=8)  # month 199
    spec = sim.spec(
        volatility="stochastic",
        stationary=["gl.a", "ru.c"],
        differences=["ru.b"],
        dummies=[{"month": last, "kind": "impulse"}],
        draws=100,
        burn=50,
        seed=9,
    )
    posterior = estimate(sim.panel(y), spec)
    assert np.all(posterior.dummy[:, 1, :] == 0.0)
    jump = posterior.history[:, -1, 1] - y[-2, 1]
    assert np.all(np.abs(jump) < 10.0)


def test_a_saved_version_reads_back_its_dummy_columns():
    from dataclasses import replace
    from datetime import date

    from mst.plugins.bvar.model import BvarModel, parts

    y = _walk(80, np.random.default_rng(10))
    step = date(2004, 1, 1)
    spec = sim.spec(
        volatility="stochastic",
        stationary=["gl.a", "ru.c"],
        differences=["ru.b"],
        dummies=[{"month": step, "kind": "step"}],
        draws=3,
        burn=0,
    )
    posterior = estimate(sim.panel(y), spec)
    posterior = replace(posterior, rotation=np.tile(np.eye(3), (3, 1, 1)), shocks=["a", "b", "c"])
    saved = parts(posterior, spec, sim.panel(y), date(2006, 9, 30))
    loaded = BvarModel().load(saved.meta, saved.arrays)
    assert [(d.month, d.kind) for d in loaded.dummies] == [(step, "step"), (step, "impulse")]
    assert loaded.dummy.shape[-1] == 2


@pytest.mark.slow
def test_coinciding_dummy_columns_keep_the_drift():
    # A level impulse in month m is read by ru.c in levels; ru.b on changes reads impulses in
    # m and m + 1, and the first of them is the same column. The posterior still recovers the
    # drift of ru.b, and its dummy coefficients stay on the scale of the data.
    rng = np.random.default_rng(11)
    y = _walk(240, rng)
    impulse = sim.START.replace(year=sim.START.year + 15)
    spec = sim.spec(
        volatility="stochastic",
        stationary=["gl.a", "ru.c"],
        differences=["ru.b"],
        dummies=[{"month": impulse, "kind": "impulse"}],
        draws=200,
        burn=50,
        seed=12,
    )
    posterior = estimate(sim.panel(y), spec)
    assert np.abs(posterior.dummy[:, 1, :]).max() < 20.0
    low, high = np.quantile(_drift_of(posterior, 1), [0.005, 0.995])
    assert low <= DRIFT[1] <= high


def test_the_start_on_changes_reads_no_dummy_column():
    # Coinciding dummy columns make the conjugate start arbitrary along their difference;
    # the start on changes leaves every dummy coefficient at nought and does not depend on
    # how many such columns there are.
    from mst.plugins.bvar.estimate import _start_without_dummies, minnesota

    y = _walk(120, np.random.default_rng(13))
    panel = sim.panel(y)
    s = np.array([0.0, 1.0, 0.0])
    w = y[1:] - s * y[:-1]
    starts = []
    for d in (1, 3):
        spec = sim.spec(dummies=[{"month": sim.START, "kind": "impulse"}] * d)
        column = np.zeros((len(w), 1))
        column[60] = 1.0
        dummies = np.repeat(column, d, axis=1)
        start, _ = _start_without_dummies(
            w, dummies, minnesota(panel, spec), 1, panel, np.random.default_rng(1)
        )
        assert np.all(start[:, 1 + 3 :] == 0.0)
        starts.append(start[:, : 1 + 3])
    assert np.array_equal(starts[0], starts[1])


def test_differences_with_dummies_need_the_stochastic_sampler():
    y = sim.simulate(60, np.random.default_rng(3))
    spec = sim.spec(differences=["ru.b"], dummies=[{"month": sim.START, "kind": "step"}])
    with pytest.raises(ValueError, match="stochastic"):
        estimate(sim.panel(y), spec)


def test_a_difference_of_an_unknown_variable_is_refused():
    y = sim.simulate(60, np.random.default_rng(3))
    with pytest.raises(ValueError, match="ru.x"):
        estimate(sim.panel(y), sim.spec(differences=["ru.x"], draws=2))


def test_the_level_dummy_rows_do_not_mix_with_differences():
    with pytest.raises(ValidationError, match="differences"):
        sim.spec(differences=["ru.b"], mu=1.0)


def test_a_dummy_is_not_read_by_the_equations_it_spares():
    # The impulse names ru.b among the variables it spares: neither its own column nor the
    # columns derived from it enter the equation of ru.b, and ru.c still reads its own.
    rng = np.random.default_rng(13)
    y = _walk(120, rng)
    impulse = sim.START.replace(year=sim.START.year + 5)
    spec = sim.spec(
        volatility="stochastic",
        stationary=["gl.a", "ru.c"],
        differences=["ru.b"],
        dummies=[{"month": impulse, "kind": "impulse", "spares": ["ru.b"]}],
        draws=20,
        burn=10,
        seed=14,
    )
    posterior = estimate(sim.panel(y), spec)
    assert np.all(posterior.dummy[:, 1, :] == 0.0)
    assert np.any(posterior.dummy[:, 2, 0] != 0.0)


def test_a_version_saved_before_jumps_loads_without_them():
    from dataclasses import replace
    from datetime import date

    from mst.plugins.bvar.model import BvarModel, parts

    y = _walk(60, np.random.default_rng(15))
    spec = sim.spec(volatility="stochastic", stationary=["gl.a", "ru.c"], draws=2, burn=0)
    posterior = estimate(sim.panel(y), spec)
    posterior = replace(posterior, rotation=np.tile(np.eye(3), (2, 1, 1)), shocks=["a", "b", "c"])
    saved = parts(posterior, spec, sim.panel(y), date(2004, 12, 31))
    arrays = {k: v for k, v in saved.arrays.items() if k not in ("jump_prob", "jump_scale")}
    loaded = BvarModel().load(saved.meta, arrays)
    assert np.all(loaded.jump_prob == 0.0)
    assert np.all(loaded.jump_scale == 1.0)
