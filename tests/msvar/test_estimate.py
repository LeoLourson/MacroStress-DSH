"""The Gibbs sampler of the MS-VAR on data simulated from known parameters."""

import numpy as np
import pytest

from mst.macro.panel import Panel
from mst.plugins.msvar import estimate
from tests.msvar import simulated as sim

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def posterior():
    y, _ = sim.simulate(240, np.random.default_rng(0))
    return estimate(sim.panel(y), sim.spec(), 2)


def traces(posterior, draw: int) -> np.ndarray:
    """The total volatility of each regime: ``trace(C diag(exp h) C')``."""
    columns = (posterior.chol[draw] ** 2).sum(axis=1)  # ||C[:, j]||^2 of each regime
    return np.einsum("mn,mn->m", columns, np.exp(posterior.logvar[draw]))


def test_the_sampler_recovers_the_transition_matrix(posterior):
    mean = posterior.transition.mean(axis=0)
    assert np.allclose(mean, sim.TRANSITION, atol=0.1)
    low, high = np.quantile(posterior.transition, [0.05, 0.95], axis=0)
    assert ((sim.TRANSITION >= low) & (sim.TRANSITION <= high)).all()


def test_the_regimes_keep_their_labels_by_volatility(posterior):
    # The declaration order is the volatility order: calm first, stress second.
    for draw in range(posterior.draws):
        assert traces(posterior, draw)[0] < traces(posterior, draw)[1]


def test_the_regime_share_of_history_matches_the_truth(posterior):
    # The true path spends 0.21 of its months in the stress regime.
    share = (posterior.states == 1).mean(axis=1)
    assert share.mean() == pytest.approx(0.21, abs=0.08)
    assert posterior.diagnostics["regime_share"][1] == pytest.approx(share.mean())


def test_the_same_seed_repeats_the_draws_bit_for_bit():
    y, _ = sim.simulate(240, np.random.default_rng(1))
    first = estimate(sim.panel(y), sim.spec(draws=8, burn=8, seed=5), 2)
    second = estimate(sim.panel(y), sim.spec(draws=8, burn=8, seed=5), 2)
    for name in ("const", "lags", "chol", "logvar", "transition", "states", "history"):
        assert np.array_equal(getattr(first, name), getattr(second, name))


def test_the_missing_months_are_drawn_not_interpolated():
    y, _ = sim.simulate(240, np.random.default_rng(0))
    holed = y.copy()
    holes = np.arange(30, 210, 12)
    holed[holes, 1] = np.nan
    holed[holes + 6, 2] = np.nan
    found = estimate(
        Panel(
            ids=sim.IDS,
            n_global=sim.N_GLOBAL,
            months=sim.panel(y).months,
            y=holed,
            approximate=np.zeros(holed.shape, dtype=bool),
        ),
        sim.spec(),
        2,
    )
    assert found.diagnostics["missing"] == 2 * len(holes)
    assert np.allclose(found.transition.mean(axis=0), sim.TRANSITION, atol=0.08)


def test_a_panel_not_starting_in_the_spec_is_refused(posterior):
    from datetime import date

    from mst.macro.panel import months_between

    y, _ = sim.simulate(60, np.random.default_rng(3))
    later = Panel(
        ids=sim.IDS,
        n_global=sim.N_GLOBAL,
        months=months_between(date(2001, 1, 1), date(2005, 12, 1)),
        y=y,
        approximate=np.zeros(y.shape, dtype=bool),
    )
    with pytest.raises(ValueError, match="the panel starts in 2001-01-01"):
        estimate(later, sim.spec(), 2)


def test_one_regime_is_refused():
    y, _ = sim.simulate(60, np.random.default_rng(3))
    with pytest.raises(ValueError, match="at least two regimes"):
        estimate(sim.panel(y), sim.spec(draws=4, burn=4), 1)
